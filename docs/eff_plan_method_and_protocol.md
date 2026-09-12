# Eff / EffPlan（方案一）：方法、训练、测试协议

本文写清楚**实现的是什么、怎么训、怎么测**。所有数值都来自
`configs/experiment/effplan_cube.yaml`（已 locked）与对应的源码，不含推测值。
决策的来龙去脉见 `docs/eff_plan_decisions.md`。

---

## 1. 方法

四个部件，其中 **F 全程冻结**，本方案只训练 G、V（合称 Eff）和可选的 P（EffPlan）。

| 部件 | 是什么 | 输入 → 输出 | 是否训练 |
|---|---|---|---|
| **F** | 预训练 LeWM 世界模型 | 5 个 25 维动作块 → z₁..z₅ | **冻结**，`198c468c…` |
| **G** | `EffSuccessor` | `(z, m)` → Ψ，192 维"未来累计特征" | 训练 |
| **V** | `EffCritic` | `(Ψ, m)` → 非负标量"累计 latent 变化" | 训练 |
| **P** | `StatePlanner` | 起止状态 → 一串中间状态节点 | 训练（仅 EffPlan） |

**`m` 是什么（最容易搞混的一点）**：

```python
def goal_task(goal):                     # src/tdwm/methods/eff.py:33
    return F.normalize(goal, dim=-1) * sqrt(192)
```

`m` 是**目标方向**（归一化到半径 √192 的球面），只留方向、丢掉长度。
**V 不直接吃状态**，一次打分的完整链路是：

```
Ψ = G(z_k, m)  →  cost = V(Ψ, m)        # 即 EffModel.value: v(g(state, task), task)
```

记作 **V(G(z, m), m)**。本文统一用这个写法，避免简写成 `V(z→z_g)` 造成误读。

**两个方法名的区别**：

- **Eff（无 planner）**：只训 G/V。测试时用 CEM 搜**动作**，用 G→V 给候选动作序列打分。
- **EffPlan**：额外训 P。P 生成/优化一串**状态节点**，CEM 再搜动作去跟踪这些节点。

---

## 2. 数据

| 项 | 值 |
|---|---|
| 划分 | `rp1_cube_8000_2000`：episode 0–7999 训练，8000–9999 评测 |
| 状态步长 | `state_stride: 5`（一个宏步 = 5 个原始步） |
| 冻结 store | `frozen_store_manifest_sha256: fc80bcc4…`，2010000 行，bfloat16 |
| 冻结 F | `lewm_checkpoint_sha256: 198c468c…`（pretrained LeWM，seed 3072 / epoch 10） |

与方案二（EffAction）**共用同一份冻结 store 和数据源**；与 C→G3 系列共用同一套
episode 划分与 start-goal 选择（O25 `56546fe8…`、O50 `e46ea81c…`、O100 `8a87815e…`）。

---

## 3. 训练

### 3.1 G 和 V（`eff_training`）

预算：10 epochs × 12796 updates，batch 256，lr 1e-4，weight_decay 1e-3，grad_clip 1.0。

**G 的 TD target（用折扣）** —— 方案原文第 448 行：

```
Y_t^G = z_{t+1} + γ_G · (1 − d_t) · Ḡ(z_{t+1}, m_g)      γ_G = 0.98
```

代码：`src/tdwm/methods/eff.py:313` → `successor_target(..., gamma=gamma_g)`。
**注意 γ_G 是按训练 transition 计的，而 stride=5，所以这是每个宏步（5 原始步）的折扣。**

**V 的目标（不用折扣）** —— 方案原文第 481 行明确"V 不使用折扣，保持 W 为完整累计变化量"：

- 目标在备份窗口内（`backup_primitive_steps: 50`）→ 直接用真实 `Σ‖z_{k+1} − z_k‖`
- 目标在窗口外 → 已知前缀 + EMA target 的 `V(G(·), ·)` 补上尾部
- 代码 `movement_target` 的注释明写 "no discount"

**效率权重**（原文第 485–500 行）：

```
η_τ = ‖z_g − z_s‖ / (Σ‖z_{k+1} − z_k‖ + ε)      # 真实数据算出来的路径效率
w_τ = sg[ exp(β (η_τ − b)) ]                      # β = 5
L   = Σ_τ ρ_τ (L_G,τ + λ_V · L_V,τ)               # λ_V = critic_coefficient = 1.0
```

权重**乘在误差外面，不改 target、不替换折扣、不进网络输入**。
比较组固定为 `time_offset_within_minibatch`（加载器强制的唯一取值）。

EMA target：`ema_rate: 0.005`。验证：`validation_batches: 16`（对齐方案二的
`validation.batches`），`validation_seed: 50`。

### 3.2 P（`planner_generation` → `planner_refinement`）

两段各 5 epochs × 2560 updates，lr 1e-4。`supervision: final`、`target_readout: true`
（必须与评测端一致，代码会断言）。

| 阶段 | 损失项 | 值 |
|---|---|---|
| **generation** | 只有 trajectory | `trajectory_coefficient: 1.0` |
| | efficiency / dynamics / CEM | **必须为 0/空**（`EffPlanTrainSettings` 会拒绝） |
| **refinement** | trajectory | 1.0 |
| | efficiency（−η，即最大化效率） | 0.1 |
| | dynamics（规划节点 vs 真实 F rollout） | 0.1 |
| | CEM（出 F rollout 用） | `search_iterations: [2,2,2,2]` |

阶段间用 `--init-from` 只传权重，优化器重建（已记入 manifest）。

---

## 4. 测试

### 4.1 协议（与 C→G3 完全一致）

| 项 | 值 |
|---|---|
| offset | O25 / O50 / O100 |
| episode 预算 | `2 × offset`（50 / 100 / 200 步） |
| **receding horizon** | **O25 = 5，O50 = 1，O100 = 1** ← 只有 O25 每 5 块重规划 |
| 每 offset episode 数 | 50 |
| planning seed | 42 |
| CEM | 300 候选 / 30 轮 / 30 精英，`cem_batch_size: 1` |
| horizon / action_block | 5 / 5 |
| 渲染后端 | `MUJOCO_GL=osmesa`（C–G3 实测产物，不是 egl） |
| 读出 | `target_readout: true`（EMA target，不是 online） |

**F-only 不重跑**：它是共享基线，直接复用既有数字作为对照行。

### 4.2 Eff 的六种评分模式

排序方向只有一种：**cost 越小越好，CEM 升序排**。六种模式的区别只在
"z₀→z₅ 这段怎么计账"。

**纯终点（C–G3 基线口径）**

| 模式 | 评分 |
|---|---|
| `terminal_eff_cost` | `V(G(z₅, m_g), m_g)` |

**累计族**（终点项 + 一项累计，权重 `eff_cumulative_weight` 默认 1.0）

| 模式 | 累计项 |
|---|---|
| `latent_path_eff_cost` | `Σ ‖z_{k+1} − z_k‖`（几何弦长，不用网络） |
| `value_path_eff_cost` | `Σ V(G(z_k, m_{k+1}), m_{k+1})`（**= η 的分母 W**，也是 P 在优化的量） |
| `goal_distance_eff_cost` | `‖z₅ − z_g‖`（几何，只看终点） |

**自含族**（不再另加终点项，k=5 那一项本身就是终点项）

| 模式 | 评分 |
|---|---|
| `value_to_goal_sum_eff_cost` | `Σ_{k=1..5} V(G(z_k, m_g), m_g)` |
| `value_to_goal_mean_eff_cost` | 同上 ÷ 5 |

**分野只看一个下标**：用 `m_{k+1}` 是"量走了多远"，用 `m_g` 是"量离目标多远"。

### 4.3 EffPlan

P 生成状态节点 → CEM 搜动作跟踪全部节点（含目标）。
CEM 预算：8 次状态改善 + 1 次最后动作搜索，**共享 30 轮**
（`effplan_search_iterations: [3,3,3,3,3,3,4,4,4]`，合计 30）。
`effplan_dynamics_coefficient: 0.1`。

---

## 5. 命令

```bash
参数名以下方 `--help` 的实际输出为准，不要凭记忆写。

```bash
# 1) 准备元数据
#    train_eff.py prepare-metadata --config --dataset --output-dir
python scripts/train_eff.py prepare-metadata \
    --config configs/experiment/effplan_cube.yaml \
    --dataset <dataset> --output-dir <meta_dir>

# 2) 训 G/V
#    train_eff.py train --config --latent-store --terminal-metadata
#                       --output-dir --device [--resume]
#    注意：这一步没有 --lewm-checkpoint，F 是从 latent store 来的。
python scripts/train_eff.py train \
    --config configs/experiment/effplan_cube.yaml \
    --latent-store <store> --terminal-metadata <meta_dir/terminal.json> \
    --output-dir <eff_out> --device cuda

# 3) 准备评测 episode 选择
#    evaluate_effplan.py prepare-selections --config --terminal-metadata --output-dir
python scripts/evaluate_effplan.py prepare-selections \
    --config configs/experiment/effplan_cube.yaml \
    --terminal-metadata <meta_dir/terminal.json> --output-dir <sel_dir>

# 4) 评测 Eff（换 --eff-score 即为扫模式，override 会记进 manifest 的
#    protocol_overrides；另有 --eff-cumulative-weight 与 --video）
python scripts/evaluate_effplan.py evaluate \
    --config configs/experiment/effplan_cube.yaml \
    --dataset <dataset> --lewm-checkpoint <lewm.pt> \
    --selection <sel_dir/selection.json> --output-dir <res> --device cuda \
    --method Eff \
    --eff-checkpoint <eff_out/eff.pt> --eff-manifest <eff_out/manifest.json> \
    --eff-score value_path_eff_cost

# 5) 训 P（两阶段）
#    train_effplan.py --config --latent-store --terminal-metadata
#                     --eff-checkpoint --eff-manifest --output-dir --device
#                     --phase {generation,refinement}
#                     [--lewm-checkpoint] [--resume] [--init-from]
python scripts/train_effplan.py \
    --config configs/experiment/effplan_cube.yaml \
    --latent-store <store> --terminal-metadata <meta_dir/terminal.json> \
    --eff-checkpoint <eff_out/eff.pt> --eff-manifest <eff_out/manifest.json> \
    --output-dir <p1_out> --device cuda --phase generation

python scripts/train_effplan.py \
    --config configs/experiment/effplan_cube.yaml \
    --latent-store <store> --terminal-metadata <meta_dir/terminal.json> \
    --eff-checkpoint <eff_out/eff.pt> --eff-manifest <eff_out/manifest.json> \
    --output-dir <p2_out> --device cuda --phase refinement \
    --init-from <p1_out/planner.pt>

# 6) 评测 EffPlan（在 4 的基础上换 --method 并补 planner 两个参数）
python scripts/evaluate_effplan.py evaluate \
    --config configs/experiment/effplan_cube.yaml \
    --dataset <dataset> --lewm-checkpoint <lewm.pt> \
    --selection <sel_dir/selection.json> --output-dir <res> --device cuda \
    --method EffPlan \
    --eff-checkpoint <eff_out/eff.pt> --eff-manifest <eff_out/manifest.json> \
    --planner-checkpoint <p2_out/planner.pt> \
    --planner-manifest <p2_out/manifest.json>
```
```

---

## 6. 结果写回

每次评测的 manifest 含 `protocol_overrides`（`{from: 原值, to: 新值}`）、
`score_mode`、`paired_protocol`、`lewm_checkpoint_sha256`、
`cuda_visible_devices` 等，可直接对照。所有配置与结果需写入 `Results TD.docx`。

**调参纪律**：方案原文第 628 行要求 β、b、n、失败处理、采样比例、损失系数
**必须在正式评测前固定，不能依据测试成功率临时挑选**。改用验证集指标调。
