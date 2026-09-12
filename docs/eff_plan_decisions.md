# Eff / EffPlan（方案一）正式运行决策记录

本文件记录方案一进入正式训练前必须锁定的选择。**只有本文件记为「已锁定」的项才能写进
`configs/experiment/effplan_cube.yaml`**；其余项在用户确认前保持 `null`，配置保持
`status: draft`，`load_eff_protocol` 会因此拒绝执行，这是设计行为，不是缺陷。

配置加载器的硬性要求（`tdwm/training/eff_protocol.py`）：阶段子树内**不允许存在任何
`null`**，且 `status` 必须为 `locked`；`eff_training` 阶段还要求
`data.efficiency_groups == "time_offset_within_minibatch"`。因此"部分填写"没有任何意义，
必须全部定完才能开跑。

## 已锁定（用户于 2026-09-12 确认）

| 项 | 决定 | 落到配置的位置 |
|---|---|---|
| Eff 的正式评分 | **rollout 累计变化 + 终点 `V(G(z_H,m),m)`**，不是纯终点评分 | `evaluation.eff_score` |
| 首版超参包 | 接受：β=5、同时间跨度组内归一化、V 的 TD 备份 n=50 原始步、G/V 各两层 512 隐藏单元、EMA 更新率 0.005、沿用 V1-C 的 10 epochs 预算 | `eff_training.settings.*`、`data.efficiency_groups` |
| EffPlan 的 CEM 预算 | **总预算共享**：8 次状态改善 + 最后 1 次动作搜索，共享总计 30 轮、每轮 300 候选 | `planner_refinement.settings.search_iterations`、`evaluation.effplan_search_iterations` |
| P 第二阶段训练数据 | 先用**同 episode、每 5 步取一个状态的 25 步时间对齐片段**；跨 episode 目标仍用于 G/V，暂不用于 P 的轨迹拟合 | `planner_*.settings` 中 `cross_episode_probability=0`、`trajectory` 分支有效 |

## 方案一原文证据（2026-09-12 复核 `results_td_current_extracted.txt` 第 410–633 行）

- **G 用折扣，V 不用折扣。** 第 448 行 G 的 TD target 为
  `Y_t^G = sg( z_{t+1} + γ_G (1−d_t) Ḡ(z_{t+1}, m_g) )`，符号就是代码里的 `gamma_g`；
  第 481 行明确「V 不使用折扣，保持 W 为完整累计变化量」。
- **γ_G 在原文中没有给数值。** 全篇方案一范围内检索 `γ / gamma / 折扣`，唯一命中就是
  第 448 行的定义式。因此上一版建议的 `0.98^5 ≈ 0.9039` 是**从 V1-C3 配置折算的推测值，
  不是方案原文的取值**，未获用户确认，不得写入配置。
- **原文自己列出了必须在正式评测前固定的项**（第 628 行）：
  β、比较组与基准 b、备份长度 n、失败终止处理、采样比例、各损失系数。
  并明确要求「**不能依据测试成功率临时挑选**」。
  第 629 行还要求为静止或近零位移的退化片段固定有效性筛选与数值保护规则。
- `load_eff_protocol` 对 `eff_training` 阶段强制
  `data.efficiency_groups == "time_offset_within_minibatch"`，这正是「比较组」的落地形式，
  代码已把它定为唯一可选值，不再作为开放项。

## 仍待定（附建议值，未经用户确认不得写入配置）

### A. Eff 训练（`eff_training.settings`）

| 字段 | 建议值 | 依据 / 备注 |
|---|---|---|
| `gamma_g` | `0.9039`（= 0.98^5） | 沿用 V1-C3 的 `gamma_per_primitive_step: 0.98`，按 5 原始步一个块折算。备选 `1.0`（无折扣）。**建议包里没有折扣这一项，必须单独确认。** |
| `critic_coefficient` | `1.0` | λ_V，即 V 损失相对 G 损失的权重。V1-C3 未单独暴露该系数；取 1.0 表示等权，需你确认。 |
| `validation_batches` | `20` | 纯工程量，用于每 epoch 的验证曲线，不影响方法。 |

已由"建议包"覆盖、无需再问：`g_hidden_dim=512`、`v_hidden_dim=512`、`beta=5`、
`ema_rate=0.005`、`backup_primitive_steps=50`、`learning_rate=1e-4`、
`weight_decay=0.001`、`epochs=10`、`updates_per_epoch=12796`、`gradient_clip=1.0`、
`cross_episode_probability=0.3`、`epsilon=1e-6`、`warmup_fraction=0.01`、
`validation_seed=50`、`include_goal_boundary=true`、`seed=3072`。
`data.efficiency_groups` 必须设为 `time_offset_within_minibatch`。

### B. EffPlan 两阶段（`planner_generation` / `planner_refinement`）

两阶段都需要各自的 `run`（`epochs`、`updates_per_epoch`）与 `settings`。`EffPlanTrainSettings`
要求：`phase ∈ {generation, refinement}`、`supervision ∈ {final, mean_rounds}` 且必须显式；
generation 只允许 trajectory loss、不能有 CEM 与效率/动力学项；refinement 必须给出正的
`search_iterations`。

| 字段 | 建议值 | 备注 |
|---|---|---|
| `supervision` | `final` | 只监督 K 轮之后的结果，而非每轮平均 |
| `target_readout` | `true` | 用 EMA target 的 G/V 读出；训练与部署必须一致（评测端会断言） |
| `generation.run` | 5 epochs × 2560 updates | 需你确认预算 |
| `refinement.run` | 5 epochs × 2560 updates | 需你确认；phase-2 优化器是否从 phase-1 权重继续也需明确（建议继续） |
| `learning_rate` | `1e-4` | 与 G/V 同量级，待确认 |
| `trajectory_coefficient` | `1.0` | 真实状态节点重建 |
| `efficiency_coefficient` | `0.01` | 测试里用过的小权重，需你确认量级 |
| `dynamics_coefficient` | `0.1` | F rollout 一致性项 |
| `cem_candidates` / `cem_elites` / `cem_batch_size` | `300` / `30` / 需定 | 候选与精英沿用 C–G3；`cem_batch_size` 是纯工程量 |

### C. 评测（`evaluation`）

| 字段 | 建议值 | 备注 |
|---|---|---|
| `receding_horizons` | `{"25": 5, "50": 5, "100": 5}` | 新 full-plan 文档对 H5 计划用 RH5；历史 O50 是 RH1，必须先确定再跑，否则成功率不可比 |
| `render_backend` | 沿用与 C–G3 完全相同的后端 | 渲染设置会直接改变成功率，绝不能让新方法换后端 |
| `eff_score` | 由已锁定的"rollout 累计 + 终点 G→V"决定 | 需落成配置能表达的具体取值 |
| `episodes_per_protocol` / `planning_seed` | 50 / 42 | 已填，可与 C–G3 对齐 |

### D. GPU（第二道门）

- 另一方案占用哪几张卡 / 对应任务名：**未知**。
- 是否允许关机切换到 5 卡 GPU 模式：**未知**。
- 在明确之前，不执行任何关机、重启或占满 5 卡的操作，也不启动正式训练。

## 与方案二（EffAction / EffActionPlan）的区分

`RP1_SAMPLING_DECISIONS.md` 里的"五组待定"属于**方案二（动作生成）**，与本文件无关。
两套方案的 η 含义、直接/TD 分支判据、网络输入输出都不同，不可互相套用。
