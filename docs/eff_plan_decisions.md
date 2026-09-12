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

## Eff 的 CEM 代价：四种累计项（已实现，待用户选定默认值）

**关键前提（此前表述不清，这里更正）：不是"两种排序方式"。排序方向只有一种 —— V 是
代价，越小越好，CEM 升序排列。四种选项只是"累计项怎么算"，它们共享同一个终点项。**

每条候选动作序列都做完整的 5 段 F rollout，得到 z1..z5（z0 是当前状态）。
**四种模式的终点项完全相同**：`V(G(z5, m_g), m_g)`，即"每个候选的最后一个状态送进 V"。
差别只在 z0→z5 这一段怎么计账（均由 `cumulative_weight` 加权，默认 1.0）：

分成两族，区别在于**要不要另外加终点项**。

**A. 累计族（在终点项 `V(G(z5,m_g),m_g)` 之上再加一项，权重 `eff_cumulative_weight`）**

| 模式常量 | 累计项 | 含义 |
|---|---|---|
| `terminal_eff_cost` | 无 | 纯终点，即 C–G3 的 state_v 口径 |
| `latent_path_eff_cost` | `Σ_{k=0..4} ‖z_{k+1} − z_k‖` | 5 段几何弦长之和，**低估**真实路径长度 |
| `value_path_eff_cost` | `Σ_{k=0..4} V(G(z_k, m_{k+1}), m_{k+1})` | 逐段用 V 估代价，**就是 `path_efficiency` 的分母** |
| `goal_distance_eff_cost` | `‖z5 − z_g‖` | 只看终点弦长（这是 D，不是 W） |

> 2026-09-12 删除 `discounted_goal_eff_cost`（原"逐步到目标的几何距离"）。它唯一的作用是
> 做 `value_to_goal_*` 的几何对照，用户决定不要，同时删掉只被它使用的
> `cumulative_gamma` / `eff_cumulative_gamma` / `--eff-cumulative-gamma` 整套开关。

**B. 自含族（不再另加终点项，因为 k=5 那一项本身就是终点项）** — 用户 2026-09-12 提出

| 模式常量 | 评分 | 含义 |
|---|---|---|
| `value_to_goal_sum_eff_cost` | `Σ_{k=1..5} V(z_k → z_g)` | 每个动作块之后问一次"离真目标还有多远"，求和 |
| `value_to_goal_mean_eff_cost` | `(1/5) Σ_{k=1..5} V(z_k → z_g)` | 同上，取平均 |

**两族的根本区别**（这是用户新提议与 A′ 的分野，容易混）：

- A′ `value_path` 每段把**下一个状态**当子目标：`V(z_k → z_{k+1})`，量的是**这条路走了多远**（W）。
- B 族 `value_to_goal` 每一步都用**真正的目标**：`V(z_k → z_g)`，量的是**一路上离目标有多远**。

（已删除的 `discounted_goal_eff_cost` 本是 `value_to_goal_*` 的几何对照，即把学到的 V
换成几何距离 `‖z_k − z_g‖`；用户 2026-09-12 决定不要这个对照。）

**坑（已在代码中防住）**：自含族的 k=5 项就是终点项，若再单加一次终点项会把终点代价重复计权。
代码里 `EFF_SELF_CONTAINED_SCORE_MODES` 走独立分支，只算累计项，不叠终点项。

### 为什么 `value_path_eff_cost` 是推荐项

- 方案原文第 486 行定义 `η = ‖z_g − z_s‖₂ / (Σ_{k=s}^{g−1} ‖z_{k+1} − z_k‖₂ + ε)`，
  分母是**原始步分辨率**的几何累计。
- 但 CEM 只能看到**宏步**（每步 5 个原始步）。在宏步分辨率下用弦长 `‖z_{k+1} − z_k‖`
  会漏掉段内绕行，系统性低估路径长度。
- V 训练的正是"沿数据路径的真实累计 latent 变化"，所以 `V(z_k, z_{k+1})` 才是宏步分辨率下
  对原文分母的**正确推广**；弦长是它的粗糙近似。
- 代码里 `path_efficiency`（P 的优化目标）用的已经是 V 版：
  `costs = value(states[:, :-1], states[:, 1:])`。**选 `value_path_eff_cost` 才能让
  Eff 的 CEM 与 EffPlan 的 P 优化同一个目标**，否则两个方法不可比。

### 落地位置

- `src/tdwm/adapters/effplan.py`：`EFF_SCORE_MODES` / `cumulative_eff_cost()` /
  `EffCEMCost.get_cost`。
- `src/tdwm/evaluation/effplan.py`：放宽原先只允许 `terminal_eff_cost` 的硬门。
- 配置：`evaluation.eff_score` 取上表六个值之一；`eff_cumulative_weight`（默认 1.0，
  仅对累计族生效）。
- 回归测试：`tests/integration/test_effplan_cem_public_api.py`（10 项，含几何解析解校验）。

## F-only 不重跑（用户 2026-09-12 决定）

F-only **不列入本次运行计划**。理由（用户原话要点）：

- F-only 已经有结果了。
- 本方案的所有训练都是"拿以前 pre-trained 的 F 冻结住，再在上面训自己的模块"，
  跟 C 到 G3 系列的做法完全一样。
- 因此协议一样、训练集测试集也一样，F-only 是**共享基线**，直接复用既有数字，
  不需要（也不应该）重跑一遍。

需要补的只有：把既有 F-only 的 O25/O50/O100 数字从 `Results TD.docx` 或服务器产物里
**抄进结果文档**，作为对照行。

**注意（尚未解决）**：本地 `outputs/g_weighted_cem_c1573c1_20260910/matched_f_only/`
虽然名字带 f_only，但 manifest 里是 `method: actor_free_td_lewm_v1_c`、
`score_mode: f_plus_g`，**那是 V1-C 不是纯 F-only**，不能拿来当 F-only 用；
且它的 `o100/` 目录**没有 results.json**。真的 F-only 数字还没有在本地找到。

## 与 C–G3 可比性的三个前提

1. **同一个冻结的 F。** 全部沿用 pre-trained LeWM，冻结后不更新。
2. **同一套训练集/测试集。** RP1 划分 `rp1_cube_8000_2000`：episode 0–7999 训练、
   8000–9999 评测；评测 episode 选择沿用 C–G3 的 selection（O25 `56546fe8…`、
   O50 `e46ea81c…`、O100 `8a87815e…`）。
3. **同一套协议。** O25/O50/O100、预算 2×offset、CEM 300/30/30、RH 见下节。

**未解决的 provenance 风险**：`effplan_cube.yaml` 声明冻结 F 为 `198c468c…`
（pretrained LeWM，`test_actor_free_td_lewm_v1_results.py` 里的 `PRETRAINED_SHA256`），
而 C–G3 系列产物记录的是 `88bd65c4…`（V1-C epoch 10）。两者是不同文件。
若两次加载的 F 权重不一致，Eff 与基线就不可比。**上服务器确认前不要开跑。**

## 基线强制值（2026-09-12 从 V1-C3 实际运行产物提取，非自由选择）

用户要求"和我的其他 C-G3 实验一样的流程"，以下几项因此**被基线钉死**，不是可调项。
来源：`configs/experiment/actor_free_td_lewm_v1_c3_cube_checkpoint_{o25,o50,o100}.yaml`
与 `outputs/g_weighted_cem_v1_cg3_completion_644f12b_20260910/**/launcher_manifest.json`。

| 项 | 基线实际取值 | 证据 |
|---|---|---|
| `render_backend` | `osmesa` | C–G3 正式运行 manifest 里 `MUJOCO_GL: "osmesa"`（不是 egl） |
| `cem_batch_size` | `1` | C–G3 `planning.solver_batch_size: 1` |
| `receding_horizons` | `{"25": 5, "50": 1, "100": 1}` | O25 显式 `receding_horizon: 5`；O50 显式 `1`；O100 继承 O50 的 `1` |
| `target_readout` | `true` | C–G3 `critic: ema_target`、`online_critic_used: false` |
| `episodes_per_protocol` / `planning_seed` | `50` / `42` | C–G3 `evaluation.episodes: 50`、`planning.planning_seed: 42` |
| `episode_budget_multiplier` | `2` | O25=50、O50=100、O100=200 步预算 |

**更正此前的一个错误**：我先前在本文档写 `receding_horizons` 建议 `{"25":5,"50":5,"100":5}`，
并说"历史 O50 是 RH1"。核对基线后确认 **O50 与 O100 都是 RH1，只有 O25 是 RH5**，
且 O25 还额外执行全部五个动作块（`executed_action_block: all_five_blocks`）。这正是之前
"测试协议弄错"的典型来源，已按基线改正。

另一处基线事实：C–G3 的 `inference_objective.score_mode` 是 `state_v_terminal`，即**纯终点
评分**。所以 Eff 相对基线的唯一差别就是本次新增的累计项——这也说明为什么要把它做成可扫的
四种模式，而不是拍一个。

**注意：C–G3 的 `state_critic` 用 `gamma_per_primitive_step: 0.98` + expectile + Huber，
但方案一原文第 481 行明确要求 V 不用折扣、用 MSE，两者不可互相套用。** 方案一的 V 只能按
原文实现，不能照抄 C–G3 的 critic 配置。

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
