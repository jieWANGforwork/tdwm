# TDWM 实验结果总览

最后更新：2026-09-01（Asia/Shanghai）

本目录汇总当前仓库报告、趋动云下载结果和本地 3090 服务器下载结果。大型日志、数据、
视频和 checkpoint 不进入 Git；本地轻量原始文件位于被 Git 忽略的
`outputs/server_experiments/`。下表中的 success rate 均来自实际 `results.json`，训练指标
来自 `training_result.json`、metrics 或对应的详细报告。

例外是经确认长期保存的轻量 A/C/D 结果归档
[`artifacts/aligned_acd_o50_seed3072/`](artifacts/aligned_acd_o50_seed3072/README.md)：
它只含 start/goal 索引、成功布尔值、哈希和 provenance，不含数据、模型或原始日志。

Actor-Free TD-LeWM 现在有一份统一总账，并保留各阶段报告作为来源说明。统一总账覆盖
**465 个 O50 结果单元、23,250 个逐 episode 布尔 outcome**；所有格使用同一组 50 个
start-goal pair。固定 E10 比较与 E3--E10 checkpoint sweep 在总账中分开标记，不能把
事后最佳 epoch 当成固定 checkpoint 主结果。

| 记录 | 正式范围 | 报告 / 轻量证据 |
| --- | --- | --- |
| **统一 Results TD 总账** | **Legacy + V0 + V1 + V2 + V2-EMA-SG = 465 O50** | [`actor_free_td_lewm_complete_cube_seed3072.md`](actor_free_td_lewm_complete_cube_seed3072.md) / [`results_td_actor_free_td_lewm_complete_cube_seed3072.docx`](results_td_actor_free_td_lewm_complete_cube_seed3072.docx) / [`artifacts/actor_free_td_lewm_complete_cube_seed3072/`](artifacts/actor_free_td_lewm_complete_cube_seed3072/) |
| 较早结构消融 | 7 methods × 3 scores = 21 O50 | [`actor_free_td_lewm_cube_seed3072.md`](actor_free_td_lewm_cube_seed3072.md) / [`artifacts/actor_free_td_lewm_cube_seed3072/`](artifacts/actor_free_td_lewm_cube_seed3072/README.md) |
| C–G3 raw-action V0 | 18 原评分 + 6 first-Q = 24 O50 | [`actor_free_td_lewm_v0_v1_cube_seed3072.md`](actor_free_td_lewm_v0_v1_cube_seed3072.md) / [`formal_o50_summary.json`](artifacts/actor_free_td_lewm_v0_cube_seed3072/formal_o50_summary.json) |
| C–G3 action-encoder V1 | 18 原评分 + 6 first-Q = 24 O50 | [`actor_free_td_lewm_v1_cube_seed3072.md`](actor_free_td_lewm_v1_cube_seed3072.md) / [`artifacts/actor_free_td_lewm_v1_cube_seed3072/`](artifacts/actor_free_td_lewm_v1_cube_seed3072/README.md) |
| V2 joint fine-tune | 144 原评分 epoch cells + 12 E10 新评分 = 156 O50 | 统一总账中的 V2 exact checkpoint trajectory / [`artifacts/actor_free_td_lewm_complete_cube_seed3072/`](artifacts/actor_free_td_lewm_complete_cube_seed3072/) |
| V2-EMA-SG | E3--E10 × 6 methods × 5 scores = 240 O50 | [`actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072.md`](actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072.md) / [`artifacts/actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072/`](artifacts/actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072/) |

统一总账由 `scripts/archive_actor_free_td_lewm_complete.py` 对各阶段已锁定归档和原始
训练/评测证据再次 fail-closed 对账；465/465 格均从 50 个严格布尔 outcome 重算成功数，
并记录 checkpoint、commit、来源路径及 SHA-256。所有训练仍只有 seed 3072；V0/V1/V2
虽然使用同一 selection，但网络更新范围、action 表示和参数量不同，不能把单组百分比
解释为多 seed 总体优劣。

## 结论边界

- 当前几乎所有控制结果都只有一个训练 seed。Aligned A/C/D 已扩展到六个 matched
  planning-selection seeds，但它们不是六个独立训练模型，不能据此声称方法在统计意义上
  优于 baseline。
- O25 与 O50 表示 start--goal offset，属于不同难度和 episode budget 的协议，不能横向
  排名。
- 趋动云 LeWM 48% checkpoint 与 3090 LeWM 72% checkpoint 的训练数据读取、checkpoint
  和运行代码状态不同，不能把二者当作同一次 baseline。
- 部分历史 3090 结果采集时工作树落后远程并包含未提交修改；本次 Aligned 正式评测使用
  锁定提交的干净工作树，其他运行在正式复现前仍须逐项核对 manifest、配置、checkpoint
  SHA-256 和 Git revision。
- 部分旧报告在生成正式 CEM 结果前写成“尚未评测”。本 README 以之后保存的
  `results.json` 为最新事实，并保留旧报告作为训练过程记录。

## Cube：O25 主结果（50 episodes）

| 来源 | 方法/运行 | Success | 成功数 | 耗时 | 解释 |
| --- | --- | ---: | ---: | ---: | --- |
| 3090 | LeWM `seed_3072_3090_episode_stream_formal` | 72% | 36/50 | 64.21 s | 3090 配对比较使用的 baseline |
| 3090 | MC-GT-LeWM | 74% | 37/50 | 62.93 s | 比配对 LeWM 多成功 1 个 episode，证据不足 |
| 3090 | TD-GT-LeWM | 72% | 36/50 | 64.52 s | 与配对 LeWM 持平 |
| 3090 | Joint TD-GT-LeWM | 62% | 31/50 | 66.76 s | checkpoint 初始化/fine-tuning 诊断 |
| 3090 | E2E Joint TD-GT-LeWM world-only | 62% | 31/50 | 65.91 s | 不加入 tail 的诊断 |
| 3090 | E2E Joint TD-GT-LeWM + tail | 56% | 28/50 | 69.34 s | 相对配对 LeWM 下降 16 个百分点 |
| 趋动云 | LeWM，30-iteration CEM | 48% | 24/50 | 144.67 s | 该云端 checkpoint 的规范主结果 |
| 趋动云 | LeWM，10-iteration CEM | 54% | 27/50 | 99.32 s | 非规范早期诊断，不作为主结果 |

MC-GT-LeWM 相对配对 baseline 的变化只有 `+2` 个百分点，即净增 1 个 episode；
E2E 实验的配对统计和失败机制审计见
[`e2e_joint_td_gt_lewm_cube_seed3072.md`](e2e_joint_td_gt_lewm_cube_seed3072.md)。

## Cube：O50 Aligned A/C/D 配对消融（一个 training seed）

固定 training seed 3072，在 planning-selection seeds 42--47 上各运行 50 个 matched
episodes：

| Cell | 定义 | Success | 成功数 |
| --- | --- | ---: | ---: |
| A | Original LeWM world + terminal | 51.0% | 153/300 |
| C | Aligned world + terminal | **56.0%** | **168/300** |
| D | Aligned world + anchored tail | 55.67% | 167/300 |

`C-A` 在 6/6 个 selections 中均为正，总差为 `+5.0 pp`；exact McNemar
`p=0.02007`，作为三个平级 contrasts 之一的 Holm-adjusted `p=0.06022`。`D-C` 为
`-0.33 pp`、exact `p=1.0`，即 tail 在 300 个 episode 中净少成功 1 个。当前证据支持
“Aligned supervision 的主要观测价值来自训练后的 world model”，不支持
“inference-time anchored tail 带来稳定总体增益”。完整报告和机器可读逐 episode 归档见
[`aligned_acd_cube_o50_seed3072_planning_seeds42_47.md`](aligned_acd_cube_o50_seed3072_planning_seeds42_47.md)。

真正的 `2x2` 设计仍缺 Original LeWM coordinates 中训练的 matched anchored tail（B）。
跨坐标 B-prime 为 46%，只能诊断 latent-coordinate incompatibility，不能作为 B。

## Cube：O50 历史/其他结果（多数为 50 episodes）

以下运行都使用 goal offset 50，但方法、checkpoint、planner cost 和部分推理实现仍有差异；
本表是完整结果索引，不是公平排行榜。

| 运行 | Success | 耗时 |
| --- | ---: | ---: |
| Aligned E2E MC-GT-LeWM，epoch 10，planning seed 42（D） | **62%** | 332.36 s |
| LeWM `lewm_seed3072_e10_matched_o50_retry` | 54% | 379.93 s |
| RF-Successor-LeWM，epoch 10 | **32%** | 40.76 s |
| Successor Geometry LeWM，epoch 3 | 46% | 422.45 s |
| Policy-Auxiliary Successor Geometry，epoch 3 | 40% | 463.07 s |
| Residual-Policy Successor Geometry，epoch 3 | 32% | 522.18 s |
| RF Successor Sequence WM，epoch 1 | 24% | 43.12 s |
| RF Balanced Successor Sequence，epoch 1 | 22% | 44.33 s |
| RF Balanced Successor Sequence，epoch 3 | 30% | 41.82 s |
| RF Balanced Successor Sequence，epoch 4 | 38% | 122.82 s |
| RF Balanced Successor Sequence，epoch 4 projected | 34% | 41.07 s |
| RF Balanced Successor Sequence，epoch 5 | 36% | 40.40 s |
| RF E2E Moment Sequence，epoch 1 | 30% | 41.78 s |
| RF E2E Moment Sequence，epoch 2 | 34% | 124.97 s |
| RF E2E Moment Sequence，epoch 2 projected | 36% | 40.05 s |
| RF E2E Moment Sequence，epoch 2 terminal | 34% | 38.11 s |
| RF E2E Moment Sequence，epoch 3 | 30% | 42.24 s |
| RF EMA Manifold Prefix，epoch 1 terminal | 30% | 94.08 s |
| RF Frozen Manifold Prefix，epoch 1 path | 42% | 86.04 s |
| RF Frozen Manifold Prefix，epoch 1 terminal | 44% | 81.68 s |
| RF Frozen Manifold Prefix，validation-fit cost mix | 40% | 510.97 s |
| RF Frozen Manifold Prefix，validation-fit hybrid | 56% | 428.56 s |
| RF Frozen Residual Prefix，epoch 1 terminal | 48% | 430.24 s |
| RF Manifold Prefix，epoch 1 blend | 38% | 89.06 s |
| RF Manifold Prefix，epoch 1 path/default | 40% | 86.59 s |
| RF Manifold Prefix，epoch 1 terminal | 44% | 80.88 s |
| RF Manifold Prefix，epoch 2 path/default | 40% | 86.45 s |
| RF Manifold Prefix，epoch 2 terminal | 30% | 94.10 s |
| RF Manifold Prefix Head Refine，epoch 2 path | 38% | 88.36 s |
| RF Manifold Prefix Head Refine，epoch 2 terminal | 42% | 83.21 s |

另有 Successor Geometry world-only O50 诊断：`40%`（20/50，458.83 s）。
`validation-fit hybrid` 等名称明确表示使用了 validation-fit/post-hoc 选择，不应与预先锁定
的主方法结果混写。

planning seed 42 的旧单次观察为 D `62%`、A `54%`，配对 exact McNemar
`p=0.21875`；该事实保留在
[`aligned_e2e_mc_gt_lewm_cube_seed3072.md`](aligned_e2e_mc_gt_lewm_cube_seed3072.md)。
后续 300-episode 结果改变了核心解释：C 为 `56.0%`，A 为 `51.0%`，D 为
`55.67%`，因此不能继续把 seed-42 的 `+8 pp` 归因于推理 tail。

## 3090 Pilot 与 Smoke 结果

Pilot 只有 10 episodes，方差很大，只用于决定是否继续正式评测：

| Pilot | Success | 成功数 |
| --- | ---: | ---: |
| Residual-Policy Successor Geometry，epoch 1 | 70% | 7/10 |
| Residual-Policy Successor Geometry，epoch 2 | 60% | 6/10 |
| Residual-Policy Successor Geometry，epoch 3 | 70% | 7/10 |
| RF Balanced Successor Sequence，epoch 1 | 60% | 6/10 |
| RF EMA Manifold Prefix，epoch 1 terminal | 60% | 6/10 |
| RF Successor Sequence WM，epoch 1 | 60% | 6/10 |
| Successor Geometry LeWM，epoch 3 | 70% | 7/10 |

Smoke 只有 1 episode，`0%` 或 `100%` 仅表示链路是否运行，不能解释性能：

| Smoke 运行 | Success |
| --- | ---: |
| E2E Joint TD-GT-LeWM EGL | 100% |
| GT-LeWM 3090 EGL | 100% |
| Joint TD-GT-LeWM | 100% |
| Policy-Auxiliary Successor Geometry | 0% |
| Residual-Policy Successor Geometry | 0% |
| RF Balanced Successor Sequence，epoch 1 | 0% |
| RF Balanced Successor Sequence，epoch 3 | 0% |
| RF E2E Moment Sequence，epoch 2 projected | 0% |
| RF E2E Moment Sequence，epoch 2 terminal | 0% |
| RF Manifold Prefix，epoch 2 | 0% |
| RF Successor Sequence WM，epoch 1 | 0% |
| Successor Geometry LeWM，epoch 3 | 0% |

## 主要训练与离线指标

| 运行 | 训练状态 | 关键结果 |
| --- | --- | --- |
| 趋动云 LeWM Cube seed 3072 | epoch 10，127,960 steps | train loss `0.13555546`；validation loss `0.22475678`；validation prediction loss `0.01425450` |
| 3090 LeWM Cube seed 3072 | epoch 10，127,960 steps | train loss `0.09266791`；validation loss `0.64971817`；validation prediction loss `0.00541754` |
| MC-GT-LeWM | epoch 20，4,720 steps | validation MSE `0.00607095`；MAE `0.05518595`；Spearman `0.92718685` |
| TD-GT-LeWM | epoch 19 best / epoch 20 complete，4,720 steps | best validation MC MSE `0.00864504`；epoch-20 Spearman `0.90725171` |
| E2E Joint TD-GT-LeWM | epoch 9 selected，127,960 updates | validation prediction MSE `0.008731`；5-step terminal rollout MSE `0.019560`；TD tail MSE `0.003121` |
| Joint TD-GT-LeWM | 10 epochs，28,830 steps | best validation joint loss `0.00758652` |
| Goal Tail Value v0.1 | 150 steps | best validation MSE `0.07888254`；未接 planner |
| Goal Tail Value v0.1 1000-step | 1,000 steps | best validation MSE `0.06026462`；未接 planner |
| 趋动云 RF Successor Sequence WM | epoch 10，127,960 steps | train loss `0.06427395`；validation loss `0.23557366`；successor sequence loss `0.00111796`；峰值显存约 3.83 GiB |
| 3090 Successor Geometry LeWM | epoch 3，12,000 steps | 正式 O50 `46%`；world-only `40%` |
| 3090 Policy-Auxiliary Successor Geometry | epoch 3，12,000 steps | 正式 O50 `40%` |
| 3090 Residual-Policy Successor Geometry | epoch 3，12,000 steps | 正式 O50 `32%` |
| 3090 Aligned E2E MC-GT-LeWM | epoch 10，127,960 steps | seed-42 D `62%`；六组配对 selections 汇总 A/C/D=`51.0%/56.0%/55.67%` |
| 3090 GT-LeWM v2 | epoch 10，127,960 steps | 训练完成；当前下载结果中没有对应正式 CEM `results.json` |
| 3090 RF-Successor-LeWM | epoch 10，127,960 steps | 正式 O50 `32%`（16/50，40.76 s） |

## 其他环境

### TD-MPC2 CartPole

单 seed、每次 3 episodes。Sparse 是包中官方任务；Dense 是只修改 reward density 的工程
诊断，二者不能直接比较。

| Step | Sparse mean reward | Dense mean reward |
| ---: | ---: | ---: |
| 10k | 0.00 | 371.54 |
| 20k | 0.00 | 770.54 |
| 30k | 0.00 | 832.31 |
| 40k | 0.00 | 833.11 |
| 50k | 0.00 | 835.32 |
| 60k | 0.00 | 863.23 |
| 70k | 0.00 | 857.97 |
| 80k | 0.00 | 820.82 |
| 90k | 4.67 | 853.58 |
| 100k | 0.00 | 864.55 |

Dense `best_model.pt` 独立重载后的 3-episode mean reward 为 `864.4722`。原始逐 episode
结果和 checkpoint 已清理，因此该实验只能作为链路诊断。

### LeWM PushT

仅完成 epoch 0 训练和部分验证，不是完整 baseline：13,933/13,933 train steps；最近训练
loss `0.16631`、prediction loss `0.03154`、SIGReg `1.34375`；当时验证约完成
831/1,549 batches。没有 success rate，且该过渡运行不满足当前只使用安装包公开 API 的
最终合规要求。

## 完整评测文件索引

本地 3090 目录包含 55 个 `results.json`；上文已经覆盖全部 36 个正式/主结果和诊断、
7 个 pilot、12 个 smoke。趋动云另有 2 个 LeWM `results.json`。具体相对路径、协议 manifest
和逐 episode 布尔结果保存在本地忽略目录，并由以下轻量索引记录：

- [`server_3090_experiment_index.md`](server_3090_experiment_index.md)
- [`server_experiment_index.md`](server_experiment_index.md)

新的 A/C/D 结果未并入上述被忽略目录的历史文件计数；其 300-row、可提交轻量归档位于
[`artifacts/aligned_acd_o50_seed3072/`](artifacts/aligned_acd_o50_seed3072/README.md)。

<details>
<summary>完整 training_result.json 索引（57 个）</summary>

### 趋动云 LeWM（18 个）

| 运行 | 类型 | Epoch | Step |
| --- | --- | ---: | ---: |
| `seed_0_loader_w6p2_s50_epoch` | loader/profile | 1 | 50 |
| `seed_3072_baseline_w0_20_v1` | trial | 1 | 20 |
| `seed_3072_blockprefetch512_w2_100_v2` | loader/profile | 1 | 100 |
| `seed_3072_blockseq_w2_100` | loader/profile | 1 | 100 |
| `seed_3072_blockshuffle_w0_100_profile` | loader/profile | 1 | 100 |
| `seed_3072_blockshuffle_w0_20_v2` | loader/profile | 1 | 20 |
| `seed_3072_blockshuffle_w2_100_profile` | loader/profile | 1 | 100 |
| `seed_3072_blockshuffle_w2_b192_100_profile_v2` | loader/profile | 1 | 100 |
| `seed_3072_blockshuffle_w2_pf1_500` | loader/profile | 1 | 500 |
| `seed_3072_blockshuffle_w2_pf2_100` | loader/profile | 1 | 100 |
| `seed_3072_blockshuffle_w2_pf2_500` | loader/profile | 1 | 500 |
| `seed_3072_blockshuffle_w2_pf2_compile_100` | loader/profile | 1 | 100 |
| `seed_3072_blockshuffle_w2_pf2_valblock_resume_e01` | full | 10 | 127,960 |
| `seed_3072_blockshuffle_w4_100_profile` | loader/profile | 1 | 100 |
| `seed_3072_episode_stream_profile100` | loader/profile | 1 | 100 |
| `seed_3072_smoke_episode_stream_smoke_v2` | smoke | 1 | 2 |
| `seed_3072_smoke_scheduler_epoch_w2_pf2` | smoke | 1 | 2 |
| `seed_42_smoke` | smoke | 1 | 2 |

### 趋动云 Successor（2 个）

| 运行 | 类型 | Epoch | Step |
| --- | --- | ---: | ---: |
| `rf_successor_sequence_wm_cube_training/seed_0` | full | 10 | 127,960 |
| `rf_successor_sequence_wm_trial_200/seed_0` | trial | 1 | 200 |

### 3090（37 个）

| 运行 | 类型 | Epoch | Step |
| --- | --- | ---: | ---: |
| `aligned_e2e_mc_gt_lewm_cube_full_v2/seed_3072` | full | 10 | 127,960 |
| `aligned_e2e_mc_gt_lewm_cube_full_v2/seed_3072_smoke` | smoke | 1 | 2 |
| `aligned_e2e_mc_gt_lewm_smoke_resume_v2/seed_3072_smoke` | resume smoke | 2 | 4 |
| `e2e_joint_td_gt_lewm_cube_full/seed_3072` | full | 10 | 127,960 |
| `e2e_joint_td_gt_lewm_cube_full/seed_3072_smoke` | smoke | 1 | 2 |
| `goal_tail_value_cube_v0_1/seed_3072` | trial | - | 150 |
| `goal_tail_value_cube_v0_1_1000/seed_3072` | offline diagnostic | - | 1,000 |
| `gt_lewm_cube_training_3090_100/seed_0` | trial | 1 | 100 |
| `gt_lewm_cube_training_3090_b16/seed_0_smoke` | smoke | 1 | 2 |
| `gt_lewm_cube_training_v2/seed_0` | full | 10 | 127,960 |
| `joint_td_gt_lewm_cube_full_c1ebd96/seed_3072` | full | 10 | 28,830 |
| `joint_td_gt_lewm_cube_smoke/seed_3072` | smoke | 1 | 2 |
| `joint_td_gt_lewm_cube_smoke_batch256/seed_3072` | smoke | 1 | 2 |
| `joint_td_gt_lewm_cube_smoke_bnfix/seed_3072` | resume smoke | 2 | 4 |
| `ls_lewm_cube_training/seed_0_smoke` | smoke | 1 | 2 |
| `mc_gt_lewm_cube_full/seed_3072` | full | 20 | 4,720 |
| `policy_auxiliary_successor_geometry_lewm_cube_seed399_qtricks_v1/seed_399` | full | 3 | 12,000 |
| `policy_auxiliary_successor_geometry_lewm_cube_seed399_qtricks_v1/seed_399_smoke` | smoke | 1 | 2 |
| `residual_policy_successor_geometry_lewm_cube_seed399_qtricks_v1/seed_399` | full | 3 | 12,000 |
| `residual_policy_successor_geometry_lewm_cube_seed399_qtricks_v1/seed_399_smoke` | smoke | 1 | 2 |
| `rf_balanced_successor_sequence_wm_cube_v1/seed_0_smoke` | resume smoke | 2 | 4 |
| `rf_direct_moment_sequence_wm_smoke/seed_0_smoke` | resume smoke | 2 | 4 |
| `rf_e2e_moment_sequence_smoke/seed_0_smoke` | resume smoke | 2 | 4 |
| `rf_ema_balanced_successor_sequence_wm_cube_v1/seed_0_smoke` | resume smoke | 2 | 4 |
| `rf_frozen_manifold_prefix_successor_wm_cube_v1/seed_0` | full | 1 | 12,796 |
| `rf_frozen_residual_prefix_wm_cube_lr5e5_20260824/seed_0` | full | 1 | 12,796 |
| `rf_manifold_prefix_successor_wm_cube_head_refine_v1/seed_0` | full | 2 | 25,592 |
| `rf_manifold_prefix_successor_wm_smoke_v1/seed_0_smoke` | resume smoke | 2 | 4 |
| `rf_successor_lewm_cube_v1/seed_0` | full | 10 | 127,960 |
| `rf_successor_lewm_cube_v1/seed_0_smoke` | resume smoke | 2 | 4 |
| `rf_successor_lewm_cube_v1_preflight/seed_0` | preflight | 1 | 10 |
| `rf_successor_sequence_wm_cube_v1/seed_0_smoke` | resume smoke | 2 | 4 |
| `seed_3072_3090_episode_stream_formal` | full | 10 | 127,960 |
| `seed_3072_smoke_3090_deployment_smoke` | smoke | 1 | 2 |
| `successor_geometry_lewm_cube_seed399_v2/seed_399` | full | 3 | 12,000 |
| `td_gt_lewm_cube_full/seed_3072` | full | 20 | 4,720 |
| `td_gt_lewm_cube_full_resumable/seed_3072` | full/resume copy | 20 | 4,720 |

</details>

## 详细报告

| 报告 | 内容 |
| --- | --- |
| [`baseline_lewm_cube_seed3072.md`](baseline_lewm_cube_seed3072.md) | 趋动云 LeWM Cube 训练和 48%/54% 评测协议 |
| [`mc_gt_lewm_cube_seed3072.md`](mc_gt_lewm_cube_seed3072.md) | MC-GT-LeWM 离线训练指标与审计信息 |
| [`td_gt_lewm_cube_seed3072.md`](td_gt_lewm_cube_seed3072.md) | TD-GT-LeWM 离线训练指标与审计信息 |
| [`e2e_joint_td_gt_lewm_cube_seed3072.md`](e2e_joint_td_gt_lewm_cube_seed3072.md) | E2E 负结果、配对统计和 formulation 偏差审计 |
| [`aligned_acd_cube_o50_seed3072_planning_seeds42_47.md`](aligned_acd_cube_o50_seed3072_planning_seeds42_47.md) | 六组 planning selections、300-episode A/C/D 消融和机器可读归档 |
| [`aligned_e2e_mc_gt_lewm_cube_seed3072.md`](aligned_e2e_mc_gt_lewm_cube_seed3072.md) | planning seed 42 的单次 62%/54% 历史结果与复现信息 |
| [`actor_free_td_lewm_cube_seed3072.md`](actor_free_td_lewm_cube_seed3072.md) | 较早的 7-method × 3-score Actor-Free TD-LeWM 正式消融 |
| [`actor_free_td_lewm_v0_v1_cube_seed3072.md`](actor_free_td_lewm_v0_v1_cube_seed3072.md) | C–G3 raw-action V0 与 action-encoder V1 的共同 6×3 对照 |
| [`actor_free_td_lewm_v1_cube_seed3072.md`](actor_free_td_lewm_v1_cube_seed3072.md) | V1 六法训练、18 个 O50、方法/loss/推理定义与审计边界 |
| [`baseline_tdmpc2_cartpole.md`](baseline_tdmpc2_cartpole.md) | TD-MPC2 CartPole sparse/dense 诊断 |
| [`baseline_lewm_pusht_training.md`](baseline_lewm_pusht_training.md) | 未完成的 PushT 过渡训练记录 |
| [`server_experiment_index.md`](server_experiment_index.md) | 趋动云日志和轻量 artifact 索引 |
| [`server_3090_experiment_index.md`](server_3090_experiment_index.md) | 3090 日志、训练和评测 artifact 索引 |
| [`world_model_research_review.md`](world_model_research_review.md) | 研究背景和方法边界，不是实验结果 |

7×3 legacy bundle 和 6×3 V1 bundle 均已完成正式归档；版本化报告与 artifact 路径见上表。

## 当前总体判断

1. Stable World Model、LeWM、TD-MPC2、checkpoint 恢复和 CEM 评测链路均已实际运行。
2. 单次 3090 O25 结果中，MC-GT-LeWM 为 74%，只比配对 LeWM 的 72% 多一个 episode；
   TD-GT-LeWM 为 72%，没有观测提升。
3. 第一版 E2E Joint TD-GT-LeWM 为 56%，且 world-only 已降至 62%；这是带明确协议偏差的
   负结果，不构成对修正 formulation 的最终证伪。
4. 固定 training seed 3072 的六组 O50 paired selections 中，A/C/D 为
   `51.0%/56.0%/55.67%`。`C-A` 在 6/6 selections 中为正，而 `D-C=-0.33 pp`；当前主要
   观测增益来自 Aligned world-model training，inference-time tail 没有稳定总体收益。
5. O50 successor 系列当前正式结果为 22%--56%，但存在协议、checkpoint、post-hoc 选择
   和实现差异，尚不能据此选出优于 LeWM 的方法。
6. 项目仍缺少受控多 seed、统一 checkpoint、统一 planner 和锁定 episode split 的最终比较；
   因此尚未达到“提出方法优于 baseline”的里程碑。
