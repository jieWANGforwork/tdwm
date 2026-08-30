# 服务器实验产物索引

首次记录：2026-08-26；最后更新：2026-08-31（Asia/Shanghai）

## AutoDL：较早 Actor-Free TD-LeWM 7×3 正式归档

Actor-Free TD-LeWM 当前比较范围固定为 7 个方法：`serial_decoupled`、
`serial_coupled`、`hybrid`、`parallel_real`、`goal_hybrid`、`imaginary_hybrid` 和
`direct_goal_hybrid`。前 6 个 Successor 方法各收集 `f_only/g_only/f_plus_g`；Direct
Goal Critic 收集 `f_only/c_only/f_plus_c`，合计 21 个正式 O50。

服务器导出目录需使用以下轻量结构：

```text
<bundle>/<variant>/training_result.json
<bundle>/<variant>/training_manifest.json
<bundle>/<variant>/metrics.csv
<bundle>/<variant>/<score_mode>/results.json
<bundle>/<variant>/<score_mode>/protocol_manifest.json
<bundle>/<variant>/<score_mode>/episode_selection.json
```

三个训练文件必须由 trainer 原样导出；不接受人工 `training_summary.json` 或整理后的
曲线。归档器从 Lightning `metrics.csv` 的 `train/loss_epoch` 与 `validation/loss` 自动
抽取完整 epochs 1--10 曲线并计算三个原始文件的 SHA-256。正式归档前运行：

```bash
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle> --validate-only
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle>
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle> --check
```

硬性验收包括：7×3 文件齐全；全部为 O50 且非 smoke/pilot；selection 必须能按 StableWM
0.1.1 seed 42 重算且精确 SHA-256 为 `e46ea81c…ee7`；每方法 3 种 mode 使用同一路径、
同一 SHA-256 checkpoint，且路径对应训练器 epoch-10 export；逐 episode canonical
`metrics.episode_successes` 与汇总 rate 一致；10 个 epoch aggregate、最终 step 与
trainer 的 127,960 global steps 一致。21 个运行还必须共享完整正式 runtime/image/dataset/
model/world/evaluation/planning、关键软件版本、数据 format/size/conversion provenance、
action normalization 和 world 参数量指纹。每个训练 protocol 必须完整匹配其 variant 的
锁定 YAML；七个 split 的 train/validation 样本数与索引 SHA-256 必须相同，绝对 path
不进入指纹。旧版 evaluator 没有显式 score-mode 字段时，只允许其 combined 结果进入
`f_plus_g/f_plus_c`；formal protocol 的 mode 也只能缺失或为 combined，不能把它解释为
F-only 或 G/C-only。

21 个正式 O50 已完成并归档到
[`actor_free_td_lewm_cube_seed3072.md`](actor_free_td_lewm_cube_seed3072.md) 与
[`artifacts/actor_free_td_lewm_cube_seed3072/`](artifacts/actor_free_td_lewm_cube_seed3072/README.md)。
checkpoint、数据、视频和原始日志继续只保存在外部服务器 artifact 目录。

## AutoDL：C–G3 raw-action V0 / action-encoder V1

后来单独实现的 controlled study 固定为 `c,d,f,g1,g2,g3` 六种方法；它不是上面
7-method 结构消融的改名。V0 将 normalized raw 25D action block 直接送入 G；V1 先通过
预训练 LeWM 的共享冻结 Action Encoder 得到 192D embedding。两版均无 Actor。

- V0 output root：`/root/autodl-tmp/tdwm/outputs/actor_free_td_lewm_v0_cg3_79706d3_20260830`
- V1 repo root：`/root/autodl-tmp/tdwm-v1-3c4e62e`
- V1 output root：`/root/autodl-tmp/tdwm/outputs/actor_free_td_lewm_v1_cg3_3c4e62e_20260830`
- V1 训练提交：`3c4e62ef2ab72387536433f27ef11bce75477e7e`
- V1 训练验收：`training_acceptance.json` 为 `PASS_WITH_WARNINGS`；六个 warning 仅说明
  原 launcher exit code 在进程被回收后不可恢复，checkpoint、10 epochs、127,960 steps、
  metrics 和有限值检查均通过。

两版各完成 18 个正式 O50，使用同一 50-pair selection SHA-256
`e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`。V0 F+G 最好为
G1/G2 的 50%；V1 F+G 最好为 G3 的 54%。完整共同表见
[`actor_free_td_lewm_v0_v1_cube_seed3072.md`](actor_free_td_lewm_v0_v1_cube_seed3072.md)，
V1 完整训练/评测归档见
[`actor_free_td_lewm_v1_cube_seed3072.md`](actor_free_td_lewm_v1_cube_seed3072.md) 与
[`artifacts/actor_free_td_lewm_v1_cube_seed3072/`](artifacts/actor_free_td_lewm_v1_cube_seed3072/README.md)。

## AutoDL 云端 RTX 4080：Aligned A/C/D 补充评测

2026-08-27 的 paired ablation 使用代码目录 `/root/tdwm`、数据目录
`/root/autodl-tmp/tdwm/data/lewm-cube/` 和结果目录
`/root/autodl-tmp/tdwm/outputs/diagnostics/`。云端 RTX 4080 承担 planning-selection
seeds 44、46；每个 seed 内 A/C/D 的 selection 文件完全一致。

| Planning seed | A：Original terminal | C：Aligned terminal | D：Aligned + tail |
| ---: | ---: | ---: | ---: |
| 44 | 44% (22/50) | 46% (23/50) | 46% (23/50) |
| 46 | 48% (24/50) | 52% (26/50) | 58% (29/50) |

同一云端还运行了两个隔离诊断：

- B-prime（Original world + Aligned-coordinate tail）：`46%`（23/50）。该组合存在
  latent-coordinate mismatch，不能作为干净 factorial cell B。
- Aligned planner O5：保留 124 条 iteration-0/29 诊断记录；terminal-total rank
  correlation `0.92978`，最终 iteration 改变最佳 candidate `27.42%`，boundary maximum
  为 `0.0`。O5 不用于 success-rate 声明。

RTX 3090 承担 seeds 42、43、45、47。六组跨服务器汇总和逐 episode 机器可读归档见
[`aligned_acd_cube_o50_seed3072_planning_seeds42_47.md`](aligned_acd_cube_o50_seed3072_planning_seeds42_47.md)
及 [`artifacts/aligned_acd_o50_seed3072/`](artifacts/aligned_acd_o50_seed3072/README.md)。
GitHub 只保存轻量 CSV/JSON 和哈希，不保存数据集、checkpoint 或原始日志。

## 范围与保存策略

本索引记录通过 SSH 从服务器读取到的实验产物。服务器侧来源为：

- `/gemini/code/tdwm/outputs/lewm_cube_training/`
- `/gemini/code/tdwm-successor-run/outputs/`

原始日志、metrics、manifest 和结果 JSON 已保存到本地被 Git 忽略的目录：

- `outputs/server_experiments/tdwm/`
- `outputs/server_experiments/tdwm_successor/`

本次本地下载目录中**没有 checkpoint 文件**（无 `.ckpt`、`.pt` 或 `.pth`）。原始大型
产物不进入 GitHub；GitHub 保存本索引、轻量汇总，以及经确认归档的 A/C/D 选择索引、
成功布尔值和哈希。

## 当前服务器保留的原始日志

- LeWM/Cube：39 个日志文件，包含 smoke、DataLoader/Lance 读取、block shuffle、
  prefetch、compile、scheduler、正式训练和两次 CEM 评测。
- Reward-free successor sequence WM：1 个训练日志，包含 Cube 10-epoch 训练。
- 合计：40 个日志文件。

### LeWM/Cube 日志清单

相对于服务器目录 `/gemini/code/tdwm/outputs/lewm_cube_training/`：

```text
loader_w12p2_s50_epoch.log
loader_w6p1_s50_epoch.log
loader_w6p2_s100.log
loader_w6p2_s50_epoch.log
loader_w8p2_s50_epoch.log
logs/baseline_w0_20_v1.console.log
logs/blockprefetch512_w2_100.console.log
logs/blockprefetch512_w2_100_v2.console.log
logs/blockseq_w2_100.console.log
logs/blockshuffle_w0_100_profile.console.log
logs/blockshuffle_w0_20.console.log
logs/blockshuffle_w0_20_v2.console.log
logs/blockshuffle_w2_100_profile.console.log
logs/blockshuffle_w2_b192_100_profile.console.log
logs/blockshuffle_w2_b192_100_profile_v2.console.log
logs/blockshuffle_w2_pf1_500.console.log
logs/blockshuffle_w2_pf2_100.console.log
logs/blockshuffle_w2_pf2_500.console.log
logs/blockshuffle_w2_pf2_compile_100.console.log
logs/blockshuffle_w4_100_profile.console.log
logs/seed_3072_blockshuffle_w2_pf2_formal.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.evaluation.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.evaluation_cem30.console.log
logs/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01.evaluation_egl.console.log
logs/seed_3072_scheduler_epoch_w2_pf2.console.log
logs/seed_3072_smoke_optimizer_step_fixed.console.log
logs/seed_3072_smoke_scheduler_epoch_w2_pf2.console.log
logs/smoke_resume_seed_42.log
logs/smoke_seed_3072.log
logs/smoke_seed_42.log
logs/stage_cube_to_code_and_train_seed_0.log
logs/train_seed_0.log
seed_3072_episode_stream_formal_v1/stdout.log
seed_3072_formal_compiled/console-restart.log
seed_3072_formal_compiled/console.log
seed_3072_formal_eager/console.log
seed_3072_formal_eager_w0/console.log
train_seed_0_loader_w6p1_formal.log
```

### Successor sequence WM 日志

```text
/gemini/code/tdwm-successor-run/outputs/logs/rf_successor_sequence_wm_seed0.log
```

## 关键可复核结果

### LeWM Cube：seed 3072

产物目录：

`/gemini/code/tdwm/outputs/lewm_cube_training/seed_3072_blockshuffle_w2_pf2_valblock_resume_e01/`

- 10 epochs，`global_step=127960`
- `train/loss=0.13555546`
- `train/prediction_loss=0.01771848`
- `validation/loss=0.22475678`
- `validation/prediction_loss=0.01425450`
- 50-episode、30-iteration CEM：`48.0%`
- 早期非规范 EGL 诊断：`54.0%`
- 参数量：18,034,628

对应评测结果文件位于两个 `evaluation_o25_*` 目录的 `results.json` 中。54% 只保留为
诊断结果，正式主结果是 30-iteration CEM 的 48%。

### Reward-free successor sequence WM：Cube seed 0

产物目录：

`/gemini/code/tdwm-successor-run/outputs/rf_successor_sequence_wm_cube_training/seed_0/`

- 10 epochs，`global_step=127960`
- `train/loss=0.06427395`
- `validation/loss=0.23557366`
- `train/successor_sequence_loss=0.00111796`
- successor MSE h1/hK：`0.00504898 / 0.00104171`
- 峰值 CUDA 显存：约 3.83 GiB
- 当前没有对应的 policy/CEM success-rate 结果。

## 只有报告、当前服务器没有原始日志的实验

以下记录在仓库报告中，但报告注明原始日志或 checkpoint 已清理/保存在其他外部
artifact 位置，不能当作本次下载到的 raw log：

- [`reports/mc_gt_lewm_cube_seed3072.md`](mc_gt_lewm_cube_seed3072.md)：MC-GT-LeWM
  value head，20 epochs；尚未做 CEM。
- [`reports/td_gt_lewm_cube_seed3072.md`](td_gt_lewm_cube_seed3072.md)：TD-GT-LeWM
  value head，20 epochs；尚未做 CEM。
- [`reports/e2e_joint_td_gt_lewm_cube_seed3072.md`](e2e_joint_td_gt_lewm_cube_seed3072.md)：
  E2E Joint TD-GT-LeWM，报告的 baseline/world-only/tail 结果为 72%/62%/56%。
- [`reports/baseline_tdmpc2_cartpole.md`](baseline_tdmpc2_cartpole.md)：TD-MPC2
  CartPole sparse/dense 诊断。
- [`reports/baseline_lewm_pusht_training.md`](baseline_lewm_pusht_training.md)：LeWM
  PushT，仅记录到第 0 个 epoch 的中途状态。

## 解释边界

服务器代码中还有尚未产生当前 raw output 的 experiment 配置，因此本索引只代表当前
服务器上实际保留并成功读取到的日志和轻量结果，不代表所有曾经启动过的任务或所有
研究设想均有可恢复 artifact。服务器侧 `/gemini/code/tdwm` 工作树在采集时不是干净
工作树，正式比较前仍需核对代码版本和未提交修改。
