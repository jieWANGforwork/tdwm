# Actor-Free TD-LeWM 7×3 结果包格式

`scripts/archive_actor_free_td_lewm_o50.py` 只接受完整、轻量、可审计的服务器原始产物包。
它不会读取 checkpoint、数据集、视频或控制台日志，也不会为缺失单元生成占位结果。

## 目录

```text
<bundle>/
  serial_decoupled/
    training_result.json
    training_manifest.json
    metrics.csv
    f_only/{results.json,protocol_manifest.json,episode_selection.json}
    g_only/{results.json,protocol_manifest.json,episode_selection.json}
    f_plus_g/{results.json,protocol_manifest.json,episode_selection.json}
  serial_coupled/...
  hybrid/...
  parallel_real/...
  goal_hybrid/...
  imaginary_hybrid/...
  direct_goal_hybrid/
    training_result.json
    training_manifest.json
    metrics.csv
    f_only/{results.json,protocol_manifest.json,episode_selection.json}
    c_only/{results.json,protocol_manifest.json,episode_selection.json}
    f_plus_c/{results.json,protocol_manifest.json,episode_selection.json}
```

三个训练文件必须是训练器原样写出的文件。归档器自行计算其 SHA-256，不接受手填的
`training_summary.json`、checkpoint 哈希或整理后的曲线。评测目录中的三个 JSON 同样必须
来自 evaluator，不能重新计算或手工改写。

## 原始训练文件验收

`training_result.json` 必须记录方法、variant、seed 3072、`final_epoch=10`、
`global_step=127960`、run directory、Lightning `last.ckpt` 和峰值 CUDA 显存。
`training_manifest.json` 必须记录 StableWM 0.1.1、训练 Git revision、完整 protocol、
数据/模型/参数量，以及以下正式训练状态：

- 10 epochs；
- 每 epoch 12,796 optimizer steps；
- 总计 127,960 optimizer steps；
- validation 未跳过；
- deployment checkpoint version 1。

每个 variant 的 `training_manifest.protocol` 还必须与仓库中对应的锁定训练 YAML 完整
canonical hash 一致。因此检查不只覆盖共享字段，也覆盖 variant 专属语义：Serial 是否
detach、Hybrid real/predicted TD 权重、Goal readout 与 goal loss、Imaginary bootstrap、
Direct critic/joint objective，以及 head gamma、EMA 和 warm-up 等。专属字段只与该
variant 的 YAML 比较，不会错误地跨 variant 求同。

训练 dataset 的 `split` 必须使用训练器 `save_split` 的真实结构：`train_samples`、
`validation_samples`、`train_indices_sha256`、`validation_indices_sha256` 和 `path`。
前两个样本数之和必须等于 `sequence_samples`；四个语义字段必须在七个训练运行中一致。
`path` 是 run-specific 绝对路径，只检查其存在，不进入跨运行指纹。

`metrics.csv` 使用 Lightning 原始稀疏 CSV。归档器从每个 zero-based epoch `0..9` 的
`train/loss_epoch` 与 `validation/loss` 各提取唯一一个 aggregate；二者必须落在该 epoch
最后的 zero-based step（12,795、25,591、…、127,959）。归档器自动生成面向报告的
epoch 1--10 曲线、epoch-10 指标和最低 validation 指标。列缺失、重复 aggregate、少于
10 点、NaN/Inf、负 loss 或 step 不一致都会被拒绝。

不同方法的 total loss 包含数量和定义不同的 auxiliary loss。曲线只用于判断每个方法
自己的收敛状态，不能按曲线高低比较方法性能。

## 评测与公平性锁

- `results.json.metrics.episode_successes` 是逐 episode 结果的规范字段；兼容旧字段
  `metrics.success`。若两者同时存在，必须逐项完全一致。
- success rate 必须能由 50 个逐 episode 布尔值重新计算得到。
- 21 个运行必须共享 runtime 关键版本、图像预处理、数据集、模型、world、evaluation、
  planning、action normalization 和 world-model 参数量的完整指纹。planning 锁包含
  `solver=CEM` 与 `history_len=1`。
- 每个 variant 的三个 score mode 必须共享完整 formal protocol、同一路径/同一 SHA-256
  checkpoint 和同一 head 参数量；Goal/Imaginary/Direct 等 variant 专属字段不会被错误地
  要求跨方法相同。
- 数据来源指纹包含 path、format、size 和 Lance conversion manifest provenance。
- 每个评测 checkpoint 路径必须对应训练 run directory 下的
  `checkpoints/actor_free_td_lewm/<variant>/epoch_10.pt`，参数量也必须与训练 manifest 一致。
- 固定 selection 必须是 StableWM 0.1.1 Cube seed 42 的 O50 重算结果，且
  `episode_selection.json` 的精确字节 SHA-256 为
  `e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`。每行还必须满足
  `episode < 10000`、`0 <= start < goal < 201` 和 `goal-start=50`。

旧版 evaluator 没有 `score_mode` 字段时，只允许原始 combined 结果放入 `f_plus_g` 或
`f_plus_c`；其余模式必须带显式字段。`formal_protocol.inference_objective.score_mode` 只能
缺失（旧 evaluator）或保留 variant 的 combined mode，不能被改成 `f_only`、`g_only`
或 `c_only`；实际运行 mode 只出现在 configured protocol 和结果元数据中。

## 运行

```bash
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle> --validate-only
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle>
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle> --check
```

生成结果包括：

- `reports/actor_free_td_lewm_cube_seed3072.md`；
- `reports/artifacts/actor_free_td_lewm_cube_seed3072/summary.json`；
- `paired_outcomes.csv`（50 行、21 个 success 列）；
- `training_loss_curves.csv` 与可嵌入文档的 `training_loss_curves.svg`；
- `README.md` 与 `checksums.sha256`。

需要 PNG 时，可在归档生成后运行：

```bash
python scripts/plot_actor_free_td_lewm_losses.py --output <curves.png>
```

SVG 是标准归档的一部分并进入校验文件；PNG 只是为特定文档渲染器生成的派生文件。
