# Actor-Free TD-LeWM V1：6×3 结果包与 Results TD 流程

本文只描述 V1（共享冻结 LeWM Action Encoder 的版本）正式结果如何进入归档、曲线图和
`Results TD.docx`。结果层不会从日志猜测数值，不会为未完成的训练或 O50 单元生成
`TBD`、零值或示例值。

raw-action V0 的 C–G3 正式 O50 索引位于
`reports/artifacts/actor_free_td_lewm_v0_cube_seed3072/formal_o50_summary.json`；更早的
7-method bundle 格式仍见 `docs/actor_free_td_lewm_result_bundle.md`。三者不可混称为同一
实验。

## 正式输入包

六种方法固定为 `c`、`d`、`f`、`g1`、`g2`、`g3`；每种方法固定评测
`f_only`、`g_only`、`f_plus_g` 三种分数。

```text
<bundle>/
  training_acceptance.json
  c/
    training_result.json
    training_manifest.json
    execution_evidence.json
    metrics.csv
    f_only/
      results.json
      protocol_manifest.json
      episode_selection.json
      action_normalization.json
    g_only/{同上四个 JSON}
    f_plus_g/{同上四个 JSON}
  d/{同样结构}
  f/{同样结构}
  g1/{同样结构}
  g2/{同样结构}
  g3/{同样结构}
```

训练输入必须是每法 seed 3072 的完整 10-epoch 运行：最终 deployment checkpoint 为
`epoch_10.pt`，`global_step=127960`，validation 未跳过。评测输入必须是正式 O50，不能是
smoke 或 pilot。

归档核心负责验证 checkpoint 身份、V1 action 维度、训练/评测 protocol、数据来源、runtime、
同方法三种评分所用 checkpoint，以及跨 18 项的共同 start–goal selection。绘图和 DOCX
脚本位于归档之后，不绕过这些检查。

先用正式 bundle 生成归档派生物：

```bash
python scripts/archive_actor_free_td_lewm_v1_o50.py --bundle <bundle>
```

只做输入审计、不写派生文件时可加 `--validate-only`。

## 公平性与完整性锁

- 六种方法共享 LeWM、训练 seed 3072、10 epochs 和 127,960 optimizer updates。
- 18 个 O50 共享 50 个 start–goal pair、planning seed 42 和 goal offset 50。
- 正式 CEM 参数为 candidates 300、iterations 30、elites 30、action block 5、episode
  budget 100。
- `f_only`、`g_only`、`f_plus_g` 的 planning horizon 分别为 5、1、5。
- 每项 success rate 必须由 50 个逐 episode 布尔结果重新计算。
- 每法三种 score mode 必须使用同一个正式 `epoch_10.pt`。
- checkpoint 必须记录 V1 method/family/variant、`epoch=10`、`global_step=127960`，以及
  raw action 25D、共享冻结 Action Encoder 192D embedding 和单 predictor 身份。

任一单元缺失、重复、非 finite 或协议不一致时，结果包不完整；不应继续生成图或文档。

## 归档派生物

通过归档核心后，默认派生目录为：

```text
reports/artifacts/actor_free_td_lewm_v1_cube_seed3072/
  summary.json
  paired_outcomes.csv
  training_loss_curves.csv
  training_loss_curves.svg
  README.md
  checksums.sha256
```

其中：

- `summary.json` 是 DOCX 中方法、训练摘要、18 项 O50 和 combined 排名的唯一数值来源；
- `paired_outcomes.csv` 必须有 50 行与 18 个 `success_<variant>__<mode>` 布尔列；
- `training_loss_curves.csv` 必须恰有 60 行，即六种方法 × epochs 1--10；
- SVG 属于可审计归档，PNG 是 DOCX 使用的派生图。

## 训练曲线的两种含义

V1 的训练和 validation 不是同一种 total loss：

- **Train — method objective**：每种方法自己的正式优化目标。C/D/F/G1/G2/G3 的辅助项
  不同，因此曲线只用于检查该方法自己的收敛状态，不能按高低做跨方法排名。
- **Validation — common base TD**：六种方法统一计算的 base TD loss，用于共同的 validation
  监控语义。

正式归档 CSV 使用 `train_method_loss` / `validation_base_td_loss`。绘图脚本也兼容
`train_method_objective` / `validation_common_base_td`、对应 `_loss` 名称，以及旧的
`train_loss` / `validation_loss`。无论列名采用哪一组，图的两块面板都会明确写出上述不同含义。正式归档还会逐行记录
`train_metric_semantics=method_specific_objective` 和
`validation_metric_semantics=common_base_td`，并写出
`cross_method_ranking_metric=false_use_formal_o50_f_plus_g`；绘图脚本若看到这些列，会拒绝其他值。

生成 DOCX 用 PNG：

```bash
python scripts/plot_actor_free_td_lewm_v1_losses.py
```

也可显式指定：

```bash
python scripts/plot_actor_free_td_lewm_v1_losses.py \
  --input reports/artifacts/actor_free_td_lewm_v1_cube_seed3072/training_loss_curves.csv \
  --output reports/artifacts/actor_free_td_lewm_v1_cube_seed3072/training_loss_curves.png
```

绘图脚本要求恰好 60 个 data rows、六个固定 variant、每法 epochs 1--10、所有 loss finite。
不完整 CSV 不会留下部分 PNG。

## 生成 Results TD V1

DOCX builder 接受五项已存在的输入：

1. `summary.json`；
2. `paired_outcomes.csv`；
3. `training_loss_curves.png`；
4. V0 C–G3 的 `formal_o50_summary.json`；
5. 已跟踪且锁定 SHA-256 的旧 7-method `Results TD` DOCX 作为不可变 base。

默认运行：

```bash
python scripts/build_results_td_v1.py
```

默认同时写出：

- 仓库报告副本：`reports/results_td_actor_free_td_lewm_v0_v1_cube_seed3072.docx`；
- 项目根交付副本：`Results TD.docx`。

每次都从锁定的 7-method base 重新生成，再追加 C–G3 V0/V1 对照和 V1 详细页，因此重复
运行不会把同一附录追加多次。项目根 `Results TD.docx` 是最新交付副本；版本身份应以仓库
中的版本化文件名为准。

显式输入/输出示例：

```bash
python scripts/build_results_td_v1.py \
  --summary reports/artifacts/actor_free_td_lewm_v1_cube_seed3072/summary.json \
  --paired reports/artifacts/actor_free_td_lewm_v1_cube_seed3072/paired_outcomes.csv \
  --loss-chart reports/artifacts/actor_free_td_lewm_v1_cube_seed3072/training_loss_curves.png \
  --v0-summary reports/artifacts/actor_free_td_lewm_v0_cube_seed3072/formal_o50_summary.json \
  --base-document reports/results_td_actor_free_cube_seed3072.docx \
  --output reports/results_td_actor_free_td_lewm_v0_v1_cube_seed3072.docx \
  --project-copy "Results TD.docx"
```

只需要仓库副本时可加 `--no-project-copy`。

builder 会在写文件前重新检查：

- summary 声明六种方法、18 个正式评测、seed/epoch/step/O50 元数据完整；
- 每法三种 score mode 与 combined 排名完整；
- paired CSV 恰有 50 行，所有 goal offset 为 50，共享同一个 selection SHA-256；
- 18 个 paired outcome 计数逐项等于 summary 中的 success 数；
- 每法有 10 个真实 loss-curve 点；
- loss chart 是非空 PNG。
- V0/V1 共享 selection SHA、seed、50 pairs、goal offset 和 5/1/5 horizon；
- base DOCX 的 SHA-256 与锁定的旧 7-method 文档一致，并且不会被当作输出覆盖。

`training_acceptance.status` 可以是 `PASS` 或 `PASS_WITH_WARNINGS`。如果是后者，DOCX 会原样
披露该状态，并明确写出 launcher 退出码不可恢复的 provenance warning；不会把它改写成
无条件的 `PASS`。

任一输入缺失或不一致时，两个 DOCX 都不写。脚本不会使用默认成功率、模拟 outcome、占位
排名或“待补充”曲线。

## 报告解释边界

Combined 排名只使用预先规定的 `f_plus_g`，不能在看完结果后从三种 score mode 中挑最好
的一列重排。当前设计仍只有一个 training seed 和一个固定 O50 selection，适合作为结构及
推理评分消融；它不能替代多随机种子均值、方差或统计显著性分析。
