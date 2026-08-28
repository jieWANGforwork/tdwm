# Actor-Free TD-LeWM 7×3 结果包格式

`scripts/archive_actor_free_td_lewm_o50.py` 只接受完整、轻量、可审计的服务器导出包。
它不会读取 checkpoint、数据集、视频或原始日志，也不会为缺失单元生成占位结果。

## 目录

```text
<bundle>/
  serial_decoupled/
    training_summary.json
    training_curve.csv
    f_only/{results.json,protocol_manifest.json,episode_selection.json}
    g_only/{results.json,protocol_manifest.json,episode_selection.json}
    f_plus_g/{results.json,protocol_manifest.json,episode_selection.json}
  serial_coupled/...
  hybrid/...
  parallel_real/...
  goal_hybrid/...
  imaginary_hybrid/...
  direct_goal_hybrid/
    training_summary.json
    training_curve.csv
    f_only/{results.json,protocol_manifest.json,episode_selection.json}
    c_only/{results.json,protocol_manifest.json,episode_selection.json}
    f_plus_c/{results.json,protocol_manifest.json,episode_selection.json}
```

评测目录中的三个 JSON 必须是 evaluator 原样写出的文件，不能重新计算或手工改写。旧版
evaluator 没有 `score_mode` 字段时，只允许原始 combined 结果放入 `f_plus_g` 或
`f_plus_c`；其余模式必须带显式字段。

## `training_summary.json`

每个方法需要一个由服务器训练产物提取的摘要：

```json
{
  "schema_version": 1,
  "method": "actor_free_td_lewm",
  "variant": "<variant>",
  "seed": 3072,
  "status": "complete",
  "epochs_completed": 10,
  "global_step": 127960,
  "training_commit": "<7-to-40-character-lowercase-git-revision>",
  "checkpoint_sha256": "<64-character-lowercase-sha256>",
  "runtime": {
    "stable_worldmodel": "0.1.1",
    "cuda_device": "<recorded-device-name>"
  },
  "metrics": {
    "final_epoch": {
      "epoch": 10,
      "train/loss": "<finite-number>",
      "validation/loss": "<finite-number>"
    },
    "best_validation": {
      "epoch": "<integer-1-through-10>",
      "metric": "validation/loss",
      "value": "<finite-number>"
    }
  },
  "source_files": {
    "training_result.json": "<sha256>",
    "training_manifest.json": "<sha256>",
    "metrics.csv": "<sha256>"
  }
}
```

尖括号表示待导出的真实值；不得把示例字符串留在正式包中。`checkpoint_sha256` 必须等于
该方法三次评测 manifest 中的 checkpoint SHA-256。

## `training_curve.csv`

每个方法恰好 10 行，epoch 使用面向报告的 1--10 编号：

```csv
epoch,train_loss,validation_loss
1,<epoch-1-train-total-loss>,<epoch-1-validation-total-loss>
...
10,<epoch-10-train-total-loss>,<epoch-10-validation-total-loss>
```

epoch 10 的两项值必须与 `training_summary.json.metrics.final_epoch` 一致；最低 validation
loss 的 epoch/value 必须与 `best_validation` 一致。曲线中的 loss 是每个方法自己的 total
loss。由于 auxiliary loss 的数量和定义不同，它只用于检查单个方法是否收敛，不能跨方法
比较绝对高低。

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

SVG 是标准归档的一部分并进入校验文件；PNG 只作为特定文档渲染器的派生文件。
