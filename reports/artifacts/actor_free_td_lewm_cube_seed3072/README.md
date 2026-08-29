# Actor-Free TD-LeWM Cube O50 7×3 可审计归档

该目录由服务器导出的完整轻量结果包生成。它包含 7 个方法 × 3 种推理分数的
同一 O50 selection 配对结果，以及训练器原始 JSON/metrics.csv 的派生摘要；不包含
数据集、checkpoint、图像、视频或控制台日志。

## 文件

- `summary.json`：combined 排名、21 个汇总结果、训练摘要、runtime 和来源文件哈希。
- `paired_outcomes.csv`：50 个固定 pair × 21 个 success 布尔列。
- `training_loss_curves.csv`：7 个方法 × 10 epochs 的统一 train/validation total loss。
- `training_loss_curves.svg`：可直接嵌入报告/文档的两面板曲线图。
- `checksums.sha256`：归档器生成文件与上级完整报告的 SHA-256。
- `../../actor_free_td_lewm_cube_seed3072.md`：人类可读的完整 Results TD 报告。

## Combined 排名

1. Hybrid: 27/50 (54%, f_plus_g)
2. Serial Decoupled: 23/50 (46%, f_plus_g)
2. Imaginary Hybrid: 23/50 (46%, f_plus_g)
4. Parallel Real: 22/50 (44%, f_plus_g)
5. Serial Coupled: 20/50 (40%, f_plus_g)
6. Goal Hybrid: 19/50 (38%, f_plus_g)
6. Direct Goal Critic Hybrid: 19/50 (38%, f_plus_c)

同 success 数使用同一名次。排名只使用 `f_plus_g` / `f_plus_c`，不会按三列中的
最佳 post-hoc 数值重新排序。

## 输入包目录

```text
<bundle>/<variant>/training_result.json
<bundle>/<variant>/training_manifest.json
<bundle>/<variant>/metrics.csv
<bundle>/<variant>/<score_mode>/results.json
<bundle>/<variant>/<score_mode>/protocol_manifest.json
<bundle>/<variant>/<score_mode>/episode_selection.json
```

Successor variants 使用 `f_only/g_only/f_plus_g`；`direct_goal_hybrid` 使用
`f_only/c_only/f_plus_c`。旧 evaluator 只有 combined 没有显式 `score_mode` 字段，
归档器只允许它进入 combined 单元，并在 summary 中标记
`legacy_combined_default`；非 combined 单元必须显式记录 mode。
Formal protocol 的 mode 只能缺失或保留 combined；F/G/C-only 只允许出现在 configured protocol。

## 重建与验证

从仓库根目录运行：

```bash
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle>
python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle> --check
python scripts/plot_actor_free_td_lewm_losses.py --output <curves.png>
cd reports/artifacts/actor_free_td_lewm_cube_seed3072
shasum -a 256 -c checksums.sha256
```

归档器会从原始训练文件自行计算 SHA-256、10-epoch 曲线和最终/最佳 validation；
拒绝不完整的 7×3 bundle、smoke/pilot、非 O50、selection 非固定 seed-42 文件、同方法
不同 checkpoint、训练 export 路径不匹配、任何公平协议/runtime/数据来源指纹漂移，
或 success rate 与逐 episode 结果不一致。

## 选择与哈希

共同 `episode_selection.json` SHA-256：`e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`。

`pair_hash` 是只含 `episode_index`、`goal_step`、`start_step` 的 compact、
key-sorted JSON 的 SHA-256。来源 JSON 的 exact byte SHA-256 保存在 `summary.json`。
