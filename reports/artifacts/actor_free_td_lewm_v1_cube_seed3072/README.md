# Actor-Free TD-JEPA V1 Cube O50 machine-readable archive

This directory is generated from a validated six-method by three-score-mode
bundle.  It contains no checkpoints, datasets, videos, or full console logs.

- `summary.json`: methods, 18 formal scores, protocol/checkpoint/source hashes,
  loss semantics, and explicit trainer provenance gaps.
- `paired_outcomes.csv`: the locked 50 start-goal pairs and 18 success columns.
- `training_loss_curves.csv`: 6 x 10 epoch diagnostics.  Train is each method's
  objective; validation is the common base TD.
- `training_loss_curves.svg`: deterministic report visualization.
- `checksums.sha256`: hashes for generated archive files and the Markdown report.

Locked selection SHA-256: `e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`.
Locked action-encoder state SHA-256: `2657b55140013b4b071cd8cdea63f1eac5c65c498d55331c7499744ef31a9cd3`.

F-only and F+G use horizon 5.  G-only uses horizon 1.  Ranking is by F+G only.
Missing CUDA peak memory and trainer CUDA-device fields were not fabricated;
their external evidence hashes are recorded in `summary.json`.
