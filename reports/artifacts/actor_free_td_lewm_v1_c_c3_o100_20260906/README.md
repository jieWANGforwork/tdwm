# V1-C and V1-C3 formal O100 evidence

This directory is the auditable evidence bundle for seven completed formal
Cube O100 evaluation cells: all six score modes for Actor-Free TD-LeWM V1-C
at epoch 10, plus the V1-C3 epoch-12 State-V + First-Q2 score at alpha 0.10.
It does not contain pilot, smoke, or C4 results.

## Locked protocol

All seven cells use:

- training seed 3072 and planning seed 42;
- the same ordered set of 50 fixed start-goal pairs;
- goal offset 100 and an episode budget of 200;
- OSMesa rendering (`MUJOCO_GL=osmesa`), recorded independently in every job
  log;
- episode-selection file SHA-256
  `8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c`;
- ordered valid-row-ranks SHA-256
  `36994b1ab36656666ff91b379a59829c4b2af150b1f4ed23d409deb5cca9654e`;
- action-normalization SHA-256
  `57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723`.

The launcher manifest reports all seven jobs as `SUCCEEDED` with exit code 0.
Every `results.json` contains exactly 50 Boolean episode outcomes, and its
aggregate success rate agrees with those outcomes.

## Checkpoints

| Method | Epoch | Checkpoint SHA-256 |
|---|---:|---|
| V1-C | 10 | `88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3` |
| V1-C3 | 12 | `5e240053d7c33fc016ef2ff64f3a4a79706dbe10dfde347d5c5f3cd45043e5b2` |

The server checkpoint paths and their identities are preserved in
`checkpoint_manifest.json`; the checkpoint binaries are not duplicated in
this lightweight bundle.

## Formal results

| Method | Checkpoint | Score | Stored score mode | Success |
|---|---:|---|---|---:|
| V1-C | E10 | F-only | `f_only` | 25/50 (50%) |
| V1-C | E10 | G-only | `g_only` | 24/50 (48%) |
| V1-C | E10 | F+G tail | `f_plus_g` | 22/50 (44%) |
| V1-C | E10 | First-Q, alpha=.25 | `f_plus_g_first` | **32/50 (64%)** |
| V1-C | E10 | Mean-Q | `g_only_f_rollout_mean` | 25/50 (50%) |
| V1-C | E10 | First-Q2, alpha=.25 | `f_plus_g_first_q2` | 26/50 (52%) |
| V1-C3 | E12 | State-V + First-Q2, alpha=.10 | `state_v_plus_first_q2` | 25/50 (50%) |

These are paired single-training-seed, single-planning-seed results. The table
reports every formal cell in this bundle; it does not treat the highest value
as a post-hoc selected controller or claim statistical superiority.

## Bundle layout

- `checkpoint_manifest.json` records the two checkpoint identities and the
  locked selection and normalization hashes.
- `formal/_launcher/launcher_manifest.json` records the seven invocations,
  completion states, and aggregate results.
- `formal/_launcher/jobs/*.log` retains the execution logs, including the
  OSMesa backend evidence.
- `formal/o100/` contains one directory per score mode. Each directory keeps
  `results.json`, `protocol_manifest.json`, `episode_selection.json`, and
  `action_normalization.json`.
- `checksums.sha256` covers every file in this directory except the checksum
  file itself.
