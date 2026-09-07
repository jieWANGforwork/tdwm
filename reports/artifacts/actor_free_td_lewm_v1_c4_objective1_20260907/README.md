# V1-C4 objective-v1 formal evidence

This directory archives the completed V1-C4 training run and its preregistered O25/O50/O100 evaluation matrix. It supersedes the earlier objective-v0 C4 run without deleting that historical evidence.

## Method contract

C4 keeps the V1-C training protocol and changes only the route by which action affects the successor head. The observation encoder, action encoder, and LeWM predictor `F` are frozen; their outputs are stop-gradient. The optimizer contains only the online state-only successor head `G_C4`, with a frozen EMA target head.

For transition `i`, the online head receives the post-action predicted state

`x_i = stop_gradient(F(E(o_i), a_i))`.

Its TD target is

`Y_i = stop_gradient(E(o_(i+1)) + gamma * (1-d_i) * Gbar_C4(F(E(o_(i+1)), a_(i+1)), m))`.

The optimized loss is

`L_C4 = mean(||G_C4(x_i,m)-Y_i||^2) + mean_goal(((G_C4(x_i,m)-Y_i)^T m)^2)`.

Random-task samples participate only in vector TD; goal-derived samples participate in both terms. The terminal mask makes `Y_i = E(o_(i+1))` when the reached state is terminal. There is no actor and no action input to either online or target `G_C4`.

## Training identity

- Seed: 3072
- Epochs: 10
- Optimizer updates: 127,960
- Frozen LeWM checkpoint SHA-256: `198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257`
- Frozen world-model state SHA-256: `0ef286ac11ae41fbda29b97b781c0bf81332a5ceed1924d0d21d5756566bbe25`
- Deployed epoch-10 C4 checkpoint: `/root/autodl-tmp/tdwm/outputs/actor_free_td_lewm_v1_c4_ea1f64b_obj1_20260907/seed_3072/checkpoints/actor_free_td_lewm_v1_c4/c4/epoch_10.pt`
- Deployed checkpoint SHA-256: `ae4112aa8ca9810040bb23ad9ee314445ea003c264985cd9ce3070f61c3f2c7d`
- Final train losses: vector 1,804.82; goal 38,950.51; total 40,755.39
- Final validation losses: vector 1,590.51; goal 16,062.43; total 17,652.91
- Best validation total: 17,637.54 at epoch 9; the preregistered epoch-10 checkpoint was evaluated.

## Formal evaluation

Every cell uses planning seed 42 and the same 50 fixed start-goal pairs within its protocol. All six score modes were fixed before evaluation.

| Protocol | F-only | C4-only | F+C4 tail | First-Q | Mean-Q | First-Q2 |
|---|---:|---:|---:|---:|---:|---:|
| O25 | 37/50 (74%) | 32/50 (64%) | 38/50 (76%) | 32/50 (64%) | 31/50 (62%) | 34/50 (68%) |
| O50 | 26/50 (52%) | 19/50 (38%) | 24/50 (48%) | 22/50 (44%) | 23/50 (46%) | 25/50 (50%) |
| O100 | 25/50 (50%) | 25/50 (50%) | 24/50 (48%) | 27/50 (54%) | 28/50 (56%) | 28/50 (56%) |

Exact paired changes relative to the same-run, same-backend F-only baseline are:

| Protocol | C4-only New/Lost | Tail New/Lost | First-Q New/Lost | Mean-Q New/Lost | First-Q2 New/Lost |
|---|---:|---:|---:|---:|---:|
| O25 | 3/8 | 5/4 | 3/8 | 3/9 | 2/5 |
| O50 | 3/10 | 3/5 | 2/6 | 3/6 | 3/4 |
| O100 | 7/7 | 5/6 | 6/4 | 6/3 | 8/5 |

C4 is horizon-dependent rather than a uniform improvement. Of 15 non-baseline cells, four improve, one ties, and ten regress. O25 tail adds one net success; O50 has no C4-assisted winner; O100 Mean-Q and First-Q2 add three net successes. None of the paired changes is statistically significant at 0.05 with this single training seed and one planning selection.

## Backend audit

The finalized C4 evaluation used EGL. Historical V1-C evidence used OSMesa, so historical C4-versus-V1-C differences are descriptive, not causal. Fresh V1-C F-only rechecks under the same EGL runtime exactly match C4 F-only episode by episode: O25 37/50, O50 26/50, and O100 25/50. The frozen LeWM configurations are identical and all 303 stored predictor tensors match exactly.

## Recovery record

The original parallel launcher completed 17 of 18 cells. Its O100 `f_plus_g` worker stalled after more than three hours and was terminated only after an isolated, unchanged retry completed. The recovered formal root combines the 17 original cells with that retry. `formal/recovery/recovery_validation_manifest.json` records all 18 validated result, protocol, checkpoint, selection, and action-normalization hashes. `formal/launchers/original_formal.exit` remains `1` intentionally so the incomplete original launcher is never misrepresented as successful.

## Directory map

- `training/`: training manifest, raw metrics, epoch losses, loss curve, logs, and summary.
- `formal/results/`: all 18 result directories, including per-episode outcomes and protocol manifests.
- `formal/summary/`: machine-readable aggregate JSON, episode matrix CSV, and Markdown summary.
- `formal/recovery/`: isolated retry evidence and the 18-cell recovery validation manifest.
- `formal/launchers/`: original launch manifests, exit status, and individual job logs.
- `runtime_audit/`: same-EGL V1-C rechecks and historical OSMesa logs.
- `checksums.sha256`: byte-level integrity manifest for this evidence bundle.

The canonical narrative report is `reports/actor_free_td_lewm_complete_cube_seed3072.md`; the matching Word report is `reports/results_td_actor_free_td_lewm_complete_cube_seed3072.docx` and the project-facing copy is `Results TD.docx`.
