# Fixed-four-node P from 5/10/20-action source spans

This is an opt-in P-training experiment, not variable-output P and not a new V.
Use `configs/experiment/effplan_fixed4_discrete_5_10_20_old_v.yaml` with the
existing `scripts/train_effplan.py` entrypoint. It retains the original frozen
total-work G/V training identity from `effplan_cube_stable_p_sparse_v1.yaml`.
Pass the existing paired G/V checkpoint and its completed training manifest;
the loader verifies settings, parameterization, source identity and update count,
and records the actual checkpoint SHA. Do not run G/V training for this experiment.

## Sampling

An action here is a five-primitive-action block. Choose the source span uniformly
from exactly {5, 10, 20}; choose uniformly an eligible episode, then a legal
anchor. A full source interval contains 6, 11 or 21 stride-five states.
Select actual frozen encoded states at these offsets, including endpoints:

| Source action blocks | Selected offsets in blocks | Interior labels |
|---|---|---|
| 5 | 0, 1, 2, 3, 4, 5 | 1, 2, 3, 4 |
| 10 | 0, 2, 4, 6, 8, 10 | 2, 4, 6, 8 |
| 20 | 0, 4, 8, 12, 16, 20 | 4, 8, 12, 16 |

These are time-position samples, never averages of latent vectors. All labels
remain in the same episode. Check terminals throughout the full primitive
interval, including unselected states; a terminal at the final goal is allowed.
Fail explicitly if a requested span has no legal source window. Training uses
episodes 0..7999; deterministic validation uses 8000..9999 via the existing
replay loader. Validation uses the same three-span recipe and a separate RNG.

The batch is always `[B,6,192]`. Its `state_stride=5` describes source replay
resolution, NOT the interval between selected labels. Per-sample `action_spans`,
`node_offsets_blocks` and global `rows` record the actual spacing. Logs record
each training batch's counts of the three spans; checkpoint identity contains
the complete sampling recipe. Resume restores the existing saved NumPy RNG and
rejects a different sampling identity. Legacy configurations have no added
identity fields and continue to use the unchanged original sampler.

## Networks, losses and action semantics

P remains the existing residual MLP, recursively producing exactly four interior
states. Generation fits the four selected states with the existing trajectory
MSE. Refinement retains trajectory, efficiency and dynamics losses, original
coefficients, sparse CEM cadence, safety limits and frozen G/V/F. Only P is trained.

Both training CEM and evaluation retain **five newly searched action blocks**.
For a long source fragment this deliberately tests compressed-path supervision:
it is a soft fitting objective, not a claim that each source segment already took
one action. Recorded source actions are not shortened, concatenated, averaged,
or fed to F with an incorrect duration. Sampling does NOT switch CEM to H10/H20.
Evaluation remains H5/RH5 (25 primitive steps per decision), with original
O25/O50/O100 pairs and total budgets 50/100/200. Longer-span labels can conflict
with five-action dynamics matching; that is an experiment risk, not silently
removed by changing losses or planning protocol.

## Usage and recovery

Supply the existing artifact paths via arguments, e.g.:

```sh
python scripts/train_effplan.py \
  --config configs/experiment/effplan_fixed4_discrete_5_10_20_old_v.yaml \
  --phase generation --latent-store "$EFF_LATENTS" \
  --terminal-metadata "$EFF_TERMINALS" \
  --eff-checkpoint "$EFF_OLD_V_CHECKPOINT" --eff-manifest "$EFF_OLD_V_MANIFEST" \
  --output-dir "$EFF_NEW_GENERATION_OUTPUT" --device cuda
```

`--init-from OLD_P.pt` optionally starts generation from an old P with the SAME
frozen G/V/F; it is weights-only, with a new optimizer and update counter.
Run refinement in a separate new output directory using the same configuration,
`--phase refinement --init-from GENERATION/last.pt --lewm-checkpoint F_WEIGHTS`.
All other artifact arguments remain the same. The shipped update budget remains
5 x 2560 per phase. Never overwrite an existing run. `--resume OUTPUT/last.pt`
resumes the same experiment; `--stop-after-updates N` allows bounded verification.

No training or formal evaluation is automatically launched by adding this code.
