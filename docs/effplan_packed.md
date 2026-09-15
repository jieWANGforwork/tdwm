# Packed-compute P continuation (additive)

`scripts/train_effplan_packed.py` is an independent backend for the existing
variable-length P. It consumes the **unchanged**
`effplan_variable_same_episode_extra_work_v1.yaml` configuration. The fixed-H5
P and original variable-length P files, checkpoints and running processes are
not edited or stopped.

## What accelerates

Generation pools all independent binary-tree splits at each depth across all
path lengths. It constructs exactly the same L-2 intermediate states for an
L-state path, rather than running the tree separately for every length bucket.
Topology and real labels are assembled once per batch and reused. Only genuine
edges are evaluated; temporary reduction matrices contain zeros in unused
slots, never fictitious candidate states or transitions.

Ordinary refinement pools all genuine edges for G/V evaluation and all interior
states for the synchronous P update. Loss keeps the original per-path weighting
and per-path interior-node averaging. With final-round supervision, loss-only
computations for unused earlier rounds are omitted; their state updates and
feedback gradients are still computed. Generation's efficiency logging term
has no backward graph because its coefficient is zero.

Calibration uses the original implementation unchanged, including its cadence,
path cap, candidates, iteration counts, horizon, solver RNG order and F rollout.
No optimizer updates, training paths, intermediate nodes, refinement rounds,
loss coefficients or model parameters are removed. No AMP/TF32 policy changes
are introduced. Frozen F/G/V retain their original roles.

Safety rules remain identical. Packed diagnostic means/extrema describe the
actual pooled nodes/paths; the older backend averaged per-bucket diagnostic
summaries. This affects diagnostic aggregation only, not caps or losses.

## Validate, then branch from a checkpoint

`scripts/benchmark_effplan_packed.py` compares both backends on an immutable
checkpoint and the same real batch, checking loss and relative gradient error.
It never steps an optimizer or saves a model. CUDA timings synchronize before
and after computation; reported compute speedup excludes replay setup, sampling,
checkpoint IO and the other training stage. Calibration is timed separately.

Run the packed trainer with a **new output directory** and
`--resume IMMUTABLE_SOURCE.pt`. The original checkpoint format and training
identity remain compatible: P weights, AdamW moments, optimizer update count,
sampling RNG, torch/CUDA RNG and CEM RNG are all restored. The full training
schedule is retained, not restarted. The new manifest records the resume source
and `runtime.execution_backend=packed_cross_length_v1`. After the generation
stage finishes, start packed refinement using that branch's final generation
weights with `--init-from`, as in the original two-stage protocol.

Reduction/batching order can cause small floating-point differences, so this is
an objective- and protocol-preserving backend, not a promise of bitwise-identical
future training trajectories. Tests cover loss, gradients, parameter updates,
fixed endpoint preservation, two-state boundary samples, calibration budgets,
and optimizer/RNG checkpoint continuity. Evaluation settings are unchanged.
