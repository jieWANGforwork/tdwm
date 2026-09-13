# Eff extra-work V (independent variant)

This implements the requested V option C, not historical V1-C/C4. No actor,
new action input, sampling change, score sweep, or baseline replacement.

## Model and loss

Let `m = sqrt(192) normalize(z_goal)`, `Psi = G(z,m)`. G remains the existing
192-dimensional future-successor predictor, beginning at the next real state.
The V head retains its two-hidden-layer MLP (384 -> 512 -> 512 -> 1).

```
r(z,g) = softplus(V_raw(stop_gradient(Psi), m))  # V training only
D(z,g) = ||z - z_goal||_2                       # unsquared raw latent distance
W(z,g) = D(z,g) + r(z,g)                        # nonterminal readout
W(g,g) = 0                                     # exact empty-path boundary
```

During planning, G output is NOT detached, so candidate-state and goal
derivatives flow through G and V and through D. Parameters can remain frozen.
`EffModel.value` returns **total W** in both online and EMA modes. `model.v`
alone returns the positive extra-work residual in this variant; it must not be
used as the planning cost or the bootstrap tail by itself.

Each real training transition costs `c_i = ||z_(i+1)-z_i||_2`. The V target is:

```
Y = sum(c_i, from anchor to goal)                    if goal reached in window
Y = sum(c_i, over backup prefix) + EMA_W(z_backup,g)  otherwise, if continuation valid
L_V = existing efficiency-weighted mean of (W - stop_gradient(Y))^2
L = L_G + 1.0 * L_V
```

No discount is applied to W. G's vector TD discount remains 0.98 per five-step
training block. Invalid failed-terminal continuations remain masked out.
V regression still detaches G; only G's vector loss updates G. F and its
encoders do not enter the optimizer. No separate residual target or loss is
introduced. Both online and EMA tails include the geometric base.

The exact-goal branch uses equality of raw latent states, NOT an environment
success threshold. It enforces the existing zero-cost endpoint label but can
be discontinuous near the endpoint when the residual remains positive. It is
not a claim of smooth calibration; proximity gating/multiplicative residuals
would be separate parameterizations. Euclidean gradient is zero at equality.
No output upper clipping is added. The existing P safety limits remain enabled.

## Protocol, artifacts and compatibility

Use `configs/experiment/effplan_cube_extra_work_v1.yaml`; it differs semantically
from stable_p_v1 only by `eff_training.settings.v_parameterization: extra_work`.
Default `total_work` retains old numerical paths, checkpoint keys and manifest
settings hashes. New checkpoints record `extra_work`; loaders restore it, and
resume rejects cross-mode use. Same-shaped old weights are NOT residual weights.
Train in a NEW output directory; never overwrite or relabel an old run.

The existing seed 3072, 10 epochs / 127960 updates, episode split 0-7999 versus
8000-9999, frozen store, goal sampling, efficiency weighting, LR schedule and
MC/TD split are unchanged. Backup remains 50 primitive steps / 10 stride-five
blocks; this is our configured choice, not a verified RP1 unit equivalence.

Full training entrypoint (supply existing audited store and terminal metadata):

```sh
PYTHONPATH=src python scripts/train_eff.py train \
  --config configs/experiment/effplan_cube_extra_work_v1.yaml \
  --latent-store /absolute/path/to/frozen_store \
  --terminal-metadata /absolute/path/to/terminal_metadata_directory \
  --output-dir /absolute/path/to/NEW_extra_work_run/train --device cuda:0
```

The unchanged train_effplan.py and evaluate_effplan.py entrypoints accept this
configuration and the corresponding new Eff checkpoint/manifest. P refinement
and Eff CEM scoring both read total W through EffModel.value. Existing complete
training, identity, checkpoint update-budget and protocol gates still apply.
O25/O50/O100 retain H5/RH5, not historical RH1. No evaluation formula changes.

Unit and small-loop integration tests exercise MC/TD labels, detach semantics,
exact-goal boundary, candidate gradients, adapter readout, legacy behavior,
checkpoint mode, cross-mode rejection and deterministic full-loop resume.
These tests do not constitute full-budget training or formal success results.
