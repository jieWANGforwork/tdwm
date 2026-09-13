# EffPlan P stability v1

Opt-in numerical safeguards, not a new V objective or a claim of improved
success. Configuration: `configs/experiment/effplan_cube_stable_p_v1.yaml`.
Historical configuration, checkpoints, Eff-only scores and execution budgets
remain unchanged. No Actor or action decoder is added.

## P computation

F, G and V stay frozen. P retains its 769 -> 512 -> 512 -> 192 MLP and
trajectory / efficiency / dynamics losses with coefficients 1 / 0.1 / 0.1.
For candidate path x_0,...,x_H, only inside P's objective:

    W_raw,k = V(G(x_k,m_{k+1}),m_{k+1})
    W_safe,k = max(W_raw,k, ||x_{k+1}-x_k||_2)
    eta_P = ||x_H-x_0||_2 / (sum_k W_safe,k + epsilon)

Unsquared Euclidean distance in numerator and denominator gives eta_P<=1
by the triangle inequality, up to roundoff. Coincident endpoints give zero
efficiency, including the all-identical-state path. This floor is NOT proof
of dynamic reachability. Where active, its derivative replaces the raw V
derivative. A high activation fraction can reduce useful critic guidance.

Feedback remains dJ/dx with J=-eta_P+0.1*L_dynamics, then detached before P.
The existing feedback coefficient 1 (versus final-loss coefficient 0.1) is
retained; these are not silently equated. Project each 192-D candidate gradient
to L2 norm <=10 and each P increment to L2 norm <=5. Five means latent units,
not actions. These are pre-evaluation engineering limits, not optimized values.
Recursive generation and all refinement rounds use the same settings, keeping
endpoints fixed. Norm projection accumulates in float64; NaN/Inf are rejected,
not replaced by zeros. Parameter-gradient clipping remains a separate limit 1.

Logs include raw V values, floor-hit fraction, protected efficiency, raw/capped
candidate-gradient and increment norms, cap fractions, and the individual
losses. Finiteness alone is insufficient: inspect whether updates merely
saturate their limits and whether held-out dynamics consistency improves.

## Run and compatibility

Initialize refinement from the existing completed generation P in a NEW output
directory. Keep G/V/F sources, splits, batch/CEM budgets and 5*2560 updates.
Safety-v1 also enables the existing 1%-warmup/cosine LR function which the old
runner defined but never called. Legacy runs retain constant LR. This is a
stabilization bundle, not an isolated ablation attributing gains to one fix.

Optional `settings.safety` is hashed and saved. Without it, serialization omits
the new field, retaining old checkpoint settings/hashes. Different safety
settings cannot resume each other. `--init-from` is weights-only; `--resume`
restores optimizer and sampling/CEM RNG state. Deployment refuses discrepancies
between checkpoint, training and evaluation safety settings.

Safety runs additionally checkpoint at updates 1,5,10,25,50,100. Optional
`--stop-after-updates N` pauses at absolute update N, saves last.pt, and marks
the run paused, NOT complete. It leaves the full LR budget unchanged. Resume
without that option to finish. Never report a paused check as formal training.

```bash
python scripts/train_effplan.py \
  --config configs/experiment/effplan_cube_stable_p_v1.yaml \
  --phase refinement --device cuda:0 \
  --latent-store "$EFF_LATENT_STORE" --terminal-metadata "$EFF_METADATA" \
  --eff-checkpoint "$EFF_CHECKPOINT" --eff-manifest "$EFF_MANIFEST" \
  --lewm-checkpoint "$LEWM_CHECKPOINT" \
  --init-from "$P_GENERATION_CHECKPOINT" --output-dir "$P_STABLE_OUTPUT" \
  --stop-after-updates 25
```

Public SWM CEM still finds actions matching the path; frozen F rerolls the
returned actions. CPU fixture tests are not a substitute for actual-checkpoint
F+CEM refinement or formal task success evaluation.

## V: discussion only, not implemented

V already uses MSE against observed/TD targets. Softplus ensures nonnegativity,
not W>=distance, and very negative preactivations can saturate near zero.
First examine observed short paths, full-path targets, TD targets and generated
states separately, including goal-boundary behavior. A global rescaling alone
cannot repair both near-zero estimates and overestimates.

Controlled alternatives: linear final output with the same MSE; or
W(z,g)=D(z,g)+softplus(R(G(z,m),m)), learning extra path movement beyond direct
distance. The latter requires retraining V and explicit zero-at-goal handling.
Keep the total-W target, or subtract D from that target when supervising the
residual; simply using c+next residual is incorrect. Old total-cost weights
cannot be relabeled residual weights. Direct efficiency prediction is another
objective: a ratio does not obey the additive cumulative-cost TD backup.
