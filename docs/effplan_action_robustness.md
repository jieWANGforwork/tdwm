# EffPlan action-perturbation robustness v1

Independent inference-only option. Reuse the frozen F, G/V and P checkpoints;
do not train anything, perturb actual environment actions, or add a network.
The default evaluator without the new flag is unchanged.

For a fixed P path X, let J(A) be the existing mean squared tracking error of
the full five-block F rollout against x1,...,x4,z_goal. Nominally preselect
the best 60 of 300 candidates. Within that shortlist use

    R(A) = mean_j max(0, J(A + delta_j) - J(A))
    J_robust(A) = J(A) + lambda R(A).

Use the same start, F and P nodes for every perturbation. Non-shortlisted
candidates receive infinity, so unchecked candidates cannot displace checked
elites merely by lacking a risk penalty. CEM retains its original 30 elites
and public update rule, including returning the elite mean. The returned
mean is not guaranteed to minimize this sampled objective (as in baseline).
P refinement still uses the nominal returned-action F rollout.

## Predeclared first comparison

Use the retrained same-episode n=10 total-work G/V and its completed refined P,
with the identical frozen LeWM and O25 50 start-goal pairs. This is a single
paired pilot, not hyperparameter selection or proof of superiority. Do not
retrain or rerun F-only. Keep H=RH=5, block=5, total primitive budget=50,
planning seed=42, 300 candidates, 30 rounds and 30 elites. Replan after each
25-step window if unfinished, stop on real environment success/budget.

Fixed settings: sigma=0.05 in standardized action coordinates, four noise
samples (two fixed Gaussian directions and their negatives), weight=1,
independent noise seed=43017, nominal shortlist=60. Clip noise components to
+/-3 sigma; do **not** add candidate action clipping. The installed baseline
CEM does not clip its model-space candidates; adding clip(A+delta) only to
perturbed evaluations would confound sensitivity with a new projection at
out-of-range nominal actions. Execution normalization/clipping remains the
existing baseline path. This distinction is explicit in every manifest.

All candidates use common random directions, preserving comparison under
candidate ordering/environment minibatching and not consuming CEM RNG. This
fixed finite bank may miss other sensitive directions; low R does not certify
robustness. Stable-but-wrong F predictions can still receive low R.

9000 nominal plus 7200 perturbed candidate rollouts and eight nominal returned
action rerolls per decision: 16208 total, versus baseline 9008. Environment
budget and nominal CEM sampling are matched; F compute is **not** matched.
Zero weight or zero sigma returns the exact original scores without extra
rollouts or preselection. No score scaling or result-dependent tuning.

## Entry point and outputs

Pass the original evaluation command, checkpoint/manifest paths and selection,
plus:

    --action-robustness-config configs/experiment/effplan_action_robustness_v1.json

Only fixed H5/RH5 EffPlan accepts this option. Use a fresh output directory.
Outputs include the ordinary full evaluation manifest, per-episode outcomes,
result.json, and action_robustness_diagnostics.json (cost/risk summaries and
actual nominal/perturbation rollout counts for every scoring call). Manifest
records config hash, formula, action units, noise/preselection settings and
extra compute. Run with the same runtime/render settings as the paired run.

Report paired New/Lost relative to the unchanged P+CEM, checking all checkpoint
hashes, episode identities and paired_protocol first. Do not equate smaller R
with lower true F error or claim the method repairs F without real evidence.
