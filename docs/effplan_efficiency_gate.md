# Local-distance AND efficiency recursion

## Additional distance-only ablation

Keep the combined rule below unchanged. The separate option
`--adaptive-distance-only --adaptive-rolling --adaptive-local-distance-limit <P95>`
(method EffPlan) stops each generated segment solely when
`D(u,v) = ||v-u||_2 <= d_local`. Do not supply an efficiency threshold.
Use the same training-only P95 calibration artifact as the combined rule:
`d_local = quantile_0.95({||E(o[t+5])-E(o[t])||_2})`, episodes 0..7999,
all valid within-episode t. These are adjacent **big-action boundary states**,
five primitive steps apart, not adjacent raw frames. P95 is not a mean:
95% of those training transition distances are at or below this cutoff.
Do not select between the mean and P95 using evaluation outcomes.

For every current leaf, including newly generated child segments:
first stop if D <= 1e-6 (degenerate, not environment success);
otherwise stop if D <= d_local; otherwise ask the existing P for a
finite, nonduplicate midpoint and check both children independently.
Retain existing remaining-budget cap and duplicate guard. Neither guard
certifies short distance or reachability. Efficiency and predicted work are null
in stopping records: G/V is not called to decide whether to stop.
G/V still supplies the unchanged P generation/refinement feedback; this is
**distance-only stopping**, not removing G/V from the entire planner.
Keep subsequent CEM/F tracking, real-environment success checking, and
replanning until success or total budget unchanged.

Record score mode `adaptive_local_distance_only_rolling_v1` and criterion
`local_distance` separately. Do not overwrite or mix with the combined study.
The shared study launcher now supports `--distance-only` for preview/run;
its analyzer detects and validates this separate score mode. Supply the same
calibrated `--local-distance-limit`, existing model/data arguments, explicit
GPU indices, and a new output directory. Preview performs no GPU work.
Run launches all six evaluations concurrently and refuses existing output/logs
or a previously launched study. It automatically audits completed results and
writes one comparison table, original V O25/O50/O100 followed by extra-work V
O25/O50/O100. Do not launch until CUDA GPUs are available; CPU-only preparation
does not count as a formal evaluation. No training or F-only rerun is needed.
A small latent distance still cannot prove
one-action reachability or exclude an obstacle between the endpoints.

The revised rule stops a leaf if and only if:

`D <= d_local AND eta >= tau`.

Before computing eta or calling G/V, handle `D <= epsilon` separately: stop
subdivision with `degenerate_segment`, keep eta undefined (null), and never turn
this numerical guard into real-environment success. The existing epsilon is 1e-6.
The raw distance limit must be explicit, finite and positive; it has no default.
It can be calibrated from a predeclared quantile of training-only
`||z[t+5]-z[t]||_2` (one big A). No quantile or value is selected here, and no
test-pair outcomes may be used to select it.

New study preview/run requires `--local-distance-limit <d_local>`. It forwards
`--adaptive-local-distance-limit <d_local>` to every evaluation. Score mode is
`adaptive_local_distance_efficiency_rolling_v2`; the manifest and every leaf
record the distance limit. Efficiency tau remains 0.8 in the study launcher.
A far but straight path must continue subdividing; a close but inefficient path
also continues. Both children are checked independently. A budget-cap or duplicate
guard is NOT evidence that either stopping condition is satisfied.

The evaluator retains the no-distance, efficiency-only option solely to reproduce
historical runs; the study CLI no longer silently launches that old option.
The analyzer supports both versions but rejects mixing versions/scales in a study.
No server jobs, checkpoints or training settings are changed by this revision.

The user has now fixed calibration to P95. Run
`scripts/calibrate_effplan_distance.py --latent-store <store> --config <config> --output <new-json>`
before launching the study. It validates the frozen store hashes, uses episodes
0..7999 only, computes all within-episode pairs separated by exactly 5 primitive
steps, and saves the exact linear-interpolated P95 and provenance. Pass its
`local_distance_limit` unchanged to both V variants; do not recalibrate on outcomes.

## Historical efficiency-only option and shared mechanics

Opt-in only. Existing fixed-25, offset-window and work-gain adaptive evaluations
retain their defaults and numerical paths. No training or checkpoint changes.

For each current leaf `(u,v)`, use the existing frozen target G/V and safeguards:

```
D = ||v-u||_2
W = max(V(G(u,v),v), D)
eta = D / (W + epsilon)
```

If `eta >= tau`, stop subdividing that leaf. Otherwise P proposes a midpoint
using its existing midpoint initialization, efficiency feedback and capped state
increment. Keep a finite, nonduplicate midpoint without comparing parent work
against summed child work. Test both children independently with the same rule.
Breadth-first traversal is scheduling only; no larger-side selection/ranking.
The remaining primitive budget caps blocks at `remaining//5`, with `K` interior
states mapping to `K+1` blocks. Budget exhaustion does not imply sufficient eta.

The threshold has NO default. Supply an explicit finite `0 < tau < 1` before
evaluation. With the geometric floor and epsilon, tau=1 would almost always
continue until the cap, so it is rejected. Equal/near-equal endpoints stop via a
separate degeneracy guard, not an invented efficiency of one. Nonfinite values
fail fast. Duplicate proposed midpoints stop via a separate diagnostic.

Use the existing evaluation command and required artifact arguments, adding:

```
--method EffPlan --adaptive-rolling --adaptive-efficiency-threshold <tau>
```

Not valid with one-shot, offset-window, F-only or Eff. The score mode is
`adaptive_local_efficiency_rolling_v1`; the effective threshold and formula are
stored in `protocol_overrides.adaptive_rolling`. `adaptive_planning.json` records
each branch's distance, raw/protected work, efficiency, threshold and stop reason.
No formal tau is selected by this implementation or tuned on held-out successes.

## Predeclared first study (user approved)

The first study now fixes tau=0.8 for both original V and extra-work V, each on
O25/O50/O100 (six jobs, 300 episodes). The generic evaluator still has no default.
Reuse each variant's existing G/V and 12800-update P; no training or F-only rerun.

`scripts/run_effplan_efficiency_study.py` has `preview`, `run` and `analyze` modes.
Provide `--runs-root`, a fresh `--output-root`, and for preview/run also
`--dataset`, `--lewm-checkpoint`, and explicit `--devices 0 1 ...`.
Preview is read-only. Run checks all inputs before launching six concurrent jobs,
round-robin across the supplied GPUs, records PIDs/logs and refuses occupied runs.
Do not run until the GPU server is available. On interruption, inspect recorded
PIDs; never launch duplicates or overwrite unfinished output directories.

Once all jobs succeed, run automatically audits and writes `study_analysis.json`
and `study_analysis.md` inside the new external output directory. Compare against
`sparse_v_compare_20260914` (fixed five blocks) and `adaptive_rolling_20260915`
(first work-gain adaptive rule), verifying pair and checkpoint identities.
Report original V's three protocol columns first, then extra-work V's three.
Also report paired New/Lost, action counts, budget-cap stops, degenerate/duplicate
stops, floor-driven efficiency stops, and actual execution budgets. These files
are intermediate analysis artifacts for incorporating into the existing Results
TD document, not a replacement results document. No study has been run merely by
adding this launcher. Formal outcomes must not be used to tune the threshold.

After generation, keep the existing alternating P refinement and joint CEM/F
path tracking with the selected count fixed within that decision. Those later
P updates can change endpoint efficiencies; no claim is made that final refined
leaves still pass the generation-time gate. Every new real-observation decision
generates and tests a fresh tree. Execute all chosen blocks, stop early on real
success, otherwise replan until cumulative budget is exhausted (50/100/200).

This is a path-efficiency stopping heuristic, not a learned action-count estimator
or a one-block reachability certificate. A long straight efficient path can still
need many actions. No F-based reachability gate or new network is added.
