# Independent local-efficiency recursion

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
