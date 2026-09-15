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

After generation, keep the existing alternating P refinement and joint CEM/F
path tracking with the selected count fixed within that decision. Those later
P updates can change endpoint efficiencies; no claim is made that final refined
leaves still pass the generation-time gate. Every new real-observation decision
generates and tests a fresh tree. Execute all chosen blocks, stop early on real
success, otherwise replan until cumulative budget is exhausted (50/100/200).

This is a path-efficiency stopping heuristic, not a learned action-count estimator
or a one-block reachability certificate. A long straight efficient path can still
need many actions. No F-based reachability gate or new network is added.
