# Adaptive one-shot EffPlan v1 (inference only)

Use `scripts/evaluate_effplan.py evaluate --method EffPlan --adaptive-one-shot`
with the existing completed stable/sparse P checkpoint, its matching G/V, and
unchanged frozen LeWM. No new network, parameter fitting, or checkpoint changes.
The old Eff/EffPlan execution and scoring paths are unchanged without this flag.

## Predeclared node rule

Start with `[start, goal]`. For each leaf in breadth-first order initialize its
midpoint at the endpoint mean and apply the existing P once with local negative
efficiency and its candidate gradient. Evaluate W(left,right) and
W(left,midpoint)+W(midpoint,right) using the EMA G/V and existing per-segment
geometric floor. Retain a distinct midpoint only if the relative predicted work
decrease is strictly greater than 1e-6 (numerical improvement tolerance, not a
test-selected threshold). Rejecting a split stops that leaf. An accepted midpoint
adds exactly one action block. Stop accepting at budget/5 blocks. There is no
forced first split, no minimum five-block path, and no post-result threshold sweep.

This is a critic-based heuristic, NOT evidence of one-block reachability or
optimal step count. W already predicts cumulative work; critic inconsistency can
create artificial split gains. High efficiency does not imply a short path.
Log every proposed split and its before/after costs, decision and depth.

## Planning and execution

K intermediate states correspond to K+1 five-primitive-action blocks. Every F
rollout begins from the observed initial state, never resets at a generated node,
and covers all K+1 candidate blocks. Track the squared distance at every node,
including the final goal. Keep the previous eight synchronous state refinements
and final action search, entirely BEFORE any environment action. The nine CEM
searches use [3,3,3,3,3,3,4,4,4] iterations, 300 candidates, 30 elites. Refinement
does not change the already selected node count. Reset CEM seed 42 per pair.

One planning call per episode. No environment-feedback replanning. Execute the
returned action sequence once, stopping at actual environment success or the end
of that sequence. Exhaustion without success is truncation/failure. Unused budget
is not filled by extra planning or zero actions. Public SWM WorldModelPolicy uses
a rectangular transport tensor; public gym wrappers truncate BEFORE any transport
padding could execute. These wrappers never set a success flag. No installed SWM
code is changed.

O25/O50/O100 caps are 50/100/200 primitive actions: at most 10/20/40 blocks and
9/19/39 intermediate nodes. Same fixed 50 pairs, renderer, source checkpoint and
environment success test as previous EffPlan. This is a DIFFERENT planning and
execution protocol, not a matched-H5/RH5 compute comparison. Record actual horizon,
candidate macro transitions, rerolls, planning time and executed primitive steps.

Run both original-V and extra-work-V existing final P checkpoints on all three
protocols. Save per-episode results, attempted splits, initial/final planned nodes,
normalized actions and identities independently of previous results. Never infer
success from F or the critic. No F-only or legacy scoring reruns are required.
