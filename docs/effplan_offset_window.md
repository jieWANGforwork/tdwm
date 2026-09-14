# Fixed goal-offset execution window (independent evaluation)

Use `scripts/evaluate_effplan.py evaluate --method EffPlan --offset-window`
with the existing config, frozen checkpoints and selection. No training changes.
Do not combine this flag with either adaptive flag. O25 is intentionally excluded.

| Protocol | Intermediate nodes | Action blocks | Execute per decision | Total primitive budget |
|---|---:|---:|---:|---:|
| O50 | 9 | 10 | 50 | 100 |
| O100 | 19 | 20 | 100 | 200 |

Each block contains five primitive actions. Existing recursive midpoint generation
always fills the fixed node count; no work-gain rejection or adaptive subdivision.
The existing P refinement, geometric safeguards, G/V readout and full-path F
tracking score are retained. CEM still uses 300 candidates, 30 iterations split
as `[3,3,3,3,3,3,4,4,4]`, 30 elites, seed 42 and batch size 1. F rolls forward
from the observed anchor through all actions, never resetting to P's nodes.

The public WorldModelPolicy executes the full window, then replans from the new
real observation if unsuccessful. Success may terminate any primitive step.
There are at most two decisions. H equals RH, so no unexecuted warm-start tail
is carried to the next decision. The original H5/RH5 and both adaptive modes
remain available unchanged. The longer horizon changes search dimensionality
and F computation; this is not a matched-compute comparison.

Evaluate both original V and extra-work V, using their completed 12800-update P
checkpoints, on the same O50/O100 pair files, with isolated output directories.
The manifest records the original locked config plus an explicit `offset_window`
override; `paired_protocol` records the effective H/RH and unchanged total budget.

Validation: the execution-protocol integration test uses the installed public SWM
action queue to verify decisions at 0/50 and 0/100, full ordered consumption,
termination in either window, and unchanged legacy scheduling. The CEM integration
test checks fixed 5/10/20-block path shapes and final-search state feedback.
