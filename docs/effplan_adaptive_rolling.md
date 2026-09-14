# Adaptive rolling EffPlan: corrected total-budget execution

Use `scripts/evaluate_effplan.py evaluate --method EffPlan --adaptive-rolling`.
The earlier `--adaptive-one-shot` remains an isolated historical ablation, not the
user's requested total-budget adaptive evaluation. Do not overwrite its results.

This change retains the exact midpoint proposal, critic work-gain acceptance
threshold (1e-6), breadth-first subdivision, geometric protection, eight P
refinements and final CEM search from the adaptive one-shot implementation.
No thresholds are retuned from results. P, G, V and LeWM checkpoints are reused
without training. No new learned module or actor is introduced.

At each REAL decision boundary, use the latest environment observation as start
and the unchanged final goal. A path with K internal states requires N=K+1 action
blocks (5 primitive actions each). Bound N by floor(REMAINING budget/5), not the
original full budget. CEM tracks all N nodes through one continuous F rollout.
Execute all N blocks unless the environment reports success. If still unfinished,
plan another adaptive path from the newly observed state. Never truncate an
episode merely because one adaptive plan was exhausted. Non-success episodes use
the full 50/100/200 primitive-action cap for O25/O50/O100. Goal success can terminate
inside any action block. No padded transport actions are sent to the environment.

Implementation uses a thin per-episode controller around the PUBLIC SWM
WorldModelPolicy, which still performs preprocessing, normalization and buffering;
the installed CEM and World.evaluate remain unchanged. No private package API or
installed-package modifications are used. Separate delegates allow different
episodes to have different N and decision boundaries in the same vector World.

Each decision uses [3,3,3,3,3,3,4,4,4] CEM iterations, 300 candidates, 30 elites and
seed 42. Total planning compute is therefore variable and can exceed fixed H5/RH5;
record every decision's horizon, remaining budget, start step, actual execution,
candidate macro transitions, duration, initial/final nodes and normalized actions.
This is an execution-protocol ablation, not a matched-compute superiority claim.

Run both existing final P variants on the same three fixed 50-pair selections in
new output directories. Guard against a second plan using an imagined start,
budget reset, premature failure, goal drift, wrong node/action count, or counting
an execution limit as success. Preserve original fixed and one-shot results.
