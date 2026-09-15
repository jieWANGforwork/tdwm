# Variable-length P (additive experiment)

The original fixed-H5 P, its entrypoint/configuration, checkpoints and evaluation
remain unchanged. The new entrypoint is `scripts/train_effplan_variable.py` with
`configs/experiment/effplan_variable_same_episode_extra_work_v1.yaml`.

## Sampling and time units

Use the paired frozen V's exact same-episode sampler: uniformly draw an episode,
uniformly draw a legal future offset in its stride-5 state sequence, then draw a
legal anchor. Keep **every** stride-5 state up to the selected hindsight goal.
V's 50-primitive-step bootstrap boundary does not shorten the P training path.
For the current 201-frame episodes there are 41 stride-5 states; the sampled
path has L=2..41 states, L-1=1..40 macro actions, and L-2=0..39 interior nodes.
L counts states, not actions. L=10 means 9 macro actions and 8 interior nodes.
Never stretch a fixed four-node path across a longer segment.

Each minibatch contains IID variable-length paths. Bucket by length without
padding; sum bucket gradients weighted by the fraction of sampled paths, then
perform exactly one optimizer update. Independent midpoint splits at each tree
depth are batched without changing their state values or gradient paths.
Trajectory error is averaged over each
path's interior nodes, as in the original method. Two-state paths have no P
output: record them, but do not invent a midpoint or a P gradient. An all-boundary
batch does not apply AdamW weight decay. No terminal-crossing or cross-episode
trajectory labels are created.

## Networks, objectives and stages

P is the unchanged 769->512->512->192 residual MLP. G/V, observation encoder,
action encoder and F remain frozen, using the paired same-episode extra-work V
checkpoint. Only P is initialized and trained anew.

Generation fits L-2 real interior states. Refinement uses the original trajectory,
efficiency and dynamics coefficients and safety limits. Sparse CEM calibration
retains its original cadence and sample cap, but sets H=L-1 for each length
bucket and rolls every one of those actions through frozen F. Non-calibration
updates retain efficiency/trajectory refinement without searching actions.
The existing variable-horizon tracking adapter is reused; no new CEM is written.
Longer paths necessarily increase training computation; optimizer-update count
and loss coefficients are held fixed, not wall time or number of predicted nodes.

Each stage retains 5 epochs x 2560 updates, seed 3072, batch size 256, original
optimizer/warmup-cosine settings and deterministic validation seed 50. The shared
latent store and episode partition come from the paired V configuration.
Existing formal O25/O50/O100 pair lists and fixed 25-step decision windows are
not changed by this training experiment.

## Run and recover

Both stages accept the same artifact arguments as the old entrypoint. Run
generation with a **new** output directory, then refinement with another new
directory and `--init-from GENERATION/last.pt`. Refinement also needs
`--lewm-checkpoint`. The initialization is checked against the new sampling
identity and paired frozen source. Resume with `--resume OUTPUT/last.pt` in that
same stage. `--stop-after-updates N` pauses at absolute update N while preserving
the original full training schedule. No old fixed-H5 checkpoint is silently
treated as a variable-length continuation.

Sampling identity, per-batch sampled state counts, optimizer updates, source
hashes, RNG states and checkpoint hashes are recorded. The original evaluation
checkpoint loader remains usable because P's architecture and weight format
are unchanged; sampling provenance is carried in the checkpoint identity.
