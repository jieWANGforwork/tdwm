# G-weighted CEM evaluation

These are two additional inference tests of existing checkpoints. They do
not modify training, add an actor, update G/F, or overwrite earlier readouts.

## Fixed rules

The existing F-only full five-block rollout assigns each candidate its
terminal squared latent goal distance. The installed CEM selects its usual
30 elites from 300 candidates. These are the only paths receiving G reads.
Existing CEM iterations, seeds, warm starts, primitive-action normalization,
feedback interval, environment budget, dataset and start-goal pairs remain
those of the supplied historical config.

For action-conditioned C--G3 and C2, the per-position score is
q[i,h] = G(z[i,h-1], E_A(a[i,h]), m)^T m (V0 uses its native raw action).
The first state is the encoded real observation; later states come from the
same candidate's prefix F rollout. For C4, q[i,h] = G(F-produced z[i,h],m)^T m:
every action, including the last, passes through F and never enters G directly.
Online deployment G is used, not the EMA training target.

The two predefined update rules are:

| Evaluation mode | G weighting of the selected elites |
| --- | --- |
| g_path_weighted_cem | s[i] = mean over all five q[i,h]; w[i] = softmax over elite paths of s[i]/temperature; all five action positions share that weight |
| g_action_weighted_cem | w[i,h] = softmax over elite paths of q[i,h]/temperature separately for each position h |

For either rule, mu[h] = sum_i w[i,h] a[i,h]. Standard deviation uses the
same position-specific weights and variance denominator 1-sum_i w[i,h]^2.
Thus uniform weights recover the original sample-standard-deviation rule,
including its K-1 correction. Exactly equal weights use the original mean
and std operations, preserving numerical equality. A degenerate denominator
is clamped to the weight dtype's epsilon to avoid NaN; one-hot weights can
collapse the distribution. This is an explicit weighted-moment convention,
not an unbiasedness claim for value-dependent weights.

Default temperature is 1.0, with raw Q: no z-score, extra F/G mixture, value
clipping, minimum-spread floor, or automatic tuning. Temperature is recorded
in each manifest; choose it before evaluation, never by selecting the best
formal result afterward. Raw Q scales may make weights overly concentrated.
The callback exposes effective elite counts and maximum weights in the
upstream solver's callback history for diagnostic inspection.

This does not greedily concatenate individual winning actions. After the
weighted moment update, the unchanged solver samples and evaluates complete
new sequences. Its final output remains the final fitted mean action plan,
not a guaranteed best sampled trajectory. G does not see the candidate's
entire remaining suffix, and per-position weighting can mix incompatible
prefix-dependent actions; improved control is an empirical question.

## Implementation and supported checkpoints

The added adapter attaches to the installed stable-worldmodel 0.1.1 public
CEM callback interface. It changes the live mean/std tensors after ordinary
elite selection. No upstream package file, solver loop or execution policy
is copied or changed. F states are cached from the same full rollout; G
evaluates only 30 x 5 elite positions, not another full 300-path rollout.

The entry point supports C/D/F/G1/G2/G3 for V0, V1, V2 and V2 EMA, and V1
C2/C4 where their existing configs exist. It does not invent missing O25 or
O100 configs. C3's separately trained State-V is not silently substituted for
G or added to the cost; a State-V variant needs its own explicit definition.
Existing ordinary score modes are unchanged when this feature is not enabled.

O25/O50/O100 denote the original goal offsets, not the feedback interval.
This entry point preserves the source config's feedback interval and does
not implicitly enable the separate 25-step full-plan revalidation experiment.

## Run

Inspect a fully resolved test on CPU, without loading weights or data:

    python scripts/evaluate_actor_free_td_lewm_g_weighted_cem.py \
      --version v1 --variant c --weight-mode action --temperature 1.0 \
      --config configs/experiment/actor_free_td_lewm_v1_c_cube_checkpoint_o50.yaml \
      --dry-run

Replace action with path for the other predefined test. To run, remove
--dry-run and additionally supply:

    --dataset <existing-cube-dataset>
    --checkpoint-path <previously-selected-checkpoint.pt>
    --checkpoint-sha256 <previously-recorded-64-character-digest>
    --output-dir <new-empty-result-directory>

Formal runs require an available CUDA GPU and retain the existing checkpoint
completion and fixed-pair validation. Selected intermediate V2/EMA epochs
3--9 require --checkpoint-epoch, as in the historical evaluator. Each result
has its own g_path_weighted_cem or g_action_weighted_cem identity, complete
formula/configuration, checkpoint digest, selected pairs and per-episode
outcomes. No test is launched by --dry-run.

The runtime integration tests use synthetic models/environments and the
actual installed solver; they are not formal Cube success-rate results.

## Complete the V1 C--G3 matrix without repeating C

`run_actor_free_td_lewm_v1_g_weighted_completion.py` requires the completed
six-cell C output root. It verifies those results and schedules exactly
D/F/G1/G2/G3 x O25/O50/O100 x path/action = 30 new formal cells. It never
schedules training or F-only repeats. The new D--G3 O25/O100 configs inherit
their respective O50 method/checkpoint contracts and match C's goal-offset,
pair-selection, feedback and budget settings.

    python scripts/run_actor_free_td_lewm_v1_g_weighted_completion.py \
      --checkpoint-root <historical-v1-cg3-training-root> \
      --dataset <cube.lance> \
      --reuse-c-root <completed-six-cell-C-root> \
      --output-root <new-output-root> \
      --gpus 0 1 2 3 --max-jobs-per-gpu 3 --max-concurrency 12

Pass `--preflight-only` in CPU mode to check all checkpoint hashes, dataset
manifest and six reused C results without starting evaluations. Supply the
existing runtime's `STABLEWM_HOME`, `LD_PRELOAD` and `LD_LIBRARY_PATH` as
environment variables; the launcher retains OSMesa to match the completed C
cells and records those settings. Temperature stays fixed at 1, without
z-score or additional score changes.

All offsets share one worker pool. Longest-budget jobs are queued first;
any freed GPU slot admits the next job. Three workers per GPU is an initial
configurable capacity, not a GPU hardware limit. Pair equality is checked
within each offset, never incorrectly between different goal offsets. A
failed child/output check stops new dispatches, while already-running jobs
finish and retain their outputs. Reusing a nonempty output root is rejected.
Power control is not part of this launcher: the authorized operator must
verify completion, preserve results and shut down the AutoDL instance.
