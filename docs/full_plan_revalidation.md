# C–G3: full-plan execution revalidation

This is an opt-in evaluation protocol, not a training change. Historical
configs, checkpoints, score formulas, weights, and result directories are
preserved. New results must not be combined with historical 5-step-feedback
results as if their execution protocols were identical.

## What changes

For a five-block candidate plan, each block contains five primitive actions:

1. Encode the current **real environment observation**.
2. Run CEM on five-block candidate action sequences. Score every candidate
   within every CEM iteration, before selecting elites.
3. Execute all five blocks from the chosen plan, in order. There is no new
   CEM search between those blocks, and no predicted state is substituted for
   an observation to start a new search.
4. If the episode is still active and budget remains, encode the new real
   observation after 25 primitives and plan the next five-block sequence.

The installed `stable_worldmodel.WorldModelPolicy` implements the action
buffer and replanning. No alternative policy or solver is introduced.
The transform sets `receding_horizon=5`; `horizon=5` and `action_block=5`
remain unchanged.

O25/O50/O100 describe the dataset goal offset, not the feedback interval.
Existing supported configs retain their episode budgets of 50/100/200
primitive actions, respectively. At most 2/4/8 complete 25-step plans fit
those budgets; success or termination can end an episode earlier.

Historical O50 and O100 used `receding_horizon=1` (five primitives per
real-observation replanning). The existing V1-C O25 five-block modes already
used `receding_horizon=5`. The revalidation interface does not broaden a
version's supported goal offsets or invent missing experiment configs.

## Where Q is used

Let `z0` be the encoded real observation, `A1,...,A5` the candidate blocks,
`zk = F(z[k-1], Ak)` imagined states, and `D` the existing terminal latent
goal-distance cost. For V1, `Q(z,A,m) = G(z,E_A(A),m)^T m`.

| Score mode | Existing candidate cost, preserved in this revalidation |
| --- | --- |
| `f_only` | `D(z5, goal)`; no G call |
| `f_plus_g_first` | `D(z5, goal) - alpha * Q(z0,A1,m)` |
| `f_plus_g_first_q2` | Candidate-wise z-score of `D(z5,goal)` minus `alpha` times the independently normalized first Q; only where the source config supports it |
| `f_plus_g` | Existing legacy tail: `D(z4,goal) - gamma^4 * Q(z4,A5,m)`; F receives the first four blocks, G the final block |
| `g_only_f_rollout_mean` | `-mean(Q(z[k-1],Ak,m), k=1,...,5)`; F supplies imagined states, with no terminal distance term |

First-Q evaluates G at the first block only. The remaining four blocks still
affect the F terminal cost. Changing the feedback interval does not silently
turn First-Q into Mean-Q, or change the tail formula to full-five-block F.
Which readout works best with 25-step feedback is an empirical question; old
rankings do not establish the new ranking.

Strict historical `g_only` is deliberately rejected: it plans one block with
G and does not roll out F. Extending it to five blocks needs a separately
agreed definition. Keep it labeled as a different protocol unless that
decision is made.

## Running an existing checkpoint

The entry point supports V0, V1, V2, and V2 EMA (`v2_ema_sg`), each with
C/D/F/G1/G2/G3. Source config validation remains authoritative. Existing
checkpoint and selected-episode provenance checks still run.

Inspect the resolved protocol without data, weights, or an environment:

```bash
python scripts/evaluate_actor_free_td_lewm_full_plan.py \
  --version v1 --variant c \
  --config configs/experiment/actor_free_td_lewm_v1_c_cube_checkpoint_o50.yaml \
  --score-mode f_plus_g_first --g-first-weight 0.25 --dry-run
```

For formal evaluation, add these arguments and remove `--dry-run`:

```text
--dataset <existing-cube-dataset>
--checkpoint-path <historical-checkpoint.pt>
--checkpoint-sha256 <historical-64-character-sha256>
--output-dir <new-empty-result-directory>
```

Use the previously declared alpha, not a newly selected value. The example
does not declare 0.25 universally optimal. Intermediate V2/EMA checkpoints
from epochs 3–9 additionally require `--checkpoint-epoch`; omit it for the
formal final checkpoint. Formal runs require CUDA. An unavailable GPU fails
before environment evaluation, rather than silently starting a CPU run.

Each output records `execution_protocol=full_plan_25_steps_real_feedback_v1`,
the original configured-protocol hash, both feedback intervals, the existing
checkpoint hash and selected pairs, and per-episode results. Existing or
partially populated output directories are refused, not overwritten. To
retry an interrupted evaluation, keep its artifacts and use a new directory.

The focused regression tests include the actual installed CEM solver and
WorldModelPolicy with a synthetic recording cost. They verify CEM origins
at primitive steps `0,25` for the new protocol versus `0,5,10,15,20,25` for
the old protocol. They are not substitutes for formal environment results.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/unit/test_full_plan_revalidation.py
```
