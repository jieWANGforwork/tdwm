# Executed-rollout recording for Eff/EffPlan

Formal `scripts/evaluate_effplan.py evaluate` runs now record by default for
F-only, Eff, and EffPlan (including action-perturbation scoring). The explicit
`--no-record-rollouts` flag disables recording for scheduling-only diagnostics.
Existing `--video` remains independent and optional. No trained weight, scoring
formula, random seed, action queue, execution budget or replanning rule changes.

Recording uses SWM's public `extra_wrappers` and a delegating policy. It copies
commands actually passed to each environment's `step`, after the policy's
inverse normalization. Commands are NOT CEM candidates or internal actuator
controls. The recorder does not clip or modify commands. Internal environment
clipping, if any, remains the environment's responsibility.

## Website-readable layout

```
evaluation/
  result.json
  protocol_manifest.json
  rollouts/
    manifest.json
    episode_0000/
      episode.json
      goal.png
      frames/000000.png
      frames/000001.png
      ...
      steps.jsonl
      trajectory.json
    ...
```

The index is the fixed pair's position, not its source dataset episode number.
Metadata includes both IDs, dataset start/goal indices, protocol, score mode,
selection SHA and checkpoint identities. `manifest.json` links episodes using
paths relative to `rollouts/`; paths inside each trajectory are relative to its
episode directory. These JSON/PNG assets can be served directly by a website.

Each transition has zero-based `step`, the executed `action` and dtype,
`before_frame`, `after_frame`, reward, terminated/truncated flags, and available
numeric state (`qpos`, `qvel`, observation, privileged coordinates). Big-action
grouping is explicit: `action_block_index = step // 5` and
`primitive_index_in_block = step % 5`. It does not imply extra replanning.

Frame 0 is the selected dataset start image supplied to the policy AFTER SWM
restores the episode state, not the unrelated reset image. `goal.png` is the
same dataset goal supplied to planning. Frame t+1 is the actual canonical
environment observation AFTER executed action t, copied from SWM's existing
pixels. These are not F predictions. A trajectory with T executed actions has
exactly T+1 observation PNGs, plus its goal PNG. No additional render or model
rollout is performed to capture these images.

Masked/completed environments are never stepped or padded by this recorder;
the last success/truncation image is retained. PNG is lossless RGB. Each
transition is also streamed into `steps.jsonl`; failure leaves an `incomplete`
manifest and available partial trajectories rather than a false success.
Recording refuses to overwrite any prior rollout directory. Final validation
checks all pair outcomes against actual recorded termination and budget.

## Tests and paired reruns

```
PYTHONPATH=src python -m pytest -q tests/unit/test_rollout_recording.py tests/integration/test_rollout_recording_public_api.py tests/integration/test_effplan_execution_protocol.py
```

The public-API test calls the installed SWM World.evaluate, comparing recording
on/off for identical actions, outcomes, render counts and early termination.
Other tests cover restored frame zero, PNG/action alignment, immutable stored
frames, dtype, action mutation by an environment, terminal masks, no RNG use,
partial output, and overwrite rejection.

An optional real-Cube CPU check executes at most three zero-command steps in
one environment using existing data (no model inference and no formal score):
set `TDWM_RECORDING_CUBE_DATASET` and `TDWM_RECORDING_CUBE_SELECTION`, retain the
locked `MUJOCO_GL=osmesa` backend, and run
`tests/integration/test_rollout_recording_cube.py`. It verifies actual 224x224
RGB files, five-dimensional commands, qpos/qvel and the final observation.

The requested original P+CEM and action-risk O25/O50/O100 re-evaluations must
use new output directories, the original checkpoints and exact fixed pairs.
Recorded reruns are new executions, not recovered recordings of the old runs.
Only code/tests/docs belong in Git; images and raw trajectories remain external
artifacts. Do not publish raw observations or checkpoint files to GitHub.
