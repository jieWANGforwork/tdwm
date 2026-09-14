# Sparse dynamics calibration continuation

Opt-in configurations: `effplan_cube_stable_p_sparse_v1.yaml` (old V) and
`effplan_cube_extra_work_sparse_v1.yaml` (extra-work V). Old configurations,
Eff weights, generation weights, and evaluation formulas remain unchanged.

Regular updates keep the batch of 256 and four state-refinement rounds, but
use G/V efficiency feedback only: trajectory loss + 0.1 * negative efficiency.
They never call CEM/F. Every tenth absolute optimizer update uses the first
32 IID paths of the sampled batch and the original four CEM/refinement rounds
([2,2,2,2], 300 candidates). That calibration update uses trajectory + 0.1 *
negative efficiency + 0.1 * dynamics consistency. It is one optimizer update,
not an extra optimizer step. The IID-prefix subset adds no sampling RNG draws.
Calibration updates reduce the whole supervised minibatch to 32, not just the
dynamics term. This deliberate new training schedule must not be relabeled as
the original dense experiment. No rescaling by the interval is applied.

Validation always runs calibration on the capped subset, independent of the
current step modulo ten; validation preserves training and CEM RNG states.
Metrics report whether the update calibrated and the actual CEM rollout count.
Nominal training CEM count is reduced by 80x, but wall-clock speed must be
measured: state feedback, regression and validation still cost time.

Use `scripts/train_effplan.py --branch-from PARENT.pt` with a NEW output dir.
The transition accepts only the two new calibration setting differences from
a dense refinement checkpoint. F/G/V source identity and all other settings
must agree. It restores P, optimizer moments, NumPy/Torch/CUDA/CEM RNG and
global_step, preserving the full-budget LR schedule (no new warmup). Parent
path, SHA and switch step are saved in checkpoint and training manifest.
Subsequent resumes use ordinary `--resume`, not `--branch-from` again.

New runner processes honor a `STOP_REQUESTED` file in their output directory
between accepted updates, save an atomic full checkpoint and mark themselves
paused. They do not publish a completed planner manifest. Remove the marker
before explicitly resuming. This feature cannot be hot-loaded into an already
running old Python process. Never claim it saved that old process's current
in-memory state: use a verified completed checkpoint and record the exact
branch step. Existing old logs/checkpoints are retained, never overwritten.

Inference is unchanged: P state planning and CEM/F action matching; H5/RH5,
O25/O50/O100 fixed pairs and budgets. The 30 completed Eff-only results remain
valid because G/V and their scoring are unchanged; do not rerun them, F-only,
or the excluded mean mode. Only the six final EffPlan evaluations are pending.
