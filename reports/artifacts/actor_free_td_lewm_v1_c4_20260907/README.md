# V1-C4 formal Cube result bundle

This directory is the lightweight, auditable result bundle for Actor-Free TD-LeWM V1-C4. The formal run uses training seed 3072, planning seed 42, the frozen V1 LeWM checkpoint, and the fixed O25/O50/O100 start-goal selections. It contains 18 completed evaluation cells and 900 per-episode Boolean outcomes.

## Method and objective

C4 removes action from the successor network interface:

\[
G_{C4}(z_i,m)\rightarrow\Psi_i\in\mathbb{R}^{192},\qquad
q_{C4}(z_i,m)=\Psi_i^\top m.
\]

The observation encoder, action encoder, and LeWM predictor \(F\) are frozen. Action can influence C4 only through a stopped predicted state from \(F\). The aligned online inputs are

\[
x_i^{real}=z_i,\qquad x_i^{pred}=\operatorname{sg}[F(z_{i-1},a_{i-1})].
\]

Both branches share

\[
Y_i=\operatorname{sg}\left[z_i+\gamma(1-d_i)\bar G_{C4}(z_{i+1},m)\right].
\]

For a terminal current state, \(Y_i=z_i\). The objective is

\[
L_{C4}=\tfrac12\left[(L_{vec}^{real}+L_{goal}^{real})+(L_{vec}^{pred}+L_{goal}^{pred})\right],
\]

where vector TD is the mean squared 192-D L2 error and goal TD is the projected scalar MSE on goal-derived samples only. \(\lambda_C=1\).

## Formal results

Each entry is successes out of the same 50 pairs for that protocol.

| Protocol | F-only | C4-only | F+C4 tail | First-Q, alpha=.25 | Mean-Q | First-Q2, alpha=.25 |
|---|---:|---:|---:|---:|---:|---:|
| O25 | 37 (74%) | 33 (66%) | 36 (72%) | 33 (66%) | 34 (68%) | 32 (64%) |
| O50 | 23 (46%) | 18 (36%) | 23 (46%) | 25 (50%) | 21 (42%) | 24 (48%) |
| O100 | 25 (50%) | 25 (50%) | 24 (48%) | 29 (58%) | 28 (56%) | 31 (62%) |

The predeclared scorers are:

- `F-only`: terminal five-step LeWM goal distance.
- `C4-only`: \(-q_{C4}(\hat z_1^F,m)\); the first action reaches C4 only through \(F\).
- `F+C4 tail`: terminal \(F\) cost through \(\hat z_4^F\) minus \(\gamma^4 q_{C4}(\hat z_5^F,m)\); the fifth action is first passed through \(F\).
- `First-Q`: full five-step \(F\) rollout cost minus \(0.25q_{C4}(\hat z_1^F,m)\), using raw scales.
- `Mean-Q`: \(-\frac15\sum_{k=1}^{5}q_{C4}(\hat z_k^F,m)\).
- `First-Q2`: population-z-scored \(F\) cost minus 0.25 times population-z-scored first Q, normalized over the CEM candidate axis per environment.

Relative to the same-protocol F-only baseline, the best deployed C4 score is unchanged F-only on O25, First-Q on O50 (+2 successes), and First-Q2 on O100 (+6 successes). The largest oracle `F+New` unions are 43/50 on O25 (Mean-Q), 29/50 on O50 (tail or First-Q), and 34/50 on O100 (First-Q2); these unions are diagnostic ceilings, not implemented controllers.

## Training evidence

- Training: 10 epochs, 12,796 optimizer steps per epoch, 127,960 total updates.
- Optimizer scope: online `G_C4` only; 0 trainable LeWM parameters; EMA target frozen.
- Frozen LeWM source SHA-256: `198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257`.
- Deployment checkpoint on server: `/root/autodl-tmp/tdwm/outputs/actor_free_td_lewm_v1_c4_f29f779_20260906/seed_3072/checkpoints/actor_free_td_lewm_v1_c4/c4/epoch_10.pt`.
- Deployment checkpoint SHA-256: `28a59d0b07cb2e0ea66b34c57fdc1eb8dce513ca80b8a8700cc36ad9458ef99b`.
- Clean training audit: `PASS`; best validation total was epoch 9.
- Epoch-10 train total: 41,290.0664; validation total: 17,941.4453.
- Epoch-10 train goal/vector scale ratio: 22.12x; validation ratio: 10.42x. This motivates a predeclared loss-scale balancing experiment rather than a post-hoc scorer choice.

## Verification

- Three formal launchers: `SUCCEEDED`.
- 18/18 jobs: `state=SUCCEEDED`, `exit_code=0`.
- 18 result files: exactly 50 Boolean outcomes each.
- C4/formal-summary/report-update test set: 106 passed, 0 skipped, 0 failed.
- Ruff: 19 C4-related Python files, 0 errors.
- Python AST parse: 19 C4-related Python files, 0 errors.
- The updated Results TD DOCX renders to 48 pages; all pages were inspected and the promoted copy is pixel-identical to the staged render.

## Bundle layout

- `training/`: training manifest, raw scalar metrics, epoch summary, clean audit, and loss figure.
- `formal/summary/`: formal matrix, episode matrix, and human-readable summary.
- `formal/launchers/`: the three successful O25/O50/O100 launcher manifests.
- `formal/results/`: per-mode result, selection, normalization, and protocol manifests.
- `checksums.sha256`: hashes for every tracked evidence file in this bundle.

The full method implementation spans the C4 method, adapter, training, evaluation, result summarization, launchers, configurations, and unit tests under `src/`, `scripts/`, `configs/experiment/`, and `tests/unit/`. The canonical consolidated report is `reports/actor_free_td_lewm_complete_cube_seed3072.md`, with the matching Word artifact at `reports/results_td_actor_free_td_lewm_complete_cube_seed3072.docx`.
