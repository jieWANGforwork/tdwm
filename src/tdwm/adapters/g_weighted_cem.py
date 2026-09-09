"""G-weighted elite moments using the installed CEM's public callback API.

F's full-rollout terminal cost still selects the elites. Only the contribution
of those elites to the next Gaussian changes; no solver, policy, sampling loop,
or baseline package code is copied. CEM 0.1.1 passes its live mean and standard
deviation tensors to callbacks, so updating those tensors also updates the
next iteration and the final returned mean plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import project_tasks_to_sphere_v1


@dataclass(frozen=True)
class GWeightedCEMConfig:
    mode: str
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"path", "action"}:
            raise ValueError("G weighting mode must be 'path' or 'action'.")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("G weighting temperature must be finite and positive.")

    @property
    def evaluation_mode(self) -> str:
        return f"g_{self.mode}_weighted_cem"


def elite_g_weights(scores: torch.Tensor, config: GWeightedCEMConfig) -> torch.Tensor:
    """Normalize over elite paths independently for each environment/position."""
    if scores.ndim != 3 or min(scores.shape) <= 0 or scores.shape[1] < 2:
        raise ValueError("G scores must have shape (batch, elites>=2, horizon).")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("G scores must be finite floating-point values.")
    working = (
        scores.float() if scores.dtype in {torch.float16, torch.bfloat16} else scores
    )
    if config.mode == "path":
        working = working.mean(dim=-1, keepdim=True).expand_as(working)
    # Subtract before dividing to avoid overflow for large finite positive Q.
    logits = (working - working.amax(dim=1, keepdim=True)) / config.temperature
    return torch.softmax(logits, dim=1)


def weighted_elite_moments(
    candidates: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit weighted means/stds, preserving the baseline's sample-std convention.

    The variance denominator is 1-sum(w^2), hence uniform weights give the same
    K-1 correction as upstream torch.std. This is a declared weighted-moment
    convention, not a claim of unbiasedness for value-dependent sample weights.
    Fully concentrated weights have zero spread rather than a 0/0 NaN.
    """
    if candidates.ndim != 4 or weights.shape != candidates.shape[:-1]:
        raise ValueError(
            "Actions and weights must align as (batch, elites, horizon[, dim])."
        )
    if candidates.shape[1] < 2 or not candidates.is_floating_point():
        raise ValueError("At least two floating-point action elites are required.")
    if (
        not bool(torch.isfinite(candidates).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or not torch.allclose(weights.sum(1), torch.ones_like(weights.sum(1)))
    ):
        raise ValueError("Finite nonnegative weights must sum to one over elites.")
    working = candidates.to(weights.dtype)
    mean = (weights[..., None] * working).sum(dim=1)
    residual = working - mean.unsqueeze(1)
    numerator = (weights[..., None] * residual.square()).sum(dim=1)
    denominator = 1.0 - weights.square().sum(dim=1)
    std = (
        numerator / denominator.clamp_min(torch.finfo(weights.dtype).eps)[..., None]
    ).sqrt()
    # Equal scores must reproduce upstream exactly, including operation order.
    uniform = (weights.amax(1) == weights.amin(1))[..., None]
    mean = torch.where(uniform, candidates.mean(1), mean)
    std = torch.where(uniform, candidates.std(1), std)
    return mean.to(candidates.dtype), std.to(candidates.dtype)


def _select_paths(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    shape = (*indices.shape, *values.shape[2:])
    expanded = indices.reshape(*indices.shape, *([1] * (values.ndim - 2))).expand(shape)
    return values.gather(1, expanded)


class GWeightedPlanningModel(nn.Module):
    """Reuse each full F rollout; run G only on the selected elite states."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if getattr(base, "score_mode", None) != "f_only":
            raise ValueError(
                "G-weighted CEM requires unchanged F-only elite selection."
            )
        if base.training or any(
            parameter.requires_grad for parameter in base.parameters()
        ):
            raise ValueError(
                "The evaluation world model and G must both be frozen/eval."
            )
        self.base = base
        self.state_only = hasattr(base, "_rollout_next_ghost_states")
        if not self.state_only and not hasattr(base, "_rollout_future"):
            raise ValueError("This adapter does not expose aligned per-step G states.")
        self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.eval()

    def clear_cache(self) -> None:
        self._cache = None

    @torch.inference_mode()
    def get_cost(
        self, info_dict: dict[str, Any], actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim != 4 or actions.shape[-2:] != (5, 25):
            raise ValueError(
                "G-weighted CEM requires five normalized 25D action blocks."
            )
        if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
            raise ValueError(
                "Candidate action blocks must be finite floating-point values."
            )
        batch, samples, horizon = actions.shape[:3]
        # Upstream owns this per-solve dictionary. Retain its observation/goal
        # embedding cache across CEM iterations, just as the baseline does.
        info = info_dict
        kwargs = dict(batch=batch, samples=samples, reference=actions)
        goal = self.base._goal_for_samples(info, **kwargs)
        current = (
            None
            if self.state_only
            else self.base._current_state_for_samples(info, **kwargs)
        )
        rollout = (
            self.base._rollout_next_ghost_states
            if self.state_only
            else self.base._rollout_future
        )
        future = rollout(info, actions, batch=batch, samples=samples, horizon=horizon)
        states = (
            future
            if self.state_only
            else torch.cat((current.unsqueeze(-2), future[..., :-1, :]), dim=-2)
        )
        tasks = project_tasks_to_sphere_v1(goal).unsqueeze(-2).expand_as(states)
        self._cache = (actions, states.detach(), tasks.detach())
        return self.base._explicit_terminal_cost(future, goal)

    @torch.inference_mode()
    def score_elites(
        self, candidates: torch.Tensor, indices: torch.Tensor, elites: torch.Tensor
    ) -> torch.Tensor:
        if self._cache is None or self._cache[0] is not candidates:
            raise RuntimeError("G elite scores require the matching cached F rollout.")
        _, all_states, all_tasks = self._cache
        states = _select_paths(all_states, indices)
        tasks = _select_paths(all_tasks, indices)
        if states.shape[:-1] != elites.shape[:-1]:
            raise ValueError(
                "Selected G states and candidate action positions do not align."
            )
        if self.state_only:
            scores = self.base._goal_score(states, tasks)
        else:
            scores = self.base._goal_score(states, elites, tasks)
        if scores.shape != elites.shape[:-1]:
            raise ValueError("G must return one scalar per elite action position.")
        self.clear_cache()
        return scores


class GWeightedEliteUpdate:
    """Public CEM callback that replaces only the fitted elite moments."""

    output_key = "g_weighted_elite_update"

    def __init__(
        self, model: GWeightedPlanningModel, config: GWeightedCEMConfig
    ) -> None:
        self.model = model
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.history: list[list[dict[str, float]]] = []
        self._current: list[dict[str, float]] = []
        self.model.clear_cache()

    def start_batch(self) -> None:
        if self._current:
            self.history.append(self._current)
        self._current = []

    def end_solve(self) -> None:
        if self._current:
            self.history.append(self._current)
        self._current = []
        self.model.clear_cache()

    @torch.inference_mode()
    def __call__(self, **state: Any) -> None:
        elites = state["topk_candidates"]
        scores = self.model.score_elites(
            state["candidates"], state["topk_inds"], elites
        )
        weights = elite_g_weights(scores, self.config)
        mean, std = weighted_elite_moments(elites, weights)
        # These are the live tensors used by the next upstream CEM iteration.
        state["mean"].copy_(mean)
        state["var"].copy_(std)  # Upstream's 'var' variable is a standard deviation.
        self._current.append(
            {
                "effective_elites_mean": float(
                    weights.square().sum(1).reciprocal().mean()
                ),
                "max_weight_mean": float(weights.amax(1).mean()),
            }
        )


def attach_g_weighted_cem(policy: Any, config: GWeightedCEMConfig) -> Any:
    """Attach to a newly built upstream policy, without replacing its solver."""
    import stable_worldmodel as swm

    solver = policy.solver
    if type(solver) is not swm.solver.CEMSolver:
        raise ValueError(
            "G weighting is verified for the installed public CEMSolver only."
        )
    if solver.topk < 2:
        raise ValueError("G-weighted CEM requires at least two elites.")
    if any(isinstance(callback, GWeightedEliteUpdate) for callback in solver.callbacks):
        raise ValueError("G weighting is already attached to this policy.")
    model = GWeightedPlanningModel(solver.model)
    solver.model = model
    solver.callbacks.insert(0, GWeightedEliteUpdate(model, config))
    return policy
