"""Pure C--G3 objectives for the goal-conditioned actor-free TD-LeWM V0.

V0 keeps objective reduction separate from the predictor and data pipeline.  A
caller supplies the real-transition TD loss and detached scoring inputs.  This
module therefore cannot create Bellman targets for neighbor or prefix actions:
those actions are comparison-only signals by construction.

The mixed-task contract is shared by all objectives:

* ``goal_mask`` identifies goal-derived tasks in an otherwise mixed task batch;
* every transition contributes its ordinary TD loss;
* C applies its extra projection loss only to goal-derived transitions; and
* D/F/G apply detached weighting only within the goal-derived subset.  Random
  task transitions keep unit weight, and final weights have mean one over all
  transitions.

When fewer than two goal-derived transitions are available, a weighting
objective falls back to the unweighted base TD loss.  This keeps small or
imbalanced batches trainable and avoids assigning meaning to a singleton
softmax.  C can still use one goal-derived transition because it does not form
a cross-sample weighting distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

VARIANT_FAMILY = "actor_free_td_lewm_v0"
OBJECTIVE_VERSION = 0
ACTION_PREFIX_SLOTS = 5


@dataclass(frozen=True)
class GoalProjectedV0Output:
    """C: base TD plus a goal-subset projection residual."""

    loss: torch.Tensor
    base_td_loss: torch.Tensor
    projection_loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    prediction_score: torch.Tensor
    target_score: torch.Tensor
    score_residual: torch.Tensor
    goal_indices: torch.Tensor


@dataclass(frozen=True)
class GoalValueWeightedV0Output:
    """D: detached TD-target goal values weight real-transition TD losses."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    target_score: torch.Tensor
    weights: torch.Tensor
    goal_indices: torch.Tensor
    used_weighting: bool


@dataclass(frozen=True)
class SameFutureGoalAdvantageV0Output:
    """F: one fixed TD target is compared with goal-subset tasks."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    score_matrix: torch.Tensor
    positive_score: torch.Tensor
    all_goal_mean_score: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    goal_indices: torch.Tensor
    used_weighting: bool


@dataclass(frozen=True)
class NeighborActionAdvantageV0Output:
    """G1: detached neighbor-action advantage weights real TD only."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    positive_score: torch.Tensor
    neighbor_scores: torch.Tensor
    neighbor_attention: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    goal_indices: torch.Tensor
    used_weighting: bool


@dataclass(frozen=True)
class PrefixMeanAdvantageV0Output:
    """G2: detached full-minus-prefix-mean advantage weights real TD."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    prefix_scores: torch.Tensor
    full_score: torch.Tensor
    prefix_mean_score: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    goal_indices: torch.Tensor
    used_weighting: bool


@dataclass(frozen=True)
class PrefixMarginalAdvantageV0Output:
    """G3: detached mean adjacent-prefix marginal weights real TD."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    prefix_scores: torch.Tensor
    marginal_scores: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    goal_indices: torch.Tensor
    used_weighting: bool


def _validate_float_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    ndim: int,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} axes must be non-empty.")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on device {device}.")
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_per_transition_td(
    per_transition_td_loss: torch.Tensor,
) -> tuple[int, torch.device]:
    _validate_float_tensor(
        "per_transition_td_loss", per_transition_td_loss, ndim=1
    )
    if bool(torch.any(per_transition_td_loss.detach() < 0)):
        raise ValueError("per_transition_td_loss cannot be negative.")
    return int(per_transition_td_loss.shape[0]), per_transition_td_loss.device


def _validate_goal_mask(
    goal_mask: torch.Tensor,
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(goal_mask, torch.Tensor):
        raise TypeError("goal_mask must be a torch.Tensor.")
    if goal_mask.dtype != torch.bool:
        raise TypeError("goal_mask must have dtype torch.bool.")
    if goal_mask.shape != (count,):
        raise ValueError(f"goal_mask must have shape {(count,)}.")
    if goal_mask.device != device:
        raise ValueError("goal_mask and TD losses must be on the same device.")
    return torch.nonzero(goal_mask, as_tuple=False).flatten()


def _validate_temperature(value: float, *, name: str) -> float:
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return temperature


def _validate_td_and_mask(
    per_transition_td_loss: torch.Tensor,
    goal_mask: torch.Tensor,
) -> tuple[int, torch.device, torch.Tensor]:
    count, device = _validate_per_transition_td(per_transition_td_loss)
    goal_indices = _validate_goal_mask(goal_mask, count=count, device=device)
    return count, device, goal_indices


def _validate_targets_and_tasks(
    td_targets: torch.Tensor,
    tasks: torch.Tensor,
    *,
    count: int,
    device: torch.device,
) -> int:
    _validate_float_tensor(
        "td_targets", td_targets, ndim=2, device=device
    )
    _validate_float_tensor("tasks", tasks, ndim=2, device=device)
    if td_targets.shape != tasks.shape:
        raise ValueError("td_targets and tasks must have identical shapes.")
    if td_targets.shape[0] != count:
        raise ValueError("td_targets must contain one row per transition.")
    return int(td_targets.shape[-1])


def _goal_subset_weights(
    signal: torch.Tensor,
    goal_indices: torch.Tensor,
    *,
    count: int,
    temperature: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Return detached full-batch weights with random-task weights fixed at one."""

    tau = _validate_temperature(temperature, name="temperature")
    expected = (int(goal_indices.numel()),)
    if signal.shape != expected:
        raise ValueError(f"weighting signal must have shape {expected}.")
    if not signal.is_floating_point():
        raise TypeError("weighting signal must be floating point.")
    if signal.device != device:
        raise ValueError("weighting signal must share the TD-loss device.")
    if not bool(torch.isfinite(signal.detach()).all()):
        raise ValueError("weighting signal must contain only finite values.")

    # A singleton softmax is always one and cannot express an advantage.  Use
    # the same explicit base-TD fallback for zero and one goal transitions.
    if goal_indices.numel() < 2:
        return torch.ones(count, dtype=dtype, device=device), False

    with torch.no_grad():
        subset_weights = torch.softmax(signal.detach().float() / tau, dim=0)
        subset_weights = subset_weights * float(goal_indices.numel())
        weights = torch.ones(count, dtype=torch.float32, device=device)
        weights.index_copy_(0, goal_indices, subset_weights)
        # The subset already has mean one.  Normalizing once over the complete
        # mixed batch makes the full-N invariant explicit and robust to roundoff.
        weights = weights / weights.mean()
    return weights.to(dtype=dtype), True


def _weighted_td_loss(
    per_transition_td_loss: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (per_transition_td_loss * weights).mean()


def goal_projected_v0_loss(
    online_predictions: torch.Tensor,
    td_targets: torch.Tensor,
    tasks: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
    *,
    projection_coefficient: float,
) -> GoalProjectedV0Output:
    """Compute C using the single symmetric online predictor.

    ``online_predictions`` has shape ``(transitions, features)``.  The ordinary
    TD term is always averaged over the full mixed-task batch.
    """

    count, device, goal_indices = _validate_td_and_mask(
        per_transition_td_loss, goal_mask
    )
    _validate_float_tensor(
        "online_predictions", online_predictions, ndim=2, device=device
    )
    feature_dim = _validate_targets_and_tasks(
        td_targets, tasks, count=count, device=device
    )
    if online_predictions.shape != (count, feature_dim):
        raise ValueError(
            "online_predictions must have shape (transitions, features)."
        )
    coefficient = float(projection_coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("projection_coefficient must be finite and non-negative.")

    detached_tasks = tasks.detach().float()
    prediction_score = (online_predictions.float() * detached_tasks).sum(dim=-1)
    with torch.no_grad():
        target_score = (td_targets.detach().float() * detached_tasks).sum(dim=-1)
    score_residual = prediction_score - target_score
    if goal_indices.numel() == 0:
        projection_loss = prediction_score.sum() * 0.0
    else:
        projection_loss = score_residual.index_select(0, goal_indices).square().mean()
    base_td_loss = per_transition_td_loss.mean()
    return GoalProjectedV0Output(
        loss=base_td_loss + coefficient * projection_loss,
        base_td_loss=base_td_loss,
        projection_loss=projection_loss,
        per_transition_td_loss=per_transition_td_loss,
        prediction_score=prediction_score,
        target_score=target_score,
        score_residual=score_residual,
        goal_indices=goal_indices,
    )


def goal_value_weighted_v0_loss(
    td_targets: torch.Tensor,
    tasks: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
    *,
    temperature: float,
) -> GoalValueWeightedV0Output:
    """Compute D from detached TD-target/task dot products."""

    count, device, goal_indices = _validate_td_and_mask(
        per_transition_td_loss, goal_mask
    )
    _validate_targets_and_tasks(
        td_targets, tasks, count=count, device=device
    )
    with torch.no_grad():
        target_score = (
            td_targets.detach().float() * tasks.detach().float()
        ).sum(dim=-1)
    goal_signal = target_score.index_select(0, goal_indices)
    weights, used_weighting = _goal_subset_weights(
        goal_signal,
        goal_indices,
        count=count,
        temperature=temperature,
        dtype=per_transition_td_loss.dtype,
        device=device,
    )
    return GoalValueWeightedV0Output(
        loss=_weighted_td_loss(per_transition_td_loss, weights),
        per_transition_td_loss=per_transition_td_loss,
        target_score=target_score,
        weights=weights.detach(),
        goal_indices=goal_indices,
        used_weighting=used_weighting,
    )


def same_future_goal_advantage_v0_loss(
    td_targets: torch.Tensor,
    tasks: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
    *,
    temperature: float,
) -> SameFutureGoalAdvantageV0Output:
    """Compute F on the goal-derived subset while keeping each ``Y_i`` fixed.

    For goal indices ``I``, the detached matrix is
    ``td_targets[I] @ tasks[I].T``.  Its diagonal is the matched positive,
    and its row mean includes that positive exactly as specified by method F.
    """

    count, device, goal_indices = _validate_td_and_mask(
        per_transition_td_loss, goal_mask
    )
    feature_dim = _validate_targets_and_tasks(
        td_targets, tasks, count=count, device=device
    )
    goal_count = int(goal_indices.numel())
    with torch.no_grad():
        selected_targets = td_targets.detach().float().index_select(
            0, goal_indices
        )
        selected_tasks = tasks.detach().float().index_select(0, goal_indices)
        if goal_count == 0:
            score_matrix = torch.empty(
                0, 0, dtype=torch.float32, device=device
            )
            # Keep the feature validation above meaningful even for this branch.
            assert feature_dim > 0
        else:
            score_matrix = selected_targets @ selected_tasks.transpose(0, 1)
        positive_score = score_matrix.diagonal()
        all_goal_mean_score = (
            score_matrix.mean(dim=1)
            if goal_count > 0
            else torch.empty(0, dtype=torch.float32, device=device)
        )
        advantage = positive_score - all_goal_mean_score
    weights, used_weighting = _goal_subset_weights(
        advantage,
        goal_indices,
        count=count,
        temperature=temperature,
        dtype=per_transition_td_loss.dtype,
        device=device,
    )
    return SameFutureGoalAdvantageV0Output(
        loss=_weighted_td_loss(per_transition_td_loss, weights),
        per_transition_td_loss=per_transition_td_loss,
        score_matrix=score_matrix,
        positive_score=positive_score,
        all_goal_mean_score=all_goal_mean_score,
        advantage=advantage.detach(),
        weights=weights.detach(),
        goal_indices=goal_indices,
        used_weighting=used_weighting,
    )


def neighbor_action_advantage_v0_loss(
    positive_scores: torch.Tensor,
    neighbor_scores: torch.Tensor,
    neighbor_distances: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
    *,
    neighbor_temperature: float,
    weight_temperature: float,
) -> NeighborActionAdvantageV0Output:
    """Compute G1 from already-evaluated action scores and neighbor distances."""

    count, device, goal_indices = _validate_td_and_mask(
        per_transition_td_loss, goal_mask
    )
    _validate_float_tensor("positive_scores", positive_scores, ndim=1, device=device)
    _validate_float_tensor("neighbor_scores", neighbor_scores, ndim=2, device=device)
    _validate_float_tensor(
        "neighbor_distances", neighbor_distances, ndim=2, device=device
    )
    if positive_scores.shape != (count,):
        raise ValueError("positive_scores must contain one score per transition.")
    if neighbor_scores.shape[0] != count:
        raise ValueError("neighbor_scores must contain one row per transition.")
    if neighbor_distances.shape != neighbor_scores.shape:
        raise ValueError("neighbor_distances must match neighbor_scores.")
    if bool(torch.any(neighbor_distances.detach() < 0)):
        raise ValueError("neighbor_distances cannot be negative.")
    neighbor_tau = _validate_temperature(
        neighbor_temperature, name="neighbor_temperature"
    )

    with torch.no_grad():
        selected_positive = positive_scores.detach().float().index_select(
            0, goal_indices
        )
        selected_neighbors = neighbor_scores.detach().float().index_select(
            0, goal_indices
        )
        selected_distances = neighbor_distances.detach().float().index_select(
            0, goal_indices
        )
        neighbor_attention = torch.softmax(
            -selected_distances / neighbor_tau, dim=-1
        )
        advantage = selected_positive - (
            neighbor_attention * selected_neighbors
        ).sum(dim=-1)
    weights, used_weighting = _goal_subset_weights(
        advantage,
        goal_indices,
        count=count,
        temperature=weight_temperature,
        dtype=per_transition_td_loss.dtype,
        device=device,
    )
    return NeighborActionAdvantageV0Output(
        loss=_weighted_td_loss(per_transition_td_loss, weights),
        per_transition_td_loss=per_transition_td_loss,
        positive_score=selected_positive,
        neighbor_scores=selected_neighbors,
        neighbor_attention=neighbor_attention,
        advantage=advantage.detach(),
        weights=weights.detach(),
        goal_indices=goal_indices,
        used_weighting=used_weighting,
    )


def _validate_prefix_inputs(
    prefix_scores: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
) -> tuple[int, torch.device, torch.Tensor, torch.Tensor]:
    count, device, goal_indices = _validate_td_and_mask(
        per_transition_td_loss, goal_mask
    )
    _validate_float_tensor("prefix_scores", prefix_scores, ndim=2, device=device)
    expected = (count, ACTION_PREFIX_SLOTS)
    if prefix_scores.shape != expected:
        raise ValueError(f"prefix_scores must have shape {expected}.")
    selected_prefixes = prefix_scores.detach().float().index_select(0, goal_indices)
    return count, device, goal_indices, selected_prefixes


def prefix_mean_advantage_v0_loss(
    prefix_scores: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
    *,
    temperature: float,
) -> PrefixMeanAdvantageV0Output:
    """Compute G2 exactly as ``q5 - mean(q1, ..., q5)``."""

    count, device, goal_indices, selected_prefixes = _validate_prefix_inputs(
        prefix_scores, goal_mask, per_transition_td_loss
    )
    with torch.no_grad():
        full_score = selected_prefixes[:, -1]
        prefix_mean_score = selected_prefixes.mean(dim=-1)
        advantage = full_score - prefix_mean_score
    weights, used_weighting = _goal_subset_weights(
        advantage,
        goal_indices,
        count=count,
        temperature=temperature,
        dtype=per_transition_td_loss.dtype,
        device=device,
    )
    return PrefixMeanAdvantageV0Output(
        loss=_weighted_td_loss(per_transition_td_loss, weights),
        per_transition_td_loss=per_transition_td_loss,
        prefix_scores=selected_prefixes,
        full_score=full_score,
        prefix_mean_score=prefix_mean_score,
        advantage=advantage.detach(),
        weights=weights.detach(),
        goal_indices=goal_indices,
        used_weighting=used_weighting,
    )


def prefix_marginal_advantage_v0_loss(
    prefix_scores: torch.Tensor,
    goal_mask: torch.Tensor,
    per_transition_td_loss: torch.Tensor,
    *,
    temperature: float,
) -> PrefixMarginalAdvantageV0Output:
    """Compute G3 as the mean of ``q2-q1`` through ``q5-q4``."""

    count, device, goal_indices, selected_prefixes = _validate_prefix_inputs(
        prefix_scores, goal_mask, per_transition_td_loss
    )
    with torch.no_grad():
        marginal_scores = selected_prefixes[:, 1:] - selected_prefixes[:, :-1]
        advantage = marginal_scores.mean(dim=-1)
    weights, used_weighting = _goal_subset_weights(
        advantage,
        goal_indices,
        count=count,
        temperature=temperature,
        dtype=per_transition_td_loss.dtype,
        device=device,
    )
    return PrefixMarginalAdvantageV0Output(
        loss=_weighted_td_loss(per_transition_td_loss, weights),
        per_transition_td_loss=per_transition_td_loss,
        prefix_scores=selected_prefixes,
        marginal_scores=marginal_scores,
        advantage=advantage.detach(),
        weights=weights.detach(),
        goal_indices=goal_indices,
        used_weighting=used_weighting,
    )


__all__ = [
    "ACTION_PREFIX_SLOTS",
    "OBJECTIVE_VERSION",
    "VARIANT_FAMILY",
    "GoalProjectedV0Output",
    "GoalValueWeightedV0Output",
    "NeighborActionAdvantageV0Output",
    "PrefixMarginalAdvantageV0Output",
    "PrefixMeanAdvantageV0Output",
    "SameFutureGoalAdvantageV0Output",
    "goal_projected_v0_loss",
    "goal_value_weighted_v0_loss",
    "neighbor_action_advantage_v0_loss",
    "prefix_marginal_advantage_v0_loss",
    "prefix_mean_advantage_v0_loss",
    "same_future_goal_advantage_v0_loss",
]
