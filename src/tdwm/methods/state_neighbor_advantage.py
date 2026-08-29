"""Detached state-neighbor advantages for weighting real-data TD updates.

Counterfactual actions are used only to compare the current successor's scores.
They never receive Bellman targets and cannot contribute gradients to the
weighted TD loss.  The caller remains responsible for constructing the real
data-transition TD loss that this module reweights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tdwm.methods.successor_geometry import successor_goal_cost

VARIANT = "neighbor_action_advantage"
OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class StateNeighborAdvantageOutput:
    """Weighted real-data TD loss and detached weighting diagnostics."""

    loss: torch.Tensor
    weights: torch.Tensor
    advantage: torch.Tensor
    positive_scores: torch.Tensor
    neighbor_scores: torch.Tensor
    neighbor_attention: torch.Tensor


def _validate_temperature(value: float, *, name: str) -> float:
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return temperature


def _validate_g1_inputs(
    real_history: torch.Tensor,
    previous_actions: torch.Tensor,
    positive_actions: torch.Tensor,
    neighbor_actions: torch.Tensor,
    goals: torch.Tensor,
    distances: torch.Tensor,
    real_td_per_transition: torch.Tensor,
) -> tuple[int, int, int]:
    if real_history.ndim != 3:
        raise ValueError("real_history must have shape (N, history, latent_dim).")
    count, history_size, latent_dim = real_history.shape
    if min(count, history_size, latent_dim) <= 0:
        raise ValueError("real_history axes must be non-empty.")
    if positive_actions.ndim != 2 or positive_actions.shape[0] != count:
        raise ValueError("positive_actions must have shape (N, action_dim).")
    action_dim = int(positive_actions.shape[-1])
    if action_dim <= 0:
        raise ValueError("positive_actions must have a non-empty action dimension.")
    expected_previous = (count, history_size - 1, action_dim)
    if previous_actions.shape != expected_previous:
        raise ValueError(
            "previous_actions must have shape "
            f"{expected_previous}, found {tuple(previous_actions.shape)}."
        )
    if neighbor_actions.ndim != 3 or neighbor_actions.shape[0] != count:
        raise ValueError("neighbor_actions must have shape (N, neighbors, action_dim).")
    neighbor_count = int(neighbor_actions.shape[1])
    if neighbor_count <= 0 or neighbor_actions.shape[2] != action_dim:
        raise ValueError(
            "neighbor_actions must contain at least one action with the positive "
            "action dimension."
        )
    if goals.shape != (count, latent_dim):
        raise ValueError(
            f"goals must have shape {(count, latent_dim)}, found {tuple(goals.shape)}."
        )
    if distances.shape != (count, neighbor_count):
        raise ValueError(
            "distances must match the leading neighbor action axes "
            f"{(count, neighbor_count)}."
        )
    if real_td_per_transition.shape != (count,):
        raise ValueError(
            "real_td_per_transition must contain one scalar loss per transition."
        )

    tensors = (
        real_history,
        previous_actions,
        positive_actions,
        neighbor_actions,
        goals,
        distances,
        real_td_per_transition,
    )
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("State-neighbor advantage inputs must be floating tensors.")
    if any(tensor.device != real_history.device for tensor in tensors[1:]):
        raise ValueError("State-neighbor advantage inputs must share one device.")
    if not torch.isfinite(distances).all():
        raise ValueError("distances must be finite.")
    if torch.any(distances < 0):
        raise ValueError("distances cannot be negative.")
    if not torch.isfinite(real_td_per_transition).all():
        raise ValueError("real_td_per_transition must be finite.")
    if torch.any(real_td_per_transition < 0):
        raise ValueError("real_td_per_transition cannot be negative.")
    return count, neighbor_count, latent_dim


def state_neighbor_advantage_weighted_td_loss(
    successor,
    real_history: torch.Tensor,
    previous_actions: torch.Tensor,
    positive_actions: torch.Tensor,
    neighbor_actions: torch.Tensor,
    goals: torch.Tensor,
    distances: torch.Tensor,
    real_td_per_transition: torch.Tensor,
    *,
    neighbor_temperature: float,
    weight_temperature: float,
) -> StateNeighborAdvantageOutput:
    """Use detached counterfactual action advantage to weight real TD losses.

    ``positive_actions`` and each row of ``neighbor_actions`` are evaluated with
    the exact same state history, previous actions, and goal.  A smaller
    successor goal cost is a larger score.  Neighbor distances define a local
    softmax baseline, and a second stable softmax maps detached advantages to
    non-negative transition weights whose batch mean is exactly one.

    Only ``real_td_per_transition`` contributes gradients to ``loss``.  In
    particular, neighbor actions are never assigned TD targets.
    """

    neighbor_tau = _validate_temperature(
        neighbor_temperature, name="neighbor_temperature"
    )
    weight_tau = _validate_temperature(weight_temperature, name="weight_temperature")
    count, neighbor_count, latent_dim = _validate_g1_inputs(
        real_history,
        previous_actions,
        positive_actions,
        neighbor_actions,
        goals,
        distances,
        real_td_per_transition,
    )

    candidate_actions = torch.cat(
        (positive_actions.unsqueeze(1), neighbor_actions), dim=1
    )
    candidate_count = neighbor_count + 1
    candidate_history = real_history.unsqueeze(1).expand(-1, candidate_count, -1, -1)
    candidate_previous_actions = previous_actions.unsqueeze(1).expand(
        -1, candidate_count, -1, -1
    )
    # Counterfactual scores are a detached weighting signal, never a learning
    # target.  Avoid constructing a successor autograd graph that would be
    # discarded immediately after the advantage is formed.
    with torch.no_grad():
        candidate_successor = successor(
            candidate_history,
            candidate_previous_actions,
            candidate_actions,
        )
    expected_output_shape = (count, candidate_count, latent_dim + 2)
    if candidate_successor.shape != expected_output_shape:
        raise ValueError(
            "successor output must have shape "
            f"{expected_output_shape}, found {tuple(candidate_successor.shape)}."
        )

    # Goal readout and both softmaxes stay in FP32 under mixed precision.  The
    # detach is the key causal boundary: counterfactual scores select the size
    # of a real-data update but are not themselves an optimization objective.
    with torch.no_grad():
        scores = -successor_goal_cost(
            candidate_successor.float(),
            goals.float().unsqueeze(1),
        )
        positive_scores = scores[:, 0]
        neighbor_scores = scores[:, 1:]
        neighbor_attention = torch.softmax(
            -distances.detach().float() / neighbor_tau, dim=-1
        )
        advantage = positive_scores - (neighbor_attention * neighbor_scores).sum(dim=-1)
    weights = torch.softmax(advantage / weight_tau, dim=0) * float(count)
    weights = weights.to(real_td_per_transition)
    loss = (weights * real_td_per_transition).mean()

    return StateNeighborAdvantageOutput(
        loss=loss,
        weights=weights.detach(),
        advantage=advantage,
        positive_scores=positive_scores.detach(),
        neighbor_scores=neighbor_scores.detach(),
        neighbor_attention=neighbor_attention.detach(),
    )


__all__ = [
    "OBJECTIVE_VERSION",
    "StateNeighborAdvantageOutput",
    "VARIANT",
    "state_neighbor_advantage_weighted_td_loss",
]
