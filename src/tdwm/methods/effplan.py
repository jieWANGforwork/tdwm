"""EffPlan's state generator/refiner; actions remain the job of SWM CEM.

The candidate states are optimization variables, not new observations. All
action matching is performed by an injected public-framework solver/rollout
adapter. No CEM, simulator or world-model implementation is duplicated here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from tdwm.methods.eff import STATE_DIM

ValueFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class StatePlanner(nn.Module):
    """P(left, candidate, right, value, dJ/dcandidate) -> state increment."""

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("Planner hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(4 * STATE_DIM + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )
        # Initial residual = 0; initialization is explicitly the endpoint mean.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        left: torch.Tensor,
        candidate: torch.Tensor,
        right: torch.Tensor,
        value: torch.Tensor,
        state_gradient: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not (left.shape == candidate.shape == right.shape == state_gradient.shape)
            or candidate.shape[-1] != STATE_DIM
        ):
            raise ValueError("Planner state inputs must match [..., 192].")
        if value.shape != candidate.shape[:-1]:
            raise ValueError("Planner feedback must have one scalar per state.")
        return self.network(
            torch.cat(
                (
                    left,
                    candidate,
                    right,
                    value.detach()[..., None],
                    state_gradient.detach(),
                ),
                dim=-1,
            )
        )


def binary_midpoint_order(horizon: int) -> list[tuple[int, int, int]]:
    """Parent-first subdivision; indices denote macro-step positions."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    order = []

    def split(left: int, right: int) -> None:
        if right - left < 2:
            return
        middle = (left + right) // 2
        order.append((left, middle, right))
        split(left, middle)
        split(middle, right)

    split(0, horizon)
    return order


def path_efficiency(
    states: torch.Tensor, value: ValueFunction, *, epsilon: float
) -> torch.Tensor:
    """Endpoint net distance / sum of segment costs predicted by G -> V."""
    if states.ndim != 3 or states.shape[-1] != STATE_DIM or states.shape[1] < 2:
        raise ValueError("states must be [batch, >=2 nodes, 192].")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    costs = value(states[:, :-1], states[:, 1:])
    if costs.shape != states.shape[:2][0:1] + (states.shape[1] - 1,):
        raise ValueError("G -> V must return one scalar per segment.")
    net = torch.linalg.vector_norm(states[:, -1] - states[:, 0], dim=-1)
    return net / (costs.sum(-1) + epsilon)


def dynamics_consistency(
    states: torch.Tensor, predicted_future: torch.Tensor
) -> torch.Tensor:
    """Compare nodes 1..H to one continuous F rollout anchored at node 0."""
    if predicted_future.shape != states[:, 1:].shape:
        raise ValueError("F outputs must align with every future state node.")
    return (states[:, 1:] - predicted_future.detach()).square().sum(-1).mean(-1)


def state_objective(
    states: torch.Tensor,
    value: ValueFunction,
    *,
    epsilon: float,
    predicted_future: torch.Tensor | None,
    dynamics_coefficient: float,
) -> torch.Tensor:
    if not math.isfinite(dynamics_coefficient) or dynamics_coefficient < 0:
        raise ValueError("dynamics_coefficient must be finite and nonnegative.")
    objective = -path_efficiency(states, value, epsilon=epsilon)
    if predicted_future is not None:
        objective = objective + dynamics_coefficient * dynamics_consistency(
            states, predicted_future
        )
    elif dynamics_coefficient:
        raise ValueError("A nonzero dynamics term requires an F reference.")
    return objective


def state_feedback(
    states: torch.Tensor,
    value: ValueFunction,
    *,
    epsilon: float,
    predicted_future: torch.Tensor | None = None,
    dynamics_coefficient: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach feedback only; never detach the final planner-loss computation.

    Sum (not batch mean) obtains each sample's own gradient independent of
    batch size. This also works inside a caller's inference/no_grad context.
    """
    with torch.inference_mode(False), torch.enable_grad():
        candidate = states.detach().clone().requires_grad_(True)
        reference = (
            None if predicted_future is None else predicted_future.detach().clone()
        )
        objective = state_objective(
            candidate,
            value,
            epsilon=epsilon,
            predicted_future=reference,
            dynamics_coefficient=dynamics_coefficient,
        )
        gradient = torch.autograd.grad(objective.sum(), candidate)[0]
    return objective.detach(), gradient.detach()


def generate_state_path(
    planner: StatePlanner,
    start: torch.Tensor,
    goal: torch.Tensor,
    value: ValueFunction,
    *,
    horizon: int,
    epsilon: float,
) -> torch.Tensor:
    """Recursively generate midpoint nodes; never use ground-truth midpoints."""
    if start.ndim != 2 or start.shape != goal.shape or start.shape[-1] != STATE_DIM:
        raise ValueError("start and goal must match [batch, 192].")
    nodes = {0: start, horizon: goal}
    for left, middle, right in binary_midpoint_order(horizon):
        candidate = (nodes[left] + nodes[right]) / 2
        local = torch.stack((nodes[left], candidate, nodes[right]), dim=1)
        objective, gradient = state_feedback(local, value, epsilon=epsilon)
        nodes[middle] = candidate + planner(
            nodes[left], candidate, nodes[right], objective, gradient[:, 1]
        )
    return torch.stack([nodes[i] for i in range(horizon + 1)], dim=1)


def refine_state_path(
    planner: StatePlanner,
    states: torch.Tensor,
    value: ValueFunction,
    *,
    predicted_future: torch.Tensor,
    epsilon: float,
    dynamics_coefficient: float,
) -> torch.Tensor:
    """One synchronous improvement of the SAME time-indexed state candidates."""
    objective, gradient = state_feedback(
        states,
        value,
        epsilon=epsilon,
        predicted_future=predicted_future,
        dynamics_coefficient=dynamics_coefficient,
    )
    if states.shape[1] == 2:
        return states
    interior = states[:, 1:-1]
    delta = planner(
        states[:, :-2],
        interior,
        states[:, 2:],
        objective[:, None].expand(interior.shape[:2]),
        gradient[:, 1:-1],
    )
    return torch.cat((states[:, :1], interior + delta, states[:, -1:]), dim=1)


@dataclass(frozen=True)
class PlannerLoss:
    total: torch.Tensor
    trajectory: torch.Tensor
    efficiency: torch.Tensor
    dynamics: torch.Tensor


def planner_loss(
    states: torch.Tensor,
    real_states: torch.Tensor,
    value: ValueFunction,
    *,
    predicted_future: torch.Tensor | None,
    epsilon: float,
    trajectory_coefficient: float,
    efficiency_coefficient: float,
    dynamics_coefficient: float,
    trajectory_valid: torch.Tensor | None = None,
) -> PlannerLoss:
    """Phase 1 uses trajectory only; phase 2 adds efficiency and F consistency."""
    if states.shape != real_states.shape or states.shape[1] < 3:
        raise ValueError("Planner labels must match a path with interior nodes.")
    coefficients = (
        trajectory_coefficient,
        efficiency_coefficient,
        dynamics_coefficient,
    )
    if any(not math.isfinite(x) or x < 0 for x in coefficients):
        raise ValueError("Planner loss coefficients must be finite >= 0.")
    per_path = (
        (states[:, 1:-1] - real_states[:, 1:-1].detach()).square().sum(-1).mean(-1)
    )
    if trajectory_valid is None:
        trajectory = per_path.mean()
    else:
        if (
            trajectory_valid.dtype != torch.bool
            or trajectory_valid.shape != states.shape[:1]
        ):
            raise ValueError("trajectory_valid must be one Boolean per path.")
        trajectory = (
            per_path[trajectory_valid].mean()
            if bool(trajectory_valid.any())
            else per_path.sum() * 0
        )
    efficiency = -path_efficiency(states, value, epsilon=epsilon).mean()
    if predicted_future is None:
        if dynamics_coefficient:
            raise ValueError("Dynamics loss requires an F reference.")
        dynamics = states.sum() * 0
    else:
        dynamics = dynamics_consistency(states, predicted_future).mean()
    total = (
        trajectory_coefficient * trajectory
        + efficiency_coefficient * efficiency
        + dynamics_coefficient * dynamics
    )
    return PlannerLoss(total, trajectory, efficiency, dynamics)
