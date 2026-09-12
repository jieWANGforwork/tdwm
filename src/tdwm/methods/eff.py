"""Eff: state-only future features and an efficiency-weighted path critic.

G predicts discounted *future* features, beginning at the next real state.
V(G(z, m), m) predicts undiscounted cumulative latent movement, not efficiency,
physical energy, a policy, or the Euclidean distance between the endpoints.
Only the training-data weights use measured trajectory efficiency. Numerical
hyperparameters are explicit arguments so unfinished protocol choices cannot
silently become experiment defaults.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM = 192


def _vector_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape or left.ndim < 2 or left.shape[-1] != STATE_DIM:
        raise ValueError("Eff inputs must have matching [..., 192] shapes.")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("Eff inputs must be floating point.")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("Eff inputs must share device and dtype.")


def goal_task(goal: torch.Tensor) -> torch.Tensor:
    """Existing C-family sqrt(192)-sphere task, retaining input derivatives."""
    if goal.ndim < 2 or goal.shape[-1] != STATE_DIM:
        raise ValueError("goal must have shape [..., 192].")
    return F.normalize(goal, dim=-1) * math.sqrt(STATE_DIM)


def _mlp(hidden_dim: int, output_dim: int) -> nn.Sequential:
    if isinstance(hidden_dim, bool) or hidden_dim < 1:
        raise ValueError("hidden_dim must be positive.")
    return nn.Sequential(
        nn.Linear(2 * STATE_DIM, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class EffSuccessor(nn.Module):
    """G(z, m) -> 192D; deliberately has no action input or action module."""

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = _mlp(hidden_dim, STATE_DIM)

    def forward(self, state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        _vector_pair(state, task)
        return self.network(torch.cat((state, task), dim=-1))


class EffCritic(nn.Module):
    """V(Psi, m) -> nonnegative scalar cumulative movement."""

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = _mlp(hidden_dim, 1)

    def forward(self, successor: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        _vector_pair(successor, task)
        return F.softplus(self.network(torch.cat((successor, task), dim=-1)))[..., 0]


class EffModel(nn.Module):
    """Online G/V and frozen EMA copies; LeWM is deliberately not owned here."""

    def __init__(self, *, g_hidden_dim: int, v_hidden_dim: int) -> None:
        super().__init__()
        self.g = EffSuccessor(hidden_dim=g_hidden_dim)
        self.v = EffCritic(hidden_dim=v_hidden_dim)
        self.target_g = copy.deepcopy(self.g).requires_grad_(False).eval()
        self.target_v = copy.deepcopy(self.v).requires_grad_(False).eval()

    def train(self, mode: bool = True) -> EffModel:
        super().train(mode)
        self.target_g.eval()
        self.target_v.eval()
        return self

    def online_parameters(self) -> list[nn.Parameter]:
        return list(self.g.parameters()) + list(self.v.parameters())

    def value(
        self, state: torch.Tensor, goal: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        """Compose G -> V, retaining candidate-state AND goal derivatives.

        Frozen parameters do not imply no_grad: the state planner needs input
        derivatives. Training uses the separately detached path in eff_loss.
        """
        task = goal_task(goal)
        g, v = (self.target_g, self.target_v) if target else (self.g, self.v)
        return v(g(state, task), task)

    @torch.no_grad()
    def update_targets(self, *, rate: float) -> None:
        if not math.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("EMA rate must be in (0, 1].")
        for online, target in ((self.g, self.target_g), (self.v, self.target_v)):
            for p, q in zip(online.parameters(), target.parameters(), strict=True):
                q.lerp_(p, rate)
            for p, q in zip(online.buffers(), target.buffers(), strict=True):
                q.copy_(p)


@torch.no_grad()
def successor_target(
    next_state: torch.Tensor,
    next_successor: torch.Tensor,
    terminal_after_transition: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """z_(t+1) + gamma * (1-d_t) * target_G(z_(t+1), m)."""
    _vector_pair(next_state, next_successor)
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    if terminal_after_transition.shape != next_state.shape[:-1]:
        raise ValueError("terminal mask must match the state batch axes.")
    if terminal_after_transition.dtype != torch.bool:
        raise TypeError("terminal mask must be Boolean, not a truncation flag.")
    tail = torch.where(terminal_after_transition[..., None], 0, next_successor)
    return next_state + gamma * tail


@torch.no_grad()
def movement_target(
    observed_cost: torch.Tensor,
    target_remaining: torch.Tensor,
    goal_reached: torch.Tensor,
    continuation_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact cost if goal reached, otherwise prefix + EMA tail (no discount).

    An actual failed terminal with no continuation is invalid, not zero-cost
    success. A finite loader window is not by itself a terminal. The caller
    must derive both flags from actual episode/goal metadata.
    """
    if not (
        observed_cost.shape
        == target_remaining.shape
        == goal_reached.shape
        == continuation_valid.shape
    ):
        raise ValueError("movement target fields must have identical shapes.")
    if goal_reached.dtype != torch.bool or continuation_valid.dtype != torch.bool:
        raise TypeError("goal and continuation masks must be Boolean.")
    valid = goal_reached | continuation_valid
    target = observed_cost + torch.where(goal_reached, 0, target_remaining)
    target = torch.where(valid, target, 0)
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("Valid cumulative movement targets must be finite >= 0.")
    return target, valid


@torch.no_grad()
def trajectory_efficiency(
    states: torch.Tensor, lengths: torch.Tensor, *, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Net endpoint distance / sum of consecutive distances on real paths.

    lengths counts transitions; states may have right-padding. A stationary
    zero-length-displacement path is not assigned a fabricated efficiency.
    The returned valid mask makes that distinction explicit.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if states.ndim != 3 or states.shape[-1] != STATE_DIM:
        raise ValueError("states must be [batch, time, 192].")
    if lengths.shape != states.shape[:1] or lengths.dtype != torch.long:
        raise ValueError("lengths must be one int64 transition count per path.")
    if bool(((lengths < 0) | (lengths >= states.shape[1])).any()):
        raise ValueError("Path endpoint lies outside the supplied states.")
    end = states[torch.arange(len(states), device=states.device), lengths]
    net = torch.linalg.vector_norm(end - states[:, 0], dim=-1)
    changes = torch.linalg.vector_norm(states[:, 1:] - states[:, :-1], dim=-1)
    mask = torch.arange(changes.shape[1], device=states.device)[None] < lengths[:, None]
    cost = torch.where(mask, changes, 0).sum(-1)
    valid = (lengths > 0) & (cost > epsilon) & (net > epsilon)
    eta = torch.where(valid, net / (cost + epsilon), 0)
    return eta, valid


@torch.no_grad()
def efficiency_weights(
    efficiency: torch.Tensor,
    known: torch.Tensor,
    groups: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Positive, detached, mean-one weights within comparable path groups.

    Unknown/unreachable/degenerate full paths keep weight 1. Normalization
    avoids changing a group's total mass solely because its typical geometry
    is easier. The log-domain form is the normalized exp(beta*(eta-baseline));
    the shared group baseline cancels, so no learned advantage network exists.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if efficiency.ndim != 1 or known.shape != efficiency.shape:
        raise ValueError("One efficiency and known flag are required per path.")
    if not efficiency.is_floating_point():
        raise TypeError("Efficiency values must be floating point.")
    if groups.shape != efficiency.shape or groups.dtype != torch.long:
        raise ValueError("groups must be int64 with one entry per path.")
    if known.dtype != torch.bool:
        raise TypeError("known must be Boolean.")
    measured = efficiency[known]
    if not bool(torch.isfinite(measured).all()) or bool(
        ((measured < 0) | (measured > 1 + 1e-5)).any()
    ):
        raise ValueError("Known path efficiency must be in [0, 1].")
    weights = torch.ones_like(efficiency)
    for group in torch.unique(groups[known]):
        selected = known & (groups == group)
        measured_group = efficiency[selected].double()
        logits = (measured_group - measured_group.max()) * beta
        # A numerical floor preserves positivity even at extreme beta; this
        # is machine precision protection, not a tunable advantage clip.
        probabilities = torch.softmax(logits, dim=0).clamp_min(
            torch.finfo(efficiency.dtype).tiny
        )
        weights[selected] = (probabilities / probabilities.sum() * selected.sum()).to(
            efficiency
        )
    return weights


def weighted_path_loss(
    losses: torch.Tensor,
    valid: torch.Tensor,
    path_ids: torch.Tensor,
    path_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean within a path first, then weighted mean across nonempty paths."""
    if losses.ndim != 1 or valid.shape != losses.shape:
        raise ValueError("losses/valid must be matching sample vectors.")
    if path_ids.shape != losses.shape or path_ids.dtype != torch.long:
        raise ValueError("path_ids must be one int64 ID per sample.")
    if valid.dtype != torch.bool or path_weights.ndim != 1:
        raise ValueError("Expected Boolean validity and vector path weights.")
    if bool(((path_ids < 0) | (path_ids >= len(path_weights))).any()):
        raise ValueError("path_ids index outside path_weights.")
    weights = path_weights.detach().to(losses)
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("path weights must be finite and strictly positive.")
    totals = torch.zeros_like(weights).scatter_add(
        0, path_ids, torch.where(valid, losses, 0)
    )
    counts = torch.zeros_like(weights).scatter_add(0, path_ids, valid.to(losses))
    present = counts > 0
    if not bool(present.any()):
        return torch.where(valid, losses, 0).sum() * 0
    means = totals / counts.clamp_min(1)
    return (weights[present] * means[present]).sum() / weights[present].sum()


@dataclass(frozen=True)
class EffLoss:
    total: torch.Tensor
    vector: torch.Tensor
    critic: torch.Tensor
    vector_target: torch.Tensor
    critic_target: torch.Tensor
    critic_valid: torch.Tensor


def eff_loss(
    model: EffModel,
    *,
    state: torch.Tensor,
    next_state: torch.Tensor,
    bootstrap_state: torch.Tensor,
    goal: torch.Tensor,
    terminal_after_transition: torch.Tensor,
    observed_cost: torch.Tensor,
    goal_reached: torch.Tensor,
    continuation_valid: torch.Tensor,
    vector_valid: torch.Tensor,
    path_ids: torch.Tensor,
    path_weights: torch.Tensor,
    gamma_g: float,
    critic_coefficient: float,
) -> EffLoss:
    """Train G with vector TD and V with detached-G movement regression."""
    if not math.isfinite(critic_coefficient) or critic_coefficient < 0:
        raise ValueError("critic_coefficient must be finite and nonnegative.")
    state, next_state = state.detach(), next_state.detach()
    goal, bootstrap_state = goal.detach(), bootstrap_state.detach()
    task = goal_task(goal)
    prediction = model.g(state, task)
    value = model.v(prediction.detach(), task)
    with torch.no_grad():
        yg = successor_target(
            next_state,
            model.target_g(next_state, task),
            terminal_after_transition,
            gamma=gamma_g,
        )
        remaining = model.target_v(model.target_g(bootstrap_state, task), task)
        yv, v_valid = movement_target(
            observed_cost.detach(), remaining, goal_reached, continuation_valid
        )
    lg = weighted_path_loss(
        (prediction - yg).square().mean(-1), vector_valid, path_ids, path_weights
    )
    lv = weighted_path_loss((value - yv).square(), v_valid, path_ids, path_weights)
    return EffLoss(lg + critic_coefficient * lv, lg, lv, yg, yv, v_valid)
