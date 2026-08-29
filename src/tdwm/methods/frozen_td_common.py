"""Shared alignment and weighting primitives for frozen-LeWM TD methods.

This module owns only mechanics common to C, D, F, and G1.  Each research
method keeps its objective and output type in its own module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch

from tdwm.methods.successor_geometry import (
    successor_feature_basis,
    successor_goal_cost,
)


class SuccessorModule(Protocol):
    """Structural type required by :func:`build_frozen_real_td_batch`."""

    embed_dim: int
    action_dim: int
    history_size: int
    output_dim: int

    def __call__(
        self,
        latent_history: torch.Tensor,
        previous_actions: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class DistributionDiagnostics:
    """Detached scalar summary of a tensor distribution."""

    mean: torch.Tensor
    std: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor


@dataclass(frozen=True)
class WeightDiagnostics:
    """Detached diagnostics for a temperature-softmax weighting signal."""

    signal: DistributionDiagnostics
    softmax_weights: DistributionDiagnostics
    normalized_weights: DistributionDiagnostics
    clipped_fraction: torch.Tensor
    effective_sample_size: torch.Tensor
    effective_sample_fraction: torch.Tensor


@dataclass(frozen=True)
class FrozenRealTDBatch:
    """Strictly aligned real-history successor prediction and TD target."""

    prediction: torch.Tensor
    target: torch.Tensor
    current_history: torch.Tensor
    previous_actions: torch.Tensor
    current_actions: torch.Tensor
    aligned_terminal: torch.Tensor


def validate_floating_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if tensor.ndim < 1 or tensor.shape[-1] <= 0:
        raise ValueError(f"{name} must have a non-empty final dimension.")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ValueError(f"{name} must contain only finite values.")


def validate_vector_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    validate_floating_tensor("prediction", prediction)
    validate_floating_tensor("target", target)
    if prediction.ndim < 2:
        raise ValueError(
            "prediction and target must contain a transition axis and a vector axis."
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes, found "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device.")


def _successor_dimensions(successor: SuccessorModule) -> tuple[int, int, int, int]:
    names = ("embed_dim", "action_dim", "history_size", "output_dim")
    try:
        dimensions = tuple(int(getattr(successor, name)) for name in names)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "successor modules must expose integer embed_dim, action_dim, "
            "history_size, and output_dim attributes."
        ) from error
    embed_dim, action_dim, history_size, output_dim = dimensions
    if min(embed_dim, action_dim, history_size) <= 0:
        raise ValueError("successor dimensions must be positive.")
    if output_dim != embed_dim + 2:
        raise ValueError("successor output_dim must equal embed_dim plus two.")
    return dimensions


def _latent_histories(
    latents: torch.Tensor,
    *,
    history_size: int,
    current_count: int,
    shift: int,
) -> torch.Tensor:
    return torch.stack(
        [
            latents[:, shift + index : shift + index + history_size]
            for index in range(current_count)
        ],
        dim=1,
    )


def _previous_action_histories(
    actions: torch.Tensor,
    *,
    history_size: int,
    current_count: int,
    shift: int,
) -> torch.Tensor:
    previous = history_size - 1
    if previous == 0:
        return actions.new_empty(actions.shape[0], current_count, 0, actions.shape[-1])
    return torch.stack(
        [
            actions[:, shift + index : shift + index + previous]
            for index in range(current_count)
        ],
        dim=1,
    )


def _normalize_terminals(
    terminals: torch.Tensor | None,
    *,
    real_latents: torch.Tensor,
) -> torch.Tensor:
    if terminals is None:
        return torch.zeros(
            real_latents.shape[:2],
            device=real_latents.device,
            dtype=torch.bool,
        )
    if not isinstance(terminals, torch.Tensor):
        raise TypeError("terminals must be a torch.Tensor or None.")
    if terminals.shape != real_latents.shape[:2]:
        raise ValueError("terminals must have shape (batch, time).")
    terminal_on_device = terminals.to(device=real_latents.device)
    terminal_bool = terminal_on_device.to(dtype=torch.bool)
    if bool(torch.any(terminal_on_device != terminal_bool)):
        raise ValueError("terminals must contain only binary values.")
    return terminal_bool


def build_frozen_real_td_batch(
    successor: SuccessorModule,
    target_successor: SuccessorModule,
    real_latents: torch.Tensor,
    real_ema_latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    gamma: float,
    terminals: torch.Tensor | None = None,
    first_current_index: int | None = None,
) -> FrozenRealTDBatch:
    """Build the shared real-history TD batch for frozen-LeWM objectives.

    If the first current state is ``t``, the online successor sees the real
    history ending at ``t`` and recorded action ``a_t``.  The detached target
    successor sees the history ending at ``t + 1`` and recorded next action
    ``a_(t + 1)``.  Only the explicit terminal mask disables bootstrapping.
    """

    online_dimensions = _successor_dimensions(successor)
    target_dimensions = _successor_dimensions(target_successor)
    if online_dimensions != target_dimensions:
        raise ValueError("online and target successors must share dimensions.")
    embed_dim, action_dim, history_size, output_dim = online_dimensions
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or not 0.0 <= gamma_value < 1.0:
        raise ValueError("gamma must be finite and lie in [0, 1).")

    validate_floating_tensor("real_latents", real_latents)
    validate_floating_tensor("real_ema_latents", real_ema_latents)
    validate_floating_tensor("actions", actions)
    if real_latents.ndim != 3:
        raise ValueError("real_latents must have shape (batch, time, latent_dim).")
    if real_ema_latents.shape != real_latents.shape:
        raise ValueError("real_ema_latents must match real_latents.")
    if actions.ndim != 3 or actions.shape[:2] != real_latents.shape[:2]:
        raise ValueError("actions must share the latent batch and time axes.")
    if real_latents.device != real_ema_latents.device or (
        real_latents.device != actions.device
    ):
        raise ValueError("latents and actions must be on the same device.")
    if real_latents.shape[-1] != embed_dim:
        raise ValueError("latents have the wrong embedding dimension.")
    if actions.shape[-1] != action_dim:
        raise ValueError("actions have the wrong action dimension.")
    if real_latents.shape[1] <= history_size:
        raise ValueError(
            "the clip must contain a full history and a dataset next action."
        )

    terminal_mask = _normalize_terminals(terminals, real_latents=real_latents)
    first_current = (
        history_size if first_current_index is None else int(first_current_index)
    )
    if first_current < history_size:
        raise ValueError("first_current_index must be at least history_size.")
    if first_current >= real_latents.shape[1] - 1:
        raise ValueError(
            "first_current_index must leave a next state and dataset next action."
        )

    current_count = real_latents.shape[1] - first_current - 1
    history_shift = first_current - history_size + 1
    frozen_real_latents = real_latents.detach()
    frozen_actions = actions.detach()
    current_history = _latent_histories(
        frozen_real_latents,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    previous_actions = _previous_action_histories(
        frozen_actions,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    current_actions = frozen_actions[:, first_current:-1]
    prediction = successor(current_history, previous_actions, current_actions)
    expected_output_shape = (real_latents.shape[0], current_count, output_dim)
    if prediction.shape != expected_output_shape:
        raise ValueError(
            "successor prediction must have shape "
            f"{expected_output_shape}, found {tuple(prediction.shape)}."
        )

    with torch.no_grad():
        frozen_ema_latents = real_ema_latents.detach()
        next_history = _latent_histories(
            frozen_ema_latents,
            history_size=history_size,
            current_count=current_count,
            shift=history_shift + 1,
        )
        next_previous_actions = _previous_action_histories(
            frozen_actions,
            history_size=history_size,
            current_count=current_count,
            shift=history_shift + 1,
        )
        dataset_next_action = frozen_actions[:, first_current + 1 :]
        bootstrap = target_successor(
            next_history, next_previous_actions, dataset_next_action
        )
        if bootstrap.shape != expected_output_shape:
            raise ValueError(
                "target successor prediction must have shape "
                f"{expected_output_shape}, found {tuple(bootstrap.shape)}."
            )
        next_latent = frozen_ema_latents[:, first_current + 1 :]
        aligned_terminal = terminal_mask[:, first_current:-1]
        immediate = (1.0 - gamma_value) * successor_feature_basis(next_latent)
        continuation = (~aligned_terminal).to(immediate.dtype).unsqueeze(-1)
        target = immediate + gamma_value * continuation * bootstrap.detach()

    return FrozenRealTDBatch(
        prediction=prediction,
        target=target.detach(),
        current_history=current_history,
        previous_actions=previous_actions,
        current_actions=current_actions,
        aligned_terminal=aligned_terminal,
    )


def gather_hindsight_goals(
    real_ema_latents: torch.Tensor,
    terminals: torch.Tensor,
    first_current_index: int,
    goal_offsets: torch.Tensor,
) -> torch.Tensor:
    """Gather one frozen, reachable future-goal latent per transition."""

    validate_floating_tensor("real_ema_latents", real_ema_latents)
    if real_ema_latents.ndim != 3:
        raise ValueError("real_ema_latents must have shape (batch, time, latent_dim).")
    terminal_mask = _normalize_terminals(terminals, real_latents=real_ema_latents)
    first_current = int(first_current_index)
    if first_current < 0 or first_current >= real_ema_latents.shape[1] - 1:
        raise ValueError("first_current_index must leave at least one future state.")
    current_count = real_ema_latents.shape[1] - first_current - 1
    expected_offsets_shape = (real_ema_latents.shape[0], current_count)
    if not isinstance(goal_offsets, torch.Tensor):
        raise TypeError("goal_offsets must be a torch.Tensor.")
    if goal_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "goal_offsets must have shape "
            f"{expected_offsets_shape}, found {tuple(goal_offsets.shape)}."
        )
    offsets_on_device = goal_offsets.to(device=real_ema_latents.device)
    offsets = offsets_on_device.to(dtype=torch.int64)
    if bool(torch.any(offsets_on_device != offsets)):
        raise ValueError("goal_offsets must contain integer values.")
    if bool(torch.any(offsets < 1)):
        raise ValueError("goal_offsets must be at least one.")

    aligned_terminal = terminal_mask[:, first_current:-1]
    limits: list[torch.Tensor] = []
    for index in range(current_count):
        remaining = aligned_terminal[:, index:]
        active = ~aligned_terminal[:, :index].any(dim=-1)
        has_terminal = remaining.any(dim=-1)
        first_terminal = remaining.to(dtype=torch.int64).argmax(dim=-1) + 1
        clip_limit = torch.full_like(first_terminal, current_count - index)
        reachable_limit = torch.where(has_terminal, first_terminal, clip_limit)
        limits.append(
            torch.where(active, reachable_limit, torch.zeros_like(clip_limit))
        )
    offset_limits = torch.stack(limits, dim=1)
    if bool(torch.any(offsets > offset_limits)):
        raise ValueError(
            "goal_offsets must select reachable future states without crossing "
            "a terminal."
        )

    current_indices = torch.arange(
        first_current,
        first_current + current_count,
        device=real_ema_latents.device,
        dtype=torch.int64,
    ).unsqueeze(0)
    goal_indices = current_indices + offsets
    frozen_latents = real_ema_latents.detach()
    return frozen_latents.gather(
        1,
        goal_indices.unsqueeze(-1).expand(
            real_ema_latents.shape[0], current_count, real_ema_latents.shape[-1]
        ),
    )


def validate_aligned_goal(
    successor: torch.Tensor,
    goal: torch.Tensor,
    *,
    goal_name: str = "goal",
) -> None:
    validate_floating_tensor(goal_name, goal)
    expected_shape = successor.shape[:-1] + (successor.shape[-1] - 2,)
    if successor.shape[-1] <= 2:
        raise ValueError("successor vectors must contain at least three features.")
    if goal.shape != expected_shape:
        raise ValueError(
            f"{goal_name} must have shape {expected_shape}, found {tuple(goal.shape)}."
        )
    if successor.device != goal.device:
        raise ValueError(f"successor and {goal_name} must be on the same device.")


def distribution_diagnostics(values: torch.Tensor) -> DistributionDiagnostics:
    detached = values.detach().float().reshape(-1)
    return DistributionDiagnostics(
        mean=detached.mean(),
        std=detached.std(unbiased=False),
        minimum=detached.min(),
        maximum=detached.max(),
    )


def _validate_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as error:
        raise TypeError("temperature must be a finite positive number.") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be a finite positive number.")
    return value


def _validate_weight_clip(
    weight_clip: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if weight_clip is None:
        return None
    if not isinstance(weight_clip, tuple) or len(weight_clip) != 2:
        raise TypeError("weight_clip must be a (minimum, maximum) tuple or None.")
    minimum, maximum = (float(value) for value in weight_clip)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or maximum <= 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "weight_clip bounds must be finite, non-negative, and ordered with "
            "a positive maximum."
        )
    return minimum, maximum


def temperature_softmax_weights(
    signal: torch.Tensor,
    *,
    temperature: float,
    weight_clip: tuple[float, float] | None,
) -> tuple[torch.Tensor, WeightDiagnostics]:
    """Return detached, globally mean-one weights for ``signal``."""

    temperature = _validate_temperature(temperature)
    bounds = _validate_weight_clip(weight_clip)
    detached = signal.detach().float()
    if detached.numel() == 0 or not bool(torch.isfinite(detached).all()):
        raise ValueError("weighting signal must be non-empty and finite.")

    flat = detached.reshape(-1)
    softmax_weights = torch.softmax(flat / temperature, dim=0) * flat.numel()
    clipped = softmax_weights
    if bounds is not None:
        clipped = softmax_weights.clamp(min=bounds[0], max=bounds[1])
    weights = clipped / clipped.mean()
    weights = weights.reshape(detached.shape).detach()

    flat_weights = weights.reshape(-1)
    effective_sample_size = flat_weights.sum().square() / flat_weights.square().sum()
    clipped_fraction = (clipped != softmax_weights).float().mean()
    diagnostics = WeightDiagnostics(
        signal=distribution_diagnostics(detached),
        softmax_weights=distribution_diagnostics(softmax_weights),
        normalized_weights=distribution_diagnostics(weights),
        clipped_fraction=clipped_fraction.detach(),
        effective_sample_size=effective_sample_size.detach(),
        effective_sample_fraction=(
            effective_sample_size / flat_weights.numel()
        ).detach(),
    )
    return weights, diagnostics


def per_transition_vector_td_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 vector TD MSE per transition, reducing only features."""

    validate_vector_pair(prediction, target)
    prediction_fp32 = prediction.float()
    target_fp32 = target.detach().float()
    return (prediction_fp32 - target_fp32).square().mean(dim=-1)


def successor_goal_score(successor: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return ``-successor_goal_cost(successor, goal)`` in FP32."""

    validate_floating_tensor("successor", successor)
    validate_floating_tensor("goal", goal)
    if successor.device != goal.device:
        raise ValueError("successor and goal must be on the same device.")
    if successor.shape[-1] != goal.shape[-1] + 2:
        raise ValueError(
            "successor final dimension must equal goal final dimension plus two."
        )
    try:
        torch.broadcast_shapes(successor.shape[:-1], goal.shape[:-1])
    except RuntimeError as error:
        raise ValueError(
            "successor and goal leading axes must be broadcastable."
        ) from error
    return -successor_goal_cost(successor.float(), goal.detach().float())


__all__ = [
    "DistributionDiagnostics",
    "FrozenRealTDBatch",
    "SuccessorModule",
    "WeightDiagnostics",
    "build_frozen_real_td_batch",
    "distribution_diagnostics",
    "gather_hindsight_goals",
    "per_transition_vector_td_mse",
    "successor_goal_score",
    "temperature_softmax_weights",
    "validate_aligned_goal",
    "validate_floating_tensor",
    "validate_vector_pair",
]
