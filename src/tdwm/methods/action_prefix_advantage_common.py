"""Shared zero-mean action-prefix scoring for the G2 and G3 objectives.

The frozen Cube action cache stores one normalized 25-dimensional coarse
action as five consecutive five-dimensional primitive-action slots.  A short
prefix is represented without changing the successor architecture: retained
slots keep their recorded normalized actions and every later slot is filled
with zero, i.e. the mean action in normalized space.

Prefix candidates are only a detached scoring signal.  They never receive a
Bellman target and never contribute a direct gradient to the successor.
"""

from __future__ import annotations

import torch

from tdwm.methods.frozen_td_common import successor_goal_score, validate_floating_tensor

ACTION_PREFIX_SLOTS = 5
ACTION_PREFIX_DIM = 25


def _validate_action_layout(action_dim: int, prefix_slots: int) -> None:
    if int(prefix_slots) != ACTION_PREFIX_SLOTS:
        raise ValueError("G2/G3 require exactly five action-prefix slots.")
    if int(action_dim) != ACTION_PREFIX_DIM:
        raise ValueError("G2/G3 require a 25D Cube action block (5 slots x 5D).")


def build_zero_mean_action_prefixes(
    full_actions: torch.Tensor,
    *,
    prefix_slots: int = ACTION_PREFIX_SLOTS,
) -> torch.Tensor:
    """Return all fixed-shape prefixes with zero-filled normalized suffixes.

    For ``[a1, a2, a3, a4, a5]`` the candidate axis contains
    ``[a1, 0, 0, 0, 0]`` through ``[a1, a2, a3, a4, a5]``.  The output shape is
    ``(*leading, prefix_slots, action_dim)``.
    """

    validate_floating_tensor("full_actions", full_actions)
    slots = int(prefix_slots)
    action_dim = int(full_actions.shape[-1])
    _validate_action_layout(action_dim, slots)

    primitive_dim = action_dim // slots
    action_slots = full_actions.reshape(
        *full_actions.shape[:-1], slots, primitive_dim
    )
    mask = torch.ones(
        (slots, slots),
        device=full_actions.device,
        dtype=full_actions.dtype,
    ).tril()
    mask = mask.reshape((1,) * (full_actions.ndim - 1) + (slots, slots, 1))
    prefixes = action_slots.unsqueeze(-3) * mask
    return prefixes.flatten(start_dim=-2)


def _validate_prefix_score_inputs(
    real_history: torch.Tensor,
    previous_actions: torch.Tensor,
    full_actions: torch.Tensor,
    goals: torch.Tensor,
    *,
    prefix_slots: int,
) -> tuple[int, int, int]:
    validate_floating_tensor("real_history", real_history)
    validate_floating_tensor("previous_actions", previous_actions)
    validate_floating_tensor("full_actions", full_actions)
    validate_floating_tensor("goals", goals)
    if real_history.ndim != 3:
        raise ValueError("real_history must have shape (N, history, latent_dim).")
    count, history_size, latent_dim = real_history.shape
    if full_actions.ndim != 2 or full_actions.shape[0] != count:
        raise ValueError("full_actions must have shape (N, action_dim).")
    action_dim = int(full_actions.shape[-1])
    _validate_action_layout(action_dim, prefix_slots)
    expected_previous = (count, history_size - 1, action_dim)
    if previous_actions.shape != expected_previous:
        raise ValueError(
            "previous_actions must have shape "
            f"{expected_previous}, found {tuple(previous_actions.shape)}."
        )
    if goals.shape != (count, latent_dim):
        raise ValueError(
            f"goals must have shape {(count, latent_dim)}, "
            f"found {tuple(goals.shape)}."
        )
    if any(
        tensor.device != real_history.device
        for tensor in (previous_actions, full_actions, goals)
    ):
        raise ValueError("Action-prefix scoring inputs must share one device.")
    return count, action_dim, latent_dim


def validate_real_td_per_transition(
    real_td_per_transition: torch.Tensor,
    *,
    count: int,
    device: torch.device,
) -> None:
    """Validate the full-action scalar TD losses that prefix scores reweight."""

    validate_floating_tensor("real_td_per_transition", real_td_per_transition)
    if real_td_per_transition.shape != (count,):
        raise ValueError(
            "real_td_per_transition must contain one scalar loss per transition."
        )
    if real_td_per_transition.device != device:
        raise ValueError("real_td_per_transition must share the scoring device.")
    if bool(torch.any(real_td_per_transition.detach() < 0)):
        raise ValueError("real_td_per_transition cannot be negative.")


def score_zero_mean_action_prefixes(
    successor,
    real_history: torch.Tensor,
    previous_actions: torch.Tensor,
    full_actions: torch.Tensor,
    goals: torch.Tensor,
    *,
    prefix_slots: int = ACTION_PREFIX_SLOTS,
) -> torch.Tensor:
    """Score all prefixes with the online successor under a no-gradient boundary."""

    count, _, latent_dim = _validate_prefix_score_inputs(
        real_history,
        previous_actions,
        full_actions,
        goals,
        prefix_slots=prefix_slots,
    )
    prefixes = build_zero_mean_action_prefixes(
        full_actions.detach(),
        prefix_slots=prefix_slots,
    )
    candidate_history = real_history.unsqueeze(1).expand(-1, prefix_slots, -1, -1)
    candidate_previous_actions = previous_actions.unsqueeze(1).expand(
        -1, prefix_slots, -1, -1
    )
    with torch.no_grad():
        candidate_successor = successor(
            candidate_history,
            candidate_previous_actions,
            prefixes,
        )
        expected_shape = (count, int(prefix_slots), latent_dim + 2)
        if candidate_successor.shape != expected_shape:
            raise ValueError(
                "successor output must have shape "
                f"{expected_shape}, found {tuple(candidate_successor.shape)}."
            )
        scores = successor_goal_score(
            candidate_successor.float(),
            goals.detach().float().unsqueeze(1),
        )
    return scores.detach()


__all__ = [
    "ACTION_PREFIX_DIM",
    "ACTION_PREFIX_SLOTS",
    "build_zero_mean_action_prefixes",
    "score_zero_mean_action_prefixes",
    "validate_real_td_per_transition",
]
