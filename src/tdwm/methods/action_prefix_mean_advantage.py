"""G2: prefix-mean action advantage weighting for frozen real TD."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tdwm.methods.action_prefix_advantage_common import (
    ACTION_PREFIX_SLOTS,
    score_zero_mean_action_prefixes,
    validate_real_td_per_transition,
)
from tdwm.methods.frozen_td_common import WeightDiagnostics, temperature_softmax_weights

VARIANT = "prefix_mean_action_advantage"
OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class ActionPrefixMeanAdvantageTDOutput:
    """Full-action TD loss weighted by detached prefix-mean advantage."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    prefix_scores: torch.Tensor
    full_score: torch.Tensor
    prefix_mean_score: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    weight_diagnostics: WeightDiagnostics


def action_prefix_mean_advantage_td_loss(
    successor,
    real_history: torch.Tensor,
    previous_actions: torch.Tensor,
    full_actions: torch.Tensor,
    goals: torch.Tensor,
    real_td_per_transition: torch.Tensor,
    *,
    weight_temperature: float,
) -> ActionPrefixMeanAdvantageTDOutput:
    """Compute G2 and use it only to weight the real full-action TD loss."""

    if full_actions.ndim != 2:
        raise ValueError("full_actions must have shape (N, action_dim).")
    count = int(full_actions.shape[0])
    validate_real_td_per_transition(
        real_td_per_transition,
        count=count,
        device=full_actions.device,
    )
    prefix_scores = score_zero_mean_action_prefixes(
        successor,
        real_history,
        previous_actions,
        full_actions,
        goals,
        prefix_slots=ACTION_PREFIX_SLOTS,
    )
    full_score = prefix_scores[:, -1]
    prefix_mean_score = prefix_scores.mean(dim=-1)
    advantage = (full_score - prefix_mean_score).detach()
    weights, diagnostics = temperature_softmax_weights(
        advantage,
        temperature=weight_temperature,
        weight_clip=None,
    )
    weights = weights.to(real_td_per_transition)
    return ActionPrefixMeanAdvantageTDOutput(
        loss=(weights * real_td_per_transition).mean(),
        per_transition_td_loss=real_td_per_transition,
        prefix_scores=prefix_scores,
        full_score=full_score.detach(),
        prefix_mean_score=prefix_mean_score.detach(),
        advantage=advantage,
        weights=weights.detach(),
        weight_diagnostics=diagnostics,
    )


__all__ = [
    "ActionPrefixMeanAdvantageTDOutput",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "action_prefix_mean_advantage_td_loss",
]
