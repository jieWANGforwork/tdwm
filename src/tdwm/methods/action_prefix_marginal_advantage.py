"""G3: mean prefix-marginal action advantage weighting for frozen real TD."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tdwm.methods.action_prefix_advantage_common import (
    ACTION_PREFIX_SLOTS,
    score_zero_mean_action_prefixes,
    validate_real_td_per_transition,
)
from tdwm.methods.frozen_td_common import WeightDiagnostics, temperature_softmax_weights

VARIANT = "prefix_marginal_action_advantage"
OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class ActionPrefixMarginalAdvantageTDOutput:
    """Full-action TD loss weighted by detached mean prefix marginal."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    prefix_scores: torch.Tensor
    marginal_scores: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    weight_diagnostics: WeightDiagnostics


def action_prefix_marginal_advantage_td_loss(
    successor,
    real_history: torch.Tensor,
    previous_actions: torch.Tensor,
    full_actions: torch.Tensor,
    goals: torch.Tensor,
    real_td_per_transition: torch.Tensor,
    *,
    weight_temperature: float,
) -> ActionPrefixMarginalAdvantageTDOutput:
    """Compute G3 as mean adjacent-prefix progress and weight only real TD.

    The four retained marginal diagnostics are ``q2-q1`` through ``q5-q4``.
    Their sample-level mean is algebraically ``(q5-q1)/4``.
    """

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
    marginal_scores = prefix_scores[:, 1:] - prefix_scores[:, :-1]
    advantage = marginal_scores.mean(dim=-1).detach()
    weights, diagnostics = temperature_softmax_weights(
        advantage,
        temperature=weight_temperature,
        weight_clip=None,
    )
    weights = weights.to(real_td_per_transition)
    return ActionPrefixMarginalAdvantageTDOutput(
        loss=(weights * real_td_per_transition).mean(),
        per_transition_td_loss=real_td_per_transition,
        prefix_scores=prefix_scores,
        marginal_scores=marginal_scores.detach(),
        advantage=advantage,
        weights=weights.detach(),
        weight_diagnostics=diagnostics,
    )


__all__ = [
    "ActionPrefixMarginalAdvantageTDOutput",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "action_prefix_marginal_advantage_td_loss",
]
