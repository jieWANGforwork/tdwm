"""Method D: teacher-goal-weighted TD with a frozen pretrained LeWM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tdwm.methods.frozen_td_common import (
    WeightDiagnostics,
    per_transition_vector_td_mse,
    successor_goal_score,
    temperature_softmax_weights,
    validate_aligned_goal,
    validate_vector_pair,
)

VARIANT = "goal_value_weighted_td"
OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class GoalWeightedTDOutput:
    """Vector TD loss weighted by the detached teacher goal score."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    teacher_score: torch.Tensor
    weights: torch.Tensor
    weight_diagnostics: WeightDiagnostics


def teacher_goal_weighted_td_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    *,
    temperature: float,
    weight_clip: tuple[float, float] | None = None,
) -> GoalWeightedTDOutput:
    """Compute D using globally mean-one detached teacher-score weights."""

    validate_vector_pair(prediction, target)
    validate_aligned_goal(target, goal)
    per_transition_td_loss = per_transition_vector_td_mse(prediction, target)
    teacher_score = successor_goal_score(target.detach(), goal).detach()
    weights, diagnostics = temperature_softmax_weights(
        teacher_score,
        temperature=temperature,
        weight_clip=weight_clip,
    )
    return GoalWeightedTDOutput(
        loss=(weights * per_transition_td_loss).mean(),
        per_transition_td_loss=per_transition_td_loss,
        teacher_score=teacher_score,
        weights=weights,
        weight_diagnostics=diagnostics,
    )


__all__ = [
    "GoalWeightedTDOutput",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "teacher_goal_weighted_td_loss",
]
