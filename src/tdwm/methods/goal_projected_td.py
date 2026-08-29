"""Method C: goal-projected TD with a frozen pretrained LeWM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tdwm.methods.frozen_td_common import (
    DistributionDiagnostics,
    distribution_diagnostics,
    successor_goal_score,
    validate_aligned_goal,
    validate_vector_pair,
)

VARIANT = "goal_projected_td"
OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class GoalProjectedTDOutput:
    """TD error projected onto the frozen goal-cost direction."""

    loss: torch.Tensor
    per_transition_loss: torch.Tensor
    prediction_score: torch.Tensor
    target_score: torch.Tensor
    score_residual: torch.Tensor
    residual_diagnostics: DistributionDiagnostics


def goal_projected_td_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
) -> GoalProjectedTDOutput:
    """Compute C, the squared vector TD error projected onto each goal."""

    validate_vector_pair(prediction, target)
    validate_aligned_goal(prediction, goal)
    prediction_score = successor_goal_score(prediction, goal)
    target_score = successor_goal_score(target.detach(), goal)
    score_residual = prediction_score - target_score
    per_transition_loss = score_residual.square()
    return GoalProjectedTDOutput(
        loss=per_transition_loss.mean(),
        per_transition_loss=per_transition_loss,
        prediction_score=prediction_score,
        target_score=target_score,
        score_residual=score_residual,
        residual_diagnostics=distribution_diagnostics(score_residual),
    )


__all__ = [
    "GoalProjectedTDOutput",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "goal_projected_td_loss",
]
