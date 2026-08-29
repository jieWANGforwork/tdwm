"""Method F: same-future/different-goal advantage-weighted TD."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tdwm.methods.frozen_td_common import (
    WeightDiagnostics,
    per_transition_vector_td_mse,
    successor_goal_score,
    temperature_softmax_weights,
    validate_floating_tensor,
    validate_vector_pair,
)

VARIANT = "same_future_goal_advantage"
OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class SameFutureGoalAdvantageTDOutput:
    """TD loss weighted by same-future/different-goal advantage."""

    loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    score_matrix: torch.Tensor
    positive_score: torch.Tensor
    all_goal_mean_score: torch.Tensor
    advantage: torch.Tensor
    weights: torch.Tensor
    weight_diagnostics: WeightDiagnostics


def same_future_goal_advantage_td_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    goals: torch.Tensor,
    *,
    temperature: float,
    weight_clip: tuple[float, float] | None = None,
) -> SameFutureGoalAdvantageTDOutput:
    """Compute F by scoring every teacher future against every batch goal."""

    validate_vector_pair(prediction, target)
    if prediction.ndim != 2:
        raise ValueError("prediction and target must have shape (batch, features).")
    if prediction.shape[0] < 2:
        raise ValueError("same-future goal advantage requires batch size at least 2.")
    validate_floating_tensor("goals", goals)
    expected_goal_shape = (target.shape[0], target.shape[-1] - 2)
    if goals.shape != expected_goal_shape:
        raise ValueError(
            f"goals must have shape {expected_goal_shape}, found {tuple(goals.shape)}."
        )
    if target.device != goals.device:
        raise ValueError("target and goals must be on the same device.")

    per_transition_td_loss = per_transition_vector_td_mse(prediction, target)
    score_matrix = successor_goal_score(
        target.detach()[:, None, :], goals.detach()[None, :, :]
    ).detach()
    positive_score = score_matrix.diagonal()
    all_goal_mean_score = score_matrix.mean(dim=1)
    advantage = (positive_score - all_goal_mean_score).detach()
    weights, diagnostics = temperature_softmax_weights(
        advantage,
        temperature=temperature,
        weight_clip=weight_clip,
    )
    return SameFutureGoalAdvantageTDOutput(
        loss=(weights * per_transition_td_loss).mean(),
        per_transition_td_loss=per_transition_td_loss,
        score_matrix=score_matrix,
        positive_score=positive_score,
        all_goal_mean_score=all_goal_mean_score,
        advantage=advantage,
        weights=weights,
        weight_diagnostics=diagnostics,
    )


__all__ = [
    "OBJECTIVE_VERSION",
    "SameFutureGoalAdvantageTDOutput",
    "VARIANT",
    "same_future_goal_advantage_td_loss",
]
