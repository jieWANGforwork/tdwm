"""Standalone training integration for method C (goal-projected TD)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tdwm.methods.goal_projected_td import (
    OBJECTIVE_VERSION,
    VARIANT,
    goal_projected_td_loss,
)
from tdwm.training.frozen_actor_free_td import (
    FrozenActorFreeTDSpec,
    FrozenTDContext,
    FrozenTDLoss,
    load_frozen_actor_free_td_training_protocol,
    train_frozen_actor_free_td,
    validate_frozen_actor_free_td_training_protocol,
)

METHOD = "actor_free_td_lewm_c"


def _validate_objective(objective: dict[str, Any]) -> None:
    weight = float(objective.get("goal_projection_weight", 0.0))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            "joint_objective.goal_projection_weight must be finite and positive."
        )


def _compute_loss(context: FrozenTDContext) -> FrozenTDLoss:
    projected = goal_projected_td_loss(
        context.td_batch.prediction,
        context.td_batch.target,
        context.goals,
    )
    projection_weight = float(
        context.protocol["joint_objective"]["goal_projection_weight"]
    )
    loss = context.base_td_loss + projection_weight * projected.loss
    stage = context.stage
    return FrozenTDLoss(
        loss=loss,
        metrics={
            f"{stage}/goal_projected_td_loss": projected.loss.detach(),
            f"{stage}/goal_projection_weight": loss.new_tensor(projection_weight),
            f"{stage}/goal_prediction_score_mean": (
                projected.prediction_score.detach().mean()
            ),
            f"{stage}/goal_target_score_mean": projected.target_score.detach().mean(),
            f"{stage}/goal_score_residual_mean": projected.residual_diagnostics.mean,
            f"{stage}/goal_score_residual_std": projected.residual_diagnostics.std,
            f"{stage}/goal_score_residual_min": projected.residual_diagnostics.minimum,
            f"{stage}/goal_score_residual_max": projected.residual_diagnostics.maximum,
        },
    )


def _successor_config_fields(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_projection_weight": float(
            protocol["joint_objective"]["goal_projection_weight"]
        )
    }


SPEC = FrozenActorFreeTDSpec(
    method=METHOD,
    variant=VARIANT,
    objective_version=OBJECTIVE_VERSION,
    requires_neighbor_index=False,
    validate_objective=_validate_objective,
    compute_loss=_compute_loss,
    successor_config_fields=_successor_config_fields,
)


def load_actor_free_td_lewm_c_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_c_training_protocol(
    protocol: dict[str, Any],
) -> None:
    validate_frozen_actor_free_td_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_c(**kwargs: Any) -> dict[str, Any]:
    return train_frozen_actor_free_td(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "OBJECTIVE_VERSION",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_c_training_protocol",
    "train_actor_free_td_lewm_c",
    "validate_actor_free_td_lewm_c_training_protocol",
]
