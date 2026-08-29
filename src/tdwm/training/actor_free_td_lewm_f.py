"""Standalone training integration for method F (goal-advantage weighting)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tdwm.methods.same_future_goal_advantage import (
    OBJECTIVE_VERSION,
    VARIANT,
    same_future_goal_advantage_td_loss,
)
from tdwm.training.frozen_actor_free_td import (
    FrozenActorFreeTDSpec,
    FrozenTDContext,
    FrozenTDLoss,
    frozen_weight_metrics,
    load_frozen_actor_free_td_training_protocol,
    train_frozen_actor_free_td,
    validate_frozen_actor_free_td_training_protocol,
)

METHOD = "actor_free_td_lewm_f"


def _weight_clip(objective: dict[str, Any]) -> tuple[float, float] | None:
    raw = objective.get("weight_clip")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(
            "joint_objective.weight_clip must be [minimum, maximum] or null."
        )
    minimum, maximum = (float(value) for value in raw)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or maximum <= 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "joint_objective.weight_clip bounds must be finite, non-negative, "
            "and ordered."
        )
    return minimum, maximum


def _validate_objective(objective: dict[str, Any]) -> None:
    if objective.get("weight_gradient") != "stop_gradient":
        raise ValueError("joint_objective.weight_gradient must be 'stop_gradient'.")
    temperature = float(objective.get("weight_temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "joint_objective.weight_temperature must be finite and positive."
        )
    _weight_clip(objective)


def _compute_loss(context: FrozenTDContext) -> FrozenTDLoss:
    objective = context.protocol["joint_objective"]
    weighted = same_future_goal_advantage_td_loss(
        context.td_batch.prediction.flatten(0, -2),
        context.td_batch.target.flatten(0, -2),
        context.goals.flatten(0, -2),
        temperature=float(objective["weight_temperature"]),
        weight_clip=_weight_clip(objective),
    )
    stage = context.stage
    return FrozenTDLoss(
        loss=weighted.loss,
        metrics={
            f"{stage}/positive_goal_score_mean": weighted.positive_score.mean(),
            f"{stage}/all_goal_baseline_score_mean": (
                weighted.all_goal_mean_score.mean()
            ),
            f"{stage}/goal_advantage_mean": weighted.advantage.mean(),
            f"{stage}/goal_advantage_std": weighted.advantage.std(unbiased=False),
            **frozen_weight_metrics(stage, weighted.weight_diagnostics),
        },
    )


def _successor_config_fields(protocol: dict[str, Any]) -> dict[str, Any]:
    objective = protocol["joint_objective"]
    return {
        "weight_temperature": float(objective["weight_temperature"]),
        "weight_clip": objective.get("weight_clip"),
        "weight_gradient": objective["weight_gradient"],
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


def load_actor_free_td_lewm_f_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_f_training_protocol(
    protocol: dict[str, Any],
) -> None:
    validate_frozen_actor_free_td_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_f(**kwargs: Any) -> dict[str, Any]:
    return train_frozen_actor_free_td(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "OBJECTIVE_VERSION",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_f_training_protocol",
    "train_actor_free_td_lewm_f",
    "validate_actor_free_td_lewm_f_training_protocol",
]
