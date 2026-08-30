"""Standalone training integration for G2 prefix-mean action weighting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tdwm.methods.action_prefix_advantage_common import ACTION_PREFIX_SLOTS
from tdwm.methods.action_prefix_mean_advantage import (
    OBJECTIVE_VERSION,
    VARIANT,
    action_prefix_mean_advantage_td_loss,
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

METHOD = "actor_free_td_lewm_g2"

_LOCKED_OBJECTIVE = {
    "candidate_source": "same_transition_normalized_action_zero_mean_suffix_prefixes",
    "candidate_td_targets": "none",
    "prefix_slots": ACTION_PREFIX_SLOTS,
    "suffix_fill": "normalized_zero_mean_action",
    "advantage_reducer": "full_score_minus_all_prefix_mean",
    "weight_gradient": "stop_gradient",
}


def _validate_objective(objective: dict[str, Any]) -> None:
    for key, expected in _LOCKED_OBJECTIVE.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    temperature = float(objective.get("weight_temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "joint_objective.weight_temperature must be finite and positive."
        )


def _compute_loss(context: FrozenTDContext) -> FrozenTDLoss:
    objective = context.protocol["joint_objective"]
    weighted = action_prefix_mean_advantage_td_loss(
        context.successor,
        context.td_batch.current_history.flatten(0, 1),
        context.td_batch.previous_actions.flatten(0, 1),
        context.td_batch.current_actions.flatten(0, 1),
        context.goals.flatten(0, 1),
        context.per_transition_td.flatten(),
        weight_temperature=float(objective["weight_temperature"]),
    )
    stage = context.stage
    return FrozenTDLoss(
        loss=weighted.loss,
        metrics={
            f"{stage}/prefix_score_mean": weighted.prefix_scores.mean(),
            f"{stage}/prefix_score_std": weighted.prefix_scores.std(unbiased=False),
            f"{stage}/full_action_score_mean": weighted.full_score.mean(),
            f"{stage}/prefix_mean_score_mean": weighted.prefix_mean_score.mean(),
            f"{stage}/prefix_mean_advantage_mean": weighted.advantage.mean(),
            f"{stage}/prefix_mean_advantage_std": weighted.advantage.std(
                unbiased=False
            ),
            **frozen_weight_metrics(stage, weighted.weight_diagnostics),
        },
    )


def _successor_config_fields(protocol: dict[str, Any]) -> dict[str, Any]:
    objective = protocol["joint_objective"]
    return {
        **_LOCKED_OBJECTIVE,
        "weight_temperature": float(objective["weight_temperature"]),
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


def load_actor_free_td_lewm_g2_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_g2_training_protocol(
    protocol: dict[str, Any],
) -> None:
    validate_frozen_actor_free_td_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_g2(**kwargs: Any) -> dict[str, Any]:
    return train_frozen_actor_free_td(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "OBJECTIVE_VERSION",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_g2_training_protocol",
    "train_actor_free_td_lewm_g2",
    "validate_actor_free_td_lewm_g2_training_protocol",
]
