"""Standalone training integration for original G1 neighbor-action weighting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from tdwm.methods.state_neighbor_advantage import (
    OBJECTIVE_VERSION,
    VARIANT,
    state_neighbor_advantage_weighted_td_loss,
)
from tdwm.training.frozen_actor_free_td import (
    FrozenActorFreeTDSpec,
    FrozenTDContext,
    FrozenTDLoss,
    load_frozen_actor_free_td_training_protocol,
    train_frozen_actor_free_td,
    validate_frozen_actor_free_td_training_protocol,
)

METHOD = "actor_free_td_lewm_g1"


def _validate_objective(objective: dict[str, Any]) -> None:
    locked = {
        "candidate_source": "other_episode_frozen_latent_knn_real_action_blocks",
        "candidate_td_targets": "none",
        "weight_gradient": "stop_gradient",
    }
    for key, expected in locked.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    for key in ("neighbor_temperature", "weight_temperature"):
        temperature = float(objective.get(key, 0.0))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"joint_objective.{key} must be finite and positive.")
    if int(objective.get("neighbors_per_anchor", 0)) <= 0:
        raise ValueError("joint_objective.neighbors_per_anchor must be positive.")


def _compute_loss(context: FrozenTDContext) -> FrozenTDLoss:
    stage = context.stage
    if stage == "validation":
        return FrozenTDLoss(
            loss=context.base_td_loss,
            metrics={
                f"{stage}/neighbor_objective_available": (
                    context.base_td_loss.new_zeros(())
                )
            },
        )
    if context.neighbor_index is None:
        raise RuntimeError("G1 training is missing its state-neighbor index.")
    global_starts = context.batch.get("_tdwm_global_start")
    if not isinstance(global_starts, torch.Tensor):
        raise RuntimeError("G1 batches must carry _tdwm_global_start.")

    current_count = int(context.td_batch.prediction.shape[1])
    current_steps = torch.arange(
        context.td_batch.current_history.shape[-2],
        context.td_batch.current_history.shape[-2] + current_count,
        device=global_starts.device,
        dtype=torch.int64,
    )
    current_rows = (
        global_starts.to(dtype=torch.int64).reshape(context.batch_size, 1)
        + int(context.protocol["sequence"]["frame_skip"]) * current_steps
    )
    neighbors = context.neighbor_index.lookup(
        current_rows,
        device=context.td_batch.current_actions.device,
        dtype=context.td_batch.current_actions.dtype,
    )
    objective = context.protocol["joint_objective"]
    weighted = state_neighbor_advantage_weighted_td_loss(
        context.successor,
        context.td_batch.current_history.flatten(0, 1),
        context.td_batch.previous_actions.flatten(0, 1),
        context.td_batch.current_actions.flatten(0, 1),
        neighbors.actions.flatten(0, 1),
        context.goals.flatten(0, 1),
        neighbors.distances.flatten(0, 1),
        context.per_transition_td.flatten(),
        neighbor_temperature=float(objective["neighbor_temperature"]),
        weight_temperature=float(objective["weight_temperature"]),
    )
    return FrozenTDLoss(
        loss=weighted.loss,
        metrics={
            f"{stage}/neighbor_objective_available": context.base_td_loss.new_ones(
                ()
            ),
            f"{stage}/neighbor_advantage_mean": weighted.advantage.mean(),
            f"{stage}/neighbor_advantage_std": weighted.advantage.std(unbiased=False),
            f"{stage}/positive_action_score_mean": weighted.positive_scores.mean(),
            f"{stage}/neighbor_action_score_mean": weighted.neighbor_scores.mean(),
            f"{stage}/neighbor_weight_mean": weighted.weights.mean(),
            f"{stage}/neighbor_weight_std": weighted.weights.std(unbiased=False),
            f"{stage}/neighbor_weight_min": weighted.weights.min(),
            f"{stage}/neighbor_weight_max": weighted.weights.max(),
            f"{stage}/neighbor_distance_mean": neighbors.distances.mean(),
        },
    )


def _successor_config_fields(protocol: dict[str, Any]) -> dict[str, Any]:
    objective = protocol["joint_objective"]
    return {
        "candidate_source": objective["candidate_source"],
        "candidate_td_targets": objective["candidate_td_targets"],
        "neighbor_temperature": float(objective["neighbor_temperature"]),
        "weight_temperature": float(objective["weight_temperature"]),
        "weight_gradient": objective["weight_gradient"],
        "neighbors_per_anchor": int(objective["neighbors_per_anchor"]),
    }


SPEC = FrozenActorFreeTDSpec(
    method=METHOD,
    variant=VARIANT,
    objective_version=OBJECTIVE_VERSION,
    requires_neighbor_index=True,
    validate_objective=_validate_objective,
    compute_loss=_compute_loss,
    successor_config_fields=_successor_config_fields,
)


def load_actor_free_td_lewm_g1_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_g1_training_protocol(
    protocol: dict[str, Any],
) -> None:
    validate_frozen_actor_free_td_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_g1(**kwargs: Any) -> dict[str, Any]:
    return train_frozen_actor_free_td(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "OBJECTIVE_VERSION",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_g1_training_protocol",
    "train_actor_free_td_lewm_g1",
    "validate_actor_free_td_lewm_g1_training_protocol",
]
