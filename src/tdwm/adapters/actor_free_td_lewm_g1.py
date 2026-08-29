"""Planning adapter for the independent Actor-Free TD-LeWM G1 method."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.frozen_actor_free_td_common import (
    FrozenActorFreeTDMethodSpec,
    load_frozen_actor_free_td_checkpoint,
    make_frozen_actor_free_td_policy,
    require_exact_values,
    require_positive_float,
)

METHOD = "actor_free_td_lewm_g1"
VARIANT = "neighbor_action_advantage"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1


def _validate_method_config(config: Mapping[str, Any]) -> None:
    require_exact_values(
        config,
        {
            "candidate_source": ("other_episode_frozen_latent_knn_real_action_blocks"),
            "candidate_td_targets": "none",
            "weight_gradient": "stop_gradient",
        },
        label="successor_config",
    )
    require_positive_float(config, "neighbor_temperature", label="successor_config")
    require_positive_float(config, "weight_temperature", label="successor_config")
    try:
        positive_neighbors = int(config.get("neighbors_per_anchor")) > 0
    except (TypeError, ValueError) as error:
        raise ValueError(
            "successor_config.neighbors_per_anchor must be positive."
        ) from error
    if not positive_neighbors:
        raise ValueError("successor_config.neighbors_per_anchor must be positive.")


METHOD_SPEC = FrozenActorFreeTDMethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM G1",
    objective_keys=(
        "candidate_source",
        "candidate_td_targets",
        "neighbor_temperature",
        "weight_temperature",
        "weight_gradient",
        "neighbors_per_anchor",
    ),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_g1_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_frozen_actor_free_td_checkpoint(
        checkpoint_path,
        spec=METHOD_SPEC,
        map_location=map_location,
    )


def make_actor_free_td_lewm_g1_policy(**kwargs):
    return make_frozen_actor_free_td_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_g1_checkpoint",
    "make_actor_free_td_lewm_g1_policy",
]
