"""Deployment adapter for Actor-Free TD-LeWM V2 G1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.actor_free_td_lewm_v2_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    OBJECTIVE_VERSION,
    ActorFreeTDV2MethodSpec,
    load_actor_free_td_v2_checkpoint,
    make_actor_free_td_v2_policy,
    require_exact_values,
    require_positive_float,
)

METHOD = "actor_free_td_lewm_v2_g1"
VARIANT = "g1"


def _validate_method_config(config: Mapping[str, Any]) -> None:
    objective = config["joint_objective"]
    require_positive_float(
        objective, "weight_temperature", label="predictor_config.joint_objective"
    )
    require_positive_float(
        objective, "neighbor_temperature", label="predictor_config.joint_objective"
    )
    require_exact_values(
        objective,
        {
            "objective": "neighbor_action_advantage",
            "score_source": "detached_online_predictor",
            "candidate_source": (
                "other_episode_frozen_latent_knn_real_action_blocks"
            ),
            "candidate_action_processing": "online_shared_lewm_action_encoder",
            "neighbors_per_anchor": 8,
            "goal_subset_weighting": "softmax_mean_one",
        },
        label="predictor_config.joint_objective",
    )


METHOD_SPEC = ActorFreeTDV2MethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM V2 G1",
    objective_keys=(
        "weight_temperature",
        "neighbor_temperature",
        "neighbors_per_anchor",
    ),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_v2_g1_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_actor_free_td_v2_checkpoint(
        checkpoint_path, spec=METHOD_SPEC, map_location=map_location
    )


def make_actor_free_td_lewm_v2_g1_policy(**kwargs):
    return make_actor_free_td_v2_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_v2_g1_checkpoint",
    "make_actor_free_td_lewm_v2_g1_policy",
]
