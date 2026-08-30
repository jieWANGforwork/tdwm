"""Deployment adapter for Actor-Free TD-LeWM V2 C."""

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

METHOD = "actor_free_td_lewm_v2_c"
VARIANT = "c"


def _validate_method_config(config: Mapping[str, Any]) -> None:
    objective = config["joint_objective"]
    require_positive_float(
        objective,
        "goal_projection_weight",
        label="predictor_config.joint_objective",
    )
    require_exact_values(
        objective,
        {
            "objective": "goal_projected_td",
            "goal_signal": "matched_future_latent",
            "goal_projection_target": "detached_td_target_projection",
            "goal_projection_prediction_gradient": "online_predictor",
            "projection_population": "goal_derived_tasks_only",
        },
        label="predictor_config.joint_objective",
    )


METHOD_SPEC = ActorFreeTDV2MethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM V2 C",
    objective_keys=("goal_projection_weight",),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_v2_c_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_actor_free_td_v2_checkpoint(
        checkpoint_path, spec=METHOD_SPEC, map_location=map_location
    )


def make_actor_free_td_lewm_v2_c_policy(**kwargs):
    return make_actor_free_td_v2_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_v2_c_checkpoint",
    "make_actor_free_td_lewm_v2_c_policy",
]
