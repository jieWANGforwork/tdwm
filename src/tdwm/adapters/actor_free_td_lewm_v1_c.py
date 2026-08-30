"""Deployment adapter for Actor-Free TD-LeWM V1 C."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.frozen_actor_free_td_v1_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    OBJECTIVE_VERSION,
    FrozenActorFreeTDV1MethodSpec,
    load_frozen_actor_free_td_v1_checkpoint,
    make_frozen_actor_free_td_v1_policy,
    require_positive_float,
)

METHOD = "actor_free_td_lewm_v1_c"
VARIANT = "c"


def _validate_method_config(config: Mapping[str, Any]) -> None:
    objective = config["joint_objective"]
    require_positive_float(
        objective, "goal_projection_weight", label="predictor_config.joint_objective"
    )


METHOD_SPEC = FrozenActorFreeTDV1MethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM V1 C",
    objective_keys=("goal_projection_weight",),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_v1_c_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_frozen_actor_free_td_v1_checkpoint(
        checkpoint_path, spec=METHOD_SPEC, map_location=map_location
    )


def make_actor_free_td_lewm_v1_c_policy(**kwargs):
    return make_frozen_actor_free_td_v1_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_v1_c_checkpoint",
    "make_actor_free_td_lewm_v1_c_policy",
]
