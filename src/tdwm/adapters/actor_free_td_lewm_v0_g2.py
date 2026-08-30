"""Deployment adapter for Actor-Free TD-LeWM V0 G2."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.frozen_actor_free_td_v0_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    OBJECTIVE_VERSION,
    FrozenActorFreeTDV0MethodSpec,
    load_frozen_actor_free_td_v0_checkpoint,
    make_frozen_actor_free_td_v0_policy,
    require_exact_values,
    require_positive_float,
)

METHOD = "actor_free_td_lewm_v0_g2"
VARIANT = "g2"


def _validate_method_config(config: Mapping[str, Any]) -> None:
    objective = config["joint_objective"]
    require_positive_float(
        objective, "weight_temperature", label="predictor_config.joint_objective"
    )
    require_exact_values(
        objective,
        {
            "prefix_slots": 5,
            "suffix_fill": "normalized_zero_mean_action",
        },
        label="predictor_config.joint_objective",
    )


METHOD_SPEC = FrozenActorFreeTDV0MethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM V0 G2",
    objective_keys=("weight_temperature", "prefix_slots", "suffix_fill"),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_v0_g2_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_frozen_actor_free_td_v0_checkpoint(
        checkpoint_path, spec=METHOD_SPEC, map_location=map_location
    )


def make_actor_free_td_lewm_v0_g2_policy(**kwargs):
    return make_frozen_actor_free_td_v0_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_v0_g2_checkpoint",
    "make_actor_free_td_lewm_v0_g2_policy",
]
