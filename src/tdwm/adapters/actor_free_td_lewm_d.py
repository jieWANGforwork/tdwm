"""Planning adapter for the independent Actor-Free TD-LeWM D method."""

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
    validate_weight_clip,
)

METHOD = "actor_free_td_lewm_d"
VARIANT = "goal_value_weighted_td"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1


def _validate_method_config(config: Mapping[str, Any]) -> None:
    require_exact_values(
        config,
        {"weight_gradient": "stop_gradient"},
        label="successor_config",
    )
    require_positive_float(config, "weight_temperature", label="successor_config")
    validate_weight_clip(config, label="successor_config")


METHOD_SPEC = FrozenActorFreeTDMethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM D",
    objective_keys=("weight_temperature", "weight_clip", "weight_gradient"),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_d_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_frozen_actor_free_td_checkpoint(
        checkpoint_path,
        spec=METHOD_SPEC,
        map_location=map_location,
    )


def make_actor_free_td_lewm_d_policy(**kwargs):
    return make_frozen_actor_free_td_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_d_checkpoint",
    "make_actor_free_td_lewm_d_policy",
]
