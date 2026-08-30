"""Planning adapter for the independent Actor-Free TD-LeWM G3 method."""

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

METHOD = "actor_free_td_lewm_g3"
VARIANT = "prefix_marginal_action_advantage"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1


def _validate_method_config(config: Mapping[str, Any]) -> None:
    require_exact_values(
        config,
        {
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "candidate_td_targets": "none",
            "prefix_slots": 5,
            "suffix_fill": "normalized_zero_mean_action",
            "advantage_reducer": "mean_adjacent_prefix_score_deltas",
            "weight_gradient": "stop_gradient",
        },
        label="successor_config",
    )
    require_positive_float(config, "weight_temperature", label="successor_config")


METHOD_SPEC = FrozenActorFreeTDMethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM G3",
    objective_keys=(
        "candidate_source",
        "candidate_td_targets",
        "prefix_slots",
        "suffix_fill",
        "advantage_reducer",
        "weight_temperature",
        "weight_gradient",
    ),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_g3_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_frozen_actor_free_td_checkpoint(
        checkpoint_path,
        spec=METHOD_SPEC,
        map_location=map_location,
    )


def make_actor_free_td_lewm_g3_policy(**kwargs):
    return make_frozen_actor_free_td_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_g3_checkpoint",
    "make_actor_free_td_lewm_g3_policy",
]
