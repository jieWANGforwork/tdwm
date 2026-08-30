"""Controlled Cube evaluation for Actor-Free TD-LeWM V1 G2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v1_g2 import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    METHOD,
    METHOD_SPEC,
    OBJECTIVE_VERSION,
    VARIANT,
    load_actor_free_td_lewm_v1_g2_checkpoint,
    make_actor_free_td_lewm_v1_g2_policy,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    FORMAL_O50_PLANNING,
    configure_frozen_actor_free_td_v1_evaluation_mode,
    evaluate_frozen_actor_free_td_v1,
    load_frozen_actor_free_td_v1_evaluation_protocol,
    validate_frozen_actor_free_td_v1_checkpoint_protocol,
    validate_frozen_actor_free_td_v1_evaluation_protocol,
)


def validate_actor_free_td_lewm_v1_g2_evaluation_protocol(protocol) -> None:
    validate_frozen_actor_free_td_v1_evaluation_protocol(protocol, spec=METHOD_SPEC)


def load_actor_free_td_lewm_v1_g2_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_v1_evaluation_protocol(path, spec=METHOD_SPEC)


def validate_actor_free_td_lewm_v1_g2_checkpoint_protocol(**kwargs) -> None:
    validate_frozen_actor_free_td_v1_checkpoint_protocol(spec=METHOD_SPEC, **kwargs)


def configure_actor_free_td_lewm_v1_g2_evaluation_mode(protocol, **kwargs):
    return configure_frozen_actor_free_td_v1_evaluation_mode(protocol, **kwargs)


def evaluate_actor_free_td_lewm_v1_g2(**kwargs) -> dict[str, Any]:
    return evaluate_frozen_actor_free_td_v1(
        spec=METHOD_SPEC,
        checkpoint_loader=load_actor_free_td_lewm_v1_g2_checkpoint,
        policy_factory=make_actor_free_td_lewm_v1_g2_policy,
        **kwargs,
    )


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_O50_PLANNING",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "configure_actor_free_td_lewm_v1_g2_evaluation_mode",
    "evaluate_actor_free_td_lewm_v1_g2",
    "load_actor_free_td_lewm_v1_g2_evaluation_protocol",
    "validate_actor_free_td_lewm_v1_g2_checkpoint_protocol",
    "validate_actor_free_td_lewm_v1_g2_evaluation_protocol",
]
