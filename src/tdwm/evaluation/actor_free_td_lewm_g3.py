"""Controlled Cube evaluation for Actor-Free TD-LeWM G3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_g3 import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    METHOD,
    METHOD_SPEC,
    OBJECTIVE_VERSION,
    VARIANT,
    load_actor_free_td_lewm_g3_checkpoint,
    make_actor_free_td_lewm_g3_policy,
)
from tdwm.evaluation.frozen_actor_free_td_common import (
    FORMAL_O50_PLANNING,
    configure_frozen_actor_free_td_evaluation_mode,
    evaluate_frozen_actor_free_td,
    load_frozen_actor_free_td_evaluation_protocol,
    validate_frozen_actor_free_td_checkpoint_protocol,
    validate_frozen_actor_free_td_evaluation_protocol,
)


def validate_actor_free_td_lewm_g3_evaluation_protocol(protocol) -> None:
    validate_frozen_actor_free_td_evaluation_protocol(protocol, spec=METHOD_SPEC)


def load_actor_free_td_lewm_g3_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_evaluation_protocol(path, spec=METHOD_SPEC)


def validate_actor_free_td_lewm_g3_checkpoint_protocol(**kwargs) -> None:
    validate_frozen_actor_free_td_checkpoint_protocol(spec=METHOD_SPEC, **kwargs)


def configure_actor_free_td_lewm_g3_evaluation_mode(protocol, **kwargs):
    return configure_frozen_actor_free_td_evaluation_mode(protocol, **kwargs)


def evaluate_actor_free_td_lewm_g3(**kwargs) -> dict[str, Any]:
    return evaluate_frozen_actor_free_td(
        spec=METHOD_SPEC,
        checkpoint_loader=load_actor_free_td_lewm_g3_checkpoint,
        policy_factory=make_actor_free_td_lewm_g3_policy,
        **kwargs,
    )


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_O50_PLANNING",
    "METHOD",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "configure_actor_free_td_lewm_g3_evaluation_mode",
    "evaluate_actor_free_td_lewm_g3",
    "load_actor_free_td_lewm_g3_evaluation_protocol",
    "validate_actor_free_td_lewm_g3_checkpoint_protocol",
    "validate_actor_free_td_lewm_g3_evaluation_protocol",
]
