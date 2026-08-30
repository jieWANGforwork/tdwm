"""Controlled Cube evaluation for Actor-Free TD-LeWM V0 F."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v0_f import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    METHOD,
    METHOD_SPEC,
    OBJECTIVE_VERSION,
    VARIANT,
    load_actor_free_td_lewm_v0_f_checkpoint,
    make_actor_free_td_lewm_v0_f_policy,
)
from tdwm.evaluation.frozen_actor_free_td_v0_common import (
    FORMAL_O50_PLANNING,
    configure_frozen_actor_free_td_v0_evaluation_mode,
    evaluate_frozen_actor_free_td_v0,
    load_frozen_actor_free_td_v0_evaluation_protocol,
    validate_frozen_actor_free_td_v0_checkpoint_protocol,
    validate_frozen_actor_free_td_v0_evaluation_protocol,
)


def validate_actor_free_td_lewm_v0_f_evaluation_protocol(protocol) -> None:
    validate_frozen_actor_free_td_v0_evaluation_protocol(protocol, spec=METHOD_SPEC)


def load_actor_free_td_lewm_v0_f_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_frozen_actor_free_td_v0_evaluation_protocol(path, spec=METHOD_SPEC)


def validate_actor_free_td_lewm_v0_f_checkpoint_protocol(**kwargs) -> None:
    validate_frozen_actor_free_td_v0_checkpoint_protocol(spec=METHOD_SPEC, **kwargs)


def configure_actor_free_td_lewm_v0_f_evaluation_mode(protocol, **kwargs):
    return configure_frozen_actor_free_td_v0_evaluation_mode(protocol, **kwargs)


def evaluate_actor_free_td_lewm_v0_f(**kwargs) -> dict[str, Any]:
    return evaluate_frozen_actor_free_td_v0(
        spec=METHOD_SPEC,
        checkpoint_loader=load_actor_free_td_lewm_v0_f_checkpoint,
        policy_factory=make_actor_free_td_lewm_v0_f_policy,
        **kwargs,
    )


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_O50_PLANNING",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "configure_actor_free_td_lewm_v0_f_evaluation_mode",
    "evaluate_actor_free_td_lewm_v0_f",
    "load_actor_free_td_lewm_v0_f_evaluation_protocol",
    "validate_actor_free_td_lewm_v0_f_checkpoint_protocol",
    "validate_actor_free_td_lewm_v0_f_evaluation_protocol",
]
