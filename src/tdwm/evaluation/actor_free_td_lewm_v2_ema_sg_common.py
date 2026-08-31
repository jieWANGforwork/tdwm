"""Controlled Cube O50 evaluation for Actor-Free TD-LeWM V2-EMA-SG."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v2_common import ActorFreeTDV2MethodSpec
from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    FORMAL_HORIZON_BY_SCORE_MODE,
    FORMAL_O50_PLANNING,
    actor_free_td_v2_output_directory_name,
    configure_actor_free_td_v2_evaluation_mode,
    evaluate_actor_free_td_v2,
    load_actor_free_td_v2_evaluation_protocol,
    validate_actor_free_td_v2_checkpoint_protocol,
    validate_actor_free_td_v2_evaluation_protocol,
    validate_v2_raw_action_compatibility,
    validate_v2_score_mode,
)


def actor_free_td_v2_ema_sg_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> str:
    return actor_free_td_v2_output_directory_name(
        protocol,
        smoke=smoke,
        pilot=pilot,
        score_mode=score_mode,
    )


def validate_actor_free_td_v2_ema_sg_evaluation_protocol(
    protocol: Mapping[str, Any],
    *,
    spec: ActorFreeTDV2MethodSpec,
) -> None:
    validate_actor_free_td_v2_evaluation_protocol(protocol, spec=spec)


def load_actor_free_td_v2_ema_sg_evaluation_protocol(
    path: str | Path,
    *,
    spec: ActorFreeTDV2MethodSpec,
) -> dict[str, Any]:
    return load_actor_free_td_v2_evaluation_protocol(path, spec=spec)


def configure_actor_free_td_v2_ema_sg_evaluation_mode(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> dict[str, Any]:
    return configure_actor_free_td_v2_evaluation_mode(
        protocol,
        smoke=smoke,
        pilot=pilot,
        score_mode=score_mode,
    )


def validate_actor_free_td_v2_ema_sg_checkpoint_protocol(
    *,
    payload: Mapping[str, Any],
    predictor_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    spec: ActorFreeTDV2MethodSpec,
    require_formal_completion: bool = True,
) -> None:
    validate_actor_free_td_v2_checkpoint_protocol(
        payload=payload,
        predictor_config=predictor_config,
        protocol=protocol,
        spec=spec,
        require_formal_completion=require_formal_completion,
    )


def evaluate_actor_free_td_v2_ema_sg(
    *,
    spec: ActorFreeTDV2MethodSpec,
    checkpoint_loader,
    policy_factory,
    **kwargs,
) -> dict[str, Any]:
    return evaluate_actor_free_td_v2(
        spec=spec,
        checkpoint_loader=checkpoint_loader,
        policy_factory=policy_factory,
        **kwargs,
    )


__all__ = [
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "FORMAL_O50_PLANNING",
    "actor_free_td_v2_ema_sg_output_directory_name",
    "configure_actor_free_td_v2_ema_sg_evaluation_mode",
    "evaluate_actor_free_td_v2_ema_sg",
    "load_actor_free_td_v2_ema_sg_evaluation_protocol",
    "validate_actor_free_td_v2_ema_sg_checkpoint_protocol",
    "validate_actor_free_td_v2_ema_sg_evaluation_protocol",
    "validate_v2_raw_action_compatibility",
    "validate_v2_score_mode",
]
