"""Controlled Cube O50 evaluation for Actor-Free TD-LeWM V2-EMA-SG."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v2_common import ActorFreeTDV2MethodSpec
from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    FORMAL_HORIZON_BY_SCORE_MODE as V2_FORMAL_HORIZON_BY_SCORE_MODE,
)
from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    FORMAL_O50_PLANNING,
    actor_free_td_v2_output_directory_name,
    configure_actor_free_td_v2_evaluation_mode,
    evaluate_actor_free_td_v2,
    load_actor_free_td_v2_evaluation_protocol,
    validate_actor_free_td_v2_checkpoint_protocol,
    validate_actor_free_td_v2_evaluation_protocol,
    validate_v2_raw_action_compatibility,
)

EMA_SG_SCORE_MODES = frozenset(("f_only", "g_only", "f_plus_g"))
FORMAL_HORIZON_BY_SCORE_MODE = {
    mode: V2_FORMAL_HORIZON_BY_SCORE_MODE[mode] for mode in EMA_SG_SCORE_MODES
}


def validate_v2_score_mode(score_mode: str) -> str:
    """Keep the completed EMA-SG evaluator locked to its historical modes."""

    if score_mode not in EMA_SG_SCORE_MODES:
        raise ValueError(
            f"score_mode {score_mode!r} is incompatible with V2-EMA-SG; expected "
            f"one of {sorted(EMA_SG_SCORE_MODES)}."
        )
    return score_mode


def actor_free_td_v2_ema_sg_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> str:
    selected = score_mode or str(
        protocol.get("inference_objective", {}).get("score_mode", "f_plus_g")
    )
    validate_v2_score_mode(selected)
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
    inference = protocol["inference_objective"]
    validate_v2_score_mode(str(inference.get("score_mode", "f_plus_g")))


def load_actor_free_td_v2_ema_sg_evaluation_protocol(
    path: str | Path,
    *,
    spec: ActorFreeTDV2MethodSpec,
) -> dict[str, Any]:
    protocol = load_actor_free_td_v2_evaluation_protocol(path, spec=spec)
    validate_actor_free_td_v2_ema_sg_evaluation_protocol(protocol, spec=spec)
    return protocol


def configure_actor_free_td_v2_ema_sg_evaluation_mode(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> dict[str, Any]:
    selected = score_mode or str(
        protocol.get("inference_objective", {}).get("score_mode", "f_plus_g")
    )
    validate_v2_score_mode(selected)
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
    expected_checkpoint_epoch: int | None = None,
) -> None:
    validate_actor_free_td_v2_checkpoint_protocol(
        payload=payload,
        predictor_config=predictor_config,
        protocol=protocol,
        spec=spec,
        require_formal_completion=require_formal_completion,
        expected_checkpoint_epoch=expected_checkpoint_epoch,
    )


def evaluate_actor_free_td_v2_ema_sg(
    *,
    spec: ActorFreeTDV2MethodSpec,
    checkpoint_loader,
    policy_factory,
    **kwargs,
) -> dict[str, Any]:
    selected = kwargs.get("score_mode")
    if selected is not None:
        validate_v2_score_mode(str(selected))
    if kwargs.get("g_first_weight") is not None:
        raise ValueError("g_first_weight is unavailable for V2-EMA-SG evaluation.")
    protocol_path = kwargs.get("protocol_path")
    if protocol_path is not None:
        load_actor_free_td_v2_ema_sg_evaluation_protocol(protocol_path, spec=spec)
    return evaluate_actor_free_td_v2(
        spec=spec,
        checkpoint_loader=checkpoint_loader,
        policy_factory=policy_factory,
        checkpoint_protocol_validator=(
            validate_actor_free_td_v2_ema_sg_checkpoint_protocol
        ),
        **kwargs,
    )


__all__ = [
    "EMA_SG_SCORE_MODES",
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
