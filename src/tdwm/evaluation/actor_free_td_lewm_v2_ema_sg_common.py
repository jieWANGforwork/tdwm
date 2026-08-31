"""Controlled Cube O50 evaluation for Actor-Free TD-LeWM V2-EMA-SG."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v2_common import ActorFreeTDV2MethodSpec
from tdwm.adapters.actor_free_td_lewm_v2_ema_sg_common import (
    IMPLEMENTATION_VERSION,
    INITIALIZATION,
    METHOD_FAMILY,
    TRAINING_COMMIT,
    TRAINING_STAGE,
    VERSION_DISPLAY_NAME,
    VERSION_KEY,
)
from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    FIRST_ACTION_SCORE_MODE,
    FORMAL_O50_PLANNING,
    ROLLOUT_MEAN_SCORE_MODE,
    actor_free_td_v2_output_directory_name,
    configure_actor_free_td_v2_evaluation_mode,
    evaluate_actor_free_td_v2,
    load_actor_free_td_v2_evaluation_protocol,
    validate_actor_free_td_v2_checkpoint_protocol,
    validate_actor_free_td_v2_evaluation_protocol,
    validate_v2_raw_action_compatibility,
)
from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    FORMAL_HORIZON_BY_SCORE_MODE as V2_FORMAL_HORIZON_BY_SCORE_MODE,
)
from tdwm.evaluation.frozen_actor_free_td_common import _resolve_joint_checkpoint
from tdwm.evaluation.lewm_checkpoint import _sha256, _write_json

EMA_SG_SCORE_MODES = frozenset(
    (
        "f_only",
        "g_only",
        "f_plus_g",
        FIRST_ACTION_SCORE_MODE,
        ROLLOUT_MEAN_SCORE_MODE,
    )
)
NEW_SCORE_MODES = frozenset((FIRST_ACTION_SCORE_MODE, ROLLOUT_MEAN_SCORE_MODE))
EMA_SG_G_SCORE = "negative_goal_projection_of_v2_ema_sg_online_predictor"
FORMAL_HORIZON_BY_SCORE_MODE = {
    mode: V2_FORMAL_HORIZON_BY_SCORE_MODE[mode] for mode in EMA_SG_SCORE_MODES
}


def validate_v2_score_mode(score_mode: str) -> str:
    """Keep V2 EMA evaluation inside its audited score-mode set."""

    if score_mode not in EMA_SG_SCORE_MODES:
        raise ValueError(
            f"score_mode {score_mode!r} is incompatible with V2 EMA; expected "
            f"one of {sorted(EMA_SG_SCORE_MODES)}."
        )
    return score_mode


def actor_free_td_v2_ema_sg_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
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
        g_first_weight=g_first_weight,
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
    g_first_weight: float | None = None,
) -> dict[str, Any]:
    selected = score_mode or str(
        protocol.get("inference_objective", {}).get("score_mode", "f_plus_g")
    )
    validate_v2_score_mode(selected)
    configured = configure_actor_free_td_v2_evaluation_mode(
        protocol,
        smoke=smoke,
        pilot=pilot,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    if selected != ROLLOUT_MEAN_SCORE_MODE:
        configured["inference_objective"]["g_score"] = EMA_SG_G_SCORE
    return configured


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
    training_manifest_path = kwargs.pop("training_manifest_path", None)
    selected = kwargs.get("score_mode")
    if selected is not None:
        selected = validate_v2_score_mode(str(selected))
    protocol_path = kwargs.get("protocol_path")
    formal_protocol: Mapping[str, Any] | None = None
    if protocol_path is not None:
        formal_protocol = load_actor_free_td_v2_ema_sg_evaluation_protocol(
            protocol_path, spec=spec
        )
    if selected is None and formal_protocol is not None:
        selected = validate_v2_score_mode(
            str(formal_protocol["inference_objective"]["score_mode"])
        )
    resolved_training_manifest: Path | None = None
    if selected in NEW_SCORE_MODES:
        if formal_protocol is None:
            raise ValueError("New V2 EMA score modes require a protocol_path.")
        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("New V2 EMA score modes require a checkpoint_path.")
        checkpoint_file = _resolve_joint_checkpoint(checkpoint_path)
        resolved_training_manifest = _resolve_training_manifest(
            checkpoint_file,
            explicit=training_manifest_path,
        )
        _validate_training_manifest(
            resolved_training_manifest,
            method=spec.method,
            variant=spec.variant,
            evaluation_protocol=formal_protocol,
        )
    result = evaluate_actor_free_td_v2(
        spec=spec,
        checkpoint_loader=checkpoint_loader,
        policy_factory=policy_factory,
        checkpoint_protocol_validator=(
            validate_actor_free_td_v2_ema_sg_checkpoint_protocol
        ),
        **kwargs,
    )
    if selected in NEW_SCORE_MODES and result.get("score_mode") != selected:
        raise ValueError("V2 EMA runtime returned a different new score mode.")
    if result.get("score_mode") not in NEW_SCORE_MODES:
        return result
    assert resolved_training_manifest is not None
    return _record_v2_ema_identity(
        result,
        output_dir=kwargs["output_dir"],
        training_manifest_path=resolved_training_manifest,
    )


def _record_v2_ema_identity(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    training_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a new-score V2 EMA result to its training and evaluation revisions."""

    output_path = Path(output_dir).expanduser().resolve()
    manifest_path = output_path / "protocol_manifest.json"
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("V2 EMA protocol manifest must contain a JSON object.")
    score_mode = result.get("score_mode")
    if score_mode not in NEW_SCORE_MODES:
        raise ValueError("V2 EMA identity metadata is only for the two new modes.")
    if manifest.get("score_mode") != score_mode:
        raise ValueError("V2 EMA result and protocol manifest score modes differ.")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("V2 EMA protocol manifest is missing protocol metadata.")
    inference = protocol.get("inference_objective")
    if not isinstance(inference, Mapping) or inference.get("score_mode") != score_mode:
        raise ValueError("V2 EMA manifest protocol has the wrong score mode.")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V2 EMA protocol manifest is missing checkpoint metadata.")
    if checkpoint.get("method_family") != METHOD_FAMILY:
        raise ValueError("V2 EMA result is bound to the wrong checkpoint family.")
    if checkpoint.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("V2 EMA result has the wrong checkpoint implementation.")
    method = checkpoint.get("method")
    variant = checkpoint.get("variant")
    if (
        not isinstance(variant, str)
        or not variant
        or method != f"{METHOD_FAMILY}_{variant}"
    ):
        raise ValueError("V2 EMA checkpoint method and variant are inconsistent.")
    if result.get("method") != method or result.get("variant") != variant:
        raise ValueError("V2 EMA result identity differs from its checkpoint.")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 3 <= epoch <= 10:
        raise ValueError("V2 EMA evaluation checkpoint epoch must lie in [3, 10].")
    checkpoint_sha256 = checkpoint.get("sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("V2 EMA checkpoint SHA-256 is missing or malformed.")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("V2 EMA checkpoint path is missing from the manifest.")
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(resolved_checkpoint)
    if _sha256(resolved_checkpoint) != checkpoint_sha256:
        raise ValueError("V2 EMA checkpoint file SHA-256 differs from the manifest.")
    evaluation_runtime = manifest.get("runtime")
    if not isinstance(evaluation_runtime, Mapping):
        raise ValueError("V2 EMA protocol manifest is missing runtime metadata.")
    evaluation_commit = evaluation_runtime.get("tdwm_git_revision")
    if (
        not isinstance(evaluation_commit, str)
        or len(evaluation_commit) != 40
        or any(character not in "0123456789abcdef" for character in evaluation_commit)
    ):
        raise ValueError("V2 EMA evaluation commit is missing or malformed.")
    resolved_training_manifest = _resolve_training_manifest(
        resolved_checkpoint,
        explicit=training_manifest_path,
    )
    _validate_training_manifest(
        resolved_training_manifest,
        method=str(method),
        variant=str(variant),
        evaluation_protocol=protocol,
    )
    metadata = {
        "version_key": VERSION_KEY,
        "version_display_name": VERSION_DISPLAY_NAME,
        "training_commit": TRAINING_COMMIT,
        "evaluation_commit": evaluation_commit,
        "method": method,
        "epoch": epoch,
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_path": str(resolved_training_manifest),
        "training_manifest_sha256": _sha256(resolved_training_manifest),
    }
    recorded = copy.deepcopy(dict(result))
    manifest.update(copy.deepcopy(metadata))
    recorded.update(metadata)
    _write_json(manifest_path, manifest)
    _write_json(output_path / "results.json", recorded)
    return recorded


def _resolve_training_manifest(
    checkpoint_path: Path,
    *,
    explicit: str | Path | None,
) -> Path:
    run_root = _checkpoint_run_root(checkpoint_path)
    expected = (run_root / "training_manifest.json").resolve()
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate != expected:
            raise ValueError(
                "The explicit V2 EMA training manifest must be the checkpoint "
                "run's exact training_manifest.json."
            )
        return candidate
    if not expected.is_file():
        raise FileNotFoundError(expected)
    return expected


def _checkpoint_run_root(checkpoint_path: Path) -> Path:
    resolved = checkpoint_path.expanduser().resolve()
    for ancestor in resolved.parents:
        if ancestor.name == "checkpoints":
            return ancestor.parent
    raise ValueError(
        "V2 EMA checkpoint must live below its run's checkpoints directory."
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_training_manifest(
    path: Path,
    *,
    method: str,
    variant: str,
    evaluation_protocol: Mapping[str, Any],
) -> None:
    with path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError("V2 EMA training manifest must contain a JSON object.")
    expected_identity = {
        "method": method,
        "method_family": METHOD_FAMILY,
        "variant": variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "stage": TRAINING_STAGE,
        "initialization": INITIALIZATION,
        "seed": 3072,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"V2 EMA training manifest {key} does not match the checkpoint run."
            )
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("V2 EMA training manifest is missing runtime metadata.")
    if runtime.get("tdwm_git_revision") != TRAINING_COMMIT:
        raise ValueError("V2 EMA training manifest has the wrong training commit.")
    training_protocol = manifest.get("protocol")
    if not isinstance(training_protocol, Mapping):
        raise ValueError("V2 EMA training manifest is missing its training protocol.")
    protocol_sha256 = manifest.get("protocol_sha256")
    if protocol_sha256 != _canonical_sha256(training_protocol):
        raise ValueError("V2 EMA training protocol SHA-256 is missing or stale.")
    for key in (
        "method",
        "method_family",
        "variant",
        "implementation_version",
        "initialization",
        "initialization_contract",
        "predictor",
        "task_sampling",
        "joint_objective",
        "source_v1",
        "source_artifacts",
    ):
        if training_protocol.get(key) != evaluation_protocol.get(key):
            raise ValueError(f"V2 EMA training protocol {key} differs from evaluation.")
    if training_protocol.get("stage") != TRAINING_STAGE:
        raise ValueError("V2 EMA training protocol has the wrong stage.")


__all__ = [
    "EMA_SG_SCORE_MODES",
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "FORMAL_O50_PLANNING",
    "NEW_SCORE_MODES",
    "TRAINING_COMMIT",
    "VERSION_DISPLAY_NAME",
    "VERSION_KEY",
    "_record_v2_ema_identity",
    "actor_free_td_v2_ema_sg_output_directory_name",
    "configure_actor_free_td_v2_ema_sg_evaluation_mode",
    "evaluate_actor_free_td_v2_ema_sg",
    "load_actor_free_td_v2_ema_sg_evaluation_protocol",
    "validate_actor_free_td_v2_ema_sg_checkpoint_protocol",
    "validate_actor_free_td_v2_ema_sg_evaluation_protocol",
    "validate_v2_raw_action_compatibility",
    "validate_v2_score_mode",
]
