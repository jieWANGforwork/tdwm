"""Validate and archive the Actor-Free TD-JEPA V1 Cube O50 study.

The V1 archive is deliberately separate from the historical Actor-Free
TD-LeWM 7x3 archive.  It validates six frozen-LeWM methods (C, D, F, G1, G2,
G3), each evaluated with F-only, G-only, and F+G scoring.  The G-only planner
uses horizon one; the other two score modes use horizon five.

The current V1 trainer did not write peak CUDA memory or its CUDA device into
the trainer JSON.  This module never invents either value.  It requires a
separate execution-evidence record and preserves the gap explicitly in the
machine-readable summary.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
METHOD_FAMILY = "actor_free_td_lewm_v1"
IMPLEMENTATION_VERSION = "v1"
OBJECTIVE_VERSION = 0
DEPLOYMENT_CHECKPOINT_VERSION = 1
TRAINING_SEED = 3072
TRAINING_EPOCHS = 10
TRAINING_STEPS = 127_960
OPTIMIZER_STEPS_PER_EPOCH = 12_796
EPISODES = 50
GOAL_OFFSET = 50
PLANNING_SEED = 42
SELECTION_SHA256 = "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
PRETRAINED_WORLD_MODEL_SHA256 = (
    "198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257"
)
EXPECTED_ACTION_ENCODER_SHA256 = (
    "2657b55140013b4b071cd8cdea63f1eac5c65c498d55331c7499744ef31a9cd3"
)
WORLD_MODEL_PARAMETERS = 18_034_628
PREDICTOR_PARAMETERS = 379_072

VARIANT_ORDER = ("c", "d", "f", "g1", "g2", "g3")
SCORE_MODES = ("f_only", "g_only", "f_plus_g")
FORMAL_HORIZON_BY_SCORE_MODE = {
    "f_only": 5,
    "g_only": 1,
    "f_plus_g": 5,
}

TRAINING_PROTOCOL_SHA256 = {
    "c": "14e9b00346bab0e7b5e527544968a072987fb0bbfb752d3a9f6050c2f4f32b6b",
    "d": "cfd7af7852d8021bb59f30920577506f773ba9321482d715f1f00370e6781752",
    "f": "ba4f68cfec17d0a893b94b7d013f118536ee3e32f5aae5a46b6a90301b284750",
    "g1": "84b1c5bb9a3098eae7967e72f2b00e4f2c651fe12049532ac8a181153f2a293e",
    "g2": "409d62cfee01e5325e8220e3cfa26b8c16f5659afa314f71212f1d5bf8f844c4",
    "g3": "56dc578a7a0148d50f23bfd53bde311913b9598f3c241212f392d4417134fea1",
}
EVALUATION_PROTOCOL_SHA256 = {
    "c": "d95a33b26fa9d20fa27fa7713465c21d688112318209ec9e8d57a3818d52a0cd",
    "d": "23925a88afd560353b23fac9dffc7392d989899335d8fba169dd43e7ab5c412e",
    "f": "ce55d97dd08fa3e83ef5a319530a54f82ef85f8b42348605666a0da78e93c659",
    "g1": "e4f96a5159e2419ae610a37f1027698213b3cffdf53ee7e79f0d32701f07b673",
    "g2": "c7264490bc83cb4ad7571e32008640aa6359b12615ac792d431db4b5ce5da8fa",
    "g3": "93c5b5aee1560960d762f8a0de0421a949b318140eefaa1c5c0ddcf28e065828",
}

DISPLAY_NAMES = {
    "c": "V1-C Goal-Projected TD",
    "d": "V1-D Goal-Value Weighted TD",
    "f": "V1-F Same-Future / Different-Goal Advantage",
    "g1": "V1-G1 Neighbor Action Advantage",
    "g2": "V1-G2 Prefix-Mean Advantage",
    "g3": "V1-G3 Prefix-Marginal Advantage",
}
METHOD_SPECS = {
    "c": {
        "network": "Frozen LeWM + frozen shared action encoder + one 379,072-parameter TD-JEPA predictor",
        "loss": "Common feature TD plus goal-projected TD residual on goal-derived tasks",
        "special": "Directly constrains the detached TD target and prediction after projection onto the matched goal",
    },
    "d": {
        "network": "Frozen LeWM + frozen shared action encoder + one 379,072-parameter TD-JEPA predictor",
        "loss": "Detached target-goal scores reweight the common real-transition feature TD",
        "special": "Goal-subset softmax weights; random-task weight remains one; final weights have mean one",
    },
    "f": {
        "network": "Frozen LeWM + frozen shared action encoder + one 379,072-parameter TD-JEPA predictor",
        "loss": "Same-future/different-goal detached advantage reweights common feature TD",
        "special": "Matching task score is contrasted with all goal-derived tasks in the batch",
    },
    "g1": {
        "network": "Frozen LeWM + frozen shared action encoder + one 379,072-parameter TD-JEPA predictor",
        "loss": "Neighbor-action detached advantage reweights common real-action feature TD",
        "special": "Other-episode KNN actions are comparison-only and never create candidate TD targets",
    },
    "g2": {
        "network": "Frozen LeWM + frozen shared action encoder + one 379,072-parameter TD-JEPA predictor",
        "loss": "Full-prefix score minus mean prefix score reweights common feature TD",
        "special": "Zero-mean suffix prefixes are comparison-only; the real full action supplies the TD pair",
    },
    "g3": {
        "network": "Frozen LeWM + frozen shared action encoder + one 379,072-parameter TD-JEPA predictor",
        "loss": "Mean adjacent prefix-score improvement reweights common feature TD",
        "special": "Prefix marginal gains are detached comparison signals, not extra TD targets",
    },
}


class BundleValidationError(ValueError):
    """Raised when a purported formal V1 result bundle is incomplete or drifts."""


@dataclass(frozen=True)
class TrainingRun:
    variant: str
    method: str
    checkpoint_path: str
    training_commit: str
    protocol_sha256: str
    source_files_sha256: Mapping[str, str]
    curve: tuple[Mapping[str, float | int], ...]
    final_metrics: Mapping[str, float | int]
    best_validation: Mapping[str, float | int]
    manifest: Mapping[str, Any]
    result: Mapping[str, Any]
    execution_evidence: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationRun:
    variant: str
    method: str
    score_mode: str
    planning_horizon: int
    success_count: int
    success_rate: float
    successes: tuple[bool, ...]
    elapsed_seconds: float
    checkpoint_path: str
    checkpoint_sha256: str
    evaluation_commit: str
    formal_protocol_sha256: str
    configured_protocol_sha256: str
    source_files_sha256: Mapping[str, str]
    manifest: Mapping[str, Any]
    result: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedStudy:
    bundle_root: Path
    training: Mapping[str, TrainingRun]
    evaluations: Mapping[str, Mapping[str, EvaluationRun]]
    selection: Mapping[str, Any]
    selection_sha256: str
    training_acceptance: Mapping[str, Any]
    training_acceptance_sha256: str


def _error(context: str, message: str) -> BundleValidationError:
    return BundleValidationError(f"{context}: {message}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise _error(context, f"missing required file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _error(context, f"invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise _error(context, "JSON root must be an object")
    return value, raw


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(context, "must be an object")
    return value


def _nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(context, "must be a non-empty string")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(context, "must be a lowercase 64-character SHA-256")
    return value


def _git_revision(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(context, "must be a lowercase Git revision")
    return value


def _positive_int(value: Any, *, context: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(context, "must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise _error(context, f"must be >= {minimum}")
    return value


def _finite_float(value: Any, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise _error(context, "must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _error(context, "must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise _error(context, "must be finite" + (" and positive" if positive else ""))
    return result


def _without_absolute_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key == "path" or key.endswith("_path"):
                continue
            result[str(key)] = _without_absolute_paths(item)
        return result
    if isinstance(value, list):
        return [_without_absolute_paths(item) for item in value]
    return deepcopy(value)


def _assert_exact(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise _error(context, f"{key} must equal {expected_value!r}")


def _path(value: Any, *, context: str) -> str:
    return os.path.normpath(_nonempty_string(value, context=context))


def _validate_execution_evidence(
    path: Path,
    *,
    variant: str,
    result: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    context = f"{variant}/execution_evidence"
    evidence, raw = _load_json(path, context=context)
    _assert_exact(
        evidence,
        {
            "schema_version": 1,
            "source": "external_execution_evidence",
            "method": f"{METHOD_FAMILY}_{variant}",
            "variant": variant,
        },
        context=context,
    )
    _nonempty_string(evidence.get("hostname"), context=f"{context}.hostname")
    gpu = _mapping(evidence.get("gpu"), context=f"{context}.gpu")
    _positive_int(gpu.get("index"), context=f"{context}.gpu.index", allow_zero=True)
    _nonempty_string(gpu.get("name"), context=f"{context}.gpu.name")
    _nonempty_string(gpu.get("uuid"), context=f"{context}.gpu.uuid")

    process = _mapping(evidence.get("process"), context=f"{context}.process")
    _positive_int(process.get("pid"), context=f"{context}.process.pid")
    command = _nonempty_string(
        process.get("command"), context=f"{context}.process.command"
    )
    if f"actor_free_td_lewm_v1_{variant}" not in command:
        raise _error(
            context, "process.command does not identify the expected V1 method"
        )
    if _sha256_bytes(command.encode()) != _sha256(
        process.get("command_sha256"), context=f"{context}.process.command_sha256"
    ):
        raise _error(context, "process.command_sha256 disagrees with command")
    _path(process.get("cwd"), context=f"{context}.process.cwd")
    evidence_revision = _git_revision(
        process.get("git_revision"), context=f"{context}.process.git_revision"
    )
    if evidence_revision != runtime.get("tdwm_git_revision"):
        raise _error(context, "process.git_revision differs from trainer manifest")
    _nonempty_string(process.get("started_at"), context=f"{context}.process.started_at")
    _nonempty_string(process.get("ended_at"), context=f"{context}.process.ended_at")
    exit_code = process.get("exit_code")
    if exit_code not in (0, None):
        raise _error(
            context,
            "completed formal training must have exit_code 0 or an explicitly unavailable exit code",
        )
    if exit_code is None:
        _nonempty_string(
            process.get("exit_code_evidence"),
            context=f"{context}.process.exit_code_evidence",
        )

    log = _mapping(evidence.get("log"), context=f"{context}.log")
    _path(log.get("path"), context=f"{context}.log.path")
    _positive_int(
        log.get("size_bytes"), context=f"{context}.log.size_bytes", allow_zero=True
    )
    _sha256(log.get("sha256"), context=f"{context}.log.sha256")
    snapshot = _mapping(
        evidence.get("gpu_process_snapshot"),
        context=f"{context}.gpu_process_snapshot",
    )
    _path(snapshot.get("path"), context=f"{context}.gpu_process_snapshot.path")
    _nonempty_string(
        snapshot.get("captured_at"),
        context=f"{context}.gpu_process_snapshot.captured_at",
    )
    _sha256(snapshot.get("sha256"), context=f"{context}.gpu_process_snapshot.sha256")

    gaps = _mapping(
        evidence.get("trainer_recording_gaps"),
        context=f"{context}.trainer_recording_gaps",
    )
    peak_recorded = result.get("peak_cuda_memory_bytes") is not None
    cuda_recorded = runtime.get("cuda_device") is not None
    expected_peak = (
        "recorded_by_v1_trainer" if peak_recorded else "not_recorded_by_v1_trainer"
    )
    expected_cuda = (
        "recorded_by_v1_trainer" if cuda_recorded else "not_recorded_by_v1_trainer"
    )
    if gaps.get("peak_cuda_memory_bytes") != expected_peak:
        raise _error(
            context,
            f"trainer_recording_gaps.peak_cuda_memory_bytes must be {expected_peak!r}",
        )
    if gaps.get("runtime.cuda_device") != expected_cuda:
        raise _error(
            context,
            f"trainer_recording_gaps.runtime.cuda_device must be {expected_cuda!r}",
        )
    if peak_recorded:
        peak_value = _positive_int(
            result.get("peak_cuda_memory_bytes"),
            context=f"{variant}.training_result.peak_cuda_memory_bytes",
        )
        peak_provenance: dict[str, Any] = {
            "status": "recorded_by_v1_trainer",
            "value": peak_value,
            "source": "training_result.json",
        }
    else:
        peak_provenance = {
            "status": "not_recorded_by_v1_trainer",
            "value": None,
            "source": "external_execution_evidence",
        }
    if cuda_recorded:
        cuda_value = _nonempty_string(
            runtime.get("cuda_device"),
            context=f"{variant}.training_manifest.runtime.cuda_device",
        )
        cuda_provenance: dict[str, Any] = {
            "status": "recorded_by_v1_trainer",
            "value": cuda_value,
            "source": "training_manifest.json",
        }
    else:
        cuda_provenance = {
            "status": "not_recorded_by_v1_trainer",
            "value": None,
            "source": "external_execution_evidence",
        }
    evidence_sha = _sha256_bytes(raw)
    peak_provenance["execution_evidence_sha256"] = evidence_sha
    cuda_provenance["execution_evidence_sha256"] = evidence_sha
    return (
        evidence,
        raw,
        {
            "peak_cuda_memory_bytes": peak_provenance,
            "runtime.cuda_device": cuda_provenance,
            "gpu_identity": {
                "status": "recorded_by_external_execution_evidence",
                "index": gpu["index"],
                "name": gpu["name"],
                "uuid": gpu.get("uuid"),
                "source": "external_execution_evidence",
                "execution_evidence_sha256": evidence_sha,
            },
        },
    )


def _read_metric_value(row: Mapping[str, str], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            number = _finite_float(value, context=f"metrics.{key}")
            return number
    return None


def _extract_training_curve(
    path: Path, *, variant: str
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    context = f"{variant}/metrics.csv"
    if not path.is_file():
        raise _error(context, f"missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise _error(context, "missing CSV header")
        rows = list(reader)
    by_epoch: dict[int, dict[str, list[float]]] = {
        epoch: {"train_method": [], "train_base": [], "validation": []}
        for epoch in range(TRAINING_EPOCHS)
    }
    steps: list[int] = []
    for row in rows:
        if row.get("step") not in (None, ""):
            try:
                steps.append(int(float(row["step"])))
            except (TypeError, ValueError) as error:
                raise _error(context, "step must be integral") from error
        if row.get("epoch") in (None, ""):
            continue
        try:
            epoch = int(float(row["epoch"]))
        except (TypeError, ValueError) as error:
            raise _error(context, "epoch must be integral") from error
        if epoch not in by_epoch:
            continue
        values = {
            "train_method": _read_metric_value(row, ("train/loss_epoch", "train/loss")),
            "train_base": _read_metric_value(
                row, ("train/base_td_loss_epoch", "train/base_td_loss")
            ),
            "validation": _read_metric_value(row, ("validation/loss",)),
        }
        validation_base = _read_metric_value(
            row, ("validation/base_td_loss", "validation/base_td_loss_epoch")
        )
        if values["validation"] is not None and validation_base is not None:
            if not math.isclose(
                values["validation"], validation_base, rel_tol=1e-8, abs_tol=1e-10
            ):
                raise _error(
                    context,
                    "validation/loss must equal the common validation/base_td_loss",
                )
        for key, value in values.items():
            if value is not None:
                by_epoch[epoch][key].append(value)
    if not steps or max(steps) != TRAINING_STEPS - 1:
        raise _error(context, "final zero-based metrics step must equal 127959")

    curve: list[dict[str, Any]] = []
    for epoch in range(TRAINING_EPOCHS):
        aggregate: dict[str, float] = {}
        for key, values in by_epoch[epoch].items():
            if not values:
                raise _error(context, f"missing {key} aggregate for epoch {epoch}")
            if any(
                not math.isclose(value, values[-1], rel_tol=1e-8, abs_tol=1e-10)
                for value in values
            ):
                raise _error(context, f"conflicting {key} aggregates for epoch {epoch}")
            aggregate[key] = values[-1]
        curve.append(
            {
                "epoch": epoch + 1,
                "train_method_loss": aggregate["train_method"],
                "train_base_td_loss": aggregate["train_base"],
                "validation_base_td_loss": aggregate["validation"],
            }
        )
    best = min(curve, key=lambda item: item["validation_base_td_loss"])
    return (
        tuple(curve),
        deepcopy(curve[-1]),
        {
            "epoch": best["epoch"],
            "value": best["validation_base_td_loss"],
        },
    )


def _validate_training(variant_root: Path, *, variant: str) -> TrainingRun:
    context = f"{variant}/training"
    result, result_raw = _load_json(
        variant_root / "training_result.json", context=f"{context}.result"
    )
    manifest, manifest_raw = _load_json(
        variant_root / "training_manifest.json", context=f"{context}.manifest"
    )
    method = f"{METHOD_FAMILY}_{variant}"
    _assert_exact(
        result,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "seed": TRAINING_SEED,
            "final_epoch": TRAINING_EPOCHS,
            "global_step": TRAINING_STEPS,
            "pretrained_world_model_sha256": PRETRAINED_WORLD_MODEL_SHA256,
        },
        context=f"{context}.result",
    )
    run_dir = _path(result.get("run_dir"), context=f"{context}.result.run_dir")
    expected_last = os.path.join(run_dir, "checkpoints", "lightning", "last.ckpt")
    if (
        _path(
            result.get("last_checkpoint"), context=f"{context}.result.last_checkpoint"
        )
        != expected_last
    ):
        raise _error(
            context, "last_checkpoint must be run_dir/checkpoints/lightning/last.ckpt"
        )
    expected_deployment = os.path.join(
        run_dir,
        "checkpoints",
        method,
        variant,
        f"epoch_{TRAINING_EPOCHS:02d}.pt",
    )
    deployment = _path(
        result.get("deployment_checkpoint"),
        context=f"{context}.result.deployment_checkpoint",
    )
    if deployment != expected_deployment:
        raise _error(
            context, "deployment_checkpoint does not match the V1 epoch-10 export path"
        )

    _assert_exact(
        manifest,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "seed": TRAINING_SEED,
        },
        context=f"{context}.manifest",
    )
    protocol = _mapping(manifest.get("protocol"), context=f"{context}.protocol")
    protocol_sha = _fingerprint(protocol)
    if protocol_sha != TRAINING_PROTOCOL_SHA256[variant]:
        raise _error(context, "protocol differs from the complete locked training YAML")
    if (
        manifest.get("protocol_sha256") != protocol_sha
        or result.get("protocol_sha256") != protocol_sha
    ):
        raise _error(
            context, "protocol_sha256 disagrees with the canonical locked protocol"
        )
    _assert_exact(
        protocol,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "environment": "cube",
            "stage": "full_training",
            "initialization": "frozen_pretrained_lewm",
        },
        context=f"{context}.protocol",
    )
    predictor = _mapping(protocol.get("predictor"), context=f"{context}.predictor")
    _assert_exact(
        predictor,
        {
            "architecture": "td_jepa_forward_map_v1",
            "state_dim": 192,
            "raw_action_dim": 25,
            "action_dim": 192,
            "action_embedding_dim": 192,
            "task_dim": 192,
            "output_dim": 192,
            "num_parallel": 1,
            "action_processing": "frozen_shared_lewm_action_encoder",
            "shared_lewm_action_encoder": True,
            "action_encoder_trainable": False,
            "action_encoder_source": "world_model.action_encoder",
            "bootstrap_action": "dataset_next_action",
            "actor": "none",
            "reward": "none",
        },
        context=f"{context}.predictor",
    )
    pretrained = _mapping(
        protocol.get("pretrained_world_model"), context=f"{context}.pretrained"
    )
    _assert_exact(
        pretrained,
        {"checkpoint_sha256": PRETRAINED_WORLD_MODEL_SHA256, "frozen": True},
        context=f"{context}.pretrained",
    )

    model = _mapping(manifest.get("model"), context=f"{context}.model")
    _assert_exact(
        model,
        {
            "lewm_parameters": WORLD_MODEL_PARAMETERS,
            "trainable_lewm_parameters": 0,
            "predictor_parameters": PREDICTOR_PARAMETERS,
        },
        context=f"{context}.model",
    )
    initialization = _mapping(
        model.get("initialization"), context=f"{context}.model.initialization"
    )
    _assert_exact(
        initialization,
        {
            "strategy": "frozen_pretrained_lewm",
            "source_checkpoint_sha256": PRETRAINED_WORLD_MODEL_SHA256,
            "frozen": True,
        },
        context=f"{context}.model.initialization",
    )
    training = _mapping(manifest.get("training"), context=f"{context}.training")
    _assert_exact(
        training,
        {
            "formal_optimizer_steps": TRAINING_STEPS,
            "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "configured_optimizer_steps": TRAINING_STEPS,
            "validation_skipped": False,
            "data_source": "frozen_latent_store",
            "sampling_unit": "transition",
            "train_sampling": "random_with_replacement",
            "validation_primary_objective": "common_base_td_all_variants",
            "world_model_visual_encode_during_training": False,
            "shared_action_encoder_forward_during_training": True,
        },
        context=f"{context}.training",
    )
    runtime = _mapping(manifest.get("runtime"), context=f"{context}.runtime")
    _assert_exact(runtime, {"stable_worldmodel": "0.1.1"}, context=f"{context}.runtime")
    for key in ("torch", "python", "platform"):
        _nonempty_string(runtime.get(key), context=f"{context}.runtime.{key}")
    training_commit = _git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.runtime.tdwm_git_revision"
    )

    dataset = _mapping(manifest.get("dataset"), context=f"{context}.dataset")
    if dataset.get("format") != "lance":
        raise _error(context, "training dataset must use the audited Lance format")
    split = _mapping(dataset.get("split"), context=f"{context}.dataset.split")
    train_samples = _positive_int(
        split.get("train_samples"), context=f"{context}.split.train_samples"
    )
    validation_samples = _positive_int(
        split.get("validation_samples"),
        context=f"{context}.split.validation_samples",
    )
    if train_samples + validation_samples != dataset.get("sequence_samples"):
        raise _error(context, "split sample counts must equal sequence_samples")
    _sha256(
        split.get("train_indices_sha256"),
        context=f"{context}.split.train_indices_sha256",
    )
    _sha256(
        split.get("validation_indices_sha256"),
        context=f"{context}.split.validation_indices_sha256",
    )
    latent = _mapping(
        manifest.get("frozen_latent_store"), context=f"{context}.frozen_latent_store"
    )
    latent_sha = _sha256(
        latent.get("manifest_sha256"), context=f"{context}.latent.manifest_sha256"
    )
    if result.get("frozen_latent_store_manifest_sha256") != latent_sha:
        raise _error(
            context, "training_result latent-store SHA disagrees with manifest"
        )
    _assert_exact(
        latent,
        {
            "pretrained_checkpoint_sha256": PRETRAINED_WORLD_MODEL_SHA256,
            "total_rows": 2_010_000,
            "embed_dim": 192,
            "frame_skip": 5,
            "history_frames": 3,
            "action_dim": 5,
            "action_block_dim": 25,
            "stable_worldmodel_version": "0.1.1",
        },
        context=f"{context}.frozen_latent_store",
    )
    neighbor = manifest.get("neighbor_index")
    if variant == "g1":
        neighbor_mapping = _mapping(neighbor, context=f"{context}.neighbor_index")
        neighbor_sha = _sha256(
            neighbor_mapping.get("manifest_sha256"),
            context=f"{context}.neighbor_index.manifest_sha256",
        )
        if result.get("neighbor_index_manifest_sha256") != neighbor_sha:
            raise _error(
                context, "G1 neighbor-index SHA disagrees between trainer files"
            )
    elif (
        neighbor is not None or result.get("neighbor_index_manifest_sha256") is not None
    ):
        raise _error(context, "only V1 G1 may record a neighbor index")

    evidence, evidence_raw, provenance = _validate_execution_evidence(
        variant_root / "execution_evidence.json",
        variant=variant,
        result=result,
        runtime=runtime,
    )
    metrics_path = variant_root / "metrics.csv"
    curve, final_metrics, best_validation = _extract_training_curve(
        metrics_path, variant=variant
    )
    return TrainingRun(
        variant=variant,
        method=method,
        checkpoint_path=deployment,
        training_commit=training_commit,
        protocol_sha256=protocol_sha,
        source_files_sha256={
            "training_result.json": _sha256_bytes(result_raw),
            "training_manifest.json": _sha256_bytes(manifest_raw),
            "metrics.csv": _sha256_bytes(metrics_path.read_bytes()),
            "execution_evidence.json": _sha256_bytes(evidence_raw),
        },
        curve=curve,
        final_metrics=final_metrics,
        best_validation=best_validation,
        manifest=manifest,
        result=result,
        execution_evidence=evidence,
        provenance=provenance,
    )


def _validate_selection(
    selection: Mapping[str, Any], *, raw: bytes, context: str
) -> None:
    if _sha256_bytes(raw) != SELECTION_SHA256:
        raise _error(
            context,
            "episode_selection.json SHA-256 must equal locked seed-42 O50 selection",
        )
    keys = ("episode_indices", "start_steps", "goal_steps", "valid_row_ranks")
    for key in keys:
        values = selection.get(key)
        if not isinstance(values, list) or len(values) != EPISODES:
            raise _error(context, f"{key} must contain exactly 50 integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise _error(context, f"{key} must contain integers")
    for episode, start, goal in zip(
        selection["episode_indices"],
        selection["start_steps"],
        selection["goal_steps"],
    ):
        if not 0 <= episode < 10_000:
            raise _error(context, "episode_indices must lie in [0,10000)")
        if not 0 <= start < goal < 201 or goal - start != GOAL_OFFSET:
            raise _error(
                context, "selection must satisfy 0 <= start < goal < 201 and O50"
            )


def _validate_predictor_config(
    config: Mapping[str, Any], *, formal: Mapping[str, Any], variant: str, context: str
) -> None:
    _assert_exact(
        config,
        {
            "method": f"{METHOD_FAMILY}_{variant}",
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        },
        context=context,
    )
    predictor = _mapping(formal.get("predictor"), context=f"{context}.formal.predictor")
    for key, expected in predictor.items():
        actual = config.get(key)
        if key in ("gamma", "target_ema_decay"):
            if actual is None or not math.isclose(float(actual), float(expected)):
                raise _error(context, f"predictor_config.{key} differs from protocol")
        elif actual != expected:
            raise _error(context, f"predictor_config.{key} differs from protocol")
    if config.get("task_sampling") != formal.get("task_sampling"):
        raise _error(context, "predictor_config.task_sampling differs from protocol")
    if config.get("joint_objective") != formal.get("joint_objective"):
        raise _error(context, "predictor_config.joint_objective differs from protocol")
    if config.get("pretrained_world_model") != formal.get("pretrained_world_model"):
        raise _error(
            context, "predictor_config.pretrained_world_model differs from protocol"
        )


def _success_vector(metrics: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    canonical = metrics.get("episode_successes")
    legacy = metrics.get("success")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise _error(context, "episode_successes and legacy success disagree")
    values = canonical if canonical is not None else legacy
    if not isinstance(values, list) or len(values) != EPISODES:
        raise _error(context, "metrics.episode_successes must contain 50 booleans")
    if any(not isinstance(value, bool) for value in values):
        raise _error(context, "metrics.episode_successes must contain booleans")
    return tuple(values)


def _validate_action_normalization(stats: Mapping[str, Any], *, context: str) -> None:
    for key in ("mean", "scale", "variance"):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != 5:
            raise _error(context, f"{key} must contain the five primitive actions")
        numbers = [
            _finite_float(value, context=f"{context}.{key}[{index}]")
            for index, value in enumerate(values)
        ]
        if key == "scale" and any(number <= 0.0 for number in numbers):
            raise _error(context, "scale entries must be positive")
        if key == "variance" and any(number < 0.0 for number in numbers):
            raise _error(context, "variance entries must be non-negative")
    _positive_int(stats.get("samples"), context=f"{context}.samples")


def _validate_evaluation(
    run_root: Path,
    *,
    variant: str,
    score_mode: str,
    training: TrainingRun,
) -> EvaluationRun:
    context = f"{variant}/{score_mode}/evaluation"
    result, result_raw = _load_json(
        run_root / "results.json", context=f"{context}.result"
    )
    manifest, manifest_raw = _load_json(
        run_root / "protocol_manifest.json", context=f"{context}.manifest"
    )
    selection, selection_raw = _load_json(
        run_root / "episode_selection.json", context=f"{context}.selection"
    )
    action_stats, action_raw = _load_json(
        run_root / "action_normalization.json",
        context=f"{context}.action_normalization",
    )
    _validate_action_normalization(
        action_stats, context=f"{context}.action_normalization"
    )
    _validate_selection(selection, raw=selection_raw, context=f"{context}.selection")
    if manifest.get("selection") != selection:
        raise _error(
            context, "protocol_manifest.selection differs from episode_selection.json"
        )
    normalization = _mapping(
        manifest.get("normalization"), context=f"{context}.normalization"
    )
    if normalization.get("action") != action_stats:
        raise _error(
            context, "action_normalization.json differs from protocol_manifest"
        )

    formal = _mapping(
        manifest.get("formal_protocol"), context=f"{context}.formal_protocol"
    )
    formal_sha = _fingerprint(formal)
    if formal_sha != EVALUATION_PROTOCOL_SHA256[variant]:
        raise _error(
            context, "formal_protocol differs from the locked V1 evaluator YAML"
        )
    configured = _mapping(manifest.get("protocol"), context=f"{context}.protocol")
    expected_configured = deepcopy(dict(formal))
    expected_configured.setdefault("inference_objective", {})["score_mode"] = score_mode
    expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[score_mode]
    expected_configured.setdefault("planning", {})["horizon"] = expected_horizon
    if configured != expected_configured:
        raise _error(
            context,
            f"configured protocol must equal the locked formal protocol with {score_mode} horizon {expected_horizon}",
        )
    if manifest.get("score_mode") != score_mode:
        raise _error(context, "protocol_manifest.score_mode is wrong")
    method = f"{METHOD_FAMILY}_{variant}"
    _assert_exact(
        result,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "score_mode": score_mode,
            "planning_horizon": expected_horizon,
            "smoke": False,
            "pilot": False,
            "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
            "predictor_parameter_count": PREDICTOR_PARAMETERS,
        },
        context=f"{context}.result",
    )
    evaluation = _mapping(configured.get("evaluation"), context=f"{context}.evaluation")
    _assert_exact(
        evaluation,
        {"episodes": EPISODES, "goal_offset": GOAL_OFFSET},
        context=f"{context}.evaluation",
    )
    planning = _mapping(configured.get("planning"), context=f"{context}.planning")
    _assert_exact(
        planning,
        {
            "solver": "CEM",
            "horizon": expected_horizon,
            "candidates": 300,
            "iterations": 30,
            "elites": 30,
            "action_block": 5,
            "frame_skip": 5,
            "episode_budget": 100,
            "planning_seed": PLANNING_SEED,
            "history_len": 1,
            "initial_distribution": "cem_gaussian_no_actor",
        },
        context=f"{context}.planning",
    )

    checkpoint = _mapping(manifest.get("checkpoint"), context=f"{context}.checkpoint")
    _assert_exact(
        checkpoint,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "epoch": TRAINING_EPOCHS,
            "global_step": TRAINING_STEPS,
            "formal_completion_required": True,
        },
        context=f"{context}.checkpoint",
    )
    checkpoint_path = _path(
        checkpoint.get("path"), context=f"{context}.checkpoint.path"
    )
    if checkpoint_path != training.checkpoint_path:
        raise _error(
            context, "evaluation checkpoint path differs from the training export"
        )
    checkpoint_sha = _sha256(
        checkpoint.get("sha256"), context=f"{context}.checkpoint.sha256"
    )
    predictor_config = _mapping(
        checkpoint.get("predictor_config"), context=f"{context}.predictor_config"
    )
    _validate_predictor_config(
        predictor_config, formal=formal, variant=variant, context=context
    )
    pretrained_provenance = _mapping(
        checkpoint.get("pretrained_world_model_provenance"),
        context=f"{context}.pretrained_world_model_provenance",
    )
    if (
        pretrained_provenance.get("source_checkpoint_sha256")
        != PRETRAINED_WORLD_MODEL_SHA256
    ):
        raise _error(context, "checkpoint uses a different pretrained LeWM")

    dataset = _mapping(manifest.get("dataset"), context=f"{context}.dataset")
    _assert_exact(
        dataset,
        {"format": "lance", "episodes": 10_000, "transitions": 2_010_000},
        context=f"{context}.dataset",
    )
    runtime = _mapping(manifest.get("runtime"), context=f"{context}.runtime")
    _assert_exact(
        runtime,
        {"stable_worldmodel": "0.1.1", "device": "cuda"},
        context=f"{context}.runtime",
    )
    for key in ("torch", "python", "platform", "cuda_device"):
        _nonempty_string(runtime.get(key), context=f"{context}.runtime.{key}")
    evaluation_commit = _git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.runtime.tdwm_git_revision"
    )

    metrics = _mapping(result.get("metrics"), context=f"{context}.metrics")
    successes = _success_vector(metrics, context=context)
    success_count = sum(successes)
    success_rate_percent = _finite_float(
        metrics.get("success_rate"), context=f"{context}.metrics.success_rate"
    )
    expected_percent = 100.0 * success_count / EPISODES
    if not math.isclose(
        success_rate_percent, expected_percent, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _error(context, "success_rate disagrees with episode_successes")
    # The evaluator persists a percentage (for example 46.0), while the
    # archive uses a normalized rate so downstream arithmetic remains clear.
    success_rate = success_rate_percent / 100.0
    elapsed = _finite_float(
        result.get("elapsed_seconds"),
        context=f"{context}.elapsed_seconds",
        positive=True,
    )
    return EvaluationRun(
        variant=variant,
        method=method,
        score_mode=score_mode,
        planning_horizon=expected_horizon,
        success_count=success_count,
        success_rate=success_rate,
        successes=successes,
        elapsed_seconds=elapsed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        evaluation_commit=evaluation_commit,
        formal_protocol_sha256=formal_sha,
        configured_protocol_sha256=_fingerprint(configured),
        source_files_sha256={
            "results.json": _sha256_bytes(result_raw),
            "protocol_manifest.json": _sha256_bytes(manifest_raw),
            "episode_selection.json": _sha256_bytes(selection_raw),
            "action_normalization.json": _sha256_bytes(action_raw),
        },
        manifest=manifest,
        result=result,
    )


def _common_training_fingerprints(training: Mapping[str, TrainingRun]) -> None:
    first = training[VARIANT_ORDER[0]]
    common_split = _without_absolute_paths(first.manifest["dataset"]["split"])
    common_dataset = _without_absolute_paths(first.manifest["dataset"])
    common_latent = _without_absolute_paths(first.manifest["frozen_latent_store"])
    first_runtime = first.manifest["runtime"]
    common_runtime = {
        key: first_runtime.get(key)
        for key in (
            "stable_worldmodel",
            "torch",
            "python",
            "platform",
            "compatibility_adapter",
        )
    }
    for variant in VARIANT_ORDER[1:]:
        run = training[variant]
        if _without_absolute_paths(run.manifest["dataset"]["split"]) != common_split:
            raise _error(
                variant, "training split samples or index hashes differ across methods"
            )
        if _without_absolute_paths(run.manifest["dataset"]) != common_dataset:
            raise _error(
                variant, "training dataset source/provenance fingerprint differs"
            )
        if (
            _without_absolute_paths(run.manifest["frozen_latent_store"])
            != common_latent
        ):
            raise _error(
                variant, "frozen latent-store fingerprint differs across methods"
            )
        runtime = run.manifest["runtime"]
        runtime_common = {key: runtime.get(key) for key in common_runtime}
        if runtime_common != common_runtime:
            raise _error(variant, "training critical runtime fingerprint differs")


def _common_evaluation_fingerprints(
    evaluations: Mapping[str, Mapping[str, EvaluationRun]],
) -> None:
    first = evaluations[VARIANT_ORDER[0]][SCORE_MODES[0]]
    first_manifest = first.manifest
    common_runtime = {
        key: first_manifest["runtime"].get(key)
        for key in (
            "stable_worldmodel",
            "torch",
            "python",
            "platform",
            "device",
            "cuda_device",
            "compatibility_adapter",
        )
    }
    common_dataset = _without_absolute_paths(first_manifest["dataset"])
    common_normalization = first_manifest["normalization"]
    for variant in VARIANT_ORDER:
        runs = evaluations[variant]
        checkpoints = {
            (run.checkpoint_path, run.checkpoint_sha256) for run in runs.values()
        }
        if len(checkpoints) != 1:
            raise _error(variant, "all three score modes must use the same checkpoint")
        for mode in SCORE_MODES:
            run = runs[mode]
            runtime = run.manifest["runtime"]
            if {key: runtime.get(key) for key in common_runtime} != common_runtime:
                raise _error(
                    f"{variant}/{mode}",
                    "evaluation critical runtime fingerprint differs",
                )
            if _without_absolute_paths(run.manifest["dataset"]) != common_dataset:
                raise _error(
                    f"{variant}/{mode}", "evaluation dataset fingerprint differs"
                )
            if run.manifest["normalization"] != common_normalization:
                raise _error(
                    f"{variant}/{mode}", "action normalization fingerprint differs"
                )


def _validate_training_acceptance(
    bundle_root: Path,
    *,
    training: Mapping[str, TrainingRun],
    evaluations: Mapping[str, Mapping[str, EvaluationRun]],
) -> tuple[dict[str, Any], bytes]:
    context = "training_acceptance"
    acceptance, raw = _load_json(
        bundle_root / "training_acceptance.json", context=context
    )
    _assert_exact(
        acceptance,
        {
            "schema_version": 1,
            "seed": TRAINING_SEED,
            "expected_epoch": TRAINING_EPOCHS,
            "expected_global_step": TRAINING_STEPS,
            "expected_action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
        },
        context=context,
    )
    status = acceptance.get("status")
    if status not in ("PASS", "PASS_WITH_WARNINGS"):
        raise _error(context, "status must be PASS or PASS_WITH_WARNINGS")
    errors = acceptance.get("errors")
    if errors not in ([], None):
        raise _error(context, "errors must be empty for an accepted formal run")
    warnings = acceptance.get("warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) or not item for item in warnings
    ):
        raise _error(context, "warnings must be a list of non-empty strings")
    if status == "PASS" and warnings:
        raise _error(context, "PASS cannot hide warnings")
    if status == "PASS_WITH_WARNINGS" and not warnings:
        raise _error(context, "PASS_WITH_WARNINGS must disclose at least one warning")
    _nonempty_string(
        acceptance.get("generated_at_utc"), context=f"{context}.generated_at_utc"
    )
    _path(acceptance.get("output_root"), context=f"{context}.output_root")
    training_commit = _git_revision(
        acceptance.get("training_commit"), context=f"{context}.training_commit"
    )
    variants = _mapping(acceptance.get("variants"), context=f"{context}.variants")
    if set(variants) != set(VARIANT_ORDER):
        raise _error(context, "variants must contain exactly c,d,f,g1,g2,g3")
    for variant in VARIANT_ORDER:
        item = _mapping(variants[variant], context=f"{context}.{variant}")
        run = training[variant]
        combined = evaluations[variant]["f_plus_g"]
        _assert_exact(
            item,
            {
                "checkpoint_path": run.checkpoint_path,
                "checkpoint_sha256": combined.checkpoint_sha256,
                "checkpoint_epoch": TRAINING_EPOCHS,
                "checkpoint_global_step": TRAINING_STEPS,
                "action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
            },
            context=f"{context}.{variant}",
        )
        if run.training_commit != training_commit:
            raise _error(context, f"{variant} training commit differs from acceptance")
        _path(
            item.get("training_result_path"),
            context=f"{context}.{variant}.training_result_path",
        )
        _path(
            item.get("training_manifest_path"),
            context=f"{context}.{variant}.training_manifest_path",
        )
        expected_hashes = run.source_files_sha256
        if (
            item.get("training_result_sha256")
            != expected_hashes["training_result.json"]
        ):
            raise _error(
                context, f"{variant} training_result_sha256 disagrees with bundle"
            )
        if (
            item.get("training_manifest_sha256")
            != expected_hashes["training_manifest.json"]
        ):
            raise _error(
                context, f"{variant} training_manifest_sha256 disagrees with bundle"
            )
        metrics_files = item.get("metrics_files")
        if not isinstance(metrics_files, list) or not metrics_files:
            raise _error(context, f"{variant}.metrics_files must be a non-empty list")
        metric_hashes = set()
        for index, entry in enumerate(metrics_files):
            mapping = _mapping(
                entry, context=f"{context}.{variant}.metrics_files[{index}]"
            )
            _path(
                mapping.get("path"),
                context=f"{context}.{variant}.metrics_files[{index}].path",
            )
            metric_hashes.add(
                _sha256(
                    mapping.get("sha256"),
                    context=f"{context}.{variant}.metrics_files[{index}].sha256",
                )
            )
        if expected_hashes["metrics.csv"] not in metric_hashes:
            raise _error(
                context, f"{variant} metrics_files do not bind bundle metrics.csv"
            )
        epochs = item.get("metrics_epochs")
        if epochs not in (TRAINING_EPOCHS, list(range(1, TRAINING_EPOCHS + 1))):
            raise _error(context, f"{variant}.metrics_epochs must cover epochs 1..10")
        process = _mapping(item.get("process"), context=f"{context}.{variant}.process")
        _positive_int(process.get("pid"), context=f"{context}.{variant}.process.pid")
        if process.get("pid") != run.execution_evidence["process"]["pid"]:
            raise _error(
                context,
                f"{variant} process PID differs from external execution evidence",
            )
        if process.get("state") not in ("completed", "exited"):
            raise _error(context, f"{variant} process is not complete")
        if process.get("exit_code") not in (0, None):
            raise _error(context, f"{variant} process exit_code indicates failure")
        if process.get("exit_code") is None:
            _nonempty_string(
                process.get("exit_code_evidence"),
                context=f"{context}.{variant}.process.exit_code_evidence",
            )
    disk = _mapping(acceptance.get("disk"), context=f"{context}.disk")
    _positive_int(
        disk.get("outputs_bytes"),
        context=f"{context}.disk.outputs_bytes",
        allow_zero=True,
    )
    free_bytes = _positive_int(
        disk.get("free_bytes"), context=f"{context}.disk.free_bytes", allow_zero=True
    )
    minimum = _positive_int(
        disk.get("min_free_bytes"),
        context=f"{context}.disk.min_free_bytes",
        allow_zero=True,
    )
    if free_bytes < minimum:
        raise _error(context, "recorded free disk is below min_free_bytes")
    return acceptance, raw


def validate_bundle(bundle_root: str | Path) -> ValidatedStudy:
    """Validate a complete V1 six-method by three-score-mode formal bundle."""

    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise _error("bundle", f"not a directory: {root}")
    training: dict[str, TrainingRun] = {}
    evaluations: dict[str, dict[str, EvaluationRun]] = {}
    common_selection: Mapping[str, Any] | None = None
    for variant in VARIANT_ORDER:
        variant_root = root / variant
        training[variant] = _validate_training(variant_root, variant=variant)
        evaluations[variant] = {}
        for score_mode in SCORE_MODES:
            run = _validate_evaluation(
                variant_root / score_mode,
                variant=variant,
                score_mode=score_mode,
                training=training[variant],
            )
            selection = run.manifest["selection"]
            if common_selection is None:
                common_selection = selection
            elif selection != common_selection:
                raise _error(
                    f"{variant}/{score_mode}",
                    "all 18 evaluations must use the same selection",
                )
            evaluations[variant][score_mode] = run
    assert common_selection is not None
    _common_training_fingerprints(training)
    _common_evaluation_fingerprints(evaluations)
    acceptance, acceptance_raw = _validate_training_acceptance(
        root, training=training, evaluations=evaluations
    )
    return ValidatedStudy(
        bundle_root=root,
        training=training,
        evaluations=evaluations,
        selection=common_selection,
        selection_sha256=SELECTION_SHA256,
        training_acceptance=acceptance,
        training_acceptance_sha256=_sha256_bytes(acceptance_raw),
    )


def _pair_hash(episode: int, start: int, goal: int) -> str:
    return _fingerprint(
        {"episode_index": episode, "start_step": start, "goal_step": goal}
    )


def _ranking(study: ValidatedStudy) -> list[dict[str, Any]]:
    rows = [
        {
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "success_count": study.evaluations[variant]["f_plus_g"].success_count,
            "success_rate": study.evaluations[variant]["f_plus_g"].success_rate,
        }
        for variant in VARIANT_ORDER
    ]
    rows.sort(
        key=lambda row: (-row["success_count"], VARIANT_ORDER.index(row["variant"]))
    )
    previous_count: int | None = None
    rank = 0
    for position, row in enumerate(rows, 1):
        if row["success_count"] != previous_count:
            rank = position
            previous_count = row["success_count"]
        row["rank"] = rank
    return rows


def build_summary(study: ValidatedStudy) -> dict[str, Any]:
    """Build the deterministic machine-readable V1 study summary."""

    methods: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        evaluations = study.evaluations[variant]
        loss_curve = []
        for raw_row in training.curve:
            row = dict(raw_row)
            # Explicit aliases keep downstream document builders readable while
            # retaining the compact archive CSV column names.
            row["train_method_objective"] = row["train_method_loss"]
            row["validation_common_base_td"] = row["validation_base_td_loss"]
            loss_curve.append(row)
        method_spec = {
            "network": METHOD_SPECS[variant]["network"],
            "loss": METHOD_SPECS[variant]["loss"],
            "special": METHOD_SPECS[variant]["special"],
            "inference": (
                "F-only: LeWM horizon 5; G-only: predictor goal projection "
                "horizon 1; F+G: LeWM prefix plus predictor tail horizon 5"
            ),
        }
        methods[variant] = {
            "method": training.method,
            "display_name": DISPLAY_NAMES[variant],
            "network": METHOD_SPECS[variant]["network"],
            "training_loss": METHOD_SPECS[variant]["loss"],
            "special_mechanism": METHOD_SPECS[variant]["special"],
            "method_spec": method_spec,
            "inference": {
                "f_only": "LeWM final predicted-latent goal cost, horizon 5",
                "g_only": "Negative goal projection of the V1 predictor, horizon 1",
                "f_plus_g": "LeWM F prefix plus discounted terminal V1 G tail, horizon 5",
            },
            "training": {
                "seed": TRAINING_SEED,
                "epochs": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
                "checkpoint_path": training.checkpoint_path,
                "training_commit": training.training_commit,
                "protocol_canonical_sha256": training.protocol_sha256,
                "source_files_sha256": dict(training.source_files_sha256),
                "loss_curve": loss_curve,
                "final_epoch": dict(training.final_metrics),
                "best_validation": dict(training.best_validation),
                "train_loss_semantics": "method_specific_objective",
                "validation_loss_semantics": "common_base_td",
                "provenance": deepcopy(dict(training.provenance)),
            },
            "evaluations": {
                mode: {
                    "score_mode": mode,
                    "planning_horizon": evaluations[mode].planning_horizon,
                    "successes": evaluations[mode].success_count,
                    "success_count": evaluations[mode].success_count,
                    "success_rate": evaluations[mode].success_rate,
                    "elapsed_seconds": evaluations[mode].elapsed_seconds,
                    "checkpoint_path": evaluations[mode].checkpoint_path,
                    "checkpoint_sha256": evaluations[mode].checkpoint_sha256,
                    "evaluation_commit": evaluations[mode].evaluation_commit,
                    "formal_protocol_canonical_sha256": evaluations[
                        mode
                    ].formal_protocol_sha256,
                    "configured_protocol_canonical_sha256": evaluations[
                        mode
                    ].configured_protocol_sha256,
                    "source_files_sha256": dict(evaluations[mode].source_files_sha256),
                }
                for mode in SCORE_MODES
            },
            "combined_minus_f_only_percentage_points": 100.0
            * (
                evaluations["f_plus_g"].success_rate
                - evaluations["f_only"].success_rate
            ),
        }
    ranking = _ranking(study)
    return {
        "schema_version": SCHEMA_VERSION,
        "study": {
            "id": "actor_free_td_lewm_v1_cube_seed3072_o50_6x3",
            "method_family": METHOD_FAMILY,
            "environment": "cube",
            "training_seed": TRAINING_SEED,
            "planning_seed": PLANNING_SEED,
            "goal_offset": GOAL_OFFSET,
            "episodes_per_evaluation": EPISODES,
            "training_count": len(VARIANT_ORDER),
            "evaluation_count": len(VARIANT_ORDER) * len(SCORE_MODES),
            "score_modes": list(SCORE_MODES),
            "formal_horizon_by_score_mode": dict(FORMAL_HORIZON_BY_SCORE_MODE),
            "ranking_metric": "f_plus_g_success_count_only",
            "single_training_seed": True,
            "single_planning_selection": True,
        },
        "architecture": {
            "lewm_frozen": True,
            "shared_action_encoder_frozen": True,
            "action_processing": "raw25_to_frozen_world_model.action_encoder_to_embedding192",
            "state_dim": 192,
            "raw_action_dim": 25,
            "action_embedding_dim": 192,
            "task_dim": 192,
            "output_dim": 192,
            "world_model_parameters": WORLD_MODEL_PARAMETERS,
            "trainable_lewm_parameters": 0,
            "predictor_parameters": PREDICTOR_PARAMETERS,
            "pretrained_world_model_sha256": PRETRAINED_WORLD_MODEL_SHA256,
            "action_encoder_state_sha256": EXPECTED_ACTION_ENCODER_SHA256,
            "actor": "none",
            "reward": "none",
        },
        "selection": {
            "episode_selection_json_sha256": study.selection_sha256,
            "episode_count": EPISODES,
        },
        "training_acceptance": {
            "status": study.training_acceptance["status"],
            "warnings": list(study.training_acceptance.get("warnings", [])),
            "sha256": study.training_acceptance_sha256,
            "expected_action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
        },
        "ranking_by_f_plus_g": ranking,
        # Kept as a deterministic alias for the existing Results TD builder;
        # the study-level ranking_metric above remains the normative name.
        "ranking_by_combined": deepcopy(ranking),
        "methods": methods,
        "validation": {
            "complete_6x3_bundle": True,
            "formal_o50_only": True,
            "smoke_or_pilot_runs": 0,
            "common_selection_across_18_runs": True,
            "locked_seed42_selection_sha256": True,
            "mode_specific_horizon_5_1_5": True,
            "same_checkpoint_within_each_method": True,
            "frozen_lewm_and_action_encoder": True,
            "no_lewm_training_loss": True,
            "training_metrics_derived_from_raw_lightning_csv": True,
            "success_rates_match_episode_outcomes": True,
            "missing_trainer_provenance_not_fabricated": True,
            "external_training_acceptance_bound": True,
        },
    }


def build_paired_outcomes_csv(study: ValidatedStudy) -> bytes:
    stream = io.StringIO(newline="")
    success_columns = [
        f"success_{variant}__{mode}"
        for variant in VARIANT_ORDER
        for mode in SCORE_MODES
    ]
    fields = [
        "selection_position",
        "selection_sha256",
        "episode_index",
        "start_step",
        "goal_step",
        "valid_row_rank",
        "pair_hash",
        *success_columns,
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for index in range(EPISODES):
        episode = study.selection["episode_indices"][index]
        start = study.selection["start_steps"][index]
        goal = study.selection["goal_steps"][index]
        row: dict[str, Any] = {
            "selection_position": index,
            "selection_sha256": study.selection_sha256,
            "episode_index": episode,
            "start_step": start,
            "goal_step": goal,
            "valid_row_rank": study.selection["valid_row_ranks"][index],
            "pair_hash": _pair_hash(episode, start, goal),
        }
        for variant in VARIANT_ORDER:
            for mode in SCORE_MODES:
                row[f"success_{variant}__{mode}"] = str(
                    study.evaluations[variant][mode].successes[index]
                ).lower()
        writer.writerow(row)
    return stream.getvalue().encode()


def build_training_curves_csv(study: ValidatedStudy) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "variant",
        "method",
        "display_name",
        "epoch",
        "train_method_loss",
        "train_base_td_loss",
        "validation_base_td_loss",
        "train_metric_semantics",
        "validation_metric_semantics",
        "cross_method_ranking_metric",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for variant in VARIANT_ORDER:
        run = study.training[variant]
        for item in run.curve:
            writer.writerow(
                {
                    "variant": variant,
                    "method": run.method,
                    "display_name": DISPLAY_NAMES[variant],
                    "epoch": item["epoch"],
                    "train_method_loss": f"{item['train_method_loss']:.12g}",
                    "train_base_td_loss": f"{item['train_base_td_loss']:.12g}",
                    "validation_base_td_loss": f"{item['validation_base_td_loss']:.12g}",
                    "train_metric_semantics": "method_specific_objective",
                    "validation_metric_semantics": "common_base_td",
                    "cross_method_ranking_metric": "false_use_formal_o50_f_plus_g",
                }
            )
    return stream.getvalue().encode()


def build_training_curves_svg(study: ValidatedStudy) -> bytes:
    width, height = 1280, 620
    left, top, plot_width, plot_height = 82, 108, 520, 390
    second_left = 678
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

    train_values = [
        float(item["train_method_loss"])
        for variant in VARIANT_ORDER
        for item in study.training[variant].curve
    ]
    validation_values = [
        float(item["validation_base_td_loss"])
        for variant in VARIANT_ORDER
        for item in study.training[variant].curve
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:24px;font-weight:700}.sub{font-size:14px;fill:#58656f}.panel{font-size:17px;font-weight:700}.legend{font-size:13px}</style>",
        '<text class="title" x="640" y="36" text-anchor="middle">Actor-Free TD-JEPA V1 loss diagnostics</text>',
        '<text class="sub" x="640" y="62" text-anchor="middle">Train = method-specific objective; validation = common base TD. Formal O50 F+G is the ranking metric.</text>',
        f'<text class="panel" x="{left + plot_width / 2}" y="91" text-anchor="middle">Train method objective</text>',
        f'<text class="panel" x="{second_left + plot_width / 2}" y="91" text-anchor="middle">Validation common base TD</text>',
    ]
    for x0 in (left, second_left):
        parts.append(
            f'<rect x="{x0}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#b0bec5"/>'
        )
        for epoch in range(1, 11):
            x = x0 + plot_width * (epoch - 1) / 9
            parts.append(
                f'<text x="{x:.2f}" y="522" text-anchor="middle" font-size="12">{epoch}</text>'
            )
    for index, variant in enumerate(VARIANT_ORDER):
        color = colors[index]
        curve = study.training[variant].curve
        train = [float(item["train_method_loss"]) for item in curve]
        validation = [float(item["validation_base_td_loss"]) for item in curve]

        # Normalize each panel using all methods, while retaining exact values in CSV.
        def shared_points(values: list[float], all_values: list[float], x0: int) -> str:
            minimum, maximum = min(all_values), max(all_values)
            span = maximum - minimum or 1.0
            return " ".join(
                f"{x0 + plot_width * i / 9:.2f},{top + plot_height * (maximum - value) / span:.2f}"
                for i, value in enumerate(values)
            )

        parts.append(
            f'<polyline points="{shared_points(train, train_values, left)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<polyline points="{shared_points(validation, validation_values, second_left)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        legend_x = 145 + index * 190
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="566" x2="{legend_x + 26}" y2="566" stroke="{color}" stroke-width="3"/>',
                f'<text class="legend" x="{legend_x + 33}" y="571">{html.escape(variant.upper())}</text>',
            ]
        )
    parts.extend(
        [
            '<text class="sub" x="640" y="603" text-anchor="middle">Curves diagnose convergence; their heights are not a cross-method performance ranking.</text>',
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode()


def _percent(run: EvaluationRun) -> str:
    return f"{100.0 * run.success_rate:.0f}% ({run.success_count}/{EPISODES})"


def build_markdown_report(study: ValidatedStudy) -> bytes:
    ranking = _ranking(study)
    missing_trainer_provenance = [
        variant
        for variant in VARIANT_ORDER
        if study.training[variant].provenance["peak_cuda_memory_bytes"]["status"]
        == "not_recorded_by_v1_trainer"
        or study.training[variant].provenance["runtime.cuda_device"]["status"]
        == "not_recorded_by_v1_trainer"
    ]
    lines = [
        "# Results TD — Actor-Free TD-JEPA V1 Cube O50",
        "",
        "本报告由 6 个训练产物与 18 个正式 O50 evaluator 原始输出自动生成。",
        "排名只使用预先定义的 **F+G**，不从三种推理评分中事后取最好值。",
        "",
        "## 正式结果",
        "",
        "| Rank | Method | F-only | G-only | F+G | Δ F+G − F-only |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    rank_by_variant = {item["variant"]: item["rank"] for item in ranking}
    for variant in VARIANT_ORDER:
        runs = study.evaluations[variant]
        delta = 100.0 * (runs["f_plus_g"].success_rate - runs["f_only"].success_rate)
        lines.append(
            f"| {rank_by_variant[variant]} | {DISPLAY_NAMES[variant]} | {_percent(runs['f_only'])} | "
            f"{_percent(runs['g_only'])} | {_percent(runs['f_plus_g'])} | {delta:+.0f} pp |"
        )
    lines.extend(
        [
            "",
            "## 方法、网络、损失与推理",
            "",
            "所有方法共享 **frozen LeWM** 和同一个 frozen shared LeWM action encoder。"
            "训练只更新一个 379,072 参数的 goal-conditioned TD-JEPA predictor；没有 Actor、"
            "没有 reward loss，也没有 LeWM reconstruction/prediction loss。",
            "",
            "共同 feature TD target 为 `s_next + gamma * (1-terminal) * EMA-G(s_next, E_A(a_next), task)`；"
            "数据集 next action 经冻结的 25D→192D LeWM action encoder 后参与 bootstrap。",
            "",
            "| Method | Network | Training loss | Special mechanism | Inference |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for variant in VARIANT_ORDER:
        spec = METHOD_SPECS[variant]
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {spec['network']} | {spec['loss']} | "
            f"{spec['special']} | F-only horizon 5; G-only horizon 1; F+G horizon 5 |"
        )
    lines.extend(
        [
            "",
            "## Loss 曲线语义",
            "",
            "`train/loss` 是每个方法自己的训练 objective；`validation/loss` 对六个方法统一为 common base TD。"
            "训练曲线定义不同，不能按高低做跨方法排名；正式 O50 F+G 才是本表排名依据。",
            "",
            "![V1 loss diagnostics](artifacts/actor_free_td_lewm_v1_cube_seed3072/training_loss_curves.svg)",
            "",
            "| Method | Epoch-10 train method loss | Epoch-10 train base TD | Epoch-10 validation base TD | Best validation | Checkpoint SHA-256 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        final = training.final_metrics
        best = training.best_validation
        checkpoint_sha = study.evaluations[variant]["f_plus_g"].checkpoint_sha256
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {final['train_method_loss']:.6f} | "
            f"{final['train_base_td_loss']:.6f} | {final['validation_base_td_loss']:.6f} | "
            f"{best['value']:.6f} (E{best['epoch']}) | `{checkpoint_sha[:12]}…` |"
        )
    lines.extend(
        [
            "",
            "## 审计结论与边界",
            "",
            f"- 18 个运行共享固定 selection SHA-256：`{study.selection_sha256}`。",
            "- score-mode horizon 锁为 F-only=5、G-only=1、F+G=5；其余正式 CEM 参数一致。",
            f"- frozen shared action encoder canonical state hash：`{EXPECTED_ACTION_ENCODER_SHA256}`。",
            "- 每个方法的三种 score mode 使用同一 epoch-10 checkpoint；success rate 均由 50 个逐 episode 布尔值重算。",
            "- 只有一个 training seed 和一组 planning selection；结果适合作为结构消融，不能声称多随机种子总体最优。",
        ]
    )
    if missing_trainer_provenance:
        lines.append(
            "- 当前 V1 trainer 未记录 `peak_cuda_memory_bytes` 与/或 `runtime.cuda_device` "
            f"（{', '.join(missing_trainer_provenance)}）。归档未补造；缺失项明确标为 "
            "`not_recorded_by_v1_trainer`，GPU/PID/命令/日志来源来自独立 execution evidence。"
        )
    for warning in study.training_acceptance.get("warnings", []):
        lines.append(f"- Training acceptance warning: {warning}")
    return ("\n".join(lines) + "\n").encode()


def build_archive_readme(study: ValidatedStudy) -> bytes:
    text = f"""# Actor-Free TD-JEPA V1 Cube O50 machine-readable archive

This directory is generated from a validated six-method by three-score-mode
bundle.  It contains no checkpoints, datasets, videos, or full console logs.

- `summary.json`: methods, 18 formal scores, protocol/checkpoint/source hashes,
  loss semantics, and explicit trainer provenance gaps.
- `paired_outcomes.csv`: the locked 50 start-goal pairs and 18 success columns.
- `training_loss_curves.csv`: 6 x 10 epoch diagnostics.  Train is each method's
  objective; validation is the common base TD.
- `training_loss_curves.svg`: deterministic report visualization.
- `checksums.sha256`: hashes for generated archive files and the Markdown report.

Locked selection SHA-256: `{study.selection_sha256}`.
Locked action-encoder state SHA-256: `{EXPECTED_ACTION_ENCODER_SHA256}`.

F-only and F+G use horizon 5.  G-only uses horizon 1.  Ranking is by F+G only.
Missing CUDA peak memory and trainer CUDA-device fields were not fabricated;
their external evidence hashes are recorded in `summary.json`.
"""
    return text.encode()


def _archive_payloads(
    study: ValidatedStudy, *, report_path: Path, artifact_dir: Path
) -> tuple[dict[str, bytes], bytes]:
    summary = (
        json.dumps(build_summary(study), indent=2, sort_keys=True) + "\n"
    ).encode()
    paired = build_paired_outcomes_csv(study)
    curves = build_training_curves_csv(study)
    svg = build_training_curves_svg(study)
    readme = build_archive_readme(study)
    report = build_markdown_report(study)
    payloads = {
        "README.md": readme,
        "summary.json": summary,
        "paired_outcomes.csv": paired,
        "training_loss_curves.csv": curves,
        "training_loss_curves.svg": svg,
    }
    checksum_lines = [
        f"{_sha256_bytes(payloads[name])}  {name}" for name in sorted(payloads)
    ]
    report_relative = os.path.relpath(report_path, artifact_dir)
    checksum_lines.append(f"{_sha256_bytes(report)}  {report_relative}")
    payloads["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode()
    return payloads, report


def write_archive(
    study: ValidatedStudy,
    *,
    artifact_dir: str | Path,
    report_path: str | Path,
    check: bool = False,
) -> list[Path]:
    """Write, or byte-check, the deterministic V1 archive."""

    destination = Path(artifact_dir)
    report = Path(report_path)
    payloads, report_payload = _archive_payloads(
        study, report_path=report, artifact_dir=destination
    )
    paths = [report, *(destination / name for name in payloads)]
    if check:
        expected = {report: report_payload}
        expected.update(
            {destination / name: payload for name, payload in payloads.items()}
        )
        for path, payload in expected.items():
            if not path.is_file():
                raise _error("archive check", f"missing generated file: {path}")
            if path.read_bytes() != payload:
                raise _error("archive check", f"generated file differs: {path}")
        return paths
    destination.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(report_payload)
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    return paths


__all__ = [
    "BundleValidationError",
    "EXPECTED_ACTION_ENCODER_SHA256",
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "PREDICTOR_PARAMETERS",
    "SCORE_MODES",
    "SELECTION_SHA256",
    "VARIANT_ORDER",
    "ValidatedStudy",
    "build_markdown_report",
    "build_paired_outcomes_csv",
    "build_summary",
    "build_training_curves_csv",
    "build_training_curves_svg",
    "validate_bundle",
    "write_archive",
]
