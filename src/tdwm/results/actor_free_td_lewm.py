"""Validate and archive the fixed-seed Actor-Free TD-LeWM Cube O50 study.

The archive consumes the evaluator's JSON outputs and the trainer's original
``training_result.json``, ``training_manifest.json``, and Lightning
``metrics.csv``.  It computes source hashes and epoch summaries itself; no
hand-written result or curve summary is trusted.  Datasets, checkpoints,
videos, and console logs are never copied into the repository.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
METHOD = "actor_free_td_lewm"
TRAINING_SEED = 3072
TRAINING_EPOCHS = 10
TRAINING_STEPS = 127_960
EPISODES = 50
GOAL_OFFSET = 50
PLANNING_SEED = 42
ENVIRONMENT = "cube"
DATASET_IDENTIFIER = "quentinll/lewm-cube"
STABLE_WORLDMODEL_VERSION = "0.1.1"
SELECTION_SHA256 = "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
DATASET_EPISODES = 10_000
DATASET_TRANSITIONS = 2_010_000
EPISODE_STEPS = 201
OPTIMIZER_STEPS_PER_EPOCH = 12_796

VARIANT_ORDER = (
    "serial_decoupled",
    "serial_coupled",
    "hybrid",
    "parallel_real",
    "goal_hybrid",
    "imaginary_hybrid",
    "direct_goal_hybrid",
)
DIRECT_VARIANT = "direct_goal_hybrid"
SUCCESSOR_MODES = ("f_only", "g_only", "f_plus_g")
DIRECT_MODES = ("f_only", "c_only", "f_plus_c")

FORMAL_PLANNING = {
    "solver": "CEM",
    "horizon": 5,
    "candidates": 300,
    "iterations": 30,
    "elites": 30,
    "initial_variance": 1.0,
    "action_block": 5,
    "frame_skip": 5,
    "receding_horizon": 1,
    "episode_budget": 100,
    "planning_seed": PLANNING_SEED,
    "solver_batch_size": 1,
    "history_len": 1,
    "warm_start": True,
    "initial_distribution": "cem_gaussian_no_actor",
}

FORMAL_RUNTIME = {
    "stable_worldmodel_version": STABLE_WORLDMODEL_VERSION,
    "import": "import stable_worldmodel as swm",
    "precision": "fp32",
}
FORMAL_IMAGE_PREPROCESSING = {
    "source": "stable_pretraining.data.dataset_stats.ImageNet",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
FORMAL_DATASET = {
    "identifier": DATASET_IDENTIFIER,
    "file": "ogbench/cube_single_expert.h5",
    "expected_size_bytes": 101_942_558_720,
    "accepted_size_bytes": [101_942_558_720, 74_104_077_358],
    "expected_episodes": DATASET_EPISODES,
    "expected_transitions": DATASET_TRANSITIONS,
    "episode_steps": EPISODE_STEPS,
    "lance": {
        "manifest_suffix": ".manifest.json",
        "image_codec": "jpeg",
        "jpeg_quality": 100,
    },
    "keys_to_load": [
        "pixels",
        "action",
        "qpos",
        "qvel",
        "privileged_block_0_pos",
        "privileged_block_0_quat",
    ],
}
FORMAL_MODEL = {"embed_dim": 192}
FORMAL_WORLD = {
    "env_name": "swm/OGBCube-v0",
    "env_type": "single",
    "ob_type": "states",
    "image_size": 224,
    "multiview": False,
    "visualize_info": False,
    "terminate_at_goal": True,
    "success_threshold_meters": 0.04,
}
FORMAL_EVALUATION = {
    "episodes": EPISODES,
    "goal_offset": GOAL_OFFSET,
    "start_goal_source": "same_dataset_episode",
}
COMMON_PROTOCOL_SECTIONS = {
    "runtime": FORMAL_RUNTIME,
    "image_preprocessing": FORMAL_IMAGE_PREPROCESSING,
    "dataset": FORMAL_DATASET,
    "model": FORMAL_MODEL,
    "world": FORMAL_WORLD,
    "evaluation": FORMAL_EVALUATION,
    "planning": FORMAL_PLANNING,
}
RUNTIME_FINGERPRINT_KEYS = (
    "stable_worldmodel",
    "torch",
    "python",
    "platform",
    "device",
    "cuda_device",
    "compatibility_adapter",
)
TRAINING_COMMON_PROTOCOL_SECTIONS = (
    "runtime",
    "dataset",
    "split",
    "sequence",
    "image_preprocessing",
    "normalization",
    "model",
    "loss",
    "loader",
    "logging",
    "scheduler",
    "training",
)
OBJECTIVE_VERSIONS = {
    "serial_decoupled": 1,
    "serial_coupled": 1,
    "hybrid": 1,
    "parallel_real": 1,
    "goal_hybrid": 2,
    "imaginary_hybrid": 3,
    "direct_goal_hybrid": 3,
}
TRAINING_PROTOCOL_SHA256 = {
    "serial_decoupled": "6eaaf266bc6e303f5b72b1858925fb761d516673d7a8235d33b109fb206dcdc2",
    "serial_coupled": "e272feeb5081253d732dd761593a80750582dbaabdc6b0bff4d56d2b10497d6b",
    "hybrid": "bd0b207e27126d5534f137016a69a8402521522df60826b9bd442395484a13a9",
    "parallel_real": "f3d3ca31e1f1b6405f02f63e0b11025d273cc567ff908599347b4be79c0e4fec",
    "goal_hybrid": "fdbf618cb45e8b856f1914df752e3aa44c741b579ec88191b8f92d3785a491be",
    "imaginary_hybrid": "6e43c9f75351537bc5bec32c5f88ec47f295d9f2d943c8c9294d79423bfc7340",
    "direct_goal_hybrid": "f9e2ee091aee487bcef521390f5032d586b7e1ffa98033412fdca30f7293a471",
}

DISPLAY_NAMES = {
    "serial_decoupled": "Serial Decoupled",
    "serial_coupled": "Serial Coupled",
    "hybrid": "Hybrid",
    "parallel_real": "Parallel Real",
    "goal_hybrid": "Goal Hybrid",
    "imaginary_hybrid": "Imaginary Hybrid",
    "direct_goal_hybrid": "Direct Goal Critic Hybrid",
}

METHOD_SPECS = {
    "serial_decoupled": {
        "network": "LeWM + one successor-feature head on predicted latent history",
        "loss": "L_LeWM + alpha_u L_TD^pred",
        "special": "Predicted context is detached; TD updates the successor head only.",
        "inference": "CEM with F-only, G-only, or F+G cost.",
    },
    "serial_coupled": {
        "network": "LeWM + one successor-feature head on predicted latent history",
        "loss": "L_LeWM + alpha_u L_TD^pred",
        "special": "Predicted context stays differentiable; TD also reaches LeWM.",
        "inference": "CEM with F-only, G-only, or F+G cost.",
    },
    "hybrid": {
        "network": "LeWM + one shared successor head for real and predicted histories",
        "loss": "L_LeWM + alpha_u (L_TD^real + L_TD^pred)",
        "special": "The same head is trained on parallel-real and coupled-serial branches.",
        "inference": "CEM with F-only, G-only, or F+G cost.",
    },
    "parallel_real": {
        "network": "LeWM predictor and successor head are parallel on encoder latents",
        "loss": "L_LeWM + alpha_u L_TD^real",
        "special": "TD uses real latent history and does not pass through the predictor.",
        "inference": "CEM with F-only, G-only, or F+G cost.",
    },
    "goal_hybrid": {
        "network": "Hybrid successor head with fixed linear goal readout G^T w(g)",
        "loss": (
            "L_LeWM + alpha_u (L_SF-TD^real + L_SF-TD^pred + "
            "L_goal-TD^real + L_goal-TD^pred)"
        ),
        "special": "Goal readout is trained by hindsight goal-conditioned Bellman TD.",
        "inference": "CEM with F-only, G-only, or F+G cost.",
    },
    "imaginary_hybrid": {
        "network": "Hybrid successor head with an EMA-LeWM imagined bootstrap state",
        "loss": "L_LeWM + alpha_u (L_TD^real + L_TD^pred)",
        "special": "The TD target bootstraps one step through the stopped EMA predictor.",
        "inference": "CEM with F-only, G-only, or F+G cost.",
    },
    "direct_goal_hybrid": {
        "network": "LeWM + one scalar goal-conditioned critic for real/predicted histories",
        "loss": "L_LeWM + alpha_u (L_C-TD^real + L_C-TD^pred)",
        "special": "Goal latent enters the critic directly; there is no SF factorization.",
        "inference": "CEM with F-only, C-only, or F+C cost.",
    },
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


class BundleValidationError(ValueError):
    """Raised when an input bundle is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class EvaluationRun:
    variant: str
    score_mode: str
    score_mode_source: str
    successes: tuple[bool, ...]
    success_rate: float
    elapsed_seconds: float
    checkpoint_sha256: str
    selection: Mapping[str, tuple[int, ...]]
    selection_sha256: str
    formal_protocol_sha256: str
    evaluation_commit: str
    runtime: Mapping[str, Any]
    dataset: Mapping[str, Any]
    normalization: Mapping[str, Any]
    checkpoint_path: str
    common_protocol_sha256: str
    runtime_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    normalization_fingerprint_sha256: str
    world_model_parameter_count: int
    head_parameter_count: int
    source_sha256: Mapping[str, str]


@dataclass(frozen=True)
class TrainingRun:
    variant: str
    training_commit: str
    run_dir: str
    last_checkpoint: str
    deployment_checkpoint: str
    runtime: Mapping[str, Any]
    dataset: Mapping[str, Any]
    model: Mapping[str, Any]
    metrics: Mapping[str, Any]
    source_file_sha256: Mapping[str, str]
    world_model_parameter_count: int
    head_parameter_count: int
    common_protocol_sha256: str
    locked_protocol_sha256: str
    runtime_fingerprint_sha256: str
    dataset_source_fingerprint_sha256: str
    curve: tuple[Mapping[str, float | int], ...]


@dataclass(frozen=True)
class ValidatedStudy:
    training: Mapping[str, TrainingRun]
    evaluations: Mapping[str, Mapping[str, EvaluationRun]]
    selection: Mapping[str, tuple[int, ...]]
    selection_sha256: str


def modes_for_variant(variant: str) -> tuple[str, str, str]:
    if variant not in VARIANT_ORDER:
        raise BundleValidationError(f"Unknown variant {variant!r}.")
    return DIRECT_MODES if variant == DIRECT_VARIANT else SUCCESSOR_MODES


def combined_mode_for_variant(variant: str) -> str:
    return "f_plus_c" if variant == DIRECT_VARIANT else "f_plus_g"


def _error(context: str, message: str) -> BundleValidationError:
    return BundleValidationError(f"{context}: {message}")


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise _error(context, f"missing required file {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(context, f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(context, f"{path} must contain a JSON object")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(context, "must be a JSON object")
    return value


def _require_bool(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise _error(context, "must be a JSON boolean")
    return value


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(context, "must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _error(context, "must be a finite number")
    return number


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(context, "must be a positive integer")
    return value


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise _error(context, "must be a lowercase 64-character SHA-256")
    return value


def _require_git_revision(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _GIT_PATTERN.fullmatch(value) is None:
        raise _error(context, "must be a 7-to-40 character lowercase Git revision")
    return value


def _normalize_success_rate(value: Any, *, context: str) -> float:
    number = _finite_number(value, context=context)
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    raise _error(context, "must be a fraction in [0,1] or a percent in (1,100]")


def _success_vector(metrics: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    def normalize(values: Any, field: str) -> tuple[bool, ...]:
        if not isinstance(values, list):
            raise _error(context, f"metrics.{field} must be a JSON array")
        if len(values) != EPISODES:
            raise _error(
                context,
                f"metrics.{field} must contain exactly {EPISODES} rows",
            )
        normalized: list[bool] = []
        for index, value in enumerate(values):
            if type(value) is bool:
                normalized.append(value)
            elif isinstance(value, int) and value in (0, 1):
                normalized.append(bool(value))
            else:
                raise _error(
                    context,
                    f"metrics.{field}[{index}] is not boolean/0/1",
                )
        return tuple(normalized)

    canonical = (
        normalize(metrics["episode_successes"], "episode_successes")
        if "episode_successes" in metrics
        else None
    )
    legacy = normalize(metrics["success"], "success") if "success" in metrics else None
    if canonical is None and legacy is None:
        raise _error(
            context,
            "metrics.episode_successes is required (legacy metrics.success is accepted)",
        )
    if canonical is not None and legacy is not None and canonical != legacy:
        raise _error(
            context,
            "metrics.episode_successes and legacy metrics.success disagree",
        )
    if canonical is not None:
        return canonical
    assert legacy is not None
    return legacy


def _selection(value: Any, *, context: str) -> dict[str, tuple[int, ...]]:
    source = _require_mapping(value, context=context)
    required = ("episode_indices", "start_steps", "goal_steps", "valid_row_ranks")
    normalized: dict[str, tuple[int, ...]] = {}
    for key in required:
        values = source.get(key)
        if not isinstance(values, list) or len(values) != EPISODES:
            raise _error(context, f"{key} must contain exactly {EPISODES} integers")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise _error(context, f"{key} must contain only integers")
        normalized[key] = tuple(values)
    if any(item < 0 for key in required for item in normalized[key]):
        raise _error(context, "selection values must be non-negative")
    if any(item >= DATASET_EPISODES for item in normalized["episode_indices"]):
        raise _error(context, f"episode_indices must lie in [0,{DATASET_EPISODES})")
    for index, (start, goal) in enumerate(
        zip(normalized["start_steps"], normalized["goal_steps"])
    ):
        if not 0 <= start < goal < EPISODE_STEPS:
            raise _error(
                context,
                f"selection row {index} must satisfy 0 <= start < goal < {EPISODE_STEPS}",
            )
    if any(
        goal != start + GOAL_OFFSET
        for start, goal in zip(
            normalized["start_steps"], normalized["goal_steps"]
        )
    ):
        raise _error(context, f"every goal step must equal start + {GOAL_OFFSET}")
    ranks = normalized["valid_row_ranks"]
    if list(ranks) != sorted(ranks) or len(set(ranks)) != EPISODES:
        raise _error(context, "valid_row_ranks must be unique and sorted")
    pairs = list(
        zip(
            normalized["episode_indices"],
            normalized["start_steps"],
            normalized["goal_steps"],
        )
    )
    if len(set(pairs)) != EPISODES:
        raise _error(context, "the O50 selection contains duplicate start-goal pairs")

    valid_per_episode = EPISODE_STEPS - GOAL_OFFSET
    rng = np.random.default_rng(PLANNING_SEED)
    expected_ranks = np.sort(
        rng.choice(
            DATASET_EPISODES * valid_per_episode - 1,
            size=EPISODES,
            replace=False,
        )
    )
    expected_episodes = expected_ranks // valid_per_episode
    expected_starts = expected_ranks % valid_per_episode
    expected = {
        "episode_indices": tuple(int(item) for item in expected_episodes),
        "start_steps": tuple(int(item) for item in expected_starts),
        "goal_steps": tuple(int(item + GOAL_OFFSET) for item in expected_starts),
        "valid_row_ranks": tuple(int(item) for item in expected_ranks),
    }
    if normalized != expected:
        raise _error(
            context,
            "selection is not the StableWM 0.1.1 Cube seed-42 O50 sample",
        )
    return normalized


def _pair_hash(episode_index: int, start_step: int, goal_step: int) -> str:
    payload = {
        "episode_index": episode_index,
        "goal_step": goal_step,
        "start_step": start_step,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _parse_csv_integer(value: Any, *, context: str) -> int:
    number = _parse_csv_number(value)
    if not math.isfinite(number) or not number.is_integer():
        raise _error(context, "must be an integer")
    return int(number)


def _has_csv_value(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _validate_training_metrics(
    path: Path, variant: str
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    context = f"{variant}/metrics.csv"
    if not path.is_file():
        raise _error(context, f"missing required file {path}")
    train_by_epoch: dict[int, tuple[float, int]] = {}
    validation_by_epoch: dict[int, tuple[float, int]] = {}
    observed_steps: list[int] = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"epoch", "step", "train/loss_epoch", "validation/loss"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise _error(context, f"missing columns {sorted(missing)}")
            for position, source in enumerate(reader, start=1):
                tracked = (
                    _has_csv_value(source.get("train/loss_epoch"))
                    or _has_csv_value(source.get("validation/loss"))
                )
                if not _has_csv_value(source.get("step")):
                    if tracked:
                        raise _error(context, f"row {position} aggregate has no step")
                    continue
                step = _parse_csv_integer(
                    source.get("step"), context=f"{context} row {position} step"
                )
                if step < 0:
                    raise _error(context, f"row {position} step must be non-negative")
                observed_steps.append(step)
                if not tracked:
                    continue
                epoch = _parse_csv_integer(
                    source.get("epoch"), context=f"{context} row {position} epoch"
                )
                if not 0 <= epoch < TRAINING_EPOCHS:
                    raise _error(context, f"row {position} epoch must lie in [0,9]")
                for field, destination in (
                    ("train/loss_epoch", train_by_epoch),
                    ("validation/loss", validation_by_epoch),
                ):
                    if not _has_csv_value(source.get(field)):
                        continue
                    value = _finite_number(
                        _parse_csv_number(source[field]),
                        context=f"{context} row {position} {field}",
                    )
                    if value < 0.0:
                        raise _error(context, f"row {position} {field} is negative")
                    if epoch in destination:
                        raise _error(
                            context,
                            f"epoch {epoch} contains more than one {field} aggregate",
                        )
                    destination[epoch] = (value, step)
    except OSError as exc:
        raise _error(context, f"cannot read {path}: {exc}") from exc
    expected_epochs = set(range(TRAINING_EPOCHS))
    if set(train_by_epoch) != expected_epochs:
        raise _error(context, "must contain exactly one train/loss_epoch for epochs 0..9")
    if set(validation_by_epoch) != expected_epochs:
        raise _error(context, "must contain exactly one validation/loss for epochs 0..9")
    if not observed_steps or max(observed_steps) != TRAINING_STEPS - 1:
        raise _error(
            context,
            f"maximum CSV step must equal zero-based {TRAINING_STEPS - 1}",
        )
    curve: list[Mapping[str, Any]] = []
    for epoch in range(TRAINING_EPOCHS):
        expected_step = (epoch + 1) * OPTIMIZER_STEPS_PER_EPOCH - 1
        train_loss, train_step = train_by_epoch[epoch]
        validation_loss, validation_step = validation_by_epoch[epoch]
        if train_step != expected_step or validation_step != expected_step:
            raise _error(
                context,
                f"epoch {epoch} aggregates must use final step {expected_step}",
            )
        curve.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
    best = min(curve, key=lambda row: float(row["validation_loss"]))
    metrics = {
        "final_epoch": {
            "epoch": TRAINING_EPOCHS,
            "train/loss": curve[-1]["train_loss"],
            "validation/loss": curve[-1]["validation_loss"],
        },
        "best_validation": {
            "epoch": best["epoch"],
            "metric": "validation/loss",
            "value": best["validation_loss"],
        },
    }
    return tuple(curve), metrics


def _parse_csv_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _validate_dataset_provenance(
    dataset: Mapping[str, Any],
    *,
    context: str,
    require_embedded_conversion: bool = False,
) -> None:
    path = dataset.get("path")
    if not isinstance(path, str) or not path:
        raise _error(context, "dataset.path must be recorded")
    dataset_format = dataset.get("format")
    if dataset_format not in {"hdf5", "lance"}:
        raise _error(context, "dataset.format must be 'hdf5' or 'lance'")
    size = _positive_int(dataset.get("size_bytes"), context=f"{context}.size_bytes")
    conversion_path = dataset.get("conversion_manifest_path")
    conversion = dataset.get("conversion_manifest")
    if dataset_format == "hdf5":
        if size not in FORMAL_DATASET["accepted_size_bytes"]:
            raise _error(context, "HDF5 dataset size is not an accepted locked layout")
        if conversion_path is not None or conversion is not None:
            raise _error(context, "HDF5 dataset must not claim Lance conversion provenance")
        return
    if not isinstance(conversion_path, str) or not conversion_path.endswith(
        FORMAL_DATASET["lance"]["manifest_suffix"]
    ):
        raise _error(context, "Lance dataset must record its .manifest.json path")
    if require_embedded_conversion:
        embedded = _require_mapping(
            conversion, context=f"{context}.conversion_manifest"
        )
        destination = _require_mapping(
            embedded.get("destination"),
            context=f"{context}.conversion_manifest.destination",
        )
        conversion_details = _require_mapping(
            embedded.get("conversion"),
            context=f"{context}.conversion_manifest.conversion",
        )
        if destination.get("format") != "lance":
            raise _error(context, "conversion manifest destination must be Lance")
        if destination.get("size_bytes") != size:
            raise _error(context, "conversion manifest size differs from dataset size")
        if (
            conversion_details.get("stable_worldmodel_version")
            != STABLE_WORLDMODEL_VERSION
        ):
            raise _error(context, "conversion provenance requires StableWM 0.1.1")


def _validate_split_manifest(
    dataset: Mapping[str, Any], *, context: str
) -> dict[str, int | str]:
    split = _require_mapping(dataset.get("split"), context=f"{context}.split")
    _path_string(split.get("path"), context=f"{context}.split.path")
    train_samples = _positive_int(
        split.get("train_samples"), context=f"{context}.split.train_samples"
    )
    validation_samples = _positive_int(
        split.get("validation_samples"),
        context=f"{context}.split.validation_samples",
    )
    sequence_samples = _positive_int(
        dataset.get("sequence_samples"), context=f"{context}.sequence_samples"
    )
    if train_samples + validation_samples != sequence_samples:
        raise _error(
            context,
            "split train_samples + validation_samples must equal sequence_samples",
        )
    return {
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "train_indices_sha256": _require_sha256(
            split.get("train_indices_sha256"),
            context=f"{context}.split.train_indices_sha256",
        ),
        "validation_indices_sha256": _require_sha256(
            split.get("validation_indices_sha256"),
            context=f"{context}.split.validation_indices_sha256",
        ),
    }


def _without_absolute_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_absolute_paths(item)
            for key, item in value.items()
            if key != "path" and not key.endswith("_path")
        }
    if isinstance(value, list):
        return [_without_absolute_paths(item) for item in value]
    return deepcopy(value)


def _path_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(context, "must be a non-empty path string")
    return os.path.normpath(value)


def _validate_training_artifacts(variant_root: Path, variant: str) -> TrainingRun:
    result_path = variant_root / "training_result.json"
    manifest_path = variant_root / "training_manifest.json"
    metrics_path = variant_root / "metrics.csv"
    context = f"{variant}/training"
    result = _load_json(result_path, context=f"{context}.result")
    manifest = _load_json(manifest_path, context=f"{context}.manifest")

    expected_result = {
        "method": METHOD,
        "variant": variant,
        "seed": TRAINING_SEED,
        "final_epoch": TRAINING_EPOCHS,
        "global_step": TRAINING_STEPS,
    }
    for key, expected in expected_result.items():
        if result.get(key) != expected:
            raise _error(context, f"training_result.{key} must equal {expected!r}")
    _positive_int(
        result.get("peak_cuda_memory_bytes"),
        context=f"{context}.result.peak_cuda_memory_bytes",
    )
    run_dir = _path_string(result.get("run_dir"), context=f"{context}.result.run_dir")
    last_checkpoint = _path_string(
        result.get("last_checkpoint"), context=f"{context}.result.last_checkpoint"
    )
    expected_last = os.path.normpath(
        os.path.join(run_dir, "checkpoints", "lightning", "last.ckpt")
    )
    if last_checkpoint != expected_last:
        raise _error(
            context,
            "training_result.last_checkpoint is not run_dir/checkpoints/lightning/last.ckpt",
        )
    deployment_checkpoint = os.path.normpath(
        os.path.join(
            run_dir,
            "checkpoints",
            METHOD,
            variant,
            f"epoch_{TRAINING_EPOCHS:02d}.pt",
        )
    )

    expected_manifest = {
        "method": METHOD,
        "variant": variant,
        "seed": TRAINING_SEED,
        "objective_version": OBJECTIVE_VERSIONS[variant],
        "deployment_checkpoint_version": 1,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise _error(context, f"training_manifest.{key} must equal {expected!r}")
    _path_string(
        manifest.get("protocol_path"), context=f"{context}.manifest.protocol_path"
    )
    protocol = _require_mapping(
        manifest.get("protocol"), context=f"{context}.manifest.protocol"
    )
    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "variant": variant,
        "environment": ENVIRONMENT,
        "stage": "full_training",
    }.items():
        if protocol.get(key) != expected:
            raise _error(context, f"training protocol {key} must equal {expected!r}")
    locked_protocol_sha = _fingerprint(protocol)
    if locked_protocol_sha != TRAINING_PROTOCOL_SHA256[variant]:
        raise _error(
            context,
            "training_manifest.protocol differs from the complete locked training YAML "
            f"for {variant}",
        )
    protocol_runtime = _require_mapping(
        protocol.get("runtime"), context=f"{context}.protocol.runtime"
    )
    if protocol_runtime.get("stable_worldmodel_version") != STABLE_WORLDMODEL_VERSION:
        raise _error(context, "training protocol requires stable-worldmodel 0.1.1")
    protocol_training = _require_mapping(
        protocol.get("training"), context=f"{context}.protocol.training"
    )
    if protocol_training.get("epochs") != TRAINING_EPOCHS:
        raise _error(context, "training protocol must contain 10 epochs")
    if protocol_training.get("optimizer_steps_per_epoch") != OPTIMIZER_STEPS_PER_EPOCH:
        raise _error(context, "training optimizer_steps_per_epoch must equal 12796")
    head_section = "critic" if variant == DIRECT_VARIANT else "successor"
    protocol_head = _require_mapping(
        protocol.get(head_section), context=f"{context}.protocol.{head_section}"
    )
    if protocol_head.get("objective_version") != OBJECTIVE_VERSIONS[variant]:
        raise _error(context, f"training {head_section}.objective_version is wrong")

    runtime = _require_mapping(manifest.get("runtime"), context=f"{context}.runtime")
    if runtime.get("stable_worldmodel") != STABLE_WORLDMODEL_VERSION:
        raise _error(
            context,
            f"runtime.stable_worldmodel must equal {STABLE_WORLDMODEL_VERSION!r}",
        )
    for key in ("torch", "python", "platform", "cuda_device"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise _error(context, f"runtime.{key} must be recorded")
    if "compatibility_adapter" not in runtime or (
        runtime["compatibility_adapter"] is not None
        and not isinstance(runtime["compatibility_adapter"], dict)
    ):
        raise _error(context, "runtime.compatibility_adapter must be null or an object")
    training_commit = _require_git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.runtime.tdwm_git_revision"
    )
    training = _require_mapping(
        manifest.get("training"), context=f"{context}.manifest.training"
    )
    expected_training = {
        "formal_optimizer_steps": TRAINING_STEPS,
        "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
        "configured_optimizer_steps": TRAINING_STEPS,
        "validation_skipped": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise _error(context, f"training_manifest.training.{key} must equal {expected!r}")

    dataset = _require_mapping(
        manifest.get("dataset"), context=f"{context}.manifest.dataset"
    )
    _validate_dataset_provenance(
        dataset,
        context=f"{context}.manifest.dataset",
        require_embedded_conversion=dataset.get("format") == "lance",
    )
    split_fingerprint = _validate_split_manifest(
        dataset, context=f"{context}.manifest.dataset"
    )
    model = _require_mapping(manifest.get("model"), context=f"{context}.manifest.model")
    world_parameters = _positive_int(
        model.get("lewm_parameters"), context=f"{context}.model.lewm_parameters"
    )
    protocol_model = _require_mapping(
        protocol.get("model"), context=f"{context}.protocol.model"
    )
    protocol_world_parameters = _positive_int(
        protocol_model.get("parameters"),
        context=f"{context}.protocol.model.parameters",
    )
    if protocol_world_parameters != world_parameters:
        raise _error(context, "training protocol/model world parameter counts differ")
    head_key = "critic_parameters" if variant == DIRECT_VARIANT else "successor_parameters"
    head_parameters = _positive_int(
        model.get(head_key), context=f"{context}.model.{head_key}"
    )
    curve, metrics = _validate_training_metrics(metrics_path, variant)
    normalized_sources = {
        "training_result.json": _sha256_file(result_path),
        "training_manifest.json": _sha256_file(manifest_path),
        "metrics.csv": _sha256_file(metrics_path),
    }
    missing_common = [
        key for key in TRAINING_COMMON_PROTOCOL_SECTIONS if key not in protocol
    ]
    if missing_common:
        raise _error(context, f"training protocol is missing {missing_common}")
    common_protocol = {
        key: deepcopy(protocol[key]) for key in TRAINING_COMMON_PROTOCOL_SECTIONS
    }
    # The exact parameter count is checked separately across all training and
    # evaluation runs; keep the architecture fields in this protocol hash.
    _require_mapping(
        common_protocol["model"], context=f"{context}.common_protocol.model"
    ).pop("parameters", None)
    training_runtime_fingerprint = {
        key: deepcopy(runtime[key])
        for key in (
            "stable_worldmodel",
            "torch",
            "python",
            "platform",
            "cuda_device",
            "compatibility_adapter",
        )
    }
    dataset_source = {
        "format": dataset.get("format"),
        "size_bytes": dataset.get("size_bytes"),
        "conversion_manifest": _without_absolute_paths(
            dataset.get("conversion_manifest")
        ),
        "split": split_fingerprint,
    }
    return TrainingRun(
        variant=variant,
        training_commit=training_commit,
        run_dir=run_dir,
        last_checkpoint=last_checkpoint,
        deployment_checkpoint=deployment_checkpoint,
        runtime=deepcopy(runtime),
        dataset=deepcopy(dataset),
        model=deepcopy(model),
        metrics=deepcopy(metrics),
        source_file_sha256=normalized_sources,
        world_model_parameter_count=world_parameters,
        head_parameter_count=head_parameters,
        common_protocol_sha256=_fingerprint(common_protocol),
        locked_protocol_sha256=locked_protocol_sha,
        runtime_fingerprint_sha256=_fingerprint(training_runtime_fingerprint),
        dataset_source_fingerprint_sha256=_fingerprint(dataset_source),
        curve=curve,
    )


def _validate_score_metadata(
    *,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    score_mode: str,
    combined_mode: str,
    context: str,
) -> str:
    result_mode = result.get("score_mode")
    manifest_mode = manifest.get("score_mode")
    protocol_mode = _require_mapping(
        protocol.get("inference_objective"),
        context=f"{context}.protocol.inference_objective",
    ).get("score_mode")
    values = (result_mode, manifest_mode, protocol_mode)
    if all(value is not None for value in values):
        if any(value != score_mode for value in values):
            raise _error(context, "score_mode metadata is missing or inconsistent")
        return "explicit"
    if (
        result_mode is None
        and manifest_mode is None
        and score_mode == combined_mode
        and protocol_mode in {None, combined_mode}
    ):
        return "legacy_combined_default"
    if score_mode != combined_mode:
        raise _error(context, "non-combined scores require explicit score_mode metadata")
    raise _error(context, "score_mode metadata is missing or inconsistent")


def _validate_protocol(
    protocol: Mapping[str, Any], *, variant: str, context: str
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "variant": variant,
        "environment": ENVIRONMENT,
        "stage": "planner_evaluation",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise _error(context, f"protocol.{key} must equal {value!r}")
    for section, expected_value in COMMON_PROTOCOL_SECTIONS.items():
        actual = _require_mapping(
            protocol.get(section), context=f"{context}.protocol.{section}"
        )
        if actual != expected_value:
            raise _error(
                context,
                f"protocol.{section} differs from the complete formal Cube O50 lock",
            )

    successor_common = {
        "objective_version": OBJECTIVE_VERSIONS[variant],
        "architecture": "actor_free_successor_head",
        "history_size": 3,
        "hidden_dim": 256,
        "gamma": 0.95,
        "feature_basis": "augmented_latent_squared_distance",
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "goal_conditioning": "none",
        "actor": "none",
        "reward": "none",
        "td_bootstrap": True,
        "target_world_ema_decay": 0.995,
        "target_successor_ema_decay": 0.995,
        "planning_weight": 1.0,
        "terminal_weight": 0.0,
        "clamp_successor_cost": True,
    }
    if variant == "goal_hybrid":
        successor_common.update(
            {
                "goal_readout_training": True,
                "goal_source": "uniform_reachable_future_ema_latent_same_clip",
                "goal_offset_weighting": "uniform_per_transition",
                "goal_terminal_condition": "dataset_terminal_or_next_state_is_goal",
                "goal_readout_branches": ["real_context", "predicted_context"],
                "goal_readout_precision": "float32",
                "goal_cost": "normalized_discounted_latent_mse",
            }
        )
    elif variant == "imaginary_hybrid":
        successor_common.update(
            {
                "immediate_feature_source": "real_ema_next_latent",
                "bootstrap_state_source": (
                    "ema_lewm_predicted_next_from_real_ema_history"
                ),
                "imaginary_horizon": 1,
                "imaginary_predictor_gradient": "target_ema_stop_gradient",
            }
        )
    direct = {
        "objective_version": 3,
        "architecture": "direct_goal_critic_head",
        "history_size": 3,
        "hidden_dim": 256,
        "gamma": 0.95,
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "goal_conditioning": "direct_latent_input",
        "actor": "none",
        "reward": "none",
        "td_bootstrap": True,
        "goal_source": "uniform_reachable_future_ema_latent_same_clip",
        "goal_offset_weighting": "uniform_per_transition",
        "goal_terminal_condition": "dataset_terminal_or_next_state_is_goal",
        "td_branches": ["real_context", "predicted_context"],
        "goal_cost": "normalized_discounted_latent_mse",
        "target_world_ema_decay": 0.995,
        "target_critic_ema_decay": 0.995,
        "planning_weight": 1.0,
        "clamp_critic_cost": True,
    }
    head_section = "critic" if variant == DIRECT_VARIANT else "successor"
    expected_head = direct if variant == DIRECT_VARIANT else successor_common
    actual_head = _require_mapping(
        protocol.get(head_section), context=f"{context}.protocol.{head_section}"
    )
    if actual_head != expected_head:
        raise _error(
            context,
            f"protocol.{head_section} differs from the formal evaluator semantics",
        )
    wrong_head = "successor" if variant == DIRECT_VARIANT else "critic"
    if wrong_head in protocol:
        raise _error(context, f"protocol must not contain a {wrong_head} section")

    expected_checkpoint = {
        "source": "joint_actor_free_td_lewm_export",
        "checkpoint_path": "required_cli_argument",
        "contains_world_model": True,
        ("contains_critic" if variant == DIRECT_VARIANT else "contains_successor"): True,
    }
    checkpoint = _require_mapping(
        protocol.get("checkpoint"), context=f"{context}.protocol.checkpoint"
    )
    if checkpoint != expected_checkpoint:
        raise _error(context, "protocol.checkpoint semantics differ from the evaluator")

    expected_objective = {
        "score": (
            "discounted_direct_goal_critic_cost"
            if variant == DIRECT_VARIANT
            else "discounted_successor_feature_goal_cost"
        ),
        "goal_usage": {
            "goal_hybrid": "training_goal_readout_and_planning_linear_readout",
            DIRECT_VARIANT: "training_and_planning_direct_critic_input",
        }.get(variant, "planning_linear_readout_only"),
        (
            "goal_enters_critic_head"
            if variant == DIRECT_VARIANT
            else "goal_enters_successor_head"
        ): variant == DIRECT_VARIANT,
        "learned_actor": False,
        "replanning": "every_action_block",
    }
    objective = _require_mapping(
        protocol.get("inference_objective"),
        context=f"{context}.protocol.inference_objective",
    )
    objective_without_mode = deepcopy(objective)
    objective_without_mode.pop("score_mode", None)
    if objective_without_mode != expected_objective:
        raise _error(
            context,
            "protocol.inference_objective differs from formal evaluator semantics",
        )
    if "score_mode" in objective and objective["score_mode"] not in modes_for_variant(
        variant
    ):
        raise _error(context, "protocol score_mode is incompatible with its variant")


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _runtime_fingerprint(runtime: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    missing = [key for key in RUNTIME_FINGERPRINT_KEYS if key not in runtime]
    if missing:
        raise _error(context, f"runtime fingerprint is missing {missing}")
    fingerprint = {key: deepcopy(runtime[key]) for key in RUNTIME_FINGERPRINT_KEYS}
    if fingerprint["stable_worldmodel"] != STABLE_WORLDMODEL_VERSION:
        raise _error(context, "runtime stable_worldmodel must equal 0.1.1")
    for key in ("torch", "python", "platform", "device", "cuda_device"):
        if not isinstance(fingerprint[key], str) or not fingerprint[key]:
            raise _error(context, f"runtime.{key} must be a non-empty string")
    if fingerprint["device"] != "cuda":
        raise _error(context, "formal server evaluation must record device='cuda'")
    compatibility = fingerprint["compatibility_adapter"]
    if compatibility is not None and not isinstance(compatibility, dict):
        raise _error(context, "runtime.compatibility_adapter must be null or an object")
    return fingerprint


def _validate_checkpoint_config(
    checkpoint: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    variant: str,
    context: str,
) -> None:
    if checkpoint.get("objective_version") != OBJECTIVE_VERSIONS[variant]:
        raise _error(context, "checkpoint objective_version differs from its variant")
    config_key = "critic_config" if variant == DIRECT_VARIANT else "successor_config"
    config = _require_mapping(
        checkpoint.get(config_key), context=f"{context}.checkpoint.{config_key}"
    )
    head_key = "critic" if variant == DIRECT_VARIANT else "successor"
    head = _require_mapping(protocol.get(head_key), context=f"{context}.{head_key}")
    expected_subset = {
        "method": METHOD,
        "variant": variant,
        "objective_version": OBJECTIVE_VERSIONS[variant],
        "deployment_checkpoint_version": 1,
        "architecture": head["architecture"],
        "embed_dim": FORMAL_MODEL["embed_dim"],
        "history_size": head["history_size"],
        "hidden_dim": head["hidden_dim"],
        "gamma": head["gamma"],
        "action_conditioning": head["action_conditioning"],
        "bootstrap_action": head["bootstrap_action"],
        "terminal_source": head["terminal_source"],
        "goal_conditioning": head["goal_conditioning"],
        "actor": "none",
        "reward": "none",
    }
    if variant != DIRECT_VARIANT:
        expected_subset["feature_basis"] = head["feature_basis"]
    if variant == "goal_hybrid":
        expected_subset.update(
            {
                "goal_readout_training": True,
                "goal_source": head["goal_source"],
                "goal_offset_weighting": head["goal_offset_weighting"],
                "goal_terminal_condition": head["goal_terminal_condition"],
                "goal_readout_branches": head["goal_readout_branches"],
                "goal_readout_precision": head["goal_readout_precision"],
                "goal_cost": head["goal_cost"],
                "goal_enters_successor_head": False,
                "predicted_goal_td_weight": 1.0,
                "real_goal_td_weight": 1.0,
            }
        )
    elif variant == "imaginary_hybrid":
        expected_subset.update(
            {
                "immediate_feature_source": head["immediate_feature_source"],
                "bootstrap_state_source": head["bootstrap_state_source"],
                "imaginary_horizon": head["imaginary_horizon"],
                "imaginary_predictor_gradient": head[
                    "imaginary_predictor_gradient"
                ],
            }
        )
    elif variant == DIRECT_VARIANT:
        expected_subset.update(
            {
                "goal_source": head["goal_source"],
                "goal_offset_weighting": head["goal_offset_weighting"],
                "goal_terminal_condition": head["goal_terminal_condition"],
                "td_branches": head["td_branches"],
                "goal_cost": head["goal_cost"],
                "goal_enters_critic_head": True,
                "predicted_context_detach": False,
                "predicted_critic_td_weight": 1.0,
                "real_critic_td_weight": 1.0,
            }
        )
    for key, expected in expected_subset.items():
        if config.get(key) != expected:
            raise _error(
                context,
                f"checkpoint {config_key}.{key} differs from evaluator semantics",
            )


def _validate_evaluation_run(
    run_dir: Path, *, variant: str, score_mode: str
) -> EvaluationRun:
    context = f"{variant}/{score_mode}"
    result_path = run_dir / "results.json"
    manifest_path = run_dir / "protocol_manifest.json"
    selection_path = run_dir / "episode_selection.json"
    result = _load_json(result_path, context=f"{context}/results.json")
    manifest = _load_json(manifest_path, context=f"{context}/protocol_manifest.json")
    selection_source = _load_json(
        selection_path, context=f"{context}/episode_selection.json"
    )

    if result.get("method") != METHOD or result.get("variant") != variant:
        raise _error(context, "results method/variant does not match its bundle directory")
    if _require_bool(result.get("smoke"), context=f"{context}.smoke"):
        raise _error(context, "smoke results cannot enter the formal archive")
    if _require_bool(result.get("pilot"), context=f"{context}.pilot"):
        raise _error(context, "pilot results cannot enter the formal archive")

    protocol = _require_mapping(
        manifest.get("protocol"), context=f"{context}.manifest.protocol"
    )
    formal_protocol = _require_mapping(
        manifest.get("formal_protocol"), context=f"{context}.manifest.formal_protocol"
    )
    _validate_protocol(protocol, variant=variant, context=context)
    _validate_protocol(formal_protocol, variant=variant, context=f"{context}.formal")
    combined_mode = combined_mode_for_variant(variant)
    formal_score_mode = _require_mapping(
        formal_protocol.get("inference_objective"),
        context=f"{context}.formal_protocol.inference_objective",
    ).get("score_mode")
    if formal_score_mode is not None and formal_score_mode != combined_mode:
        raise _error(
            context,
            "formal_protocol.inference_objective.score_mode must be missing for a "
            f"legacy run or equal combined mode {combined_mode!r}",
        )
    score_source = _validate_score_metadata(
        result=result,
        manifest=manifest,
        protocol=protocol,
        score_mode=score_mode,
        combined_mode=combined_mode,
        context=context,
    )
    formal_copy = deepcopy(formal_protocol)
    formal_objective = _require_mapping(
        formal_copy.get("inference_objective"),
        context=f"{context}.formal_protocol.inference_objective",
    )
    if score_source == "explicit":
        formal_objective["score_mode"] = score_mode
        if protocol != formal_copy:
            raise _error(
                context,
                "configured protocol must differ from formal_protocol only by score_mode",
            )
    elif protocol != formal_protocol:
        raise _error(context, "legacy combined protocol must equal formal_protocol")
    canonical_formal_protocol = deepcopy(formal_protocol)
    canonical_formal_objective = _require_mapping(
        canonical_formal_protocol.get("inference_objective"),
        context=f"{context}.formal_protocol.inference_objective",
    )
    canonical_formal_objective.setdefault("score_mode", combined_mode)

    metrics = _require_mapping(result.get("metrics"), context=f"{context}.metrics")
    outcomes = _success_vector(metrics, context=context)
    if "success_rate" not in metrics:
        raise _error(context, "metrics.success_rate is required")
    success_rate = _normalize_success_rate(
        metrics["success_rate"], context=f"{context}.metrics.success_rate"
    )
    observed = sum(outcomes) / EPISODES
    if not math.isclose(success_rate, observed, rel_tol=0.0, abs_tol=1e-12):
        raise _error(
            context,
            f"success_rate {success_rate} disagrees with {sum(outcomes)}/{EPISODES}",
        )

    elapsed = _finite_number(
        result.get("elapsed_seconds"), context=f"{context}.elapsed_seconds"
    )
    if elapsed <= 0.0:
        raise _error(context, "elapsed_seconds must be positive")
    world_parameters = _positive_int(
        result.get("world_model_parameter_count"),
        context=f"{context}.world_model_parameter_count",
    )
    head_key = (
        "critic_parameter_count"
        if variant == DIRECT_VARIANT
        else "successor_parameter_count"
    )
    head_parameters = _positive_int(
        result.get(head_key), context=f"{context}.{head_key}"
    )

    normalized_selection = _selection(selection_source, context=context)
    manifest_selection = _selection(manifest.get("selection"), context=f"{context}.manifest")
    if normalized_selection != manifest_selection:
        raise _error(context, "manifest.selection differs from episode_selection.json")
    selection_sha = _sha256_file(selection_path)
    if selection_sha != SELECTION_SHA256:
        raise _error(
            context,
            f"episode_selection.json SHA-256 must equal locked {SELECTION_SHA256}",
        )

    checkpoint = _require_mapping(
        manifest.get("checkpoint"), context=f"{context}.manifest.checkpoint"
    )
    checkpoint_sha = _require_sha256(
        checkpoint.get("sha256"), context=f"{context}.checkpoint.sha256"
    )
    if checkpoint.get("method") != METHOD or checkpoint.get("variant") != variant:
        raise _error(context, "checkpoint method/variant metadata is inconsistent")
    checkpoint_path = _path_string(
        checkpoint.get("path"), context=f"{context}.checkpoint.path"
    )
    _validate_checkpoint_config(
        checkpoint,
        protocol=formal_protocol,
        variant=variant,
        context=context,
    )

    dataset = _require_mapping(
        manifest.get("dataset"), context=f"{context}.manifest.dataset"
    )
    if (
        dataset.get("episodes") != DATASET_EPISODES
        or dataset.get("transitions") != DATASET_TRANSITIONS
    ):
        raise _error(context, "manifest dataset must contain 10,000 episodes/2,010,000 rows")
    _validate_dataset_provenance(dataset, context=f"{context}.manifest.dataset")
    normalization = _require_mapping(
        manifest.get("normalization"), context=f"{context}.manifest.normalization"
    )
    action_normalization = _require_mapping(
        normalization.get("action"),
        context=f"{context}.manifest.normalization.action",
    )
    if not action_normalization:
        raise _error(context, "action normalization provenance must not be empty")
    runtime = _require_mapping(
        manifest.get("runtime"), context=f"{context}.manifest.runtime"
    )
    runtime_fingerprint = _runtime_fingerprint(runtime, context=context)
    evaluation_commit = _require_git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.runtime.tdwm_git_revision"
    )
    _path_string(
        manifest.get("protocol_path"), context=f"{context}.manifest.protocol_path"
    )
    common_protocol = {
        key: deepcopy(formal_protocol[key]) for key in COMMON_PROTOCOL_SECTIONS
    }
    return EvaluationRun(
        variant=variant,
        score_mode=score_mode,
        score_mode_source=score_source,
        successes=outcomes,
        success_rate=success_rate,
        elapsed_seconds=elapsed,
        checkpoint_sha256=checkpoint_sha,
        selection=normalized_selection,
        selection_sha256=selection_sha,
        formal_protocol_sha256=_sha256_bytes(
            _canonical_json_bytes(canonical_formal_protocol)
        ),
        evaluation_commit=evaluation_commit,
        runtime=deepcopy(runtime),
        dataset=deepcopy(dataset),
        normalization=deepcopy(normalization),
        checkpoint_path=checkpoint_path,
        common_protocol_sha256=_fingerprint(common_protocol),
        runtime_fingerprint_sha256=_fingerprint(runtime_fingerprint),
        dataset_fingerprint_sha256=_fingerprint(dataset),
        normalization_fingerprint_sha256=_fingerprint(normalization),
        world_model_parameter_count=world_parameters,
        head_parameter_count=head_parameters,
        source_sha256={
            "results.json": _sha256_file(result_path),
            "protocol_manifest.json": _sha256_file(manifest_path),
            "episode_selection.json": selection_sha,
        },
    )


def validate_bundle(bundle_root: str | Path) -> ValidatedStudy:
    """Load a complete 7x3 bundle and enforce the locked O50 invariants."""

    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleValidationError(f"Bundle root does not exist: {root}")

    training: dict[str, TrainingRun] = {}
    evaluations: dict[str, dict[str, EvaluationRun]] = {}
    reference_selection: Mapping[str, tuple[int, ...]] | None = None
    reference_selection_sha: str | None = None
    all_runs: list[EvaluationRun] = []
    for variant in VARIANT_ORDER:
        variant_root = root / variant
        training_run = _validate_training_artifacts(variant_root, variant)
        training[variant] = training_run
        runs: dict[str, EvaluationRun] = {}
        for score_mode in modes_for_variant(variant):
            run = _validate_evaluation_run(
                variant_root / score_mode,
                variant=variant,
                score_mode=score_mode,
            )
            runs[score_mode] = run
            all_runs.append(run)
            if reference_selection is None:
                reference_selection = run.selection
                reference_selection_sha = run.selection_sha256
            elif (
                run.selection != reference_selection
                or run.selection_sha256 != reference_selection_sha
            ):
                raise _error(
                    f"{variant}/{score_mode}",
                    "selection differs from the common 21-run O50 selection",
                )

        checkpoint_hashes = {run.checkpoint_sha256 for run in runs.values()}
        if len(checkpoint_hashes) != 1:
            raise _error(variant, "the three score modes use different checkpoints")
        checkpoint_paths = {run.checkpoint_path for run in runs.values()}
        if len(checkpoint_paths) != 1:
            raise _error(variant, "the three score modes use different checkpoint paths")
        if next(iter(checkpoint_paths)) != training_run.deployment_checkpoint:
            raise _error(
                variant,
                "evaluation checkpoint path differs from the trainer's epoch-10 export",
            )
        protocols = {run.formal_protocol_sha256 for run in runs.values()}
        if len(protocols) != 1:
            raise _error(variant, "the three score modes use different formal protocols")
        world_counts = {run.world_model_parameter_count for run in runs.values()}
        head_counts = {run.head_parameter_count for run in runs.values()}
        if len(world_counts) != 1 or len(head_counts) != 1:
            raise _error(variant, "parameter counts differ across score modes")
        if next(iter(world_counts)) != training_run.world_model_parameter_count:
            raise _error(variant, "evaluation world-model count differs from training")
        if next(iter(head_counts)) != training_run.head_parameter_count:
            raise _error(variant, "evaluation head count differs from training")
        evaluations[variant] = runs

    assert reference_selection is not None and reference_selection_sha is not None
    if reference_selection_sha != SELECTION_SHA256:
        raise BundleValidationError("The common selection is not the locked seed-42 O50 file.")
    shared_fingerprints = {
        "formal protocol common sections": {
            run.common_protocol_sha256 for run in all_runs
        },
        "critical runtime": {run.runtime_fingerprint_sha256 for run in all_runs},
        "dataset source/provenance": {
            run.dataset_fingerprint_sha256 for run in all_runs
        },
        "action normalization": {
            run.normalization_fingerprint_sha256 for run in all_runs
        },
    }
    for label, values in shared_fingerprints.items():
        if len(values) != 1:
            raise BundleValidationError(
                f"The 21 formal runs do not share one {label} fingerprint."
            )
    training_fingerprints = {
        "training common protocol": {
            run.common_protocol_sha256 for run in training.values()
        },
        "training critical runtime": {
            run.runtime_fingerprint_sha256 for run in training.values()
        },
        "training dataset source/provenance": {
            run.dataset_source_fingerprint_sha256 for run in training.values()
        },
    }
    for label, values in training_fingerprints.items():
        if len(values) != 1:
            raise BundleValidationError(
                f"The seven training runs do not share one {label} fingerprint."
            )
    world_counts = {run.world_model_parameter_count for run in all_runs}
    training_world_counts = {
        run.world_model_parameter_count for run in training.values()
    }
    if len(world_counts) != 1 or training_world_counts != world_counts:
        raise BundleValidationError(
            "world_model_parameter_count must be identical across all 21 evaluations "
            "and all seven training manifests."
        )
    return ValidatedStudy(
        training=training,
        evaluations=evaluations,
        selection=reference_selection,
        selection_sha256=reference_selection_sha,
    )


def _ranking(study: ValidatedStudy) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        combined = combined_mode_for_variant(variant)
        run = study.evaluations[variant][combined]
        rows.append(
            {
                "variant": variant,
                "display_name": DISPLAY_NAMES[variant],
                "combined_score_mode": combined,
                "successes": sum(run.successes),
                "episodes": EPISODES,
                "success_rate": run.success_rate,
                "success_rate_percent": 100.0 * run.success_rate,
                "elapsed_seconds": run.elapsed_seconds,
            }
        )
    rows.sort(key=lambda row: (-row["successes"], VARIANT_ORDER.index(row["variant"])))
    for row in rows:
        row["rank"] = 1 + sum(other["successes"] > row["successes"] for other in rows)
    return rows


def build_summary(study: ValidatedStudy) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        runs: dict[str, Any] = {}
        for score_mode in modes_for_variant(variant):
            run = study.evaluations[variant][score_mode]
            runs[score_mode] = {
                "successes": sum(run.successes),
                "episodes": EPISODES,
                "success_rate": run.success_rate,
                "success_rate_percent": 100.0 * run.success_rate,
                "elapsed_seconds": run.elapsed_seconds,
                "score_mode_source": run.score_mode_source,
                "checkpoint_sha256": run.checkpoint_sha256,
                "selection_sha256": run.selection_sha256,
                "formal_protocol_canonical_sha256": run.formal_protocol_sha256,
                "evaluation_commit": run.evaluation_commit,
                "runtime": run.runtime,
                "dataset": run.dataset,
                "normalization": run.normalization,
                "checkpoint_path": run.checkpoint_path,
                "common_protocol_sha256": run.common_protocol_sha256,
                "runtime_fingerprint_sha256": run.runtime_fingerprint_sha256,
                "dataset_fingerprint_sha256": run.dataset_fingerprint_sha256,
                "normalization_fingerprint_sha256": (
                    run.normalization_fingerprint_sha256
                ),
                "world_model_parameter_count": run.world_model_parameter_count,
                (
                    "critic_parameter_count"
                    if variant == DIRECT_VARIANT
                    else "successor_parameter_count"
                ): run.head_parameter_count,
                "source_files_sha256": run.source_sha256,
            }
        methods[variant] = {
            "display_name": DISPLAY_NAMES[variant],
            "family": "direct_goal_critic" if variant == DIRECT_VARIANT else "successor_feature",
            "score_modes": list(modes_for_variant(variant)),
            "combined_score_mode": combined_mode_for_variant(variant),
            "method_spec": METHOD_SPECS[variant],
            "training": {
                "seed": TRAINING_SEED,
                "status": "complete",
                "epochs_completed": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
                "training_commit": training.training_commit,
                "checkpoint_sha256": runs[
                    combined_mode_for_variant(variant)
                ]["checkpoint_sha256"],
                "run_dir": training.run_dir,
                "last_checkpoint": training.last_checkpoint,
                "deployment_checkpoint": training.deployment_checkpoint,
                "runtime": training.runtime,
                "dataset": training.dataset,
                "model": training.model,
                "metrics": training.metrics,
                "source_files_sha256": training.source_file_sha256,
                "world_model_parameter_count": training.world_model_parameter_count,
                (
                    "critic_parameter_count"
                    if variant == DIRECT_VARIANT
                    else "successor_parameter_count"
                ): training.head_parameter_count,
                "common_protocol_sha256": training.common_protocol_sha256,
                "locked_protocol_sha256": training.locked_protocol_sha256,
                "runtime_fingerprint_sha256": training.runtime_fingerprint_sha256,
                "dataset_source_fingerprint_sha256": (
                    training.dataset_source_fingerprint_sha256
                ),
                "loss_curve": list(training.curve),
            },
            "evaluations": runs,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "study": {
            "id": "actor_free_td_lewm_cube_seed3072_o50_score_modes",
            "method": METHOD,
            "environment": "swm/OGBCube-v0",
            "dataset_identifier": DATASET_IDENTIFIER,
            "training_seed": TRAINING_SEED,
            "training_epochs": TRAINING_EPOCHS,
            "optimizer_updates": TRAINING_STEPS,
            "planning_seed": PLANNING_SEED,
            "episodes": EPISODES,
            "goal_offset": GOAL_OFFSET,
            "variant_count": len(VARIANT_ORDER),
            "evaluation_count": sum(
                len(modes_for_variant(variant)) for variant in VARIANT_ORDER
            ),
            "interpretation": (
                "One training seed and one matched O50 planning selection. "
                "Rank only combined inference scores; this is not a multi-seed claim."
            ),
        },
        "score_definitions": {
            "f_only": (
                "LeWM rolls and scores all H predicted states with the normalized "
                "discounted latent-goal cost."
            ),
            "g_only": (
                "LeWM still rolls the candidate to form tail context, but ranking "
                "uses only the successor G readout."
            ),
            "f_plus_g": "F prefix cost plus discounted successor G tail cost.",
            "c_only": (
                "LeWM still rolls the candidate to form critic context, but ranking "
                "uses only the direct goal critic C."
            ),
            "f_plus_c": "F prefix cost plus discounted direct critic C tail cost.",
        },
        "selection": {
            "episode_selection_json_sha256": study.selection_sha256,
            "episodes": EPISODES,
            "pair_hash_definition": (
                "SHA-256 of compact key-sorted JSON containing episode_index, "
                "goal_step, and start_step."
            ),
        },
        "ranking_by_combined": _ranking(study),
        "methods": methods,
        "validation": {
            "complete_7x3_bundle": True,
            "formal_o50_only": True,
            "smoke_or_pilot_runs": 0,
            "common_selection_across_21_runs": True,
            "locked_seed42_selection_sha256": True,
            "same_checkpoint_within_each_variant": True,
            "training_checkpoint_path_matches_evaluation": True,
            "shared_common_protocol_fingerprint": True,
            "shared_critical_runtime_fingerprint": True,
            "shared_dataset_provenance_fingerprint": True,
            "shared_action_normalization_fingerprint": True,
            "shared_world_model_parameter_count": True,
            "shared_training_common_protocol_fingerprint": True,
            "complete_locked_training_protocol_per_variant": True,
            "shared_training_critical_runtime_fingerprint": True,
            "shared_training_dataset_provenance_fingerprint": True,
            "shared_training_split_samples_and_index_hashes": True,
            "training_metrics_derived_from_raw_lightning_csv": True,
            "success_rates_match_episode_outcomes": True,
        },
    }


def build_paired_outcomes_csv(study: ValidatedStudy) -> bytes:
    stream = io.StringIO(newline="")
    outcome_columns = [
        f"success_{variant}__{mode}"
        for variant in VARIANT_ORDER
        for mode in modes_for_variant(variant)
    ]
    fieldnames = [
        "selection_position",
        "selection_sha256",
        "episode_index",
        "start_step",
        "goal_step",
        "valid_row_rank",
        "pair_hash",
        *outcome_columns,
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
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
            for mode in modes_for_variant(variant):
                row[f"success_{variant}__{mode}"] = str(
                    study.evaluations[variant][mode].successes[index]
                ).lower()
        writer.writerow(row)
    return stream.getvalue().encode()


def build_training_curves_csv(study: ValidatedStudy) -> bytes:
    """Build a normalized long-form table used by plots and document tooling."""

    stream = io.StringIO(newline="")
    fieldnames = (
        "variant",
        "display_name",
        "epoch",
        "train_loss",
        "validation_loss",
        "loss_component",
        "cross_method_comparable",
    )
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for variant in VARIANT_ORDER:
        for row in study.training[variant].curve:
            writer.writerow(
                {
                    "variant": variant,
                    "display_name": DISPLAY_NAMES[variant],
                    "epoch": row["epoch"],
                    "train_loss": format(float(row["train_loss"]), ".12g"),
                    "validation_loss": format(
                        float(row["validation_loss"]), ".12g"
                    ),
                    "loss_component": "method_total_loss",
                    "cross_method_comparable": "false",
                }
            )
    return stream.getvalue().encode()


def build_training_curves_svg(study: ValidatedStudy) -> bytes:
    """Render a deterministic dependency-free two-panel loss curve as SVG."""

    width, height = 1600, 780
    panel_top, panel_height, panel_width = 92, 485, 650
    left_x, right_x = 105, 845
    colors = (
        "#2E74B5",
        "#D95F02",
        "#1B9E77",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#A6761D",
    )

    def panel(metric: str, title: str, origin_x: int) -> list[str]:
        values = [
            float(row[metric])
            for variant in VARIANT_ORDER
            for row in study.training[variant].curve
        ]
        y_max = max(values)
        y_max = y_max * 1.08 if y_max > 0 else 1.0

        def x(epoch: int) -> float:
            return origin_x + (epoch - 1) * panel_width / (TRAINING_EPOCHS - 1)

        def y(value: float) -> float:
            return panel_top + panel_height * (1.0 - value / y_max)

        items = [
            f'<text x="{origin_x}" y="62" class="panel-title">{html.escape(title)}</text>',
            f'<rect x="{origin_x}" y="{panel_top}" width="{panel_width}" '
            f'height="{panel_height}" class="plot-bg"/>',
        ]
        for tick in range(6):
            value = y_max * tick / 5
            py = y(value)
            items.append(
                f'<line x1="{origin_x}" y1="{py:.2f}" x2="{origin_x + panel_width}" '
                f'y2="{py:.2f}" class="grid"/>'
            )
            items.append(
                f'<text x="{origin_x - 12}" y="{py + 5:.2f}" '
                f'class="tick ytick">{value:.3f}</text>'
            )
        for epoch in range(1, TRAINING_EPOCHS + 1):
            px = x(epoch)
            items.append(
                f'<text x="{px:.2f}" y="{panel_top + panel_height + 28}" '
                f'class="tick xtick">{epoch}</text>'
            )
        items.append(
            f'<text x="{origin_x + panel_width / 2:.2f}" '
            f'y="{panel_top + panel_height + 62}" class="axis-title">Epoch</text>'
        )
        for index, variant in enumerate(VARIANT_ORDER):
            points = " ".join(
                f"{x(int(row['epoch'])):.2f},{y(float(row[metric])):.2f}"
                for row in study.training[variant].curve
            )
            items.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[index]}" '
                'stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for row in study.training[variant].curve:
                items.append(
                    f'<circle cx="{x(int(row["epoch"])):.2f}" '
                    f'cy="{y(float(row[metric])):.2f}" r="3.4" '
                    f'fill="{colors[index]}"/>'
                )
        return items

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Actor-Free TD-LeWM training and validation total-loss curves">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #17212B; }",
        ".title { font-size: 27px; font-weight: 700; }",
        ".subtitle { font-size: 16px; fill: #5C6975; }",
        ".panel-title { font-size: 21px; font-weight: 700; }",
        ".plot-bg { fill: #FAFBFC; stroke: #8795A1; stroke-width: 1.2; }",
        ".grid { stroke: #D7DEE5; stroke-width: 1; }",
        ".tick { font-size: 13px; fill: #5C6975; }",
        ".ytick { text-anchor: end; }",
        ".xtick { text-anchor: middle; }",
        ".axis-title { font-size: 14px; text-anchor: middle; fill: #5C6975; }",
        ".legend { font-size: 14px; }",
        "</style>",
        '<rect width="1600" height="780" fill="#FFFFFF"/>',
        '<text x="40" y="35" class="title">Actor-Free TD-LeWM loss curves</text>',
        '<text x="40" y="66" class="subtitle">Method total losses; use each curve for convergence only, not cross-method ranking.</text>',
        *panel("train_loss", "Training total loss", left_x),
        *panel("validation_loss", "Validation total loss", right_x),
    ]
    legend_y = 700
    for index, variant in enumerate(VARIANT_ORDER):
        column = index % 4
        row = index // 4
        x = 110 + column * 370
        y = legend_y + row * 32
        svg.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 34}" y2="{y}" '
                f'stroke="{colors[index]}" stroke-width="4"/>',
                f'<text x="{x + 44}" y="{y + 5}" class="legend">'
                f'{html.escape(DISPLAY_NAMES[variant])}</text>',
            ]
        )
    svg.append("</svg>\n")
    return "\n".join(svg).encode()


def _percent(run: EvaluationRun) -> str:
    return f"{sum(run.successes)}/{EPISODES} ({100.0 * run.success_rate:.0f}%)"


def _short_sha(value: str) -> str:
    return value[:12]


def build_report_markdown(study: ValidatedStudy) -> bytes:
    ranking = _ranking(study)
    lines = [
        "# Actor-Free TD-LeWM Cube O50：7×3 推理分数消融",
        "",
        "本报告由完整服务器结果包自动生成。归档器已核验 7 个训练方法、每个方法 3 种",
        "推理分数，共 21 个正式 O50；所有运行使用同一组 50 个 start--goal pair。排名只使用",
        "combined 列：Successor 方法为 `f_plus_g`，Direct Goal Critic 为 `f_plus_c`。",
        "",
        "## Combined 排名",
        "",
        "| 排名 | 方法 | Combined | 成功数 | Success rate | 耗时（秒） |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | {row['display_name']} (`{row['variant']}`) | "
            f"`{row['combined_score_mode']}` | {row['successes']}/{EPISODES} | "
            f"{row['success_rate_percent']:.0f}% | {row['elapsed_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 三种推理分数",
            "",
            "`F-only` 让 LeWM 对 H 个预测状态全部计分；`G/C-only` 仍运行 LeWM 来形成 tail",
            "上下文，但候选排序不加入 F cost；combined 使用 F 与对应 tail 的和。",
            "",
            "| 方法 | F-only | G/C-only | Combined |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant in VARIANT_ORDER:
        runs = study.evaluations[variant]
        if variant == DIRECT_VARIANT:
            tail, combined = "c_only", "f_plus_c"
        else:
            tail, combined = "g_only", "f_plus_g"
        lines.append(
            f"| {DISPLAY_NAMES[variant]} (`{variant}`) | {_percent(runs['f_only'])} | "
            f"{_percent(runs[tail])} | {_percent(runs[combined])} |"
        )
    lines.extend(
        [
            "",
            "## 方法、网络、损失与推理",
            "",
            "所有方法共享 LeWM encoder/predictor、Cube 数据、10 epochs / 127,960 updates、",
            "training seed 3072，以及无 Actor 的 CEM-MPC。`L_LeWM` 包含 prediction MSE 与",
            "0.09 倍 SIGReg；辅助 TD 在训练前 5% updates 线性 warm-up。",
            "",
            "| 方法 | 网络结构 | 训练损失 | 特殊设计 | 推理 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for variant in VARIANT_ORDER:
        spec = METHOD_SPECS[variant]
        lines.append(
            f"| {DISPLAY_NAMES[variant]} (`{variant}`) | {spec['network']} | "
            f"{spec['loss']} | {spec['special']} | {spec['inference']} |"
        )
    lines.extend(
        [
            "",
            "## 训练摘要与 checkpoint 来源",
            "",
            "**重要：不同方法加入的辅助 loss 数量与定义不同，因此图中的 total loss 只能",
            "用于检查各自是否收敛，不能比较曲线高低，也不能当作跨方法性能排名。**",
            "",
            "![7 methods training and validation total-loss curves](artifacts/actor_free_td_lewm_cube_seed3072/training_loss_curves.svg)",
            "",
            "| 方法 | Epoch-10 train/loss | Epoch-10 validation/loss | Best validation | Checkpoint SHA-256 | Training commit |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        checkpoint_sha = study.evaluations[variant][
            combined_mode_for_variant(variant)
        ].checkpoint_sha256
        final_epoch = training.metrics["final_epoch"]
        best = training.metrics["best_validation"]
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {float(final_epoch['train/loss']):.6f} | "
            f"{float(final_epoch['validation/loss']):.6f} | "
            f"{float(best['value']):.6f} (E{best['epoch']}) | "
            f"`{_short_sha(checkpoint_sha)}…` | "
            f"`{_short_sha(training.training_commit)}` |"
        )
    lines.extend(
        [
            "",
            "## 审计结论与边界",
            "",
            f"- 21 个运行的 selection 文件 SHA-256：`{study.selection_sha256}`。",
            "- 七个训练 manifest 的完整 protocol 分别与对应锁定 YAML 的 canonical hash 一致；split 样本数和索引哈希一致，run-specific 绝对路径不参与指纹。",
            "- 每个方法的三种 score mode 使用完全相同的 checkpoint；其路径严格对应训练器的 epoch-10 export。",
            "- 21 个运行共享完整正式协议、关键 runtime、数据格式/大小/转换来源、action normalization 与 world 参数量指纹。",
            "- 所有运行均为 50 episodes、goal offset 50、planning seed 42、完整 CEM 预算，",
            "  且 `smoke=false`、`pilot=false`。",
            "- 每个 success rate 都已由 50 个逐 episode 布尔值重新计算并核对。",
            "- 这仍然只是一个 training seed 和一个 planning selection。它适合结构/推理消融，",
            "  不足以支持跨训练 seed 的总体优越性声明。",
            "",
            "机器可读摘要、50×21 配对结果和来源哈希见",
            "[`artifacts/actor_free_td_lewm_cube_seed3072/`](artifacts/actor_free_td_lewm_cube_seed3072/README.md)。",
            "",
        ]
    )
    return "\n".join(lines).encode()


def build_artifact_readme(study: ValidatedStudy) -> bytes:
    ranking = _ranking(study)
    ranking_lines = [
        f"{row['rank']}. {row['display_name']}: {row['successes']}/{EPISODES} "
        f"({row['success_rate_percent']:.0f}%, {row['combined_score_mode']})"
        for row in ranking
    ]
    lines = [
        "# Actor-Free TD-LeWM Cube O50 7×3 可审计归档",
        "",
        "该目录由服务器导出的完整轻量结果包生成。它包含 7 个方法 × 3 种推理分数的",
        "同一 O50 selection 配对结果，以及训练器原始 JSON/metrics.csv 的派生摘要；不包含",
        "数据集、checkpoint、图像、视频或控制台日志。",
        "",
        "## 文件",
        "",
        "- `summary.json`：combined 排名、21 个汇总结果、训练摘要、runtime 和来源文件哈希。",
        "- `paired_outcomes.csv`：50 个固定 pair × 21 个 success 布尔列。",
        "- `training_loss_curves.csv`：7 个方法 × 10 epochs 的统一 train/validation total loss。",
        "- `training_loss_curves.svg`：可直接嵌入报告/文档的两面板曲线图。",
        "- `checksums.sha256`：归档器生成文件与上级完整报告的 SHA-256。",
        "- `../../actor_free_td_lewm_cube_seed3072.md`：人类可读的完整 Results TD 报告。",
        "",
        "## Combined 排名",
        "",
        *ranking_lines,
        "",
        "同 success 数使用同一名次。排名只使用 `f_plus_g` / `f_plus_c`，不会按三列中的",
        "最佳 post-hoc 数值重新排序。",
        "",
        "## 输入包目录",
        "",
        "```text",
        "<bundle>/<variant>/training_result.json",
        "<bundle>/<variant>/training_manifest.json",
        "<bundle>/<variant>/metrics.csv",
        "<bundle>/<variant>/<score_mode>/results.json",
        "<bundle>/<variant>/<score_mode>/protocol_manifest.json",
        "<bundle>/<variant>/<score_mode>/episode_selection.json",
        "```",
        "",
        "Successor variants 使用 `f_only/g_only/f_plus_g`；`direct_goal_hybrid` 使用",
        "`f_only/c_only/f_plus_c`。旧 evaluator 只有 combined 没有显式 `score_mode` 字段，",
        "归档器只允许它进入 combined 单元，并在 summary 中标记",
        "`legacy_combined_default`；非 combined 单元必须显式记录 mode。",
        "Formal protocol 的 mode 只能缺失或保留 combined；F/G/C-only 只允许出现在 configured protocol。",
        "",
        "## 重建与验证",
        "",
        "从仓库根目录运行：",
        "",
        "```bash",
        "python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle>",
        "python scripts/archive_actor_free_td_lewm_o50.py --bundle <bundle> --check",
        "python scripts/plot_actor_free_td_lewm_losses.py --output <curves.png>",
        "cd reports/artifacts/actor_free_td_lewm_cube_seed3072",
        "shasum -a 256 -c checksums.sha256",
        "```",
        "",
        "归档器会从原始训练文件自行计算 SHA-256、10-epoch 曲线和最终/最佳 validation；",
        "拒绝不完整的 7×3 bundle、smoke/pilot、非 O50、selection 非固定 seed-42 文件、同方法",
        "不同 checkpoint、训练 export 路径不匹配、任何公平协议/runtime/数据来源指纹漂移，",
        "或 success rate 与逐 episode 结果不一致。",
        "",
        "## 选择与哈希",
        "",
        f"共同 `episode_selection.json` SHA-256：`{study.selection_sha256}`。",
        "",
        "`pair_hash` 是只含 `episode_index`、`goal_step`、`start_step` 的 compact、",
        "key-sorted JSON 的 SHA-256。来源 JSON 的 exact byte SHA-256 保存在 `summary.json`。",
        "",
    ]
    return "\n".join(lines).encode()


def build_archive_payloads(
    study: ValidatedStudy,
    *,
    report_checksum_path: str = "../../actor_free_td_lewm_cube_seed3072.md",
) -> tuple[dict[str, bytes], bytes]:
    """Return artifact-directory payloads and the top-level report payload."""

    summary = _pretty_json_bytes(build_summary(study))
    csv_payload = build_paired_outcomes_csv(study)
    curves_csv = build_training_curves_csv(study)
    curves_svg = build_training_curves_svg(study)
    report = build_report_markdown(study)
    readme = build_artifact_readme(study)
    payloads = {
        "README.md": readme,
        "paired_outcomes.csv": csv_payload,
        "summary.json": summary,
        "training_loss_curves.csv": curves_csv,
        "training_loss_curves.svg": curves_svg,
    }
    checksum_entries = {
        **{name: _sha256_bytes(payload) for name, payload in payloads.items()},
        report_checksum_path: _sha256_bytes(report),
    }
    checksum_text = "".join(
        f"{checksum}  {name}\n" for name, checksum in sorted(checksum_entries.items())
    ).encode()
    payloads["checksums.sha256"] = checksum_text
    return payloads, report


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_archive(
    study: ValidatedStudy,
    *,
    artifact_dir: str | Path,
    report_path: str | Path,
    check: bool = False,
) -> list[Path]:
    """Write or byte-compare the deterministic report/archive outputs."""

    artifact_root = Path(artifact_dir)
    report_file = Path(report_path)
    report_checksum_path = Path(
        os.path.relpath(report_file.resolve(), artifact_root.resolve())
    ).as_posix()
    payloads, report = build_archive_payloads(
        study, report_checksum_path=report_checksum_path
    )
    targets = {artifact_root / name: payload for name, payload in payloads.items()}
    targets[report_file] = report
    if check:
        mismatches = [
            str(path)
            for path, payload in targets.items()
            if not path.is_file() or path.read_bytes() != payload
        ]
        if mismatches:
            raise BundleValidationError(
                "Generated archive differs from committed files: " + ", ".join(mismatches)
            )
        return sorted(targets)
    for path, payload in targets.items():
        _write_atomic(path, payload)
    return sorted(targets)


__all__ = [
    "BundleValidationError",
    "DIRECT_MODES",
    "SELECTION_SHA256",
    "TRAINING_PROTOCOL_SHA256",
    "SUCCESSOR_MODES",
    "VARIANT_ORDER",
    "ValidatedStudy",
    "build_archive_payloads",
    "build_paired_outcomes_csv",
    "build_report_markdown",
    "build_summary",
    "build_training_curves_csv",
    "build_training_curves_svg",
    "combined_mode_for_variant",
    "modes_for_variant",
    "validate_bundle",
    "write_archive",
]
