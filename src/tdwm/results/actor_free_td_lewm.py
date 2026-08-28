"""Validate and archive the fixed-seed Actor-Free TD-LeWM Cube O50 study.

The archive intentionally consumes only lightweight JSON outputs and a compact
training summary.  It never reads or copies datasets, checkpoints, videos, or
raw logs.  A complete input bundle has seven variants and three inference
scores per variant; partial bundles are rejected instead of being rendered as
partial results.
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
    "warm_start": True,
    "initial_distribution": "cem_gaussian_no_actor",
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
    world_model_parameter_count: int
    head_parameter_count: int
    source_sha256: Mapping[str, str]


@dataclass(frozen=True)
class TrainingRun:
    variant: str
    training_commit: str
    checkpoint_sha256: str
    runtime: Mapping[str, Any]
    metrics: Mapping[str, Any]
    source_file_sha256: Mapping[str, str]
    summary_sha256: str
    curve_sha256: str
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
    values = metrics.get("success")
    if not isinstance(values, list):
        raise _error(context, "metrics.success must be a JSON array")
    if len(values) != EPISODES:
        raise _error(context, f"metrics.success must contain exactly {EPISODES} rows")
    normalized: list[bool] = []
    for index, value in enumerate(values):
        if type(value) is bool:
            normalized.append(value)
        elif isinstance(value, int) and value in (0, 1):
            normalized.append(bool(value))
        else:
            raise _error(context, f"metrics.success[{index}] is not boolean/0/1")
    return tuple(normalized)


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
    return normalized


def _pair_hash(episode_index: int, start_step: int, goal_step: int) -> str:
    payload = {
        "episode_index": episode_index,
        "goal_step": goal_step,
        "start_step": start_step,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _validate_training_curve(path: Path, variant: str) -> tuple[Mapping[str, Any], ...]:
    context = f"{variant}/training_curve.csv"
    if not path.is_file():
        raise _error(context, f"missing required file {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"epoch", "train_loss", "validation_loss"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise _error(context, f"missing columns {sorted(missing)}")
            for position, source in enumerate(reader, start=1):
                try:
                    epoch = int(source["epoch"])
                except (TypeError, ValueError) as exc:
                    raise _error(context, f"row {position} epoch is not an integer") from exc
                train_loss = _finite_number(
                    _parse_csv_number(source["train_loss"]),
                    context=f"{context} row {position} train_loss",
                )
                validation_loss = _finite_number(
                    _parse_csv_number(source["validation_loss"]),
                    context=f"{context} row {position} validation_loss",
                )
                if train_loss < 0.0 or validation_loss < 0.0:
                    raise _error(context, "loss values must be non-negative")
                rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    }
                )
    except OSError as exc:
        raise _error(context, f"cannot read {path}: {exc}") from exc
    epochs = [row["epoch"] for row in rows]
    if epochs != list(range(1, TRAINING_EPOCHS + 1)):
        raise _error(context, "must contain exactly one ordered row for epochs 1..10")
    return tuple(rows)


def _parse_csv_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _validate_training_summary(path: Path, curve_path: Path, variant: str) -> TrainingRun:
    context = f"{variant}/training_summary.json"
    summary = _load_json(path, context=context)
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "variant": variant,
        "seed": TRAINING_SEED,
        "status": "complete",
        "epochs_completed": TRAINING_EPOCHS,
        "global_step": TRAINING_STEPS,
    }
    for key, expected in expected_scalars.items():
        if summary.get(key) != expected:
            raise _error(context, f"{key} must equal {expected!r}")

    training_commit = _require_git_revision(
        summary.get("training_commit"), context=f"{context}.training_commit"
    )
    checkpoint_sha = _require_sha256(
        summary.get("checkpoint_sha256"), context=f"{context}.checkpoint_sha256"
    )
    runtime = _require_mapping(summary.get("runtime"), context=f"{context}.runtime")
    if runtime.get("stable_worldmodel") != STABLE_WORLDMODEL_VERSION:
        raise _error(
            context,
            f"runtime.stable_worldmodel must equal {STABLE_WORLDMODEL_VERSION!r}",
        )
    if not isinstance(runtime.get("cuda_device"), str) or not runtime["cuda_device"]:
        raise _error(context, "runtime.cuda_device must be recorded")

    metrics = _require_mapping(summary.get("metrics"), context=f"{context}.metrics")
    final_epoch = _require_mapping(
        metrics.get("final_epoch"), context=f"{context}.metrics.final_epoch"
    )
    if final_epoch.get("epoch") != TRAINING_EPOCHS:
        raise _error(context, f"metrics.final_epoch.epoch must equal {TRAINING_EPOCHS}")
    for key in ("train/loss", "validation/loss"):
        _finite_number(
            final_epoch.get(key), context=f"{context}.metrics.final_epoch.{key}"
        )
    best = _require_mapping(
        metrics.get("best_validation"),
        context=f"{context}.metrics.best_validation",
    )
    if best.get("metric") != "validation/loss":
        raise _error(context, "best_validation.metric must equal 'validation/loss'")
    best_epoch = best.get("epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= TRAINING_EPOCHS
    ):
        raise _error(context, "best_validation.epoch must lie in [1,10]")
    _finite_number(best.get("value"), context=f"{context}.best_validation.value")

    source_files = _require_mapping(
        summary.get("source_files"), context=f"{context}.source_files"
    )
    required_sources = ("training_result.json", "training_manifest.json", "metrics.csv")
    normalized_sources = {
        name: _require_sha256(
            source_files.get(name), context=f"{context}.source_files.{name}"
        )
        for name in required_sources
    }
    curve = _validate_training_curve(curve_path, variant)
    final_curve = curve[-1]
    if not math.isclose(
        float(final_epoch["train/loss"]),
        float(final_curve["train_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _error(context, "final train/loss differs from training_curve.csv")
    if not math.isclose(
        float(final_epoch["validation/loss"]),
        float(final_curve["validation_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _error(context, "final validation/loss differs from training_curve.csv")
    curve_best = min(curve, key=lambda row: float(row["validation_loss"]))
    if best["epoch"] != curve_best["epoch"] or not math.isclose(
        float(best["value"]),
        float(curve_best["validation_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _error(context, "best_validation differs from training_curve.csv")
    return TrainingRun(
        variant=variant,
        training_commit=training_commit,
        checkpoint_sha256=checkpoint_sha,
        runtime=deepcopy(runtime),
        metrics=deepcopy(metrics),
        source_file_sha256=normalized_sources,
        summary_sha256=_sha256_file(path),
        curve_sha256=_sha256_file(curve_path),
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
    values = (
        result.get("score_mode"),
        manifest.get("score_mode"),
        _require_mapping(
            protocol.get("inference_objective"),
            context=f"{context}.protocol.inference_objective",
        ).get("score_mode"),
    )
    present = [value for value in values if value is not None]
    if present:
        if len(present) != len(values) or any(value != score_mode for value in values):
            raise _error(context, "score_mode metadata is missing or inconsistent")
        return "explicit"
    if score_mode != combined_mode:
        raise _error(context, "non-combined scores require explicit score_mode metadata")
    return "legacy_combined_default"


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
    runtime = _require_mapping(
        protocol.get("runtime"), context=f"{context}.protocol.runtime"
    )
    if runtime.get("stable_worldmodel_version") != STABLE_WORLDMODEL_VERSION:
        raise _error(context, "protocol requires stable-worldmodel 0.1.1")
    dataset = _require_mapping(
        protocol.get("dataset"), context=f"{context}.protocol.dataset"
    )
    if dataset.get("identifier") != DATASET_IDENTIFIER:
        raise _error(context, f"protocol.dataset.identifier must be {DATASET_IDENTIFIER!r}")
    evaluation = _require_mapping(
        protocol.get("evaluation"), context=f"{context}.protocol.evaluation"
    )
    if evaluation.get("episodes") != EPISODES or evaluation.get("goal_offset") != GOAL_OFFSET:
        raise _error(context, "protocol must be the formal Cube O50 evaluation")
    planning = _require_mapping(
        protocol.get("planning"), context=f"{context}.protocol.planning"
    )
    for key, expected_value in FORMAL_PLANNING.items():
        if planning.get(key) != expected_value:
            raise _error(
                context, f"protocol.planning.{key} must equal {expected_value!r}"
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

    checkpoint = _require_mapping(
        manifest.get("checkpoint"), context=f"{context}.manifest.checkpoint"
    )
    checkpoint_sha = _require_sha256(
        checkpoint.get("sha256"), context=f"{context}.checkpoint.sha256"
    )
    if checkpoint.get("method") != METHOD or checkpoint.get("variant") != variant:
        raise _error(context, "checkpoint method/variant metadata is inconsistent")

    dataset = _require_mapping(
        manifest.get("dataset"), context=f"{context}.manifest.dataset"
    )
    if dataset.get("episodes") != 10_000 or dataset.get("transitions") != 2_010_000:
        raise _error(context, "manifest dataset must contain 10,000 episodes/2,010,000 rows")
    runtime = _require_mapping(
        manifest.get("runtime"), context=f"{context}.manifest.runtime"
    )
    if runtime.get("stable_worldmodel") != STABLE_WORLDMODEL_VERSION:
        raise _error(context, "runtime stable_worldmodel must equal 0.1.1")
    evaluation_commit = _require_git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.runtime.tdwm_git_revision"
    )
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
    for variant in VARIANT_ORDER:
        variant_root = root / variant
        training_run = _validate_training_summary(
            variant_root / "training_summary.json",
            variant_root / "training_curve.csv",
            variant,
        )
        training[variant] = training_run
        runs: dict[str, EvaluationRun] = {}
        for score_mode in modes_for_variant(variant):
            run = _validate_evaluation_run(
                variant_root / score_mode,
                variant=variant,
                score_mode=score_mode,
            )
            runs[score_mode] = run
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
        checkpoint_sha = next(iter(checkpoint_hashes))
        if checkpoint_sha != training_run.checkpoint_sha256:
            raise _error(
                variant,
                "evaluation checkpoint differs from training_summary checkpoint",
            )
        protocols = {run.formal_protocol_sha256 for run in runs.values()}
        if len(protocols) != 1:
            raise _error(variant, "the three score modes use different formal protocols")
        world_counts = {run.world_model_parameter_count for run in runs.values()}
        head_counts = {run.head_parameter_count for run in runs.values()}
        if len(world_counts) != 1 or len(head_counts) != 1:
            raise _error(variant, "parameter counts differ across score modes")
        evaluations[variant] = runs

    assert reference_selection is not None and reference_selection_sha is not None
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
                "checkpoint_sha256": training.checkpoint_sha256,
                "runtime": training.runtime,
                "metrics": training.metrics,
                "source_files_sha256": training.source_file_sha256,
                "training_summary_json_sha256": training.summary_sha256,
                "training_curve_csv_sha256": training.curve_sha256,
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
            "same_checkpoint_within_each_variant": True,
            "training_checkpoint_matches_evaluation": True,
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
        final_epoch = training.metrics["final_epoch"]
        best = training.metrics["best_validation"]
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {float(final_epoch['train/loss']):.6f} | "
            f"{float(final_epoch['validation/loss']):.6f} | "
            f"{float(best['value']):.6f} (E{best['epoch']}) | "
            f"`{_short_sha(training.checkpoint_sha256)}…` | "
            f"`{_short_sha(training.training_commit)}` |"
        )
    lines.extend(
        [
            "",
            "## 审计结论与边界",
            "",
            f"- 21 个运行的 selection 文件 SHA-256：`{study.selection_sha256}`。",
            "- 每个方法的三种 score mode 使用完全相同的 checkpoint；其 SHA 也与训练摘要一致。",
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
        "同一 O50 selection 配对结果，不包含数据集、checkpoint、图像、视频或原始日志。",
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
        "<bundle>/<variant>/training_summary.json",
        "<bundle>/<variant>/training_curve.csv",
        "<bundle>/<variant>/<score_mode>/results.json",
        "<bundle>/<variant>/<score_mode>/protocol_manifest.json",
        "<bundle>/<variant>/<score_mode>/episode_selection.json",
        "```",
        "",
        "Successor variants 使用 `f_only/g_only/f_plus_g`；`direct_goal_hybrid` 使用",
        "`f_only/c_only/f_plus_c`。旧 evaluator 只有 combined 没有显式 `score_mode` 字段，",
        "归档器只允许它进入 combined 单元，并在 summary 中标记",
        "`legacy_combined_default`；非 combined 单元必须显式记录 mode。",
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
        "归档器会拒绝不完整的 7×3 bundle、smoke/pilot、非 O50、不同 selection、同方法",
        "不同 checkpoint、训练 checkpoint 不匹配、协议预算变化，或 success rate 与逐 episode",
        "结果不一致。",
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
