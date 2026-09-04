"""Fail-closed reconciliation of every formal Actor-Free TD-LeWM O50 cell.

The complete ledger intentionally keeps fixed-checkpoint comparisons separate
from epoch sweeps.  It consumes the already archived legacy/V1 summaries, the
V0 raw evaluations and compact summary, the prior V2-EMA new-score ledger, and
the raw V2/V2-EMA original-score sweeps.  Every accepted cell carries the exact
50 boolean outcomes and hashes for its source evidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from tdwm.results import actor_free_td_lewm as legacy_results
from tdwm.results import actor_free_td_lewm_v1 as v1_results
from tdwm.results import actor_free_td_lewm_v2 as v2_results
from tdwm.results import actor_free_td_lewm_v2_ema_sg as ema_results

SCHEMA_VERSION = 1
EPISODE_COUNT = 50
TRAINING_SEED = 3072
PLANNING_SEED = 42
STEPS_PER_EPOCH = 12_796
EPOCHS = tuple(range(3, 11))
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
ORIGINAL_SCORE_MODES = ("f_only", "g_only", "f_plus_g")
NEW_SCORE_MODES = ("f_plus_g_first", "g_only_f_rollout_mean")
G_FIRST_WEIGHT = 0.25
COMPLETE_CELL_COUNT = 477
COMPLETE_OUTCOME_COUNT = COMPLETE_CELL_COUNT * EPISODE_COUNT
EMA_SWEEP_NEW_SCORE_CELL_COUNT = 96
FIXED_NEW_SCORE_CELL_COUNT = 36
FIXED_MEAN_Q_CELL_COUNT = len(("v0", "v1", "v2", "v2_ema_sg")) * len(VARIANTS)

SELECTION_SHA256 = "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)
FIXED_SELECTION_RANKS_SHA256 = (
    "88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9"
)
EMA_NEW_EVALUATION_COMMIT = "5456f3d18116812d078d4ec2e85ba1f83d89c7c7"
EMA_TRAINING_COMMIT = "18cd574d522515f20f4103509b1e660b2fc89ea6"

LEGACY_SUMMARY_SHA256 = (
    "9fa31a12476ff4a332e555505fc720964d81c5640c77cc394beb5df538ae2da6"
)
LEGACY_PAIRED_SHA256 = (
    "3047c9a419f066548b6e942e29c49b35d286670dc60e5db27fcd845897a6f2ae"
)
V0_SUMMARY_SHA256 = "10ec73f4d40fd2a21e01016f9d55c4ac931ba338dde6296dad6877b167bca323"
V1_SUMMARY_SHA256 = "a0210525dfc361e98420d26e1a97cd57dc784186ef58cc3f70d99bbe8ed77922"
V1_PAIRED_SHA256 = "528aa62cafbc8c540fed65c41ee48a6a4c856b3667f8ab0d47673f387db134ca"
EMA_ORIGINAL_SUMMARY_SHA256 = (
    "735afc0cce4498fb84b05efbb185db63ff613986b1a74625106f65aa7512f47c"
)
EMA_ORIGINAL_PAIRED_SHA256 = (
    "25710a0c2383d5c5639a0ff46c76ea5e70b988476d30b60e0c176bbe357d4d1a"
)
EMA_NEW_LEDGER_SHA256 = (
    "b3dc4c77468bca45a1fd2af886f39916c4795a82fcfdffe2d02d1119edb5db29"
)

LEGACY_VARIANTS = tuple(legacy_results.VARIANT_ORDER)
LEGACY_DIRECT = legacy_results.DIRECT_VARIANT
LEGACY_MODES = {
    variant: tuple(legacy_results.modes_for_variant(variant))
    for variant in LEGACY_VARIANTS
}
RETRY_PATHS = {
    (3, "g1", "f_plus_g"): ("evaluation_retry_18cd574/epoch_03/g1/f_plus_g/attempt_02"),
    (3, "g2", "f_only"): ("evaluation_retry_18cd574/epoch_03/g2/f_only/attempt_02"),
}

CSV_FIELDS = (
    "cell_id",
    "family",
    "version",
    "method_id",
    "method_label",
    "variant",
    "epoch",
    "global_step",
    "checkpoint_sha256",
    "score_mode",
    "score_label",
    "success_count",
    "episode_count",
    "success_rate",
    "success_rate_percent",
    "training_seed",
    "planning_seed",
    "selection_sha256",
    "action_normalization_sha256",
    "training_commit",
    "evaluation_commit",
    "planning_horizon",
    "g_first_weight",
    "status",
    "comparison_role",
    "source_kind",
    "source_path",
    "source_results_sha256",
    "source_protocol_sha256",
    "outcomes_sha256",
)
ARCHIVE_FILENAMES = (
    "all_o50_results.csv",
    "reconciliation_ledger.json",
    "summary.json",
    "v0_training_loss_curves.csv",
    "v2_training_loss_curves.csv",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


class CompleteReconciliationError(ValueError):
    """Raised when the 477-cell archive cannot be proven from its sources."""


@dataclass(frozen=True)
class ValidatedCompleteStudy:
    """All validated cells plus deterministic source and training evidence."""

    cells: tuple[Mapping[str, Any], ...]
    sources: Mapping[str, Any]
    v0_training_rows: tuple[Mapping[str, Any], ...]
    v0_training_columns: tuple[str, ...]
    v2_training_rows: tuple[Mapping[str, Any], ...]
    v2_training_columns: tuple[str, ...]
    training_curve_sources: Mapping[str, Any]


def _error(context: str, message: str) -> CompleteReconciliationError:
    return CompleteReconciliationError(f"{context}: {message}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _outcomes_sha256(outcomes: Sequence[bool]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(outcomes))).hexdigest()


def _read_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise _error(context, f"missing JSON file {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(context, f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(context, "JSON root must be an object")
    return value, raw


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(context, "must be an object")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error(context, "must be a lowercase SHA-256")
    return value


def _git_revision(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise _error(context, "must be a lowercase Git revision")
    return value


def _integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(context, "must be an integer")
    return value


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(context, "must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(context, "must be finite")
    return result


def _expect_equal(value: Any, expected: Any, *, context: str) -> None:
    if value != expected:
        raise _error(context, f"found {value!r}, expected {expected!r}")


def _require_locked_file(path: Path, expected_sha256: str, *, context: str) -> bytes:
    if not path.is_file():
        raise _error(context, f"missing locked source {path}")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise _error(context, f"SHA-256 {actual} differs from {expected_sha256}")
    return raw


def _validate_outcomes(metrics: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    canonical = metrics.get("episode_successes")
    legacy = metrics.get("success")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise _error(context, "episode_successes and legacy success disagree")
    values = canonical if canonical is not None else legacy
    if (
        not isinstance(values, list)
        or len(values) != EPISODE_COUNT
        or any(not isinstance(item, bool) for item in values)
    ):
        raise _error(context, "must contain exactly 50 booleans")
    outcomes = tuple(values)
    expected_percent = 100.0 * sum(outcomes) / EPISODE_COUNT
    rate = _number(metrics.get("success_rate"), context=f"{context}.success_rate")
    if not math.isclose(rate, expected_percent, rel_tol=0.0, abs_tol=1e-12):
        raise _error(context, "success_rate does not equal the boolean outcomes")
    return outcomes


def _validate_selection_file(path: Path, *, context: str) -> tuple[dict[str, Any], str]:
    value, raw = _read_json(path, context=context)
    try:
        v2_results._validate_selection(value, raw=raw, context=context)
    except Exception as exc:
        raise _error(context, str(exc)) from exc
    sha = hashlib.sha256(raw).hexdigest()
    if sha != SELECTION_SHA256:
        raise _error(context, "selection is not the locked O50 file")
    return value, sha


def _validate_action_file(path: Path, *, context: str) -> tuple[dict[str, Any], str]:
    value, raw = _read_json(path, context=context)
    try:
        v2_results._validate_action_normalization(value, context=context)
    except Exception as exc:
        raise _error(context, str(exc)) from exc
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ACTION_NORMALIZATION_SHA256:
        raise _error(context, "action normalization is not the locked file")
    return value, sha


def _planning_horizon(manifest: Mapping[str, Any], *, context: str) -> int:
    actual_protocol = manifest.get("protocol")
    if isinstance(actual_protocol, Mapping) and isinstance(
        actual_protocol.get("planning"), Mapping
    ):
        planning = _mapping(
            actual_protocol.get("planning"), context=f"{context}.protocol.planning"
        )
    else:
        formal = _mapping(manifest.get("formal_protocol"), context=f"{context}.formal")
        planning = _mapping(formal.get("planning"), context=f"{context}.planning")
    horizon = _integer(planning.get("horizon"), context=f"{context}.horizon")
    _expect_equal(
        planning.get("planning_seed"), PLANNING_SEED, context=f"{context}.seed"
    )
    return horizon


def _checkpoint_epoch(checkpoint: Mapping[str, Any], *, context: str) -> int:
    raw_epoch = checkpoint.get("epoch")
    if raw_epoch is not None:
        return _integer(raw_epoch, context=f"{context}.epoch")
    path = checkpoint.get("path")
    if not isinstance(path, str):
        raise _error(context, "checkpoint epoch and path are both unavailable")
    match = re.search(r"/epoch_(\d{2})\.pt$", path)
    if match is None:
        raise _error(context, "checkpoint path does not prove an epoch")
    return int(match.group(1))


def _score_label(score_mode: str) -> str:
    return {
        "f_only": "F-only",
        "g_only": "G-only",
        "f_plus_g": "F+G",
        "c_only": "C-only",
        "f_plus_c": "F+C",
        "f_plus_g_first": "F+G First (alpha=0.25)",
        "g_only_f_rollout_mean": "Mean-Q over F rollout",
    }[score_mode]


def _method_label(version: str, variant: str) -> str:
    if version == "legacy":
        return legacy_results.DISPLAY_NAMES[variant]
    if version == "v1":
        return v1_results.DISPLAY_NAMES[variant]
    if version == "v2":
        return v2_results.DISPLAY_NAMES[variant]
    if version == "v2_ema_sg":
        return ema_results.DISPLAY_NAMES[variant]
    if version == "v0":
        return {
            key: value.replace("V1-", "V0-")
            for key, value in v1_results.DISPLAY_NAMES.items()
        }[variant]
    raise _error("method_label", f"unknown version {version!r}")


def _cell_id(
    *, version: str, comparison_role: str, variant: str, epoch: int, score_mode: str
) -> str:
    return f"{comparison_role}__{version}__e{epoch:02d}__{variant}__{score_mode}"


def _base_cell(
    *,
    family: str,
    version: str,
    method_id: str,
    variant: str,
    epoch: int,
    global_step: int | None,
    checkpoint_sha256: str,
    score_mode: str,
    outcomes: Sequence[bool],
    training_commit: str | None,
    evaluation_commit: str | None,
    planning_horizon: int | None,
    g_first_weight: float | None,
    status: str,
    comparison_role: str,
    source_kind: str,
    source_path: str,
    source_results_sha256: str,
    source_protocol_sha256: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_sha = _sha256(checkpoint_sha256, context="checkpoint_sha256")
    outcome_tuple = tuple(outcomes)
    if len(outcome_tuple) != EPISODE_COUNT or any(
        not isinstance(item, bool) for item in outcome_tuple
    ):
        raise _error("cell.outcomes", "must contain exactly 50 booleans")
    success_count = sum(outcome_tuple)
    cell_id = _cell_id(
        version=version,
        comparison_role=comparison_role,
        variant=variant,
        epoch=epoch,
        score_mode=score_mode,
    )
    return {
        "cell_id": cell_id,
        "family": family,
        "version": version,
        "method_id": method_id,
        "method_label": _method_label(version, variant),
        "variant": variant,
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint_sha256": checkpoint_sha,
        "score_mode": score_mode,
        "score_label": _score_label(score_mode),
        "success_count": success_count,
        "episode_count": EPISODE_COUNT,
        "success_rate": success_count / EPISODE_COUNT,
        "success_rate_percent": 100.0 * success_count / EPISODE_COUNT,
        "training_seed": TRAINING_SEED,
        "planning_seed": PLANNING_SEED,
        "selection_sha256": SELECTION_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
        "training_commit": training_commit,
        "evaluation_commit": evaluation_commit,
        "planning_horizon": planning_horizon,
        "g_first_weight": g_first_weight,
        "status": "VERIFIED",
        "comparison_role": comparison_role,
        "source_kind": source_kind,
        "source_path": source_path,
        "source_results_sha256": _sha256(
            source_results_sha256, context=f"{cell_id}.source_results"
        ),
        "source_protocol_sha256": _sha256(
            source_protocol_sha256, context=f"{cell_id}.source_protocol"
        ),
        "outcomes_sha256": _outcomes_sha256(outcome_tuple),
        "outcomes": list(outcome_tuple),
        "evidence": dict(evidence),
    }


def _validate_raw_evaluation(
    *,
    run_dir: Path,
    family: str,
    version: str,
    variant: str,
    score_mode: str,
    expected_epoch: int,
    expected_horizon: int,
    expected_method: str,
    training_commit: str | None,
    comparison_role: str,
    source_kind: str,
    locked_action: Mapping[str, Any] | None = None,
    summary_evaluation: Mapping[str, Any] | None = None,
    summary_results_sha256: str | None = None,
    source_status: str = "SUCCEEDED",
) -> dict[str, Any]:
    """Validate one raw evaluator directory and return a canonical cell."""

    context = f"{version}/{expected_epoch}/{variant}/{score_mode}"
    result, result_raw = _read_json(run_dir / "results.json", context=context)
    manifest, protocol_raw = _read_json(
        run_dir / "protocol_manifest.json", context=f"{context}.manifest"
    )
    selection, selection_sha = _validate_selection_file(
        run_dir / "episode_selection.json", context=f"{context}.selection"
    )
    action_path = run_dir / "action_normalization.json"
    if action_path.is_file():
        action, action_sha = _validate_action_file(
            action_path, context=f"{context}.action"
        )
        action_evidence: Mapping[str, Any] = {
            "source": "action_normalization.json",
            "path": str(action_path.resolve()),
            "sha256": action_sha,
        }
    else:
        if locked_action is None:
            raise _error(context, "action_normalization.json is missing")
        action = dict(locked_action)
        try:
            v2_results._validate_action_normalization(
                action, context=f"{context}.embedded_action"
            )
        except Exception as exc:
            raise _error(context, str(exc)) from exc
        action_sha = ACTION_NORMALIZATION_SHA256
        action_evidence = {
            "source": "protocol_manifest.normalization.action",
            "locked_file_sha256": ACTION_NORMALIZATION_SHA256,
        }

    expected_identity = {
        "method": expected_method,
        "variant": variant,
        "pilot": False,
        "smoke": False,
    }
    for key, expected in expected_identity.items():
        _expect_equal(result.get(key), expected, context=f"{context}.result.{key}")
    legacy_combined_default = version == "legacy" and score_mode in (
        "f_plus_g",
        "f_plus_c",
    )
    if legacy_combined_default:
        if result.get("score_mode") not in (None, score_mode):
            raise _error(context, "legacy combined result has a conflicting score mode")
    else:
        _expect_equal(
            result.get("score_mode"), score_mode, context=f"{context}.result.score_mode"
        )
    if version != "legacy":
        _expect_equal(
            result.get("method_family"), family, context=f"{context}.method_family"
        )
        _expect_equal(
            result.get("implementation_version"),
            version,
            context=f"{context}.implementation_version",
        )
        protocol = _mapping(manifest.get("protocol"), context=f"{context}.protocol")
        inference = _mapping(
            protocol.get("inference_objective"),
            context=f"{context}.protocol.inference_objective",
        )
        _expect_equal(
            inference.get("score_mode"),
            score_mode,
            context=f"{context}.protocol.inference_objective.score_mode",
        )
    if legacy_combined_default:
        if manifest.get("score_mode") not in (None, score_mode):
            raise _error(
                context, "legacy combined manifest has a conflicting score mode"
            )
    else:
        _expect_equal(
            manifest.get("score_mode"), score_mode, context=f"{context}.manifest.mode"
        )
    manifest_pointer = result.get("protocol_manifest")
    if not isinstance(manifest_pointer, str) or not manifest_pointer.endswith(
        "/protocol_manifest.json"
    ):
        raise _error(context, "results.json does not point to protocol_manifest.json")
    if manifest.get("selection") != selection:
        raise _error(context, "protocol manifest selection differs from sidecar")
    normalization = _mapping(
        manifest.get("normalization"), context=f"{context}.normalization"
    )
    if normalization.get("action") != action:
        raise _error(context, "protocol manifest action normalization differs")
    horizon = _planning_horizon(manifest, context=context)
    _expect_equal(horizon, expected_horizon, context=f"{context}.planning_horizon")
    if result.get("planning_horizon") is not None:
        _expect_equal(
            result.get("planning_horizon"),
            expected_horizon,
            context=f"{context}.result.planning_horizon",
        )

    outcomes = _validate_outcomes(
        _mapping(result.get("metrics"), context=f"{context}.metrics"),
        context=f"{context}.metrics",
    )
    checkpoint = _mapping(manifest.get("checkpoint"), context=f"{context}.checkpoint")
    epoch = _checkpoint_epoch(checkpoint, context=f"{context}.checkpoint")
    _expect_equal(epoch, expected_epoch, context=f"{context}.checkpoint.epoch")
    checkpoint_sha = _sha256(
        checkpoint.get("sha256"), context=f"{context}.checkpoint.sha256"
    )
    if checkpoint.get("method") is not None:
        _expect_equal(
            checkpoint.get("method"),
            expected_method,
            context=f"{context}.checkpoint.method",
        )
    if checkpoint.get("variant") is not None:
        _expect_equal(
            checkpoint.get("variant"), variant, context=f"{context}.checkpoint.variant"
        )
    if version != "legacy":
        _expect_equal(
            checkpoint.get("method_family"),
            family,
            context=f"{context}.checkpoint.method_family",
        )
        _expect_equal(
            checkpoint.get("implementation_version"),
            version,
            context=f"{context}.checkpoint.implementation_version",
        )
    global_step_value = checkpoint.get("global_step")
    global_step = None
    if global_step_value is not None:
        global_step = _integer(
            global_step_value, context=f"{context}.checkpoint.global_step"
        )
        _expect_equal(
            global_step,
            expected_epoch * STEPS_PER_EPOCH,
            context=f"{context}.checkpoint.global_step",
        )
    elif expected_epoch == 10 and version == "legacy":
        global_step = expected_epoch * STEPS_PER_EPOCH

    runtime = _mapping(manifest.get("runtime"), context=f"{context}.runtime")
    evaluation_commit = _git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.evaluation_commit"
    )
    result_sha = hashlib.sha256(result_raw).hexdigest()
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    if summary_results_sha256 is not None:
        _expect_equal(
            result_sha,
            summary_results_sha256,
            context=f"{context}.summary.results_sha256",
        )
    if summary_evaluation is not None:
        expected_count = summary_evaluation.get(
            "success_count", summary_evaluation.get("successes")
        )
        _expect_equal(
            expected_count, sum(outcomes), context=f"{context}.summary.success_count"
        )
        summary_rate = summary_evaluation.get("success_rate")
        if summary_rate is not None:
            _expect_equal(
                _number(summary_rate, context=f"{context}.summary.success_rate"),
                sum(outcomes) / EPISODE_COUNT,
                context=f"{context}.summary.success_rate",
            )
        summary_percent = summary_evaluation.get("success_rate_percent")
        if summary_percent is not None:
            actual_percent = _number(
                summary_percent, context=f"{context}.summary.success_rate_percent"
            )
            expected_percent = 100.0 * sum(outcomes) / EPISODE_COUNT
            if not math.isclose(
                actual_percent, expected_percent, rel_tol=0.0, abs_tol=1e-12
            ):
                raise _error(
                    f"{context}.summary.success_rate_percent",
                    f"found {actual_percent!r}, expected {expected_percent!r}",
                )
        summary_checkpoint = summary_evaluation.get("checkpoint_sha256")
        if summary_checkpoint is not None:
            _expect_equal(
                summary_checkpoint,
                checkpoint_sha,
                context=f"{context}.summary.checkpoint_sha256",
            )
        source_files = summary_evaluation.get("source_files_sha256")
        if source_files is not None:
            hashes = _mapping(source_files, context=f"{context}.summary.source_files")
            expected_hashes = {
                "results.json": result_sha,
                "protocol_manifest.json": protocol_sha,
                "episode_selection.json": selection_sha,
            }
            if action_path.is_file():
                expected_hashes["action_normalization.json"] = action_sha
            for name, sha in expected_hashes.items():
                _expect_equal(
                    hashes.get(name),
                    sha,
                    context=f"{context}.summary.source_files.{name}",
                )

    evidence = {
        "source_status": source_status,
        "source_files_sha256": {
            "results.json": result_sha,
            "protocol_manifest.json": protocol_sha,
            "episode_selection.json": selection_sha,
            **(
                {"action_normalization.json": action_sha}
                if action_path.is_file()
                else {}
            ),
        },
        "selection_evidence": {
            "source": "episode_selection.json",
            "path": str((run_dir / "episode_selection.json").resolve()),
            "sha256": selection_sha,
        },
        "action_normalization_evidence": action_evidence,
        "checkpoint": {
            "path": checkpoint.get("path"),
            "epoch_source": (
                "protocol_manifest.checkpoint.epoch"
                if checkpoint.get("epoch") is not None
                else "protocol_manifest.checkpoint.path"
            ),
            "sha256": checkpoint_sha,
        },
    }
    return _base_cell(
        family=family,
        version=version,
        method_id=expected_method,
        variant=variant,
        epoch=epoch,
        global_step=global_step,
        checkpoint_sha256=checkpoint_sha,
        score_mode=score_mode,
        outcomes=outcomes,
        training_commit=training_commit,
        evaluation_commit=evaluation_commit,
        planning_horizon=horizon,
        g_first_weight=None,
        status=source_status,
        comparison_role=comparison_role,
        source_kind=source_kind,
        source_path=str(run_dir.resolve()),
        source_results_sha256=result_sha,
        source_protocol_sha256=protocol_sha,
        evidence=evidence,
    )


def _read_paired_outcomes(
    path: Path,
    *,
    expected_sha256: str,
    expected_columns: Sequence[str],
    context: str,
) -> Mapping[str, tuple[bool, ...]]:
    raw = _require_locked_file(path, expected_sha256, context=context)
    try:
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
    except UnicodeDecodeError as exc:
        raise _error(context, "paired outcomes are not UTF-8") from exc
    required_prefix = (
        "selection_position",
        "selection_sha256",
        "episode_index",
        "start_step",
        "goal_step",
        "valid_row_rank",
        "pair_hash",
    )
    expected_header = (*required_prefix, *expected_columns)
    if not rows or tuple(rows[0]) != expected_header:
        raise _error(context, "paired outcomes header is not canonical")
    if len(rows) != EPISODE_COUNT:
        raise _error(context, "paired outcomes must contain exactly 50 rows")
    columns: dict[str, list[bool]] = {name: [] for name in expected_columns}
    seen_pairs: set[tuple[int, int, int, int]] = set()
    for position, row in enumerate(rows):
        if row["selection_position"] != str(position):
            raise _error(context, "selection_position is not 0..49")
        if row["selection_sha256"] != SELECTION_SHA256:
            raise _error(context, "selection SHA differs from the locked O50 set")
        try:
            episode = int(row["episode_index"])
            start = int(row["start_step"])
            goal = int(row["goal_step"])
            rank = int(row["valid_row_rank"])
        except ValueError as exc:
            raise _error(context, "selection coordinates must be integers") from exc
        if goal - start != 50:
            raise _error(context, "paired outcome is not an O50 pair")
        pair = (episode, start, goal, rank)
        if pair in seen_pairs:
            raise _error(context, "paired outcome row is duplicated")
        seen_pairs.add(pair)
        expected_pair_hash = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "episode_index": episode,
                    "goal_step": goal,
                    "start_step": start,
                }
            )
        ).hexdigest()
        if row["pair_hash"] != expected_pair_hash:
            raise _error(context, "pair_hash is stale")
        for column in expected_columns:
            text = row[column]
            if text not in ("true", "false"):
                raise _error(context, f"{column} is not a literal boolean")
            columns[column].append(text == "true")
    return {key: tuple(value) for key, value in columns.items()}


def _extract_training_curves(
    root: Path,
    *,
    version: str,
    family: str,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[str, ...],
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]],
]:
    """Validate six training metadata triples and retain every epoch metric."""

    all_rows: list[Mapping[str, Any]] = []
    metric_column_union: set[str] = set()
    source_variants: dict[str, Any] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    for variant in VARIANTS:
        context = f"{version}.training.{variant}"
        run = root / variant / f"seed_{TRAINING_SEED}"
        manifest_path = run / "training_manifest.json"
        result_path = run / "training_result.json"
        metrics_path = run / "metrics/version_0/metrics.csv"
        manifest, manifest_raw = _read_json(
            manifest_path, context=f"{context}.manifest"
        )
        result, result_raw = _read_json(result_path, context=f"{context}.result")
        method = f"{family}_{variant}"
        expected_identity = {
            "method": method,
            "method_family": family,
            "variant": variant,
            "implementation_version": version,
            "seed": TRAINING_SEED,
        }
        for document, label in ((manifest, "manifest"), (result, "result")):
            for key, expected in expected_identity.items():
                _expect_equal(
                    document.get(key), expected, context=f"{context}.{label}.{key}"
                )
        _expect_equal(result.get("final_epoch"), 10, context=f"{context}.final_epoch")
        _expect_equal(
            result.get("global_step"), 10 * STEPS_PER_EPOCH, context=f"{context}.step"
        )
        deployment = result.get("deployment_checkpoint")
        if not isinstance(deployment, str) or not deployment.endswith("/epoch_10.pt"):
            raise _error(context, "deployment checkpoint does not prove epoch 10")
        runtime = _mapping(manifest.get("runtime"), context=f"{context}.runtime")
        training_commit = _git_revision(
            runtime.get("tdwm_git_revision"), context=f"{context}.training_commit"
        )
        protocol = _mapping(manifest.get("protocol"), context=f"{context}.protocol")
        protocol_sha = hashlib.sha256(_canonical_json_bytes(protocol)).hexdigest()
        _expect_equal(
            manifest.get("protocol_sha256"),
            protocol_sha,
            context=f"{context}.protocol_sha256",
        )
        if result.get("protocol_sha256") is not None:
            _expect_equal(
                result.get("protocol_sha256"),
                protocol_sha,
                context=f"{context}.result.protocol_sha256",
            )
        if not metrics_path.is_file():
            raise _error(context, f"missing metrics file {metrics_path}")
        metrics_raw = metrics_path.read_bytes()
        try:
            raw_rows = list(csv.DictReader(io.StringIO(metrics_raw.decode())))
        except UnicodeDecodeError as exc:
            raise _error(context, "metrics CSV is not UTF-8") from exc
        if not raw_rows or "epoch" not in raw_rows[0] or "step" not in raw_rows[0]:
            raise _error(context, "metrics CSV has no epoch/step columns")
        columns = tuple(
            name
            for name in raw_rows[0]
            if (name.startswith("train/") and name.endswith("_epoch"))
            or name.startswith("validation/")
        )
        if not columns:
            raise _error(
                context, "metrics CSV has no epoch-level train/validation data"
            )
        metric_column_union.update(columns)
        by_epoch: dict[int, list[Mapping[str, str]]] = defaultdict(list)
        for row in raw_rows:
            try:
                epoch_index = int(row["epoch"])
                int(row["step"])
            except (TypeError, ValueError) as exc:
                raise _error(context, "metrics epoch/step is not an integer") from exc
            by_epoch[epoch_index].append(row)
        if set(by_epoch) != set(range(10)):
            raise _error(context, "metrics do not contain exactly raw epochs 0..9")
        for raw_epoch in range(10):
            epoch_rows = by_epoch[raw_epoch]
            row_out: dict[str, Any] = {
                "variant": variant,
                "method_id": method,
                "method_label": _method_label(version, variant),
                "epoch": raw_epoch + 1,
                "global_step": (raw_epoch + 1) * STEPS_PER_EPOCH,
            }
            for column in columns:
                values = [row[column] for row in epoch_rows if row[column].strip()]
                if len(values) != 1:
                    raise _error(
                        context,
                        f"epoch {raw_epoch + 1} metric {column} has {len(values)} values",
                    )
                try:
                    number = float(values[0])
                except ValueError as exc:
                    raise _error(context, f"metric {column} is not numeric") from exc
                if not math.isfinite(number):
                    raise _error(context, f"metric {column} is not finite")
                row_out[column] = number
            all_rows.append(row_out)
        source_variants[variant] = {
            "training_commit": training_commit,
            "training_manifest_path": str(manifest_path.resolve()),
            "training_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "training_result_path": str(result_path.resolve()),
            "training_result_sha256": hashlib.sha256(result_raw).hexdigest(),
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": hashlib.sha256(metrics_raw).hexdigest(),
            "metric_columns": list(columns),
            "row_count": 10,
        }
        identities[variant] = {
            "training_commit": training_commit,
            "deployment_checkpoint": deployment,
            "global_step": 10 * STEPS_PER_EPOCH,
            "protocol_sha256": protocol_sha,
        }
    metric_columns = tuple(sorted(metric_column_union))
    return (
        tuple(all_rows),
        metric_columns,
        {
            "root": str(root.resolve()),
            "version": version,
            "family": family,
            "variant_count": 6,
            "epoch_rows": 60,
            "variants": source_variants,
        },
        identities,
    )


def _validate_legacy_source(
    *,
    bundle_root: Path,
    summary_path: Path,
    paired_path: Path,
    locked_action: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    context = "legacy"
    summary_raw = _require_locked_file(
        summary_path, LEGACY_SUMMARY_SHA256, context=f"{context}.summary"
    )
    summary = json.loads(summary_raw)
    if not isinstance(summary, dict):
        raise _error(context, "summary root must be an object")
    _expect_equal(summary.get("schema_version"), 1, context=f"{context}.schema_version")
    methods = _mapping(summary.get("methods"), context=f"{context}.methods")
    if set(methods) != set(LEGACY_VARIANTS):
        raise _error(context, "summary method grid is not exactly 7 variants")
    expected_columns = [
        f"success_{variant}__{mode}"
        for variant in LEGACY_VARIANTS
        for mode in LEGACY_MODES[variant]
    ]
    paired = _read_paired_outcomes(
        paired_path,
        expected_sha256=LEGACY_PAIRED_SHA256,
        expected_columns=expected_columns,
        context=f"{context}.paired",
    )
    raw_root = (
        bundle_root / "methods" if (bundle_root / "methods").is_dir() else bundle_root
    )
    cells: list[dict[str, Any]] = []
    for variant in LEGACY_VARIANTS:
        method_summary = _mapping(methods[variant], context=f"legacy.{variant}")
        evaluations = _mapping(
            method_summary.get("evaluations"), context=f"legacy.{variant}.evaluations"
        )
        if set(evaluations) != set(LEGACY_MODES[variant]):
            raise _error(f"legacy.{variant}", "score-mode grid differs")
        training = _mapping(
            method_summary.get("training"), context=f"legacy.{variant}.training"
        )
        training_commit = _git_revision(
            training.get("training_commit"), context=f"legacy.{variant}.training_commit"
        )
        _expect_equal(
            training.get("epochs_completed"), 10, context=f"legacy.{variant}.epochs"
        )
        _expect_equal(
            training.get("global_step"),
            10 * STEPS_PER_EPOCH,
            context=f"legacy.{variant}.global_step",
        )
        for mode in LEGACY_MODES[variant]:
            evaluation = _mapping(
                evaluations[mode], context=f"legacy.{variant}.{mode}.summary"
            )
            cell = _validate_raw_evaluation(
                run_dir=raw_root / variant / "evaluations" / mode,
                family="actor_free_td_lewm",
                version="legacy",
                variant=variant,
                score_mode=mode,
                expected_epoch=10,
                expected_horizon=5,
                expected_method="actor_free_td_lewm",
                training_commit=training_commit,
                comparison_role="fixed_checkpoint_original_scores",
                source_kind="raw_evaluation_crosschecked_by_locked_summary",
                locked_action=locked_action,
                summary_evaluation=evaluation,
            )
            expected_outcomes = paired[f"success_{variant}__{mode}"]
            if tuple(cell["outcomes"]) != expected_outcomes:
                raise _error(
                    f"legacy.{variant}.{mode}", "paired outcomes differ from raw"
                )
            cells.append(cell)
    if len(cells) != 21:
        raise _error(context, "did not produce exactly 21 cells")
    return cells, {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": LEGACY_SUMMARY_SHA256,
        "paired_path": str(paired_path.resolve()),
        "paired_sha256": LEGACY_PAIRED_SHA256,
        "raw_root": str(raw_root.resolve()),
        "cell_count": 21,
    }


def _validate_v0_source(
    *,
    root: Path,
    summary_path: Path,
    training_identities: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    context = "v0"
    summary_raw = _require_locked_file(
        summary_path, V0_SUMMARY_SHA256, context=f"{context}.summary"
    )
    summary = json.loads(summary_raw)
    if not isinstance(summary, dict):
        raise _error(context, "summary root must be an object")
    study_summary = _mapping(summary.get("study"), context="v0.study")
    _expect_equal(
        study_summary.get("method_family"), "actor_free_td_lewm_v0", context="v0.family"
    )
    _expect_equal(study_summary.get("training_seed"), TRAINING_SEED, context="v0.seed")
    _expect_equal(study_summary.get("planning_seed"), PLANNING_SEED, context="v0.plan")
    _expect_equal(
        study_summary.get("episodes_per_evaluation"),
        EPISODE_COUNT,
        context="v0.episodes",
    )
    summary_commit = study_summary.get("training_commit")
    if not isinstance(summary_commit, str) or not summary_commit:
        raise _error(context, "compact summary has no training commit prefix")
    selection = _mapping(summary.get("selection"), context="v0.selection")
    _expect_equal(selection.get("sha256"), SELECTION_SHA256, context="v0.selection.sha")
    methods = _mapping(summary.get("methods"), context="v0.methods")
    if set(methods) != set(VARIANTS):
        raise _error(context, "summary method grid is not exactly six variants")
    locked_action, _ = _validate_action_file(
        root / "o50/c/f_only/action_normalization.json", context="v0.locked_action"
    )
    horizon_by_mode = {"f_only": 5, "g_only": 1, "f_plus_g": 5}
    cells: list[dict[str, Any]] = []
    references: dict[str, Any] = {}
    for variant in VARIANTS:
        identity = _mapping(
            training_identities.get(variant), context=f"v0.training_identity.{variant}"
        )
        training_commit = _git_revision(
            identity.get("training_commit"), context=f"v0.{variant}.training_commit"
        )
        if not training_commit.startswith(summary_commit):
            raise _error(
                f"v0.{variant}", "training commit does not extend summary prefix"
            )
        modes = _mapping(methods[variant], context=f"v0.{variant}.modes")
        if set(modes) != set(ORIGINAL_SCORE_MODES):
            raise _error(f"v0.{variant}", "score-mode grid differs")
        variant_cells: list[dict[str, Any]] = []
        for mode in ORIGINAL_SCORE_MODES:
            compact = _mapping(modes[mode], context=f"v0.{variant}.{mode}.compact")
            _expect_equal(
                compact.get("planning_horizon"),
                horizon_by_mode[mode],
                context=f"v0.{variant}.{mode}.horizon",
            )
            cell = _validate_raw_evaluation(
                run_dir=root / "o50" / variant / mode,
                family="actor_free_td_lewm_v0",
                version="v0",
                variant=variant,
                score_mode=mode,
                expected_epoch=10,
                expected_horizon=horizon_by_mode[mode],
                expected_method=f"actor_free_td_lewm_v0_{variant}",
                training_commit=training_commit,
                comparison_role="fixed_checkpoint_original_scores",
                source_kind="raw_evaluation_crosschecked_by_locked_summary",
                locked_action=locked_action,
                summary_evaluation=compact,
                summary_results_sha256=_sha256(
                    compact.get("results_sha256"),
                    context=f"v0.{variant}.{mode}.results_sha256",
                ),
            )
            variant_cells.append(cell)
            cells.append(cell)
        checkpoint_keys = {
            (cell["checkpoint_sha256"], cell["epoch"], cell["global_step"])
            for cell in variant_cells
        }
        if len(checkpoint_keys) != 1:
            raise _error(f"v0.{variant}", "three score modes use different checkpoints")
        checkpoint_sha, epoch, global_step = next(iter(checkpoint_keys))
        _expect_equal(epoch, 10, context=f"v0.{variant}.epoch")
        _expect_equal(global_step, 10 * STEPS_PER_EPOCH, context=f"v0.{variant}.step")
        _expect_equal(
            identity.get("deployment_checkpoint"),
            variant_cells[0]["evidence"]["checkpoint"]["path"],
            context=f"v0.{variant}.training_deployment",
        )
        references[variant] = {
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_path": variant_cells[0]["evidence"]["checkpoint"]["path"],
            "epoch": epoch,
            "global_step": global_step,
            "training_commit": training_commit,
        }
    if len(cells) != 18:
        raise _error(context, "did not produce exactly 18 cells")
    return (
        cells,
        {
            "root": str(root.resolve()),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": V0_SUMMARY_SHA256,
            "cell_count": 18,
        },
        references,
    )


def _validate_v1_source(
    *,
    bundle_root: Path,
    summary_path: Path,
    paired_path: Path,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    context = "v1"
    try:
        validated = v1_results.validate_bundle(bundle_root)
    except Exception as exc:
        raise _error(context, f"full V1 bundle validation failed: {exc}") from exc
    summary_raw = _require_locked_file(
        summary_path, V1_SUMMARY_SHA256, context=f"{context}.summary"
    )
    summary = json.loads(summary_raw)
    if not isinstance(summary, dict):
        raise _error(context, "summary root must be an object")
    methods = _mapping(summary.get("methods"), context="v1.methods")
    if set(methods) != set(VARIANTS):
        raise _error(context, "summary method grid is not exactly six variants")
    expected_columns = [
        f"success_{variant}__{mode}"
        for variant in VARIANTS
        for mode in ORIGINAL_SCORE_MODES
    ]
    paired = _read_paired_outcomes(
        paired_path,
        expected_sha256=V1_PAIRED_SHA256,
        expected_columns=expected_columns,
        context="v1.paired",
    )
    cells: list[dict[str, Any]] = []
    references: dict[str, Any] = {}
    horizon_by_mode = {"f_only": 5, "g_only": 1, "f_plus_g": 5}
    for variant in VARIANTS:
        method_summary = _mapping(methods[variant], context=f"v1.{variant}")
        evaluations = _mapping(
            method_summary.get("evaluations"), context=f"v1.{variant}.evaluations"
        )
        if set(evaluations) != set(ORIGINAL_SCORE_MODES):
            raise _error(f"v1.{variant}", "score-mode grid differs")
        training = _mapping(
            method_summary.get("training"), context=f"v1.{variant}.training"
        )
        training_commit = _git_revision(
            training.get("training_commit"), context=f"v1.{variant}.training_commit"
        )
        _expect_equal(training.get("epochs"), 10, context=f"v1.{variant}.epochs")
        _expect_equal(
            training.get("global_step"),
            10 * STEPS_PER_EPOCH,
            context=f"v1.{variant}.global_step",
        )
        variant_cells: list[dict[str, Any]] = []
        for mode in ORIGINAL_SCORE_MODES:
            evaluation = _mapping(
                evaluations[mode], context=f"v1.{variant}.{mode}.summary"
            )
            cell = _validate_raw_evaluation(
                run_dir=bundle_root / variant / mode,
                family="actor_free_td_lewm_v1",
                version="v1",
                variant=variant,
                score_mode=mode,
                expected_epoch=10,
                expected_horizon=horizon_by_mode[mode],
                expected_method=f"actor_free_td_lewm_v1_{variant}",
                training_commit=training_commit,
                comparison_role="fixed_checkpoint_original_scores",
                source_kind="raw_evaluation_crosschecked_by_locked_summary",
                summary_evaluation=evaluation,
            )
            if tuple(cell["outcomes"]) != paired[f"success_{variant}__{mode}"]:
                raise _error(f"v1.{variant}.{mode}", "paired outcomes differ from raw")
            validated_outcomes = tuple(validated.evaluations[variant][mode].successes)
            if tuple(cell["outcomes"]) != validated_outcomes:
                raise _error(f"v1.{variant}.{mode}", "full validator outcomes differ")
            variant_cells.append(cell)
            cells.append(cell)
        checkpoint_keys = {
            (cell["checkpoint_sha256"], cell["epoch"], cell["global_step"])
            for cell in variant_cells
        }
        if len(checkpoint_keys) != 1:
            raise _error(f"v1.{variant}", "three score modes use different checkpoints")
        checkpoint_sha, epoch, global_step = next(iter(checkpoint_keys))
        references[variant] = {
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_path": variant_cells[0]["evidence"]["checkpoint"]["path"],
            "epoch": epoch,
            "global_step": global_step,
            "training_commit": training_commit,
        }
    if len(cells) != 18:
        raise _error(context, "did not produce exactly 18 cells")
    return (
        cells,
        {
            "bundle_root": str(bundle_root.resolve()),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": V1_SUMMARY_SHA256,
            "paired_path": str(paired_path.resolve()),
            "paired_sha256": V1_PAIRED_SHA256,
            "cell_count": 18,
        },
        references,
    )


def _raw_sweep_path(
    root: Path,
    *,
    version: str,
    epoch: int,
    variant: str,
    score_mode: str,
) -> Path:
    if version == "v2" and epoch == 10:
        return root / "evaluations" / variant / score_mode
    if version == "v2_ema_sg" and (epoch, variant, score_mode) in RETRY_PATHS:
        return root / RETRY_PATHS[(epoch, variant, score_mode)]
    return root / "evaluation_sweeps" / f"epoch_{epoch:02d}" / variant / score_mode


def _validate_original_sweep(
    *,
    root: Path,
    version: str,
    family: str,
    training_identities: Mapping[str, Mapping[str, Any]] | None,
    ema_summary_path: Path | None = None,
    ema_paired_path: Path | None = None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[tuple[int, str], Any]]:
    context = f"{version}.original_sweep"
    horizon_by_mode = {"f_only": 5, "g_only": 1, "f_plus_g": 5}
    expected_result_paths = {
        (
            _raw_sweep_path(
                root,
                version=version,
                epoch=epoch,
                variant=variant,
                score_mode=mode,
            )
            / "results.json"
        ).resolve()
        for epoch in EPOCHS
        for variant in VARIANTS
        for mode in ORIGINAL_SCORE_MODES
    }
    actual_result_paths = {path.resolve() for path in root.rglob("results.json")}
    if actual_result_paths != expected_result_paths:
        raise _error(
            context,
            "results.json grid differs; "
            f"missing={sorted(map(str, expected_result_paths - actual_result_paths))}, "
            f"extra={sorted(map(str, actual_result_paths - expected_result_paths))}",
        )
    if version == "v2_ema_sg":
        for epoch, variant, mode in RETRY_PATHS:
            main_result = (
                root
                / "evaluation_sweeps"
                / f"epoch_{epoch:02d}"
                / variant
                / mode
                / "results.json"
            )
            if main_result.exists():
                raise _error(
                    context, f"retry cell has a duplicate main result: {main_result}"
                )

    ema_methods: Mapping[str, Any] | None = None
    ema_paired: Mapping[str, tuple[bool, ...]] | None = None
    source_summary: dict[str, Any] = {}
    if version == "v2_ema_sg":
        if ema_summary_path is None or ema_paired_path is None:
            raise _error(context, "EMA epoch-10 cross-check sources are required")
        summary_raw = _require_locked_file(
            ema_summary_path,
            EMA_ORIGINAL_SUMMARY_SHA256,
            context=f"{context}.summary",
        )
        summary = json.loads(summary_raw)
        if not isinstance(summary, dict):
            raise _error(context, "EMA summary root must be an object")
        ema_methods = _mapping(summary.get("methods"), context=f"{context}.methods")
        expected_columns = [
            f"success_{variant}__{mode}"
            for variant in VARIANTS
            for mode in ORIGINAL_SCORE_MODES
        ]
        ema_paired = _read_paired_outcomes(
            ema_paired_path,
            expected_sha256=EMA_ORIGINAL_PAIRED_SHA256,
            expected_columns=expected_columns,
            context=f"{context}.paired",
        )
        source_summary = {
            "epoch10_summary_path": str(ema_summary_path.resolve()),
            "epoch10_summary_sha256": EMA_ORIGINAL_SUMMARY_SHA256,
            "epoch10_paired_path": str(ema_paired_path.resolve()),
            "epoch10_paired_sha256": EMA_ORIGINAL_PAIRED_SHA256,
        }

    cells: list[dict[str, Any]] = []
    references: dict[tuple[int, str], Any] = {}
    for epoch in EPOCHS:
        for variant in VARIANTS:
            if training_identities is None:
                training_commit = EMA_TRAINING_COMMIT
            else:
                training_identity = _mapping(
                    training_identities.get(variant),
                    context=f"{context}.training.{variant}",
                )
                training_commit = _git_revision(
                    training_identity.get("training_commit"),
                    context=f"{context}.{variant}.training_commit",
                )
            variant_cells: list[dict[str, Any]] = []
            for mode in ORIGINAL_SCORE_MODES:
                summary_evaluation = None
                if ema_methods is not None and epoch == 10:
                    method_summary = _mapping(
                        ema_methods[variant], context=f"{context}.summary.{variant}"
                    )
                    summary_evaluation = _mapping(
                        _mapping(
                            method_summary.get("evaluations"),
                            context=f"{context}.summary.{variant}.evaluations",
                        )[mode],
                        context=f"{context}.summary.{variant}.{mode}",
                    )
                run_dir = _raw_sweep_path(
                    root,
                    version=version,
                    epoch=epoch,
                    variant=variant,
                    score_mode=mode,
                )
                cell = _validate_raw_evaluation(
                    run_dir=run_dir,
                    family=family,
                    version=version,
                    variant=variant,
                    score_mode=mode,
                    expected_epoch=epoch,
                    expected_horizon=horizon_by_mode[mode],
                    expected_method=f"{family}_{variant}",
                    training_commit=training_commit,
                    comparison_role="epoch_sweep_original_scores",
                    source_kind=(
                        "raw_retry_attempt_02"
                        if (epoch, variant, mode) in RETRY_PATHS
                        and version == "v2_ema_sg"
                        else "raw_evaluation"
                    ),
                    summary_evaluation=summary_evaluation,
                )
                if (
                    ema_paired is not None
                    and epoch == 10
                    and tuple(cell["outcomes"])
                    != ema_paired[f"success_{variant}__{mode}"]
                ):
                    raise _error(
                        f"{context}.{variant}.{mode}",
                        "epoch-10 paired outcomes differ from raw",
                    )
                variant_cells.append(cell)
                cells.append(cell)
            checkpoint_keys = {
                (cell["checkpoint_sha256"], cell["epoch"], cell["global_step"])
                for cell in variant_cells
            }
            if len(checkpoint_keys) != 1:
                raise _error(
                    f"{context}.e{epoch:02d}.{variant}",
                    "score modes use different checkpoints",
                )
            checkpoint_sha, checkpoint_epoch, global_step = next(iter(checkpoint_keys))
            references[(epoch, variant)] = {
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_path": variant_cells[0]["evidence"]["checkpoint"]["path"],
                "epoch": checkpoint_epoch,
                "global_step": global_step,
                "training_commit": training_commit,
            }
            if epoch == 10 and training_identities is not None:
                identity = _mapping(
                    training_identities[variant],
                    context=f"{context}.training.{variant}",
                )
                _expect_equal(
                    identity.get("deployment_checkpoint"),
                    references[(epoch, variant)]["checkpoint_path"],
                    context=f"{context}.{variant}.training_deployment",
                )
    if len(cells) != 144:
        raise _error(context, "did not produce exactly 144 cells")
    checkpoint_ids = {
        (cell["epoch"], cell["variant"], cell["checkpoint_sha256"]) for cell in cells
    }
    if len(checkpoint_ids) != 48:
        raise _error(context, "checkpoint identity is not exactly 8x6")
    return (
        cells,
        {
            "root": str(root.resolve()),
            "cell_count": 144,
            "epochs": list(EPOCHS),
            "variants": list(VARIANTS),
            "score_modes": list(ORIGINAL_SCORE_MODES),
            "retry_cells": (
                [
                    {
                        "epoch": epoch,
                        "variant": variant,
                        "score_mode": mode,
                        "path": str((root / relative).resolve()),
                    }
                    for (epoch, variant, mode), relative in RETRY_PATHS.items()
                ]
                if version == "v2_ema_sg"
                else []
            ),
            **source_summary,
        },
        references,
    )


def _ledger_outcomes(cell: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    outcomes = cell.get("outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EPISODE_COUNT
        or any(not isinstance(item, bool) for item in outcomes)
    ):
        raise _error(context, "outcomes must contain exactly 50 booleans")
    result = tuple(outcomes)
    success_count = sum(result)
    _expect_equal(cell.get("success_count"), success_count, context=f"{context}.count")
    rate = _number(cell.get("success_rate"), context=f"{context}.rate")
    percent = _number(cell.get("success_rate_percent"), context=f"{context}.percent")
    if not math.isclose(
        rate, success_count / EPISODE_COUNT, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _error(context, "success_rate differs from outcomes")
    if not math.isclose(
        percent, 100.0 * success_count / EPISODE_COUNT, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _error(context, "success_rate_percent differs from outcomes")
    return result


def _ledger_source_hashes(
    cell: Mapping[str, Any], *, context: str
) -> Mapping[str, str]:
    raw = _mapping(cell.get("source_files_sha256"), context=f"{context}.source_files")
    if set(raw) != {
        "results.json",
        "protocol_manifest.json",
        "episode_selection.json",
        "action_normalization.json",
    }:
        raise _error(context, "source file hash grid is not the exact four files")
    hashes = {
        name: _sha256(value, context=f"{context}.source_files.{name}")
        for name, value in raw.items()
    }
    _expect_equal(
        hashes["episode_selection.json"],
        SELECTION_SHA256,
        context=f"{context}.selection_file_sha256",
    )
    _expect_equal(
        hashes["action_normalization.json"],
        ACTION_NORMALIZATION_SHA256,
        context=f"{context}.action_sha256",
    )
    return hashes


def _validate_ema_new_ledger(
    *,
    ledger_path: Path,
    fixed_references: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ema_references: Mapping[tuple[int, str], Mapping[str, Any]],
    expected_ledger_sha256: str = EMA_NEW_LEDGER_SHA256,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    context = "ema_new_ledger"
    locked_ledger_sha = _sha256(
        expected_ledger_sha256, context=f"{context}.expected_sha256"
    )
    raw = _require_locked_file(ledger_path, locked_ledger_sha, context=context)
    try:
        ledger = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _error(context, f"invalid JSON: {exc}") from exc
    if not isinstance(ledger, dict):
        raise _error(context, "ledger root must be an object")
    expected_header = {
        "schema_version": 1,
        "cell_count": 96,
        "epochs": list(EPOCHS),
        "variants": list(VARIANTS),
        "score_modes": list(NEW_SCORE_MODES),
        "g_first_weight": G_FIRST_WEIGHT,
        "selection_sha256": SELECTION_SHA256,
        "evaluation_commit": EMA_NEW_EVALUATION_COMMIT,
        "training_commit": EMA_TRAINING_COMMIT,
    }
    for key, expected in expected_header.items():
        _expect_equal(ledger.get(key), expected, context=f"{context}.{key}")
    _expect_equal(
        ledger.get("action_normalization_sha256"),
        ACTION_NORMALIZATION_SHA256,
        context=f"{context}.action_normalization_sha256",
    )
    raw_cells = ledger.get("cells")
    if not isinstance(raw_cells, list):
        raise _error(context, "cells must be a list")
    expected_grid = {
        (epoch, variant, mode)
        for epoch in EPOCHS
        for variant in VARIANTS
        for mode in NEW_SCORE_MODES
    }
    indexed: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for value in raw_cells:
        cell = _mapping(value, context=f"{context}.cell")
        key = (cell.get("epoch"), cell.get("variant"), cell.get("score_mode"))
        if key in indexed:
            raise _error(context, f"duplicate EMA new-score cell {key}")
        indexed[key] = cell
    if set(indexed) != expected_grid:
        raise _error(context, "EMA new-score grid is not exactly 8x6x2")
    outputs: set[str] = set()
    cells: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for variant in VARIANTS:
            reference = _mapping(
                ema_references.get((epoch, variant)),
                context=f"{context}.reference.e{epoch}.{variant}",
            )
            for mode in NEW_SCORE_MODES:
                source = indexed[(epoch, variant, mode)]
                cell_context = f"{context}.e{epoch:02d}.{variant}.{mode}"
                outcomes = _ledger_outcomes(source, context=cell_context)
                hashes = _ledger_source_hashes(source, context=cell_context)
                expected_id = f"v2_ema_e{epoch:02d}_{variant}_{mode}" + (
                    "_alpha_0p25" if mode == "f_plus_g_first" else ""
                )
                _expect_equal(
                    source.get("cell_id"), expected_id, context=f"{cell_context}.id"
                )
                _expect_equal(
                    source.get("method"),
                    f"actor_free_td_lewm_v2_ema_sg_{variant}",
                    context=f"{cell_context}.method",
                )
                expected_scope = "strict_epoch3" if epoch == 3 else "original_epoch4_10"
                _expect_equal(
                    source.get("source_scope"),
                    expected_scope,
                    context=f"{cell_context}.scope",
                )
                if source.get("source_status") not in ("REUSED", "SUCCEEDED"):
                    raise _error(cell_context, "source status is not terminal")
                output = source.get("output_dir")
                if not isinstance(output, str) or not output:
                    raise _error(cell_context, "output_dir is missing")
                if output in outputs:
                    raise _error(cell_context, "output_dir is duplicated")
                outputs.add(output)
                expected_alpha = G_FIRST_WEIGHT if mode == "f_plus_g_first" else None
                _expect_equal(
                    source.get("g_first_weight"),
                    expected_alpha,
                    context=f"{cell_context}.alpha",
                )
                _expect_equal(
                    source.get("selection_sha256"),
                    SELECTION_SHA256,
                    context=f"{cell_context}.selection",
                )
                _expect_equal(
                    source.get("action_normalization_sha256"),
                    ACTION_NORMALIZATION_SHA256,
                    context=f"{cell_context}.action",
                )
                _expect_equal(
                    source.get("evaluation_commit"),
                    EMA_NEW_EVALUATION_COMMIT,
                    context=f"{cell_context}.evaluation_commit",
                )
                checkpoint_sha = _sha256(
                    source.get("checkpoint_sha256"),
                    context=f"{cell_context}.checkpoint",
                )
                _expect_equal(
                    checkpoint_sha,
                    reference.get("checkpoint_sha256"),
                    context=f"{cell_context}.checkpoint_crosslink",
                )
                _expect_equal(
                    source.get("checkpoint_path"),
                    reference.get("checkpoint_path"),
                    context=f"{cell_context}.checkpoint_path_crosslink",
                )
                _expect_equal(
                    source.get("checkpoint_epoch"),
                    epoch,
                    context=f"{cell_context}.epoch",
                )
                _expect_equal(
                    source.get("checkpoint_global_step"),
                    epoch * STEPS_PER_EPOCH,
                    context=f"{cell_context}.global_step",
                )
                training_manifest_sha = _sha256(
                    source.get("training_manifest_sha256"),
                    context=f"{cell_context}.training_manifest_sha256",
                )
                evidence = {
                    "prior_reconciliation_ledger": {
                        "path": str(ledger_path.resolve()),
                        "sha256": locked_ledger_sha,
                    },
                    "original_output_dir": output,
                    "source_state": source.get("source_state"),
                    "source_status": source.get("source_status"),
                    "source_files_sha256": dict(hashes),
                    "training_manifest_path": source.get("training_manifest_path"),
                    "training_manifest_sha256": training_manifest_sha,
                    "checkpoint_crosslink": dict(reference),
                }
                cells.append(
                    _base_cell(
                        family="actor_free_td_lewm_v2_ema_sg",
                        version="v2_ema_sg",
                        method_id=f"actor_free_td_lewm_v2_ema_sg_{variant}",
                        variant=variant,
                        epoch=epoch,
                        global_step=epoch * STEPS_PER_EPOCH,
                        checkpoint_sha256=checkpoint_sha,
                        score_mode=mode,
                        outcomes=outcomes,
                        training_commit=EMA_TRAINING_COMMIT,
                        evaluation_commit=EMA_NEW_EVALUATION_COMMIT,
                        planning_horizon=None,
                        g_first_weight=expected_alpha,
                        status=str(source["source_status"]),
                        comparison_role="epoch_sweep_new_scores",
                        source_kind="prior_fail_closed_reconciliation_ledger",
                        source_path=str(ledger_path.resolve()),
                        source_results_sha256=hashes["results.json"],
                        source_protocol_sha256=hashes["protocol_manifest.json"],
                        evidence=evidence,
                    )
                )

    fixed_section = _mapping(
        ledger.get("fixed_checkpoint_comparison"), context=f"{context}.fixed"
    )
    expected_fixed_header = {
        "included": True,
        "cell_count": FIXED_NEW_SCORE_CELL_COUNT,
        "selection_sha256": FIXED_SELECTION_RANKS_SHA256,
        "episode_selection_file_sha256": SELECTION_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
    }
    for key, expected in expected_fixed_header.items():
        _expect_equal(
            fixed_section.get(key), expected, context=f"{context}.fixed.{key}"
        )
    raw_fixed = fixed_section.get("cells")
    if not isinstance(raw_fixed, list):
        raise _error(context, "fixed cells must be a list")
    expected_fixed_grid = {
        (version, variant, mode)
        for version in ("v0", "v1", "v2")
        for variant in VARIANTS
        for mode in NEW_SCORE_MODES
    }
    fixed_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    fixed_outputs: set[str] = set()
    for value in raw_fixed:
        source = _mapping(value, context=f"{context}.fixed.cell")
        key = (source.get("version"), source.get("variant"), source.get("score_mode"))
        if key in fixed_index:
            raise _error(context, f"duplicate fixed new-score cell {key}")
        fixed_index[key] = source
    if set(fixed_index) != expected_fixed_grid:
        raise _error(
            context,
            f"fixed new-score grid is not exactly {FIXED_NEW_SCORE_CELL_COUNT} cells",
        )
    for version in ("v0", "v1", "v2"):
        version_refs = _mapping(
            fixed_references.get(version),
            context=f"{context}.fixed.references.{version}",
        )
        for variant in VARIANTS:
            reference = _mapping(
                version_refs.get(variant),
                context=f"{context}.fixed.reference.{version}.{variant}",
            )
            for mode in NEW_SCORE_MODES:
                source = fixed_index[(version, variant, mode)]
                cell_context = f"{context}.fixed.{version}.{variant}.{mode}"
                outcomes = _ledger_outcomes(source, context=cell_context)
                hashes = _ledger_source_hashes(source, context=cell_context)
                _expect_equal(
                    source.get("episode_selection_file_sha256"),
                    SELECTION_SHA256,
                    context=f"{cell_context}.selection_file",
                )
                _expect_equal(
                    source.get("selection_sha256"),
                    FIXED_SELECTION_RANKS_SHA256,
                    context=f"{cell_context}.selection_ranks",
                )
                _expect_equal(
                    source.get("action_normalization_sha256"),
                    ACTION_NORMALIZATION_SHA256,
                    context=f"{cell_context}.action",
                )
                evaluation_commit = _git_revision(
                    source.get("evaluation_commit"),
                    context=f"{cell_context}.evaluation_commit",
                )
                expected_alpha = G_FIRST_WEIGHT if mode == "f_plus_g_first" else None
                _expect_equal(
                    source.get("g_first_weight"),
                    expected_alpha,
                    context=f"{cell_context}.alpha",
                )
                checkpoint_sha = _sha256(
                    source.get("checkpoint_sha256"),
                    context=f"{cell_context}.checkpoint",
                )
                _expect_equal(
                    source.get("method"),
                    f"actor_free_td_lewm_{version}_{variant}",
                    context=f"{cell_context}.method",
                )
                output = source.get("output_dir")
                if not isinstance(output, str) or not output:
                    raise _error(cell_context, "output_dir is missing")
                if output in fixed_outputs:
                    raise _error(cell_context, "fixed output_dir is duplicated")
                fixed_outputs.add(output)
                _expect_equal(
                    checkpoint_sha,
                    reference.get("checkpoint_sha256"),
                    context=f"{cell_context}.checkpoint_crosslink",
                )
                _expect_equal(
                    source.get("checkpoint_path"),
                    reference.get("checkpoint_path"),
                    context=f"{cell_context}.checkpoint_path_crosslink",
                )
                training_commit = _git_revision(
                    reference.get("training_commit"),
                    context=f"{cell_context}.training_commit",
                )
                evidence = {
                    "prior_reconciliation_ledger": {
                        "path": str(ledger_path.resolve()),
                        "sha256": locked_ledger_sha,
                    },
                    "original_output_dir": source.get("output_dir"),
                    "source_launcher_manifest": source.get("source_launcher_manifest"),
                    "source_files_sha256": dict(hashes),
                    "fixed_selection_ranks_sha256": source.get("selection_sha256"),
                    "valid_row_ranks_sha256": source.get("selection_sha256"),
                    "checkpoint_crosslink": dict(reference),
                    "epoch_global_step_evidence": "crosslinked original formal checkpoint",
                    "evaluation_commit_evidence": source.get(
                        "evaluation_commit_evidence"
                    ),
                }
                cells.append(
                    _base_cell(
                        family=f"actor_free_td_lewm_{version}",
                        version=version,
                        method_id=f"actor_free_td_lewm_{version}_{variant}",
                        variant=variant,
                        epoch=10,
                        global_step=10 * STEPS_PER_EPOCH,
                        checkpoint_sha256=checkpoint_sha,
                        score_mode=mode,
                        outcomes=outcomes,
                        training_commit=training_commit,
                        evaluation_commit=evaluation_commit,
                        planning_horizon=None,
                        g_first_weight=expected_alpha,
                        status="SUCCEEDED",
                        comparison_role="fixed_checkpoint_new_scores",
                        source_kind="prior_fail_closed_reconciliation_ledger",
                        source_path=str(ledger_path.resolve()),
                        source_results_sha256=hashes["results.json"],
                        source_protocol_sha256=hashes["protocol_manifest.json"],
                        evidence=evidence,
                    )
                )
    expected_total = EMA_SWEEP_NEW_SCORE_CELL_COUNT + FIXED_NEW_SCORE_CELL_COUNT
    if len(cells) != expected_total:
        raise _error(
            context,
            "did not produce exactly "
            f"{EMA_SWEEP_NEW_SCORE_CELL_COUNT} + {FIXED_NEW_SCORE_CELL_COUNT} cells",
        )
    return cells, {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": locked_ledger_sha,
        "ema_epoch_sweep_cell_count": EMA_SWEEP_NEW_SCORE_CELL_COUNT,
        "fixed_checkpoint_cell_count": FIXED_NEW_SCORE_CELL_COUNT,
        "cell_count": expected_total,
    }


def reconcile_complete_o50(
    *,
    legacy_bundle_root: str | Path,
    legacy_summary: str | Path,
    legacy_paired_outcomes: str | Path,
    v0_root: str | Path,
    v0_summary: str | Path,
    v0_training_root: str | Path,
    v1_bundle_root: str | Path,
    v1_summary: str | Path,
    v1_paired_outcomes: str | Path,
    ema_new_ledger: str | Path,
    ema_new_ledger_sha256: str = EMA_NEW_LEDGER_SHA256,
    v2_root: str | Path,
    v2_training_root: str | Path,
    v2_ema_root: str | Path,
    v2_ema_epoch10_summary: str | Path,
    v2_ema_epoch10_paired_outcomes: str | Path,
) -> ValidatedCompleteStudy:
    """Validate and reconcile the exact 477-cell formal O50 universe."""

    legacy_bundle = Path(legacy_bundle_root).expanduser().resolve()
    legacy_summary_path = Path(legacy_summary).expanduser().resolve()
    legacy_paired_path = Path(legacy_paired_outcomes).expanduser().resolve()
    v0_source_root = Path(v0_root).expanduser().resolve()
    v0_summary_path = Path(v0_summary).expanduser().resolve()
    v0_training = Path(v0_training_root).expanduser().resolve()
    v1_bundle = Path(v1_bundle_root).expanduser().resolve()
    v1_summary_path = Path(v1_summary).expanduser().resolve()
    v1_paired_path = Path(v1_paired_outcomes).expanduser().resolve()
    ema_new_path = Path(ema_new_ledger).expanduser().resolve()
    v2_source_root = Path(v2_root).expanduser().resolve()
    v2_training = Path(v2_training_root).expanduser().resolve()
    ema_source_root = Path(v2_ema_root).expanduser().resolve()
    ema_summary_path = Path(v2_ema_epoch10_summary).expanduser().resolve()
    ema_paired_path = Path(v2_ema_epoch10_paired_outcomes).expanduser().resolve()

    locked_action, _ = _validate_action_file(
        v0_source_root / "o50/c/f_only/action_normalization.json",
        context="global.locked_action",
    )
    v0_training_rows, v0_training_columns, v0_training_sources, v0_identities = (
        _extract_training_curves(
            v0_training,
            version="v0",
            family="actor_free_td_lewm_v0",
        )
    )
    v2_training_rows, v2_training_columns, v2_training_sources, v2_identities = (
        _extract_training_curves(
            v2_training,
            version="v2",
            family="actor_free_td_lewm_v2",
        )
    )

    legacy_cells, legacy_sources = _validate_legacy_source(
        bundle_root=legacy_bundle,
        summary_path=legacy_summary_path,
        paired_path=legacy_paired_path,
        locked_action=locked_action,
    )
    v0_cells, v0_sources, v0_references = _validate_v0_source(
        root=v0_source_root,
        summary_path=v0_summary_path,
        training_identities=v0_identities,
    )
    v1_cells, v1_sources, v1_references = _validate_v1_source(
        bundle_root=v1_bundle,
        summary_path=v1_summary_path,
        paired_path=v1_paired_path,
    )
    v2_cells, v2_sources, v2_sweep_references = _validate_original_sweep(
        root=v2_source_root,
        version="v2",
        family="actor_free_td_lewm_v2",
        training_identities=v2_identities,
    )
    v2_references = {
        variant: v2_sweep_references[(10, variant)] for variant in VARIANTS
    }
    ema_cells, ema_sources, ema_references = _validate_original_sweep(
        root=ema_source_root,
        version="v2_ema_sg",
        family="actor_free_td_lewm_v2_ema_sg",
        training_identities=None,
        ema_summary_path=ema_summary_path,
        ema_paired_path=ema_paired_path,
    )
    ema_new_cells, ema_new_sources = _validate_ema_new_ledger(
        ledger_path=ema_new_path,
        fixed_references={
            "v0": v0_references,
            "v1": v1_references,
            "v2": v2_references,
        },
        ema_references=ema_references,
        expected_ledger_sha256=ema_new_ledger_sha256,
    )

    cells = tuple(
        legacy_cells + v0_cells + v1_cells + v2_cells + ema_cells + ema_new_cells
    )
    if len(cells) != COMPLETE_CELL_COUNT:
        raise _error(
            "complete",
            f"cell count is {len(cells)}, expected {COMPLETE_CELL_COUNT}",
        )
    identities = [str(cell["cell_id"]) for cell in cells]
    if len(set(identities)) != COMPLETE_CELL_COUNT:
        duplicates = sorted(
            cell_id for cell_id, count in Counter(identities).items() if count > 1
        )
        raise _error("complete", f"duplicate cell identities: {duplicates}")
    expected_roles = {
        "fixed_checkpoint_original_scores": 57,
        "epoch_sweep_original_scores": 288,
        "epoch_sweep_new_scores": 96,
        "fixed_checkpoint_new_scores": FIXED_NEW_SCORE_CELL_COUNT,
    }
    actual_roles = Counter(str(cell["comparison_role"]) for cell in cells)
    if dict(actual_roles) != expected_roles:
        raise _error("complete", f"comparison-role counts differ: {dict(actual_roles)}")
    expected_versions = {
        "legacy": 21,
        "v0": 30,
        "v1": 30,
        "v2": 156,
        "v2_ema_sg": 240,
    }
    actual_versions = Counter(str(cell["version"]) for cell in cells)
    if dict(actual_versions) != expected_versions:
        raise _error("complete", f"version counts differ: {dict(actual_versions)}")
    if sum(len(cell["outcomes"]) for cell in cells) != COMPLETE_OUTCOME_COUNT:
        raise _error("complete", f"outcome total is not {COMPLETE_OUTCOME_COUNT:,}")
    fixed_mean_q_grid = {
        (str(cell["version"]), str(cell["variant"]))
        for cell in cells
        if int(cell["epoch"]) == 10
        and cell["score_mode"] == "g_only_f_rollout_mean"
        and cell["version"] in ("v0", "v1", "v2", "v2_ema_sg")
    }
    expected_fixed_mean_q_grid = {
        (version, variant)
        for version in ("v0", "v1", "v2", "v2_ema_sg")
        for variant in VARIANTS
    }
    if fixed_mean_q_grid != expected_fixed_mean_q_grid:
        raise _error(
            "complete",
            f"fixed E10 Mean-Q coverage is not exactly {FIXED_MEAN_Q_CELL_COUNT} cells",
        )
    for cell in cells:
        if cell["selection_sha256"] != SELECTION_SHA256:
            raise _error("complete", "selection drift remains after reconciliation")
        if cell["action_normalization_sha256"] != ACTION_NORMALIZATION_SHA256:
            raise _error("complete", "action-normalization drift remains")
        if _outcomes_sha256(cell["outcomes"]) != cell["outcomes_sha256"]:
            raise _error("complete", f"outcomes hash is stale for {cell['cell_id']}")
        _sha256(cell["checkpoint_sha256"], context=f"{cell['cell_id']}.checkpoint")
        _sha256(cell["source_results_sha256"], context=f"{cell['cell_id']}.results")
        _sha256(cell["source_protocol_sha256"], context=f"{cell['cell_id']}.protocol")
    retry_cells = [
        cell for cell in cells if cell["source_kind"] == "raw_retry_attempt_02"
    ]
    retry_grid = {
        (cell["epoch"], cell["variant"], cell["score_mode"]) for cell in retry_cells
    }
    if retry_grid != set(RETRY_PATHS):
        raise _error(
            "complete", "retry source grid is not the exact two attempt_02 cells"
        )
    if len(v0_training_rows) != 60 or len(v2_training_rows) != 60:
        raise _error("complete", "training curves are not exactly 6x10 rows")
    sources = {
        "legacy": legacy_sources,
        "v0": v0_sources,
        "v1": v1_sources,
        "v2_original_epoch_sweep": v2_sources,
        "v2_ema_original_epoch_sweep": ema_sources,
        "ema_new_scores": ema_new_sources,
    }
    training_curve_sources = {
        "v0": v0_training_sources,
        "v2": v2_training_sources,
    }
    return ValidatedCompleteStudy(
        cells=cells,
        sources=sources,
        v0_training_rows=v0_training_rows,
        v0_training_columns=v0_training_columns,
        v2_training_rows=v2_training_rows,
        v2_training_columns=v2_training_columns,
        training_curve_sources=training_curve_sources,
    )


def build_all_results_csv(study: ValidatedCompleteStudy) -> bytes:
    """Build the fixed-schema 477-row scalar result table."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for cell in study.cells:
        row = {field: cell.get(field) for field in CSV_FIELDS}
        for field in (
            "epoch",
            "global_step",
            "training_commit",
            "evaluation_commit",
            "planning_horizon",
            "g_first_weight",
        ):
            if row[field] is None:
                row[field] = ""
        writer.writerow(row)
    return stream.getvalue().encode()


def _build_training_csv(
    rows: Sequence[Mapping[str, Any]], metric_columns: Sequence[str]
) -> bytes:
    fields = (
        "variant",
        "method_id",
        "method_label",
        "epoch",
        "global_step",
        *metric_columns,
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for source in rows:
        row = {field: source.get(field, "") for field in fields}
        for metric in metric_columns:
            if row[metric] != "":
                row[metric] = format(float(row[metric]), ".17g")
        writer.writerow(row)
    return stream.getvalue().encode()


def build_v0_training_csv(study: ValidatedCompleteStudy) -> bytes:
    return _build_training_csv(study.v0_training_rows, study.v0_training_columns)


def build_v2_training_csv(study: ValidatedCompleteStudy) -> bytes:
    return _build_training_csv(study.v2_training_rows, study.v2_training_columns)


def build_ledger(study: ValidatedCompleteStudy) -> dict[str, Any]:
    """Build the unique full-evidence ledger, including all 23,850 outcomes."""

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "actor_free_td_lewm_complete_o50_reconciliation",
        "cell_count": len(study.cells),
        "episode_count_per_cell": EPISODE_COUNT,
        "outcome_count": len(study.cells) * EPISODE_COUNT,
        "selection_sha256": SELECTION_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
        "grid": {
            "fixed_checkpoint_original_scores": 57,
            "epoch_sweep_original_scores": 288,
            "epoch_sweep_new_scores": 96,
            "fixed_checkpoint_new_scores": FIXED_NEW_SCORE_CELL_COUNT,
        },
        "sources": dict(study.sources),
        "training_curve_sources": dict(study.training_curve_sources),
        "cells": list(study.cells),
    }


def _statistics(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(cell["success_rate_percent"]) for cell in cells]
    best = max(values)
    return {
        "cell_count": len(cells),
        "mean_success_rate_percent": fmean(values),
        "population_sd_percentage_points": pstdev(values),
        "minimum_success_rate_percent": min(values),
        "maximum_success_rate_percent": best,
        "best_cells": [
            {
                "cell_id": cell["cell_id"],
                "method_id": cell["method_id"],
                "variant": cell["variant"],
                "epoch": cell["epoch"],
                "success_count": cell["success_count"],
                "success_rate_percent": cell["success_rate_percent"],
            }
            for cell in cells
            if float(cell["success_rate_percent"]) == best
        ],
    }


def build_summary(
    study: ValidatedCompleteStudy,
    *,
    generated_training_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build compact counts, best cells and dispersion without losing scopes."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in study.cells:
        grouped[
            (
                str(cell["comparison_role"]),
                str(cell["version"]),
                str(cell["score_mode"]),
            )
        ].append(cell)
    statistics = {"|".join(key): _statistics(grouped[key]) for key in sorted(grouped)}
    training = {
        version: {
            **dict(source),
            "generated_csv_sha256": (generated_training_sha256 or {}).get(version),
        }
        for version, source in study.training_curve_sources.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "actor_free_td_lewm_complete_o50_summary",
        "cell_count": len(study.cells),
        "outcome_cell_count": len(study.cells),
        "outcome_count": len(study.cells) * EPISODE_COUNT,
        "training_seed": TRAINING_SEED,
        "planning_seed": PLANNING_SEED,
        "selection_sha256": SELECTION_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
        "cell_counts_by_comparison_role": dict(
            sorted(Counter(cell["comparison_role"] for cell in study.cells).items())
        ),
        "cell_counts_by_version": dict(
            sorted(Counter(cell["version"] for cell in study.cells).items())
        ),
        "retry_attempt_02_cells": [
            {
                "cell_id": cell["cell_id"],
                "source_path": cell["source_path"],
                "source_results_sha256": cell["source_results_sha256"],
                "source_protocol_sha256": cell["source_protocol_sha256"],
            }
            for cell in study.cells
            if cell["source_kind"] == "raw_retry_attempt_02"
        ],
        "statistics_by_scope_version_score_mode": statistics,
        "training_loss_curves": training,
        "sources": dict(study.sources),
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _archive_payloads(study: ValidatedCompleteStudy) -> Mapping[str, bytes]:
    v0_training = build_v0_training_csv(study)
    v2_training = build_v2_training_csv(study)
    summary = build_summary(
        study,
        generated_training_sha256={
            "v0": hashlib.sha256(v0_training).hexdigest(),
            "v2": hashlib.sha256(v2_training).hexdigest(),
        },
    )
    return {
        "all_o50_results.csv": build_all_results_csv(study),
        "reconciliation_ledger.json": _json_bytes(build_ledger(study)),
        "summary.json": _json_bytes(summary),
        "v0_training_loss_curves.csv": v0_training,
        "v2_training_loss_curves.csv": v2_training,
    }


def write_archive(
    study: ValidatedCompleteStudy,
    *,
    artifact_dir: str | Path,
    check: bool = False,
) -> tuple[Path, ...]:
    """Atomically write or byte-check the deterministic five-file archive."""

    root = Path(artifact_dir).expanduser().resolve()
    payloads = _archive_payloads(study)
    paths = tuple(root / name for name in ARCHIVE_FILENAMES)
    if check:
        if not root.is_dir():
            raise _error("archive", f"artifact directory is missing: {root}")
        actual_files = {path.name for path in root.iterdir() if path.is_file()}
        if actual_files != set(ARCHIVE_FILENAMES):
            raise _error("archive", "artifact directory file set is not exact")
        for name, payload in payloads.items():
            if (root / name).read_bytes() != payload:
                raise _error("archive", f"generated archive differs: {root / name}")
        return paths
    root.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in root.iterdir() if path.is_file()} - set(
        ARCHIVE_FILENAMES
    )
    if unexpected:
        raise _error(
            "archive", f"refusing to mix unexpected files: {sorted(unexpected)}"
        )
    for name, payload in payloads.items():
        _atomic_write(root / name, payload)
    return paths


__all__ = [
    "ACTION_NORMALIZATION_SHA256",
    "ARCHIVE_FILENAMES",
    "CSV_FIELDS",
    "CompleteReconciliationError",
    "EPISODE_COUNT",
    "SELECTION_SHA256",
    "ValidatedCompleteStudy",
    "build_all_results_csv",
    "build_ledger",
    "build_summary",
    "build_v0_training_csv",
    "build_v2_training_csv",
    "reconcile_complete_o50",
    "write_archive",
]
