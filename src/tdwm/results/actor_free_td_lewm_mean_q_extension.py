"""Extend the sealed 96+24 score ledger with 12 V0/V1 Mean-Q cells.

This module deliberately does not rebuild the historical 96-cell EMA sweep.
It accepts that compact ledger only under an explicit SHA-256 lock, validates a
fresh 12-cell launcher against its live evidence, and returns a deterministic
96+36 compact ledger.  The source ledger is never modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tdwm.results import actor_free_td_lewm_v2_ema_new_scores as prior_results

SCHEMA_VERSION = 1
EPISODES = 50
EPOCHS = tuple(range(3, 11))
VARIANTS = tuple(prior_results.VARIANTS)
EMA_SCORE_MODES = tuple(prior_results.SCORE_MODES)
FIRST_ACTION_MODE = EMA_SCORE_MODES[0]
ROLLOUT_MEAN_MODE = EMA_SCORE_MODES[1]
FIXED_VERSIONS = tuple(prior_results.FIXED_VERSIONS)
NEW_VERSIONS = ("v0", "v1")
FIRST_ACTION_WEIGHT = prior_results.FIRST_ACTION_WEIGHT
RAW_SELECTION_SHA256 = prior_results.SELECTION_SHA256
SELECTION_RANKS_SHA256 = prior_results.FIXED_SELECTION_SHA256
ACTION_NORMALIZATION_SHA256 = prior_results.FIXED_ACTION_NORMALIZATION_SHA256
REQUIRED_OUTPUT_FILES = tuple(prior_results.REQUIRED_OUTPUT_FILES)

_fixed = prior_results._fixed
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")


class MeanQExtensionError(ValueError):
    """The old ledger or the new 12-cell evidence failed closed validation."""


@dataclass(frozen=True)
class ExtendedCompactLedger:
    """A validated deterministic ledger payload and its provenance hashes."""

    document: Mapping[str, Any]
    payload: bytes
    sha256: str
    source_ledger_sha256: str
    launcher_manifest_sha256: str
    evaluation_commit: str


def _error(context: str, message: str) -> MeanQExtensionError:
    return MeanQExtensionError(f"{context}: {message}")


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error(context, "must be a lowercase SHA-256 digest")
    return value


def _git_revision(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise _error(context, "must be a full lowercase git revision")
    return value


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(context, "must be an object")
    return value


def _expect(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise _error(context, f"found {actual!r}, expected {expected!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise _error(context, f"missing file {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(context, f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(context, "JSON root must be an object")
    return value, raw


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(context, "must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(context, "must be finite")
    return result


def _validate_outcomes(cell: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    outcomes = cell.get("outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EPISODES
        or any(not isinstance(value, bool) for value in outcomes)
    ):
        raise _error(context, "outcomes must contain exactly 50 booleans")
    result = tuple(outcomes)
    success_count = sum(result)
    _expect(cell.get("success_count"), success_count, context=f"{context}.count")
    rate = _number(cell.get("success_rate"), context=f"{context}.rate")
    percent = _number(cell.get("success_rate_percent"), context=f"{context}.percent")
    if not math.isclose(rate, success_count / EPISODES, rel_tol=0.0, abs_tol=1e-12):
        raise _error(context, "success_rate differs from outcomes")
    if not math.isclose(percent, 2.0 * success_count, rel_tol=0.0, abs_tol=1e-12):
        raise _error(context, "success_rate_percent differs from outcomes")
    return result


def _validate_source_hashes(cell: Mapping[str, Any], *, context: str) -> dict[str, str]:
    raw = _mapping(cell.get("source_files_sha256"), context=f"{context}.hashes")
    if set(raw) != set(REQUIRED_OUTPUT_FILES):
        raise _error(context, "source hashes must cover the exact four evidence files")
    hashes = {
        name: _sha256(raw[name], context=f"{context}.hashes.{name}")
        for name in REQUIRED_OUTPUT_FILES
    }
    _expect(
        hashes["episode_selection.json"],
        RAW_SELECTION_SHA256,
        context=f"{context}.selection_file_sha256",
    )
    _expect(
        hashes["action_normalization.json"],
        ACTION_NORMALIZATION_SHA256,
        context=f"{context}.action_sha256",
    )
    return hashes


def _expected_ema_grid() -> set[tuple[int, str, str]]:
    return {
        (epoch, variant, mode)
        for epoch in EPOCHS
        for variant in VARIANTS
        for mode in EMA_SCORE_MODES
    }


def _old_fixed_grid() -> set[tuple[str, str, str]]:
    return {
        (version, variant, FIRST_ACTION_MODE)
        for version in FIXED_VERSIONS
        for variant in VARIANTS
    } | {("v2", variant, ROLLOUT_MEAN_MODE) for variant in VARIANTS}


def _new_fixed_grid() -> set[tuple[str, str, str]]:
    return {
        (version, variant, ROLLOUT_MEAN_MODE)
        for version in NEW_VERSIONS
        for variant in VARIANTS
    }


def _complete_fixed_grid() -> set[tuple[str, str, str]]:
    return {
        (version, variant, mode)
        for version in FIXED_VERSIONS
        for variant in VARIANTS
        for mode in EMA_SCORE_MODES
    }


def _fixed_job_id(version: str, variant: str, mode: str) -> str:
    suffix = "__alpha_0p25" if mode == FIRST_ACTION_MODE else ""
    return f"{version}__{variant}__{mode}{suffix}"


def _validate_old_compact_ledger(
    ledger: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    context = "old_ledger"
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "source": "actor_free_td_lewm_v2_ema_new_score_reconciliation",
        "cell_count": 96,
        "epochs": list(EPOCHS),
        "variants": list(VARIANTS),
        "score_modes": list(EMA_SCORE_MODES),
        "g_first_weight": FIRST_ACTION_WEIGHT,
        "selection_sha256": RAW_SELECTION_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
    }
    for key, expected in expected_header.items():
        _expect(ledger.get(key), expected, context=f"{context}.{key}")
    sweep_commit = _git_revision(
        ledger.get("evaluation_commit"), context=f"{context}.evaluation_commit"
    )
    _git_revision(ledger.get("training_commit"), context=f"{context}.training_commit")
    _mapping(ledger.get("sources"), context=f"{context}.sources")

    raw_cells = ledger.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 96:
        raise _error(context, "cells must contain exactly the historical 96 entries")
    indexed: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    outputs: set[str] = set()
    expected_ema_grid = _expected_ema_grid()
    for index, value in enumerate(raw_cells):
        cell_context = f"{context}.cells[{index}]"
        cell = _mapping(value, context=cell_context)
        key = (cell.get("epoch"), cell.get("variant"), cell.get("score_mode"))
        if key in indexed:
            raise _error(cell_context, f"duplicate identity {key!r}")
        if key not in expected_ema_grid:
            raise _error(cell_context, f"unexpected identity {key!r}")
        indexed[key] = cell
        _validate_outcomes(cell, context=cell_context)
        _validate_source_hashes(cell, context=cell_context)
        epoch, variant, mode = key
        expected_id = f"v2_ema_e{epoch:02d}_{variant}_{mode}" + (
            "_alpha_0p25" if mode == FIRST_ACTION_MODE else ""
        )
        _expect(cell.get("cell_id"), expected_id, context=f"{cell_context}.cell_id")
        _expect(
            cell.get("method"),
            f"actor_free_td_lewm_v2_ema_sg_{variant}",
            context=f"{cell_context}.method",
        )
        _expect(
            cell.get("selection_sha256"),
            RAW_SELECTION_SHA256,
            context=f"{cell_context}.selection",
        )
        _expect(
            cell.get("action_normalization_sha256"),
            ACTION_NORMALIZATION_SHA256,
            context=f"{cell_context}.action",
        )
        _expect(
            cell.get("evaluation_commit"),
            sweep_commit,
            context=f"{cell_context}.evaluation_commit",
        )
        expected_alpha = FIRST_ACTION_WEIGHT if mode == FIRST_ACTION_MODE else None
        _expect(
            cell.get("g_first_weight"),
            expected_alpha,
            context=f"{cell_context}.alpha",
        )
        _expect(cell.get("checkpoint_epoch"), epoch, context=f"{cell_context}.epoch")
        _expect(
            cell.get("checkpoint_global_step"),
            epoch * prior_results.STEPS_PER_EPOCH,
            context=f"{cell_context}.global_step",
        )
        _sha256(cell.get("checkpoint_sha256"), context=f"{cell_context}.checkpoint")
        _sha256(
            cell.get("training_manifest_sha256"),
            context=f"{cell_context}.training_manifest",
        )
        if cell.get("source_status") not in prior_results.TERMINAL_STATES:
            raise _error(cell_context, "source status is not terminal")
        output = cell.get("output_dir")
        if not isinstance(output, str) or not output or output in outputs:
            raise _error(cell_context, "output_dir is empty or duplicated")
        outputs.add(output)
    if set(indexed) != expected_ema_grid:
        raise _error(context, "EMA new-score grid is not exactly 8x6x2")

    fixed = _mapping(
        ledger.get("fixed_checkpoint_comparison"), context=f"{context}.fixed"
    )
    expected_fixed_header = {
        "included": True,
        "cell_count": 24,
        "selection_sha256": SELECTION_RANKS_SHA256,
        "episode_selection_file_sha256": RAW_SELECTION_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
    }
    for key, expected in expected_fixed_header.items():
        _expect(fixed.get(key), expected, context=f"{context}.fixed.{key}")
    launcher_sources = fixed.get("launcher_sources")
    if not isinstance(launcher_sources, list) or len(launcher_sources) != 3:
        raise _error(context, "old fixed section must retain exactly three launchers")
    launcher_job_count = 0
    for index, value in enumerate(launcher_sources):
        source = _mapping(value, context=f"{context}.fixed.launchers[{index}]")
        _sha256(
            source.get("manifest_sha256"),
            context=f"{context}.fixed.launchers[{index}].manifest_sha256",
        )
        count = source.get("job_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise _error(context, "old launcher job_count must be positive integers")
        launcher_job_count += count
    if launcher_job_count != 24:
        raise _error(context, "old launcher sources do not account for 24 jobs")

    raw_fixed = fixed.get("cells")
    if not isinstance(raw_fixed, list) or len(raw_fixed) != 24:
        raise _error(context, "old fixed cells must contain exactly 24 entries")
    fixed_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    fixed_outputs: set[str] = set()
    expected_old_fixed_grid = _old_fixed_grid()
    for index, value in enumerate(raw_fixed):
        cell_context = f"{context}.fixed.cells[{index}]"
        cell = _mapping(value, context=cell_context)
        key = (cell.get("version"), cell.get("variant"), cell.get("score_mode"))
        if key in fixed_index:
            raise _error(cell_context, f"duplicate identity {key!r}")
        if key not in expected_old_fixed_grid:
            raise _error(cell_context, f"unexpected identity {key!r}")
        fixed_index[key] = cell
        version, variant, mode = key
        _expect(
            cell.get("job_id"),
            _fixed_job_id(version, variant, mode),
            context=f"{cell_context}.job_id",
        )
        _expect(
            cell.get("method"),
            f"actor_free_td_lewm_{version}_{variant}",
            context=f"{cell_context}.method",
        )
        _expect(
            cell.get("selection_sha256"),
            SELECTION_RANKS_SHA256,
            context=f"{cell_context}.selection_ranks",
        )
        _expect(
            cell.get("episode_selection_file_sha256"),
            RAW_SELECTION_SHA256,
            context=f"{cell_context}.selection_file",
        )
        _expect(
            cell.get("action_normalization_sha256"),
            ACTION_NORMALIZATION_SHA256,
            context=f"{cell_context}.action",
        )
        expected_alpha = FIRST_ACTION_WEIGHT if mode == FIRST_ACTION_MODE else None
        _expect(
            cell.get("g_first_weight"), expected_alpha, context=f"{cell_context}.alpha"
        )
        _git_revision(
            cell.get("evaluation_commit"),
            context=f"{cell_context}.evaluation_commit",
        )
        _mapping(
            cell.get("evaluation_commit_evidence"),
            context=f"{cell_context}.evaluation_commit_evidence",
        )
        _validate_outcomes(cell, context=cell_context)
        _validate_source_hashes(cell, context=cell_context)
        _sha256(cell.get("checkpoint_sha256"), context=f"{cell_context}.checkpoint")
        checkpoint_path = cell.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise _error(cell_context, "checkpoint_path is missing")
        output = cell.get("output_dir")
        if not isinstance(output, str) or not output or output in fixed_outputs:
            raise _error(cell_context, "output_dir is empty or duplicated")
        fixed_outputs.add(output)
        if not isinstance(cell.get("source_launcher_manifest"), str):
            raise _error(cell_context, "source_launcher_manifest is missing")
    if set(fixed_index) != expected_old_fixed_grid:
        raise _error(context, "old fixed grid is not exactly 18 first-Q + 6 V2 Mean-Q")
    for variant in VARIANTS:
        first = fixed_index[("v2", variant, FIRST_ACTION_MODE)]
        mean = fixed_index[("v2", variant, ROLLOUT_MEAN_MODE)]
        for field in ("checkpoint_path", "checkpoint_sha256"):
            _expect(
                mean.get(field),
                first.get(field),
                context=f"{context}.fixed.v2.{variant}.{field}",
            )
    return {
        (version, variant): fixed_index[(version, variant, FIRST_ACTION_MODE)]
        for version in NEW_VERSIONS
        for variant in VARIANTS
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _recorded_path(value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error(context, "must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise _error(context, "must be an absolute recorded path")
    return path


def _recorded_relative_path(value: Any, root: Path, *, context: str) -> Path:
    path = _recorded_path(value, context=context)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise _error(context, f"recorded path is outside {root}") from exc
    if not relative.parts or ".." in relative.parts:
        raise _error(context, "recorded path has an unsafe relative form")
    return relative


def _git_head(repository: Path) -> str:
    if not repository.is_dir():
        raise _error("evaluation_checkout", f"missing directory {repository}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _error("evaluation_checkout", "cannot resolve git HEAD") from exc
    return _git_revision(revision, context="evaluation_checkout.HEAD")


def _require_checkout_root(repository: Path) -> None:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _error("evaluation_checkout", "cannot resolve repository root") from exc
    if Path(top_level).resolve() != repository.resolve():
        raise _error(
            "evaluation_checkout",
            "must be the top level of the supplied git worktree",
        )


def _require_tracked_clean(repository: Path) -> None:
    try:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _error("evaluation_checkout", "cannot inspect worktree state") from exc
    if dirty.strip():
        raise _error("evaluation_checkout", "worktree is not completely clean")


def _require_tracked_file(
    repository: Path,
    relative_path: Path,
    *,
    context: str,
) -> Path:
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise _error(context, "repository-relative path is unsafe")
    lexical_path = repository / relative_path
    if lexical_path.is_symlink():
        raise _error(context, "must be a regular tracked file, not a symlink")
    resolved_path = lexical_path.resolve()
    if not _path_is_within(resolved_path, repository) or not resolved_path.is_file():
        raise _error(context, "is missing or resolves outside the supplied checkout")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "--error-unmatch",
                "--",
                relative_path.as_posix(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _error(context, "is not tracked by the supplied checkout") from exc
    return resolved_path


def _require_matching_validator(checkout: Path) -> Path:
    active_validator = Path(str(_fixed.__file__)).resolve()
    active_repository = Path(__file__).resolve().parents[3]
    _require_checkout_root(active_repository)
    if not _path_is_within(active_validator, active_repository):
        raise _error(
            "evaluation_checkout.active_validator",
            "validator module resolves outside its repository",
        )
    validator_relative = active_validator.relative_to(active_repository)
    checkout_validator = _require_tracked_file(
        checkout,
        validator_relative,
        context="evaluation_checkout.validator",
    )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(active_repository),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                validator_relative.as_posix(),
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _error(
            "evaluation_checkout.active_validator",
            "validator differs from its repository HEAD",
        ) from exc
    _expect(
        _file_sha256(active_validator),
        _file_sha256(checkout_validator),
        context="evaluation_checkout.validator_sha256",
    )
    return validator_relative


def _optional_commit_evidence(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected: str,
    context: str,
) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for document, label in ((result, "results"), (manifest, "manifest")):
        if "evaluation_commit" not in document:
            evidence[label] = "absent"
        elif document["evaluation_commit"] == expected:
            evidence[label] = "present_and_matched"
        else:
            raise _error(
                context,
                f"{label}.evaluation_commit conflicts with checkout HEAD {expected}",
            )
    return evidence


def _validate_new_launcher(
    *,
    launcher_root: Path,
    launcher_manifest: Path,
    evaluation_checkout: Path,
    old_first_q: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], str, str]:
    context = "new_launcher"
    root = launcher_root.expanduser().resolve()
    manifest_path = launcher_manifest.expanduser().resolve()
    checkout = evaluation_checkout.expanduser().resolve()
    if not root.is_dir():
        raise _error(context, f"launcher root is missing: {root}")
    if not _path_is_within(manifest_path, root):
        raise _error(context, "launcher manifest is outside its supplied root")
    _expect(
        manifest_path.relative_to(root),
        Path("_launcher/launcher_manifest.json"),
        context=f"{context}.manifest_relative_path",
    )
    _require_checkout_root(checkout)
    _require_tracked_clean(checkout)
    checkout_head = _git_head(checkout)
    validator_relative = _require_matching_validator(checkout)
    manifest, manifest_raw = _read_json(manifest_path, context=context)
    expected_header = {
        "schema_version": 1,
        "launcher": "actor_free_td_lewm_first_action_comparison",
        "inference_only": True,
        "training_performed": False,
        "alpha_selection_performed": False,
        "stage": "formal",
        "status": "SUCCEEDED",
        "shared_score_modes": [ROLLOUT_MEAN_MODE],
        "v2_only_score_modes": [],
        "alphas": [],
        "score_modes_by_version": {
            version: [ROLLOUT_MEAN_MODE] for version in NEW_VERSIONS
        },
    }
    for key, expected in expected_header.items():
        _expect(manifest.get(key), expected, context=f"{context}.{key}")
    versions = manifest.get("versions")
    variants = manifest.get("variants")
    if not isinstance(versions, list) or set(versions) != set(NEW_VERSIONS):
        raise _error(context, "versions must be exactly V0 and V1")
    if len(versions) != len(NEW_VERSIONS):
        raise _error(context, "versions contain duplicates")
    if not isinstance(variants, list) or set(variants) != set(VARIANTS):
        raise _error(context, "variants must be exactly C/D/F/G1/G2/G3")
    if len(variants) != len(VARIANTS):
        raise _error(context, "variants contain duplicates")
    recorded_repository = _recorded_path(
        manifest.get("repository"), context=f"{context}.repository"
    )
    expected_raw_selection = manifest.get("expected_selection_file_sha256")
    _expect(
        expected_raw_selection,
        RAW_SELECTION_SHA256,
        context=f"{context}.expected_selection_file_sha256",
    )
    recorded_output_root = _recorded_path(
        manifest.get("output_root"), context=f"{context}.output_root"
    )
    recorded_stage_root = recorded_output_root / "formal"
    recorded_manifest_path = recorded_stage_root / "_launcher/launcher_manifest.json"

    selection_summary = _mapping(
        manifest.get("selection"), context=f"{context}.selection"
    )
    _expect(
        selection_summary.get("valid_row_ranks_sha256"),
        SELECTION_RANKS_SHA256,
        context=f"{context}.selection.ranks_sha256",
    )
    _expect(
        selection_summary.get("selection_file_sha256"),
        RAW_SELECTION_SHA256,
        context=f"{context}.selection.file_sha256",
    )
    _expect(
        selection_summary.get("identical_across_all_jobs"),
        True,
        context=f"{context}.selection.identical",
    )
    _expect(
        selection_summary.get("selection_file_identical_across_all_jobs"),
        True,
        context=f"{context}.selection.file_identical",
    )
    summary_ranks = selection_summary.get("valid_row_ranks")
    if (
        not isinstance(summary_ranks, list)
        or len(summary_ranks) != EPISODES
        or any(
            isinstance(rank, bool) or not isinstance(rank, int)
            for rank in summary_ranks
        )
    ):
        raise _error(context, "launcher selection must contain 50 integer ranks")
    _expect(
        _fixed.canonical_json_sha256(summary_ranks),
        SELECTION_RANKS_SHA256,
        context=f"{context}.selection.canonical_sha256",
    )

    jobs = _mapping(manifest.get("jobs"), context=f"{context}.jobs")
    expected_job_ids = {
        _fixed_job_id(version, variant, ROLLOUT_MEAN_MODE)
        for version in NEW_VERSIONS
        for variant in VARIANTS
    }
    if set(jobs) != expected_job_ids:
        raise _error(
            context,
            "launcher is not the exact 12-cell V0/V1 Mean-Q grid; "
            f"missing={sorted(expected_job_ids - set(jobs))}, "
            f"extra={sorted(set(jobs) - expected_job_ids)}",
        )

    cells: list[Mapping[str, Any]] = []
    outputs: set[str] = set()
    for version in NEW_VERSIONS:
        for variant in VARIANTS:
            job_id = _fixed_job_id(version, variant, ROLLOUT_MEAN_MODE)
            cell_context = f"{context}.{job_id}"
            raw_job = _mapping(jobs[job_id], context=cell_context)
            _expect(raw_job.get("job_id"), job_id, context=f"{cell_context}.job_id")
            _expect(raw_job.get("state"), "SUCCEEDED", context=f"{cell_context}.state")
            _expect(raw_job.get("exit_code"), 0, context=f"{cell_context}.exit_code")
            _expect(raw_job.get("stage"), "formal", context=f"{cell_context}.stage")
            _expect(raw_job.get("version"), version, context=f"{cell_context}.version")
            _expect(raw_job.get("variant"), variant, context=f"{cell_context}.variant")
            _expect(
                raw_job.get("score_mode"),
                ROLLOUT_MEAN_MODE,
                context=f"{cell_context}.score_mode",
            )
            _expect(raw_job.get("alpha"), None, context=f"{cell_context}.alpha")
            argv = raw_job.get("argv")
            if (
                not isinstance(argv, list)
                or len(argv) < 2
                or any(not isinstance(value, str) for value in argv)
            ):
                raise _error(cell_context, "argv is malformed")
            evaluator_relative = _recorded_relative_path(
                argv[1], recorded_repository, context=f"{cell_context}.evaluator"
            )
            expected_relative_evaluator = (
                Path("scripts") / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
            )
            _expect(
                evaluator_relative,
                expected_relative_evaluator,
                context=f"{cell_context}.evaluator_relative_path",
            )
            recorded_evaluator = recorded_repository / evaluator_relative
            _require_tracked_file(
                checkout,
                expected_relative_evaluator,
                context=f"{cell_context}.evaluation_checkout_evaluator",
            )
            config_relative = _recorded_relative_path(
                raw_job.get("config_path"),
                recorded_repository,
                context=f"{cell_context}.config_path",
            )
            expected_config_relative = Path("configs/experiment") / (
                f"actor_free_td_lewm_{version}_{variant}_cube_checkpoint_o50_"
                f"{ROLLOUT_MEAN_MODE}.yaml"
            )
            _expect(
                config_relative,
                expected_config_relative,
                context=f"{cell_context}.config_relative_path",
            )
            validation_config = _require_tracked_file(
                checkout,
                expected_config_relative,
                context=f"{cell_context}.evaluation_checkout_config",
            )
            output_relative = _recorded_relative_path(
                raw_job.get("output_dir"),
                recorded_stage_root,
                context=f"{cell_context}.output_dir",
            )
            expected_output_relative = Path(version) / variant / ROLLOUT_MEAN_MODE
            _expect(
                output_relative,
                expected_output_relative,
                context=f"{cell_context}.output_relative_path",
            )
            recorded_output = recorded_stage_root / output_relative
            validation_output = (root / output_relative).resolve()
            if (
                not _path_is_within(validation_output, root)
                or str(recorded_output) in outputs
            ):
                raise _error(
                    cell_context, "output_dir is outside the root or duplicated"
                )
            outputs.add(str(recorded_output))

            old_reference = old_first_q[(version, variant)]
            checkpoint = _recorded_path(
                raw_job.get("checkpoint"), context=f"{cell_context}.checkpoint"
            )
            _expect(
                str(checkpoint),
                str(
                    _recorded_path(
                        old_reference["checkpoint_path"],
                        context=f"{cell_context}.old_checkpoint_path",
                    )
                ),
                context=f"{cell_context}.checkpoint_path_crosslink",
            )
            checkpoint_sha = str(old_reference["checkpoint_sha256"])
            job = _fixed.Job(
                job_id=job_id,
                stage="formal",
                version=version,
                variant=variant,
                score_mode=ROLLOUT_MEAN_MODE,
                alpha=None,
                checkpoint=str(checkpoint),
                config_path=str(validation_config),
                output_dir=str(validation_output),
                log_path=str(raw_job.get("log_path", "")),
                argv=tuple(argv),
            )
            before_hashes = {
                name: _file_sha256(validation_output / name)
                for name in REQUIRED_OUTPUT_FILES
            }
            try:
                validation = _fixed.validate_job_output(
                    job,
                    expected_selection_file_sha256=RAW_SELECTION_SHA256,
                )
            except Exception as exc:
                raise _error(
                    cell_context, f"launcher output validation failed: {exc}"
                ) from exc
            recorded_evidence = _mapping(
                raw_job.get("evidence"), context=f"{cell_context}.launcher_evidence"
            )
            for name, filename in (
                ("results_path", "results.json"),
                ("manifest_path", "protocol_manifest.json"),
                ("selection_path", "episode_selection.json"),
            ):
                _expect(
                    recorded_evidence.get(name),
                    str(recorded_output / filename),
                    context=f"{cell_context}.launcher_evidence.{name}",
                )
            for name in (
                "selection_file_sha256",
                "valid_row_ranks",
                "valid_row_ranks_sha256",
                "selection_sha256",
            ):
                _expect(
                    recorded_evidence.get(name),
                    validation.get(name),
                    context=f"{cell_context}.launcher_evidence.{name}",
                )
            result, _ = _read_json(
                validation_output / "results.json", context=f"{cell_context}.results"
            )
            protocol, _ = _read_json(
                validation_output / "protocol_manifest.json",
                context=f"{cell_context}.protocol",
            )
            selection, _ = _read_json(
                validation_output / "episode_selection.json",
                context=f"{cell_context}.selection",
            )
            action, _ = _read_json(
                validation_output / "action_normalization.json",
                context=f"{cell_context}.action",
            )
            after_hashes = {
                name: _file_sha256(validation_output / name)
                for name in REQUIRED_OUTPUT_FILES
            }
            _expect(
                after_hashes, before_hashes, context=f"{cell_context}.stable_hashes"
            )
            _expect(
                after_hashes["episode_selection.json"],
                RAW_SELECTION_SHA256,
                context=f"{cell_context}.selection_file_sha256",
            )
            _expect(
                after_hashes["action_normalization.json"],
                ACTION_NORMALIZATION_SHA256,
                context=f"{cell_context}.action_sha256",
            )
            _expect(
                protocol.get("selection"),
                selection,
                context=f"{cell_context}.protocol_selection",
            )
            normalization = _mapping(
                protocol.get("normalization"), context=f"{cell_context}.normalization"
            )
            _expect(
                normalization.get("action"),
                action,
                context=f"{cell_context}.protocol_action",
            )
            checkpoint_evidence = _mapping(
                protocol.get("checkpoint"),
                context=f"{cell_context}.protocol_checkpoint",
            )
            _expect(
                checkpoint_evidence.get("path"),
                str(checkpoint),
                context=f"{cell_context}.protocol_checkpoint.path",
            )
            _expect(
                checkpoint_evidence.get("sha256"),
                checkpoint_sha,
                context=f"{cell_context}.protocol_checkpoint.sha256",
            )
            _expect(
                checkpoint_evidence.get("epoch"),
                10,
                context=f"{cell_context}.protocol_checkpoint.epoch",
            )
            _expect(
                checkpoint_evidence.get("global_step"),
                10 * prior_results.STEPS_PER_EPOCH,
                context=f"{cell_context}.protocol_checkpoint.global_step",
            )
            runtime = _mapping(
                protocol.get("runtime"), context=f"{cell_context}.protocol_runtime"
            )
            runtime_revision = _git_revision(
                runtime.get("tdwm_git_revision"),
                context=f"{cell_context}.protocol_runtime.tdwm_git_revision",
            )
            _expect(
                runtime_revision,
                checkout_head,
                context=f"{cell_context}.protocol_runtime.tdwm_git_revision",
            )
            commit_fields = _optional_commit_evidence(
                result,
                protocol,
                expected=checkout_head,
                context=cell_context,
            )
            metrics = _mapping(result.get("metrics"), context=f"{cell_context}.metrics")
            outcomes = metrics.get("episode_successes")
            outcome_cell = {
                "outcomes": outcomes,
                "success_count": sum(outcomes) if isinstance(outcomes, list) else None,
                "success_rate": (
                    sum(outcomes) / EPISODES if isinstance(outcomes, list) else None
                ),
                "success_rate_percent": (
                    2.0 * sum(outcomes) if isinstance(outcomes, list) else None
                ),
            }
            validated_outcomes = _validate_outcomes(outcome_cell, context=cell_context)
            success_count = sum(validated_outcomes)
            cells.append(
                {
                    "version": version,
                    "variant": variant,
                    "score_mode": ROLLOUT_MEAN_MODE,
                    "g_first_weight": None,
                    "job_id": job_id,
                    "method": f"actor_free_td_lewm_{version}_{variant}",
                    "outcomes": list(validated_outcomes),
                    "success_count": success_count,
                    "success_rate": success_count / EPISODES,
                    "success_rate_percent": 2.0 * success_count,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "selection_sha256": SELECTION_RANKS_SHA256,
                    "episode_selection_file_sha256": RAW_SELECTION_SHA256,
                    "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
                    "evaluation_commit": checkout_head,
                    "evaluation_commit_evidence": {
                        "source": "launcher_repository_checkout_head",
                        "launcher_repository": str(recorded_repository),
                        "checkout_head": checkout_head,
                        "protocol_runtime_tdwm_git_revision": runtime_revision,
                        "evaluator_path": str(recorded_evaluator),
                        "result_field": commit_fields["results"],
                        "manifest_field": commit_fields["manifest"],
                    },
                    "source_launcher_manifest": str(recorded_manifest_path),
                    "output_dir": str(recorded_output),
                    "source_files_sha256": after_hashes,
                }
            )
    _require_tracked_clean(checkout)
    if _git_head(checkout) != checkout_head:
        raise _error(context, "evaluation checkout HEAD changed during validation")
    source_evidence = {
        "root": str(recorded_stage_root),
        "manifest": str(recorded_manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "job_count": len(cells),
        "evaluation_checkout": str(recorded_repository),
        "evaluation_checkout_head": checkout_head,
        "recorded_repository_from_launcher_manifest": str(recorded_repository),
        "protocol_runtime_commit_matched_for_all_jobs": True,
        "reference_checkout_was_clean_at_validation": True,
        "validator_relative_path": validator_relative.as_posix(),
        "validation_input": "caller_supplied_launcher_root_mapped_by_recorded_paths",
    }
    return (
        source_evidence,
        tuple(cells),
        source_evidence["manifest_sha256"],
        checkout_head,
    )


def _fixed_sort_key(cell: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        FIXED_VERSIONS.index(str(cell["version"])),
        VARIANTS.index(str(cell["variant"])),
        EMA_SCORE_MODES.index(str(cell["score_mode"])),
    )


def _ledger_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def extend_compact_ledger(
    *,
    old_ledger_path: str | Path,
    expected_old_ledger_sha256: str,
    launcher_root: str | Path,
    launcher_manifest: str | Path,
    evaluation_checkout: str | Path,
) -> ExtendedCompactLedger:
    """Validate the inputs and return a sealed 96+36 compact ledger payload."""

    old_path = Path(old_ledger_path).expanduser().resolve()
    expected_old_sha = _sha256(
        expected_old_ledger_sha256, context="old_ledger.expected_sha256"
    )
    old_document, old_raw = _read_json(old_path, context="old_ledger")
    actual_old_sha = hashlib.sha256(old_raw).hexdigest()
    _expect(actual_old_sha, expected_old_sha, context="old_ledger.sha256")
    old_first_q = _validate_old_compact_ledger(old_document)
    source, new_cells, manifest_sha, evaluation_commit = _validate_new_launcher(
        launcher_root=Path(launcher_root),
        launcher_manifest=Path(launcher_manifest),
        evaluation_checkout=Path(evaluation_checkout),
        old_first_q=old_first_q,
    )

    document = copy.deepcopy(old_document)
    fixed = document["fixed_checkpoint_comparison"]
    combined = list(fixed["cells"]) + [dict(cell) for cell in new_cells]
    identities = {
        (cell.get("version"), cell.get("variant"), cell.get("score_mode"))
        for cell in combined
    }
    if len(combined) != 36 or identities != _complete_fixed_grid():
        raise _error("extended_ledger", "fixed grid is not exactly 3x6x2")
    fixed["cells"] = sorted(combined, key=_fixed_sort_key)
    fixed["cell_count"] = 36
    fixed["launcher_sources"] = list(fixed["launcher_sources"]) + [dict(source)]
    commits = sorted({str(cell["evaluation_commit"]) for cell in combined})
    fixed["evaluation_commit"] = commits[0] if len(commits) == 1 else None
    fixed["evaluation_commits"] = commits
    fixed["evaluation_commit_evidence_source"] = "per_launcher_repository_checkout_head"
    fixed["extension_provenance"] = {
        "kind": "v0_v1_fixed_mean_q_12_cell_extension",
        "source_ledger_sha256": actual_old_sha,
        "source_fixed_checkpoint_cell_count": 24,
        "added_fixed_checkpoint_cell_count": 12,
        "launcher_manifest_sha256": manifest_sha,
        "evaluation_commit": evaluation_commit,
    }
    payload = _ledger_bytes(document)
    return ExtendedCompactLedger(
        document=document,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_ledger_sha256=actual_old_sha,
        launcher_manifest_sha256=manifest_sha,
        evaluation_commit=evaluation_commit,
    )


def write_extended_ledger(
    result: ExtendedCompactLedger,
    *,
    output_path: str | Path,
    old_ledger_path: str | Path,
) -> Path:
    """Atomically create a new ledger without overwriting either ledger path."""

    output = Path(output_path).expanduser().resolve()
    old = Path(old_ledger_path).expanduser().resolve()
    if output == old:
        raise _error("output", "refusing to overwrite the old compact ledger")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(result.payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, output)
        except FileExistsError as exc:
            raise _error(
                "output", f"refusing to overwrite existing file {output}"
            ) from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output


__all__ = [
    "ACTION_NORMALIZATION_SHA256",
    "ExtendedCompactLedger",
    "MeanQExtensionError",
    "RAW_SELECTION_SHA256",
    "SELECTION_RANKS_SHA256",
    "extend_compact_ledger",
    "write_extended_ledger",
]
