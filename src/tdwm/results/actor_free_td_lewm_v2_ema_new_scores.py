"""Fail-closed reconciliation for the 96-cell V2-EMA new-score sweep.

The formal sweep was intentionally split across two scheduler roots: the
strict epoch-3 replacement and the epoch-4-through-10 continuation.  This
module accepts only that exact, disjoint 12 + 84 partition.  It validates the
live source artifacts and writes a lightweight archive; raw evaluator output,
checkpoints and logs are never copied.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from tdwm.results import actor_free_td_lewm_v2_ema_sg as ema_results

SCHEMA_VERSION = 1
TRAINING_COMMIT = "18cd574d522515f20f4103509b1e660b2fc89ea6"
EVALUATION_COMMIT = "5456f3d18116812d078d4ec2e85ba1f83d89c7c7"
SELECTION_SHA256 = "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
SCORE_MODES = ("f_plus_g_first", "g_only_f_rollout_mean")
FIXED_VERSIONS = ("v0", "v1", "v2")
FIXED_SELECTION_SHA256 = (
    "88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9"
)
FIXED_ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)
EPOCHS = tuple(range(3, 11))
FIRST_ACTION_WEIGHT = 0.25
EPISODES = 50
STEPS_PER_EPOCH = 12_796
TERMINAL_STATES = frozenset({"REUSED", "SUCCEEDED"})
REQUIRED_OUTPUT_FILES = (
    "results.json",
    "protocol_manifest.json",
    "episode_selection.json",
    "action_normalization.json",
)
ARCHIVE_FILENAMES = (
    "reconciliation_ledger.json",
    "results.csv",
    "fixed_checkpoint_results.csv",
    "summary.json",
)


class NewScoreReconciliationError(ValueError):
    """The split formal sweep cannot be accepted without inventing evidence."""


@dataclass(frozen=True)
class ValidatedNewScoreStudy:
    """Validated source provenance and the exact 96 reconciled cells."""

    strict_epoch3_root: Path
    strict_epoch3_state: Path
    original_epoch4_10_root: Path
    original_epoch4_10_state: Path
    state_sha256: Mapping[str, str]
    cells: tuple[Mapping[str, Any], ...]
    action_normalization_sha256: str
    fixed_launcher_sources: tuple[Mapping[str, Any], ...] = ()
    fixed_cells: tuple[Mapping[str, Any], ...] = ()


def _load_sweep_validator() -> ModuleType:
    module_name = f"{__name__}._new_score_sweep_validator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    repository_root = Path(__file__).resolve().parents[3]
    source = (
        repository_root
        / "scripts"
        / "run_actor_free_td_lewm_v2_ema_new_score_sweeps.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the new-score sweep validator from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_sweeps = _load_sweep_validator()


def _load_fixed_validator() -> ModuleType:
    module_name = f"{__name__}._first_action_comparison_validator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    repository_root = Path(__file__).resolve().parents[3]
    source = (
        repository_root
        / "scripts"
        / "run_actor_free_td_lewm_first_action_comparison.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the fixed-score validator from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_fixed = _load_fixed_validator()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise NewScoreReconciliationError(f"{context}: missing {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NewScoreReconciliationError(
            f"{context}: cannot parse JSON at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NewScoreReconciliationError(f"{context}: JSON root must be an object")
    return value, raw


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NewScoreReconciliationError(f"{context}: expected an object")
    return value


def _same_path(left: Any, right: Path) -> bool:
    return (
        isinstance(left, str) and Path(left).expanduser().resolve() == right.resolve()
    )


def _expected_cell_id(epoch: int, variant: str, score_mode: str) -> str:
    suffix = score_mode
    if score_mode == SCORE_MODES[0]:
        suffix += "_alpha_0p25"
    return f"v2_ema_e{epoch:02d}_{variant}_{suffix}"


def _expected_output(root: Path, epoch: int, variant: str, score_mode: str) -> Path:
    output = root / "new_score_evaluation_sweeps" / f"epoch_{epoch:02d}" / variant
    if score_mode == SCORE_MODES[0]:
        return output / score_mode / "alpha_0p25"
    return output / score_mode


def _expected_grid(epochs: Sequence[int]) -> set[str]:
    return {
        _expected_cell_id(epoch, variant, score_mode)
        for epoch in epochs
        for variant in VARIANTS
        for score_mode in SCORE_MODES
    }


def _argv_value(argv: Sequence[str], option: str) -> str | None:
    if option not in argv:
        return None
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise NewScoreReconciliationError(f"argv has no value after {option}")
    return argv[index + 1]


def _validate_state_header(
    state: Mapping[str, Any],
    *,
    root: Path,
    state_path: Path,
    expected_epochs: Sequence[int],
) -> Mapping[str, Mapping[str, Any]]:
    context = str(state_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source": "actor_free_td_lewm_v2_ema_new_score_sweep_scheduler",
        "version_key": "v2_ema",
        "version_display_name": "V2 EMA",
        "training_commit": TRAINING_COMMIT,
        "expected_evaluation_commit": EVALUATION_COMMIT,
        "expected_selection_sha256": SELECTION_SHA256,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise NewScoreReconciliationError(
                f"{context}: {key}={state.get(key)!r}, expected {value!r}"
            )
    if state.get("score_modes") != list(SCORE_MODES):
        raise NewScoreReconciliationError(
            f"{context}: score_modes must be exactly {list(SCORE_MODES)!r}"
        )
    paths = _mapping(state.get("paths"), context=f"{context}.paths")
    if not _same_path(paths.get("bundle_root"), root):
        raise NewScoreReconciliationError(
            f"{context}: paths.bundle_root does not bind the supplied root"
        )
    expected_sweep = root / "new_score_evaluation_sweeps"
    if not _same_path(paths.get("sweep_root"), expected_sweep):
        raise NewScoreReconciliationError(
            f"{context}: paths.sweep_root does not bind the canonical output root"
        )
    cells = _mapping(state.get("cells"), context=f"{context}.cells")
    expected_grid = _expected_grid(expected_epochs)
    if set(cells) != expected_grid:
        missing = sorted(expected_grid - set(cells))
        extra = sorted(set(cells) - expected_grid)
        raise NewScoreReconciliationError(
            f"{context}: state grid mismatch; missing={missing}, extra={extra}"
        )
    if state.get("cell_count") != len(expected_grid):
        raise NewScoreReconciliationError(
            f"{context}: cell_count must be {len(expected_grid)}"
        )
    actual_counts: dict[str, int] = {}
    for cell in cells.values():
        cell_mapping = _mapping(cell, context=f"{context}.cells entry")
        label = str(cell_mapping.get("state"))
        actual_counts[label] = actual_counts.get(label, 0) + 1
    if state.get("state_counts") != actual_counts:
        raise NewScoreReconciliationError(f"{context}: state_counts is stale")
    unexpected_states = set(actual_counts) - TERMINAL_STATES
    if unexpected_states:
        raise NewScoreReconciliationError(
            f"{context}: non-terminal states remain: {sorted(unexpected_states)}"
        )
    return cells


def _validate_result_files(
    *,
    cell: Any,
    fresh_output_audit: Mapping[str, Any],
) -> dict[str, Any]:
    context = cell.cell_id
    output = Path(cell.output_dir)
    documents: dict[str, dict[str, Any]] = {}
    raw_files: dict[str, bytes] = {}
    for name in REQUIRED_OUTPUT_FILES:
        document, raw = _read_json(output / name, context=f"{context}.{name}")
        documents[name] = document
        raw_files[name] = raw

    result = documents["results.json"]
    manifest = documents["protocol_manifest.json"]
    selection = documents["episode_selection.json"]
    action = documents["action_normalization.json"]
    try:
        ema_results._engine._validate_selection(
            selection,
            raw=raw_files["episode_selection.json"],
            context=f"{context}.selection",
        )
        ema_results._engine._validate_action_normalization(
            action, context=f"{context}.action"
        )
    except Exception as error:
        raise NewScoreReconciliationError(str(error)) from error
    if manifest.get("selection") != selection:
        raise NewScoreReconciliationError(
            f"{context}: protocol manifest selection differs from selection file"
        )
    normalization = _mapping(
        manifest.get("normalization"), context=f"{context}.normalization"
    )
    if normalization.get("action") != action:
        raise NewScoreReconciliationError(
            f"{context}: protocol manifest action normalization differs"
        )

    metrics = _mapping(result.get("metrics"), context=f"{context}.metrics")
    outcomes = metrics.get("episode_successes")
    legacy = metrics.get("success")
    if outcomes is not None and legacy is not None and outcomes != legacy:
        raise NewScoreReconciliationError(
            f"{context}: canonical and legacy episode outcomes disagree"
        )
    if outcomes is None:
        outcomes = legacy
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EPISODES
        or any(not isinstance(value, bool) for value in outcomes)
    ):
        raise NewScoreReconciliationError(
            f"{context}: metrics.episode_successes must contain 50 booleans"
        )
    success_count = sum(outcomes)
    success_rate_percent = 100.0 * success_count / EPISODES
    recorded_rate = metrics.get("success_rate")
    if (
        isinstance(recorded_rate, bool)
        or not isinstance(recorded_rate, (int, float))
        or not math.isfinite(float(recorded_rate))
        or not math.isclose(
            float(recorded_rate), success_rate_percent, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise NewScoreReconciliationError(
            f"{context}: metrics.success_rate disagrees with episode outcomes"
        )

    checkpoint = _mapping(manifest.get("checkpoint"), context=f"{context}.checkpoint")
    if checkpoint.get("epoch") != cell.epoch:
        raise NewScoreReconciliationError(f"{context}: checkpoint epoch is wrong")
    expected_step = STEPS_PER_EPOCH * cell.epoch
    if checkpoint.get("global_step") != expected_step:
        raise NewScoreReconciliationError(f"{context}: checkpoint global_step is wrong")

    source_hashes = {
        name: _file_sha256(output / name) for name in REQUIRED_OUTPUT_FILES
    }
    if {
        name: str(_mapping(fresh_output_audit[name], context=name).get("sha256"))
        for name in REQUIRED_OUTPUT_FILES
    } != source_hashes:
        raise NewScoreReconciliationError(
            f"{context}: fresh scheduler audit hashes disagree"
        )
    return {
        "outcomes": list(outcomes),
        "success_count": success_count,
        "success_rate": success_count / EPISODES,
        "success_rate_percent": success_rate_percent,
        "checkpoint_epoch": cell.epoch,
        "checkpoint_global_step": expected_step,
        "checkpoint_path": str(Path(cell.checkpoint).resolve()),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "training_manifest_path": str(result["training_manifest_path"]),
        "training_manifest_sha256": str(result["training_manifest_sha256"]),
        "evaluation_commit": EVALUATION_COMMIT,
        "selection_sha256": source_hashes["episode_selection.json"],
        "action_normalization_sha256": source_hashes["action_normalization.json"],
        "source_files_sha256": source_hashes,
    }


def _validate_cell(
    *,
    cell_id: str,
    raw_cell: Mapping[str, Any],
    root: Path,
    state_path: Path,
    expected_epoch: int,
) -> dict[str, Any]:
    context = f"{state_path}:{cell_id}"
    if raw_cell.get("cell_id") != cell_id:
        raise NewScoreReconciliationError(f"{context}: embedded cell_id differs")
    epoch = raw_cell.get("epoch")
    variant = raw_cell.get("variant")
    score_mode = raw_cell.get("score_mode")
    if (
        epoch != expected_epoch
        or variant not in VARIANTS
        or score_mode not in SCORE_MODES
    ):
        raise NewScoreReconciliationError(f"{context}: cell identity is invalid")
    if cell_id != _expected_cell_id(epoch, variant, score_mode):
        raise NewScoreReconciliationError(f"{context}: cell_id is non-canonical")
    state_label = raw_cell.get("state")
    if state_label not in TERMINAL_STATES:
        raise NewScoreReconciliationError(f"{context}: cell is not terminal")
    if raw_cell.get("error") not in (None, ""):
        raise NewScoreReconciliationError(f"{context}: cell records an error")
    if state_label == "SUCCEEDED" and raw_cell.get("exit_code_marker") != 0:
        raise NewScoreReconciliationError(
            f"{context}: succeeded cell lacks a zero exit marker"
        )

    output = _expected_output(root, epoch, variant, score_mode).resolve()
    if not _same_path(raw_cell.get("output_dir"), output):
        raise NewScoreReconciliationError(
            f"{context}: output is not in its canonical source root"
        )
    argv_value = raw_cell.get("argv")
    if (
        not isinstance(argv_value, list)
        or not argv_value
        or any(not isinstance(item, str) for item in argv_value)
    ):
        raise NewScoreReconciliationError(f"{context}: argv is missing or malformed")
    argv = tuple(argv_value)
    if raw_cell.get("argv_sha256") != _sweeps._base.canonical_json_sha256(argv_value):
        raise NewScoreReconciliationError(f"{context}: argv_sha256 is stale")
    expected_weight = _argv_value(argv, "--g-first-weight")
    if score_mode == SCORE_MODES[0]:
        try:
            weight = float(str(expected_weight))
        except (TypeError, ValueError) as error:
            raise NewScoreReconciliationError(
                f"{context}: first-action alpha is missing"
            ) from error
        if not math.isclose(weight, FIRST_ACTION_WEIGHT, rel_tol=0.0, abs_tol=0.0):
            raise NewScoreReconciliationError(
                f"{context}: first-action alpha must be exactly {FIRST_ACTION_WEIGHT}"
            )
    elif expected_weight is not None:
        raise NewScoreReconciliationError(
            f"{context}: mean score must not contain alpha"
        )

    state_paths = _mapping(
        _read_json(state_path, context=str(state_path))[0].get("paths"),
        context=f"{context}.paths",
    )
    formal_root_text = state_paths.get("formal_root")
    if not isinstance(formal_root_text, str):
        raise NewScoreReconciliationError(f"{context}: formal_root is missing")
    paths = _sweeps.SweepPaths(
        repository=Path(str(state_paths.get("repository", "."))).resolve(),
        dataset=Path(str(state_paths.get("dataset", "."))).resolve(),
        formal_root=Path(formal_root_text).expanduser().resolve(),
        bundle_root=root,
        sweep_root=root / "new_score_evaluation_sweeps",
        launcher_root=root / "new_score_evaluation_sweep_launcher",
    )
    expected = next(
        item
        for item in _sweeps.build_cells(
            paths=paths,
            python=argv[0],
            g_first_weight=FIRST_ACTION_WEIGHT,
        )
        if item.cell_id == cell_id
    )
    for field in (
        "epoch",
        "variant",
        "score_mode",
        "checkpoint",
        "output_dir",
        "job_dir",
        "config_path",
    ):
        if raw_cell.get(field) != getattr(expected, field):
            raise NewScoreReconciliationError(f"{context}: {field} is non-canonical")
    if argv != expected.argv:
        raise NewScoreReconciliationError(f"{context}: argv is non-canonical")

    cell = _sweeps.Cell(
        cell_id=cell_id,
        epoch=epoch,
        variant=variant,
        score_mode=score_mode,
        checkpoint=str(raw_cell["checkpoint"]),
        output_dir=str(raw_cell["output_dir"]),
        job_dir=str(raw_cell["job_dir"]),
        config_path=str(raw_cell["config_path"]),
        argv=argv,
    )
    try:
        _sweeps._EXPECTED_EVALUATION_COMMIT = EVALUATION_COMMIT
        fresh_output_audit = _sweeps.audit_complete_output(cell)
    except Exception as error:
        raise NewScoreReconciliationError(
            f"{context}: scheduler output audit failed: {error}"
        ) from error
    recorded_output_audit = _mapping(
        raw_cell.get("output_audit"), context=f"{context}.output_audit"
    )
    result_evidence = _validate_result_files(
        cell=cell,
        fresh_output_audit=fresh_output_audit,
    )
    if recorded_output_audit != fresh_output_audit:
        raise NewScoreReconciliationError(f"{context}: scheduler output_audit is stale")
    return {
        "cell_id": cell_id,
        "source_scope": ("strict_epoch3" if epoch == 3 else "original_epoch4_10"),
        "source_state": str(state_path),
        "source_status": state_label,
        "epoch": epoch,
        "variant": variant,
        "method": f"{_sweeps.METHOD_FAMILY}_{variant}",
        "score_mode": score_mode,
        "g_first_weight": FIRST_ACTION_WEIGHT if score_mode == SCORE_MODES[0] else None,
        "output_dir": str(output),
        **result_evidence,
    }


def _validate_scope(
    *,
    root: Path,
    state_path: Path,
    expected_epochs: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if not root.is_dir():
        raise NewScoreReconciliationError(f"missing source root: {root}")
    state, state_raw = _read_json(state_path, context=str(state_path))
    raw_cells = _validate_state_header(
        state,
        root=root,
        state_path=state_path,
        expected_epochs=expected_epochs,
    )
    cells: list[dict[str, Any]] = []
    for epoch in expected_epochs:
        for variant in VARIANTS:
            for score_mode in SCORE_MODES:
                cell_id = _expected_cell_id(epoch, variant, score_mode)
                cells.append(
                    _validate_cell(
                        cell_id=cell_id,
                        raw_cell=_mapping(
                            raw_cells[cell_id], context=f"{state_path}:{cell_id}"
                        ),
                        root=root,
                        state_path=state_path,
                        expected_epoch=epoch,
                    )
                )
    return state, cells, hashlib.sha256(state_raw).hexdigest()


def _fixed_expected_grid() -> set[str]:
    first = {
        f"{version}__{variant}__{SCORE_MODES[0]}__alpha_0p25"
        for version in FIXED_VERSIONS
        for variant in VARIANTS
    }
    mean = {f"v2__{variant}__{SCORE_MODES[1]}" for variant in VARIANTS}
    return first | mean


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git_head(repository: Path) -> str:
    if not repository.is_dir():
        raise NewScoreReconciliationError(
            f"fixed evaluation checkout is missing: {repository}"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise NewScoreReconciliationError(
            f"cannot resolve fixed evaluation checkout HEAD: {repository}"
        ) from error
    if (
        len(revision) != 40
        or revision != revision.lower()
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise NewScoreReconciliationError(
            f"fixed evaluation checkout has malformed HEAD: {revision!r}"
        )
    return revision


def _validate_fixed_launchers(
    sources: Sequence[tuple[str | Path, str | Path]],
    *,
    evaluation_checkout: str | Path,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    if len(sources) != 3:
        raise NewScoreReconciliationError(
            "fixed-checkpoint reconciliation requires exactly three launcher root/manifest pairs"
        )
    checkout = Path(evaluation_checkout).expanduser().resolve()
    checkout_head = _git_head(checkout)
    if checkout_head != EVALUATION_COMMIT:
        raise NewScoreReconciliationError(
            "fixed evaluation checkout HEAD "
            f"{checkout_head!r} is not the locked commit {EVALUATION_COMMIT!r}"
        )
    source_evidence: list[Mapping[str, Any]] = []
    raw_jobs: dict[str, tuple[Mapping[str, Any], Path, Path]] = {}
    for index, (raw_root, raw_manifest) in enumerate(sources, 1):
        root = Path(raw_root).expanduser().resolve()
        manifest_path = Path(raw_manifest).expanduser().resolve()
        if not root.is_dir():
            raise NewScoreReconciliationError(f"fixed launcher root is missing: {root}")
        if not _path_is_within(manifest_path, root):
            raise NewScoreReconciliationError(
                f"fixed launcher manifest is outside its supplied root: {manifest_path}"
            )
        manifest, raw = _read_json(manifest_path, context=f"fixed_launcher_{index}")
        expected_header = {
            "schema_version": 1,
            "launcher": "actor_free_td_lewm_first_action_comparison",
            "inference_only": True,
            "training_performed": False,
            "alpha_selection_performed": False,
            "stage": "formal",
        }
        for key, value in expected_header.items():
            if manifest.get(key) != value:
                raise NewScoreReconciliationError(
                    f"{manifest_path}: {key}={manifest.get(key)!r}, expected {value!r}"
                )
        if manifest.get("status") != "SUCCEEDED":
            raise NewScoreReconciliationError(
                f"{manifest_path}: launcher status is not SUCCEEDED"
            )
        if not _same_path(manifest.get("repository"), checkout):
            raise NewScoreReconciliationError(
                f"{manifest_path}: launcher repository does not bind the supplied checkout"
            )
        jobs = _mapping(manifest.get("jobs"), context=f"{manifest_path}.jobs")
        for job_id, value in jobs.items():
            if job_id in raw_jobs:
                raise NewScoreReconciliationError(
                    f"fixed launcher job appears more than once: {job_id}"
                )
            raw_jobs[str(job_id)] = (
                _mapping(value, context=f"{manifest_path}.jobs.{job_id}"),
                root,
                manifest_path,
            )
        source_evidence.append(
            {
                "root": str(root),
                "manifest": str(manifest_path),
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "job_count": len(jobs),
                "evaluation_checkout": str(checkout),
                "evaluation_checkout_head": checkout_head,
            }
        )
    expected_grid = _fixed_expected_grid()
    if set(raw_jobs) != expected_grid:
        raise NewScoreReconciliationError(
            "fixed launcher union is not the exact 18 first-action + 6 V2 mean grid; "
            f"missing={sorted(expected_grid - set(raw_jobs))}, "
            f"extra={sorted(set(raw_jobs) - expected_grid)}"
        )

    cells: list[Mapping[str, Any]] = []
    selections: set[str] = set()
    outputs: set[str] = set()
    for version in FIXED_VERSIONS:
        for variant in VARIANTS:
            modes = (SCORE_MODES[0],) + ((SCORE_MODES[1],) if version == "v2" else ())
            for score_mode in modes:
                job_id = (
                    f"{version}__{variant}__{score_mode}__alpha_0p25"
                    if score_mode == SCORE_MODES[0]
                    else f"{version}__{variant}__{score_mode}"
                )
                raw_job, root, manifest_path = raw_jobs[job_id]
                if raw_job.get("job_id") != job_id:
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: embedded job_id differs"
                    )
                if raw_job.get("state") != "SUCCEEDED" or raw_job.get("exit_code") != 0:
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: job is not a zero-exit success"
                    )
                argv = raw_job.get("argv")
                if (
                    not isinstance(argv, list)
                    or len(argv) < 2
                    or any(not isinstance(item, str) for item in argv)
                ):
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: argv is malformed"
                    )
                evaluator = Path(argv[1]).expanduser().resolve()
                expected_evaluator = (
                    checkout
                    / "scripts"
                    / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
                ).resolve()
                if evaluator != expected_evaluator or not evaluator.is_file():
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: argv evaluator is not the "
                        "expected script in the supplied checkout"
                    )
                alpha = raw_job.get("alpha")
                if score_mode == SCORE_MODES[0]:
                    if (
                        isinstance(alpha, bool)
                        or not isinstance(alpha, (int, float))
                        or float(alpha) != FIRST_ACTION_WEIGHT
                    ):
                        raise NewScoreReconciliationError(
                            f"{manifest_path}:{job_id}: alpha must be 0.25"
                        )
                elif alpha is not None:
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: mean score must not carry alpha"
                    )
                output = Path(str(raw_job.get("output_dir", ""))).expanduser().resolve()
                if not _path_is_within(output, root):
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: output is outside its supplied root"
                    )
                if str(output) in outputs:
                    raise NewScoreReconciliationError(
                        f"fixed launcher output is duplicated: {output}"
                    )
                outputs.add(str(output))
                job = _fixed.Job(
                    job_id=job_id,
                    stage=str(raw_job.get("stage")),
                    version=str(raw_job.get("version")),
                    variant=str(raw_job.get("variant")),
                    score_mode=str(raw_job.get("score_mode")),
                    alpha=float(alpha) if alpha is not None else None,
                    checkpoint=str(raw_job.get("checkpoint")),
                    config_path=str(raw_job.get("config_path")),
                    output_dir=str(output),
                    log_path=str(raw_job.get("log_path")),
                    argv=tuple(argv),
                )
                if (
                    job.stage != "formal"
                    or job.version != version
                    or job.variant != variant
                    or job.score_mode != score_mode
                ):
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: job identity is inconsistent"
                    )
                try:
                    validation = _fixed.validate_job_output(job)
                except Exception as error:
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: validate_job_output failed: {error}"
                    ) from error
                recorded_evidence = raw_job.get("evidence")
                if recorded_evidence != validation:
                    raise NewScoreReconciliationError(
                        f"{manifest_path}:{job_id}: launcher evidence is stale"
                    )
                selection_sha = str(validation["selection_sha256"])
                selections.add(selection_sha)
                result_path = Path(str(validation["results_path"]))
                result, _ = _read_json(result_path, context=f"{job_id}.results")
                manifest_output, _ = _read_json(
                    Path(str(validation["manifest_path"])),
                    context=f"{job_id}.manifest",
                )
                commit_field_evidence: dict[str, str] = {}
                for document, label in (
                    (result, "results"),
                    (manifest_output, "manifest"),
                ):
                    if "evaluation_commit" not in document:
                        commit_field_evidence[label] = "absent"
                    elif document["evaluation_commit"] == EVALUATION_COMMIT:
                        commit_field_evidence[label] = "present_and_matched"
                    else:
                        raise NewScoreReconciliationError(
                            f"{job_id}: optional {label}.evaluation_commit conflicts "
                            f"with checkout HEAD {EVALUATION_COMMIT}"
                        )
                selection_path = Path(str(validation["selection_path"]))
                selection_file_sha = _file_sha256(selection_path)
                if selection_file_sha != SELECTION_SHA256:
                    raise NewScoreReconciliationError(
                        f"{job_id}: episode_selection.json is not the locked O50 file"
                    )
                action_path = output / "action_normalization.json"
                action_output, _ = _read_json(
                    action_path, context=f"{job_id}.action_normalization"
                )
                action_file_sha = _file_sha256(action_path)
                if action_file_sha != FIXED_ACTION_NORMALIZATION_SHA256:
                    raise NewScoreReconciliationError(
                        f"{job_id}: action_normalization.json is not the locked file"
                    )
                if (
                    manifest_output.get("selection")
                    != _read_json(selection_path, context=f"{job_id}.selection")[0]
                ):
                    raise NewScoreReconciliationError(
                        f"{job_id}: manifest selection differs from the locked file"
                    )
                normalization = _mapping(
                    manifest_output.get("normalization"),
                    context=f"{job_id}.normalization",
                )
                if normalization.get("action") != action_output:
                    raise NewScoreReconciliationError(
                        f"{job_id}: manifest action normalization differs"
                    )
                metrics = _mapping(result.get("metrics"), context=f"{job_id}.metrics")
                outcomes = metrics.get("episode_successes")
                if (
                    not isinstance(outcomes, list)
                    or len(outcomes) != EPISODES
                    or any(not isinstance(value, bool) for value in outcomes)
                ):
                    raise NewScoreReconciliationError(
                        f"{job_id}: metrics.episode_successes must contain 50 booleans"
                    )
                success_count = sum(outcomes)
                cells.append(
                    {
                        "job_id": job_id,
                        "version": version,
                        "variant": variant,
                        "method": f"actor_free_td_lewm_{version}_{variant}",
                        "score_mode": score_mode,
                        "g_first_weight": (
                            FIRST_ACTION_WEIGHT
                            if score_mode == SCORE_MODES[0]
                            else None
                        ),
                        "success_count": success_count,
                        "success_rate": success_count / EPISODES,
                        "success_rate_percent": 100.0 * success_count / EPISODES,
                        "outcomes": list(outcomes),
                        "checkpoint_path": str(Path(job.checkpoint).resolve()),
                        "checkpoint_sha256": _file_sha256(Path(job.checkpoint)),
                        "selection_sha256": selection_sha,
                        "episode_selection_file_sha256": selection_file_sha,
                        "action_normalization_sha256": action_file_sha,
                        "evaluation_commit": EVALUATION_COMMIT,
                        "evaluation_commit_evidence": {
                            "source": "launcher_repository_checkout_head",
                            "launcher_repository": str(checkout),
                            "checkout_head": checkout_head,
                            "evaluator_path": str(evaluator),
                            "result_field": commit_field_evidence["results"],
                            "manifest_field": commit_field_evidence["manifest"],
                        },
                        "source_launcher_manifest": str(manifest_path),
                        "output_dir": str(output),
                        "source_files_sha256": {
                            name: _file_sha256(output / name)
                            for name in REQUIRED_OUTPUT_FILES
                        },
                    }
                )
    if len(cells) != 24 or selections != {FIXED_SELECTION_SHA256}:
        raise NewScoreReconciliationError(
            "fixed checkpoint grid does not share the locked 50-pair selection"
        )
    return tuple(source_evidence), tuple(cells)


def reconcile_new_score_sweeps(
    *,
    strict_epoch3_root: str | Path,
    strict_epoch3_state: str | Path,
    original_epoch4_10_root: str | Path,
    original_epoch4_10_state: str | Path,
    fixed_launchers: Sequence[tuple[str | Path, str | Path]] | None = None,
    fixed_evaluation_checkout: str | Path | None = None,
) -> ValidatedNewScoreStudy:
    """Validate and reconcile the exact strict-epoch3 plus epoch4--10 split."""

    strict_root = Path(strict_epoch3_root).expanduser().resolve()
    strict_state_path = Path(strict_epoch3_state).expanduser().resolve()
    original_root = Path(original_epoch4_10_root).expanduser().resolve()
    original_state_path = Path(original_epoch4_10_state).expanduser().resolve()
    if strict_root == original_root:
        raise NewScoreReconciliationError("the two source roots must be disjoint")
    _, epoch3_cells, strict_state_sha = _validate_scope(
        root=strict_root,
        state_path=strict_state_path,
        expected_epochs=(3,),
    )
    _, later_cells, original_state_sha = _validate_scope(
        root=original_root,
        state_path=original_state_path,
        expected_epochs=tuple(range(4, 11)),
    )
    cells = tuple(epoch3_cells + later_cells)
    identities = {str(cell["cell_id"]) for cell in cells}
    if len(cells) != 96 or identities != _expected_grid(EPOCHS):
        raise NewScoreReconciliationError(
            "combined grid is not exactly 96 unique cells"
        )
    output_dirs = {str(cell["output_dir"]) for cell in cells}
    if len(output_dirs) != 96:
        raise NewScoreReconciliationError("combined grid contains duplicate outputs")
    normalizations = {str(cell["action_normalization_sha256"]) for cell in cells}
    if len(normalizations) != 1:
        raise NewScoreReconciliationError(
            "the 96 cells do not share one action-normalization artifact"
        )
    for epoch in EPOCHS:
        for variant in VARIANTS:
            pair = [
                cell
                for cell in cells
                if cell["epoch"] == epoch and cell["variant"] == variant
            ]
            if len(pair) != 2:
                raise NewScoreReconciliationError(
                    f"epoch {epoch} variant {variant} does not have two score modes"
                )
            if len({cell["checkpoint_sha256"] for cell in pair}) != 1:
                raise NewScoreReconciliationError(
                    f"epoch {epoch} variant {variant} score modes use different checkpoints"
                )
            if len({cell["training_manifest_sha256"] for cell in pair}) != 1:
                raise NewScoreReconciliationError(
                    f"epoch {epoch} variant {variant} score modes use different training manifests"
                )
    fixed_sources: tuple[Mapping[str, Any], ...] = ()
    fixed_cells: tuple[Mapping[str, Any], ...] = ()
    if fixed_launchers is not None:
        if fixed_evaluation_checkout is None:
            raise NewScoreReconciliationError(
                "fixed_evaluation_checkout is required with fixed_launchers"
            )
        fixed_sources, fixed_cells = _validate_fixed_launchers(
            fixed_launchers,
            evaluation_checkout=fixed_evaluation_checkout,
        )
    elif fixed_evaluation_checkout is not None:
        raise NewScoreReconciliationError(
            "fixed_launchers are required with fixed_evaluation_checkout"
        )
    return ValidatedNewScoreStudy(
        strict_epoch3_root=strict_root,
        strict_epoch3_state=strict_state_path,
        original_epoch4_10_root=original_root,
        original_epoch4_10_state=original_state_path,
        state_sha256={
            "strict_epoch3": strict_state_sha,
            "original_epoch4_10": original_state_sha,
        },
        cells=cells,
        action_normalization_sha256=next(iter(normalizations)),
        fixed_launcher_sources=fixed_sources,
        fixed_cells=fixed_cells,
    )


def build_ledger(study: ValidatedNewScoreStudy) -> dict[str, Any]:
    """Build the compact provenance ledger without copying raw artifacts."""

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "actor_free_td_lewm_v2_ema_new_score_reconciliation",
        "training_commit": TRAINING_COMMIT,
        "evaluation_commit": EVALUATION_COMMIT,
        "selection_sha256": SELECTION_SHA256,
        "action_normalization_sha256": study.action_normalization_sha256,
        "epochs": list(EPOCHS),
        "variants": list(VARIANTS),
        "score_modes": list(SCORE_MODES),
        "g_first_weight": FIRST_ACTION_WEIGHT,
        "cell_count": len(study.cells),
        "sources": {
            "strict_epoch3": {
                "root": str(study.strict_epoch3_root),
                "state": str(study.strict_epoch3_state),
                "state_sha256": study.state_sha256["strict_epoch3"],
                "epochs": [3],
                "cell_count": 12,
            },
            "original_epoch4_10": {
                "root": str(study.original_epoch4_10_root),
                "state": str(study.original_epoch4_10_state),
                "state_sha256": study.state_sha256["original_epoch4_10"],
                "epochs": list(range(4, 11)),
                "cell_count": 84,
            },
        },
        "cells": list(study.cells),
        "fixed_checkpoint_comparison": {
            "included": bool(study.fixed_cells),
            "selection_sha256": (FIXED_SELECTION_SHA256 if study.fixed_cells else None),
            "episode_selection_file_sha256": (
                SELECTION_SHA256 if study.fixed_cells else None
            ),
            "action_normalization_sha256": (
                FIXED_ACTION_NORMALIZATION_SHA256 if study.fixed_cells else None
            ),
            "evaluation_commit": EVALUATION_COMMIT if study.fixed_cells else None,
            "evaluation_commit_evidence_source": (
                "launcher_repository_checkout_head" if study.fixed_cells else None
            ),
            "cell_count": len(study.fixed_cells),
            "launcher_sources": list(study.fixed_launcher_sources),
            "cells": list(study.fixed_cells),
        },
    }


def build_results_csv(study: ValidatedNewScoreStudy) -> bytes:
    fields = (
        "epoch",
        "variant",
        "method",
        "score_mode",
        "g_first_weight",
        "success_count",
        "success_rate",
        "success_rate_percent",
        "checkpoint_epoch",
        "checkpoint_global_step",
        "checkpoint_sha256",
        "training_manifest_sha256",
        "evaluation_commit",
        "source_scope",
        "source_status",
        "cell_id",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cell in study.cells:
        row = {field: cell.get(field) for field in fields}
        row["g_first_weight"] = (
            "" if cell["g_first_weight"] is None else cell["g_first_weight"]
        )
        writer.writerow(row)
    return stream.getvalue().encode()


def build_fixed_results_csv(study: ValidatedNewScoreStudy) -> bytes:
    fields = (
        "version",
        "variant",
        "method",
        "score_mode",
        "g_first_weight",
        "success_count",
        "success_rate",
        "success_rate_percent",
        "checkpoint_sha256",
        "selection_sha256",
        "episode_selection_file_sha256",
        "action_normalization_sha256",
        "evaluation_commit",
        "job_id",
        "source_launcher_manifest",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cell in study.fixed_cells:
        row = {field: cell.get(field) for field in fields}
        row["g_first_weight"] = (
            "" if cell["g_first_weight"] is None else cell["g_first_weight"]
        )
        writer.writerow(row)
    return stream.getvalue().encode()


def build_summary(study: ValidatedNewScoreStudy) -> dict[str, Any]:
    results_by_epoch: dict[str, dict[str, dict[str, Any]]] = {}
    best_by_epoch: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for epoch in EPOCHS:
        epoch_cells = [cell for cell in study.cells if cell["epoch"] == epoch]
        results_by_epoch[str(epoch)] = {
            variant: {
                cell["score_mode"]: {
                    "success_count": cell["success_count"],
                    "success_rate": cell["success_rate"],
                    "success_rate_percent": cell["success_rate_percent"],
                }
                for cell in epoch_cells
                if cell["variant"] == variant
            }
            for variant in VARIANTS
        }
        best_by_epoch[str(epoch)] = {}
        for score_mode in SCORE_MODES:
            candidates = [
                cell for cell in epoch_cells if cell["score_mode"] == score_mode
            ]
            best_count = max(int(cell["success_count"]) for cell in candidates)
            best_by_epoch[str(epoch)][score_mode] = [
                {
                    "variant": cell["variant"],
                    "success_count": cell["success_count"],
                    "success_rate_percent": cell["success_rate_percent"],
                }
                for cell in candidates
                if cell["success_count"] == best_count
            ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "actor_free_td_lewm_v2_ema_new_score_summary",
        "cell_count": len(study.cells),
        "selection_sha256": SELECTION_SHA256,
        "training_commit": TRAINING_COMMIT,
        "evaluation_commit": EVALUATION_COMMIT,
        "fixed_checkpoint_cell_count": len(study.fixed_cells),
        "fixed_checkpoint_selection_sha256": (
            FIXED_SELECTION_SHA256 if study.fixed_cells else None
        ),
        "fixed_checkpoint_results": [
            {
                key: cell[key]
                for key in (
                    "version",
                    "variant",
                    "score_mode",
                    "success_count",
                    "success_rate",
                    "success_rate_percent",
                )
            }
            for cell in study.fixed_cells
        ],
        "results_by_epoch": results_by_epoch,
        "best_by_epoch_and_score_mode": best_by_epoch,
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


def write_archive(
    study: ValidatedNewScoreStudy,
    *,
    artifact_dir: str | Path,
    check: bool = False,
) -> tuple[Path, ...]:
    """Write or byte-check the deterministic, lightweight three-file archive."""

    root = Path(artifact_dir).expanduser().resolve()
    payloads = {
        "reconciliation_ledger.json": _json_bytes(build_ledger(study)),
        "results.csv": build_results_csv(study),
        "fixed_checkpoint_results.csv": build_fixed_results_csv(study),
        "summary.json": _json_bytes(build_summary(study)),
    }
    paths = tuple(root / name for name in ARCHIVE_FILENAMES)
    if check:
        existing_names = (
            {path.name for path in root.iterdir() if path.is_file()}
            if root.is_dir()
            else set()
        )
        if existing_names != set(ARCHIVE_FILENAMES):
            raise NewScoreReconciliationError(
                "archive directory does not contain exactly the three generated files"
            )
        for name, payload in payloads.items():
            path = root / name
            if path.read_bytes() != payload:
                raise NewScoreReconciliationError(f"generated archive differs: {path}")
        return paths
    root.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in root.iterdir() if path.is_file()} - set(
        ARCHIVE_FILENAMES
    )
    if unexpected:
        raise NewScoreReconciliationError(
            f"refusing to mix archive with unexpected files: {sorted(unexpected)}"
        )
    for name, payload in payloads.items():
        _atomic_write(root / name, payload)
    return paths


__all__ = [
    "ARCHIVE_FILENAMES",
    "EPOCHS",
    "EVALUATION_COMMIT",
    "FIXED_ACTION_NORMALIZATION_SHA256",
    "FIXED_SELECTION_SHA256",
    "FIXED_VERSIONS",
    "FIRST_ACTION_WEIGHT",
    "NewScoreReconciliationError",
    "SCORE_MODES",
    "SELECTION_SHA256",
    "TRAINING_COMMIT",
    "VARIANTS",
    "ValidatedNewScoreStudy",
    "build_ledger",
    "build_fixed_results_csv",
    "build_results_csv",
    "build_summary",
    "reconcile_new_score_sweeps",
    "write_archive",
]
