"""Fail-closed archive for the eight V1-C/C2/C3 endpoint O50 cells.

The endpoint study is intentionally an extension rather than a rewrite of the
historical result ledger.  It accepts the seven-job V1-C2 launcher (which also
contains the V1-C First-Q2 reference) and one standalone V1-C3 State-V output.
Every accepted scalar is recomputed from the copied 50-episode boolean
outcomes, and every copied source file is bound by SHA-256 in the ledger.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SOURCE = "actor_free_td_lewm_v1_c2_c3_endpoint_extension"
EPISODES = 50
TRAINING_SEED = 3072
PLANNING_SEED = 42
FIRST_Q_WEIGHT = 0.25
PARENT_V1_C_CHECKPOINT_SHA256 = (
    "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3"
)
SELECTION_SHA256 = "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
SELECTION_RANKS_SHA256 = (
    "88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9"
)
ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)

REQUIRED_OUTPUT_FILES = (
    "results.json",
    "protocol_manifest.json",
    "episode_selection.json",
    "action_normalization.json",
)
ARCHIVE_FILENAMES = (
    "all_o50_results.csv",
    "reconciliation_ledger.json",
    "summary.json",
)

# These are the two columns added to the historical master table.  C2's five
# existing score modes retain the historical column names below.
ENDPOINT_SCORE_MODES = ("first_q2", "state_v")
ALL_ENDPOINT_CELL_SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "g_only_f_rollout_mean",
    "first_q2",
    "state_v",
)
EXPECTED_ENDPOINT_IDENTITIES = (
    ("v1_c", "first_q2"),
    ("v1_c2", "f_only"),
    ("v1_c2", "g_only"),
    ("v1_c2", "f_plus_g"),
    ("v1_c2", "f_plus_g_first"),
    ("v1_c2", "first_q2"),
    ("v1_c2", "g_only_f_rollout_mean"),
    ("v1_c3", "state_v"),
)

_RAW_C2_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "f_plus_g_first_q2",
    "g_only_f_rollout_mean",
)
_FIRST_ACTION_RAW_MODES = frozenset({"f_plus_g_first", "f_plus_g_first_q2"})
_SHA256_RE_LENGTH = 64
_GIT_RE_LENGTH = 40


class EndpointReconciliationError(ValueError):
    """The endpoint sources or the sealed archive failed validation."""


@dataclass(frozen=True)
class EndpointResultCell:
    """One validated, normalized endpoint result exposed to table builders."""

    cell_id: str
    method_key: str
    method: str
    variant: str
    score_mode: str
    raw_score_mode: str
    outcomes: tuple[bool, ...]
    success_count: int
    success_rate: float
    success_rate_percent: float
    checkpoint_epoch: int
    checkpoint_global_step: int
    checkpoint_path: str
    checkpoint_sha256: str
    parent_v1_c_checkpoint_sha256: str
    training_seed: int
    planning_seed: int
    selection_sha256: str
    selection_ranks_sha256: str
    action_normalization_sha256: str
    evaluation_commit: str
    source_directory: str
    source_files_sha256: Mapping[str, str]

    @property
    def identity(self) -> tuple[str, str]:
        return self.method_key, self.score_mode


@dataclass(frozen=True)
class TrainingArtifactSet:
    """One optional, hash-bound formal training-run evidence set."""

    method_key: str
    run_dir: Path
    files: Mapping[str, Path]
    files_sha256: Mapping[str, str]


@dataclass(frozen=True)
class ValidatedEndpointStudy:
    """Validated live evidence ready for deterministic archiving."""

    cells: tuple[EndpointResultCell, ...]
    c2_launcher_manifest: Path
    c2_launcher_manifest_sha256: str
    c2_checkpoint_manifest: Path
    c2_checkpoint_manifest_sha256: str
    c3_output_dir: Path
    c2_evaluation_commit: str
    c3_evaluation_commit: str
    training_artifacts: tuple[TrainingArtifactSet, ...] = ()


def _error(context: str, message: str) -> EndpointReconciliationError:
    return EndpointReconciliationError(f"{context}: {message}")


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(context, "must be an object")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(context, "must be an array")
    return value


def _expect(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise _error(context, f"found {actual!r}, expected {expected!r}")


def _lower_hex(value: Any, *, length: int, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        label = "SHA-256" if length == _SHA256_RE_LENGTH else "git revision"
        raise _error(context, f"must be a full lowercase {label}")
    return value


def _sha256(value: Any, *, context: str) -> str:
    return _lower_hex(value, length=_SHA256_RE_LENGTH, context=context)


def _git_revision(value: Any, *, context: str) -> str:
    return _lower_hex(value, length=_GIT_RE_LENGTH, context=context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise _error(context, f"missing file {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(context, f"invalid JSON in {path}: {exc}") from exc
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


def _normalized_score_mode(raw_score_mode: str) -> str:
    if raw_score_mode == "f_plus_g_first_q2":
        return "first_q2"
    if raw_score_mode == "state_v_terminal":
        return "state_v"
    return raw_score_mode


def _cell_id(method_key: str, raw_score_mode: str) -> str:
    if method_key == "v1_c3":
        return "v1__c3__state_v_terminal"
    suffix = "__alpha_0p25" if raw_score_mode in _FIRST_ACTION_RAW_MODES else ""
    variant = method_key.removeprefix("v1_")
    return f"v1__{variant}__{raw_score_mode}{suffix}"


def _expected_raw_identities() -> tuple[tuple[str, str], ...]:
    return (
        ("v1_c", "f_plus_g_first_q2"),
        *(("v1_c2", mode) for mode in _RAW_C2_MODES),
        ("v1_c3", "state_v_terminal"),
    )


def _expected_job_ids() -> set[str]:
    return {
        _cell_id(method_key, raw_mode)
        for method_key, raw_mode in _expected_raw_identities()
        if method_key != "v1_c3"
    }


def _source_hashes(output_dir: Path, *, context: str) -> dict[str, str]:
    unexpected = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in REQUIRED_OUTPUT_FILES
    )
    # Videos/directories and evaluator logs are deliberately outside the copied
    # four-file evidence boundary, but an unexplained extra top-level file is
    # ambiguous and therefore rejected.
    if unexpected:
        raise _error(context, f"unexpected top-level evidence files: {unexpected}")
    hashes: dict[str, str] = {}
    for name in REQUIRED_OUTPUT_FILES:
        path = output_dir / name
        if not path.is_file():
            raise _error(context, f"missing {name}")
        hashes[name] = _file_sha256(path)
    return hashes


def _validate_outcomes(results: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    metrics = _mapping(results.get("metrics"), context=f"{context}.metrics")
    raw = metrics.get("episode_successes")
    if (
        not isinstance(raw, list)
        or len(raw) != EPISODES
        or any(not isinstance(value, bool) for value in raw)
    ):
        raise _error(context, "metrics.episode_successes must be exactly 50 booleans")
    return tuple(raw)


def _validate_selection(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    context: str,
) -> tuple[str, str]:
    path = output_dir / "episode_selection.json"
    raw_sha = _file_sha256(path)
    _expect(raw_sha, SELECTION_SHA256, context=f"{context}.selection.sha256")
    selection, _ = _read_json(path, context=f"{context}.selection")
    ranks = selection.get("valid_row_ranks")
    if (
        not isinstance(ranks, list)
        or len(ranks) != EPISODES
        or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in ranks
        )
        or len(set(ranks)) != EPISODES
    ):
        raise _error(context, "selection.valid_row_ranks must be 50 unique integers")
    ranks_sha = _canonical_json_sha256(ranks)
    _expect(
        ranks_sha,
        SELECTION_RANKS_SHA256,
        context=f"{context}.selection.ranks_sha256",
    )
    _expect(
        manifest.get("selection"),
        selection,
        context=f"{context}.manifest.selection",
    )
    if "selection_sha256" in manifest:
        _expect(
            manifest.get("selection_sha256"),
            raw_sha,
            context=f"{context}.manifest.selection_sha256",
        )
    return raw_sha, ranks_sha


def _validate_action_normalization(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    context: str,
) -> str:
    path = output_dir / "action_normalization.json"
    digest = _file_sha256(path)
    _expect(
        digest,
        ACTION_NORMALIZATION_SHA256,
        context=f"{context}.action_normalization.sha256",
    )
    action, _ = _read_json(path, context=f"{context}.action_normalization")
    normalization = _mapping(
        manifest.get("normalization"), context=f"{context}.manifest.normalization"
    )
    _expect(
        normalization.get("action"),
        action,
        context=f"{context}.manifest.normalization.action",
    )
    return digest


def _validate_planning(
    protocol: Mapping[str, Any],
    *,
    raw_score_mode: str,
    context: str,
) -> None:
    planning = _mapping(protocol.get("planning"), context=f"{context}.planning")
    expected_horizon = 1 if raw_score_mode == "g_only" else 5
    expected = {
        "solver": "CEM",
        "horizon": expected_horizon,
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
    for key, value in expected.items():
        _expect(planning.get(key), value, context=f"{context}.planning.{key}")
    evaluation = _mapping(protocol.get("evaluation"), context=f"{context}.evaluation")
    _expect(evaluation.get("episodes"), EPISODES, context=f"{context}.episodes")
    _expect(evaluation.get("goal_offset"), 50, context=f"{context}.goal_offset")


def _validate_parent_source(
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    method_key: str,
    context: str,
) -> str:
    if method_key == "v1_c":
        return PARENT_V1_C_CHECKPOINT_SHA256
    if method_key == "v1_c2":
        predictor = _mapping(
            checkpoint.get("predictor_config"),
            context=f"{context}.checkpoint.predictor_config",
        )
        checkpoint_source = _mapping(
            predictor.get("source_v1_c"),
            context=f"{context}.checkpoint.predictor_config.source_v1_c",
        )
        protocol_source = _mapping(
            protocol.get("source_v1_c"), context=f"{context}.protocol.source_v1_c"
        )
    else:
        checkpoint_source = _mapping(
            checkpoint.get("source_v1_c_provenance"),
            context=f"{context}.checkpoint.source_v1_c_provenance",
        )
        protocol_source = _mapping(
            protocol.get("source_v1_c"), context=f"{context}.protocol.source_v1_c"
        )
    for source, label in (
        (checkpoint_source, "checkpoint parent"),
        (protocol_source, "protocol parent"),
    ):
        _expect(
            source.get("checkpoint_sha256"),
            PARENT_V1_C_CHECKPOINT_SHA256,
            context=f"{context}.{label}.checkpoint_sha256",
        )
        _expect(
            source.get("source_seed"), TRAINING_SEED, context=f"{context}.{label}.seed"
        )
        _expect(source.get("source_epoch"), 10, context=f"{context}.{label}.epoch")
        _expect(
            source.get("source_global_step"),
            127_960,
            context=f"{context}.{label}.global_step",
        )
    return PARENT_V1_C_CHECKPOINT_SHA256


def _validate_output_dir(
    output_dir: Path,
    *,
    method_key: str,
    raw_score_mode: str,
    require_checkpoint_file: bool,
    expected_checkpoint_sha256: str | None = None,
    expected_evaluation_commit: str | None = None,
) -> EndpointResultCell:
    context = _cell_id(method_key, raw_score_mode)
    if not output_dir.is_dir():
        raise _error(context, f"missing output directory {output_dir}")
    hashes = _source_hashes(output_dir, context=context)
    results, _ = _read_json(output_dir / "results.json", context=f"{context}.results")
    manifest, _ = _read_json(
        output_dir / "protocol_manifest.json", context=f"{context}.manifest"
    )

    variant = method_key.removeprefix("v1_")
    method = f"actor_free_td_lewm_v1_{variant}"
    expected_result = {
        "method": method,
        "method_family": "actor_free_td_lewm_v1",
        "variant": variant,
        "implementation_version": "v1",
        "score_mode": raw_score_mode,
        "planning_horizon": 1 if raw_score_mode == "g_only" else 5,
        "smoke": False,
        "pilot": False,
    }
    for key, value in expected_result.items():
        _expect(results.get(key), value, context=f"{context}.results.{key}")
    outcomes = _validate_outcomes(results, context=f"{context}.results")
    success_count = sum(outcomes)

    _expect(manifest.get("score_mode"), raw_score_mode, context=f"{context}.score_mode")
    protocol = _mapping(manifest.get("protocol"), context=f"{context}.protocol")
    for key, value in (
        ("method", method),
        ("method_family", "actor_free_td_lewm_v1"),
        ("variant", variant),
        ("implementation_version", "v1"),
    ):
        _expect(protocol.get(key), value, context=f"{context}.protocol.{key}")
    inference = _mapping(
        protocol.get("inference_objective"),
        context=f"{context}.protocol.inference_objective",
    )
    _expect(
        inference.get("score_mode"),
        raw_score_mode,
        context=f"{context}.protocol.inference_objective.score_mode",
    )
    _validate_planning(protocol, raw_score_mode=raw_score_mode, context=context)
    pretrained = _mapping(
        protocol.get("pretrained_world_model"),
        context=f"{context}.protocol.pretrained_world_model",
    )
    _expect(
        pretrained.get("source_seed"),
        TRAINING_SEED,
        context=f"{context}.protocol.pretrained_world_model.source_seed",
    )

    if raw_score_mode in _FIRST_ACTION_RAW_MODES:
        for values, label in (
            (results, "results"),
            (manifest, "manifest"),
            (inference, "inference"),
        ):
            weight = _number(
                values.get("g_first_weight"), context=f"{context}.{label}.alpha"
            )
            if not math.isclose(weight, FIRST_Q_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
                raise _error(context, f"{label}.g_first_weight must be 0.25")
            _mapping(
                values.get("score_definition"),
                context=f"{context}.{label}.score_definition",
            )
    else:
        for values, label in (
            (results, "results"),
            (manifest, "manifest"),
            (inference, "inference"),
        ):
            if "g_first_weight" in values:
                raise _error(
                    context, f"{label} contains an unexpected first-action weight"
                )

    if raw_score_mode == "state_v_terminal":
        definition = _mapping(
            manifest.get("score_definition"),
            context=f"{context}.manifest.score_definition",
        )
        _expect(
            definition.get("optimization"),
            "cem_minimize",
            context=f"{context}.state_v.optimization",
        )
        _expect(
            definition.get("critic"),
            "ema_target_state_value",
            context=f"{context}.state_v.critic",
        )
        _expect(
            inference.get("parent_g_used"),
            False,
            context=f"{context}.state_v.parent_g_used",
        )
        state_critic = _mapping(
            protocol.get("state_critic"), context=f"{context}.protocol.state_critic"
        )
        _expect(
            state_critic.get("architecture"),
            "rp1_mrn_quasimetric",
            context=f"{context}.state_critic.architecture",
        )

    checkpoint = _mapping(manifest.get("checkpoint"), context=f"{context}.checkpoint")
    expected_epoch = 12 if method_key == "v1_c3" else 10
    expected_step = 12_000 if method_key == "v1_c3" else 127_960
    _expect(
        checkpoint.get("epoch"), expected_epoch, context=f"{context}.checkpoint.epoch"
    )
    if method_key == "v1_c3":
        _expect(
            checkpoint.get("logical_epoch"),
            expected_epoch,
            context=f"{context}.checkpoint.logical_epoch",
        )
    _expect(
        checkpoint.get("global_step"),
        expected_step,
        context=f"{context}.checkpoint.global_step",
    )
    _expect(
        checkpoint.get("formal_completion_required"),
        True,
        context=f"{context}.checkpoint.formal_completion_required",
    )
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise _error(context, "checkpoint.path must be a nonempty path")
    checkpoint_sha = _sha256(
        checkpoint.get("sha256"), context=f"{context}.checkpoint.sha256"
    )
    if expected_checkpoint_sha256 is not None:
        _expect(
            checkpoint_sha,
            expected_checkpoint_sha256,
            context=f"{context}.checkpoint.sha256",
        )
    if require_checkpoint_file:
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_file.is_file():
            raise _error(context, f"checkpoint file is unavailable: {checkpoint_file}")
        _expect(
            _file_sha256(checkpoint_file),
            checkpoint_sha,
            context=f"{context}.checkpoint.file_sha256",
        )
    if method_key == "v1_c":
        _expect(
            checkpoint_sha,
            PARENT_V1_C_CHECKPOINT_SHA256,
            context=f"{context}.checkpoint.parent_sha256",
        )
    parent_sha = _validate_parent_source(
        checkpoint, protocol, method_key=method_key, context=context
    )

    selection_sha, ranks_sha = _validate_selection(
        output_dir, manifest, context=context
    )
    action_sha = _validate_action_normalization(output_dir, manifest, context=context)
    runtime = _mapping(manifest.get("runtime"), context=f"{context}.runtime")
    evaluation_commit = _git_revision(
        runtime.get("tdwm_git_revision"), context=f"{context}.evaluation_commit"
    )
    if expected_evaluation_commit is not None:
        _expect(
            evaluation_commit,
            expected_evaluation_commit,
            context=f"{context}.evaluation_commit",
        )

    return EndpointResultCell(
        cell_id=context,
        method_key=method_key,
        method=method,
        variant=variant,
        score_mode=_normalized_score_mode(raw_score_mode),
        raw_score_mode=raw_score_mode,
        outcomes=outcomes,
        success_count=success_count,
        success_rate=success_count / EPISODES,
        success_rate_percent=2.0 * success_count,
        checkpoint_epoch=expected_epoch,
        checkpoint_global_step=expected_step,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        parent_v1_c_checkpoint_sha256=parent_sha,
        training_seed=TRAINING_SEED,
        planning_seed=PLANNING_SEED,
        selection_sha256=selection_sha,
        selection_ranks_sha256=ranks_sha,
        action_normalization_sha256=action_sha,
        evaluation_commit=evaluation_commit,
        source_directory=str(output_dir.resolve()),
        source_files_sha256=hashes,
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git_checkout_head(repository: Path, *, context: str) -> str:
    try:
        top = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _error(
            context, "launcher repository is not a readable git checkout"
        ) from exc
    _expect(Path(top).resolve(), repository.resolve(), context=f"{context}.top_level")
    if status:
        raise _error(context, "launcher repository must be completely clean")
    return _git_revision(head, context=f"{context}.head")


def _resolve_checkpoint_manifest(
    launcher: Mapping[str, Any], *, stage_root: Path, launcher_root: Path
) -> Path:
    recorded = launcher.get("checkpoint_manifest")
    if not isinstance(recorded, str) or not recorded:
        raise _error("c2_launcher", "checkpoint_manifest path is missing")
    recorded_path = Path(recorded).expanduser()
    candidates = [
        recorded_path,
        stage_root.parent / recorded_path.name,
        launcher_root / recorded_path.name,
    ]
    matches = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in matches:
            matches.append(resolved)
    if len(matches) != 1:
        raise _error(
            "c2_launcher",
            f"could not resolve one checkpoint manifest from {recorded_path.name}",
        )
    return matches[0]


def _validate_checkpoint_manifest(
    path: Path,
) -> tuple[dict[str, tuple[str, str]], str]:
    document, raw = _read_json(path, context="c2_checkpoint_manifest")
    _expect(
        set(document),
        {"schema_version", "purpose", "v1"},
        context="c2_checkpoint_manifest.keys",
    )
    _expect(document.get("schema_version"), 1, context="c2_checkpoint_manifest.schema")
    _expect(
        document.get("purpose"),
        "v1_c2_endpoint_o50_and_v1_c_first_q2_reference",
        context="c2_checkpoint_manifest.purpose",
    )
    v1 = _mapping(document.get("v1"), context="c2_checkpoint_manifest.v1")
    _expect(set(v1), {"c", "c2"}, context="c2_checkpoint_manifest.v1.keys")
    result: dict[str, tuple[str, str]] = {}
    for variant in ("c", "c2"):
        item = _mapping(v1.get(variant), context=f"c2_checkpoint_manifest.v1.{variant}")
        _expect(
            set(item),
            {"path", "sha256"},
            context=f"c2_checkpoint_manifest.{variant}.keys",
        )
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise _error("c2_checkpoint_manifest", f"{variant}.path is invalid")
        digest = _sha256(
            item.get("sha256"), context=f"c2_checkpoint_manifest.{variant}.sha256"
        )
        checkpoint = Path(raw_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise _error("c2_checkpoint_manifest", f"missing checkpoint {checkpoint}")
        _expect(
            _file_sha256(checkpoint),
            digest,
            context=f"c2_checkpoint_manifest.{variant}.file",
        )
        result[variant] = (str(checkpoint), digest)
    _expect(
        result["c"][1],
        PARENT_V1_C_CHECKPOINT_SHA256,
        context="c2_checkpoint_manifest.c.sha256",
    )
    if result["c"][1] == result["c2"][1]:
        raise _error(
            "c2_checkpoint_manifest", "C2 checkpoint must differ from its C parent"
        )
    return result, hashlib.sha256(raw).hexdigest()


def _relative_c2_output(method_key: str, raw_score_mode: str) -> Path:
    variant = method_key.removeprefix("v1_")
    relative = Path("v1") / variant / raw_score_mode
    if raw_score_mode in _FIRST_ACTION_RAW_MODES:
        relative /= "alpha_0p25"
    return relative


def _option_value(argv: Sequence[Any], option: str, *, context: str) -> str:
    values = [str(value) for value in argv]
    if values.count(option) != 1:
        raise _error(context, f"argv must contain {option} exactly once")
    index = values.index(option)
    if index + 1 >= len(values):
        raise _error(context, f"argv has no value after {option}")
    return values[index + 1]


def _validate_c2_launcher(
    *, launcher_root: Path, launcher_manifest: Path
) -> tuple[tuple[EndpointResultCell, ...], Path, str, str, str]:
    context = "c2_launcher"
    if not launcher_root.is_dir():
        raise _error(context, f"missing launcher root {launcher_root}")
    if not _path_is_within(launcher_manifest, launcher_root):
        raise _error(context, "launcher manifest must be inside the supplied root")
    launcher, launcher_raw = _read_json(launcher_manifest, context=context)
    header = {
        "schema_version": 1,
        "launcher": "actor_free_td_lewm_first_action_comparison",
        "inference_only": True,
        "training_performed": False,
        "alpha_selection_performed": False,
        "stage": "formal",
        "versions": ["v1"],
        "variants": ["c2", "c"],
        "shared_score_modes": list(_RAW_C2_MODES),
        "v2_only_score_modes": [],
        "alphas": [FIRST_Q_WEIGHT],
        "expected_selection_file_sha256": SELECTION_SHA256,
        "status": "SUCCEEDED",
    }
    for key, value in header.items():
        _expect(launcher.get(key), value, context=f"{context}.{key}")
    jobs = _mapping(launcher.get("jobs"), context=f"{context}.jobs")
    _expect(set(jobs), _expected_job_ids(), context=f"{context}.jobs.grid")
    stage_root = launcher_manifest.parent.parent.resolve()
    checkpoint_manifest = _resolve_checkpoint_manifest(
        launcher, stage_root=stage_root, launcher_root=launcher_root
    )
    checkpoints, checkpoint_manifest_sha = _validate_checkpoint_manifest(
        checkpoint_manifest
    )
    repository_value = launcher.get("repository")
    if not isinstance(repository_value, str):
        raise _error(context, "repository path is missing")
    repository = Path(repository_value).expanduser().resolve()
    evaluation_commit = _git_checkout_head(repository, context=f"{context}.repository")

    launcher_selection = _mapping(
        launcher.get("selection"), context=f"{context}.selection"
    )
    _expect(
        launcher_selection.get("selection_file_sha256"),
        SELECTION_SHA256,
        context=f"{context}.selection.file_sha256",
    )
    _expect(
        launcher_selection.get("valid_row_ranks_sha256"),
        SELECTION_RANKS_SHA256,
        context=f"{context}.selection.ranks_sha256",
    )
    _expect(
        launcher_selection.get("identical_across_all_jobs"),
        True,
        context=f"{context}.selection.identical",
    )
    _expect(
        launcher_selection.get("selection_file_identical_across_all_jobs"),
        True,
        context=f"{context}.selection.file_identical",
    )

    cells: list[EndpointResultCell] = []
    for method_key, raw_mode in _expected_raw_identities()[:-1]:
        job_id = _cell_id(method_key, raw_mode)
        job = _mapping(jobs.get(job_id), context=f"{context}.jobs.{job_id}")
        variant = method_key.removeprefix("v1_")
        expected_alpha = FIRST_Q_WEIGHT if raw_mode in _FIRST_ACTION_RAW_MODES else None
        for key, value in (
            ("job_id", job_id),
            ("stage", "formal"),
            ("version", "v1"),
            ("variant", variant),
            ("score_mode", raw_mode),
            ("alpha", expected_alpha),
            ("state", "SUCCEEDED"),
            ("exit_code", 0),
        ):
            _expect(job.get(key), value, context=f"{context}.jobs.{job_id}.{key}")
        expected_checkpoint = checkpoints[variant]
        _expect(
            str(Path(str(job.get("checkpoint"))).expanduser().resolve()),
            expected_checkpoint[0],
            context=f"{context}.jobs.{job_id}.checkpoint",
        )
        argv = _sequence(job.get("argv"), context=f"{context}.jobs.{job_id}.argv")
        _expect(
            _option_value(argv, "--score-mode", context=job_id),
            raw_mode,
            context=f"{context}.jobs.{job_id}.argv.score_mode",
        )
        if "--smoke" in argv or "--pilot" in argv:
            raise _error(context, f"{job_id} is not a formal invocation")
        if expected_alpha is not None:
            alpha = _number(
                float(_option_value(argv, "--g-first-weight", context=job_id)),
                context=f"{job_id}.argv.alpha",
            )
            if not math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12):
                raise _error(context, f"{job_id} alpha differs from 0.25")
        relative = _relative_c2_output(method_key, raw_mode)
        output = stage_root / relative
        recorded_output = job.get("output_dir")
        recorded_root = launcher.get("output_root")
        if not isinstance(recorded_output, str) or not isinstance(recorded_root, str):
            raise _error(context, f"{job_id} output provenance is missing")
        _expect(
            Path(recorded_output).expanduser(),
            Path(recorded_root).expanduser() / "formal" / relative,
            context=f"{context}.jobs.{job_id}.output_dir",
        )
        cell = _validate_output_dir(
            output,
            method_key=method_key,
            raw_score_mode=raw_mode,
            require_checkpoint_file=True,
            expected_checkpoint_sha256=expected_checkpoint[1],
            expected_evaluation_commit=evaluation_commit,
        )
        evidence = _mapping(
            job.get("evidence"), context=f"{context}.jobs.{job_id}.evidence"
        )
        _expect(
            evidence.get("selection_file_sha256"),
            cell.selection_sha256,
            context=f"{context}.jobs.{job_id}.evidence.selection",
        )
        _expect(
            evidence.get("valid_row_ranks_sha256"),
            cell.selection_ranks_sha256,
            context=f"{context}.jobs.{job_id}.evidence.ranks",
        )
        cells.append(cell)
    if (
        len({cell.checkpoint_sha256 for cell in cells if cell.method_key == "v1_c2"})
        != 1
    ):
        raise _error(context, "the six C2 cells do not bind one checkpoint")
    return (
        tuple(cells),
        checkpoint_manifest,
        checkpoint_manifest_sha,
        hashlib.sha256(launcher_raw).hexdigest(),
        evaluation_commit,
    )


def _find_one_metrics_csv(run_dir: Path, *, context: str) -> Path:
    candidates = sorted(
        {
            *run_dir.glob("metrics.csv"),
            *run_dir.glob("metrics/version_*/metrics.csv"),
        }
    )
    if len(candidates) != 1:
        raise _error(
            context,
            f"expected exactly one metrics.csv, found {[str(path) for path in candidates]}",
        )
    if candidates[0].stat().st_size <= 0:
        raise _error(context, "metrics.csv is empty")
    try:
        with candidates[0].open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(context, "metrics.csv is unreadable") from exc
    if not reader.fieldnames or not rows:
        raise _error(context, "metrics.csv has no header or data rows")
    loss_columns = [name for name in reader.fieldnames if "loss" in name.lower()]
    if not loss_columns:
        raise _error(context, "metrics.csv contains no loss column")
    finite_loss_seen = False
    for row in rows:
        for name in loss_columns:
            value = row.get(name)
            if value in (None, ""):
                continue
            try:
                finite_loss_seen |= math.isfinite(float(value))
            except ValueError as exc:
                raise _error(context, f"metrics.csv {name} is not numeric") from exc
    if not finite_loss_seen:
        raise _error(context, "metrics.csv contains no finite loss value")
    return candidates[0]


def _validate_training_run(
    run_dir: Path,
    *,
    method_key: str,
    expected_checkpoint_sha256: str,
) -> TrainingArtifactSet:
    context = f"{method_key}_training_run"
    if not run_dir.is_dir():
        raise _error(context, f"missing run directory {run_dir}")
    result_path = run_dir / "training_result.json"
    manifest_path = run_dir / "training_manifest.json"
    training_result, _ = _read_json(result_path, context=f"{context}.result")
    training_manifest, _ = _read_json(manifest_path, context=f"{context}.manifest")
    variant = method_key.removeprefix("v1_")
    expected_method = f"actor_free_td_lewm_v1_{variant}"
    for document, label in (
        (training_result, "result"),
        (training_manifest, "manifest"),
    ):
        _expect(
            document.get("method"),
            expected_method,
            context=f"{context}.{label}.method",
        )
        _expect(document.get("variant"), variant, context=f"{context}.{label}.variant")
        _expect(
            document.get("seed"),
            TRAINING_SEED,
            context=f"{context}.{label}.seed",
        )
    expected_step = 12_000 if method_key == "v1_c3" else 127_960
    _expect(
        training_result.get("global_step"),
        expected_step,
        context=f"{context}.result.global_step",
    )
    if method_key == "v1_c3":
        _expect(
            training_result.get("logical_epoch"),
            12,
            context=f"{context}.result.logical_epoch",
        )
        recorded_checkpoint_sha = training_result.get("deployment_checkpoint_sha256")
    else:
        recorded_checkpoint_sha = training_result.get("deployment_checkpoint_sha256")
        if recorded_checkpoint_sha is None:
            # The common V1 trainer records the final deployment path but not a
            # redundant digest in training_result.json.
            deployment = training_result.get("deployment_checkpoint")
            if not isinstance(deployment, str):
                raise _error(context, "result.deployment_checkpoint is missing")
            deployment_path = Path(deployment).expanduser().resolve()
            if not deployment_path.is_file():
                raise _error(
                    context, f"missing deployment checkpoint {deployment_path}"
                )
            recorded_checkpoint_sha = _file_sha256(deployment_path)
    _expect(
        recorded_checkpoint_sha,
        expected_checkpoint_sha256,
        context=f"{context}.result.deployment_checkpoint_sha256",
    )
    metrics = _find_one_metrics_csv(run_dir, context=context)
    files: dict[str, Path] = {
        "training_result.json": result_path,
        "training_manifest.json": manifest_path,
        "metrics.csv": metrics,
    }
    if method_key == "v1_c3":
        offline_path = run_dir / "validation_offline_metrics.json"
        offline, _ = _read_json(offline_path, context=f"{context}.offline_validation")
        epochs = offline.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != 12:
            raise _error(context, "offline validation must contain all 12 epochs")
        _expect(
            [
                entry.get("logical_epoch")
                for entry in epochs
                if isinstance(entry, Mapping)
            ],
            list(range(1, 13)),
            context=f"{context}.offline_validation.logical_epochs",
        )
        files["validation_offline_metrics.json"] = offline_path
    hashes = {name: _file_sha256(path) for name, path in files.items()}
    return TrainingArtifactSet(
        method_key=method_key,
        run_dir=run_dir,
        files=files,
        files_sha256=hashes,
    )


def reconcile_endpoint_results(
    *,
    c2_launcher_root: str | Path,
    c2_launcher_manifest: str | Path,
    c3_output_dir: str | Path,
    c2_training_run: str | Path | None = None,
    c3_training_run: str | Path | None = None,
) -> ValidatedEndpointStudy:
    """Validate the exact seven C/C2 jobs plus one C3 endpoint cell."""

    root = Path(c2_launcher_root).expanduser().resolve()
    manifest = Path(c2_launcher_manifest).expanduser().resolve()
    c3_dir = Path(c3_output_dir).expanduser().resolve()
    c2_cells, checkpoint_manifest, checkpoint_manifest_sha, launcher_sha, c2_commit = (
        _validate_c2_launcher(launcher_root=root, launcher_manifest=manifest)
    )
    c3_cell = _validate_output_dir(
        c3_dir,
        method_key="v1_c3",
        raw_score_mode="state_v_terminal",
        require_checkpoint_file=True,
    )
    cells = c2_cells + (c3_cell,)
    identities = tuple(cell.identity for cell in cells)
    _expect(
        identities,
        EXPECTED_ENDPOINT_IDENTITIES,
        context="endpoint_study.identities",
    )
    if len({cell.selection_sha256 for cell in cells}) != 1:
        raise _error("endpoint_study", "selection files differ across cells")
    if len({cell.action_normalization_sha256 for cell in cells}) != 1:
        raise _error("endpoint_study", "action normalization differs across cells")
    training_artifacts: list[TrainingArtifactSet] = []
    c2_checkpoint_sha = next(
        cell.checkpoint_sha256 for cell in cells if cell.method_key == "v1_c2"
    )
    c3_checkpoint_sha = c3_cell.checkpoint_sha256
    if c2_training_run is not None:
        training_artifacts.append(
            _validate_training_run(
                Path(c2_training_run).expanduser().resolve(),
                method_key="v1_c2",
                expected_checkpoint_sha256=c2_checkpoint_sha,
            )
        )
    if c3_training_run is not None:
        training_artifacts.append(
            _validate_training_run(
                Path(c3_training_run).expanduser().resolve(),
                method_key="v1_c3",
                expected_checkpoint_sha256=c3_checkpoint_sha,
            )
        )
    return ValidatedEndpointStudy(
        cells=cells,
        c2_launcher_manifest=manifest,
        c2_launcher_manifest_sha256=launcher_sha,
        c2_checkpoint_manifest=checkpoint_manifest,
        c2_checkpoint_manifest_sha256=checkpoint_manifest_sha,
        c3_output_dir=c3_dir,
        c2_evaluation_commit=c2_commit,
        c3_evaluation_commit=c3_cell.evaluation_commit,
        training_artifacts=tuple(training_artifacts),
    )


def _cell_document(cell: EndpointResultCell) -> dict[str, Any]:
    source_directory = f"sources/{cell.cell_id}"
    return {
        "cell_id": cell.cell_id,
        "method_key": cell.method_key,
        "method": cell.method,
        "variant": cell.variant,
        "score_mode": cell.score_mode,
        "raw_score_mode": cell.raw_score_mode,
        "outcomes": list(cell.outcomes),
        "success_count": cell.success_count,
        "success_rate": cell.success_rate,
        "success_rate_percent": cell.success_rate_percent,
        "checkpoint_epoch": cell.checkpoint_epoch,
        "checkpoint_global_step": cell.checkpoint_global_step,
        "checkpoint_path": cell.checkpoint_path,
        "checkpoint_sha256": cell.checkpoint_sha256,
        "parent_v1_c_checkpoint_sha256": cell.parent_v1_c_checkpoint_sha256,
        "training_seed": cell.training_seed,
        "planning_seed": cell.planning_seed,
        "selection_sha256": cell.selection_sha256,
        "selection_ranks_sha256": cell.selection_ranks_sha256,
        "action_normalization_sha256": cell.action_normalization_sha256,
        "evaluation_commit": cell.evaluation_commit,
        "source_directory": source_directory,
        "source_files_sha256": dict(cell.source_files_sha256),
    }


def build_ledger(study: ValidatedEndpointStudy) -> dict[str, Any]:
    """Build the deterministic eight-cell, 400-outcome reconciliation ledger."""

    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "cell_count": len(study.cells),
        "outcome_count": len(study.cells) * EPISODES,
        "episodes_per_cell": EPISODES,
        "training_seed": TRAINING_SEED,
        "planning_seed": PLANNING_SEED,
        "first_q_weight": FIRST_Q_WEIGHT,
        "selection_sha256": SELECTION_SHA256,
        "selection_ranks_sha256": SELECTION_RANKS_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
        "endpoint_score_modes": list(ENDPOINT_SCORE_MODES),
        "all_cell_score_modes": list(ALL_ENDPOINT_CELL_SCORE_MODES),
        "expected_identities": [
            {"method_key": method_key, "score_mode": score_mode}
            for method_key, score_mode in EXPECTED_ENDPOINT_IDENTITIES
        ],
        "evaluation_commits": sorted(
            {study.c2_evaluation_commit, study.c3_evaluation_commit}
        ),
        "sources": {
            "c2_launcher_manifest": {
                "path": str(study.c2_launcher_manifest),
                "sha256": study.c2_launcher_manifest_sha256,
            },
            "c2_checkpoint_manifest": {
                "path": str(study.c2_checkpoint_manifest),
                "sha256": study.c2_checkpoint_manifest_sha256,
            },
            "c3_output_dir": str(study.c3_output_dir),
        },
        "training_sources": {
            artifact.method_key: {
                "source_run_dir": str(artifact.run_dir),
                "archive_directory": f"training/{artifact.method_key}",
                "files_sha256": dict(artifact.files_sha256),
            }
            for artifact in study.training_artifacts
        },
        "cells": [_cell_document(cell) for cell in study.cells],
    }


_CSV_FIELDS = (
    "cell_id",
    "method_key",
    "method",
    "variant",
    "score_mode",
    "raw_score_mode",
    "success_count",
    "episodes",
    "success_rate",
    "success_rate_percent",
    "checkpoint_epoch",
    "checkpoint_global_step",
    "checkpoint_sha256",
    "evaluation_commit",
    "selection_sha256",
    "action_normalization_sha256",
)


def _csv_bytes(cells: Sequence[EndpointResultCell]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for cell in cells:
        writer.writerow(
            {
                "cell_id": cell.cell_id,
                "method_key": cell.method_key,
                "method": cell.method,
                "variant": cell.variant,
                "score_mode": cell.score_mode,
                "raw_score_mode": cell.raw_score_mode,
                "success_count": cell.success_count,
                "episodes": EPISODES,
                "success_rate": format(cell.success_rate, ".17g"),
                "success_rate_percent": format(cell.success_rate_percent, ".17g"),
                "checkpoint_epoch": cell.checkpoint_epoch,
                "checkpoint_global_step": cell.checkpoint_global_step,
                "checkpoint_sha256": cell.checkpoint_sha256,
                "evaluation_commit": cell.evaluation_commit,
                "selection_sha256": cell.selection_sha256,
                "action_normalization_sha256": cell.action_normalization_sha256,
            }
        )
    return stream.getvalue().encode()


def _summary_document(cells: Sequence[EndpointResultCell]) -> dict[str, Any]:
    matrix: dict[str, dict[str, Any]] = {}
    for cell in cells:
        matrix.setdefault(cell.method_key, {})[cell.score_mode] = {
            "success_count": cell.success_count,
            "episodes": EPISODES,
            "success_rate": cell.success_rate,
            "success_rate_percent": cell.success_rate_percent,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "cell_count": len(cells),
        "outcome_count": len(cells) * EPISODES,
        "methods": ["v1_c", "v1_c2", "v1_c3"],
        "score_modes": list(ALL_ENDPOINT_CELL_SCORE_MODES),
        "matrix": matrix,
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _generated_payloads(study: ValidatedEndpointStudy) -> Mapping[str, bytes]:
    return {
        "all_o50_results.csv": _csv_bytes(study.cells),
        "reconciliation_ledger.json": _json_bytes(build_ledger(study)),
        "summary.json": _json_bytes(_summary_document(study.cells)),
    }


def _expected_archive_file_set(
    cells: Sequence[EndpointResultCell],
    training_artifacts: Sequence[TrainingArtifactSet] = (),
) -> set[str]:
    expected = set(ARCHIVE_FILENAMES)
    for cell in cells:
        expected.update(
            f"sources/{cell.cell_id}/{name}" for name in REQUIRED_OUTPUT_FILES
        )
    for artifact in training_artifacts:
        expected.update(
            f"training/{artifact.method_key}/{name}" for name in artifact.files
        )
    return expected


def _archive_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def write_archive(
    study: ValidatedEndpointStudy,
    *,
    artifact_dir: str | Path,
    check: bool = False,
) -> tuple[Path, ...]:
    """Atomically write or byte-check the self-contained endpoint archive."""

    root = Path(artifact_dir).expanduser().resolve()
    payloads = _generated_payloads(study)
    expected_files = _expected_archive_file_set(study.cells, study.training_artifacts)
    if check:
        if not root.is_dir():
            raise _error("archive", f"missing artifact directory {root}")
        _expect(_archive_file_set(root), expected_files, context="archive.file_set")
        for cell in study.cells:
            source = Path(cell.source_directory)
            for name in REQUIRED_OUTPUT_FILES:
                archived = root / "sources" / cell.cell_id / name
                if archived.read_bytes() != (source / name).read_bytes():
                    raise _error("archive", f"copied source differs: {archived}")
        for artifact in study.training_artifacts:
            for name, source in artifact.files.items():
                archived = root / "training" / artifact.method_key / name
                if archived.read_bytes() != source.read_bytes():
                    raise _error(
                        "archive", f"copied training source differs: {archived}"
                    )
        for name, payload in payloads.items():
            if (root / name).read_bytes() != payload:
                raise _error("archive", f"generated file differs: {root / name}")
        return tuple(root / name for name in ARCHIVE_FILENAMES)

    if root.exists():
        raise _error("archive", f"refusing to overwrite existing path {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for cell in study.cells:
            destination = temporary / "sources" / cell.cell_id
            destination.mkdir(parents=True, exist_ok=False)
            source = Path(cell.source_directory)
            for name in REQUIRED_OUTPUT_FILES:
                shutil.copyfile(source / name, destination / name)
        for artifact in study.training_artifacts:
            destination = temporary / "training" / artifact.method_key
            destination.mkdir(parents=True, exist_ok=False)
            for name, source in artifact.files.items():
                shutil.copyfile(source, destination / name)
        for name, payload in payloads.items():
            path = temporary / name
            path.write_bytes(payload)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(root / name for name in ARCHIVE_FILENAMES)


def _cell_from_document(raw: Mapping[str, Any], *, context: str) -> EndpointResultCell:
    expected_keys = {
        "cell_id",
        "method_key",
        "method",
        "variant",
        "score_mode",
        "raw_score_mode",
        "outcomes",
        "success_count",
        "success_rate",
        "success_rate_percent",
        "checkpoint_epoch",
        "checkpoint_global_step",
        "checkpoint_path",
        "checkpoint_sha256",
        "parent_v1_c_checkpoint_sha256",
        "training_seed",
        "planning_seed",
        "selection_sha256",
        "selection_ranks_sha256",
        "action_normalization_sha256",
        "evaluation_commit",
        "source_directory",
        "source_files_sha256",
    }
    _expect(set(raw), expected_keys, context=f"{context}.keys")
    outcomes = raw.get("outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EPISODES
        or any(not isinstance(value, bool) for value in outcomes)
    ):
        raise _error(context, "outcomes must contain exactly 50 booleans")
    outcome_tuple = tuple(outcomes)
    count = sum(outcome_tuple)
    _expect(raw.get("success_count"), count, context=f"{context}.success_count")
    rate = _number(raw.get("success_rate"), context=f"{context}.success_rate")
    percent = _number(
        raw.get("success_rate_percent"), context=f"{context}.success_rate_percent"
    )
    if not math.isclose(rate, count / EPISODES, rel_tol=0.0, abs_tol=1e-12):
        raise _error(context, "success_rate differs from outcomes")
    if not math.isclose(percent, 2.0 * count, rel_tol=0.0, abs_tol=1e-12):
        raise _error(context, "success_rate_percent differs from outcomes")
    method_key = raw.get("method_key")
    score_mode = raw.get("score_mode")
    raw_mode = raw.get("raw_score_mode")
    if (
        not isinstance(method_key, str)
        or not isinstance(score_mode, str)
        or not isinstance(raw_mode, str)
    ):
        raise _error(context, "method and score identities must be strings")
    expected_cell_id = _cell_id(method_key, raw_mode)
    _expect(raw.get("cell_id"), expected_cell_id, context=f"{context}.cell_id")
    _expect(
        score_mode,
        _normalized_score_mode(raw_mode),
        context=f"{context}.score_mode",
    )
    hashes_raw = _mapping(
        raw.get("source_files_sha256"), context=f"{context}.source_files_sha256"
    )
    _expect(
        set(hashes_raw),
        set(REQUIRED_OUTPUT_FILES),
        context=f"{context}.source_hash_keys",
    )
    hashes = {
        name: _sha256(hashes_raw[name], context=f"{context}.source_files.{name}")
        for name in REQUIRED_OUTPUT_FILES
    }
    source_directory = raw.get("source_directory")
    checkpoint_path = raw.get("checkpoint_path")
    evaluation_commit = _git_revision(
        raw.get("evaluation_commit"), context=f"{context}.evaluation_commit"
    )
    if not isinstance(source_directory, str) or not isinstance(checkpoint_path, str):
        raise _error(context, "source/checkpoint paths must be strings")
    variant = method_key.removeprefix("v1_")
    _expect(raw.get("variant"), variant, context=f"{context}.variant")
    _expect(
        raw.get("method"),
        f"actor_free_td_lewm_v1_{variant}",
        context=f"{context}.method",
    )
    expected_epoch = 12 if method_key == "v1_c3" else 10
    expected_step = 12_000 if method_key == "v1_c3" else 127_960
    for key, expected in (
        ("checkpoint_epoch", expected_epoch),
        ("checkpoint_global_step", expected_step),
        ("training_seed", TRAINING_SEED),
        ("planning_seed", PLANNING_SEED),
        ("selection_sha256", SELECTION_SHA256),
        ("selection_ranks_sha256", SELECTION_RANKS_SHA256),
        ("action_normalization_sha256", ACTION_NORMALIZATION_SHA256),
        ("parent_v1_c_checkpoint_sha256", PARENT_V1_C_CHECKPOINT_SHA256),
    ):
        _expect(raw.get(key), expected, context=f"{context}.{key}")
    checkpoint_sha = _sha256(
        raw.get("checkpoint_sha256"), context=f"{context}.checkpoint_sha256"
    )
    if method_key == "v1_c":
        _expect(
            checkpoint_sha,
            PARENT_V1_C_CHECKPOINT_SHA256,
            context=f"{context}.checkpoint_sha256",
        )
    return EndpointResultCell(
        cell_id=expected_cell_id,
        method_key=method_key,
        method=str(raw.get("method")),
        variant=variant,
        score_mode=score_mode,
        raw_score_mode=raw_mode,
        outcomes=outcome_tuple,
        success_count=count,
        success_rate=rate,
        success_rate_percent=percent,
        checkpoint_epoch=expected_epoch,
        checkpoint_global_step=expected_step,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        parent_v1_c_checkpoint_sha256=PARENT_V1_C_CHECKPOINT_SHA256,
        training_seed=TRAINING_SEED,
        planning_seed=PLANNING_SEED,
        selection_sha256=SELECTION_SHA256,
        selection_ranks_sha256=SELECTION_RANKS_SHA256,
        action_normalization_sha256=ACTION_NORMALIZATION_SHA256,
        evaluation_commit=evaluation_commit,
        source_directory=source_directory,
        source_files_sha256=hashes,
    )


def load_endpoint_extension_ledger(
    path: str | Path,
) -> tuple[EndpointResultCell, ...]:
    """Load and revalidate all eight cells from a self-contained archive.

    ``path`` may name the artifact directory or its
    ``reconciliation_ledger.json`` file.  Partial ledgers are never returned.
    """

    supplied = Path(path).expanduser().resolve()
    ledger_path = (
        supplied / "reconciliation_ledger.json" if supplied.is_dir() else supplied
    )
    root = ledger_path.parent
    ledger, _ = _read_json(ledger_path, context="endpoint_ledger")
    _expect(
        set(ledger),
        {
            "schema_version",
            "source",
            "cell_count",
            "outcome_count",
            "episodes_per_cell",
            "training_seed",
            "planning_seed",
            "first_q_weight",
            "selection_sha256",
            "selection_ranks_sha256",
            "action_normalization_sha256",
            "endpoint_score_modes",
            "all_cell_score_modes",
            "expected_identities",
            "evaluation_commits",
            "sources",
            "training_sources",
            "cells",
        },
        context="endpoint_ledger.keys",
    )
    header = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "cell_count": 8,
        "outcome_count": 400,
        "episodes_per_cell": EPISODES,
        "training_seed": TRAINING_SEED,
        "planning_seed": PLANNING_SEED,
        "first_q_weight": FIRST_Q_WEIGHT,
        "selection_sha256": SELECTION_SHA256,
        "selection_ranks_sha256": SELECTION_RANKS_SHA256,
        "action_normalization_sha256": ACTION_NORMALIZATION_SHA256,
        "endpoint_score_modes": list(ENDPOINT_SCORE_MODES),
        "all_cell_score_modes": list(ALL_ENDPOINT_CELL_SCORE_MODES),
        "expected_identities": [
            {"method_key": method_key, "score_mode": score_mode}
            for method_key, score_mode in EXPECTED_ENDPOINT_IDENTITIES
        ],
    }
    for key, value in header.items():
        _expect(ledger.get(key), value, context=f"endpoint_ledger.{key}")
    raw_cells = ledger.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 8:
        raise _error("endpoint_ledger", "cells must contain exactly eight entries")
    cells = tuple(
        _cell_from_document(
            _mapping(raw, context=f"endpoint_ledger.cells[{index}]"),
            context=f"endpoint_ledger.cells[{index}]",
        )
        for index, raw in enumerate(raw_cells)
    )
    _expect(
        tuple(cell.identity for cell in cells),
        EXPECTED_ENDPOINT_IDENTITIES,
        context="endpoint_ledger.identities",
    )
    expected_commits = sorted({cell.evaluation_commit for cell in cells})
    _expect(
        ledger.get("evaluation_commits"),
        expected_commits,
        context="endpoint_ledger.evaluation_commits",
    )
    sources = _mapping(ledger.get("sources"), context="endpoint_ledger.sources")
    _expect(
        set(sources),
        {"c2_launcher_manifest", "c2_checkpoint_manifest", "c3_output_dir"},
        context="endpoint_ledger.sources.keys",
    )
    for name in ("c2_launcher_manifest", "c2_checkpoint_manifest"):
        source = _mapping(sources[name], context=f"endpoint_ledger.sources.{name}")
        _expect(
            set(source),
            {"path", "sha256"},
            context=f"endpoint_ledger.sources.{name}.keys",
        )
        if not isinstance(source.get("path"), str):
            raise _error(f"endpoint_ledger.sources.{name}", "path must be a string")
        _sha256(source.get("sha256"), context=f"endpoint_ledger.sources.{name}.sha256")
    if not isinstance(sources.get("c3_output_dir"), str):
        raise _error("endpoint_ledger.sources.c3_output_dir", "must be a string")
    training_sources = _mapping(
        ledger.get("training_sources"), context="endpoint_ledger.training_sources"
    )
    if not set(training_sources) <= {"v1_c2", "v1_c3"}:
        raise _error(
            "endpoint_ledger.training_sources", "contains an unsupported method"
        )
    archived_training: list[TrainingArtifactSet] = []
    for method_key, raw_source in training_sources.items():
        source = _mapping(
            raw_source, context=f"endpoint_ledger.training_sources.{method_key}"
        )
        _expect(
            set(source),
            {"source_run_dir", "archive_directory", "files_sha256"},
            context=f"endpoint_ledger.training_sources.{method_key}.keys",
        )
        archive_directory = f"training/{method_key}"
        _expect(
            source.get("archive_directory"),
            archive_directory,
            context=f"endpoint_ledger.training_sources.{method_key}.archive_directory",
        )
        if not isinstance(source.get("source_run_dir"), str):
            raise _error(
                f"endpoint_ledger.training_sources.{method_key}",
                "source_run_dir must be a string",
            )
        raw_hashes = _mapping(
            source.get("files_sha256"),
            context=f"endpoint_ledger.training_sources.{method_key}.files_sha256",
        )
        expected_names = {
            "training_result.json",
            "training_manifest.json",
            "metrics.csv",
        }
        if method_key == "v1_c3":
            expected_names.add("validation_offline_metrics.json")
        _expect(
            set(raw_hashes),
            expected_names,
            context=f"endpoint_ledger.training_sources.{method_key}.files",
        )
        archive_run = root / archive_directory
        files = {name: archive_run / name for name in expected_names}
        hashes = {
            name: _sha256(
                raw_hashes[name],
                context=f"endpoint_ledger.training_sources.{method_key}.{name}",
            )
            for name in expected_names
        }
        archived_training.append(
            TrainingArtifactSet(
                method_key=method_key,
                run_dir=archive_run,
                files=files,
                files_sha256=hashes,
            )
        )
    _expect(
        _archive_file_set(root),
        _expected_archive_file_set(cells, archived_training),
        context="endpoint_archive.file_set",
    )
    for artifact in archived_training:
        for name, path_value in artifact.files.items():
            if not path_value.is_file():
                raise _error(
                    "endpoint_archive", f"missing training evidence {path_value}"
                )
            _expect(
                _file_sha256(path_value),
                artifact.files_sha256[name],
                context=f"endpoint_archive.training.{artifact.method_key}.{name}",
            )

    validated: list[EndpointResultCell] = []
    for recorded in cells:
        source = root / recorded.source_directory
        if not _path_is_within(source, root / "sources"):
            raise _error(recorded.cell_id, "source directory escapes archive")
        _expect(
            source.resolve(),
            (root / "sources" / recorded.cell_id).resolve(),
            context=f"{recorded.cell_id}.source_directory",
        )
        actual_hashes = _source_hashes(source, context=recorded.cell_id)
        _expect(
            actual_hashes,
            dict(recorded.source_files_sha256),
            context=f"{recorded.cell_id}.source_files_sha256",
        )
        actual = _validate_output_dir(
            source,
            method_key=recorded.method_key,
            raw_score_mode=recorded.raw_score_mode,
            require_checkpoint_file=False,
            expected_checkpoint_sha256=recorded.checkpoint_sha256,
            expected_evaluation_commit=recorded.evaluation_commit,
        )
        for field in (
            "outcomes",
            "success_count",
            "success_rate",
            "success_rate_percent",
            "checkpoint_epoch",
            "checkpoint_global_step",
            "parent_v1_c_checkpoint_sha256",
            "training_seed",
            "planning_seed",
            "selection_sha256",
            "selection_ranks_sha256",
            "action_normalization_sha256",
        ):
            _expect(
                getattr(actual, field),
                getattr(recorded, field),
                context=f"{recorded.cell_id}.{field}",
            )
        validated.append(recorded)
    validated_tuple = tuple(validated)
    c2_hashes = {
        cell.checkpoint_sha256 for cell in validated_tuple if cell.method_key == "v1_c2"
    }
    if len(c2_hashes) != 1:
        raise _error("endpoint_ledger", "the six C2 cells do not share one checkpoint")
    if (root / "all_o50_results.csv").read_bytes() != _csv_bytes(validated_tuple):
        raise _error("endpoint_archive", "all_o50_results.csv differs from ledger")
    if (root / "summary.json").read_bytes() != _json_bytes(
        _summary_document(validated_tuple)
    ):
        raise _error("endpoint_archive", "summary.json differs from ledger")
    return validated_tuple


__all__ = [
    "ACTION_NORMALIZATION_SHA256",
    "ALL_ENDPOINT_CELL_SCORE_MODES",
    "ENDPOINT_SCORE_MODES",
    "EXPECTED_ENDPOINT_IDENTITIES",
    "EndpointReconciliationError",
    "EndpointResultCell",
    "PARENT_V1_C_CHECKPOINT_SHA256",
    "SELECTION_RANKS_SHA256",
    "SELECTION_SHA256",
    "TrainingArtifactSet",
    "ValidatedEndpointStudy",
    "build_ledger",
    "load_endpoint_extension_ledger",
    "reconcile_endpoint_results",
    "write_archive",
]
