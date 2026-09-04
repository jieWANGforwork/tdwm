#!/usr/bin/env python3
"""Build the complete Results TD ledger report from 477 audited O50 cells.

The existing Results TD reports were produced incrementally.  This builder is
the consolidation layer: it requires the exact legacy/V0/V1/V2/V2-EMA grid,
checks every identity and rate, then produces a standalone Markdown report and
a standalone DOCX which presents one fixed-E10 decision matrix, derives the
cross-version decision analysis from it, and points to superseded source pages
only in a clearly marked historical appendix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent


def _load_v2_report_builder() -> ModuleType:
    module_name = "_tdwm_results_td_v2_ema_builder"
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPT_DIR / "build_results_td_v2_ema_new_scores.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the V2-EMA Results TD builder.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


v2_report = _load_v2_report_builder()


DEFAULT_LEDGER = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/all_o50_results.csv"
)
DEFAULT_RECONCILIATION_LEDGER = DEFAULT_LEDGER.parent / "reconciliation_ledger.json"
DEFAULT_MARKDOWN_OUTPUT = (
    REPOSITORY_ROOT / "reports/actor_free_td_lewm_complete_cube_seed3072.md"
)
DEFAULT_DOCX_OUTPUT = (
    REPOSITORY_ROOT
    / "reports/results_td_actor_free_td_lewm_complete_cube_seed3072.docx"
)
DEFAULT_ROOT_DOCX_OUTPUT = REPOSITORY_ROOT.parents[1] / "Results TD.docx"
DEFAULT_BASE_DOCUMENT = v2_report.DEFAULT_BASE_DOCUMENT
DEFAULT_CHART_DIR = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_complete_cube_seed3072/figures"
)
DEFAULT_V0_TRAINING_CSV = DEFAULT_LEDGER.parent / "v0_training_loss_curves.csv"
DEFAULT_V2_TRAINING_CSV = DEFAULT_LEDGER.parent / "v2_training_loss_curves.csv"

EPISODES = 50
COMPLETE_CELL_COUNT = 477
COMPLETE_OUTCOME_COUNT = COMPLETE_CELL_COUNT * EPISODES
TRAINING_SEED = 3072
PLANNING_SEED = 42
STEPS_PER_EPOCH = 12_796
SELECTION_SHA256 = v2_report.SHARED_EPISODE_SELECTION_SHA256
ACTION_NORMALIZATION_SHA256 = v2_report.ACTION_NORMALIZATION_SHA256
FIXED_SELECTION_RANKS_SHA256 = (
    "88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9"
)
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
VERSIONS = ("v0", "v1", "v2", "v2_ema_sg")
VERSION_LABELS = {
    "v0": "V0",
    "v1": "V1",
    "v2": "V2",
    "v2_ema_sg": "V2-EMA",
}
ORIGINAL_MODES = ("f_only", "g_only", "f_plus_g")
NEW_MODES = ("f_plus_g_first", "g_only_f_rollout_mean")
ALL_CONTROLLED_MODES = ORIGINAL_MODES + NEW_MODES
EMA_EPOCHS = tuple(range(3, 11))
LEGACY_METHODS = (
    "serial_decoupled",
    "serial_coupled",
    "hybrid",
    "parallel_real",
    "goal_hybrid",
    "imaginary_hybrid",
    "direct_goal_hybrid",
)
LEGACY_LABELS = {
    "serial_decoupled": "Serial Decoupled",
    "serial_coupled": "Serial Coupled",
    "hybrid": "Hybrid",
    "parallel_real": "Parallel Real",
    "goal_hybrid": "Goal Hybrid",
    "imaginary_hybrid": "Imaginary Hybrid",
    "direct_goal_hybrid": "Direct Goal Critic Hybrid",
}
MODE_LABELS = {
    "f_only": "F-only",
    "g_only": "G-only",
    "f_plus_g": "F+G tail",
    "c_only": "C-only",
    "f_plus_c": "F+C",
    "f_plus_g_first": "F + first-Q",
    "g_only_f_rollout_mean": "Mean-Q rollout",
}
LEDGER_SCORE_LABELS = {
    "f_only": "F-only",
    "g_only": "G-only",
    "f_plus_g": "F+G",
    "c_only": "C-only",
    "f_plus_c": "F+C",
    "f_plus_g_first": "F+G First (alpha=0.25)",
    "g_only_f_rollout_mean": "Mean-Q over F rollout",
}

FIELDS = (
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


class CompleteResultsError(ValueError):
    """The complete Results TD evidence is missing or inconsistent."""


@dataclass(frozen=True)
class ResultCell:
    values: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]


@dataclass(frozen=True)
class FixedAnalysis:
    """Derived fixed-E10 comparisons used by every report surface."""

    mode_means: Mapping[str, float]
    mode_best_counts: Mapping[str, int]
    mode_winners: Mapping[str, tuple[str, ...]]
    overall_best_count: int
    overall_winners: tuple[str, ...]
    best_mean_modes: tuple[str, ...]
    best_mean_percent: float
    version_mode_means: Mapping[tuple[str, str], float]
    ema_variant_means: Mapping[str, float]
    ema_best_variants: tuple[str, ...]


@dataclass(frozen=True)
class LedgerEvidence:
    """Companion files that preserve the full scalar and per-pair audit trail."""

    csv_path: str
    csv_sha256: str
    json_path: str
    json_sha256: str
    cell_count: int
    outcome_count: int


def _sha(value: str, *, context: str, allow_empty: bool = False) -> str:
    value = value.strip()
    if allow_empty and value == "":
        return value
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CompleteResultsError(f"{context} must be a lowercase SHA-256.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_ledger_evidence(csv_path: str | Path, json_path: str | Path) -> LedgerEvidence:
    """Validate and fingerprint the two files that preserve the complete ledger."""

    csv_source = Path(csv_path)
    json_source = Path(json_path)
    if not csv_source.is_file():
        raise FileNotFoundError(f"Complete CSV ledger does not exist: {csv_source}")
    if not json_source.is_file():
        raise FileNotFoundError(
            f"Reconciliation JSON ledger does not exist: {json_source}"
        )

    with csv_source.open(newline="", encoding="utf-8") as stream:
        csv_cell_count = sum(1 for _ in csv.DictReader(stream))
    if csv_cell_count != COMPLETE_CELL_COUNT:
        raise CompleteResultsError(
            "Companion CSV must contain "
            f"{COMPLETE_CELL_COUNT} cells, found {csv_cell_count}."
        )

    try:
        reconciliation = json.loads(json_source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompleteResultsError(
            f"Reconciliation ledger is not valid JSON: {json_source}."
        ) from exc
    if not isinstance(reconciliation, dict):
        raise CompleteResultsError("Reconciliation ledger root must be an object.")
    if reconciliation.get("cell_count") != COMPLETE_CELL_COUNT:
        raise CompleteResultsError(
            f"Reconciliation ledger cell_count must be {COMPLETE_CELL_COUNT}."
        )
    if reconciliation.get("episode_count_per_cell") != EPISODES:
        raise CompleteResultsError(
            f"Reconciliation ledger must preserve {EPISODES} outcomes per cell."
        )
    if reconciliation.get("outcome_count") != COMPLETE_OUTCOME_COUNT:
        raise CompleteResultsError(
            f"Reconciliation ledger outcome_count must be {COMPLETE_OUTCOME_COUNT}."
        )
    json_cells = reconciliation.get("cells")
    if not isinstance(json_cells, list) or len(json_cells) != COMPLETE_CELL_COUNT:
        raise CompleteResultsError(
            "Reconciliation ledger cells must contain exactly "
            f"{COMPLETE_CELL_COUNT} entries."
        )
    for index, cell in enumerate(json_cells):
        if not isinstance(cell, dict):
            raise CompleteResultsError(
                f"reconciliation.cells[{index}] is not an object."
            )
        outcomes = cell.get("outcomes")
        if (
            not isinstance(outcomes, list)
            or len(outcomes) != EPISODES
            or any(type(value) is not bool for value in outcomes)
        ):
            raise CompleteResultsError(
                f"reconciliation.cells[{index}].outcomes must contain "
                f"{EPISODES} Booleans."
            )

    return LedgerEvidence(
        csv_path=_display_path(csv_source),
        csv_sha256=_file_sha256(csv_source),
        json_path=_display_path(json_source),
        json_sha256=_file_sha256(json_source),
        cell_count=csv_cell_count,
        outcome_count=sum(len(cell["outcomes"]) for cell in json_cells),
    )


def _int(value: str, *, context: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CompleteResultsError(f"{context} must be an integer.") from exc
    if str(parsed) != value.strip():
        raise CompleteResultsError(f"{context} is not an exact integer: {value!r}.")
    return parsed


def _float(value: str, *, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CompleteResultsError(f"{context} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise CompleteResultsError(f"{context} must be finite.")
    return parsed


def _optional_int(value: str, *, context: str) -> int | None:
    if value.strip() == "":
        return None
    return _int(value, context=context)


def _expected_identities() -> set[tuple[str, str, int, str]]:
    expected: set[tuple[str, str, int, str]] = set()
    for method in LEGACY_METHODS:
        modes = (
            ("f_only", "c_only", "f_plus_c")
            if method == "direct_goal_hybrid"
            else ORIGINAL_MODES
        )
        expected.update(("legacy", method, 10, mode) for mode in modes)
    for version in ("v0", "v1"):
        expected.update(
            (version, variant, 10, mode)
            for variant in VARIANTS
            for mode in ALL_CONTROLLED_MODES
        )
    expected.update(
        ("v2", variant, epoch, mode)
        for epoch in EMA_EPOCHS
        for variant in VARIANTS
        for mode in ORIGINAL_MODES
    )
    expected.update(
        ("v2", variant, 10, mode) for variant in VARIANTS for mode in NEW_MODES
    )
    expected.update(
        ("v2_ema_sg", variant, epoch, mode)
        for epoch in EMA_EPOCHS
        for variant in VARIANTS
        for mode in ALL_CONTROLLED_MODES
    )
    return expected


def load_complete_ledger(path: str | Path) -> tuple[ResultCell, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Complete O50 ledger does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise CompleteResultsError(
                f"Complete ledger columns changed: {reader.fieldnames!r}."
            )
        raw_rows = tuple(dict(row) for row in reader)
    if len(raw_rows) != COMPLETE_CELL_COUNT:
        raise CompleteResultsError(
            "Complete ledger must contain "
            f"{COMPLETE_CELL_COUNT} cells, found {len(raw_rows)}."
        )

    cells: list[ResultCell] = []
    identities: set[tuple[str, str, int, str]] = set()
    cell_ids: set[str] = set()
    for index, row in enumerate(raw_rows):
        context = f"ledger[{index}]"
        version = row["version"].strip()
        method_key = row["variant"].strip()
        epoch = _int(row["epoch"], context=f"{context}.epoch")
        global_step = _int(row["global_step"], context=f"{context}.global_step")
        mode = row["score_mode"].strip()
        identity = (version, method_key, epoch, mode)
        if identity in identities:
            raise CompleteResultsError(f"Duplicate result identity: {identity}.")
        identities.add(identity)
        cell_id = row["cell_id"].strip()
        if not cell_id or cell_id in cell_ids:
            raise CompleteResultsError(f"{context}.cell_id is empty or duplicated.")
        cell_ids.add(cell_id)
        successes = _int(row["success_count"], context=f"{context}.success_count")
        episodes = _int(row["episode_count"], context=f"{context}.episode_count")
        rate = _float(row["success_rate"], context=f"{context}.success_rate")
        percent = _float(
            row["success_rate_percent"], context=f"{context}.success_rate_percent"
        )
        if episodes != EPISODES or not 0 <= successes <= episodes:
            raise CompleteResultsError(f"{context} must be a valid O50 result.")
        if not math.isclose(rate, successes / episodes, rel_tol=0.0, abs_tol=1e-12):
            raise CompleteResultsError(f"{context}.success_rate disagrees with count.")
        if not math.isclose(percent, 100.0 * rate, rel_tol=0.0, abs_tol=1e-9):
            raise CompleteResultsError(f"{context}.success_rate_percent is stale.")
        if (
            _int(row["training_seed"], context=f"{context}.training_seed")
            != TRAINING_SEED
        ):
            raise CompleteResultsError(f"{context}.training_seed changed.")
        if (
            _int(row["planning_seed"], context=f"{context}.planning_seed")
            != PLANNING_SEED
        ):
            raise CompleteResultsError(f"{context}.planning_seed changed.")
        if row["selection_sha256"] != SELECTION_SHA256:
            raise CompleteResultsError(f"{context} uses a different O50 selection.")
        if row["action_normalization_sha256"] != ACTION_NORMALIZATION_SHA256:
            raise CompleteResultsError(f"{context} action normalization changed.")
        _sha(row["checkpoint_sha256"], context=f"{context}.checkpoint_sha256")
        _sha(row["source_results_sha256"], context=f"{context}.source_results_sha256")
        _sha(
            row["source_protocol_sha256"],
            context=f"{context}.source_protocol_sha256",
            allow_empty=True,
        )
        _sha(row["outcomes_sha256"], context=f"{context}.outcomes_sha256")
        if global_step != epoch * STEPS_PER_EPOCH:
            raise CompleteResultsError(f"{context}.global_step disagrees with epoch.")
        if row["status"] != "VERIFIED":
            raise CompleteResultsError(f"{context}.status must be VERIFIED.")
        expected_family = (
            "actor_free_td_lewm"
            if version == "legacy"
            else f"actor_free_td_lewm_{version}"
        )
        expected_method_id = (
            "actor_free_td_lewm"
            if version == "legacy"
            else f"{expected_family}_{method_key}"
        )
        if row["family"] != expected_family:
            raise CompleteResultsError(
                f"{context}.family is inconsistent with version."
            )
        if row["method_id"] != expected_method_id:
            raise CompleteResultsError(f"{context}.method_id is inconsistent.")
        if (
            not row["method_label"].strip()
            or row["score_label"] != LEDGER_SCORE_LABELS[mode]
        ):
            raise CompleteResultsError(
                f"{context} has a missing or stale display label."
            )
        if not row["training_commit"].strip() or not row["evaluation_commit"].strip():
            raise CompleteResultsError(f"{context} is missing commit provenance.")
        if not row["source_kind"].strip() or not row["source_path"].strip():
            raise CompleteResultsError(f"{context} is missing source provenance.")
        alpha = row["g_first_weight"].strip()
        if mode == "f_plus_g_first":
            if not math.isclose(
                _float(alpha, context=f"{context}.g_first_weight"),
                0.25,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise CompleteResultsError(f"{context}.g_first_weight changed.")
        elif alpha != "":
            raise CompleteResultsError(f"{context} has an unexpected g_first_weight.")
        planning_horizon = _optional_int(
            row["planning_horizon"], context=f"{context}.planning_horizon"
        )
        if planning_horizon is not None:
            expected_horizon = (
                5 if version == "legacy" else (1 if mode == "g_only" else 5)
            )
            if planning_horizon != expected_horizon:
                raise CompleteResultsError(f"{context}.planning_horizon changed.")
        normalized = dict(row)
        normalized.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "success_count": successes,
                "episode_count": episodes,
                "success_rate": rate,
                "success_rate_percent": percent,
                "planning_horizon": planning_horizon,
            }
        )
        cells.append(ResultCell(normalized))
    expected = _expected_identities()
    if identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise CompleteResultsError(
            f"Complete grid mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}."
        )
    checkpoints: dict[tuple[str, str, int], set[str]] = {}
    for cell in cells:
        version = str(cell["version"])
        method_key = str(cell["variant"])
        key = (version, method_key, int(cell["epoch"]))
        checkpoints.setdefault(key, set()).add(str(cell["checkpoint_sha256"]))
    inconsistent = {
        key: values for key, values in checkpoints.items() if len(values) != 1
    }
    if inconsistent:
        raise CompleteResultsError(
            f"Score modes do not share a checkpoint: {next(iter(inconsistent.items()))!r}."
        )
    fixed_mean_q = {
        (str(cell["version"]), str(cell["variant"]))
        for cell in cells
        if int(cell["epoch"]) == 10
        and cell["score_mode"] == "g_only_f_rollout_mean"
        and cell["version"] in VERSIONS
    }
    expected_fixed_mean_q = {
        (version, variant) for version in VERSIONS for variant in VARIANTS
    }
    if fixed_mean_q != expected_fixed_mean_q:
        raise CompleteResultsError(
            "Fixed E10 Mean-Q coverage must contain exactly 24 version/variant cells."
        )
    return tuple(cells)


def _find(
    cells: Iterable[ResultCell],
    *,
    version: str,
    epoch: int,
    mode: str,
    variant: str | None = None,
    method_id: str | None = None,
) -> ResultCell:
    matches = [
        cell
        for cell in cells
        if cell["version"] == version
        and cell["epoch"] == epoch
        and cell["score_mode"] == mode
        and (variant is None or cell["variant"] == variant)
        and (method_id is None or cell["method_id"] == method_id)
    ]
    if len(matches) != 1:
        raise CompleteResultsError(
            f"Expected one cell for {version}/{variant or method_id}/E{epoch}/{mode}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _result(cell: ResultCell) -> str:
    return f"{cell['success_count']}/50 ({cell['success_rate_percent']:.0f}%)"


def _percent(cell: ResultCell) -> float:
    return float(cell["success_rate_percent"])


def _mean(cells: Iterable[ResultCell]) -> float:
    selected = tuple(cells)
    if not selected:
        raise CompleteResultsError("Cannot average an empty result set.")
    return sum(_percent(cell) for cell in selected) / len(selected)


def _fixed_cell_label(cell: ResultCell) -> str:
    return f"{VERSION_LABELS[str(cell['version'])]}-{str(cell['variant']).upper()}"


def _fixed_analysis(cells: Sequence[ResultCell]) -> FixedAnalysis:
    fixed = tuple(
        _find(cells, version=version, variant=variant, epoch=10, mode=mode)
        for version in VERSIONS
        for variant in VARIANTS
        for mode in ALL_CONTROLLED_MODES
    )
    if len(fixed) != len(VERSIONS) * len(VARIANTS) * len(ALL_CONTROLLED_MODES):
        raise CompleteResultsError("Fixed E10 decision grid is incomplete.")
    version_mode_means = {
        (version, mode): _mean(
            _find(cells, version=version, variant=variant, epoch=10, mode=mode)
            for variant in VARIANTS
        )
        for version in VERSIONS
        for mode in ALL_CONTROLLED_MODES
    }
    mode_means = {
        mode: _mean(cell for cell in fixed if cell["score_mode"] == mode)
        for mode in ALL_CONTROLLED_MODES
    }
    mode_best_counts: dict[str, int] = {}
    mode_winners: dict[str, tuple[str, ...]] = {}
    for mode in ALL_CONTROLLED_MODES:
        selected = tuple(cell for cell in fixed if cell["score_mode"] == mode)
        best = max(int(cell["success_count"]) for cell in selected)
        mode_best_counts[mode] = best
        mode_winners[mode] = tuple(
            _fixed_cell_label(cell)
            for cell in selected
            if int(cell["success_count"]) == best
        )
    overall_best = max(int(cell["success_count"]) for cell in fixed)
    overall_winners = tuple(
        f"{_fixed_cell_label(cell)} + {MODE_LABELS[str(cell['score_mode'])]}"
        for cell in fixed
        if int(cell["success_count"]) == overall_best
    )
    best_mean = max(mode_means.values())
    best_mean_modes = tuple(
        mode for mode in ALL_CONTROLLED_MODES if mode_means[mode] == best_mean
    )
    ema_variant_means = {
        variant: sum(
            _percent(
                _find(
                    cells,
                    version="v2_ema_sg",
                    variant=variant,
                    epoch=10,
                    mode=mode,
                )
            )
            for mode in ALL_CONTROLLED_MODES
        )
        / len(ALL_CONTROLLED_MODES)
        for variant in VARIANTS
    }
    ema_best_mean = max(ema_variant_means.values())
    return FixedAnalysis(
        mode_means=mode_means,
        mode_best_counts=mode_best_counts,
        mode_winners=mode_winners,
        overall_best_count=overall_best,
        overall_winners=overall_winners,
        best_mean_modes=best_mean_modes,
        best_mean_percent=best_mean,
        version_mode_means=version_mode_means,
        ema_variant_means=ema_variant_means,
        ema_best_variants=tuple(
            variant
            for variant in VARIANTS
            if ema_variant_means[variant] == ema_best_mean
        ),
    )


def _joined(values: Sequence[str]) -> str:
    return ", ".join(values)


def _compact_join(values: Sequence[str], *, limit: int = 3) -> str:
    """Keep tied-winner summaries bounded; the matrix retains every highlight."""

    items = tuple(values)
    if len(items) <= limit:
        return _joined(items)
    return f"{len(items)} tied: {_joined(items[:limit])}, …"


def _winner_result(analysis: FixedAnalysis, mode: str) -> str:
    count = analysis.mode_best_counts[mode]
    return f"{_compact_join(analysis.mode_winners[mode])}: {count}/50 ({count * 2}%)"


def _column_winner_summary(analysis: FixedAnalysis) -> str:
    return "; ".join(
        f"{MODE_LABELS[mode]} = {_winner_result(analysis, mode)}"
        for mode in ALL_CONTROLLED_MODES
    )


def _signed_pp(value: float) -> str:
    return f"{value:+.1f} pp"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join("---" if index == 0 else "---:" for index in range(len(headers)))
        + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _legacy_rows(cells: Sequence[ResultCell]) -> tuple[tuple[str, ...], ...]:
    rows = []
    for method in LEGACY_METHODS:
        secondary = "c_only" if method == "direct_goal_hybrid" else "g_only"
        combined = "f_plus_c" if method == "direct_goal_hybrid" else "f_plus_g"
        rows.append(
            (
                LEGACY_LABELS[method],
                _result(
                    _find(
                        cells, version="legacy", variant=method, epoch=10, mode="f_only"
                    )
                ),
                _result(
                    _find(
                        cells,
                        version="legacy",
                        variant=method,
                        epoch=10,
                        mode=secondary,
                    )
                ),
                _result(
                    _find(
                        cells, version="legacy", variant=method, epoch=10, mode=combined
                    )
                ),
            )
        )
    return tuple(rows)


def _fixed_master_rows(
    cells: Sequence[ResultCell], *, epoch: int = 10
) -> tuple[tuple[str, ...], ...]:
    """Return one comparison matrix across every controlled training setup.

    The matrix deliberately uses one prespecified checkpoint per version and
    requires all five inference scores for all 24 training configurations.
    """

    version_labels = {
        "v0": "V0",
        "v1": "V1",
        "v2": "V2",
        "v2_ema_sg": "V2-EMA",
    }
    rows: list[tuple[str, ...]] = []
    for version in VERSIONS:
        for variant in VARIANTS:
            values = [
                _result(
                    _find(
                        cells,
                        version=version,
                        variant=variant,
                        epoch=epoch,
                        mode=mode,
                    )
                )
                for mode in ALL_CONTROLLED_MODES
            ]
            rows.append((version_labels[version], variant.upper(), *values))
    return tuple(rows)


def _fixed_master_counts(
    cells: Sequence[ResultCell], *, epoch: int = 10
) -> tuple[tuple[int, ...], ...]:
    """Numeric companion to ``_fixed_master_rows`` for deterministic highlighting."""

    counts: list[tuple[int, ...]] = []
    for version in VERSIONS:
        for variant in VARIANTS:
            row: list[int] = []
            for mode in ALL_CONTROLLED_MODES:
                row.append(
                    int(
                        _find(
                            cells,
                            version=version,
                            variant=variant,
                            epoch=epoch,
                            mode=mode,
                        )["success_count"]
                    )
                )
            counts.append(tuple(row))
    return tuple(counts)


def _fixed_master_markdown_rows(
    cells: Sequence[ResultCell], *, epoch: int = 10
) -> tuple[tuple[str, ...], ...]:
    """Format the master matrix so row and column winners are visible at a glance."""

    display_rows = _fixed_master_rows(cells, epoch=epoch)
    count_rows = _fixed_master_counts(cells, epoch=epoch)
    column_maxima = tuple(
        max(row[index] for row in count_rows)
        for index in range(len(ALL_CONTROLLED_MODES))
    )
    formatted: list[tuple[str, ...]] = []
    for display_row, count_row in zip(display_rows, count_rows):
        row_maximum = max(value for value in count_row if value is not None)
        scores = []
        for index, (display, count) in enumerate(zip(display_row[2:], count_row)):
            text = f"**{display}**" if count == row_maximum else display
            if count == column_maxima[index]:
                text = f"★ {text}"
            scores.append(text)
        formatted.append((*display_row[:2], *scores))
    return tuple(formatted)


def _trajectory_rows(
    cells: Sequence[ResultCell], version: str, mode: str
) -> tuple[tuple[str, ...], ...]:
    epochs = EMA_EPOCHS if version in ("v2", "v2_ema_sg") else (10,)
    return tuple(
        (
            f"E{epoch}",
            *(
                _result(
                    _find(
                        cells, version=version, variant=variant, epoch=epoch, mode=mode
                    )
                )
                for variant in VARIANTS
            ),
        )
        for epoch in epochs
    )


def _global_posthoc_rows(cells: Sequence[ResultCell]) -> tuple[tuple[str, ...], ...]:
    rows = []
    for version, modes in (("v2", ORIGINAL_MODES), ("v2_ema_sg", ALL_CONTROLLED_MODES)):
        for mode in modes:
            candidates = [
                cell
                for cell in cells
                if cell["version"] == version and cell["score_mode"] == mode
            ]
            best = max(int(cell["success_count"]) for cell in candidates)
            winners = [
                cell for cell in candidates if int(cell["success_count"]) == best
            ]
            labels = ", ".join(
                f"{cell['variant'].upper()}-E{cell['epoch']}" for cell in winners
            )
            formal_e10 = max(
                int(cell["success_count"])
                for cell in candidates
                if int(cell["epoch"]) == 10
            )
            rows.append(
                (
                    "V2" if version == "v2" else "V2-EMA",
                    MODE_LABELS[mode],
                    labels,
                    f"{best}/50 ({best * 2}%)",
                    f"{formal_e10}/50 ({formal_e10 * 2}%)",
                )
            )
    return tuple(rows)


def _ema_new_stability_rows(cells: Sequence[ResultCell]) -> tuple[tuple[str, ...], ...]:
    rows = []
    for variant in VARIANTS:
        for mode in NEW_MODES:
            selected = [
                cell
                for cell in cells
                if cell["version"] == "v2_ema_sg"
                and cell["variant"] == variant
                and cell["score_mode"] == mode
            ]
            values = [_percent(cell) for cell in selected]
            mean = sum(values) / len(values)
            sigma = math.sqrt(
                sum((value - mean) ** 2 for value in values) / len(values)
            )
            best = max(values)
            best_epochs = ",".join(
                f"E{cell['epoch']}" for cell in selected if _percent(cell) == best
            )
            e10 = _percent(
                _find(cells, version="v2_ema_sg", variant=variant, epoch=10, mode=mode)
            )
            rows.append(
                (
                    variant.upper(),
                    MODE_LABELS[mode],
                    f"{mean:.1f}%",
                    f"{sigma:.1f}",
                    f"{best_epochs} / {best:.0f}%",
                    f"{e10:.0f}%",
                )
            )
    return tuple(rows)


def _epoch10_mode_mean(cells: Sequence[ResultCell], version: str, mode: str) -> float:
    return _mean(
        cell
        for cell in cells
        if cell["version"] == version
        and int(cell["epoch"]) == 10
        and cell["score_mode"] == mode
    )


def _fixed_version_mean_rows(
    cells: Sequence[ResultCell],
) -> tuple[tuple[str, ...], ...]:
    """Summarize fixed-E10 version quality without mixing in post-hoc epochs."""

    rows = []
    for version, label in (
        ("v0", "V0 raw action"),
        ("v1", "V1 action encoder"),
        ("v2", "V2 joint fine-tune"),
        ("v2_ema_sg", "V2-EMA-SG"),
    ):
        values = tuple(
            _epoch10_mode_mean(cells, version, mode) for mode in ALL_CONTROLLED_MODES
        )
        rows.append(
            (
                label,
                *(f"{value:.1f}%" for value in values),
                f"{sum(values) / len(values):.1f}%",
            )
        )
    return tuple(rows)


def _fixed_score_mean_rows(
    cells: Sequence[ResultCell],
) -> tuple[tuple[str, ...], ...]:
    """Compare readouts only on versions for which each readout was evaluated."""

    rows = []
    for mode in ALL_CONTROLLED_MODES:
        versions = VERSIONS
        per_version = [_epoch10_mode_mean(cells, version, mode) for version in versions]
        fixed_cells = [
            cell
            for cell in cells
            if cell["version"] in versions
            and int(cell["epoch"]) == 10
            and cell["score_mode"] == mode
        ]
        best = max(int(cell["success_count"]) for cell in fixed_cells)
        winner_labels = tuple(
            f"{str(cell['version']).replace('v2_ema_sg', 'V2-EMA').upper()}-"
            f"{str(cell['variant']).upper()}"
            for cell in fixed_cells
            if int(cell["success_count"]) == best
        )
        rows.append(
            (
                MODE_LABELS[mode],
                "/".join(
                    version.replace("v2_ema_sg", "V2-EMA").upper()
                    for version in versions
                ),
                str(len(fixed_cells)),
                f"{sum(per_version) / len(per_version):.1f}%",
                f"{_compact_join(winner_labels)}: {best}/50 ({best * 2}%)",
            )
        )
    return tuple(rows)


def _ema_variant_mean_rows(
    cells: Sequence[ResultCell],
) -> tuple[tuple[str, ...], ...]:
    rows = []
    for variant in VARIANTS:
        values = [
            _percent(
                _find(
                    cells,
                    version="v2_ema_sg",
                    variant=variant,
                    epoch=10,
                    mode=mode,
                )
            )
            for mode in ALL_CONTROLLED_MODES
        ]
        best_index = max(range(len(values)), key=values.__getitem__)
        rows.append(
            (
                variant.upper(),
                f"{sum(values) / len(values):.1f}%",
                MODE_LABELS[ALL_CONTROLLED_MODES[best_index]],
                f"{values[best_index]:.0f}%",
            )
        )
    return tuple(rows)


def build_training_chart_from_archive(path: str | Path, *, title: str) -> bytes:
    """Render the common total-loss fields from a reconciled 60-row training CSV."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = tuple(dict(row) for row in csv.DictReader(stream))
    required = {"variant", "epoch", "train/loss_epoch", "validation/loss"}
    if len(rows) != 60 or not rows or not required.issubset(rows[0]):
        raise CompleteResultsError(f"Training curve archive is incomplete: {source}.")

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (2160, 860), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 28),
        title,
        fill="#0B2545",
        font=v2_report._chart_font(34, bold=True),
    )
    train_series = []
    validation_series = []
    all_values: list[float] = []
    for variant, color in zip(VARIANTS, v2_report.CHART_COLORS):
        variant_rows = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: _int(row["epoch"], context=f"{source}.{variant}.epoch"),
        )
        if [int(row["epoch"]) for row in variant_rows] != list(range(1, 11)):
            raise CompleteResultsError(
                f"{source} has an incomplete {variant} trajectory."
            )
        train_values = [
            _float(row["train/loss_epoch"], context=f"{source}.{variant}.train")
            for row in variant_rows
        ]
        validation_values = [
            _float(row["validation/loss"], context=f"{source}.{variant}.validation")
            for row in variant_rows
        ]
        if any(value <= 0 for value in train_values + validation_values):
            raise CompleteResultsError(f"{source} contains non-positive loss values.")
        train_series.append((variant.upper(), train_values, color))
        validation_series.append((variant.upper(), validation_values, color))
        all_values.extend(train_values + validation_values)
    log_min = math.floor(min(math.log10(value) for value in all_values) * 2) / 2
    log_max = math.ceil(max(math.log10(value) for value in all_values) * 2) / 2
    y_min = 10**log_min
    y_max = 10**log_max
    ticks = tuple(
        (10.0**power, f"1e{power}")
        for power in range(math.ceil(log_min), math.floor(log_max) + 1)
    )
    if not ticks:
        ticks = ((y_min, f"{y_min:.2g}"), (y_max, f"{y_max:.2g}"))
    for box, panel_title, series in (
        ((55, 90, 1060, 825), "Training total loss", train_series),
        ((1100, 90, 2105, 825), "Validation total loss", validation_series),
    ):
        v2_report._draw_line_panel(
            draw,
            box=box,
            title=panel_title,
            x_values=tuple(range(1, 11)),
            series=series,
            y_min=y_min,
            y_max=y_max,
            y_ticks=ticks,
            y_transform=math.log10,
            y_label="Loss (log scale)",
        )
    return v2_report._save_chart(image)


def build_markdown(cells: Sequence[ResultCell], ledger_evidence: LedgerEvidence) -> str:
    analysis = _fixed_analysis(cells)
    f_plus_g_winner = _winner_result(analysis, "f_plus_g")
    overall_winner = (
        f"{_compact_join(analysis.overall_winners)}: {analysis.overall_best_count}/50 "
        f"({analysis.overall_best_count * 2}%)"
    )
    best_score_names = _joined(
        tuple(MODE_LABELS[mode] for mode in analysis.best_mean_modes)
    )
    ema_best_variants = _compact_join(
        tuple(variant.upper() for variant in analysis.ema_best_variants)
    )
    ema_best_mean = analysis.ema_variant_means[analysis.ema_best_variants[0]]
    v1_f = analysis.version_mode_means[("v1", "f_only")]
    v2_f = analysis.version_mode_means[("v2", "f_only")]
    v1_tail = analysis.version_mode_means[("v1", "f_plus_g")]
    v2_tail = analysis.version_mode_means[("v2", "f_plus_g")]
    ema_mode_summary = "、".join(
        f"{MODE_LABELS[mode]} {analysis.version_mode_means[('v2_ema_sg', mode)]:.1f}%"
        for mode in ALL_CONTROLLED_MODES
    )
    mean_q_winner = _winner_result(analysis, "g_only_f_rollout_mean")
    tail_harm_count = sum(
        _percent(
            _find(cells, version=version, variant=variant, epoch=10, mode="f_plus_g")
        )
        < _percent(
            _find(cells, version=version, variant=variant, epoch=10, mode="f_only")
        )
        for version in VERSIONS
        for variant in VARIANTS
    )
    lines = [
        "# Results TD — 全部 Actor-Free TD-LeWM 实验总账（Cube seed 3072）",
        "",
        f"本报告基于 **{COMPLETE_CELL_COUNT} 个已核验正式 O50 单元**，但主结果只展示每个训练配置的最终 E10 checkpoint，避免把逐 epoch 诊断结果与正式横向比较混在一起。每格均为同一组 50 个 start-goal pair；训练 seed=3072，planning seed=42。模型均不训练 Actor。",
        "",
        "## 一句话结论",
        "",
        f"- **按原先固定的主评分列 F+G，描述性领先配置为 {f_plus_g_winner}。**",
        f"- **所有固定 E10 单格的最高结果为 {overall_winner}。**",
        f"- **按四版本、24 个训练配置的固定 E10 均值，描述性领先测试评分为 {best_score_names}（{analysis.best_mean_percent:.1f}%）。** 单 seed 下不把它表述为统计稳健最优。",
        f"- V1→V2 联合微调后，F-only 均值由 {v1_f:.1f}% 变为 {v2_f:.1f}%（{_signed_pp(v2_f - v1_f)}），F+G 由 {v1_tail:.1f}% 变为 {v2_tail:.1f}%（{_signed_pp(v2_tail - v1_tail)}）；这首先提示 world-model/control representation 变化，而不只是 G 的读出形式。",
        "",
        "## 完整全账伴随文件",
        "",
    ]
    lines += _markdown_table(
        ("文件", "保留内容", "路径", "SHA-256"),
        (
            (
                "CSV scalar ledger",
                f"{ledger_evidence.cell_count} 个 O50 单元",
                f"`{ledger_evidence.csv_path}`",
                f"`{ledger_evidence.csv_sha256}`",
            ),
            (
                "JSON reconciliation ledger",
                f"{ledger_evidence.cell_count} 格 × {EPISODES} outcomes = {ledger_evidence.outcome_count:,}",
                f"`{ledger_evidence.json_path}`",
                f"`{ledger_evidence.json_sha256}`",
            ),
        ),
    )
    lines += [
        "",
        f"主表只使用固定 E10；全部 {ledger_evidence.cell_count} 格和 {ledger_evidence.outcome_count:,} 个逐-pair 布尔结果仍由上述伴随文件完整保留。",
        "",
        "## 结果覆盖与版本定义",
        "",
    ]
    coverage = (
        ("Legacy", "7", "E10", "F / G(C) / combined", "21"),
        ("V0 raw action", "6", "E10", "all five scores", "30"),
        ("V1 action encoder", "6", "E10", "all five scores", "30"),
        ("V2 joint fine-tune", "6", "E3-E10", "3 original + E10 first/Mean", "156"),
        ("V2-EMA-SG", "6", "E3-E10", "all five scores", "240"),
    )
    lines += _markdown_table(
        ("版本/家族", "方法数", "Checkpoint", "评分覆盖", "O50 格数"), coverage
    )
    lines += [
        "",
        "## 方法、网络和训练 loss",
        "",
        "旧结构消融比较 Successor/critic head 与 LeWM predictor 的连接方式：Serial Decoupled、Serial Coupled、Hybrid、Parallel Real、Goal Hybrid、Imaginary Hybrid、Direct Goal Critic Hybrid。其总目标均为 `L_LeWM + α_u L_TD`，区别在 real/predicted 支路、是否让 TD 梯度进入 LeWM、是否使用 goal projection/imaginary bootstrap/direct scalar critic。旧版阶段性页面不再置于新版决策视图之前；来源文档在历史附录中明确列出。",
        "",
        "C–G3 家族共享同一个 TD-JEPA predictor `G`。V0 输入归一化 raw action；V1 改用冻结的 LeWM Action Encoder；V2 联合微调 LeWM/Action Encoder/G；V2-EMA-SG 进一步用 EMA world model、EMA action encoder 与 EMA G 构造完全 stop-gradient target。",
        "",
        "基础 target 与总目标：",
        "",
        r"$$Y_t=\operatorname{sg}[\bar z_{t+1}+\gamma(1-d_t)G_{\bar\phi}(\bar z_{t+1},\bar e_{t+1},m_t)],\quad \gamma=0.95,$$",
        "",
        r"$$L_{total}=L_{pred}+0.09L_{SIGReg}+\rho(u)(L_{method}^{real}+L_{method}^{pred}).$$",
        "",
        "其中分支 `b∈{real,pred}`；逐样本基础 TD 残差为 `l_i^b=||G_φ(s_i^b,e_i,m_i)-Y_i||²`，`m_i` 是 goal/random task vector，`ρ(u)` 是 TD warm-up 权重。C 额外训练 goal scalar residual；D/F/G1/G2/G3 只用 stop-gradient 的优势信号重加权 `l_i^b`。",
        "",
    ]
    lines += _markdown_table(
        ("方法", "特殊训练信号", "支路 loss", "作用"),
        tuple(
            tuple(str(value) for value in row) for row in v2_report.METHOD_FORMULA_ROWS
        ),
    )
    lines += [
        "",
        "## 五种推理评分",
        "",
        "- `F-only`：五步 LeWM rollout 的 terminal goal latent distance。",
        "- `G-only`：只用 `-Q_G(z_0,A_1,g)`，horizon=1。",
        "- `F+G tail`：F rollout cost 加 final successor tail。",
        "- `F + first-Q`：`||z_hat_5-z_g||² - 0.25 Q_G(z_0,A_1,g)`。",
        "- `Mean-Q rollout`：`-(1/5) Σ_k Q_G(z^F_{k-1},A_k,g)`；F 只生成 imagined states。",
        "",
        "## Legacy 7 方法：完整 21 格",
        "",
    ]
    lines += _markdown_table(
        ("方法", "F-only", "G/C-only", "Combined"), _legacy_rows(cells)
    )
    lines += [
        "",
        "## C–G3 固定 E10 主结果矩阵",
        "",
        "横向读每一行，可以直接比较同一个训练方法最适合哪一种评分；纵向读每一列，可以比较固定评分下哪个训练方法最好。**粗体**是该行最佳评分，`★` 是该列全局最佳训练配置（并列全部标记）。五种评分在四个版本的 24 个训练配置上均有正式结果。",
        "",
    ]
    lines += _markdown_table(
        ("版本", "训练方法", "F-only", "G-only", "F+G tail", "First-Q", "Mean-Q"),
        _fixed_master_markdown_rows(cells),
    )
    lines += [
        "",
        f"**逐列赢家：** {_column_winner_summary(analysis)}。",
        "",
        "## 训练 / validation loss 证据",
        "",
        "训练总 loss 含不同辅助项，绝对数值不能直接给 C–G3 排名，只用于判断各自是否收敛。Legacy 与 V1 曲线保留在历史来源文档；V0、V2、V2-EMA 的逐 epoch 数值和全部 E3–E10 O50 轨迹继续保留在总账 artifacts 中，但不再塞进主结果表。",
        "",
        "## 最佳训练方法与最佳测试评分",
        "",
        "### 固定 E10 四版本均值",
        "",
    ]
    lines += _markdown_table(
        ("训练版本", "F-only", "G-only", "F+G", "First-Q", "Mean-Q", "五评分均值"),
        _fixed_version_mean_rows(cells),
    )
    lines += [
        "",
        "### 五种评分的四版本汇总",
        "",
    ]
    lines += _markdown_table(
        ("评分方式", "覆盖版本", "固定格数", "四版本均值", "最高固定单格"),
        _fixed_score_mean_rows(cells),
    )
    lines += [
        "",
        "### 1. 哪个训练方法最好",
        "",
        f"不存在脱离测试评分的唯一训练赢家。按原研究固定的 F+G 主列，领先配置为 **{f_plus_g_winner}**；若按全部五种评分寻找最高单格，则为 **{overall_winner}**。在 V2-EMA E10 内，五评分均值最高的训练变体为 **{ema_best_variants}（{ema_best_mean:.1f}%）**。这些都是描述性单 seed 结果。",
        "",
        "### 2. 哪个测试方法最好",
        "",
        f"V2-EMA E10 六法均值为：{ema_mode_summary}。跨 V0/V1/V2/V2-EMA 的固定 E10，**{best_score_names}** 的 24 配置均值最高（{analysis.best_mean_percent:.1f}%），因此它是当前描述性默认主测试方式。Mean-Q 的最高固定配置为 {mean_q_winner}。",
        "",
        "### 3. 原因分析",
        "",
        f"- V0→V1 后 G-only 均值从 {analysis.version_mode_means[('v0', 'g_only')]:.1f}% 到 {analysis.version_mode_means[('v1', 'g_only')]:.1f}%，First-Q 从 {analysis.version_mode_means[('v0', 'f_plus_g_first')]:.1f}% 到 {analysis.version_mode_means[('v1', 'f_plus_g_first')]:.1f}%；这与共享 Action Encoder 改善 G 读出一致。",
        "- V1→V2 后 F-only 与 F+G 同时大幅下降，与 TD 梯度进入 online LeWM/Action Encoder 后产生 latent/control representation drift 的假设一致；单 seed 不能证明因果，问题也不能只归因于 critic。",
        f"- 当前均值领先读出是 {best_score_names}；不同读出暴露于真实状态、imagined states 与 G 尺度的程度不同，OOD/rollout 误差仍是需要用独立 dev pairs 验证的解释。",
        f"- Mean-Q 在 24 个固定训练配置上的均值为 {analysis.mode_means['g_only_f_rollout_mean']:.1f}%，最高格为 {mean_q_winner}；它是否应进入主评测由这一完整覆盖结果决定，不再按旧的 V2-only 结论处理。",
        "",
        "## 负面结果与下一轮目标",
        "",
    ]
    lines += _markdown_table(
        ("发现", "证据", "含义/下一步"),
        (
            (
                "V2 world model 退化",
                f"V1→V2 F-only {v1_f:.1f}%→{v2_f:.1f}%",
                "先恢复 F，再谈 G 增益",
            ),
            (
                "EMA 未根治",
                "V2→EMA 五评分变化："
                + "，".join(
                    f"{MODE_LABELS[mode]} {_signed_pp(analysis.version_mode_means[('v2_ema_sg', mode)] - analysis.version_mode_means[('v2', mode)])}"
                    for mode in ALL_CONTROLLED_MODES
                ),
                "冻结或低 LR 微调 encoder/world",
            ),
            (
                "tail 干扰",
                f"{tail_harm_count}/24 个配置的 F+G 低于 F-only",
                f"以 {best_score_names} 为默认；G readout 加 gate/校准",
            ),
            (
                "Mean-Q 完整覆盖",
                f"24 配置均值 {analysis.mode_means['g_only_f_rollout_mean']:.1f}%；{mean_q_winner}",
                "按完整四版本结果决定主评测或消融地位",
            ),
            (
                "checkpoint 选择偏差",
                "同一 O50 上看 E3–E10 再取最大",
                "dev pairs 选 epoch/alpha，正式 O50 只跑锁定配置",
            ),
            (
                "单 seed",
                "全部训练 seed=3072",
                "至少 3 个训练 seeds，保存逐-pair outcome",
            ),
        ),
    )
    lines += [
        "",
        "下一轮优先目标：",
        "",
        f"1. 把联合模型固定 E10 的 F-only 六法均值从 {analysis.version_mode_means[('v2_ema_sg', 'f_only')]:.1f}% 恢复到至少 V1 的 {v1_f:.1f}%。",
        f"2. 预注册 `{_compact_join(analysis.overall_winners)}` 与 `{_compact_join(analysis.mode_winners['f_plus_g'])} + F+G tail` 作为主基线；正式 O50 不再事后选 epoch。",
        "3. 若继续 joint training，先冻结 encoder/world 或给 TD 极低学习率，再分阶段解冻；增加对 V1 latent/prediction 的 anchor，并限制 TD 梯度进入 F。",
        "4. 在独立 dev pair set 上选择 α、epoch 与 Q 校准；对 CEM 候选/imagined-state 分布加入 conservative/calibration 训练，抑制 OOD 高估。",
        "5. 至少 3 个、最好 5 个 training seeds；用 paired bootstrap/McNemar 分析固定配置。",
        "",
        "## 审计边界",
        "",
        f"- {COMPLETE_CELL_COUNT}/{COMPLETE_CELL_COUNT} 格共享 episode-selection 文件 SHA-256 `{SELECTION_SHA256}` 与 action normalization SHA-256 `{ACTION_NORMALIZATION_SHA256}`。",
        f"- fixed 新评分 launcher 另有 valid-row-ranks SHA-256 `{FIXED_SELECTION_RANKS_SHA256}`；它是规范化索引哈希，不是 episode-selection 文件哈希，二者不能混写。",
        f"- 每格成功数都由 50 个布尔 outcome 重算；CSV `{ledger_evidence.csv_path}`（SHA-256 `{ledger_evidence.csv_sha256}`）与 JSON `{ledger_evidence.json_path}`（SHA-256 `{ledger_evidence.json_sha256}`）共同保留 {ledger_evidence.cell_count} 格 / {ledger_evidence.outcome_count:,} 个 outcomes。",
        "- EMA E3 的 G1/F+G 与 G2/F-only 使用隔离 retry attempt_02；原失败调度证据保留，不把失败单元伪装成原调度成功。",
        "- 固定 E10 Mean-Q 覆盖 V0/V1/V2/V2-EMA × C/D/F/G1/G2/G3，共 24 格；主结果表不允许缺格或使用占位符。",
        "- 主结果表只展示 E10；E3–E10 全轨迹仍保存在 `all_o50_results.csv`，没有因版式精简而删除。",
        "- 只有一个 training seed；所有跨版本结果都是描述性结构消融，不声称多 seed 总体最优或统计显著。",
        "",
    ]
    return "\n".join(lines)


def _add_heading(
    document: Any, text: str, *, level: int = 1, page_break: bool = False
) -> Any:
    return v2_report._add_heading(document, text, level=level, page_break=page_break)


def _add_body(
    document: Any, text: str, *, color: str = "111827", bold: bool = False
) -> Any:
    return v2_report._add_body(document, text, color=color, bold=bold)


def _add_table(
    document: Any,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[int],
) -> Any:
    return v2_report._v1._add_table(document, headers=headers, rows=rows, widths=widths)


def _add_fixed_master_table(document: Any, cells: Sequence[ResultCell]) -> Any:
    """Add the single E10 decision matrix with row/column winner highlighting."""

    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    display_rows = _fixed_master_rows(cells)
    count_rows = _fixed_master_counts(cells)
    column_maxima = tuple(
        max(row[index] for row in count_rows)
        for index in range(len(ALL_CONTROLLED_MODES))
    )
    table = _add_table(
        document,
        ("Version", "Method", "F-only", "G-only", "F+G tail", "First-Q", "Mean-Q"),
        display_rows,
        (1200, 900, 2460, 2460, 2460, 2460, 2460),
    )
    for column, cell in enumerate(table.rows[0].cells):
        cell.paragraphs[0].alignment = (
            WD_ALIGN_PARAGRAPH.LEFT if column < 2 else WD_ALIGN_PARAGRAPH.CENTER
        )
    for row in table.rows:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
            properties = cell._tc.get_or_add_tcPr()
            margins = properties.find(qn("w:tcMar"))
            if margins is None:
                margins = OxmlElement("w:tcMar")
                properties.append(margins)
            for side in ("top", "bottom"):
                element = margins.find(qn(f"w:{side}"))
                if element is None:
                    element = OxmlElement(f"w:{side}")
                    margins.append(element)
                element.set(qn("w:w"), "55")
                element.set(qn("w:type"), "dxa")
    for row_index, (row, counts) in enumerate(zip(table.rows[1:], count_rows), start=0):
        row_maximum = max(value for value in counts if value is not None)
        band_fill = "F8FAFC" if (row_index // len(VARIANTS)) % 2 == 0 else "EEF4F8"
        for column in (0, 1):
            v2_report._v1._shade_cell(row.cells[column], band_fill)
            row.cells[column].paragraphs[0].runs[0].bold = True
        for score_index, count in enumerate(counts):
            cell = row.cells[score_index + 2]
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.runs[0]
            run.font.size = Pt(8.5)
            row_best = count == row_maximum
            column_best = count == column_maxima[score_index]
            if row_best and column_best:
                v2_report._v1._shade_cell(cell, "B7DEE8")
            elif column_best:
                v2_report._v1._shade_cell(cell, "DDEBF7")
            elif row_best:
                v2_report._v1._shade_cell(cell, "FFF2CC")
            run.bold = row_best or column_best
    return table


def _configure_primary_document(document: Any) -> None:
    """Configure a standalone landscape report with no stale pages in front."""

    from docx.enum.section import WD_ORIENT
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.5)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.25)

    if "Report Kicker" not in document.styles:
        kicker = document.styles.add_style("Report Kicker", WD_STYLE_TYPE.PARAGRAPH)
    else:
        kicker = document.styles["Report Kicker"]
    kicker.font.name = "Arial Unicode MS"
    kicker._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial Unicode MS")
    kicker._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial Unicode MS")
    kicker._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
    kicker.font.size = Pt(9.5)
    kicker.font.bold = True
    kicker.font.color.rgb = RGBColor.from_string("5C6975")
    kicker.paragraph_format.space_before = Pt(4)
    kicker.paragraph_format.space_after = Pt(4)

    v2_report._set_header_text(
        section,
        f"Results TD complete ledger · Cube O50 · {COMPLETE_CELL_COUNT} verified cells",
    )
    for footer in (section.footer, section.even_page_footer):
        footer.is_linked_to_previous = False
        paragraph = footer.paragraphs[0]
        paragraph.text = ""
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer_run = paragraph.add_run("Complete fixed-E10 decision report · Page ")
        v2_report._v1._set_run_font(footer_run, size=8.5, color="6B7280")
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        paragraph._p.append(field)

    core = document.core_properties
    core.title = "Results TD Complete Experiment Ledger"
    core.subject = (
        f"{COMPLETE_CELL_COUNT} verified O50 cells and fixed E10 five-score comparison"
    )
    core.keywords = (
        "TD-JEPA, Actor-Free TD-LeWM, fixed E10, five scores, "
        f"{COMPLETE_CELL_COUNT} cells, {COMPLETE_OUTCOME_COUNT} outcomes"
    )


def build_docx(
    cells: Sequence[ResultCell],
    ledger_evidence: LedgerEvidence,
    *,
    v0_training_chart: bytes,
    v2_training_chart: bytes,
    training_chart: bytes,
) -> bytes:
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required from the workspace runtime."
        ) from exc

    analysis = _fixed_analysis(cells)
    f_plus_g_winner = _winner_result(analysis, "f_plus_g")
    overall_winner = (
        f"{_compact_join(analysis.overall_winners)}: {analysis.overall_best_count}/50 "
        f"({analysis.overall_best_count * 2}%)"
    )
    best_score_names = _joined(
        tuple(MODE_LABELS[mode] for mode in analysis.best_mean_modes)
    )
    ema_best_variants = _compact_join(
        tuple(variant.upper() for variant in analysis.ema_best_variants)
    )
    ema_best_mean = analysis.ema_variant_means[analysis.ema_best_variants[0]]
    v1_f = analysis.version_mode_means[("v1", "f_only")]
    v2_f = analysis.version_mode_means[("v2", "f_only")]
    mean_q_winner = _winner_result(analysis, "g_only_f_rollout_mean")
    tail_harm_count = sum(
        _percent(
            _find(cells, version=version, variant=variant, epoch=10, mode="f_plus_g")
        )
        < _percent(
            _find(cells, version=version, variant=variant, epoch=10, mode="f_only")
        )
        for version in VERSIONS
        for variant in VARIANTS
    )
    document = Document()
    _configure_primary_document(document)

    kicker = document.add_paragraph(style="Report Kicker")
    kicker.add_run("RESULTS TD / COMPLETE EXPERIMENT LEDGER")
    v2_report._font_paragraph(kicker, size=9.5, color="5C6975")
    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("Fixed E10 decision matrix across methods and scores")
    v2_report._v1._set_run_font(run, size=24, color="0B2545", bold=True)
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(10)
    run = subtitle.add_run(
        f"Cube · seed 3072 · {COMPLETE_CELL_COUNT} O50 cells · legacy + V0 + V1 + V2 + V2-EMA-SG"
    )
    v2_report._v1._set_run_font(run, size=12, color="4B5563")
    _add_body(
        document,
        f"This decision view is backed by the complete {COMPLETE_CELL_COUNT}-cell audit, but its main "
        "result table shows only the final E10 checkpoint. Every displayed result "
        f"uses the same 50 start-goal pairs; the companion ledgers retain all "
        f"{COMPLETE_OUTCOME_COUNT:,} Boolean outcomes. All training uses one seed, so "
        "no statistical-significance claim is made.",
        color="7A5A00",
        bold=True,
    )

    _add_heading(document, "Decision summary")
    _add_table(
        document,
        ("Question", "Answer", "Evidence", "Decision"),
        (
            (
                "Best on prespecified F+G column",
                _compact_join(analysis.mode_winners["f_plus_g"]),
                f"{analysis.mode_best_counts['f_plus_g']}/50 "
                f"({analysis.mode_best_counts['f_plus_g'] * 2}%)",
                "Keep as primary baseline",
            ),
            (
                "Best fixed result overall",
                _compact_join(analysis.overall_winners),
                f"{analysis.overall_best_count}/50 ({analysis.overall_best_count * 2}%)",
                "Primary fixed-E10 baseline",
            ),
            (
                "Best V2-EMA training variant",
                ema_best_variants,
                f"E10 five-score mean {ema_best_mean:.1f}%",
                "Retain this variant if joint training continues",
            ),
            (
                "Descriptive default test score",
                best_score_names,
                f"24-configuration mean {analysis.best_mean_percent:.1f}%",
                "Default readout candidate; confirm on dev pairs",
            ),
            (
                "Leading failure hypothesis",
                "Joint world-model drift",
                f"V1→V2 F-only {v1_f:.1f}%→{v2_f:.1f}%",
                f"Restore F toward the V1 mean ({v1_f:.1f}%)",
            ),
        ),
        (3300, 3400, 3600, 4100),
    )

    _add_heading(document, "Complete companion ledgers")
    _add_table(
        document,
        ("Artifact", "Coverage", "Repository-relative path", "SHA-256"),
        (
            (
                "CSV scalar ledger",
                f"{ledger_evidence.cell_count} O50 cells",
                ledger_evidence.csv_path,
                ledger_evidence.csv_sha256,
            ),
            (
                "JSON reconciliation ledger",
                f"{ledger_evidence.cell_count} x {EPISODES} = "
                f"{ledger_evidence.outcome_count:,} outcomes",
                ledger_evidence.json_path,
                ledger_evidence.json_sha256,
            ),
        ),
        (2600, 2900, 4500, 4400),
    )
    _add_body(
        document,
        f"The main matrix is fixed E10 only. The two fingerprinted files above retain "
        f"all {ledger_evidence.cell_count} scalar cells and all "
        f"{ledger_evidence.outcome_count:,} per-pair Boolean outcomes.",
        bold=True,
    )

    _add_heading(
        document,
        "C-G3 fixed E10 master matrix: 24 training configurations x five scores",
        page_break=True,
    )
    _add_body(
        document,
        "Read across a row to compare five evaluation scores for one training method; "
        "read down a column to compare 24 training configurations under one score. "
        "Gold marks a row winner, blue marks a column winner, and teal marks both. "
        "Ties are all highlighted in the matrix.",
        bold=True,
    )
    _add_fixed_master_table(document, cells)
    _add_body(
        document,
        "All five scores have formal E10 results for every one of the 24 training "
        f"configurations. Column winners: {_column_winner_summary(analysis)}.",
        color="5C6975",
    )

    _add_heading(
        document,
        "Best training method and evaluation score",
        page_break=True,
    )
    _add_body(
        document,
        "There is no evaluation-independent training winner. The prespecified F+G "
        f"leader is {f_plus_g_winner}; the highest fixed cell is {overall_winner}. "
        f"Across all 24 configurations, {best_score_names} has the largest descriptive "
        f"mean ({analysis.best_mean_percent:.1f}%). These are single-seed comparisons.",
        color="7A5A00",
        bold=True,
    )
    _add_heading(document, "Fixed E10 means by training version", level=2)
    _add_table(
        document,
        (
            "Training version",
            "F-only",
            "G-only",
            "F+G",
            "First-Q",
            "Mean-Q",
            "Five-score mean",
        ),
        _fixed_version_mean_rows(cells),
        (3000, 1850, 1850, 1850, 1850, 1850, 2150),
    )
    _add_heading(document, "Fixed E10 means by evaluation score", level=2)
    _add_table(
        document,
        ("Score", "Versions", "Cells", "Four-version mean", "Best fixed cell"),
        _fixed_score_mean_rows(cells),
        (2500, 3000, 1500, 2400, 5000),
    )

    _add_heading(document, "Causes and next objectives", page_break=True)
    _add_table(
        document,
        ("Finding", "Evidence", "Interpretation", "Next objective"),
        (
            (
                "Action embedding helps G",
                f"V0→V1 G-only {analysis.version_mode_means[('v0', 'g_only')]:.1f}→"
                f"{analysis.version_mode_means[('v1', 'g_only')]:.1f}%; First-Q "
                f"{analysis.version_mode_means[('v0', 'f_plus_g_first')]:.1f}→"
                f"{analysis.version_mode_means[('v1', 'f_plus_g_first')]:.1f}%",
                "Semantic action representation improves critic readout",
                "Keep shared V1 Action Encoder",
            ),
            (
                "Joint tuning damages F",
                f"V1→V2 F-only {v1_f:.1f}→{v2_f:.1f}%",
                "Consistent with TD-gradient representation drift",
                f"Restore F-only mean to >={v1_f:.1f}%",
            ),
            (
                "EMA is insufficient",
                "Five-score mean delta "
                f"{_signed_pp(sum(analysis.version_mode_means[('v2_ema_sg', mode)] - analysis.version_mode_means[('v2', mode)] for mode in ALL_CONTROLLED_MODES) / len(ALL_CONTROLLED_MODES))}",
                "Target stabilization does not undo online drift",
                "Freeze/low-LR then staged unfreeze",
            ),
            (
                f"{best_score_names} leads descriptively",
                f"24-configuration mean {analysis.best_mean_percent:.1f}%",
                "Readout exposure to rollout OOD remains a testable hypothesis",
                "Use as default candidate; tune score parameters on dev",
            ),
            (
                "Tail interference",
                f"F+G is below F-only in {tail_harm_count}/24 fixed configurations",
                "Final imagined-state G can add OOD or scale error",
                "Gate/calibrate the tail on independent dev pairs",
            ),
            (
                "Mean-Q complete coverage",
                f"Mean {analysis.mode_means['g_only_f_rollout_mean']:.1f}%; {mean_q_winner}",
                "All four versions now contribute equally to the comparison",
                "Use the complete result to decide primary versus ablation status",
            ),
            (
                "Single-seed boundary",
                "All training seed 3072",
                "No population-level claim",
                ">=3 seeds + paired statistics",
            ),
        ),
        (2800, 4100, 4200, 3300),
    )
    _add_body(
        document,
        "Recommended next experiment: freeze the V1 encoder/world model, compare "
        f"{_compact_join(analysis.overall_winners)} against "
        f"{_compact_join(analysis.mode_winners['f_plus_g'])} + F+G tail on independent development pairs, "
        "pre-register epoch/alpha, then run at least three training seeds. If joint "
        "training is retained, use a much smaller TD learning rate, latent/prediction "
        "anchors to V1 and conservative calibration on CEM candidate actions.",
        bold=True,
    )

    _add_heading(document, "Coverage map", page_break=True)
    _add_table(
        document,
        ("Family/version", "Methods", "Epochs", "Scores", "Cells"),
        (
            ("Legacy structures", "7", "E10", "F / G(C) / combined", "21"),
            ("V0 raw action", "6", "E10", "All five", "30"),
            ("V1 action encoder", "6", "E10", "All five", "30"),
            ("V2 joint", "6", "E3-E10", "3 original + E10 First/Mean", "156"),
            ("V2-EMA-SG", "6", "E3-E10", "All five", "240"),
            ("TOTAL", "-", "-", "Same O50 selection", str(COMPLETE_CELL_COUNT)),
        ),
        (3600, 1800, 2200, 4900, 1900),
    )
    _add_body(
        document,
        "The complete E3-E10 trajectories remain in the audited companion ledgers; "
        "the decision matrix in this document deliberately uses one fixed E10 checkpoint.",
    )

    _add_heading(document, "V2 / V2-EMA network, target and loss", page_break=True)
    _add_body(
        document,
        "V2 jointly updates online LeWM, the shared 25D-to-192D Action Encoder and "
        "G. V2-EMA-SG uses an EMA world model, EMA Action Encoder and EMA G to form "
        "the fully stopped next-frame target. Neither version trains an Actor or reward model.",
    )
    v2_report._add_formula(
        document,
        "TD target",
        "Y_t = sg[zbar_(t+1) + gamma(1-d_t) Gbar(zbar_(t+1),ebar_(t+1),m_t)]",
        "zbar comes from the real next frame; ebar comes from the known dataset next action. gamma=0.95.",
    )
    v2_report._add_formula(
        document,
        "Total",
        "L_total = L_pred + 0.09 L_SIGReg + rho(u)(L_method^real + L_method^pred)",
        "The predicted branch is differentiable to online LeWM/Action Encoder; EMA modules only form targets.",
    )
    _add_body(
        document,
        "For branch b in {real,pred}, the per-sample base residual is "
        "l_i^b = ||G_phi(s_i^b,e_i,m_i)-Y_i||^2. Here m_i is the goal/random "
        "task vector and rho(u) is the TD warm-up. C adds a scalar goal residual; "
        "D/F/G1/G2/G3 use stopped advantage signals to reweight l_i^b.",
    )
    _add_table(
        document,
        ("Method", "Special signal", "Per-branch loss", "Role"),
        tuple(
            tuple(str(value) for value in row) for row in v2_report.METHOD_FORMULA_ROWS
        ),
        (1200, 4300, 4300, 4600),
    )

    _add_heading(document, "Five inference scores", page_break=True)
    _add_table(
        document,
        ("Score", "Planner signal", "G location", "Main risk"),
        (
            (
                "F-only",
                "Five-step terminal latent distance",
                "None",
                "World-model drift",
            ),
            (
                "G-only",
                "-Q_G(z0,A1,g), H=1",
                "Real z0 / first action",
                "Weak without F",
            ),
            (
                "F+G tail",
                "F prefix + final G tail",
                "Final imagined state",
                "Tail OOD / interference",
            ),
            (
                "First-Q",
                "F terminal cost - 0.25 Q_G(z0,A1,g)",
                "Real z0 / first action",
                "Q scale / alpha",
            ),
            (
                "Mean-Q",
                "-(1/5) sum Q over F rollout",
                "Five real/imagined predecessors",
                "Accumulated drift / OOD",
            ),
        ),
        (2500, 5000, 3700, 3200),
    )

    _add_heading(document, "Training / validation trajectories", page_break=True)
    _add_body(
        document,
        "Training totals are method-specific and cannot be ranked by absolute height. "
        "Use them only as within-method convergence diagnostics. The following charts "
        "show V0, V2 and V2-EMA; legacy and V1 curves remain in the historical source "
        "report. All numeric rows and E3-E10 O50 trajectories remain in the companion ledgers.",
    )
    v2_report._add_picture(
        document,
        v0_training_chart,
        title="Figure 1. V0 training and validation total loss (E1-E10)",
        description="Raw-action C, D, F, G1, G2 and G3. Log scale; compare convergence within a method, not absolute objective height across methods.",
    )
    v2_report._add_picture(
        document,
        v2_training_chart,
        title="Figure 2. V2 joint training and validation total loss (E1-E10)",
        description="Joint LeWM/Action Encoder/G fine-tuning trajectories for all six methods.",
    )
    v2_report._add_picture(
        document,
        training_chart,
        title="Figure 3. V2-EMA-SG training and validation loss (E1-E10)",
        description="Training and validation total-loss trajectories for C, D, F, G1, G2 and G3.",
    )

    _add_heading(document, "Audit boundary", page_break=True)
    _add_table(
        document,
        ("Audit field", "Locked value"),
        (
            (
                "Verified O50 cells",
                f"{COMPLETE_CELL_COUNT} / {COMPLETE_CELL_COUNT}",
            ),
            ("Scalar CSV path", ledger_evidence.csv_path),
            ("Scalar CSV SHA-256", ledger_evidence.csv_sha256),
            ("Reconciliation JSON path", ledger_evidence.json_path),
            ("Reconciliation JSON SHA-256", ledger_evidence.json_sha256),
            (
                "Per-pair Boolean outcomes",
                f"{ledger_evidence.outcome_count:,} / {COMPLETE_OUTCOME_COUNT:,}",
            ),
            ("Fixed E10 Mean-Q cells", "24 / 24"),
            ("Episode selection", SELECTION_SHA256),
            ("Fixed valid-row-ranks hash", FIXED_SELECTION_RANKS_SHA256),
            ("Action normalization", ACTION_NORMALIZATION_SHA256),
            ("Training / planning seed", "3072 / 42"),
            ("EMA E3 retry", "G1/F+G and G2/F-only use isolated attempt_02"),
            (
                "Claim boundary",
                "One training seed; the decision matrix is fixed E10 only",
            ),
        ),
        (3500, 10900),
    )

    _add_heading(
        document,
        "Historical appendix — superseded locked source report",
        page_break=True,
    )
    _add_body(
        document,
        "The earlier locked V0/V1 report is retained as a historical source at "
        f"{_display_path(Path(DEFAULT_BASE_DOCUMENT))}. Its stage-specific three-score "
        "and 18-run statements are not embedded ahead of this report and do not override "
        "the complete 24-by-five fixed-E10 decision matrix.",
        color="7A5A00",
        bold=True,
    )
    _add_body(
        document,
        "Use the historical source for legacy/V0/V1 architecture detail and training "
        "curves only. Use this document and its fingerprinted companion ledgers for all "
        "current cross-version result comparisons.",
    )

    stream = io.BytesIO()
    document.save(stream)
    payload = stream.getvalue()
    if not payload.startswith(b"PK"):
        raise RuntimeError("python-docx did not produce OOXML.")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument(
        "--reconciliation-ledger", default=str(DEFAULT_RECONCILIATION_LEDGER)
    )
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    parser.add_argument("--docx-output", default=str(DEFAULT_DOCX_OUTPUT))
    parser.add_argument("--root-docx-output", default=str(DEFAULT_ROOT_DOCX_OUTPUT))
    parser.add_argument("--chart-dir", default=str(DEFAULT_CHART_DIR))
    parser.add_argument("--v0-training-csv", default=str(DEFAULT_V0_TRAINING_CSV))
    parser.add_argument("--v2-training-csv", default=str(DEFAULT_V2_TRAINING_CSV))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cells = load_complete_ledger(args.ledger)
    ledger_evidence = load_ledger_evidence(args.ledger, args.reconciliation_ledger)
    if args.validate_only:
        print(
            "PASS: complete Results TD ledger contains "
            f"{COMPLETE_CELL_COUNT} verified O50 cells and "
            f"{COMPLETE_OUTCOME_COUNT} reconciled outcomes."
        )
        return 0
    base_inputs = v2_report.load_validated_report_inputs(
        original_summary_path=v2_report.DEFAULT_ORIGINAL_SUMMARY,
        training_loss_csv_path=v2_report.DEFAULT_TRAINING_LOSS_CSV,
        new_summary_path=v2_report.DEFAULT_NEW_SUMMARY,
        new_results_csv_path=v2_report.DEFAULT_NEW_RESULTS_CSV,
        fixed_results_csv_path=v2_report.DEFAULT_FIXED_RESULTS_CSV,
        base_document_path=DEFAULT_BASE_DOCUMENT,
    )
    v0_training_chart = build_training_chart_from_archive(
        args.v0_training_csv, title="V0 raw-action training and validation loss"
    )
    v2_training_chart = build_training_chart_from_archive(
        args.v2_training_csv, title="V2 joint training and validation loss"
    )
    training_chart = v2_report.build_training_loss_chart(base_inputs)
    score_chart = v2_report.build_new_score_chart(base_inputs)
    markdown = build_markdown(cells, ledger_evidence)
    document = build_docx(
        cells,
        ledger_evidence,
        v0_training_chart=v0_training_chart,
        v2_training_chart=v2_training_chart,
        training_chart=training_chart,
    )
    markdown_path = Path(args.markdown_output)
    docx_path = Path(args.docx_output)
    root_docx_path = Path(args.root_docx_output)
    chart_dir = Path(args.chart_dir)
    _atomic_write(markdown_path, markdown.encode("utf-8"))
    _atomic_write(docx_path, document)
    _atomic_write(root_docx_path, document)
    _atomic_write(chart_dir / "v2_ema_training_validation_loss.png", training_chart)
    _atomic_write(chart_dir / "v2_ema_new_score_trajectories.png", score_chart)
    _atomic_write(chart_dir / "v0_training_validation_loss.png", v0_training_chart)
    _atomic_write(chart_dir / "v2_training_validation_loss.png", v2_training_chart)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {docx_path}")
    print(f"Updated {root_docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
