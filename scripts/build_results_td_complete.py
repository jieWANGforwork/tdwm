#!/usr/bin/env python3
"""Build the complete Results TD ledger report from 465 audited O50 cells.

The existing Results TD reports were produced incrementally.  This builder is
the consolidation layer: it requires the exact legacy/V0/V1/V2/V2-EMA grid,
checks every identity and rate, then produces a standalone Markdown report and
a DOCX which keeps the established legacy/V0/V1 material and appends the full
V2/V2-EMA trajectories plus the cross-version decision analysis.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
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
TRAINING_SEED = 3072
PLANNING_SEED = 42
STEPS_PER_EPOCH = 12_796
SELECTION_SHA256 = v2_report.SHARED_EPISODE_SELECTION_SHA256
ACTION_NORMALIZATION_SHA256 = v2_report.ACTION_NORMALIZATION_SHA256
FIXED_SELECTION_RANKS_SHA256 = (
    "88c204c83daf2157334d4ce9ecf7f18dcd11f778fbb80c310e31f322bfe5aed7"
)
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
VERSIONS = ("v0", "v1", "v2", "v2_ema_sg")
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


def _sha(value: str, *, context: str, allow_empty: bool = False) -> str:
    value = value.strip()
    if allow_empty and value == "":
        return value
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CompleteResultsError(f"{context} must be a lowercase SHA-256.")
    return value


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
            for mode in ORIGINAL_MODES + ("f_plus_g_first",)
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
    if len(raw_rows) != 465:
        raise CompleteResultsError(
            f"Complete ledger must contain 465 cells, found {len(raw_rows)}."
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


def _fixed_rows(
    cells: Sequence[ResultCell], version: str
) -> tuple[tuple[str, ...], ...]:
    modes = ORIGINAL_MODES
    if version in ("v0", "v1"):
        modes += ("f_plus_g_first",)
    else:
        modes += NEW_MODES
    return tuple(
        (
            variant.upper(),
            *(
                _result(
                    _find(cells, version=version, variant=variant, epoch=10, mode=mode)
                )
                for mode in modes
            ),
        )
        for variant in VARIANTS
    )


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
        common = tuple(
            _epoch10_mode_mean(cells, version, mode)
            for mode in ORIGINAL_MODES + ("f_plus_g_first",)
        )
        mean_q = (
            f"{_epoch10_mode_mean(cells, version, 'g_only_f_rollout_mean'):.1f}%"
            if version in ("v2", "v2_ema_sg")
            else "—"
        )
        rows.append(
            (
                label,
                *(f"{value:.1f}%" for value in common),
                mean_q,
                f"{sum(common) / len(common):.1f}%",
            )
        )
    return tuple(rows)


def _fixed_score_mean_rows(
    cells: Sequence[ResultCell],
) -> tuple[tuple[str, ...], ...]:
    """Compare readouts only on versions for which each readout was evaluated."""

    rows = []
    for mode in ALL_CONTROLLED_MODES:
        versions = VERSIONS if mode != "g_only_f_rollout_mean" else ("v2", "v2_ema_sg")
        per_version = [_epoch10_mode_mean(cells, version, mode) for version in versions]
        fixed_cells = [
            cell
            for cell in cells
            if cell["version"] in versions
            and int(cell["epoch"]) == 10
            and cell["score_mode"] == mode
        ]
        best = max(int(cell["success_count"]) for cell in fixed_cells)
        winners = ", ".join(
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
                f"{winners}: {best}/50 ({best * 2}%)",
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


def build_markdown(cells: Sequence[ResultCell]) -> str:
    lines = [
        "# Results TD — 全部 Actor-Free TD-LeWM 实验总账（Cube seed 3072）",
        "",
        "本报告是统一总账，不是新增结果附录。它覆盖 **465 个正式 O50 单元**：旧 7 方法 21 格、V0 24 格、V1 24 格、V2 156 格、V2-EMA-SG 240 格。每格均为同一组 50 个 start-goal pair；训练 seed=3072，planning seed=42。模型均不训练 Actor。",
        "",
        "## 一句话结论",
        "",
        "- **按原先固定的主评分列 F+G，受控 C–G3 中描述性最好的训练方案是 V1-G3：27/50（54%）。**",
        "- **所有固定 checkpoint 的最高单格是 V1-C + first-Q：28/50（56%）。**",
        "- **当前描述性最好的默认测试评分是 first-Q。** 它保留 F 的五步 terminal goal cost，只用真实当前 latent 与第一动作读取 G，受 imagined-state 漂移较少；单 seed 下不把它表述为统计稳健最优。",
        "- V1→V2 联合微调后，F-only 均值由 46.0% 降到 26.0%，F+G 由 47.7% 降到 27.3%；EMA 只恢复约 1–3 pp，核心问题首先是 world-model/control representation 退化，而不只是 G 的读出形式。",
        "",
        "## 结果覆盖与版本定义",
        "",
    ]
    coverage = (
        ("Legacy", "7", "E10", "F / G(C) / combined", "21"),
        ("V0 raw action", "6", "E10", "F / G / F+G / first-Q", "24"),
        ("V1 action encoder", "6", "E10", "F / G / F+G / first-Q", "24"),
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
        "旧结构消融比较 Successor/critic head 与 LeWM predictor 的连接方式：Serial Decoupled、Serial Coupled、Hybrid、Parallel Real、Goal Hybrid、Imaginary Hybrid、Direct Goal Critic Hybrid。其总目标均为 `L_LeWM + α_u L_TD`，区别在 real/predicted 支路、是否让 TD 梯度进入 LeWM、是否使用 goal projection/imaginary bootstrap/direct scalar critic。完整逐方法网络、loss 与推理说明保留在本总账 DOCX 的前置锁定页。",
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
    for version, label in (
        ("v0", "V0 fixed E10：完整 24 格"),
        ("v1", "V1 fixed E10：完整 24 格"),
        ("v2", "V2 fixed E10：完整 30 格中的 5 评分"),
        ("v2_ema_sg", "V2-EMA fixed E10：完整 30 格中的 5 评分"),
    ):
        headers = ("方法", "F-only", "G-only", "F+G", "first-Q")
        if version in ("v2", "v2_ema_sg"):
            headers += ("Mean-Q",)
        lines += ["", f"## {label}", ""]
        lines += _markdown_table(headers, _fixed_rows(cells, version))

    lines += ["", "## V2 checkpoint 轨迹：原三评分 144 格", ""]
    for mode in ORIGINAL_MODES:
        lines += [f"### {MODE_LABELS[mode]}", ""]
        lines += _markdown_table(
            ("Epoch", "C", "D", "F", "G1", "G2", "G3"),
            _trajectory_rows(cells, "v2", mode),
        )
        lines.append("")

    lines += ["## V2-EMA checkpoint 轨迹：五评分 240 格", ""]
    for mode in ALL_CONTROLLED_MODES:
        lines += [f"### {MODE_LABELS[mode]}", ""]
        lines += _markdown_table(
            ("Epoch", "C", "D", "F", "G1", "G2", "G3"),
            _trajectory_rows(cells, "v2_ema_sg", mode),
        )
        lines.append("")

    lines += [
        "## 训练 / validation loss 证据",
        "",
        "训练总 loss 含不同辅助项，绝对数值不能直接给 C–G3 排名，只用于判断各自是否收敛。Legacy 与 V1 曲线保留在锁定的前置报告；V0、V2 的 60 行逐 epoch 数值分别存于 `v0_training_loss_curves.csv`、`v2_training_loss_curves.csv`；V2-EMA 的完整曲线与图也保留在总账 artifacts 中。",
        "",
        "## 最佳训练方法与最佳测试评分",
        "",
        "### 固定 E10 的版本 × 评分均值",
        "",
    ]
    lines += _markdown_table(
        ("训练版本", "F-only", "G-only", "F+G", "first-Q", "Mean-Q", "前四列均值"),
        _fixed_version_mean_rows(cells),
    )
    lines += [
        "",
        "### 测试评分的固定结果汇总",
        "",
    ]
    lines += _markdown_table(
        ("评分方式", "覆盖版本", "固定格数", "版本均值", "最高固定单格"),
        _fixed_score_mean_rows(cells),
    )
    lines += [
        "",
        "### V2-EMA 内训练变体的五评分均值",
        "",
    ]
    lines += _markdown_table(
        ("训练变体", "E10五评分均值", "该变体最佳评分", "最佳率"),
        _ema_variant_mean_rows(cells),
    )
    lines += [
        "",
        "### 1. 哪个训练方法最好",
        "",
        "不存在脱离测试评分的唯一训练赢家。按原研究固定的 F+G 主列，受控 C–G3 中 **V1-G3=54%** 描述性最好；旧结构家族的 **Hybrid=54%** 与其同率，但两者不是同构版本。若统一使用新增 first-Q，最高固定单格变成 **V1-C=56%**。在 V2-EMA E10 内，**训练变体 F（Same-Future Advantage）** 的五评分均值为 38.4%，在六个 EMA 训练变体中最高，因此若只继续 V2-EMA 家族，优先保留训练变体 F。",
        "",
        "### 2. 哪个测试方法最好",
        "",
        "V2-EMA E10 六法均值：F-only 27.0%、G-only 36.0%、F+G 28.3%、first-Q 40.7%、Mean-Q 37.0%。跨 V0/V1/V2/V2-EMA 的固定 E10，first-Q 的版本均值为 44.8%，也是五种读出中最高。因此 **first-Q 是当前描述性默认主测试方式**；Mean-Q 只在训练变体 F 上达到 46%，应保留为方法特定消融。",
        "",
        "### 3. 原因分析",
        "",
        "- V0→V1 后 G-only 均值从 35.7% 到 41.3%，first-Q 从 47.0% 到 52.0%，说明共享 Action Encoder 的语义动作表示明显有利于 G。",
        "- V1→V2 后 F-only 与 F+G 同时大幅下降，与 TD 梯度进入 online LeWM/Action Encoder 后产生 latent/control representation drift 的假设一致；单 seed 不能证明因果，问题也不能只归因于 critic。",
        "- first-Q 只在真实 `z0` 与将执行的第一动作上读取 G，F 仍承担五步目标距离；它较少暴露于 CEM 候选动作和 imagined states，因此 OOD/rollout 误差是目前最合理的解释之一。",
        "- Mean-Q 在五个 imagined predecessor states 上反复读取 G，world-model 漂移、Q 尺度失配和 OOD 高估可能累积。训练变体 F 的 same-future 对比可能让 goal projection 沿轨迹更可用，但目前只有单 seed。",
        "",
        "## 负面结果与下一轮目标",
        "",
    ]
    lines += _markdown_table(
        ("发现", "证据", "含义/下一步"),
        (
            (
                "V2 world model 退化",
                "V1→V2 F-only 46.0%→26.0%",
                "先恢复 F，再谈 G 增益",
            ),
            (
                "EMA 未根治",
                "V2→EMA 各固定评分只恢复约 1–3 pp",
                "冻结或低 LR 微调 encoder/world",
            ),
            (
                "tail 干扰",
                "多组 F+G 低于 F-only",
                "优先 first-Q；G readout 加 gate/校准",
            ),
            ("Mean-Q 不通用", "V2-C 20%、V2-G2 26%", "只作为 F 等方法特定消融"),
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
        "1. 把联合模型固定 E10 的 F-only 六法均值从 27% 恢复到至少 V0/V1 的 46%。",
        "2. 预注册 `V1-C + first-Q` 与 `V1-G3 + F+G` 两条主基线；正式 O50 不再事后选 epoch。",
        "3. 若继续 joint training，先冻结 encoder/world 或给 TD 极低学习率，再分阶段解冻；增加对 V1 latent/prediction 的 anchor，并限制 TD 梯度进入 F。",
        "4. 在独立 dev pair set 上选择 α、epoch 与 Q 校准；对 CEM 候选/imagined-state 分布加入 conservative/calibration 训练，抑制 OOD 高估。",
        "5. 至少 3 个、最好 5 个 training seeds；用 paired bootstrap/McNemar 分析固定配置。",
        "",
        "## 事后 checkpoint 诊断（不能替代固定 E10）",
        "",
        "> 下表在看完同一 O50 的 E3–E10 后选择最大值，存在 selection bias，只能用于诊断训练轨迹。",
        "",
    ]
    lines += _markdown_table(
        ("版本", "评分", "事后最佳方法/epoch", "最佳", "E10最佳"),
        _global_posthoc_rows(cells),
    )
    lines += ["", "### V2-EMA 新评分稳定性", ""]
    lines += _markdown_table(
        ("方法", "评分", "E3–E10均值", "σ(pp)", "最佳", "E10"),
        _ema_new_stability_rows(cells),
    )
    lines += [
        "",
        "## 审计边界",
        "",
        f"- 465/465 格共享 episode-selection 文件 SHA-256 `{SELECTION_SHA256}` 与 action normalization SHA-256 `{ACTION_NORMALIZATION_SHA256}`。",
        f"- fixed 新评分 launcher 另有 valid-row-ranks SHA-256 `{FIXED_SELECTION_RANKS_SHA256}`；它是规范化索引哈希，不是 episode-selection 文件哈希，二者不能混写。",
        "- 每格成功数都由 50 个布尔 outcome 重算；完整来源路径与 SHA 位于 `all_o50_results.csv` 和 `reconciliation_ledger.json`。",
        "- EMA E3 的 G1/F+G 与 G2/F-only 使用隔离 retry attempt_02；原失败调度证据保留，不把失败单元伪装成原调度成功。",
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


def _fixed_headers(version: str) -> tuple[str, ...]:
    headers = ("Method", "F-only", "G-only", "F+G", "First-Q")
    return headers + (("Mean-Q",) if version in ("v2", "v2_ema_sg") else ())


def _fixed_widths(version: str) -> tuple[int, ...]:
    return (
        (1800, 3150, 3150, 3150, 3150)
        if version in ("v0", "v1")
        else (1400, 2600, 2600, 2600, 2600, 2600)
    )


def build_docx(
    cells: Sequence[ResultCell],
    base_inputs: v2_report.ValidatedReportInputs,
    *,
    v0_training_chart: bytes,
    v2_training_chart: bytes,
    training_chart: bytes,
    score_chart: bytes,
) -> bytes:
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required from the workspace runtime."
        ) from exc

    document = Document(str(base_inputs.base_document))
    v2_report._v1._configure_append_section(document)
    v2_report._set_header_text(
        document.sections[-1],
        "Results TD complete ledger · Cube O50 · 465 verified cells",
    )
    kicker = document.add_paragraph(style="Report Kicker")
    kicker.add_run("RESULTS TD / COMPLETE EXPERIMENT LEDGER")
    v2_report._font_paragraph(kicker, size=9.5, color="5C6975")
    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("All methods, versions, checkpoints and scores")
    v2_report._v1._set_run_font(run, size=24, color="0B2545", bold=True)
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(10)
    run = subtitle.add_run(
        "Cube · seed 3072 · 465 O50 cells · legacy + V0 + V1 + V2 + V2-EMA-SG"
    )
    v2_report._v1._set_run_font(run, size=12, color="4B5563")
    _add_body(
        document,
        "This is the consolidated ledger, not a new-score appendix. Every result "
        "uses the same 50 start-goal pairs. Best-epoch findings are explicitly "
        "post-hoc; all training uses one seed and no statistical-significance claim is made.",
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
                "V1-G3",
                "27/50 (54%)",
                "Keep as primary baseline",
            ),
            (
                "Best fixed result overall",
                "V1-C + First-Q",
                "28/50 (56%)",
                "Primary new-readout baseline",
            ),
            (
                "Best V2-EMA training variant",
                "F (Same-Future)",
                "E10 five-score mean 38.4%",
                "Retain this variant if joint training continues",
            ),
            (
                "Descriptive default test score",
                "First-Q",
                "4-version mean 44.8%",
                "Default readout; Mean-Q is an ablation",
            ),
            (
                "Leading failure hypothesis",
                "Joint world-model drift",
                "V1→V2 F-only 46%→26%",
                "Restore F before optimizing G",
            ),
        ),
        (3300, 3400, 3600, 4100),
    )

    _add_heading(document, "Coverage map", page_break=True)
    _add_table(
        document,
        ("Family/version", "Methods", "Epochs", "Scores", "Cells"),
        (
            ("Legacy structures", "7", "E10", "F / G(C) / combined", "21"),
            ("V0 raw action", "6", "E10", "F / G / F+G / First-Q", "24"),
            ("V1 action encoder", "6", "E10", "F / G / F+G / First-Q", "24"),
            ("V2 joint", "6", "E3-E10", "3 original + E10 First/Mean", "156"),
            ("V2-EMA-SG", "6", "E3-E10", "All five", "240"),
            ("TOTAL", "-", "-", "Same O50 selection", "465"),
        ),
        (3600, 1800, 2200, 4900, 1900),
    )
    _add_body(
        document,
        "The legacy and V0/V1 method/network/loss definitions remain in the "
        "preceding locked pages. The following pages add the complete V2 and "
        "V2-EMA trajectories and the cross-version decision analysis.",
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

    for version, title_text in (
        ("v0", "V0 fixed E10 — complete four-score matrix"),
        ("v1", "V1 fixed E10 — complete four-score matrix"),
        ("v2", "V2 fixed E10 — complete five-score matrix"),
        ("v2_ema_sg", "V2-EMA fixed E10 — complete five-score matrix"),
    ):
        _add_heading(document, title_text, page_break=True)
        _add_table(
            document,
            _fixed_headers(version),
            _fixed_rows(cells, version),
            _fixed_widths(version),
        )

    for version, modes, label in (
        ("v2", ORIGINAL_MODES, "V2"),
        ("v2_ema_sg", ALL_CONTROLLED_MODES, "V2-EMA"),
    ):
        _add_heading(document, f"{label} exact checkpoint trajectory", page_break=True)
        _add_body(
            document,
            f"Every entry below is an exact O50 result. Together these tables contain "
            f"{144 if version == 'v2' else 240} unique {label} cells.",
        )
        for index, mode in enumerate(modes):
            _add_heading(
                document,
                MODE_LABELS[mode],
                level=2,
                page_break=index > 0,
            )
            _add_table(
                document,
                ("Epoch", "C", "D", "F", "G1", "G2", "G3"),
                _trajectory_rows(cells, version, mode),
                (1600, 2130, 2130, 2130, 2130, 2130, 2150),
            )

    _add_heading(document, "Training / validation trajectories", page_break=True)
    _add_body(
        document,
        "Training totals are method-specific and cannot be ranked by absolute height. "
        "Use them only as within-method convergence diagnostics. The locked opening pages "
        "contain the legacy and V1 curves; the following charts add V0, V2 and V2-EMA. "
        "All numeric rows are retained in the reconciled CSV archives.",
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
    v2_report._add_picture(
        document,
        score_chart,
        title="Figure 4. V2-EMA new-score trajectories (E3-E10)",
        description="First-Q and Mean-Q O50 trajectories for all six methods.",
    )

    _add_heading(document, "Post-hoc checkpoint diagnosis", page_break=True)
    _add_body(
        document,
        "WARNING: maxima below were selected after observing E3-E10 on the same O50 "
        "pairs. They are selection-biased trajectory diagnostics and do not replace the fixed E10 results.",
        color="9C2F17",
        bold=True,
    )
    _add_table(
        document,
        ("Version", "Score", "Post-hoc best", "Best", "Best at E10"),
        _global_posthoc_rows(cells),
        (1900, 3000, 3700, 2900, 2900),
    )
    _add_heading(document, "V2-EMA new-score stability", level=2)
    _add_table(
        document,
        ("Method", "Score", "Mean", "Sigma pp", "Best", "E10"),
        _ema_new_stability_rows(cells),
        (1500, 3600, 2100, 2200, 2900, 2100),
    )

    _add_heading(document, "Interpretation and next objectives", page_break=True)
    _add_heading(document, "Fixed E10 version means", level=2)
    _add_table(
        document,
        (
            "Training version",
            "F-only",
            "G-only",
            "F+G",
            "First-Q",
            "Mean-Q",
            "Common-4 mean",
        ),
        _fixed_version_mean_rows(cells),
        (3000, 1850, 1850, 1850, 1850, 1850, 2150),
    )
    _add_heading(document, "Fixed-score comparison", level=2)
    _add_table(
        document,
        ("Score", "Versions", "Cells", "Version mean", "Best fixed cell"),
        _fixed_score_mean_rows(cells),
        (2500, 3000, 1500, 2400, 5000),
    )
    _add_heading(document, "V2-EMA training variants", level=2)
    _add_table(
        document,
        ("Training variant", "Five-score mean", "Best readout", "Best rate"),
        _ema_variant_mean_rows(cells),
        (2600, 3100, 5600, 3100),
    )
    _add_body(
        document,
        "There is no readout-independent training winner: V1-G3 leads the "
        "prespecified F+G column, whereas V1-C leads First-Q. These are descriptive "
        "single-seed results, not statistical superiority claims.",
        color="7A5A00",
        bold=True,
    )
    _add_table(
        document,
        ("Finding", "Evidence", "Interpretation", "Next objective"),
        (
            (
                "Action embedding helps G",
                "V0→V1 G-only 35.7→41.3%; First-Q 47→52%",
                "Semantic action representation improves critic readout",
                "Keep shared V1 Action Encoder",
            ),
            (
                "Joint tuning damages F",
                "V1→V2 F-only 46→26%",
                "Consistent with TD-gradient representation drift",
                "Restore F-only mean to >=46%",
            ),
            (
                "EMA is insufficient",
                "Only ~1–3 pp recovery",
                "Target stabilization does not undo online drift",
                "Freeze/low-LR then staged unfreeze",
            ),
            (
                "First-Q leads descriptively",
                "4-version mean 44.8%",
                "Real-state/first-action readout may limit rollout OOD",
                "Default test score; calibrate alpha on dev",
            ),
            (
                "Mean-Q is method-specific",
                "F=46%; V2-C=20%",
                "Repeated imagined-state readout compounds error",
                "Keep only as registered ablation",
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
        "V1-C + First-Q against V1-G3 + F+G on independent development pairs, "
        "pre-register epoch/alpha, then run at least three training seeds. If joint "
        "training is retained, use a much smaller TD learning rate, latent/prediction "
        "anchors to V1 and conservative calibration on CEM candidate actions.",
        bold=True,
    )
    _add_table(
        document,
        ("Audit field", "Locked value"),
        (
            ("Verified O50 cells", "465 / 465"),
            ("Episode selection", SELECTION_SHA256),
            ("Fixed valid-row-ranks hash", FIXED_SELECTION_RANKS_SHA256),
            ("Action normalization", ACTION_NORMALIZATION_SHA256),
            ("Training / planning seed", "3072 / 42"),
            ("EMA E3 retry", "G1/F+G and G2/F-only use isolated attempt_02"),
            (
                "Claim boundary",
                "One training seed; post-hoc epoch maxima are diagnostic only",
            ),
        ),
        (3500, 10900),
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
    base_inputs = v2_report.load_validated_report_inputs(
        original_summary_path=v2_report.DEFAULT_ORIGINAL_SUMMARY,
        training_loss_csv_path=v2_report.DEFAULT_TRAINING_LOSS_CSV,
        new_summary_path=v2_report.DEFAULT_NEW_SUMMARY,
        new_results_csv_path=v2_report.DEFAULT_NEW_RESULTS_CSV,
        fixed_results_csv_path=v2_report.DEFAULT_FIXED_RESULTS_CSV,
        base_document_path=DEFAULT_BASE_DOCUMENT,
    )
    if args.validate_only:
        print("PASS: complete Results TD ledger contains 465 verified O50 cells.")
        return 0
    v0_training_chart = build_training_chart_from_archive(
        args.v0_training_csv, title="V0 raw-action training and validation loss"
    )
    v2_training_chart = build_training_chart_from_archive(
        args.v2_training_csv, title="V2 joint training and validation loss"
    )
    training_chart = v2_report.build_training_loss_chart(base_inputs)
    score_chart = v2_report.build_new_score_chart(base_inputs)
    markdown = build_markdown(cells)
    document = build_docx(
        cells,
        base_inputs,
        v0_training_chart=v0_training_chart,
        v2_training_chart=v2_training_chart,
        training_chart=training_chart,
        score_chart=score_chart,
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
