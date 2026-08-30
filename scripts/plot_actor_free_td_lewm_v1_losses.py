#!/usr/bin/env python3
"""Plot the six-method V1 training and validation loss archive.

The two panels intentionally have different semantics.  Training reports each
variant's own method objective, while validation reports the common base TD
objective used for all six variants.  The loader fails closed on incomplete or
non-finite 6 x 10 archives instead of drawing a partial figure.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    REPOSITORY_ROOT / "reports/artifacts/actor_free_td_lewm_v1_cube_seed3072"
)
DEFAULT_INPUT = DEFAULT_ARTIFACT_ROOT / "training_loss_curves.csv"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "training_loss_curves.png"

VARIANT_ORDER = ("c", "d", "f", "g1", "g2", "g3")
EXPECTED_EPOCHS = tuple(range(1, 11))
EXPECTED_ROWS = len(VARIANT_ORDER) * len(EXPECTED_EPOCHS)
TRAIN_COLUMNS = (
    "train_method_loss",
    "train_method_objective",
    "train_method_objective_loss",
    "train_loss",
)
VALIDATION_COLUMNS = (
    "validation_base_td_loss",
    "validation_common_base_td",
    "validation_common_base_td_loss",
    "validation_loss",
)


@dataclass(frozen=True)
class LossPoint:
    variant: str
    display_name: str
    epoch: int
    train_method_objective: float
    validation_common_base_td: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the complete 60-row Actor-Free TD-LeWM V1 loss archive. "
            "Train and validation use deliberately different objectives."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _select_column(
    fieldnames: Iterable[str] | None,
    candidates: tuple[str, ...],
    *,
    semantic_name: str,
) -> str:
    available = set(fieldnames or ())
    matches = [candidate for candidate in candidates if candidate in available]
    if len(matches) != 1:
        raise ValueError(
            f"Loss CSV must contain exactly one {semantic_name} column from "
            f"{list(candidates)}; found {matches}."
        )
    return matches[0]


def _finite_float(value: str | None, *, context: str) -> float:
    try:
        number = float(value) if value is not None else math.nan
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric, found {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, found {value!r}.")
    return number


def load_loss_points(path: str | Path) -> dict[str, tuple[LossPoint, ...]]:
    """Load and validate one exact six-variant, ten-epoch loss table."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Loss CSV does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        train_column = _select_column(
            reader.fieldnames,
            TRAIN_COLUMNS,
            semantic_name="train method objective",
        )
        validation_column = _select_column(
            reader.fieldnames,
            VALIDATION_COLUMNS,
            semantic_name="validation common base TD",
        )
        required = {"variant", "display_name", "epoch"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Loss CSV is missing columns: {sorted(missing)}")
        raw_rows = list(reader)

    if len(raw_rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Loss CSV must contain exactly {EXPECTED_ROWS} data rows "
            f"(6 variants x 10 epochs), found {len(raw_rows)}."
        )

    grouped: dict[str, list[LossPoint]] = {variant: [] for variant in VARIANT_ORDER}
    for row_index, row in enumerate(raw_rows, start=2):
        variant = row.get("variant", "")
        if variant not in grouped:
            raise ValueError(
                f"Loss CSV row {row_index} has unsupported variant {variant!r}."
            )
        display_name = (row.get("display_name") or "").strip()
        if not display_name:
            raise ValueError(f"Loss CSV row {row_index} has no display_name.")
        try:
            epoch = int(row.get("epoch", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Loss CSV row {row_index} has invalid epoch {row.get('epoch')!r}."
            ) from exc
        grouped[variant].append(
            LossPoint(
                variant=variant,
                display_name=display_name,
                epoch=epoch,
                train_method_objective=_finite_float(
                    row.get(train_column),
                    context=f"row {row_index} {train_column}",
                ),
                validation_common_base_td=_finite_float(
                    row.get(validation_column),
                    context=f"row {row_index} {validation_column}",
                ),
            )
        )
        comparable = row.get("cross_method_comparable")
        if comparable is not None and comparable.strip().lower() not in {
            "false",
            "0",
            "no",
        }:
            raise ValueError(
                "V1 method-objective loss rows must not claim cross-method "
                f"comparability (row {row_index})."
            )
        train_semantics = row.get("train_metric_semantics")
        if (
            train_semantics is not None
            and train_semantics != "method_specific_objective"
        ):
            raise ValueError(
                f"Loss CSV row {row_index} must label train semantics as "
                "method_specific_objective."
            )
        validation_semantics = row.get("validation_metric_semantics")
        if (
            validation_semantics is not None
            and validation_semantics != "common_base_td"
        ):
            raise ValueError(
                f"Loss CSV row {row_index} must label validation semantics as "
                "common_base_td."
            )
        ranking_semantics = row.get("cross_method_ranking_metric")
        if (
            ranking_semantics is not None
            and ranking_semantics != "false_use_formal_o50_f_plus_g"
        ):
            raise ValueError(
                f"Loss CSV row {row_index} must label cross-method ranking as "
                "false_use_formal_o50_f_plus_g."
            )

    validated: dict[str, tuple[LossPoint, ...]] = {}
    for variant in VARIANT_ORDER:
        points = sorted(grouped[variant], key=lambda point: point.epoch)
        epochs = tuple(point.epoch for point in points)
        if epochs != EXPECTED_EPOCHS:
            raise ValueError(
                f"Variant {variant!r} must contain epochs 1..10 exactly once; "
                f"found {list(epochs)}."
            )
        names = {point.display_name for point in points}
        if len(names) != 1:
            raise ValueError(
                f"Variant {variant!r} has inconsistent display names: {names}."
            )
        validated[variant] = tuple(points)
    return validated


def render_loss_chart(
    curves: Mapping[str, tuple[LossPoint, ...]],
    output_path: str | Path,
    *,
    dpi: int = 180,
) -> Path:
    """Render one atomic, document-ready PNG from validated curves."""

    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if tuple(curves) != VARIANT_ORDER:
        raise ValueError(
            f"Curves must use canonical variant order {list(VARIANT_ORDER)}."
        )
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    colors = ("#2F5597", "#C55A11", "#548235", "#7030A0", "#0070C0", "#A61C3C")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.35), dpi=dpi)
    try:
        for color, variant in zip(colors, VARIANT_ORDER):
            points = curves[variant]
            epochs = [point.epoch for point in points]
            label = points[0].display_name
            train_values = [point.train_method_objective for point in points]
            if any(value <= 0.0 for value in train_values):
                raise ValueError(
                    "Train method objectives must be positive for the declared "
                    "log-scale diagnostic."
                )
            axes[0].plot(
                epochs,
                train_values,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=3.4,
                label=label,
            )
            axes[1].plot(
                epochs,
                [point.validation_common_base_td for point in points],
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=3.4,
                label=label,
            )
        panel_specs = (
            (
                axes[0],
                "Train — method objective (log scale)",
                "Method-specific training objective (log scale)",
            ),
            (
                axes[1],
                "Validation — common base TD",
                "Common base TD loss",
            ),
        )
        for axis, title, ylabel in panel_specs:
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_xticks(EXPECTED_EPOCHS)
            axis.grid(alpha=0.24)
        axes[0].set_yscale("log")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        fig.suptitle("Actor-Free TD-LeWM V1 loss curves", fontweight="bold")
        fig.text(
            0.5,
            0.925,
            (
                "Train curves use each method's objective; validation uses one "
                "shared base-TD objective across all six methods."
            ),
            ha="center",
            color="#5C6975",
        )
        fig.tight_layout(rect=(0, 0.13, 1, 0.9))
        fig.savefig(
            temporary,
            format="png",
            bbox_inches="tight",
            metadata={"Software": "tdwm"},
        )
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    args = parse_args()
    try:
        curves = load_loss_points(args.input)
        output = render_loss_chart(curves, args.output, dpi=args.dpi)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(output)


if __name__ == "__main__":
    main()
