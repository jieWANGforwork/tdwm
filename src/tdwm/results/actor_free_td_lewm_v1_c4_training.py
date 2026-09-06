"""Strict training-loss summary and plots for the formal V1-C4 run.

The reporter is deliberately result driven: it accepts only a completed
10-epoch/127,960-update C4 run and derives every exported value from the
Lightning CSV.  It never fabricates missing epochs or silently falls back to
step-level metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

METHOD = "actor_free_td_lewm_v1_c4"
METHOD_FAMILY = "actor_free_td_lewm_v1"
VARIANT = "c4"
FORMAL_EPOCHS = 10
FORMAL_STEPS_PER_EPOCH = 12_796
FORMAL_OPTIMIZER_UPDATES = FORMAL_EPOCHS * FORMAL_STEPS_PER_EPOCH

LOSS_FIELDS = (
    "vector_td_loss",
    "goal_projection_loss",
    "c4_total_loss",
)

CSV_FIELDS = (
    "epoch",
    "optimizer_updates",
    "metrics_step_zero_based",
    *(f"train_{name}" for name in LOSS_FIELDS),
    *(f"validation_{name}" for name in LOSS_FIELDS),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _require_equal(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} must equal {expected!r}; found {value!r}.")


def _validate_run_contract(
    training_result: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
) -> None:
    identity = {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
    }
    for source_name, source in (
        ("training_result", training_result),
        ("training_manifest", training_manifest),
    ):
        for name, expected in identity.items():
            _require_equal(
                source.get(name),
                expected,
                label=f"{source_name}.{name}",
            )

    _require_equal(
        training_result.get("final_epoch"),
        FORMAL_EPOCHS,
        label="training_result.final_epoch",
    )
    _require_equal(
        training_result.get("global_step"),
        FORMAL_OPTIMIZER_UPDATES,
        label="training_result.global_step",
    )
    _require_equal(
        training_result.get("frozen_world_model_verified"),
        True,
        label="training_result.frozen_world_model_verified",
    )

    training = training_manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training_manifest.training must be an object.")
    expected_training = {
        "formal_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
        "optimizer_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "configured_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
        "epochs": FORMAL_EPOCHS,
        "validation_skipped": False,
        "loss_metrics": list(LOSS_FIELDS),
    }
    for name, expected in expected_training.items():
        _require_equal(
            training.get(name),
            expected,
            label=f"training_manifest.training.{name}",
        )

    protocol = training_manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("training_manifest.protocol must be an object.")
    protocol_training = protocol.get("training")
    if not isinstance(protocol_training, Mapping):
        raise ValueError("training_manifest.protocol.training must be an object.")
    _require_equal(
        protocol_training.get("epochs"),
        FORMAL_EPOCHS,
        label="training_manifest.protocol.training.epochs",
    )
    _require_equal(
        protocol_training.get("optimizer_steps_per_epoch"),
        FORMAL_STEPS_PER_EPOCH,
        label="training_manifest.protocol.training.optimizer_steps_per_epoch",
    )

    result_protocol = training_result.get("protocol_sha256")
    manifest_protocol = training_manifest.get("protocol_sha256")
    if not isinstance(result_protocol, str) or len(result_protocol) != 64:
        raise ValueError("training_result.protocol_sha256 must be a SHA-256 digest.")
    _require_equal(
        manifest_protocol,
        result_protocol,
        label="training_manifest.protocol_sha256",
    )


def _finite_number(value: str | None, *, label: str) -> float:
    try:
        number = float(value) if value not in (None, "") else math.nan
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric; found {value!r}.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; found {value!r}.")
    return number


def _integral_number(value: str | None, *, label: str) -> int:
    number = _finite_number(value, label=label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer; found {value!r}.")
    return int(number)


def _metric_aliases(stage: str, field: str) -> tuple[str, str]:
    if stage == "train":
        return (f"train/{field}_epoch", f"train/{field}")
    return (f"validation/{field}", f"validation/{field}_epoch")


def _one_consistent_value(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"{label} is missing.")
    reference = values[-1]
    if not all(
        math.isclose(value, reference, rel_tol=1e-8, abs_tol=1e-10) for value in values
    ):
        raise ValueError(f"{label} has conflicting aggregate values: {values!r}.")
    if reference < 0.0:
        raise ValueError(f"{label} must be non-negative; found {reference!r}.")
    return reference


def load_epoch_metrics(path: str | Path) -> list[dict[str, int | float]]:
    """Load the ten formal C4 epoch aggregates from one Lightning CSV."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Lightning metrics CSV does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"Lightning metrics CSV is empty: {source}")

    for stage in ("train", "validation"):
        for field in LOSS_FIELDS:
            aliases = _metric_aliases(stage, field)
            if not fieldnames.intersection(aliases):
                raise ValueError(
                    f"Lightning metrics CSV is missing {stage}/{field}; "
                    f"expected one of {list(aliases)}."
                )

    expected_epochs = set(range(FORMAL_EPOCHS))
    observed_epochs: set[int] = set()
    observed_steps: list[int] = []
    aggregate_steps: dict[int, list[int]] = {epoch: [] for epoch in expected_epochs}
    values: dict[tuple[int, str, str], list[float]] = {
        (epoch, stage, field): []
        for epoch in expected_epochs
        for stage in ("train", "validation")
        for field in LOSS_FIELDS
    }

    for line_number, row in enumerate(rows, start=2):
        raw_epoch = row.get("epoch")
        raw_step = row.get("step")
        if raw_step not in (None, ""):
            observed_steps.append(
                _integral_number(raw_step, label=f"metrics.csv:{line_number} step")
            )
        if raw_epoch in (None, ""):
            continue
        epoch = _integral_number(raw_epoch, label=f"metrics.csv:{line_number} epoch")
        if epoch not in expected_epochs:
            raise ValueError(
                "Lightning metrics CSV must contain only zero-based epochs 0..9; "
                f"found epoch {epoch} on line {line_number}."
            )
        observed_epochs.add(epoch)

        row_has_aggregate = False
        for stage in ("train", "validation"):
            for field in LOSS_FIELDS:
                for alias in _metric_aliases(stage, field):
                    raw_value = row.get(alias)
                    if raw_value in (None, ""):
                        continue
                    values[(epoch, stage, field)].append(
                        _finite_number(
                            raw_value,
                            label=f"metrics.csv:{line_number} {alias}",
                        )
                    )
                    row_has_aggregate = True
        if row_has_aggregate and raw_step not in (None, ""):
            aggregate_steps[epoch].append(
                _integral_number(
                    raw_step,
                    label=f"metrics.csv:{line_number} aggregate step",
                )
            )

    if observed_epochs != expected_epochs:
        raise ValueError(
            "Lightning metrics CSV must cover exactly zero-based epochs 0..9; "
            f"found {sorted(observed_epochs)}."
        )
    expected_final_step = FORMAL_OPTIMIZER_UPDATES - 1
    if not observed_steps or max(observed_steps) != expected_final_step:
        raise ValueError(
            "Lightning metrics CSV final zero-based step must equal "
            f"{expected_final_step}; found "
            f"{max(observed_steps) if observed_steps else None}."
        )

    epochs: list[dict[str, int | float]] = []
    for zero_based_epoch in range(FORMAL_EPOCHS):
        if not aggregate_steps[zero_based_epoch]:
            raise ValueError(
                f"epoch {zero_based_epoch + 1} has no aggregate metric step."
            )
        item: dict[str, int | float] = {
            "epoch": zero_based_epoch + 1,
            "optimizer_updates": (zero_based_epoch + 1) * FORMAL_STEPS_PER_EPOCH,
            "metrics_step_zero_based": max(aggregate_steps[zero_based_epoch]),
        }
        for stage in ("train", "validation"):
            for field in LOSS_FIELDS:
                item[f"{stage}_{field}"] = _one_consistent_value(
                    values[(zero_based_epoch, stage, field)],
                    label=f"epoch {zero_based_epoch + 1} {stage}/{field}",
                )
            components = sum(
                float(item[f"{stage}_{field}"]) for field in LOSS_FIELDS[:-1]
            )
            expected_total = components
            actual_total = float(item[f"{stage}_c4_total_loss"])
            if not math.isclose(
                actual_total, expected_total, rel_tol=2e-5, abs_tol=1e-7
            ):
                raise ValueError(
                    f"epoch {zero_based_epoch + 1} {stage} C4 total does not "
                    "equal vector_td_loss + goal_projection_loss: "
                    f"expected {expected_total}, found "
                    f"{actual_total}."
                )
        epochs.append(item)
    return epochs


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_epoch_csv(path: Path, epochs: Sequence[Mapping[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for epoch in epochs:
            writer.writerow(
                {
                    name: (
                        format(float(epoch[name]), ".12g")
                        if isinstance(epoch[name], float)
                        else epoch[name]
                    )
                    for name in CSV_FIELDS
                }
            )
    temporary.replace(path)


def render_loss_chart(
    epochs: Sequence[Mapping[str, int | float]],
    output_path: str | Path,
    *,
    dpi: int = 180,
) -> Path:
    """Render total and component curves as one document-ready PNG."""

    if len(epochs) != FORMAL_EPOCHS:
        raise ValueError(f"Chart requires exactly {FORMAL_EPOCHS} epochs.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    epoch_numbers = [int(row["epoch"]) for row in epochs]
    component_specs = (
        ("vector_td_loss", "Vector TD", "#2F5597"),
        ("goal_projection_loss", "Goal projection", "#C55A11"),
    )

    figure = plt.figure(figsize=(13.2, 9.0), dpi=dpi)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.9, 1.1))
    total_axis = figure.add_subplot(grid[0, :])
    train_axis = figure.add_subplot(grid[1, 0])
    validation_axis = figure.add_subplot(grid[1, 1])
    try:
        total_values: list[float] = []
        for stage, label, color in (
            ("train", "Train total", "#2F5597"),
            ("validation", "Validation total", "#C55A11"),
        ):
            series = [float(row[f"{stage}_c4_total_loss"]) for row in epochs]
            total_values.extend(series)
            total_axis.plot(
                epoch_numbers,
                series,
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                label=label,
            )
        if total_values and min(total_values) > 0.0:
            total_axis.set_yscale("log")
        total_axis.set_title(
            "C4 total objective — train vs validation", fontweight="bold"
        )
        total_axis.set_ylabel("Loss (log scale)" if min(total_values) > 0 else "Loss")
        total_axis.legend(frameon=False, ncol=2)

        for axis, stage, title in (
            (train_axis, "train", "Train components"),
            (validation_axis, "validation", "Validation components"),
        ):
            panel_values: list[float] = []
            for field, label, color in component_specs:
                series = [float(row[f"{stage}_{field}"]) for row in epochs]
                panel_values.extend(series)
                axis.plot(
                    epoch_numbers,
                    series,
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                    label=label,
                )
            if panel_values and min(panel_values) > 0.0:
                axis.set_yscale("log")
            axis.set_title(title, fontweight="bold")
            axis.set_ylabel("Loss (log scale)" if min(panel_values) > 0 else "Loss")
            axis.legend(frameon=False, fontsize=8)

        for axis in (total_axis, train_axis, validation_axis):
            axis.set_xlabel("Epoch")
            axis.set_xticks(range(1, FORMAL_EPOCHS + 1))
            axis.grid(alpha=0.24)
        figure.suptitle("Actor-Free TD-LeWM V1-C4 loss curves", fontweight="bold")
        figure.text(
            0.5,
            0.94,
            "Total = vector TD + goal projection",
            ha="center",
            color="#5C6975",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(
            temporary,
            format="png",
            bbox_inches="tight",
            metadata={"Software": "tdwm"},
        )
        temporary.replace(output)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    return output


def build_training_summary(
    *,
    metrics_path: str | Path,
    training_result_path: str | Path,
    training_manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate a completed C4 run and return its loss-report payload."""

    metrics = Path(metrics_path)
    result_path = Path(training_result_path)
    manifest_path = Path(training_manifest_path)
    result = _read_mapping(result_path, label="training result")
    manifest = _read_mapping(manifest_path, label="training manifest")
    _validate_run_contract(result, manifest)
    epochs = load_epoch_metrics(metrics)

    train_totals = [float(row["train_c4_total_loss"]) for row in epochs]
    validation_totals = [float(row["validation_c4_total_loss"]) for row in epochs]
    best_validation_index = min(
        range(len(validation_totals)), key=validation_totals.__getitem__
    )
    return {
        "schema_version": 1,
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "formal_contract": {
            "epochs": FORMAL_EPOCHS,
            "optimizer_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
            "optimizer_updates": FORMAL_OPTIMIZER_UPDATES,
            "lightning_final_step_zero_based": FORMAL_OPTIMIZER_UPDATES - 1,
            "loss_fields": list(LOSS_FIELDS),
            "total_loss_definition": "vector_td_loss+goal_projection_loss",
        },
        "source_files": {
            "metrics_csv": {
                "path": str(metrics.resolve()),
                "sha256": _sha256(metrics),
            },
            "training_result": {
                "path": str(result_path.resolve()),
                "sha256": _sha256(result_path),
            },
            "training_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
            },
        },
        "run": {
            "seed": result.get("seed"),
            "protocol_sha256": result["protocol_sha256"],
            "deployment_checkpoint": result.get("deployment_checkpoint"),
            "deployment_checkpoint_sha256": result.get("deployment_checkpoint_sha256"),
        },
        "summary": {
            "train_total_initial": train_totals[0],
            "train_total_final": train_totals[-1],
            "validation_total_initial": validation_totals[0],
            "validation_total_final": validation_totals[-1],
            "best_validation_epoch": best_validation_index + 1,
            "best_validation_total": validation_totals[best_validation_index],
        },
        "epochs": epochs,
    }


def write_training_metrics_report(
    *,
    metrics_path: str | Path,
    training_result_path: str | Path,
    training_manifest_path: str | Path,
    output_dir: str | Path,
    dpi: int = 180,
) -> dict[str, Path]:
    """Write validated JSON, epoch CSV and a single three-panel PNG."""

    output = Path(output_dir)
    summary = build_training_summary(
        metrics_path=metrics_path,
        training_result_path=training_result_path,
        training_manifest_path=training_manifest_path,
    )
    json_path = output / "actor_free_td_lewm_v1_c4_training_summary.json"
    csv_path = output / "actor_free_td_lewm_v1_c4_epoch_losses.csv"
    png_path = output / "actor_free_td_lewm_v1_c4_loss_curves.png"
    _atomic_text(
        json_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    _write_epoch_csv(csv_path, summary["epochs"])
    render_loss_chart(summary["epochs"], png_path, dpi=dpi)
    return {"json": json_path, "csv": csv_path, "png": png_path}


__all__ = [
    "CSV_FIELDS",
    "FORMAL_EPOCHS",
    "FORMAL_OPTIMIZER_UPDATES",
    "FORMAL_STEPS_PER_EPOCH",
    "LOSS_FIELDS",
    "build_training_summary",
    "load_epoch_metrics",
    "render_loss_chart",
    "write_training_metrics_report",
]
