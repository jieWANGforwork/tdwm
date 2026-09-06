from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tdwm.results.actor_free_td_lewm_v1_c4_training import (
    CSV_FIELDS,
    FORMAL_EPOCHS,
    FORMAL_OPTIMIZER_UPDATES,
    FORMAL_STEPS_PER_EPOCH,
    LOSS_FIELDS,
    build_training_summary,
    load_epoch_metrics,
    write_training_metrics_report,
)

PROTOCOL_SHA256 = "a" * 64


def _component_values(epoch: int, *, validation: bool) -> dict[str, float]:
    offset = 0.25 if validation else 0.0
    scale = float(epoch + 1)
    components = {
        "real_vector_loss": 8.0 / scale + offset,
        "real_goal_loss": 16.0 / scale + offset,
        "predicted_vector_loss": 10.0 / scale + offset,
        "predicted_goal_loss": 18.0 / scale + offset,
    }
    components["c4_total_loss"] = 0.5 * sum(components.values())
    return components


def _write_completed_run(root: Path) -> tuple[Path, Path, Path]:
    run = root / "seed_3072"
    metrics_path = run / "metrics/version_1/metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    train_columns = [f"train/{field}_epoch" for field in LOSS_FIELDS]
    validation_columns = [f"validation/{field}" for field in LOSS_FIELDS]
    step_columns = [f"train/{field}_step" for field in LOSS_FIELDS]
    fieldnames = ["epoch", "step", *train_columns, *validation_columns, *step_columns]
    rows: list[dict[str, str | int | float]] = []
    for epoch in range(FORMAL_EPOCHS):
        final_step = (epoch + 1) * FORMAL_STEPS_PER_EPOCH - 1
        step_row: dict[str, str | int | float] = {
            "epoch": epoch,
            "step": final_step,
        }
        step_row.update(
            {
                f"train/{field}_step": value
                for field, value in _component_values(epoch, validation=False).items()
            }
        )
        rows.append(step_row)

        train_row: dict[str, str | int | float] = {
            "epoch": epoch,
            "step": final_step,
        }
        train_row.update(
            {
                f"train/{field}_epoch": value
                for field, value in _component_values(epoch, validation=False).items()
            }
        )
        rows.append(train_row)

        validation_row: dict[str, str | int | float] = {
            "epoch": epoch,
            "step": final_step,
        }
        validation_row.update(
            {
                f"validation/{field}": value
                for field, value in _component_values(epoch, validation=True).items()
            }
        )
        rows.append(validation_row)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result_path = run / "training_result.json"
    result_path.write_text(
        json.dumps(
            {
                "method": "actor_free_td_lewm_v1_c4",
                "method_family": "actor_free_td_lewm_v1",
                "variant": "c4",
                "seed": 3072,
                "final_epoch": FORMAL_EPOCHS,
                "global_step": FORMAL_OPTIMIZER_UPDATES,
                "protocol_sha256": PROTOCOL_SHA256,
                "frozen_world_model_verified": True,
                "deployment_checkpoint": "/formal/epoch_10.pt",
                "deployment_checkpoint_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = run / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": "actor_free_td_lewm_v1_c4",
                "method_family": "actor_free_td_lewm_v1",
                "variant": "c4",
                "protocol_sha256": PROTOCOL_SHA256,
                "protocol": {
                    "training": {
                        "epochs": FORMAL_EPOCHS,
                        "optimizer_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
                    }
                },
                "training": {
                    "formal_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
                    "optimizer_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
                    "configured_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
                    "epochs": FORMAL_EPOCHS,
                    "validation_skipped": False,
                    "loss_metrics": list(LOSS_FIELDS),
                },
            }
        ),
        encoding="utf-8",
    )
    return metrics_path, result_path, manifest_path


def test_completed_c4_run_writes_json_epoch_csv_and_loss_png(
    tmp_path: Path,
) -> None:
    metrics, result, manifest = _write_completed_run(tmp_path)
    output = tmp_path / "report"

    paths = write_training_metrics_report(
        metrics_path=metrics,
        training_result_path=result,
        training_manifest_path=manifest,
        output_dir=output,
        dpi=72,
    )

    assert set(paths) == {"json", "csv", "png"}
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["formal_contract"]["epochs"] == 10
    assert payload["formal_contract"]["optimizer_updates"] == 127_960
    assert payload["formal_contract"]["loss_fields"] == list(LOSS_FIELDS)
    assert len(payload["epochs"]) == 10
    assert payload["epochs"][-1]["optimizer_updates"] == 127_960
    assert payload["summary"]["best_validation_epoch"] == 10

    with paths["csv"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert len(rows) == 10
    assert rows[-1]["epoch"] == "10"
    assert rows[-1]["optimizer_updates"] == "127960"
    assert paths["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_c4_metrics_ignore_step_losses_but_require_every_epoch_aggregate(
    tmp_path: Path,
) -> None:
    metrics, _result, _manifest = _write_completed_run(tmp_path)
    rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8")))
    fieldnames = list(rows[0])
    for row in rows:
        if row["epoch"] == "4":
            row["validation/predicted_goal_loss"] = ""
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        ValueError, match=r"epoch 5 validation/predicted_goal_loss is missing"
    ):
        load_epoch_metrics(metrics)


def test_c4_metrics_require_exactly_127960_optimizer_updates(
    tmp_path: Path,
) -> None:
    metrics, _result, _manifest = _write_completed_run(tmp_path)
    rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8")))
    fieldnames = list(rows[0])
    for row in rows:
        if row["step"] == str(FORMAL_OPTIMIZER_UPDATES - 1):
            row["step"] = str(FORMAL_OPTIMIZER_UPDATES - 2)
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="final zero-based step"):
        load_epoch_metrics(metrics)


def test_c4_report_rejects_nonformal_result_or_loss_identity(tmp_path: Path) -> None:
    metrics, result, manifest = _write_completed_run(tmp_path)
    result_payload = json.loads(result.read_text(encoding="utf-8"))
    result_payload["global_step"] = FORMAL_OPTIMIZER_UPDATES - 1
    result.write_text(json.dumps(result_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training_result.global_step"):
        build_training_summary(
            metrics_path=metrics,
            training_result_path=result,
            training_manifest_path=manifest,
        )


def test_c4_report_rejects_total_inconsistent_with_four_components(
    tmp_path: Path,
) -> None:
    metrics, _result, _manifest = _write_completed_run(tmp_path)
    rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8")))
    fieldnames = list(rows[0])
    for row in rows:
        if row["epoch"] == "7" and row["validation/c4_total_loss"]:
            row["validation/c4_total_loss"] = "999"
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=r"epoch 8 validation C4 total"):
        load_epoch_metrics(metrics)
