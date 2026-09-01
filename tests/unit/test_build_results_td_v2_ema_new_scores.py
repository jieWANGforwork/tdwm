from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_results_td_v2_ema_new_scores.py"
SPEC = importlib.util.spec_from_file_location("build_results_td_v2_ema", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _count(epoch: int, variant: str, mode: str) -> int:
    return (
        10
        + (
            epoch * 3
            + BUILDER.VARIANTS.index(variant) * 4
            + BUILDER.SCORE_MODES.index(mode) * 2
        )
        % 31
    )


def _fixed_count(version: str, variant: str, mode: str) -> int:
    return (
        8
        + (
            BUILDER.FIXED_VERSIONS.index(version) * 7
            + BUILDER.VARIANTS.index(variant) * 3
            + BUILDER.SCORE_MODES.index(mode) * 5
        )
        % 33
    )


def _rate_fields(count: int) -> dict[str, float | int]:
    return {
        "success_count": count,
        "success_rate": count / BUILDER.EPISODES,
        "success_rate_percent": count * 100.0 / BUILDER.EPISODES,
    }


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_new_archive(tmp_path: Path) -> tuple[Path, Path, Path]:
    results_by_epoch: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    best_by_epoch: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}
    result_rows: list[dict[str, object]] = []
    for epoch in BUILDER.EMA_EPOCHS:
        epoch_results: dict[str, dict[str, dict[str, float | int]]] = {}
        for variant in BUILDER.VARIANTS:
            epoch_results[variant] = {}
            checkpoint_sha = _sha(f"checkpoint-{epoch}-{variant}")
            training_sha = _sha(f"training-{epoch}-{variant}")
            for mode in BUILDER.SCORE_MODES:
                count = _count(epoch, variant, mode)
                epoch_results[variant][mode] = _rate_fields(count)
                result_rows.append(
                    {
                        "epoch": epoch,
                        "variant": variant,
                        "method": f"actor_free_td_lewm_v2_ema_sg_{variant}",
                        "score_mode": mode,
                        "g_first_weight": (
                            BUILDER.FIRST_ACTION_WEIGHT
                            if mode == "f_plus_g_first"
                            else ""
                        ),
                        **_rate_fields(count),
                        "checkpoint_epoch": epoch,
                        "checkpoint_global_step": epoch * BUILDER.STEPS_PER_EPOCH,
                        "checkpoint_sha256": checkpoint_sha,
                        "training_manifest_sha256": training_sha,
                        "evaluation_commit": BUILDER.EVALUATION_COMMIT,
                        "source_scope": (
                            "strict_epoch3" if epoch == 3 else "original_epoch4_10"
                        ),
                        "source_status": "SUCCEEDED" if epoch == 3 else "REUSED",
                        "cell_id": BUILDER._expected_cell_id(epoch, variant, mode),
                    }
                )
        results_by_epoch[str(epoch)] = epoch_results
        best_by_epoch[str(epoch)] = {}
        for mode in BUILDER.SCORE_MODES:
            best_count = max(
                int(epoch_results[variant][mode]["success_count"])
                for variant in BUILDER.VARIANTS
            )
            best_by_epoch[str(epoch)][mode] = [
                {
                    "variant": variant,
                    **_rate_fields(best_count),
                }
                for variant in BUILDER.VARIANTS
                if epoch_results[variant][mode]["success_count"] == best_count
            ]

    fixed_summary: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    fixed_grid = sorted(
        BUILDER._expected_fixed_grid(),
        key=lambda identity: (
            BUILDER.FIXED_VERSIONS.index(identity[0]),
            BUILDER.VARIANTS.index(identity[1]),
            BUILDER.SCORE_MODES.index(identity[2]),
        ),
    )
    for version, variant, mode in fixed_grid:
        count = _fixed_count(version, variant, mode)
        summary_row = {
            "version": version,
            "variant": variant,
            "score_mode": mode,
            **_rate_fields(count),
        }
        fixed_summary.append(summary_row)
        fixed_rows.append(
            {
                "version": version,
                "variant": variant,
                "method": f"actor_free_td_lewm_{version}_{variant}",
                "score_mode": mode,
                "g_first_weight": (
                    BUILDER.FIRST_ACTION_WEIGHT if mode == "f_plus_g_first" else ""
                ),
                **_rate_fields(count),
                "checkpoint_sha256": _sha(f"fixed-{version}-{variant}"),
                "selection_sha256": BUILDER.FIXED_SELECTION_SHA256,
                "episode_selection_file_sha256": (
                    BUILDER.SHARED_EPISODE_SELECTION_SHA256
                ),
                "action_normalization_sha256": (BUILDER.ACTION_NORMALIZATION_SHA256),
                "evaluation_commit": BUILDER.EVALUATION_COMMIT,
                "job_id": f"fixed-{version}-{variant}-{mode}",
                "source_launcher_manifest": f"/archive/{version}/{variant}.json",
            }
        )

    summary = {
        "schema_version": BUILDER.SCHEMA_VERSION,
        "source": "actor_free_td_lewm_v2_ema_new_score_summary",
        "cell_count": 96,
        "selection_sha256": BUILDER.EMA_SELECTION_SHA256,
        "training_commit": BUILDER.TRAINING_COMMIT,
        "evaluation_commit": BUILDER.EVALUATION_COMMIT,
        "fixed_checkpoint_cell_count": 24,
        "fixed_checkpoint_selection_sha256": BUILDER.FIXED_SELECTION_SHA256,
        "fixed_checkpoint_results": fixed_summary,
        "results_by_epoch": results_by_epoch,
        "best_by_epoch_and_score_mode": best_by_epoch,
    }
    summary_path = tmp_path / "summary.json"
    new_results_path = tmp_path / "results.csv"
    fixed_results_path = tmp_path / "fixed_checkpoint_results.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(new_results_path, BUILDER.NEW_RESULTS_FIELDS, result_rows)
    _write_csv(fixed_results_path, BUILDER.FIXED_RESULTS_FIELDS, fixed_rows)
    return summary_path, new_results_path, fixed_results_path


def _load_inputs(tmp_path: Path):
    summary, results, fixed = _write_new_archive(tmp_path)
    return BUILDER.load_validated_report_inputs(
        original_summary_path=BUILDER.DEFAULT_ORIGINAL_SUMMARY,
        training_loss_csv_path=BUILDER.DEFAULT_TRAINING_LOSS_CSV,
        new_summary_path=summary,
        new_results_csv_path=results,
        fixed_results_csv_path=fixed,
        base_document_path=BUILDER.DEFAULT_BASE_DOCUMENT,
    )


def test_complete_96_plus_24_bundle_builds_required_report_sections(
    tmp_path: Path,
) -> None:
    inputs = _load_inputs(tmp_path)

    assert len(inputs.new_rows) == 96
    assert len(inputs.fixed_rows) == 24
    assert len(inputs.training_loss_rows) == 60
    assert len(BUILDER._epoch10_rows(inputs)) == 6
    assert len(BUILDER._fixed_table_rows(inputs)) == 24
    assert len(BUILDER._best_epoch_table_rows(inputs)) == 12

    markdown = BUILDER.build_markdown_report(
        inputs,
        training_chart_reference="figures/training.png",
        score_chart_reference="figures/scores.png",
    )
    assert "L_{total}=L_{pred}+0.09L_{SIGReg}" in markdown
    assert "f_plus_g_first" in markdown
    assert "g_only_f_rollout_mean" in markdown
    assert "J_{first}" in markdown
    assert "J_{mean}" in markdown
    assert "没有 `γ^4`" in markdown
    assert "事后选择 checkpoint" in markdown
    assert BUILDER.TRAINING_COMMIT in markdown
    assert BUILDER.EVALUATION_COMMIT in markdown


def test_changed_evaluation_commit_is_rejected(tmp_path: Path) -> None:
    summary, results, fixed = _write_new_archive(tmp_path)
    with results.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    rows[0]["evaluation_commit"] = "0" * 40
    _write_csv(results, BUILDER.NEW_RESULTS_FIELDS, rows)

    with pytest.raises(BUILDER.ResultsTDV2EMAError, match="evaluation_commit"):
        BUILDER.load_validated_report_inputs(
            original_summary_path=BUILDER.DEFAULT_ORIGINAL_SUMMARY,
            training_loss_csv_path=BUILDER.DEFAULT_TRAINING_LOSS_CSV,
            new_summary_path=summary,
            new_results_csv_path=results,
            fixed_results_csv_path=fixed,
            base_document_path=BUILDER.DEFAULT_BASE_DOCUMENT,
        )


def test_incomplete_fixed_grid_is_rejected(tmp_path: Path) -> None:
    summary, results, fixed = _write_new_archive(tmp_path)
    with fixed.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    _write_csv(fixed, BUILDER.FIXED_RESULTS_FIELDS, rows[:-1])

    with pytest.raises(BUILDER.ResultsTDV2EMAError, match="24 rows"):
        BUILDER.load_validated_report_inputs(
            original_summary_path=BUILDER.DEFAULT_ORIGINAL_SUMMARY,
            training_loss_csv_path=BUILDER.DEFAULT_TRAINING_LOSS_CSV,
            new_summary_path=summary,
            new_results_csv_path=results,
            fixed_results_csv_path=fixed,
            base_document_path=BUILDER.DEFAULT_BASE_DOCUMENT,
        )


def test_fixed_cells_must_share_the_episode_selection_file(tmp_path: Path) -> None:
    summary, results, fixed = _write_new_archive(tmp_path)
    with fixed.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    rows[0]["episode_selection_file_sha256"] = "0" * 64
    _write_csv(fixed, BUILDER.FIXED_RESULTS_FIELDS, rows)

    with pytest.raises(BUILDER.ResultsTDV2EMAError, match="shared 50-pair"):
        BUILDER.load_validated_report_inputs(
            original_summary_path=BUILDER.DEFAULT_ORIGINAL_SUMMARY,
            training_loss_csv_path=BUILDER.DEFAULT_TRAINING_LOSS_CSV,
            new_summary_path=summary,
            new_results_csv_path=results,
            fixed_results_csv_path=fixed,
            base_document_path=BUILDER.DEFAULT_BASE_DOCUMENT,
        )


def test_stale_best_by_epoch_summary_is_rejected(tmp_path: Path) -> None:
    summary, _results, _fixed = _write_new_archive(tmp_path)
    payload = json.loads(summary.read_text())
    payload["best_by_epoch_and_score_mode"]["3"]["f_plus_g_first"][0][
        "success_count"
    ] -= 1
    summary.write_text(json.dumps(payload))

    with pytest.raises(BUILDER.ResultsTDV2EMAError, match="Best-by-epoch"):
        BUILDER._validate_new_summary(summary)
