from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tdwm.results import actor_free_td_lewm_complete as results


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_hashes(label: str) -> dict[str, str]:
    return {
        "results.json": _sha(f"{label}:results"),
        "protocol_manifest.json": _sha(f"{label}:protocol"),
        "episode_selection.json": results.SELECTION_SHA256,
        "action_normalization.json": results.ACTION_NORMALIZATION_SHA256,
    }


def _prior_ledger() -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, Any]]],
    dict[tuple[int, str], dict[str, Any]],
]:
    ema_references: dict[tuple[int, str], dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    for epoch in results.EPOCHS:
        for variant in results.VARIANTS:
            checkpoint = _sha(f"ema:{epoch}:{variant}")
            ema_references[(epoch, variant)] = {
                "checkpoint_sha256": checkpoint,
                "checkpoint_path": f"/server/ema/{variant}/epoch_{epoch:02d}.pt",
                "epoch": epoch,
                "global_step": epoch * results.STEPS_PER_EPOCH,
                "training_commit": results.EMA_TRAINING_COMMIT,
            }
            for mode in results.NEW_SCORE_MODES:
                alpha = results.G_FIRST_WEIGHT if mode == "f_plus_g_first" else None
                label = f"ema:{epoch}:{variant}:{mode}"
                cells.append(
                    {
                        "cell_id": (
                            f"v2_ema_e{epoch:02d}_{variant}_{mode}"
                            + ("_alpha_0p25" if alpha is not None else "")
                        ),
                        "source_scope": (
                            "strict_epoch3" if epoch == 3 else "original_epoch4_10"
                        ),
                        "source_state": "/server/state.json",
                        "source_status": "SUCCEEDED",
                        "epoch": epoch,
                        "variant": variant,
                        "method": f"actor_free_td_lewm_v2_ema_sg_{variant}",
                        "score_mode": mode,
                        "g_first_weight": alpha,
                        "output_dir": f"/server/{label}",
                        "outcomes": [False] * 49 + [True],
                        "success_count": 1,
                        "success_rate": 0.02,
                        "success_rate_percent": 2.0,
                        "checkpoint_epoch": epoch,
                        "checkpoint_global_step": epoch * results.STEPS_PER_EPOCH,
                        "checkpoint_path": f"/server/ema/{variant}/epoch_{epoch:02d}.pt",
                        "checkpoint_sha256": checkpoint,
                        "training_manifest_path": "/server/training_manifest.json",
                        "training_manifest_sha256": _sha(f"{label}:training"),
                        "evaluation_commit": results.EMA_NEW_EVALUATION_COMMIT,
                        "selection_sha256": results.SELECTION_SHA256,
                        "action_normalization_sha256": (
                            results.ACTION_NORMALIZATION_SHA256
                        ),
                        "source_files_sha256": _source_hashes(label),
                    }
                )

    fixed_references: dict[str, dict[str, dict[str, Any]]] = {}
    fixed_cells: list[dict[str, Any]] = []
    for version in ("v0", "v1", "v2"):
        fixed_references[version] = {}
        for variant in results.VARIANTS:
            checkpoint = _sha(f"{version}:{variant}:checkpoint")
            checkpoint_path = f"/server/{version}/{variant}/epoch_10.pt"
            fixed_references[version][variant] = {
                "checkpoint_sha256": checkpoint,
                "checkpoint_path": checkpoint_path,
                "epoch": 10,
                "global_step": 10 * results.STEPS_PER_EPOCH,
                "training_commit": _sha(f"{version}:training")[:40],
            }
            for mode in results.NEW_SCORE_MODES:
                alpha = results.G_FIRST_WEIGHT if mode == "f_plus_g_first" else None
                label = f"fixed:{version}:{variant}:{mode}"
                fixed_cells.append(
                    {
                        "version": version,
                        "variant": variant,
                        "score_mode": mode,
                        "g_first_weight": alpha,
                        "job_id": label,
                        "method": f"actor_free_td_lewm_{version}_{variant}",
                        "outcomes": [False] * 48 + [True, True],
                        "success_count": 2,
                        "success_rate": 0.04,
                        "success_rate_percent": 4.0,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": checkpoint,
                        "selection_sha256": results.FIXED_SELECTION_RANKS_SHA256,
                        "episode_selection_file_sha256": results.SELECTION_SHA256,
                        "action_normalization_sha256": (
                            results.ACTION_NORMALIZATION_SHA256
                        ),
                        "evaluation_commit": results.EMA_NEW_EVALUATION_COMMIT,
                        "evaluation_commit_evidence": {"source": "test"},
                        "source_launcher_manifest": "/server/launcher.json",
                        "output_dir": f"/server/{label}",
                        "source_files_sha256": _source_hashes(label),
                    }
                )
    ledger = {
        "schema_version": 1,
        "cell_count": 96,
        "epochs": list(results.EPOCHS),
        "variants": list(results.VARIANTS),
        "score_modes": list(results.NEW_SCORE_MODES),
        "g_first_weight": results.G_FIRST_WEIGHT,
        "selection_sha256": results.SELECTION_SHA256,
        "action_normalization_sha256": results.ACTION_NORMALIZATION_SHA256,
        "evaluation_commit": results.EMA_NEW_EVALUATION_COMMIT,
        "training_commit": results.EMA_TRAINING_COMMIT,
        "cells": cells,
        "fixed_checkpoint_comparison": {
            "included": True,
            "cell_count": results.FIXED_NEW_SCORE_CELL_COUNT,
            "selection_sha256": results.FIXED_SELECTION_RANKS_SHA256,
            "episode_selection_file_sha256": results.SELECTION_SHA256,
            "action_normalization_sha256": results.ACTION_NORMALIZATION_SHA256,
            "evaluation_commit": results.EMA_NEW_EVALUATION_COMMIT,
            "cells": fixed_cells,
        },
    }
    return ledger, fixed_references, ema_references


def _write_prior_ledger(tmp_path: Path, ledger: dict[str, Any]) -> tuple[Path, str]:
    path = tmp_path / "reconciliation_ledger.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return path, _sha_file(path)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prior_ledger_reconciles_exact_96_plus_36(tmp_path: Path) -> None:
    ledger, fixed_references, ema_references = _prior_ledger()
    path, ledger_sha = _write_prior_ledger(tmp_path, ledger)

    cells, evidence = results._validate_ema_new_ledger(
        ledger_path=path,
        fixed_references=fixed_references,
        ema_references=ema_references,
        expected_ledger_sha256=ledger_sha,
    )

    assert len(cells) == 132
    assert (
        evidence["ema_epoch_sweep_cell_count"] == results.EMA_SWEEP_NEW_SCORE_CELL_COUNT
    )
    assert evidence["fixed_checkpoint_cell_count"] == results.FIXED_NEW_SCORE_CELL_COUNT
    assert {cell["status"] for cell in cells} == {"VERIFIED"}
    fixed = [
        cell
        for cell in cells
        if cell["comparison_role"] == "fixed_checkpoint_new_scores"
    ]
    assert len(fixed) == results.FIXED_NEW_SCORE_CELL_COUNT
    assert {cell["epoch"] for cell in fixed} == {10}
    assert {cell["global_step"] for cell in fixed} == {10 * results.STEPS_PER_EPOCH}
    assert {cell["planning_horizon"] for cell in cells} == {None}
    fixed_mean_q = [
        cell
        for cell in cells
        if cell["epoch"] == 10 and cell["score_mode"] == "g_only_f_rollout_mean"
    ]
    assert len(fixed_mean_q) == results.FIXED_MEAN_Q_CELL_COUNT
    assert {cell["selection_sha256"] for cell in fixed_mean_q} == {
        results.SELECTION_SHA256
    }
    assert {
        cell["evidence"]["valid_row_ranks_sha256"]
        for cell in fixed_mean_q
        if cell["version"] in ("v0", "v1", "v2")
    } == {results.FIXED_SELECTION_RANKS_SHA256}
    assert {(cell["version"], cell["variant"]) for cell in fixed_mean_q} == {
        (version, variant)
        for version in ("v0", "v1", "v2", "v2_ema_sg")
        for variant in results.VARIANTS
    }


def test_prior_ledger_rejects_missing_v0_mean_q(tmp_path: Path) -> None:
    ledger, fixed_references, ema_references = _prior_ledger()
    fixed = ledger["fixed_checkpoint_comparison"]
    fixed["cells"] = [
        cell
        for cell in fixed["cells"]
        if not (
            cell["version"] == "v0"
            and cell["variant"] == "c"
            and cell["score_mode"] == "g_only_f_rollout_mean"
        )
    ]
    fixed["cell_count"] = len(fixed["cells"])
    path, ledger_sha = _write_prior_ledger(tmp_path, ledger)

    with pytest.raises(results.CompleteReconciliationError, match="cell_count"):
        results._validate_ema_new_ledger(
            ledger_path=path,
            fixed_references=fixed_references,
            ema_references=ema_references,
            expected_ledger_sha256=ledger_sha,
        )


def test_fixed_mean_q_rejects_raw_selection_file_hash_drift(tmp_path: Path) -> None:
    ledger, fixed_references, ema_references = _prior_ledger()
    target = next(
        cell
        for cell in ledger["fixed_checkpoint_comparison"]["cells"]
        if cell["version"] == "v0"
        and cell["variant"] == "c"
        and cell["score_mode"] == "g_only_f_rollout_mean"
    )
    assert target["selection_sha256"] == results.FIXED_SELECTION_RANKS_SHA256
    target["source_files_sha256"]["episode_selection.json"] = _sha(
        "wrong episode selection file"
    )
    path, ledger_sha = _write_prior_ledger(tmp_path, ledger)

    with pytest.raises(
        results.CompleteReconciliationError, match="selection_file_sha256"
    ):
        results._validate_ema_new_ledger(
            ledger_path=path,
            fixed_references=fixed_references,
            ema_references=ema_references,
            expected_ledger_sha256=ledger_sha,
        )


@pytest.mark.parametrize("mutation", ["integer_outcome", "checkpoint", "duplicate"])
def test_prior_ledger_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    ledger, fixed_references, ema_references = _prior_ledger()
    if mutation == "integer_outcome":
        ledger["cells"][0]["outcomes"][0] = 0
    elif mutation == "checkpoint":
        ledger["cells"][0]["checkpoint_sha256"] = _sha("wrong")
    else:
        ledger["cells"][1] = dict(ledger["cells"][0])
    path, ledger_sha = _write_prior_ledger(tmp_path, ledger)

    with pytest.raises(results.CompleteReconciliationError):
        results._validate_ema_new_ledger(
            ledger_path=path,
            fixed_references=fixed_references,
            ema_references=ema_references,
            expected_ledger_sha256=ledger_sha,
        )


def _fake_sweep_cell(kwargs: dict[str, Any]) -> dict[str, Any]:
    epoch = kwargs["expected_epoch"]
    variant = kwargs["variant"]
    mode = kwargs["score_mode"]
    checkpoint = _sha(f"checkpoint:{epoch}:{variant}")
    return {
        "cell_id": f"fake:{epoch}:{variant}:{mode}",
        "epoch": epoch,
        "variant": variant,
        "score_mode": mode,
        "checkpoint_sha256": checkpoint,
        "global_step": epoch * results.STEPS_PER_EPOCH,
        "outcomes": [False] * results.EPISODE_COUNT,
        "evidence": {
            "checkpoint": {
                "path": f"/server/{variant}/epoch_{epoch:02d}.pt",
            }
        },
        "source_kind": kwargs["source_kind"],
    }


def _materialize_ema_result_grid(root: Path) -> None:
    for epoch in results.EPOCHS:
        for variant in results.VARIANTS:
            for mode in results.ORIGINAL_SCORE_MODES:
                run = results._raw_sweep_path(
                    root,
                    version="v2_ema_sg",
                    epoch=epoch,
                    variant=variant,
                    score_mode=mode,
                )
                run.mkdir(parents=True, exist_ok=True)
                (run / "results.json").write_text("{}\n")


def test_ema_original_sweep_uses_only_attempt_02_retry_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ema"
    _materialize_ema_result_grid(root)
    summary = {
        "methods": {
            variant: {
                "evaluations": {mode: {} for mode in results.ORIGINAL_SCORE_MODES}
            }
            for variant in results.VARIANTS
        }
    }
    summary_bytes = (json.dumps(summary) + "\n").encode()
    paired = {
        f"success_{variant}__{mode}": tuple([False] * results.EPISODE_COUNT)
        for variant in results.VARIANTS
        for mode in results.ORIGINAL_SCORE_MODES
    }
    monkeypatch.setattr(
        results, "_require_locked_file", lambda *args, **kwargs: summary_bytes
    )
    monkeypatch.setattr(
        results, "_read_paired_outcomes", lambda *args, **kwargs: paired
    )
    seen: list[Path] = []

    def fake_validate(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["run_dir"])
        return _fake_sweep_cell(kwargs)

    monkeypatch.setattr(results, "_validate_raw_evaluation", fake_validate)
    cells, _, _ = results._validate_original_sweep(
        root=root,
        version="v2_ema_sg",
        family="actor_free_td_lewm_v2_ema_sg",
        training_identities=None,
        ema_summary_path=tmp_path / "summary.json",
        ema_paired_path=tmp_path / "paired.csv",
    )

    assert len(cells) == 144
    assert (root / "evaluation_retry_18cd574/epoch_03/g1/f_plus_g/attempt_02") in seen
    assert (root / "evaluation_retry_18cd574/epoch_03/g2/f_only/attempt_02") in seen
    assert sum(cell["source_kind"] == "raw_retry_attempt_02" for cell in cells) == 2

    duplicate = root / "evaluation_sweeps/epoch_03/g1/f_plus_g/results.json"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_text("{}\n")
    with pytest.raises(results.CompleteReconciliationError, match="grid differs"):
        results._validate_original_sweep(
            root=root,
            version="v2_ema_sg",
            family="actor_free_td_lewm_v2_ema_sg",
            training_identities=None,
            ema_summary_path=tmp_path / "summary.json",
            ema_paired_path=tmp_path / "paired.csv",
        )


def _synthetic_study() -> results.ValidatedCompleteStudy:
    cells = []
    for index in range(results.COMPLETE_CELL_COUNT):
        outcome = [False] * 49 + [index % 2 == 0]
        count = sum(outcome)
        cells.append(
            {
                **{field: "" for field in results.CSV_FIELDS},
                "cell_id": f"cell-{index:03d}",
                "family": "synthetic",
                "version": "legacy",
                "method_id": "synthetic",
                "method_label": "Synthetic",
                "variant": "synthetic",
                "epoch": 10,
                "global_step": 127_960,
                "checkpoint_sha256": _sha(f"checkpoint:{index}"),
                "score_mode": "f_only",
                "score_label": "F-only",
                "success_count": count,
                "episode_count": 50,
                "success_rate": count / 50,
                "success_rate_percent": count * 2.0,
                "training_seed": 3072,
                "planning_seed": 42,
                "selection_sha256": results.SELECTION_SHA256,
                "action_normalization_sha256": results.ACTION_NORMALIZATION_SHA256,
                "training_commit": "abcdef0",
                "evaluation_commit": "abcdef0",
                "planning_horizon": 5,
                "g_first_weight": None,
                "status": "VERIFIED",
                "comparison_role": "synthetic_scope",
                "source_kind": "synthetic",
                "source_path": "/synthetic",
                "source_results_sha256": _sha(f"results:{index}"),
                "source_protocol_sha256": _sha(f"protocol:{index}"),
                "outcomes_sha256": results._outcomes_sha256(outcome),
                "outcomes": outcome,
                "evidence": {},
            }
        )
    training_rows = tuple(
        {
            "variant": "c",
            "method_id": "method",
            "method_label": "Method",
            "epoch": index + 1,
            "global_step": (index + 1) * results.STEPS_PER_EPOCH,
            "train/loss_epoch": float(index),
            "validation/loss": float(index + 1),
        }
        for index in range(60)
    )
    return results.ValidatedCompleteStudy(
        cells=tuple(cells),
        sources={},
        v0_training_rows=training_rows,
        v0_training_columns=("train/loss_epoch", "validation/loss"),
        v2_training_rows=training_rows,
        v2_training_columns=("train/loss_epoch", "validation/loss"),
        training_curve_sources={"v0": {}, "v2": {}},
    )


def test_archive_writes_exact_schema_and_byte_checks(tmp_path: Path) -> None:
    study = _synthetic_study()
    paths = results.write_archive(study, artifact_dir=tmp_path)

    assert {path.name for path in paths} == set(results.ARCHIVE_FILENAMES)
    csv_rows = list(
        csv.DictReader(io.StringIO((tmp_path / "all_o50_results.csv").read_text()))
    )
    assert len(csv_rows) == results.COMPLETE_CELL_COUNT
    assert tuple(csv_rows[0]) == results.CSV_FIELDS
    assert {row["status"] for row in csv_rows} == {"VERIFIED"}
    ledger = json.loads((tmp_path / "reconciliation_ledger.json").read_text())
    assert ledger["cell_count"] == results.COMPLETE_CELL_COUNT
    assert ledger["outcome_count"] == results.COMPLETE_OUTCOME_COUNT
    results.write_archive(study, artifact_dir=tmp_path, check=True)
    (tmp_path / "all_o50_results.csv").write_text("tampered\n")
    with pytest.raises(results.CompleteReconciliationError, match="differs"):
        results.write_archive(study, artifact_dir=tmp_path, check=True)


def _real_source_arguments() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    project = repository.parents[1]
    artifacts = repository / "reports/artifacts"
    ema = artifacts / "actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072"
    temporary = Path("/private/tmp/td-results-full.y1qTdB")
    return {
        "legacy_bundle_root": (
            project / "artifacts/actor_free_td_lewm_final_lightweight_bundle_fa46ed9"
        ),
        "legacy_summary": artifacts / "actor_free_td_lewm_cube_seed3072/summary.json",
        "legacy_paired_outcomes": (
            artifacts / "actor_free_td_lewm_cube_seed3072/paired_outcomes.csv"
        ),
        "v0_root": temporary / "v0",
        "v0_summary": (
            artifacts / "actor_free_td_lewm_v0_cube_seed3072/formal_o50_summary.json"
        ),
        "v0_training_root": temporary / "v0/training_metadata",
        "v1_bundle_root": project / "tmp/v1-bundle-3c4e62e",
        "v1_summary": artifacts / "actor_free_td_lewm_v1_cube_seed3072/summary.json",
        "v1_paired_outcomes": (
            artifacts / "actor_free_td_lewm_v1_cube_seed3072/paired_outcomes.csv"
        ),
        "ema_new_ledger": ema / "new_scores/reconciliation_ledger.json",
        "v2_root": temporary / "v2",
        "v2_training_root": temporary / "v2/formal_metadata",
        "v2_ema_root": temporary / "v2_ema",
        "v2_ema_epoch10_summary": ema / "original_scores/summary.json",
        "v2_ema_epoch10_paired_outcomes": ema / "original_scores/paired_outcomes.csv",
    }


def test_real_sources_reconcile_to_exact_complete_ledger() -> None:
    arguments = _real_source_arguments()
    if not all(path.exists() for path in arguments.values()):
        pytest.skip("external read-only result evidence is not present")
    prior_ledger = json.loads(arguments["ema_new_ledger"].read_text())
    fixed = prior_ledger.get("fixed_checkpoint_comparison", {})
    if fixed.get("cell_count") != results.FIXED_NEW_SCORE_CELL_COUNT:
        pytest.skip("477-cell formal input is not present yet")
    arguments["ema_new_ledger_sha256"] = _sha_file(arguments["ema_new_ledger"])

    study = results.reconcile_complete_o50(**arguments)

    assert len(study.cells) == results.COMPLETE_CELL_COUNT
    assert sum(len(cell["outcomes"]) for cell in study.cells) == (
        results.COMPLETE_OUTCOME_COUNT
    )
    assert {cell["status"] for cell in study.cells} == {"VERIFIED"}
    assert len(study.v0_training_rows) == 60
    assert len(study.v2_training_rows) == 60
    retries = [
        cell for cell in study.cells if cell["source_kind"] == "raw_retry_attempt_02"
    ]
    assert {
        (cell["epoch"], cell["variant"], cell["score_mode"]) for cell in retries
    } == set(results.RETRY_PATHS)
