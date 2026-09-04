from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tdwm.results import actor_free_td_lewm_v2_ema_new_scores as results


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _selection() -> dict[str, list[int]]:
    ranks = np.sort(np.random.default_rng(42).choice(1_510_000 - 1, 50, replace=False))
    start = ranks % 151
    return {
        "episode_indices": (ranks // 151).tolist(),
        "start_steps": start.tolist(),
        "goal_steps": (start + 50).tolist(),
        "valid_row_ranks": ranks.tolist(),
    }


def _training_manifest(formal_root: Path, variant: str) -> Path:
    sweeps = results._sweeps
    path = sweeps.training_manifest_path(formal_root, variant)
    protocol = {
        "method": f"{sweeps.METHOD_FAMILY}_{variant}",
        "method_family": sweeps.METHOD_FAMILY,
        "variant": variant,
        "implementation_version": sweeps.IMPLEMENTATION_VERSION,
        "stage": "coupled_hybrid_ema_target_finetuning",
        "initialization": "corresponding_v1_deployment_finetune",
    }
    _write_json(
        path,
        {
            "method": f"{sweeps.METHOD_FAMILY}_{variant}",
            "method_family": sweeps.METHOD_FAMILY,
            "variant": variant,
            "implementation_version": sweeps.IMPLEMENTATION_VERSION,
            "objective_version": 0,
            "deployment_checkpoint_version": 1,
            "stage": "coupled_hybrid_ema_target_finetuning",
            "initialization": "corresponding_v1_deployment_finetune",
            "seed": sweeps.SEED,
            "protocol": protocol,
            "protocol_sha256": sweeps._base.canonical_json_sha256(protocol),
            "runtime": {"tdwm_git_revision": results.TRAINING_COMMIT},
        },
    )
    return path


def _argument(cell: object, option: str) -> str | None:
    argv = list(cell.argv)
    if option not in argv:
        return None
    return argv[argv.index(option) + 1]


def _write_output(
    cell: object,
    *,
    training_manifest: Path,
    selection: dict[str, list[int]],
    action: dict[str, object],
) -> None:
    sweeps = results._sweeps
    checkpoint = Path(cell.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(f"checkpoint-{cell.epoch}-{cell.variant}".encode())
    checkpoint_sha = sweeps._base.file_sha256(checkpoint)
    training_sha = sweeps._base.file_sha256(training_manifest)
    method = f"{sweeps.METHOD_FAMILY}_{cell.variant}"
    identity = {
        "version_key": sweeps.VERSION_KEY,
        "version_display_name": sweeps.VERSION_DISPLAY_NAME,
        "training_commit": results.TRAINING_COMMIT,
        "method": method,
        "epoch": cell.epoch,
        "checkpoint_epoch": cell.epoch,
        "checkpoint_sha256": checkpoint_sha,
        "training_manifest_path": str(training_manifest.resolve()),
        "training_manifest_sha256": training_sha,
        "evaluation_commit": results.EVALUATION_COMMIT,
    }
    inference: dict[str, object] = {"score_mode": cell.score_mode}
    if cell.score_mode == sweeps.FIRST_ACTION_SCORE_MODE:
        weight = float(str(_argument(cell, "--g-first-weight")))
        inference.update(
            {
                "g_first_weight": weight,
                "score_definition": sweeps.FIRST_ACTION_SCORE_DEFINITION,
            }
        )
        score_metadata = {
            "g_first_weight": weight,
            "planning": {"horizon": 5},
            "score_definition": sweeps.FIRST_ACTION_SCORE_DEFINITION,
        }
    else:
        inference.update(
            {
                "f_goal_distance_used": False,
                "f_transition_used": True,
                "g_aggregation": "mean_over_5_blocks",
                "rollout_horizon": 5,
                "score_definition": sweeps.ROLLOUT_MEAN_SCORE_DEFINITION,
            }
        )
        score_metadata = {
            "g_aggregation": "mean_over_5_blocks",
            "state_source_for_q1": "current_online_encoder_state",
            "state_source_for_q2_to_q5": "online_lewm_rollout_predicted_states",
            "f_goal_distance_used": False,
            "f_transition_used": True,
            "planning_horizon": 5,
            "rollout_horizon": 5,
            "executed_action_block": "first_block_only",
            "replanning": "every_action_block",
            "score_definition": sweeps.ROLLOUT_MEAN_SCORE_DEFINITION,
        }
    successes = [
        index < (cell.epoch + results.VARIANTS.index(cell.variant))
        for index in range(50)
    ]
    output = Path(cell.output_dir)
    common = {
        "method": method,
        "method_family": sweeps.METHOD_FAMILY,
        "variant": cell.variant,
        "score_mode": cell.score_mode,
        "smoke": False,
        "pilot": False,
        **identity,
        **score_metadata,
    }
    _write_json(
        output / "results.json",
        {
            **common,
            "metrics": {
                "episode_successes": successes,
                "success_rate": 100.0 * sum(successes) / 50,
            },
        },
    )
    _write_json(
        output / "protocol_manifest.json",
        {
            **common,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "method": method,
                "method_family": sweeps.METHOD_FAMILY,
                "variant": cell.variant,
                "implementation_version": sweeps.IMPLEMENTATION_VERSION,
                "epoch": cell.epoch,
                "global_step": results.STEPS_PER_EPOCH * cell.epoch,
            },
            "protocol": {
                "planning": {"horizon": 5},
                "inference_objective": inference,
            },
            "runtime": {"tdwm_git_revision": results.EVALUATION_COMMIT},
            "selection": selection,
            "normalization": {"action": action},
        },
    )
    _write_json(output / "episode_selection.json", selection)
    _write_json(output / "action_normalization.json", action)


def _make_scope(
    root: Path,
    *,
    formal_root: Path,
    epochs: tuple[int, ...],
    status: str,
) -> Path:
    sweeps = results._sweeps
    paths = sweeps.SweepPaths(
        repository=Path(__file__).resolve().parents[2],
        dataset=root.parent / "cube.lance",
        formal_root=formal_root,
        bundle_root=root,
        sweep_root=root / "new_score_evaluation_sweeps",
        launcher_root=root / "new_score_evaluation_sweep_launcher",
    )
    manifests = {
        variant: _training_manifest(formal_root, variant)
        for variant in results.VARIANTS
    }
    selection = _selection()
    reference = root / "selection-reference.json"
    _write_json(reference, selection)
    assert results._file_sha256(reference) == results.SELECTION_SHA256
    reference.unlink()
    action = {
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "variance": [1.0] * 5,
        "samples": 2_010_000,
    }
    all_cells = sweeps.build_cells(
        paths=paths,
        python="/venv/bin/python",
        g_first_weight=results.FIRST_ACTION_WEIGHT,
    )
    cells: dict[str, dict[str, object]] = {}
    sweeps._EXPECTED_EVALUATION_COMMIT = results.EVALUATION_COMMIT
    for cell in all_cells:
        if cell.epoch not in epochs:
            continue
        _write_output(
            cell,
            training_manifest=manifests[cell.variant],
            selection=selection,
            action=action,
        )
        output_audit = sweeps.audit_complete_output(cell)
        entry = {
            **asdict(cell),
            "argv": list(cell.argv),
            "argv_sha256": sweeps._base.canonical_json_sha256(list(cell.argv)),
            "state": status,
            "output_audit": output_audit,
        }
        if status == "SUCCEEDED":
            entry.update({"exit_code_marker": 0, "error": None})
        cells[cell.cell_id] = entry
    state_counts = {status: len(cells)}
    state = {
        "schema_version": 1,
        "source": "actor_free_td_lewm_v2_ema_new_score_sweep_scheduler",
        "version_key": "v2_ema",
        "version_display_name": "V2 EMA",
        "training_commit": results.TRAINING_COMMIT,
        "expected_evaluation_commit": results.EVALUATION_COMMIT,
        "expected_selection_sha256": results.SELECTION_SHA256,
        "score_modes": list(results.SCORE_MODES),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "cell_count": len(cells),
        "state_counts": state_counts,
        "cells": cells,
    }
    state_path = (
        root / "new_score_evaluation_sweep_launcher" / "runs" / "final" / "state.json"
    )
    _write_json(state_path, state)
    return state_path


def _make_split(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    original_root = tmp_path / "original"
    strict_root = tmp_path / "strict"
    formal_root = original_root / "formal"
    strict_state = _make_scope(
        strict_root,
        formal_root=formal_root,
        epochs=(3,),
        status="SUCCEEDED",
    )
    original_state = _make_scope(
        original_root,
        formal_root=formal_root,
        epochs=tuple(range(4, 11)),
        status="REUSED",
    )
    return strict_root, strict_state, original_root, original_state


def _reconcile(split: tuple[Path, Path, Path, Path]) -> results.ValidatedNewScoreStudy:
    strict_root, strict_state, original_root, original_state = split
    return results.reconcile_new_score_sweeps(
        strict_epoch3_root=strict_root,
        strict_epoch3_state=strict_state,
        original_epoch4_10_root=original_root,
        original_epoch4_10_state=original_state,
    )


def _write_fixed_job_output(job: object, selection: dict[str, list[int]]) -> None:
    fixed = results._fixed
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "method": f"actor_free_td_lewm_{job.version}_{job.variant}",
        "method_family": f"actor_free_td_lewm_{job.version}",
        "variant": job.variant,
        "implementation_version": job.version,
        "score_mode": job.score_mode,
        "planning_horizon": 5,
        "smoke": False,
        "pilot": False,
        "metrics": {"episode_successes": [False] * 50},
    }
    inference: dict[str, object] = {"score_mode": job.score_mode}
    manifest: dict[str, object] = {
        "score_mode": job.score_mode,
        "protocol": {
            "method": result["method"],
            "method_family": result["method_family"],
            "variant": job.variant,
            "implementation_version": job.version,
            "inference_objective": inference,
            "planning": {"horizon": 5},
        },
        "checkpoint": {"path": job.checkpoint},
        "selection": selection,
    }
    action = {
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "variance": [1.0] * 5,
        "samples": 2_010_000,
    }
    manifest["normalization"] = {"action": action}
    if job.score_mode == fixed.FIRST_ACTION_MODE:
        definition = {"formula": "f_cost - g_first_weight * q_first"}
        for value in (result, inference, manifest):
            value["g_first_weight"] = job.alpha
            value["score_definition"] = definition
    else:
        definition = {
            "formula": "mean(q_1, q_2, q_3, q_4, q_5)",
            "action_processing": fixed.ROLLOUT_MEAN_ACTION_PROCESSING_BY_VERSION[
                job.version
            ],
        }
        for value in (result, manifest):
            value.update(fixed.ROLLOUT_MEAN_METADATA_BY_VERSION[job.version])
            value["score_definition"] = definition
        inference.update(fixed.ROLLOUT_MEAN_INFERENCE_METADATA_BY_VERSION[job.version])
        inference["score_definition"] = definition
    _write_json(output / "results.json", result)
    _write_json(output / "protocol_manifest.json", manifest)
    _write_json(output / "episode_selection.json", selection)
    _write_json(output / "action_normalization.json", action)


def _make_evaluation_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "evaluation-checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    for version in results.FIXED_VERSIONS:
        for variant in results.VARIANTS:
            (
                scripts / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
            ).write_text("# evaluator fixture\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "scripts"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "fixture checkout"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, head


def _make_fixed_launchers(
    tmp_path: Path, *, evaluation_checkout: Path
) -> list[tuple[Path, Path]]:
    fixed = results._fixed
    selection = _selection()
    groups = (
        [
            (version, variant, mode)
            for version in results.FIXED_VERSIONS
            for variant in results.VARIANTS
            for mode in (results.SCORE_MODES[0],)
            if variant != "c"
        ]
        + [
            ("v2", variant, results.SCORE_MODES[1])
            for variant in results.VARIANTS
            if variant != "c"
        ],
        [(version, "c", results.SCORE_MODES[0]) for version in results.FIXED_VERSIONS],
        [("v2", "c", results.SCORE_MODES[1])],
    )
    sources: list[tuple[Path, Path]] = []
    for group_index, identities in enumerate(groups):
        root = tmp_path / f"fixed-{group_index}"
        jobs: dict[str, dict[str, object]] = {}
        for version, variant, mode in identities:
            checkpoint = tmp_path / "fixed-checkpoints" / version / f"{variant}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.exists():
                checkpoint.write_bytes(f"{version}-{variant}".encode())
            alpha = (
                results.FIRST_ACTION_WEIGHT if mode == results.SCORE_MODES[0] else None
            )
            suffix = "__alpha_0p25" if alpha is not None else ""
            job_id = f"{version}__{variant}__{mode}{suffix}"
            output = root / "formal" / version / variant / mode
            if alpha is not None:
                output /= "alpha_0p25"
            job = fixed.Job(
                job_id=job_id,
                stage="formal",
                version=version,
                variant=variant,
                score_mode=mode,
                alpha=alpha,
                checkpoint=str(checkpoint),
                config_path=str(tmp_path / "config.yaml"),
                output_dir=str(output),
                log_path=str(root / "formal/_launcher/jobs" / f"{job_id}.log"),
                argv=(
                    "python",
                    str(
                        evaluation_checkout
                        / "scripts"
                        / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
                    ),
                ),
            )
            _write_fixed_job_output(job, selection)
            evidence = fixed.validate_job_output(job)
            jobs[job_id] = {
                **asdict(job),
                "argv": list(job.argv),
                "state": "SUCCEEDED",
                "exit_code": 0,
                "evidence": evidence,
            }
        manifest = {
            "schema_version": 1,
            "launcher": "actor_free_td_lewm_first_action_comparison",
            "inference_only": True,
            "training_performed": False,
            "alpha_selection_performed": False,
            "stage": "formal",
            "status": "SUCCEEDED",
            "repository": str(evaluation_checkout),
            "jobs": jobs,
        }
        manifest_path = root / "formal/_launcher/launcher_manifest.json"
        _write_json(manifest_path, manifest)
        sources.append((root, manifest_path))
    return sources


def _prepare_fixed_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[Path, Path, Path, Path], list[tuple[Path, Path]], Path, str]:
    checkout, head = _make_evaluation_checkout(tmp_path)
    monkeypatch.setattr(results, "EVALUATION_COMMIT", head)
    monkeypatch.setattr(
        results,
        "FIXED_ACTION_NORMALIZATION_SHA256",
        "16a3af277ac5a389dffe9f4c9a95fc340c73a8df5ed08a4f581052f635c9455a",
    )
    split = _make_split(tmp_path)
    fixed_launchers = _make_fixed_launchers(tmp_path, evaluation_checkout=checkout)
    return split, fixed_launchers, checkout, head


def test_reconciles_exact_12_plus_84_and_writes_lightweight_archive(
    tmp_path: Path,
) -> None:
    study = _reconcile(_make_split(tmp_path))
    assert len(study.cells) == 96
    assert {cell["source_scope"] for cell in study.cells if cell["epoch"] == 3} == {
        "strict_epoch3"
    }
    assert {cell["source_scope"] for cell in study.cells if cell["epoch"] >= 4} == {
        "original_epoch4_10"
    }
    artifact_dir = tmp_path / "archive"
    paths = results.write_archive(study, artifact_dir=artifact_dir)
    assert {path.name for path in paths} == set(results.ARCHIVE_FILENAMES)
    with (artifact_dir / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    assert rows[0]["epoch"] == "3"
    assert rows[-1]["epoch"] == "10"
    ledger = json.loads((artifact_dir / "reconciliation_ledger.json").read_text())
    assert ledger["sources"]["strict_epoch3"]["cell_count"] == 12
    assert ledger["sources"]["original_epoch4_10"]["cell_count"] == 84
    assert results.write_archive(study, artifact_dir=artifact_dir, check=True) == paths


def test_optionally_reconciles_exact_three_launcher_fixed_24_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert results.FIXED_ACTION_NORMALIZATION_SHA256 == (
        "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
    )
    assert results.EVALUATION_COMMIT == ("5456f3d18116812d078d4ec2e85ba1f83d89c7c7")
    split, fixed_launchers, checkout, head = _prepare_fixed_reconciliation(
        tmp_path, monkeypatch
    )
    study = results.reconcile_new_score_sweeps(
        strict_epoch3_root=split[0],
        strict_epoch3_state=split[1],
        original_epoch4_10_root=split[2],
        original_epoch4_10_state=split[3],
        fixed_launchers=fixed_launchers,
        fixed_evaluation_checkout=checkout,
    )
    assert len(study.fixed_launcher_sources) == 3
    assert len(study.fixed_cells) == 24
    assert {cell["selection_sha256"] for cell in study.fixed_cells} == {
        results.FIXED_SELECTION_SHA256
    }
    assert {cell["evaluation_commit"] for cell in study.fixed_cells} == {head}
    assert {
        cell["evaluation_commit_evidence"]["result_field"] for cell in study.fixed_cells
    } == {"absent"}
    assert {
        cell["evaluation_commit_evidence"]["manifest_field"]
        for cell in study.fixed_cells
    } == {"absent"}
    assert {
        cell["evaluation_commit_evidence"]["source"] for cell in study.fixed_cells
    } == {"launcher_repository_checkout_head"}
    artifact_dir = tmp_path / "fixed-archive"
    results.write_archive(study, artifact_dir=artifact_dir)
    with (artifact_dir / "fixed_checkpoint_results.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 24


def _run_fixed_reconciliation(
    split: tuple[Path, Path, Path, Path],
    fixed_launchers: list[tuple[Path, Path]],
    checkout: Path,
) -> results.ValidatedNewScoreStudy:
    return results.reconcile_new_score_sweeps(
        strict_epoch3_root=split[0],
        strict_epoch3_state=split[1],
        original_epoch4_10_root=split[2],
        original_epoch4_10_state=split[3],
        fixed_launchers=fixed_launchers,
        fixed_evaluation_checkout=checkout,
    )


def test_fixed_optional_evaluation_commit_must_match_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split, fixed_launchers, checkout, _ = _prepare_fixed_reconciliation(
        tmp_path, monkeypatch
    )
    launcher = json.loads(fixed_launchers[0][1].read_text())
    first_job = next(iter(launcher["jobs"].values()))
    result_path = Path(first_job["output_dir"]) / "results.json"
    result = json.loads(result_path.read_text())
    result["evaluation_commit"] = "f" * 40
    _write_json(result_path, result)
    with pytest.raises(
        results.NewScoreReconciliationError,
        match="optional results.evaluation_commit conflicts",
    ):
        _run_fixed_reconciliation(split, fixed_launchers, checkout)


def test_fixed_checkout_head_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split, fixed_launchers, checkout, _ = _prepare_fixed_reconciliation(
        tmp_path, monkeypatch
    )
    (checkout / "HEAD_CHANGED").write_text("changed\n")
    subprocess.run(["git", "-C", str(checkout), "add", "HEAD_CHANGED"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "change head"],
        check=True,
    )
    with pytest.raises(
        results.NewScoreReconciliationError,
        match="is not the locked commit",
    ):
        _run_fixed_reconciliation(split, fixed_launchers, checkout)


def test_fixed_launcher_repository_or_evaluator_argv_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split, fixed_launchers, checkout, _ = _prepare_fixed_reconciliation(
        tmp_path, monkeypatch
    )
    manifest_path = fixed_launchers[0][1]
    manifest = json.loads(manifest_path.read_text())
    manifest["repository"] = str(checkout.parent)
    _write_json(manifest_path, manifest)
    with pytest.raises(
        results.NewScoreReconciliationError,
        match="repository does not bind",
    ):
        _run_fixed_reconciliation(split, fixed_launchers, checkout)

    manifest["repository"] = str(checkout)
    first_job = next(iter(manifest["jobs"].values()))
    first_job["argv"][1] = str(tmp_path / "outside_evaluator.py")
    _write_json(manifest_path, manifest)
    with pytest.raises(
        results.NewScoreReconciliationError,
        match="argv evaluator is not",
    ):
        _run_fixed_reconciliation(split, fixed_launchers, checkout)


def test_rejects_old_epoch3_cell_in_original_state(tmp_path: Path) -> None:
    split = _make_split(tmp_path)
    original_state = split[3]
    state = json.loads(original_state.read_text())
    first_id = next(iter(state["cells"]))
    stale_id = first_id.replace("e04", "e03", 1)
    state["cells"][stale_id] = state["cells"].pop(first_id)
    _write_json(original_state, state)
    with pytest.raises(results.NewScoreReconciliationError, match="grid mismatch"):
        _reconcile(split)


def test_rejects_non_boolean_outcome_even_when_length_is_fifty(
    tmp_path: Path,
) -> None:
    split = _make_split(tmp_path)
    strict_state = json.loads(split[1].read_text())
    first_cell = next(iter(strict_state["cells"].values()))
    result_path = Path(first_cell["output_dir"]) / "results.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["episode_successes"][0] = 1
    _write_json(result_path, result)
    with pytest.raises(results.NewScoreReconciliationError, match="50 booleans"):
        _reconcile(split)


def test_rejects_training_manifest_changed_after_evaluation(tmp_path: Path) -> None:
    split = _make_split(tmp_path)
    manifest = results._sweeps.training_manifest_path(split[2] / "formal", "c")
    value = json.loads(manifest.read_text())
    value["post_evaluation_mutation"] = True
    _write_json(manifest, value)
    with pytest.raises(
        results.NewScoreReconciliationError, match="training_manifest_sha256"
    ):
        _reconcile(split)


def test_rejects_stale_scheduler_output_audit(tmp_path: Path) -> None:
    split = _make_split(tmp_path)
    state = json.loads(split[1].read_text())
    first_cell = next(iter(state["cells"].values()))
    first_cell["output_audit"]["results.json"]["sha256"] = "0" * 64
    _write_json(split[1], state)
    with pytest.raises(
        results.NewScoreReconciliationError, match="output_audit is stale"
    ):
        _reconcile(split)
