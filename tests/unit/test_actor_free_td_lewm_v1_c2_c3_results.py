from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tdwm.results import actor_free_td_lewm_v1_c2_c3 as results


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_checkout(path: Path) -> tuple[Path, str]:
    path.mkdir(parents=True)
    (path / "README.md").write_text("evaluation fixture\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, head


def _planning(raw_score_mode: str) -> dict[str, Any]:
    return {
        "solver": "CEM",
        "horizon": 1 if raw_score_mode == "g_only" else 5,
        "candidates": 300,
        "iterations": 30,
        "elites": 30,
        "initial_variance": 1.0,
        "action_block": 5,
        "frame_skip": 5,
        "receding_horizon": 1,
        "episode_budget": 100,
        "planning_seed": 42,
        "solver_batch_size": 1,
        "warm_start": True,
        "initial_distribution": "cem_gaussian_no_actor",
    }


def _parent_source(parent_sha: str) -> dict[str, Any]:
    return {
        "checkpoint_sha256": parent_sha,
        "source_seed": 3072,
        "source_epoch": 10,
        "source_global_step": 127_960,
    }


def _write_output(
    *,
    output: Path,
    method_key: str,
    raw_score_mode: str,
    checkpoint: Path,
    checkpoint_sha: str,
    parent_sha: str,
    selection: dict[str, Any],
    action: dict[str, Any],
    evaluation_commit: str,
    successes: int,
) -> None:
    variant = method_key.removeprefix("v1_")
    method = f"actor_free_td_lewm_v1_{variant}"
    inference: dict[str, Any] = {"score_mode": raw_score_mode}
    result: dict[str, Any] = {
        "method": method,
        "method_family": "actor_free_td_lewm_v1",
        "variant": variant,
        "implementation_version": "v1",
        "score_mode": raw_score_mode,
        "planning_horizon": 1 if raw_score_mode == "g_only" else 5,
        "smoke": False,
        "pilot": False,
        "metrics": {
            "episode_successes": [True] * successes
            + [False] * (results.EPISODES - successes)
        },
    }
    manifest: dict[str, Any] = {"score_mode": raw_score_mode}
    if raw_score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}:
        definition = {"formula": "fixture first-Q score"}
        for value in (result, manifest, inference):
            value["g_first_weight"] = 0.25
            value["score_definition"] = definition
    if raw_score_mode == "state_v_terminal":
        definition = {
            "optimization": "cem_minimize",
            "critic": "ema_target_state_value",
        }
        result["score_definition"] = definition
        manifest["score_definition"] = definition
        inference["parent_g_used"] = False

    protocol: dict[str, Any] = {
        "method": method,
        "method_family": "actor_free_td_lewm_v1",
        "variant": variant,
        "implementation_version": "v1",
        "pretrained_world_model": {"source_seed": 3072},
        "evaluation": {"episodes": 50, "goal_offset": 50},
        "planning": _planning(raw_score_mode),
        "inference_objective": inference,
    }
    checkpoint_data: dict[str, Any] = {
        "path": str(checkpoint.resolve()),
        "sha256": checkpoint_sha,
        "epoch": 12 if method_key == "v1_c3" else 10,
        "global_step": 12_000 if method_key == "v1_c3" else 127_960,
        "formal_completion_required": True,
    }
    if method_key == "v1_c2":
        source = _parent_source(parent_sha)
        protocol["source_v1_c"] = source
        checkpoint_data["predictor_config"] = {"source_v1_c": source}
    elif method_key == "v1_c3":
        source = _parent_source(parent_sha)
        protocol["source_v1_c"] = source
        protocol["state_critic"] = {"architecture": "rp1_mrn_quasimetric"}
        checkpoint_data["source_v1_c_provenance"] = source
        checkpoint_data["logical_epoch"] = 12

    manifest.update(
        {
            "protocol": protocol,
            "checkpoint": checkpoint_data,
            "selection": selection,
            "normalization": {"action": action},
            "runtime": {"tdwm_git_revision": evaluation_commit},
        }
    )
    if raw_score_mode == "state_v_terminal":
        manifest["selection_sha256"] = results.SELECTION_SHA256
    _write_json(output / "results.json", result)
    _write_json(output / "protocol_manifest.json", manifest)
    _write_json(output / "episode_selection.json", selection)
    _write_json(output / "action_normalization.json", action)


@dataclass(frozen=True)
class Case:
    root: Path
    formal_root: Path
    launcher_manifest: Path
    c3_output: Path
    c_checkpoint: Path
    c2_checkpoint: Path
    c3_checkpoint: Path
    parent_sha: str
    c2_sha: str
    c3_sha: str


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    selection = {
        "episode_indices": list(range(50)),
        "start_steps": [0] * 50,
        "goal_steps": [50] * 50,
        "valid_row_ranks": list(range(50)),
    }
    action = {
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "variance": [1.0] * 5,
        "samples": 2_010_000,
    }
    hash_inputs = tmp_path / "hash-inputs"
    _write_json(hash_inputs / "episode_selection.json", selection)
    _write_json(hash_inputs / "action_normalization.json", action)
    selection_sha = _file_sha(hash_inputs / "episode_selection.json")
    action_sha = _file_sha(hash_inputs / "action_normalization.json")
    ranks_sha = results._canonical_json_sha256(selection["valid_row_ranks"])
    monkeypatch.setattr(results, "SELECTION_SHA256", selection_sha)
    monkeypatch.setattr(results, "SELECTION_RANKS_SHA256", ranks_sha)
    monkeypatch.setattr(results, "ACTION_NORMALIZATION_SHA256", action_sha)

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    c_checkpoint = checkpoints / "v1_c_epoch10.pt"
    c2_checkpoint = checkpoints / "v1_c2_epoch10.pt"
    c3_checkpoint = checkpoints / "v1_c3_epoch12.pt"
    c_checkpoint.write_bytes(b"v1-c-parent-checkpoint")
    c2_checkpoint.write_bytes(b"v1-c2-checkpoint")
    c3_checkpoint.write_bytes(b"v1-c3-checkpoint")
    parent_sha = _file_sha(c_checkpoint)
    c2_sha = _file_sha(c2_checkpoint)
    c3_sha = _file_sha(c3_checkpoint)
    monkeypatch.setattr(results, "PARENT_V1_C_CHECKPOINT_SHA256", parent_sha)

    repository, c2_commit = _git_checkout(tmp_path / "evaluation-checkout")
    c3_commit = "b" * 40
    root = tmp_path / "c2-launcher"
    formal_root = root / "formal"
    checkpoint_manifest = root / "checkpoint_manifest.json"
    _write_json(
        checkpoint_manifest,
        {
            "schema_version": 1,
            "purpose": "v1_c2_endpoint_o50_and_v1_c_first_q2_reference",
            "v1": {
                "c": {"path": str(c_checkpoint), "sha256": parent_sha},
                "c2": {"path": str(c2_checkpoint), "sha256": c2_sha},
            },
        },
    )

    jobs: dict[str, Any] = {}
    identities = results._expected_raw_identities()[:-1]
    for index, (method_key, raw_mode) in enumerate(identities):
        variant = method_key.removeprefix("v1_")
        checkpoint = c_checkpoint if variant == "c" else c2_checkpoint
        checkpoint_sha = parent_sha if variant == "c" else c2_sha
        relative = results._relative_c2_output(method_key, raw_mode)
        output = formal_root / relative
        _write_output(
            output=output,
            method_key=method_key,
            raw_score_mode=raw_mode,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            parent_sha=parent_sha,
            selection=selection,
            action=action,
            evaluation_commit=c2_commit,
            successes=index + 1,
        )
        job_id = results._cell_id(method_key, raw_mode)
        argv = [
            "python",
            str(
                repository / "scripts" / f"evaluate_actor_free_td_lewm_v1_{variant}.py"
            ),
            "--score-mode",
            raw_mode,
            "--checkpoint-path",
            str(checkpoint),
            "--output-dir",
            str(output),
        ]
        alpha = 0.25 if raw_mode in results._FIRST_ACTION_RAW_MODES else None
        if alpha is not None:
            argv.extend(["--g-first-weight", "0.25"])
        jobs[job_id] = {
            "job_id": job_id,
            "stage": "formal",
            "version": "v1",
            "variant": variant,
            "score_mode": raw_mode,
            "alpha": alpha,
            "checkpoint": str(checkpoint),
            "config_path": "/fixture/config.yaml",
            "output_dir": str(output),
            "log_path": "/fixture/job.log",
            "argv": argv,
            "state": "SUCCEEDED",
            "exit_code": 0,
            "evidence": {
                "selection_file_sha256": selection_sha,
                "valid_row_ranks_sha256": ranks_sha,
            },
        }

    launcher = {
        "schema_version": 1,
        "launcher": "actor_free_td_lewm_first_action_comparison",
        "inference_only": True,
        "training_performed": False,
        "alpha_selection_performed": False,
        "stage": "formal",
        "versions": ["v1"],
        "variants": ["c2", "c"],
        "shared_score_modes": list(results._RAW_C2_MODES),
        "v2_only_score_modes": [],
        "alphas": [0.25],
        "expected_selection_file_sha256": selection_sha,
        "status": "SUCCEEDED",
        "repository": str(repository),
        "checkpoint_manifest": str(checkpoint_manifest),
        "output_root": str(root),
        "selection": {
            "selection_file_sha256": selection_sha,
            "valid_row_ranks_sha256": ranks_sha,
            "identical_across_all_jobs": True,
            "selection_file_identical_across_all_jobs": True,
        },
        "jobs": jobs,
    }
    launcher_manifest = formal_root / "_launcher/launcher_manifest.json"
    _write_json(launcher_manifest, launcher)

    c3_output = tmp_path / "c3-output"
    _write_output(
        output=c3_output,
        method_key="v1_c3",
        raw_score_mode="state_v_terminal",
        checkpoint=c3_checkpoint,
        checkpoint_sha=c3_sha,
        parent_sha=parent_sha,
        selection=selection,
        action=action,
        evaluation_commit=c3_commit,
        successes=8,
    )
    return Case(
        root=root,
        formal_root=formal_root,
        launcher_manifest=launcher_manifest,
        c3_output=c3_output,
        c_checkpoint=c_checkpoint,
        c2_checkpoint=c2_checkpoint,
        c3_checkpoint=c3_checkpoint,
        parent_sha=parent_sha,
        c2_sha=c2_sha,
        c3_sha=c3_sha,
    )


def _reconcile(case: Case) -> results.ValidatedEndpointStudy:
    return results.reconcile_endpoint_results(
        c2_launcher_root=case.root,
        c2_launcher_manifest=case.launcher_manifest,
        c3_output_dir=case.c3_output,
    )


def _write_training_run(
    run_dir: Path,
    *,
    method_key: str,
    checkpoint: Path,
    checkpoint_sha: str,
) -> None:
    variant = method_key.removeprefix("v1_")
    method = f"actor_free_td_lewm_v1_{variant}"
    training_result: dict[str, Any] = {
        "method": method,
        "variant": variant,
        "seed": 3072,
        "global_step": 12_000 if method_key == "v1_c3" else 127_960,
        "deployment_checkpoint": str(checkpoint),
    }
    if method_key == "v1_c3":
        training_result.update(
            logical_epoch=12,
            deployment_checkpoint_sha256=checkpoint_sha,
        )
    _write_json(run_dir / "training_result.json", training_result)
    _write_json(
        run_dir / "training_manifest.json",
        {"method": method, "variant": variant, "seed": 3072},
    )
    metrics = run_dir / "metrics/version_0/metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("epoch,train/loss\n0,1.25\n")
    if method_key == "v1_c3":
        _write_json(
            run_dir / "validation_offline_metrics.json",
            {"epochs": [{"logical_epoch": epoch} for epoch in range(1, 13)]},
        )


def test_reconciles_writes_and_loads_exact_eight_cell_archive(
    case: Case, tmp_path: Path
) -> None:
    study = _reconcile(case)
    assert (
        tuple(cell.identity for cell in study.cells)
        == results.EXPECTED_ENDPOINT_IDENTITIES
    )
    assert tuple(cell.cell_id for cell in study.cells) == (
        "v1__c__f_plus_g_first_q2__alpha_0p25",
        "v1__c2__f_only",
        "v1__c2__g_only",
        "v1__c2__f_plus_g",
        "v1__c2__f_plus_g_first__alpha_0p25",
        "v1__c2__f_plus_g_first_q2__alpha_0p25",
        "v1__c2__g_only_f_rollout_mean",
        "v1__c3__state_v_terminal",
    )
    assert len(study.cells) == 8
    assert sum(len(cell.outcomes) for cell in study.cells) == 400
    assert {
        cell.checkpoint_sha256 for cell in study.cells if cell.method_key == "v1_c2"
    } == {case.c2_sha}

    artifact_dir = tmp_path / "archive"
    paths = results.write_archive(study, artifact_dir=artifact_dir)
    assert {path.name for path in paths} == set(results.ARCHIVE_FILENAMES)
    loaded = results.load_endpoint_extension_ledger(artifact_dir)
    assert loaded == tuple(
        cell.__class__(
            **{
                **cell.__dict__,
                "source_directory": f"sources/{cell.cell_id}",
            }
        )
        for cell in study.cells
    )
    results.write_archive(study, artifact_dir=artifact_dir, check=True)
    with (artifact_dir / "all_o50_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert rows[-1]["method_key"] == "v1_c3"
    assert rows[-1]["score_mode"] == "state_v"


def test_optionally_archives_c2_c3_training_metrics(case: Case, tmp_path: Path) -> None:
    c2_run = tmp_path / "c2-training"
    c3_run = tmp_path / "c3-training"
    _write_training_run(
        c2_run,
        method_key="v1_c2",
        checkpoint=case.c2_checkpoint,
        checkpoint_sha=case.c2_sha,
    )
    _write_training_run(
        c3_run,
        method_key="v1_c3",
        checkpoint=case.c3_checkpoint,
        checkpoint_sha=case.c3_sha,
    )
    study = results.reconcile_endpoint_results(
        c2_launcher_root=case.root,
        c2_launcher_manifest=case.launcher_manifest,
        c3_output_dir=case.c3_output,
        c2_training_run=c2_run,
        c3_training_run=c3_run,
    )
    assert [item.method_key for item in study.training_artifacts] == [
        "v1_c2",
        "v1_c3",
    ]
    archive = tmp_path / "archive-with-training"
    results.write_archive(study, artifact_dir=archive)
    assert (archive / "training/v1_c2/metrics.csv").is_file()
    assert (archive / "training/v1_c3/validation_offline_metrics.json").is_file()
    assert len(results.load_endpoint_extension_ledger(archive)) == 8


def test_rejects_non_exact_launcher_grid_and_nonformal_output(case: Case) -> None:
    launcher = json.loads(case.launcher_manifest.read_text())
    removed_id = next(iter(launcher["jobs"]))
    removed = launcher["jobs"].pop(removed_id)
    _write_json(case.launcher_manifest, launcher)
    with pytest.raises(results.EndpointReconciliationError, match="jobs.grid"):
        _reconcile(case)

    launcher["jobs"][removed_id] = removed
    _write_json(case.launcher_manifest, launcher)
    output = case.formal_root / results._relative_c2_output("v1_c", "f_plus_g_first_q2")
    result = json.loads((output / "results.json").read_text())
    result["smoke"] = True
    _write_json(output / "results.json", result)
    with pytest.raises(results.EndpointReconciliationError, match="results.smoke"):
        _reconcile(case)


def test_rejects_outcome_selection_and_action_drift(case: Case) -> None:
    output = case.formal_root / results._relative_c2_output("v1_c2", "f_only")
    result = json.loads((output / "results.json").read_text())
    result["metrics"]["episode_successes"] = [True]
    _write_json(output / "results.json", result)
    with pytest.raises(results.EndpointReconciliationError, match="exactly 50"):
        _reconcile(case)

    result["metrics"]["episode_successes"] = [False] * 50
    _write_json(output / "results.json", result)
    selection = json.loads((output / "episode_selection.json").read_text())
    selection["valid_row_ranks"][0] = 999
    _write_json(output / "episode_selection.json", selection)
    with pytest.raises(results.EndpointReconciliationError, match="selection.sha256"):
        _reconcile(case)

    selection["valid_row_ranks"][0] = 0
    _write_json(output / "episode_selection.json", selection)
    action = json.loads((output / "action_normalization.json").read_text())
    action["samples"] += 1
    _write_json(output / "action_normalization.json", action)
    with pytest.raises(
        results.EndpointReconciliationError, match="action_normalization.sha256"
    ):
        _reconcile(case)


def test_rejects_checkpoint_or_parent_provenance_drift(case: Case) -> None:
    output = case.formal_root / results._relative_c2_output("v1_c2", "g_only")
    manifest = json.loads((output / "protocol_manifest.json").read_text())
    manifest["checkpoint"]["global_step"] = 1
    _write_json(output / "protocol_manifest.json", manifest)
    with pytest.raises(
        results.EndpointReconciliationError, match="checkpoint.global_step"
    ):
        _reconcile(case)

    manifest["checkpoint"]["global_step"] = 127_960
    manifest["checkpoint"]["predictor_config"]["source_v1_c"]["checkpoint_sha256"] = (
        "f" * 64
    )
    _write_json(output / "protocol_manifest.json", manifest)
    with pytest.raises(results.EndpointReconciliationError, match="checkpoint_sha256"):
        _reconcile(case)

    manifest["checkpoint"]["predictor_config"]["source_v1_c"]["checkpoint_sha256"] = (
        case.parent_sha
    )
    _write_json(output / "protocol_manifest.json", manifest)
    checkpoint_manifest = case.root / "checkpoint_manifest.json"
    checkpoint_data = json.loads(checkpoint_manifest.read_text())
    checkpoint_data["v1"]["c2"]["sha256"] = "e" * 64
    _write_json(checkpoint_manifest, checkpoint_data)
    with pytest.raises(results.EndpointReconciliationError, match="c2.file"):
        _reconcile(case)


def test_rejects_c3_epoch_commit_and_state_value_definition_drift(case: Case) -> None:
    manifest_path = case.c3_output / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint"]["logical_epoch"] = 11
    _write_json(manifest_path, manifest)
    with pytest.raises(results.EndpointReconciliationError, match="logical_epoch"):
        _reconcile(case)

    manifest["checkpoint"]["logical_epoch"] = 12
    manifest["runtime"]["tdwm_git_revision"] = "short"
    _write_json(manifest_path, manifest)
    with pytest.raises(results.EndpointReconciliationError, match="git revision"):
        _reconcile(case)

    manifest["runtime"]["tdwm_git_revision"] = "b" * 40
    manifest["score_definition"]["critic"] = "online"
    _write_json(manifest_path, manifest)
    with pytest.raises(results.EndpointReconciliationError, match="state_v.critic"):
        _reconcile(case)


def test_loader_rejects_source_tamper_unexpected_files_and_scalar_drift(
    case: Case, tmp_path: Path
) -> None:
    study = _reconcile(case)
    archive = tmp_path / "archive"
    results.write_archive(study, artifact_dir=archive)
    source_result = archive / "sources" / "v1__c2__f_only" / "results.json"
    source_result.write_text(source_result.read_text() + " ")
    with pytest.raises(
        results.EndpointReconciliationError, match="source_files_sha256"
    ):
        results.load_endpoint_extension_ledger(archive)

    source_result.write_bytes(
        Path(study.cells[1].source_directory, "results.json").read_bytes()
    )
    (archive / "unexpected.txt").write_text("unexpected\n")
    with pytest.raises(results.EndpointReconciliationError, match="file_set"):
        results.load_endpoint_extension_ledger(archive)

    (archive / "unexpected.txt").unlink()
    ledger_path = archive / "reconciliation_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["cells"][1]["success_count"] += 1
    _write_json(ledger_path, ledger)
    with pytest.raises(results.EndpointReconciliationError, match="success_count"):
        results.load_endpoint_extension_ledger(archive)


def test_archive_never_overwrites_existing_destination(
    case: Case, tmp_path: Path
) -> None:
    study = _reconcile(case)
    destination = tmp_path / "archive"
    destination.mkdir()
    sentinel = destination / "user-file.txt"
    sentinel.write_text("keep\n")
    with pytest.raises(
        results.EndpointReconciliationError, match="refusing to overwrite"
    ):
        results.write_archive(study, artifact_dir=destination)
    assert sentinel.read_text() == "keep\n"
