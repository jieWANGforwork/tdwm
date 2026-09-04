from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

from tdwm.results import actor_free_td_lewm_complete as complete_results
from tdwm.results import actor_free_td_lewm_mean_q_extension as extension


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_hashes(selection_sha: str, action_sha: str, label: str) -> dict[str, str]:
    return {
        "results.json": _sha(f"{label}:results"),
        "protocol_manifest.json": _sha(f"{label}:protocol"),
        "episode_selection.json": selection_sha,
        "action_normalization.json": action_sha,
    }


def _old_ledger(
    *,
    selection_file_sha: str,
    selection_ranks_sha: str,
    action_sha: str,
    checkpoint_paths: dict[tuple[str, str], str],
    checkpoint_hashes: dict[tuple[str, str], str],
) -> dict[str, Any]:
    old_commit = "1" * 40
    cells: list[dict[str, Any]] = []
    for epoch in extension.EPOCHS:
        for variant in extension.VARIANTS:
            for mode in extension.EMA_SCORE_MODES:
                label = f"ema:{epoch}:{variant}:{mode}"
                cells.append(
                    {
                        "cell_id": f"v2_ema_e{epoch:02d}_{variant}_{mode}"
                        + (
                            "_alpha_0p25" if mode == extension.FIRST_ACTION_MODE else ""
                        ),
                        "epoch": epoch,
                        "variant": variant,
                        "method": f"actor_free_td_lewm_v2_ema_sg_{variant}",
                        "score_mode": mode,
                        "g_first_weight": (
                            extension.FIRST_ACTION_WEIGHT
                            if mode == extension.FIRST_ACTION_MODE
                            else None
                        ),
                        "outcomes": [False] * 50,
                        "success_count": 0,
                        "success_rate": 0.0,
                        "success_rate_percent": 0.0,
                        "checkpoint_epoch": epoch,
                        "checkpoint_global_step": epoch * 12_796,
                        "checkpoint_path": f"/remote/ema/{variant}/epoch_{epoch:02d}.pt",
                        "checkpoint_sha256": _sha(f"ema:{epoch}:{variant}"),
                        "training_manifest_path": f"/remote/ema/{variant}/training.json",
                        "training_manifest_sha256": _sha(f"training:{variant}"),
                        "evaluation_commit": old_commit,
                        "selection_sha256": selection_file_sha,
                        "action_normalization_sha256": action_sha,
                        "source_scope": (
                            "strict_epoch3" if epoch == 3 else "original_epoch4_10"
                        ),
                        "source_state": "/remote/state.json",
                        "source_status": "SUCCEEDED",
                        "output_dir": f"/remote/ema/{epoch}/{variant}/{mode}",
                        "source_files_sha256": _source_hashes(
                            selection_file_sha, action_sha, label
                        ),
                    }
                )

    fixed_cells: list[dict[str, Any]] = []
    for version in extension.FIXED_VERSIONS:
        for variant in extension.VARIANTS:
            modes = (extension.FIRST_ACTION_MODE,) + (
                (extension.ROLLOUT_MEAN_MODE,) if version == "v2" else ()
            )
            for mode in modes:
                label = f"fixed:{version}:{variant}:{mode}"
                fixed_cells.append(
                    {
                        "version": version,
                        "variant": variant,
                        "score_mode": mode,
                        "g_first_weight": (
                            extension.FIRST_ACTION_WEIGHT
                            if mode == extension.FIRST_ACTION_MODE
                            else None
                        ),
                        "job_id": extension._fixed_job_id(version, variant, mode),
                        "method": f"actor_free_td_lewm_{version}_{variant}",
                        "outcomes": [False] * 49 + [True],
                        "success_count": 1,
                        "success_rate": 0.02,
                        "success_rate_percent": 2.0,
                        "checkpoint_path": checkpoint_paths[(version, variant)],
                        "checkpoint_sha256": checkpoint_hashes[(version, variant)],
                        "selection_sha256": selection_ranks_sha,
                        "episode_selection_file_sha256": selection_file_sha,
                        "action_normalization_sha256": action_sha,
                        "evaluation_commit": old_commit,
                        "evaluation_commit_evidence": {"source": "fixture"},
                        "source_launcher_manifest": "/remote/old/launcher.json",
                        "output_dir": f"/remote/old/{version}/{variant}/{mode}",
                        "source_files_sha256": _source_hashes(
                            selection_file_sha, action_sha, label
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "source": "actor_free_td_lewm_v2_ema_new_score_reconciliation",
        "cell_count": 96,
        "epochs": list(extension.EPOCHS),
        "variants": list(extension.VARIANTS),
        "score_modes": list(extension.EMA_SCORE_MODES),
        "g_first_weight": extension.FIRST_ACTION_WEIGHT,
        "selection_sha256": selection_file_sha,
        "action_normalization_sha256": action_sha,
        "evaluation_commit": old_commit,
        "training_commit": "2" * 40,
        "sources": {"strict_epoch3": {}, "original_epoch4_10": {}},
        "cells": cells,
        "fixed_checkpoint_comparison": {
            "included": True,
            "cell_count": 24,
            "selection_sha256": selection_ranks_sha,
            "episode_selection_file_sha256": selection_file_sha,
            "action_normalization_sha256": action_sha,
            "evaluation_commit": old_commit,
            "evaluation_commit_evidence_source": ("launcher_repository_checkout_head"),
            "launcher_sources": [
                {"manifest_sha256": _sha("launcher-1"), "job_count": 20},
                {"manifest_sha256": _sha("launcher-2"), "job_count": 3},
                {"manifest_sha256": _sha("launcher-3"), "job_count": 1},
            ],
            "cells": fixed_cells,
        },
    }


def _make_checkout(tmp_path: Path) -> tuple[Path, str, Path]:
    checkout = tmp_path / "local-evaluation-checkout"
    validator_source = Path(str(extension._fixed.__file__)).resolve()
    validator_relative = validator_source.relative_to(
        Path(extension.__file__).resolve().parents[3]
    )
    validator = checkout / validator_relative
    validator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(validator_source, validator)
    for version in extension.NEW_VERSIONS:
        for variant in extension.VARIANTS:
            evaluator = (
                checkout
                / "scripts"
                / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
            )
            evaluator.parent.mkdir(parents=True, exist_ok=True)
            evaluator.write_text("# committed evaluator fixture\n")
            config = extension._fixed.evaluation_config_path(
                checkout,
                version=version,
                variant=variant,
                score_mode=extension.ROLLOUT_MEAN_MODE,
            )
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("stage: formal\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
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
    return checkout, head, Path("/remote/evaluation-checkout")


def _write_new_output(
    *,
    output: Path,
    version: str,
    variant: str,
    checkpoint_path: str,
    checkpoint_sha: str,
    selection: dict[str, list[int]],
    action: dict[str, Any],
    evaluation_commit: str,
) -> None:
    fixed = extension._fixed
    result: dict[str, Any] = {
        "method": f"actor_free_td_lewm_{version}_{variant}",
        "method_family": f"actor_free_td_lewm_{version}",
        "variant": variant,
        "implementation_version": version,
        "score_mode": extension.ROLLOUT_MEAN_MODE,
        "planning_horizon": 5,
        "smoke": False,
        "pilot": False,
        "metrics": {"episode_successes": [False] * 48 + [True, True]},
    }
    definition = {
        "formula": "mean(q_1, q_2, q_3, q_4, q_5)",
        "action_processing": fixed.ROLLOUT_MEAN_ACTION_PROCESSING_BY_VERSION[version],
    }
    result.update(fixed.ROLLOUT_MEAN_METADATA_BY_VERSION[version])
    result["score_definition"] = definition
    inference = {
        "score_mode": extension.ROLLOUT_MEAN_MODE,
        **fixed.ROLLOUT_MEAN_INFERENCE_METADATA_BY_VERSION[version],
        "score_definition": definition,
    }
    manifest = {
        "score_mode": extension.ROLLOUT_MEAN_MODE,
        "runtime": {"tdwm_git_revision": evaluation_commit},
        "protocol": {
            "method": result["method"],
            "method_family": result["method_family"],
            "variant": variant,
            "implementation_version": version,
            "inference_objective": inference,
            "planning": {"horizon": 5},
        },
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
            "epoch": 10,
            "global_step": 127_960,
        },
        "selection": selection,
        "normalization": {"action": action},
        **fixed.ROLLOUT_MEAN_METADATA_BY_VERSION[version],
        "score_definition": definition,
    }
    _write_json(output / "results.json", result)
    _write_json(output / "protocol_manifest.json", manifest)
    _write_json(output / "episode_selection.json", selection)
    _write_json(output / "action_normalization.json", action)


@dataclass(frozen=True)
class FixtureCase:
    old_ledger: Path
    old_sha256: str
    old_bytes: bytes
    launcher_root: Path
    launcher_manifest: Path
    checkout: Path
    checkout_head: str
    selection_sha256: str
    ranks_sha256: str
    action_sha256: str


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FixtureCase:
    selection = {"valid_row_ranks": list(range(50))}
    action = {
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "variance": [1.0] * 5,
        "samples": 2_010_000,
    }
    scratch = tmp_path / "hash-inputs"
    _write_json(scratch / "episode_selection.json", selection)
    _write_json(scratch / "action_normalization.json", action)
    selection_sha = _file_sha(scratch / "episode_selection.json")
    action_sha = _file_sha(scratch / "action_normalization.json")
    ranks_sha = extension._fixed.canonical_json_sha256(selection["valid_row_ranks"])
    monkeypatch.setattr(extension, "RAW_SELECTION_SHA256", selection_sha)
    monkeypatch.setattr(extension, "SELECTION_RANKS_SHA256", ranks_sha)
    monkeypatch.setattr(extension, "ACTION_NORMALIZATION_SHA256", action_sha)

    checkout, checkout_head, recorded_repository = _make_checkout(tmp_path)
    checkpoint_paths = {
        (version, variant): str(
            Path("/remote/checkpoints") / version / f"{variant}_epoch_10.pt"
        )
        for version in extension.FIXED_VERSIONS
        for variant in extension.VARIANTS
    }
    checkpoint_hashes = {
        identity: _sha(f"checkpoint:{identity[0]}:{identity[1]}")
        for identity in checkpoint_paths
    }
    old = _old_ledger(
        selection_file_sha=selection_sha,
        selection_ranks_sha=ranks_sha,
        action_sha=action_sha,
        checkpoint_paths=checkpoint_paths,
        checkpoint_hashes=checkpoint_hashes,
    )
    old_path = tmp_path / "old/reconciliation_ledger.json"
    _write_json(old_path, old)
    old_bytes = old_path.read_bytes()

    recorded_output_root = Path("/remote/outputs/v0-v1-mean-q")
    recorded_stage_root = recorded_output_root / "formal"
    local_stage_root = tmp_path / "copied-launcher/formal"
    jobs: dict[str, dict[str, Any]] = {}
    for version in extension.NEW_VERSIONS:
        for variant in extension.VARIANTS:
            job_id = extension._fixed_job_id(
                version, variant, extension.ROLLOUT_MEAN_MODE
            )
            relative_output = Path(version) / variant / extension.ROLLOUT_MEAN_MODE
            local_output = local_stage_root / relative_output
            recorded_output = recorded_stage_root / relative_output
            _write_new_output(
                output=local_output,
                version=version,
                variant=variant,
                checkpoint_path=checkpoint_paths[(version, variant)],
                checkpoint_sha=checkpoint_hashes[(version, variant)],
                selection=selection,
                action=action,
                evaluation_commit=checkout_head,
            )
            recorded_evaluator = (
                recorded_repository
                / "scripts"
                / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
            )
            recorded_config = (
                recorded_repository
                / extension._fixed.evaluation_config_path(
                    checkout,
                    version=version,
                    variant=variant,
                    score_mode=extension.ROLLOUT_MEAN_MODE,
                ).relative_to(checkout)
            )
            job = extension._fixed.Job(
                job_id=job_id,
                stage="formal",
                version=version,
                variant=variant,
                score_mode=extension.ROLLOUT_MEAN_MODE,
                alpha=None,
                checkpoint=checkpoint_paths[(version, variant)],
                config_path=str(recorded_config),
                output_dir=str(local_output),
                log_path=str(recorded_stage_root / "_launcher/jobs" / f"{job_id}.log"),
                argv=("python", str(recorded_evaluator)),
            )
            evidence = extension._fixed.validate_job_output(
                job, expected_selection_file_sha256=selection_sha
            )
            evidence.update(
                {
                    "results_path": str(recorded_output / "results.json"),
                    "manifest_path": str(recorded_output / "protocol_manifest.json"),
                    "selection_path": str(recorded_output / "episode_selection.json"),
                }
            )
            raw_job = asdict(job)
            raw_job.update(
                {
                    "config_path": str(recorded_config),
                    "output_dir": str(recorded_output),
                    "argv": list(job.argv),
                    "state": "SUCCEEDED",
                    "exit_code": 0,
                    "evidence": evidence,
                }
            )
            jobs[job_id] = raw_job
    manifest = {
        "schema_version": 1,
        "launcher": "actor_free_td_lewm_first_action_comparison",
        "inference_only": True,
        "training_performed": False,
        "alpha_selection_performed": False,
        "stage": "formal",
        "status": "SUCCEEDED",
        "versions": list(extension.NEW_VERSIONS),
        "variants": list(extension.VARIANTS),
        "shared_score_modes": [extension.ROLLOUT_MEAN_MODE],
        "v2_only_score_modes": [],
        "score_modes_by_version": {
            version: [extension.ROLLOUT_MEAN_MODE] for version in extension.NEW_VERSIONS
        },
        "alphas": [],
        "repository": str(recorded_repository),
        "output_root": str(recorded_output_root),
        "expected_selection_file_sha256": selection_sha,
        "selection": {
            "valid_row_ranks": selection["valid_row_ranks"],
            "sha256": ranks_sha,
            "valid_row_ranks_sha256": ranks_sha,
            "selection_file_sha256": selection_sha,
            "selection_file_identical_across_all_jobs": True,
            "identical_across_all_jobs": True,
        },
        "jobs": jobs,
    }
    manifest_path = local_stage_root / "_launcher/launcher_manifest.json"
    _write_json(manifest_path, manifest)
    return FixtureCase(
        old_ledger=old_path,
        old_sha256=_file_sha(old_path),
        old_bytes=old_bytes,
        launcher_root=local_stage_root,
        launcher_manifest=manifest_path,
        checkout=checkout,
        checkout_head=checkout_head,
        selection_sha256=selection_sha,
        ranks_sha256=ranks_sha,
        action_sha256=action_sha,
    )


def _extend(case: FixtureCase) -> extension.ExtendedCompactLedger:
    return extension.extend_compact_ledger(
        old_ledger_path=case.old_ledger,
        expected_old_ledger_sha256=case.old_sha256,
        launcher_root=case.launcher_root,
        launcher_manifest=case.launcher_manifest,
        evaluation_checkout=case.checkout,
    )


def test_extends_relocated_launcher_deterministically_without_touching_old(
    case: FixtureCase, tmp_path: Path
) -> None:
    first = _extend(case)
    second = _extend(case)
    second_copy_root = tmp_path / "second-copy/formal"
    shutil.copytree(case.launcher_root, second_copy_root)
    relocated = extension.extend_compact_ledger(
        old_ledger_path=case.old_ledger,
        expected_old_ledger_sha256=case.old_sha256,
        launcher_root=second_copy_root,
        launcher_manifest=second_copy_root / "_launcher/launcher_manifest.json",
        evaluation_checkout=case.checkout,
    )

    assert first.payload == second.payload == relocated.payload
    assert first.sha256 == hashlib.sha256(first.payload).hexdigest()
    assert case.old_ledger.read_bytes() == case.old_bytes
    fixed = first.document["fixed_checkpoint_comparison"]
    assert fixed["cell_count"] == 36
    assert len(fixed["cells"]) == 36
    assert len(fixed["launcher_sources"]) == 4
    assert fixed["evaluation_commits"] == sorted(["1" * 40, case.checkout_head])
    new_cells = [
        cell
        for cell in fixed["cells"]
        if cell["version"] in extension.NEW_VERSIONS
        and cell["score_mode"] == extension.ROLLOUT_MEAN_MODE
    ]
    assert len(new_cells) == 12
    assert {cell["evaluation_commit"] for cell in new_cells} == {case.checkout_head}
    assert {
        cell["source_files_sha256"]["episode_selection.json"] for cell in new_cells
    } == {case.selection_sha256}
    assert {
        cell["source_files_sha256"]["action_normalization.json"] for cell in new_cells
    } == {case.action_sha256}

    output = tmp_path / "new-source/reconciliation_ledger.json"
    written = extension.write_extended_ledger(
        first, output_path=output, old_ledger_path=case.old_ledger
    )
    assert written.read_bytes() == first.payload
    with pytest.raises(extension.MeanQExtensionError, match="existing file"):
        extension.write_extended_ledger(
            first, output_path=output, old_ledger_path=case.old_ledger
        )
    with pytest.raises(extension.MeanQExtensionError, match="old compact ledger"):
        extension.write_extended_ledger(
            first,
            output_path=case.old_ledger,
            old_ledger_path=case.old_ledger,
        )
    assert case.old_ledger.read_bytes() == case.old_bytes


def test_rejects_wrong_old_ledger_sha(case: FixtureCase) -> None:
    with pytest.raises(extension.MeanQExtensionError, match="old_ledger.sha256"):
        extension.extend_compact_ledger(
            old_ledger_path=case.old_ledger,
            expected_old_ledger_sha256="0" * 64,
            launcher_root=case.launcher_root,
            launcher_manifest=case.launcher_manifest,
            evaluation_checkout=case.checkout,
        )


def test_rejects_non_exact_grid_and_repository_mismatch(case: FixtureCase) -> None:
    manifest = json.loads(case.launcher_manifest.read_text())
    removed_id = next(iter(manifest["jobs"]))
    removed = manifest["jobs"].pop(removed_id)
    _write_json(case.launcher_manifest, manifest)
    with pytest.raises(extension.MeanQExtensionError, match="exact 12-cell"):
        _extend(case)

    manifest["jobs"][removed_id] = removed
    manifest["repository"] = "/remote/wrong-checkout"
    _write_json(case.launcher_manifest, manifest)
    with pytest.raises(extension.MeanQExtensionError, match="recorded path is outside"):
        _extend(case)


def test_rejects_checkpoint_hash_drift_and_non_h5_mean_q(case: FixtureCase) -> None:
    manifest = json.loads(case.launcher_manifest.read_text())
    job = next(iter(manifest["jobs"].values()))
    relative = Path(job["version"]) / job["variant"] / extension.ROLLOUT_MEAN_MODE
    protocol_path = case.launcher_root / relative / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["checkpoint"]["sha256"] = "f" * 64
    _write_json(protocol_path, protocol)
    with pytest.raises(extension.MeanQExtensionError, match="checkpoint.sha256"):
        _extend(case)

    protocol["checkpoint"]["sha256"] = json.loads(case.old_ledger.read_text())[
        "fixed_checkpoint_comparison"
    ]["cells"][0]["checkpoint_sha256"]
    protocol["protocol"]["planning"]["horizon"] = 4
    _write_json(protocol_path, protocol)
    with pytest.raises(extension.MeanQExtensionError, match="planning.horizon"):
        _extend(case)


def test_rejects_locked_selection_or_action_hash_drift(case: FixtureCase) -> None:
    manifest = json.loads(case.launcher_manifest.read_text())
    job = next(iter(manifest["jobs"].values()))
    relative = Path(job["version"]) / job["variant"] / extension.ROLLOUT_MEAN_MODE
    action_path = case.launcher_root / relative / "action_normalization.json"
    action = json.loads(action_path.read_text())
    action["samples"] += 1
    _write_json(action_path, action)
    with pytest.raises(extension.MeanQExtensionError, match="action_sha256"):
        _extend(case)


def test_rejects_runtime_commit_drift_and_non_root_or_dirty_checkout(
    case: FixtureCase,
) -> None:
    manifest = json.loads(case.launcher_manifest.read_text())
    job = next(iter(manifest["jobs"].values()))
    relative = Path(job["version"]) / job["variant"] / extension.ROLLOUT_MEAN_MODE
    protocol_path = case.launcher_root / relative / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["runtime"]["tdwm_git_revision"] = "f" * 40
    _write_json(protocol_path, protocol)
    with pytest.raises(
        extension.MeanQExtensionError,
        match="protocol_runtime.tdwm_git_revision",
    ):
        _extend(case)

    protocol["runtime"]["tdwm_git_revision"] = case.checkout_head
    _write_json(protocol_path, protocol)
    with pytest.raises(extension.MeanQExtensionError, match="top level"):
        extension.extend_compact_ledger(
            old_ledger_path=case.old_ledger,
            expected_old_ledger_sha256=case.old_sha256,
            launcher_root=case.launcher_root,
            launcher_manifest=case.launcher_manifest,
            evaluation_checkout=case.checkout / "scripts",
        )

    dirty = case.checkout / "untracked.txt"
    dirty.write_text("untracked evidence\n")
    with pytest.raises(extension.MeanQExtensionError, match="completely clean"):
        _extend(case)


def test_extended_ledger_is_accepted_as_complete_477_source(
    case: FixtureCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _extend(case)
    ledger_path = tmp_path / "complete-source/reconciliation_ledger.json"
    extension.write_extended_ledger(
        result, output_path=ledger_path, old_ledger_path=case.old_ledger
    )
    monkeypatch.setattr(complete_results, "SELECTION_SHA256", case.selection_sha256)
    monkeypatch.setattr(
        complete_results, "ACTION_NORMALIZATION_SHA256", case.action_sha256
    )
    monkeypatch.setattr(
        complete_results, "FIXED_SELECTION_RANKS_SHA256", case.ranks_sha256
    )
    monkeypatch.setattr(complete_results, "EMA_NEW_EVALUATION_COMMIT", "1" * 40)
    monkeypatch.setattr(complete_results, "EMA_TRAINING_COMMIT", "2" * 40)

    document = result.document
    ema_index = {(cell["epoch"], cell["variant"]): cell for cell in document["cells"]}
    ema_references = {
        identity: {
            "checkpoint_sha256": cell["checkpoint_sha256"],
            "checkpoint_path": cell["checkpoint_path"],
            "epoch": identity[0],
            "global_step": identity[0] * 12_796,
            "training_commit": "2" * 40,
        }
        for identity, cell in ema_index.items()
    }
    first_q = {
        (cell["version"], cell["variant"]): cell
        for cell in document["fixed_checkpoint_comparison"]["cells"]
        if cell["score_mode"] == extension.FIRST_ACTION_MODE
    }
    fixed_references = {
        version: {
            variant: {
                "checkpoint_sha256": first_q[(version, variant)]["checkpoint_sha256"],
                "checkpoint_path": first_q[(version, variant)]["checkpoint_path"],
                "epoch": 10,
                "global_step": 127_960,
                "training_commit": "3" * 40,
            }
            for variant in extension.VARIANTS
        }
        for version in extension.FIXED_VERSIONS
    }

    cells, evidence = complete_results._validate_ema_new_ledger(
        ledger_path=ledger_path,
        fixed_references=fixed_references,
        ema_references=ema_references,
        expected_ledger_sha256=result.sha256,
    )

    assert len(cells) == 132
    assert evidence["fixed_checkpoint_cell_count"] == 36
