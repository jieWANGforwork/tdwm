from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tdwm.results import actor_free_td_lewm_v2 as results

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_REVISION = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_protocol(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    parent = value.get("extends")
    if parent is None:
        return value
    return _merge(_load_protocol(path.parent / parent), value)


def _selection() -> dict:
    ranks = np.sort(np.random.default_rng(42).choice(1_510_000 - 1, 50, replace=False))
    start = ranks % 151
    return {
        "episode_indices": (ranks // 151).tolist(),
        "start_steps": start.tolist(),
        "goal_steps": (start + 50).tolist(),
        "valid_row_ranks": ranks.tolist(),
    }


def _source_file(root: Path, name: str, payload: bytes) -> tuple[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path.resolve()), _sha256(path)


def _make_acceptance(root: Path) -> dict:
    common_config_hash = "1" * 64
    variants = {}
    for index, variant in enumerate(results.VARIANT_ORDER):
        source_root = root / "accepted_sources" / variant
        checkpoint_path, checkpoint_sha = _source_file(
            source_root, "epoch_10.pt", f"checkpoint-{variant}".encode()
        )
        training_result_path, training_result_sha = _source_file(
            source_root, "training_result.json", f"result-{variant}".encode()
        )
        training_manifest = source_root / "training_manifest.json"
        _write_json(
            training_manifest,
            {"runtime": {"tdwm_git_revision": TEST_REVISION}},
        )
        training_manifest_path = str(training_manifest.resolve())
        training_manifest_sha = _sha256(training_manifest)
        evidence = source_root / "execution_evidence.json"
        _write_json(
            evidence,
            {
                "process": {"git_revision": TEST_REVISION},
                "outputs": {
                    "lightning_last": {
                        "resume_identity": {"v2_start_revision": TEST_REVISION}
                    }
                },
            },
        )
        evidence_path = str(evidence.resolve())
        evidence_sha = _sha256(evidence)
        last_checkpoint_path, last_checkpoint_sha = _source_file(
            source_root, "last.ckpt", f"last-{variant}".encode()
        )
        curve = [
            {
                "epoch": epoch,
                "train_loss": 1.0 + index + epoch / 100,
                "train_base_hybrid_td": 2.0 + index + epoch / 100,
                "validation_loss": 3.0 + index + epoch / 100,
                "validation_base_hybrid_td": 4.0 + index + epoch / 100,
            }
            for epoch in range(1, 11)
        ]
        variants[variant] = {
            "method": f"{results.METHOD_FAMILY}_{variant}",
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": 10,
            "checkpoint_global_step": 127_960,
            "training_revision": TEST_REVISION,
            "world_model_config_sha256": common_config_hash,
            "source_v1_world_model_config_sha256": common_config_hash,
            "world_model_parameter_count": results.WORLD_MODEL_PARAMETERS,
            "predictor_parameter_count": results.PREDICTOR_PARAMETERS,
            "protocol_sha256": results.TRAINING_PROTOCOL_SHA256[variant],
            "online_world_state_sha256": "2" * 64,
            "target_world_state_sha256": "3" * 64,
            "online_predictor_state_sha256": "4" * 64,
            "target_predictor_state_sha256": "5" * 64,
            "online_action_encoder_state_sha256": "6" * 64,
            "target_action_encoder_state_sha256": "7" * 64,
            "training_result_path": training_result_path,
            "training_result_sha256": training_result_sha,
            "training_manifest_path": training_manifest_path,
            "training_manifest_sha256": training_manifest_sha,
            "execution_evidence_path": evidence_path,
            "execution_evidence_sha256": evidence_sha,
            "last_checkpoint_path": last_checkpoint_path,
            "last_checkpoint_sha256": last_checkpoint_sha,
            "source_v1": {
                "checkpoint_path": f"/source/{variant}.pt",
                "checkpoint_sha256": results.SOURCE_V1_SHA256[variant],
                "epoch": 10,
                "global_step": 127_960,
            },
            "metrics": {"epochs": curve, "final_step": 127_959},
        }
    acceptance = {
        "schema_version": 1,
        "training_revision": TEST_REVISION,
        "seed": 3072,
        "expected_epoch": 10,
        "expected_global_step": 127_960,
        "world_model_parameter_count": results.WORLD_MODEL_PARAMETERS,
        "predictor_parameter_count": results.PREDICTOR_PARAMETERS,
        "stable_worldmodel_version": "0.1.1",
        "common_world_model_config_sha256": common_config_hash,
        "variants": variants,
        "status": "PASS",
        "warnings": [],
        "errors": [],
    }
    _write_json(root / "training_acceptance.json", acceptance)
    return acceptance


def _make_evaluations(root: Path, acceptance: dict) -> None:
    selection = _selection()
    selection_path = root / "selection_reference.json"
    _write_json(selection_path, selection)
    assert _sha256(selection_path) == results.SELECTION_SHA256
    action = {
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "variance": [1.0] * 5,
        "samples": 2_010_000,
    }
    for variant_index, variant in enumerate(results.VARIANT_ORDER):
        formal = _load_protocol(
            REPOSITORY_ROOT
            / "configs"
            / "experiment"
            / f"actor_free_td_lewm_v2_{variant}_cube_checkpoint_o50.yaml"
        )
        assert (
            results.canonical_sha256(formal)
            == results.EVALUATION_PROTOCOL_SHA256[variant]
        )
        training = acceptance["variants"][variant]
        predictor_config = {
            **copy.deepcopy(formal["predictor"]),
            "method": f"{results.METHOD_FAMILY}_{variant}",
            "method_family": results.METHOD_FAMILY,
            "variant": variant,
            "implementation_version": "v2",
            "objective_version": 0,
            "deployment_checkpoint_version": 1,
            "task_sampling": copy.deepcopy(formal["task_sampling"]),
            "joint_objective": copy.deepcopy(formal["joint_objective"]),
            "source_v1": copy.deepcopy(formal["source_v1"]),
            "source_artifacts": copy.deepcopy(formal["source_artifacts"]),
        }
        for mode_index, score_mode in enumerate(results.SCORE_MODES):
            configured = copy.deepcopy(formal)
            configured["inference_objective"]["score_mode"] = score_mode
            configured["planning"]["horizon"] = results.FORMAL_HORIZON_BY_SCORE_MODE[
                score_mode
            ]
            assert (
                results.canonical_sha256(configured)
                == results.CONFIGURED_PROTOCOL_SHA256[variant][score_mode]
            )
            successes = [pair < 10 + variant_index + mode_index for pair in range(50)]
            run_root = root / "evaluations" / variant / score_mode
            _write_json(run_root / "episode_selection.json", selection)
            _write_json(run_root / "action_normalization.json", action)
            manifest = {
                "score_mode": score_mode,
                "protocol": configured,
                "formal_protocol": formal,
                "dataset": {
                    "path": "/data/cube.lance",
                    "format": "lance",
                    "episodes": 10_000,
                    "transitions": 2_010_000,
                    "source_sha256": results.DATASET_SOURCE_SHA256,
                    "conversion_manifest_sha256": results.LANCE_MANIFEST_SHA256,
                },
                "checkpoint": {
                    "path": training["checkpoint_path"],
                    "sha256": training["checkpoint_sha256"],
                    "method": f"{results.METHOD_FAMILY}_{variant}",
                    "method_family": results.METHOD_FAMILY,
                    "variant": variant,
                    "implementation_version": "v2",
                    "objective_version": 0,
                    "epoch": 10,
                    "global_step": 127_960,
                    "formal_completion_required": True,
                    "predictor_config": predictor_config,
                    "source_v1_provenance": {
                        "checkpoint_sha256": results.SOURCE_V1_SHA256[variant],
                        "source_epoch": 10,
                        "source_global_step": 127_960,
                        "optimizer_state_loaded": False,
                        "target_world_initialization": (
                            "copy_of_v1_online_world_model"
                        ),
                    },
                },
                "selection": selection,
                "normalization": {"action": action},
                "runtime": {
                    "stable_worldmodel": "0.1.1",
                    "torch": "2.7.1",
                    "python": "3.11.12",
                    "platform": "Linux-test",
                    "tdwm_git_revision": TEST_REVISION,
                    "device": "cuda",
                    "cuda_device": "NVIDIA-test",
                    "compatibility_adapter": {"status": "not_needed"},
                },
            }
            _write_json(run_root / "protocol_manifest.json", manifest)
            _write_json(
                run_root / "results.json",
                {
                    "metrics": {
                        "episode_successes": successes,
                        "success_rate": 100.0 * sum(successes) / 50,
                    },
                    "elapsed_seconds": 10.0 + variant_index + mode_index,
                    "world_model_parameter_count": results.WORLD_MODEL_PARAMETERS,
                    "predictor_parameter_count": results.PREDICTOR_PARAMETERS,
                    "method": f"{results.METHOD_FAMILY}_{variant}",
                    "method_family": results.METHOD_FAMILY,
                    "variant": variant,
                    "implementation_version": "v2",
                    "score_mode": score_mode,
                    "planning_horizon": results.FORMAL_HORIZON_BY_SCORE_MODE[
                        score_mode
                    ],
                    "smoke": False,
                    "pilot": False,
                },
            )


@pytest.fixture
def formal_bundle(tmp_path: Path) -> Path:
    acceptance = _make_acceptance(tmp_path)
    _make_evaluations(tmp_path, acceptance)
    return tmp_path


def test_complete_bundle_builds_deterministic_archive(formal_bundle: Path) -> None:
    study = results.validate_bundle(formal_bundle)
    summary = results.build_summary(study)
    assert summary["study"]["evaluation_count"] == 18
    assert summary["validation"]["complete_6x3_bundle"] is True
    assert (
        summary["success_threshold_contract"]["explicit_constructor_argument_supported"]
        is False
    )
    paired = list(
        csv.DictReader(io.StringIO(results.build_paired_outcomes_csv(study).decode()))
    )
    curves = list(
        csv.DictReader(io.StringIO(results.build_training_curves_csv(study).decode()))
    )
    assert len(paired) == 50
    assert len(curves) == 60
    assert "success_g3__f_plus_g" in paired[0]
    assert "validation_common_base_td" in curves[0]

    artifact_dir = formal_bundle / "archive"
    report_path = formal_bundle / "report.md"
    written = results.write_archive(
        study, artifact_dir=artifact_dir, report_path=report_path
    )
    assert len(written) == 6
    assert "stable-worldmodel==0.1.1" in report_path.read_text()
    assert (
        results.write_archive(
            study, artifact_dir=artifact_dir, report_path=report_path, check=True
        )
        == written
    )


def test_bundle_rejects_missing_o50_cell(formal_bundle: Path) -> None:
    target = formal_bundle / "evaluations" / "g3" / "g_only"
    for path in target.iterdir():
        path.unlink()
    target.rmdir()
    with pytest.raises(results.V2ResultValidationError, match="exactly"):
        results.validate_bundle(formal_bundle)


def test_bundle_rejects_checkpoint_identity_drift(formal_bundle: Path) -> None:
    manifest_path = (
        formal_bundle / "evaluations" / "c" / "g_only" / "protocol_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint"]["sha256"] = "a" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(results.V2ResultValidationError, match="checkpoint SHA"):
        results.validate_bundle(formal_bundle)


def test_bundle_rejects_non_boolean_or_incomplete_outcomes(formal_bundle: Path) -> None:
    result_path = formal_bundle / "evaluations" / "d" / "f_plus_g" / "results.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["episode_successes"][-1] = 1
    _write_json(result_path, result)
    with pytest.raises(results.V2ResultValidationError, match="50 booleans"):
        results.validate_bundle(formal_bundle)


def test_bundle_rejects_evaluation_revision_drift(formal_bundle: Path) -> None:
    manifest_path = (
        formal_bundle / "evaluations" / "f" / "f_only" / "protocol_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"]["tdwm_git_revision"] = "b" * 40
    _write_json(manifest_path, manifest)
    with pytest.raises(results.V2ResultValidationError, match="tdwm_git_revision"):
        results.validate_bundle(formal_bundle)
