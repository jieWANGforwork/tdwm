from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.results.actor_free_td_lewm import (
    BundleValidationError,
    VARIANT_ORDER,
    build_summary,
    combined_mode_for_variant,
    modes_for_variant,
    validate_bundle,
    write_archive,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _protocol(variant: str, score_mode: str | None) -> dict:
    objective = {
        "goal_usage": (
            "training_and_planning_direct_critic_input"
            if variant == "direct_goal_hybrid"
            else "planning_linear_readout_only"
        ),
        "learned_actor": False,
    }
    if score_mode is not None:
        objective["score_mode"] = score_mode
    return {
        "schema_version": 1,
        "id": f"actor_free_td_lewm_{variant}_cube_o50",
        "method": "actor_free_td_lewm",
        "variant": variant,
        "environment": "cube",
        "stage": "planner_evaluation",
        "runtime": {"stable_worldmodel_version": "0.1.1"},
        "dataset": {"identifier": "quentinll/lewm-cube"},
        "evaluation": {"episodes": 50, "goal_offset": 50},
        "planning": {
            "horizon": 5,
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
        },
        "inference_objective": objective,
    }


def _selection() -> dict[str, list[int]]:
    return {
        "episode_indices": list(range(100, 150)),
        "start_steps": [index % 20 for index in range(50)],
        "goal_steps": [50 + index % 20 for index in range(50)],
        "valid_row_ranks": [1000 + index * 7 for index in range(50)],
    }


def _success_count(variant_index: int, mode_index: int) -> int:
    return 12 + 3 * variant_index + mode_index


def _make_bundle(root: Path, *, legacy_combined: bool = False) -> Path:
    selection = _selection()
    for variant_index, variant in enumerate(VARIANT_ORDER):
        variant_root = root / variant
        checkpoint_sha = _sha(f"checkpoint-{variant}")
        curve = []
        for epoch in range(1, 11):
            curve.append(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / (epoch + variant_index + 1),
                    "validation_loss": 1.2 / (epoch + variant_index + 1),
                }
            )
        curve_path = variant_root / "training_curve.csv"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        with curve_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("epoch", "train_loss", "validation_loss")
            )
            writer.writeheader()
            writer.writerows(curve)
        final = curve[-1]
        best = min(curve, key=lambda row: row["validation_loss"])
        _write_json(
            variant_root / "training_summary.json",
            {
                "schema_version": 1,
                "method": "actor_free_td_lewm",
                "variant": variant,
                "seed": 3072,
                "status": "complete",
                "epochs_completed": 10,
                "global_step": 127960,
                "training_commit": _sha(f"training-{variant}")[:40],
                "checkpoint_sha256": checkpoint_sha,
                "runtime": {
                    "stable_worldmodel": "0.1.1",
                    "cuda_device": "Synthetic GPU",
                },
                "metrics": {
                    "final_epoch": {
                        "epoch": 10,
                        "train/loss": final["train_loss"],
                        "validation/loss": final["validation_loss"],
                    },
                    "best_validation": {
                        "epoch": best["epoch"],
                        "metric": "validation/loss",
                        "value": best["validation_loss"],
                    },
                },
                "source_files": {
                    "training_result.json": _sha(f"result-{variant}"),
                    "training_manifest.json": _sha(f"manifest-{variant}"),
                    "metrics.csv": _sha(f"metrics-{variant}"),
                },
            },
        )
        combined = combined_mode_for_variant(variant)
        for mode_index, mode in enumerate(modes_for_variant(variant)):
            is_legacy = legacy_combined and mode == combined
            formal = _protocol(variant, None if is_legacy else combined)
            configured = deepcopy(formal)
            if not is_legacy:
                configured["inference_objective"]["score_mode"] = mode
            count = _success_count(variant_index, mode_index)
            outcomes = [index < count for index in range(50)]
            run_root = variant_root / mode
            _write_json(run_root / "episode_selection.json", selection)
            manifest = {
                "protocol": configured,
                "formal_protocol": formal,
                "dataset": {
                    "format": "lance",
                    "episodes": 10000,
                    "transitions": 2010000,
                },
                "checkpoint": {
                    "sha256": checkpoint_sha,
                    "method": "actor_free_td_lewm",
                    "variant": variant,
                },
                "selection": selection,
                "runtime": {
                    "stable_worldmodel": "0.1.1",
                    "tdwm_git_revision": _sha(f"eval-{variant}-{mode}")[:40],
                    "cuda_device": "Synthetic GPU",
                },
            }
            result = {
                "metrics": {"success": outcomes, "success_rate": count / 50},
                "elapsed_seconds": 10.0 + variant_index + mode_index,
                "world_model_parameter_count": 18_034_628,
                (
                    "critic_parameter_count"
                    if variant == "direct_goal_hybrid"
                    else "successor_parameter_count"
                ): 1000 + variant_index,
                "method": "actor_free_td_lewm",
                "variant": variant,
                "smoke": False,
                "pilot": False,
                "protocol_manifest": str(run_root / "protocol_manifest.json"),
            }
            if not is_legacy:
                manifest["score_mode"] = mode
                result["score_mode"] = mode
            _write_json(run_root / "protocol_manifest.json", manifest)
            _write_json(run_root / "results.json", result)
    return root


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_complete_bundle_generates_deterministic_auditable_archive(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle", legacy_combined=True)
    study = validate_bundle(bundle)
    artifact_dir = tmp_path / "reports/artifacts/actor_free_td_lewm_cube_seed3072"
    report = tmp_path / "reports/actor_free_td_lewm_cube_seed3072.md"

    paths = write_archive(study, artifact_dir=artifact_dir, report_path=report)

    assert len(paths) == 7
    summary = _read_json(artifact_dir / "summary.json")
    assert summary["study"]["evaluation_count"] == 21
    assert summary["validation"] == {
        "common_selection_across_21_runs": True,
        "complete_7x3_bundle": True,
        "formal_o50_only": True,
        "same_checkpoint_within_each_variant": True,
        "smoke_or_pilot_runs": 0,
        "success_rates_match_episode_outcomes": True,
        "training_checkpoint_matches_evaluation": True,
    }
    assert len(summary["ranking_by_combined"]) == 7
    assert summary["ranking_by_combined"][0]["variant"] == "direct_goal_hybrid"
    for variant in VARIANT_ORDER:
        combined = combined_mode_for_variant(variant)
        assert (
            summary["methods"][variant]["evaluations"][combined]["score_mode_source"]
            == "legacy_combined_default"
        )

    with (artifact_dir / "paired_outcomes.csv").open(newline="") as stream:
        outcomes = list(csv.DictReader(stream))
    assert len(outcomes) == 50
    assert len(outcomes[0]) == 7 + 21
    with (artifact_dir / "training_loss_curves.csv").open(newline="") as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 70
    assert {row["cross_method_comparable"] for row in curves} == {"false"}
    svg = (artifact_dir / "training_loss_curves.svg").read_text()
    assert "Training total loss" in svg
    assert "Validation total loss" in svg
    assert "not cross-method ranking" in svg
    assert "不能比较曲线高低" in report.read_text()

    write_archive(
        study, artifact_dir=artifact_dir, report_path=report, check=True
    )
    checksum_lines = (artifact_dir / "checksums.sha256").read_text().splitlines()
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((artifact_dir / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_explicit_score_modes_are_recorded(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    study = validate_bundle(bundle)

    assert all(
        run.score_mode_source == "explicit"
        for runs in study.evaluations.values()
        for run in runs.values()
    )


def test_combined_ranking_uses_shared_ranks_for_ties(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    target = bundle / "serial_coupled/f_plus_g/results.json"
    result = _read_json(target)
    result["metrics"]["success"] = [index < 14 for index in range(50)]
    result["metrics"]["success_rate"] = 14 / 50
    _write_json(target, result)

    ranking = build_summary(validate_bundle(bundle))["ranking_by_combined"]
    by_variant = {row["variant"]: row["rank"] for row in ranking}

    assert by_variant["serial_decoupled"] == by_variant["serial_coupled"]


def test_non_combined_mode_cannot_use_legacy_implicit_metadata(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    run_root = bundle / "hybrid/g_only"
    result = _read_json(run_root / "results.json")
    manifest = _read_json(run_root / "protocol_manifest.json")
    del result["score_mode"]
    del manifest["score_mode"]
    del manifest["protocol"]["inference_objective"]["score_mode"]
    del manifest["formal_protocol"]["inference_objective"]["score_mode"]
    _write_json(run_root / "results.json", result)
    _write_json(run_root / "protocol_manifest.json", manifest)

    with pytest.raises(BundleValidationError, match="non-combined scores require"):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("checkpoint", "different checkpoints"),
        ("selection", "common 21-run O50 selection"),
        ("smoke", "smoke results cannot"),
        ("rate", "disagrees with"),
        ("curve", "epochs 1..10"),
    ),
)
def test_bundle_rejects_invalid_or_incomplete_evidence(tmp_path, mutation, message):
    bundle = _make_bundle(tmp_path / "bundle")
    run_root = bundle / "serial_decoupled/g_only"
    if mutation == "checkpoint":
        manifest = _read_json(run_root / "protocol_manifest.json")
        manifest["checkpoint"]["sha256"] = _sha("wrong-checkpoint")
        _write_json(run_root / "protocol_manifest.json", manifest)
    elif mutation == "selection":
        manifest = _read_json(run_root / "protocol_manifest.json")
        selection = _read_json(run_root / "episode_selection.json")
        selection["start_steps"][0] += 1
        selection["goal_steps"][0] += 1
        manifest["selection"] = selection
        _write_json(run_root / "episode_selection.json", selection)
        _write_json(run_root / "protocol_manifest.json", manifest)
    elif mutation == "smoke":
        result = _read_json(run_root / "results.json")
        result["smoke"] = True
        _write_json(run_root / "results.json", result)
    elif mutation == "rate":
        result = _read_json(run_root / "results.json")
        result["metrics"]["success_rate"] = 0.98
        _write_json(run_root / "results.json", result)
    elif mutation == "curve":
        curve = bundle / "serial_decoupled/training_curve.csv"
        rows = curve.read_text().splitlines()
        curve.write_text("\n".join(rows[:-1]) + "\n")

    with pytest.raises(BundleValidationError, match=message):
        validate_bundle(bundle)
