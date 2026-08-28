from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from tdwm.results.actor_free_td_lewm import (
    BundleValidationError,
    SELECTION_SHA256,
    VARIANT_ORDER,
    build_summary,
    combined_mode_for_variant,
    modes_for_variant,
    validate_bundle,
    write_archive,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORLD_PARAMETERS = 18_034_628
DATASET_SIZE = 23_456_789


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _config(variant: str, *, training: bool) -> dict:
    suffix = "cube_train" if training else "cube_checkpoint_o50"
    path = (
        REPOSITORY_ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_{variant}_{suffix}.yaml"
    )
    return yaml.safe_load(path.read_text())


def _selection() -> dict[str, list[int]]:
    source = (
        REPOSITORY_ROOT
        / "reports"
        / "artifacts"
        / "aligned_acd_o50_seed3072"
        / "paired_outcomes.csv"
    )
    with source.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream) if int(row["planning_seed"]) == 42
        ]
    assert len(rows) == 50
    selection = {
        "episode_indices": [int(row["episode_index"]) for row in rows],
        "start_steps": [int(row["start_step"]) for row in rows],
        "goal_steps": [int(row["goal_step"]) for row in rows],
        "valid_row_ranks": [int(row["valid_row_rank"]) for row in rows],
    }
    encoded = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(encoded).hexdigest() == SELECTION_SHA256
    return selection


def _head_count(variant_index: int) -> int:
    return 2_000 + variant_index


def _training_dataset() -> dict:
    return {
        "path": "/srv/datasets/cube_single_expert.lance",
        "format": "lance",
        "size_bytes": DATASET_SIZE,
        "conversion_manifest_path": (
            "/srv/datasets/cube_single_expert.lance.manifest.json"
        ),
        "conversion_manifest": {
            "schema_version": 1,
            "destination": {"format": "lance", "size_bytes": DATASET_SIZE},
            "conversion": {"stable_worldmodel_version": "0.1.1"},
        },
        "sequence_samples": 1_279_600,
        "split": {
            "train_size": 1_151_640,
            "validation_size": 127_960,
            "seed": 3072,
        },
    }


def _evaluation_dataset() -> dict:
    return {
        "path": "/srv/datasets/cube_single_expert.lance",
        "format": "lance",
        "size_bytes": DATASET_SIZE,
        "conversion_manifest_path": (
            "/srv/datasets/cube_single_expert.lance.manifest.json"
        ),
        "episodes": 10_000,
        "transitions": 2_010_000,
    }


def _write_lightning_metrics(path: Path, variant_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "epoch",
        "step",
        "train/loss_step",
        "train/loss_epoch",
        "train/prediction_loss_epoch",
        "validation/loss",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for zero_epoch in range(10):
            final_step = (zero_epoch + 1) * 12_796 - 1
            train_loss = 1.0 / (zero_epoch + variant_index + 2)
            validation_loss = 1.2 / (zero_epoch + variant_index + 2)
            writer.writerow(
                {
                    "epoch": zero_epoch,
                    "step": max(0, final_step - 50),
                    "train/loss_step": train_loss + 0.01,
                }
            )
            # Real Lightning order: validation aggregate, then train aggregate.
            writer.writerow(
                {
                    "epoch": zero_epoch,
                    "step": final_step,
                    "validation/loss": validation_loss,
                }
            )
            writer.writerow(
                {
                    "epoch": zero_epoch,
                    "step": final_step,
                    "train/loss_epoch": train_loss,
                    "train/prediction_loss_epoch": train_loss / 2,
                }
            )


def _make_training(variant_root: Path, variant: str, variant_index: int) -> None:
    run_dir = f"/srv/runs/actor-free/{variant}"
    protocol = _config(variant, training=True)
    _write_json(
        variant_root / "training_result.json",
        {
            "method": "actor_free_td_lewm",
            "variant": variant,
            "run_dir": run_dir,
            "seed": 3072,
            "last_checkpoint": f"{run_dir}/checkpoints/lightning/last.ckpt",
            "final_epoch": 10,
            "global_step": 127_960,
            "peak_cuda_memory_bytes": 8_000_000_000,
        },
    )
    head_key = "critic" if variant == "direct_goal_hybrid" else "successor"
    _write_json(
        variant_root / "training_manifest.json",
        {
            "method": "actor_free_td_lewm",
            "variant": variant,
            "objective_version": protocol[head_key]["objective_version"],
            "deployment_checkpoint_version": 1,
            "protocol": protocol,
            "protocol_path": (
                f"/srv/repo/configs/experiment/actor_free_td_lewm_{variant}_cube_train.yaml"
            ),
            "seed": 3072,
            "dataset": _training_dataset(),
            "model": {
                "config": {"_target_": "stable_worldmodel.LeWM"},
                "lewm_parameters": WORLD_PARAMETERS,
                (
                    "critic_parameters"
                    if variant == "direct_goal_hybrid"
                    else "successor_parameters"
                ): _head_count(variant_index),
                "action_block_dim": 25,
            },
            "training": {
                "formal_optimizer_steps": 127_960,
                "optimizer_steps_per_epoch": 12_796,
                "available_batches_per_epoch": 130_000,
                "configured_optimizer_steps": 127_960,
                "resume_mode": "auto",
                "resumed_from": None,
                "episode_streaming": True,
                "validation_batches": 1_600,
                "validation_skipped": False,
            },
            "runtime": {
                "stable_worldmodel": "0.1.1",
                "torch": "2.5.1+cu124",
                "python": "3.10.13",
                "platform": "Linux-6.5-x86_64",
                "tdwm_git_revision": _sha(f"training-{variant}")[:40],
                "compatibility_adapter": None,
                "cuda_device": "NVIDIA GeForce RTX 4090",
            },
        },
    )
    _write_lightning_metrics(variant_root / "metrics.csv", variant_index)


def _checkpoint_config(formal: dict, variant: str) -> dict:
    head_key = "critic" if variant == "direct_goal_hybrid" else "successor"
    head = formal[head_key]
    config = {
        "method": "actor_free_td_lewm",
        "variant": variant,
        "objective_version": head["objective_version"],
        "deployment_checkpoint_version": 1,
        "architecture": head["architecture"],
        "embed_dim": formal["model"]["embed_dim"],
        "action_dim": 25,
        "history_size": head["history_size"],
        "hidden_dim": head["hidden_dim"],
        "gamma": head["gamma"],
        "action_conditioning": head["action_conditioning"],
        "bootstrap_action": head["bootstrap_action"],
        "terminal_source": head["terminal_source"],
        "goal_conditioning": head["goal_conditioning"],
        "actor": "none",
        "reward": "none",
    }
    if head_key == "successor":
        config["feature_basis"] = head["feature_basis"]
    if variant == "goal_hybrid":
        config.update(
            {
                "goal_readout_training": True,
                "goal_source": head["goal_source"],
                "goal_offset_weighting": head["goal_offset_weighting"],
                "goal_terminal_condition": head["goal_terminal_condition"],
                "goal_readout_branches": head["goal_readout_branches"],
                "goal_readout_precision": head["goal_readout_precision"],
                "goal_cost": head["goal_cost"],
                "goal_enters_successor_head": False,
                "predicted_goal_td_weight": 1.0,
                "real_goal_td_weight": 1.0,
            }
        )
    elif variant == "imaginary_hybrid":
        config.update(
            {
                "immediate_feature_source": head["immediate_feature_source"],
                "bootstrap_state_source": head["bootstrap_state_source"],
                "imaginary_horizon": head["imaginary_horizon"],
                "imaginary_predictor_gradient": head[
                    "imaginary_predictor_gradient"
                ],
            }
        )
    elif variant == "direct_goal_hybrid":
        config.update(
            {
                "goal_source": head["goal_source"],
                "goal_offset_weighting": head["goal_offset_weighting"],
                "goal_terminal_condition": head["goal_terminal_condition"],
                "td_branches": head["td_branches"],
                "goal_cost": head["goal_cost"],
                "goal_enters_critic_head": True,
                "predicted_context_detach": False,
                "predicted_critic_td_weight": 1.0,
                "real_critic_td_weight": 1.0,
            }
        )
    return config


def _success_count(variant_index: int, mode_index: int) -> int:
    return 12 + 3 * variant_index + mode_index


def _make_bundle(root: Path, *, legacy_combined: bool = False) -> Path:
    selection = _selection()
    for variant_index, variant in enumerate(VARIANT_ORDER):
        variant_root = root / variant
        _make_training(variant_root, variant, variant_index)
        checkpoint_sha = _sha(f"checkpoint-{variant}")
        combined = combined_mode_for_variant(variant)
        for mode_index, mode in enumerate(modes_for_variant(variant)):
            is_legacy = legacy_combined and mode == combined
            formal = _config(variant, training=False)
            if is_legacy:
                del formal["inference_objective"]["score_mode"]
            configured = deepcopy(formal)
            if not is_legacy:
                configured["inference_objective"]["score_mode"] = mode
            count = _success_count(variant_index, mode_index)
            outcomes = [index < count for index in range(50)]
            run_root = variant_root / mode
            _write_json(run_root / "episode_selection.json", selection)
            head_key = "critic" if variant == "direct_goal_hybrid" else "successor"
            checkpoint = {
                "path": (
                    f"/srv/runs/actor-free/{variant}/checkpoints/actor_free_td_lewm/"
                    f"{variant}/epoch_10.pt"
                ),
                "sha256": checkpoint_sha,
                "method": "actor_free_td_lewm",
                "variant": variant,
                "objective_version": formal[head_key]["objective_version"],
                (
                    "critic_config"
                    if variant == "direct_goal_hybrid"
                    else "successor_config"
                ): _checkpoint_config(formal, variant),
            }
            manifest = {
                "protocol": configured,
                "formal_protocol": formal,
                "protocol_path": (
                    f"/srv/repo/configs/experiment/actor_free_td_lewm_{variant}_"
                    "cube_checkpoint_o50.yaml"
                ),
                "dataset": _evaluation_dataset(),
                "checkpoint": checkpoint,
                "selection": selection,
                "normalization": {
                    "action": {"mean": [0.0] * 5, "std": [1.0] * 5}
                },
                "runtime": {
                    "stable_worldmodel": "0.1.1",
                    "torch": "2.5.1+cu124",
                    "python": "3.10.13",
                    "platform": "Linux-6.5-x86_64",
                    "tdwm_git_revision": _sha(f"evaluation-{variant}-{mode}")[:40],
                    "device": "cuda",
                    "stablewm_home": "/srv/stablewm",
                    "compatibility_adapter": None,
                    "cuda_device": "NVIDIA GeForce RTX 4090",
                },
            }
            result = {
                "metrics": {
                    "episode_successes": outcomes,
                    "success_rate": count / 50,
                },
                "elapsed_seconds": 10.0 + variant_index + mode_index,
                "world_model_parameter_count": WORLD_PARAMETERS,
                (
                    "critic_parameter_count"
                    if variant == "direct_goal_hybrid"
                    else "successor_parameter_count"
                ): _head_count(variant_index),
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


def _rewrite_csv(path: Path, mutate) -> None:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    mutate(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_complete_real_style_bundle_generates_auditable_archive(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle", legacy_combined=True)
    study = validate_bundle(bundle)
    artifact_dir = tmp_path / "reports/artifacts/actor_free_td_lewm_cube_seed3072"
    report = tmp_path / "reports/actor_free_td_lewm_cube_seed3072.md"

    paths = write_archive(study, artifact_dir=artifact_dir, report_path=report)

    assert len(paths) == 7
    summary = _read_json(artifact_dir / "summary.json")
    assert summary["study"]["evaluation_count"] == 21
    assert summary["validation"]["smoke_or_pilot_runs"] == 0
    assert all(
        value is True
        for key, value in summary["validation"].items()
        if key != "smoke_or_pilot_runs"
    )
    assert len(summary["ranking_by_combined"]) == 7
    assert summary["ranking_by_combined"][0]["variant"] == "direct_goal_hybrid"
    assert summary["selection"]["episode_selection_json_sha256"] == SELECTION_SHA256
    for variant in VARIANT_ORDER:
        combined = combined_mode_for_variant(variant)
        method = summary["methods"][variant]
        assert (
            method["evaluations"][combined]["score_mode_source"]
            == "legacy_combined_default"
        )
        assert len(method["training"]["loss_curve"]) == 10
        metrics_hash = hashlib.sha256(
            (bundle / variant / "metrics.csv").read_bytes()
        ).hexdigest()
        assert method["training"]["source_files_sha256"]["metrics.csv"] == metrics_hash

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

    write_archive(study, artifact_dir=artifact_dir, report_path=report, check=True)
    for line in (artifact_dir / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((artifact_dir / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_explicit_new_score_outputs_are_recorded(tmp_path):
    study = validate_bundle(_make_bundle(tmp_path / "bundle"))
    assert all(
        run.score_mode_source == "explicit"
        for runs in study.evaluations.values()
        for run in runs.values()
    )


def test_old_combined_can_keep_hardcoded_mode_inside_protocol(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    run_root = bundle / "hybrid/f_plus_g"
    result = _read_json(run_root / "results.json")
    manifest = _read_json(run_root / "protocol_manifest.json")
    del result["score_mode"]
    del manifest["score_mode"]
    _write_json(run_root / "results.json", result)
    _write_json(run_root / "protocol_manifest.json", manifest)

    study = validate_bundle(bundle)
    assert (
        study.evaluations["hybrid"]["f_plus_g"].score_mode_source
        == "legacy_combined_default"
    )


def test_legacy_success_and_matching_dual_fields_are_accepted(tmp_path):
    bundle = _make_bundle(tmp_path / "legacy")
    target = bundle / "hybrid/g_only/results.json"
    result = _read_json(target)
    result["metrics"]["success"] = result["metrics"].pop("episode_successes")
    _write_json(target, result)
    validate_bundle(bundle)

    bundle = _make_bundle(tmp_path / "dual")
    target = bundle / "hybrid/g_only/results.json"
    result = _read_json(target)
    result["metrics"]["success"] = list(result["metrics"]["episode_successes"])
    _write_json(target, result)
    validate_bundle(bundle)


def test_inconsistent_canonical_and_legacy_success_fields_are_rejected(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    target = bundle / "hybrid/g_only/results.json"
    result = _read_json(target)
    result["metrics"]["success"] = list(result["metrics"]["episode_successes"])
    result["metrics"]["success"][0] = not result["metrics"]["success"][0]
    _write_json(target, result)

    with pytest.raises(BundleValidationError, match="episode_successes.*disagree"):
        validate_bundle(bundle)


def test_combined_ranking_uses_shared_ranks_for_ties(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    target = bundle / "serial_coupled/f_plus_g/results.json"
    result = _read_json(target)
    result["metrics"]["episode_successes"] = [index < 14 for index in range(50)]
    result["metrics"]["success_rate"] = 14 / 50
    _write_json(target, result)

    ranking = build_summary(validate_bundle(bundle))["ranking_by_combined"]
    by_variant = {row["variant"]: row["rank"] for row in ranking}
    assert by_variant["serial_decoupled"] == by_variant["serial_coupled"]


def test_non_combined_mode_cannot_use_old_implicit_metadata(tmp_path):
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


def test_same_selection_content_with_noncanonical_bytes_is_rejected(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/g_only/episode_selection.json"
    path.write_text(json.dumps(_read_json(path), sort_keys=True) + "\n")

    with pytest.raises(BundleValidationError, match="SHA-256 must equal locked"):
        validate_bundle(bundle)


def test_wrong_seed_selection_is_rejected(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    valid_per_episode = 151
    ranks = np.sort(
        np.random.default_rng(43).choice(10_000 * valid_per_episode - 1, 50, False)
    )
    wrong = {
        "episode_indices": (ranks // valid_per_episode).astype(int).tolist(),
        "start_steps": (ranks % valid_per_episode).astype(int).tolist(),
        "goal_steps": ((ranks % valid_per_episode) + 50).astype(int).tolist(),
        "valid_row_ranks": ranks.astype(int).tolist(),
    }
    run_root = bundle / "hybrid/g_only"
    manifest = _read_json(run_root / "protocol_manifest.json")
    manifest["selection"] = wrong
    _write_json(run_root / "protocol_manifest.json", manifest)
    _write_json(run_root / "episode_selection.json", wrong)

    with pytest.raises(BundleValidationError, match="seed-42 O50 sample"):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("episode_indices", 10_000, "episode_indices"),
        ("start_steps", 151, "start < goal < 201"),
        ("goal_steps", 201, "start < goal < 201"),
    ),
)
def test_selection_cube_bounds_are_rejected(tmp_path, field, value, message):
    bundle = _make_bundle(tmp_path / "bundle")
    run_root = bundle / "hybrid/g_only"
    selection = _read_json(run_root / "episode_selection.json")
    selection[field][0] = value
    if field == "start_steps":
        selection["goal_steps"][0] = 201
    manifest = _read_json(run_root / "protocol_manifest.json")
    manifest["selection"] = selection
    _write_json(run_root / "protocol_manifest.json", manifest)
    _write_json(run_root / "episode_selection.json", selection)

    with pytest.raises(BundleValidationError, match=message):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("runtime", "precision", "fp16"),
        ("image_preprocessing", "source", "other"),
        ("dataset", "expected_size_bytes", 1),
        ("model", "embed_dim", 384),
        ("world", "env_name", "other"),
        ("evaluation", "start_goal_source", "other"),
        ("planning", "solver", "MPPI"),
        ("planning", "history_len", 2),
    ),
)
def test_complete_formal_protocol_lock_rejects_drift(
    tmp_path, section, key, value
):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/g_only/protocol_manifest.json"
    manifest = _read_json(path)
    manifest["protocol"][section][key] = value
    manifest["formal_protocol"][section][key] = value
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="formal Cube O50 lock"):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("runtime", "critical runtime fingerprint"),
        ("dataset", "dataset source/provenance fingerprint"),
        ("normalization", "action normalization fingerprint"),
    ),
)
def test_21_run_manifest_fingerprints_reject_drift(tmp_path, kind, message):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/g_only/protocol_manifest.json"
    manifest = _read_json(path)
    if kind == "runtime":
        manifest["runtime"]["torch"] = "2.6.0"
    elif kind == "dataset":
        manifest["dataset"]["path"] = "/srv/datasets/a-different-copy.lance"
    else:
        manifest["normalization"]["action"]["mean"][0] = 0.5
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match=message):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("protocol", "training common protocol fingerprint"),
        ("runtime", "training critical runtime fingerprint"),
        ("dataset", "training dataset source/provenance fingerprint"),
    ),
)
def test_seven_training_run_fingerprints_reject_drift(tmp_path, kind, message):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/training_manifest.json"
    manifest = _read_json(path)
    if kind == "protocol":
        manifest["protocol"]["loader"]["batch_size"] = 16
    elif kind == "runtime":
        manifest["runtime"]["torch"] = "2.6.0"
    else:
        manifest["dataset"]["path"] = "/srv/datasets/a-different-copy.lance"
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match=message):
        validate_bundle(bundle)


def test_world_parameter_count_is_global_not_only_per_method(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    variant = "hybrid"
    for mode in modes_for_variant(variant):
        path = bundle / variant / mode / "results.json"
        result = _read_json(path)
        result["world_model_parameter_count"] += 1
        _write_json(path, result)
    manifest_path = bundle / variant / "training_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["model"]["lewm_parameters"] += 1
    manifest["protocol"]["model"]["parameters"] += 1
    _write_json(manifest_path, manifest)

    with pytest.raises(BundleValidationError, match="identical across all 21"):
        validate_bundle(bundle)


def test_variant_specific_heads_are_not_incorrectly_cross_compared(tmp_path):
    study = validate_bundle(_make_bundle(tmp_path / "bundle"))
    assert study.evaluations["goal_hybrid"]["g_only"].head_parameter_count != (
        study.evaluations["hybrid"]["g_only"].head_parameter_count
    )
    assert study.evaluations["direct_goal_hybrid"]["c_only"].head_parameter_count != (
        study.evaluations["goal_hybrid"]["g_only"].head_parameter_count
    )


def test_full_variant_protocol_must_match_across_its_three_modes(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/g_only/protocol_manifest.json"
    manifest = _read_json(path)
    manifest["protocol"]["provenance"]["status"] = "drifted"
    manifest["formal_protocol"]["provenance"]["status"] = "drifted"
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="different formal protocols"):
        validate_bundle(bundle)


def test_goal_checkpoint_config_is_validated_like_the_formal_evaluator(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "goal_hybrid/g_only/protocol_manifest.json"
    manifest = _read_json(path)
    del manifest["checkpoint"]["successor_config"]["goal_terminal_condition"]
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="goal_terminal_condition"):
        validate_bundle(bundle)


def test_training_curve_is_derived_from_real_sparse_lightning_columns(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    study = validate_bundle(bundle)
    curve = study.training["hybrid"].curve

    assert [row["epoch"] for row in curve] == list(range(1, 11))
    assert curve[-1]["train_loss"] == pytest.approx(1.0 / 13)
    assert curve[-1]["validation_loss"] == pytest.approx(1.2 / 13)
    assert study.training["hybrid"].metrics["final_epoch"]["epoch"] == 10


def test_training_metrics_requires_ten_validation_aggregates(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/metrics.csv"

    def remove_epoch_nine_validation(rows):
        for row in rows:
            if row["epoch"] == "9" and row["validation/loss"]:
                row["validation/loss"] = ""

    _rewrite_csv(path, remove_epoch_nine_validation)

    with pytest.raises(BundleValidationError, match="validation/loss for epochs 0..9"):
        validate_bundle(bundle)


def test_training_metrics_and_result_zero_based_step_must_match(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/training_result.json"
    result = _read_json(path)
    result["global_step"] = 127_959
    _write_json(path, result)

    with pytest.raises(BundleValidationError, match="global_step must equal 127960"):
        validate_bundle(bundle)


def test_training_raw_files_are_required(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    (bundle / "hybrid/training_manifest.json").unlink()

    with pytest.raises(BundleValidationError, match="missing required file"):
        validate_bundle(bundle)


def test_training_export_path_must_match_evaluation_checkpoint(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "hybrid/g_only/protocol_manifest.json"
    manifest = _read_json(path)
    manifest["checkpoint"]["path"] = "/srv/wrong/epoch_10.pt"
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="different checkpoint paths"):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("checkpoint", "different checkpoints"),
        ("smoke", "smoke results cannot"),
        ("rate", "disagrees with"),
    ),
)
def test_bundle_rejects_invalid_evaluation_evidence(tmp_path, mutation, message):
    bundle = _make_bundle(tmp_path / "bundle")
    run_root = bundle / "serial_decoupled/g_only"
    if mutation == "checkpoint":
        manifest = _read_json(run_root / "protocol_manifest.json")
        manifest["checkpoint"]["sha256"] = _sha("wrong-checkpoint")
        _write_json(run_root / "protocol_manifest.json", manifest)
    else:
        result = _read_json(run_root / "results.json")
        if mutation == "smoke":
            result["smoke"] = True
        else:
            result["metrics"]["success_rate"] = 0.98
        _write_json(run_root / "results.json", result)

    with pytest.raises(BundleValidationError, match=message):
        validate_bundle(bundle)
