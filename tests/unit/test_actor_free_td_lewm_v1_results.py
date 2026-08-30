from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tdwm.results.actor_free_td_lewm_v1 import (
    EXPECTED_ACTION_ENCODER_SHA256,
    FORMAL_HORIZON_BY_SCORE_MODE,
    PREDICTOR_PARAMETERS,
    SCORE_MODES,
    SELECTION_SHA256,
    VARIANT_ORDER,
    BundleValidationError,
    build_summary,
    validate_bundle,
    write_archive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORLD_PARAMETERS = 18_034_628
PRETRAINED_SHA256 = "198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _protocol(variant: str, *, training: bool) -> dict:
    suffix = "cube_train" if training else "cube_checkpoint_o50"
    path = (
        REPOSITORY_ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v1_{variant}_{suffix}.yaml"
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
    result = {
        "episode_indices": [int(row["episode_index"]) for row in rows],
        "start_steps": [int(row["start_step"]) for row in rows],
        "goal_steps": [int(row["goal_step"]) for row in rows],
        "valid_row_ranks": [int(row["valid_row_rank"]) for row in rows],
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(encoded).hexdigest() == SELECTION_SHA256
    return result


def _execution_evidence(variant: str) -> dict:
    command = (
        "python scripts/train_actor_free_td_lewm_v1_"
        f"{variant}.py --seed 3072 --resume auto"
    )
    return {
        "schema_version": 1,
        "source": "external_execution_evidence",
        "method": f"actor_free_td_lewm_v1_{variant}",
        "variant": variant,
        "hostname": "autodl-container",
        "gpu": {
            "index": VARIANT_ORDER.index(variant),
            "name": "NVIDIA GeForce RTX 4090",
            "uuid": f"GPU-test-{variant}",
        },
        "process": {
            "pid": 1000 + VARIANT_ORDER.index(variant),
            "command": command,
            "command_sha256": _sha(command),
            "cwd": "/srv/repo/tdwm-v1-action-encoder",
            "git_revision": _sha("training-commit")[:40],
            "started_at": "2026-08-31T00:00:00+08:00",
            "ended_at": "2026-08-31T06:00:00+08:00",
            "exit_code": 0,
        },
        "log": {
            "path": f"/srv/logs/{variant}.log",
            "size_bytes": 1234,
            "sha256": _sha(f"log-{variant}"),
        },
        "gpu_process_snapshot": {
            "path": f"/srv/logs/{variant}.nvidia-smi.txt",
            "captured_at": "2026-08-31T00:10:00+08:00",
            "sha256": _sha(f"snapshot-{variant}"),
        },
        "trainer_recording_gaps": {
            "peak_cuda_memory_bytes": "not_recorded_by_v1_trainer",
            "runtime.cuda_device": "not_recorded_by_v1_trainer",
        },
    }


def _write_metrics(path: Path, variant_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "step",
        "train/loss_epoch",
        "train/base_td_loss_epoch",
        "validation/loss",
        "validation/base_td_loss",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for zero_epoch in range(10):
            step = (zero_epoch + 1) * 12_796 - 1
            train_method = 1.0 / (zero_epoch + variant_index + 2)
            train_base = 0.8 / (zero_epoch + variant_index + 2)
            validation_base = 0.9 / (zero_epoch + variant_index + 2)
            writer.writerow(
                {
                    "epoch": zero_epoch,
                    "step": step,
                    "validation/loss": validation_base,
                    "validation/base_td_loss": validation_base,
                }
            )
            writer.writerow(
                {
                    "epoch": zero_epoch,
                    "step": step,
                    "train/loss_epoch": train_method,
                    "train/base_td_loss_epoch": train_base,
                }
            )


def _training_dataset(variant: str) -> dict:
    return {
        "path": "/srv/data/cube.lance",
        "format": "lance",
        "size_bytes": 74_104_077_358,
        "conversion_manifest_path": "/srv/data/cube.lance.manifest.json",
        "conversion_manifest": {
            "source": {"sha256": _sha("dataset-source")},
            "destination": {"format": "lance"},
        },
        "sequence_samples": 1_279_600,
        "train_transition_population": {"size": 1_151_640},
        "validation_transition_population": {"size": 127_960},
        "cross_split_transition_overlap": 0,
        "cross_split_overlap_source": "inherited_sequence_clip_split",
        "split": {
            "path": f"/srv/runs/{variant}/split_indices.npz",
            "file_sha256": _sha("shared-split-file"),
            "train_samples": 1_151_640,
            "validation_samples": 127_960,
            "validation_array_key": "validation_indices",
            "train_indices_sha256": _sha("shared-train-indices"),
            "validation_indices_sha256": _sha("shared-validation-indices"),
            "array_hash_algorithm": "dtype_text_then_shape_text_then_c_order_bytes",
            "binding": "externally_supplied_exact_artifact",
        },
    }


def _make_training(root: Path, variant: str, variant_index: int) -> None:
    protocol = _protocol(variant, training=True)
    protocol_sha = _canonical_sha256(protocol)
    run_dir = f"/srv/runs/v1/{variant}"
    latent_sha = _sha("shared-frozen-latent-store")
    result = {
        "method": f"actor_free_td_lewm_v1_{variant}",
        "method_family": "actor_free_td_lewm_v1",
        "variant": variant,
        "implementation_version": "v1",
        "run_dir": run_dir,
        "seed": 3072,
        "last_checkpoint": f"{run_dir}/checkpoints/lightning/last.ckpt",
        "deployment_checkpoint": (
            f"{run_dir}/checkpoints/actor_free_td_lewm_v1_{variant}/"
            f"{variant}/epoch_10.pt"
        ),
        "final_epoch": 10,
        "global_step": 127_960,
        "protocol_sha256": protocol_sha,
        "pretrained_world_model_sha256": PRETRAINED_SHA256,
        "frozen_latent_store_manifest_sha256": latent_sha,
    }
    neighbor = None
    if variant == "g1":
        neighbor = {
            "path": "/srv/artifacts/g1-neighbors",
            "manifest_sha256": _sha("g1-neighbor-index"),
        }
        result["neighbor_index_manifest_sha256"] = neighbor["manifest_sha256"]
    _write_json(root / "training_result.json", result)
    _write_json(
        root / "training_manifest.json",
        {
            "method": f"actor_free_td_lewm_v1_{variant}",
            "method_family": "actor_free_td_lewm_v1",
            "variant": variant,
            "implementation_version": "v1",
            "objective_version": 0,
            "deployment_checkpoint_version": 1,
            "protocol": protocol,
            "protocol_path": (
                f"/srv/repo/configs/experiment/actor_free_td_lewm_v1_{variant}_"
                "cube_train.yaml"
            ),
            "protocol_sha256": protocol_sha,
            "seed": 3072,
            "dataset": _training_dataset(variant),
            "frozen_latent_store": {
                "path": "/srv/artifacts/frozen-latents",
                "manifest_path": "/srv/artifacts/frozen-latents/manifest.json",
                "manifest_sha256": latent_sha,
                "pretrained_checkpoint_sha256": PRETRAINED_SHA256,
                "dataset_source_sha256": _sha("dataset-source"),
                "dataset_manifest_path": "/srv/data/cube.lance.manifest.json",
                "dataset_manifest_sha256": _sha("dataset-manifest"),
                "column_normalization_path": "/srv/data/normalization.json",
                "column_normalization_sha256": _sha("normalization"),
                "input_file_sha256": {
                    "latents": _sha("latents"),
                    "action_blocks": _sha("action-blocks"),
                    "episode_ids": _sha("episode-ids"),
                },
                "total_rows": 2_010_000,
                "embed_dim": 192,
                "frame_skip": 5,
                "history_frames": 3,
                "action_dim": 5,
                "action_block_dim": 25,
                "git_revision": _sha("latent-store-git")[:40],
                "stable_worldmodel_version": "0.1.1",
                "extraction_precision": "bfloat16",
            },
            "neighbor_index": neighbor,
            "model": {
                "config": {"_target_": "stable_worldmodel.LeWM"},
                "initialization": {
                    "strategy": "frozen_pretrained_lewm",
                    "source_checkpoint_sha256": PRETRAINED_SHA256,
                    "frozen": True,
                },
                "lewm_parameters": WORLD_PARAMETERS,
                "trainable_lewm_parameters": 0,
                "predictor_parameters": PREDICTOR_PARAMETERS,
            },
            "training": {
                "formal_optimizer_steps": 127_960,
                "optimizer_steps_per_epoch": 12_796,
                "configured_optimizer_steps": 127_960,
                "available_batches_per_epoch": 12_796,
                "validation_batches": 500,
                "validation_skipped": False,
                "resumed_from": None,
                "data_source": "frozen_latent_store",
                "sampling_unit": "transition",
                "train_sampling": "random_with_replacement",
                "validation_goal_sampling": (
                    "uniform_reachable_future_fixed_per_epoch"
                ),
                "validation_task_sampling": "bernoulli_mixture_fixed_per_epoch",
                "validation_primary_objective": "common_base_td_all_variants",
                "transition_batch_size": 256,
                "world_model_visual_encode_during_training": False,
                "shared_action_encoder_forward_during_training": True,
            },
            # The current V1 trainer did not record runtime.cuda_device.
            "runtime": {
                "stable_worldmodel": "0.1.1",
                "torch": "2.5.1+cu124",
                "python": "3.10.13",
                "platform": "Linux-6.5-x86_64",
                "tdwm_git_revision": _sha("training-commit")[:40],
                "compatibility_adapter": None,
            },
        },
    )
    _write_json(root / "execution_evidence.json", _execution_evidence(variant))
    _write_metrics(root / "metrics.csv", variant_index)


def _predictor_config(protocol: dict, variant: str) -> dict:
    return {
        "method": f"actor_free_td_lewm_v1_{variant}",
        "method_family": "actor_free_td_lewm_v1",
        "variant": variant,
        "implementation_version": "v1",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        **deepcopy(protocol["predictor"]),
        "task_sampling": deepcopy(protocol["task_sampling"]),
        "joint_objective": deepcopy(protocol["joint_objective"]),
        "pretrained_world_model": deepcopy(protocol["pretrained_world_model"]),
    }


def _make_bundle(root: Path) -> Path:
    selection = _selection()
    action_stats = {
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "variance": [1.0] * 5,
        "samples": 2_010_000,
    }
    for variant_index, variant in enumerate(VARIANT_ORDER):
        variant_root = root / variant
        _make_training(variant_root, variant, variant_index)
        base_protocol = _protocol(variant, training=False)
        checkpoint_sha = _sha(f"checkpoint-{variant}")
        checkpoint_path = (
            f"/srv/runs/v1/{variant}/checkpoints/actor_free_td_lewm_v1_"
            f"{variant}/{variant}/epoch_10.pt"
        )
        for mode_index, mode in enumerate(SCORE_MODES):
            configured = deepcopy(base_protocol)
            configured["inference_objective"]["score_mode"] = mode
            configured["planning"]["horizon"] = FORMAL_HORIZON_BY_SCORE_MODE[mode]
            run_root = variant_root / mode
            _write_json(run_root / "episode_selection.json", selection)
            _write_json(run_root / "action_normalization.json", action_stats)
            checkpoint = {
                "path": checkpoint_path,
                "sha256": checkpoint_sha,
                "method": f"actor_free_td_lewm_v1_{variant}",
                "method_family": "actor_free_td_lewm_v1",
                "variant": variant,
                "implementation_version": "v1",
                "objective_version": 0,
                "epoch": 10,
                "global_step": 127_960,
                "formal_completion_required": True,
                "predictor_config": _predictor_config(base_protocol, variant),
                "pretrained_world_model_provenance": {
                    "source_checkpoint_sha256": PRETRAINED_SHA256,
                    "frozen": True,
                },
            }
            manifest = {
                "score_mode": mode,
                "protocol": configured,
                "formal_protocol": base_protocol,
                "protocol_path": (
                    f"/srv/repo/configs/experiment/actor_free_td_lewm_v1_"
                    f"{variant}_cube_checkpoint_o50.yaml"
                ),
                "dataset": {
                    "path": "/srv/data/cube.lance",
                    "format": "lance",
                    "size_bytes": 74_104_077_358,
                    "conversion_manifest_path": "/srv/data/cube.lance.manifest.json",
                    "episodes": 10_000,
                    "transitions": 2_010_000,
                },
                "checkpoint": checkpoint,
                "selection": selection,
                "normalization": {"action": action_stats},
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
            successes = 12 + variant_index * 3 + mode_index
            outcomes = [index < successes for index in range(50)]
            result = {
                "metrics": {
                    "episode_successes": outcomes,
                    "success_rate": successes * 100.0 / 50,
                },
                "elapsed_seconds": 10.0 + variant_index + mode_index,
                "world_model_parameter_count": WORLD_PARAMETERS,
                "predictor_parameter_count": PREDICTOR_PARAMETERS,
                "method": f"actor_free_td_lewm_v1_{variant}",
                "method_family": "actor_free_td_lewm_v1",
                "variant": variant,
                "implementation_version": "v1",
                "score_mode": mode,
                "planning_horizon": FORMAL_HORIZON_BY_SCORE_MODE[mode],
                "smoke": False,
                "pilot": False,
                "protocol_manifest": str(run_root / "protocol_manifest.json"),
            }
            _write_json(run_root / "protocol_manifest.json", manifest)
            _write_json(run_root / "results.json", result)
    variants = {}
    for variant in VARIANT_ORDER:
        variant_root = root / variant
        training_result = variant_root / "training_result.json"
        training_manifest = variant_root / "training_manifest.json"
        metrics = variant_root / "metrics.csv"
        result = _read_json(training_result)
        variants[variant] = {
            "checkpoint_path": result["deployment_checkpoint"],
            "checkpoint_sha256": _sha(f"checkpoint-{variant}"),
            "checkpoint_epoch": 10,
            "checkpoint_global_step": 127_960,
            "action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
            "training_result_path": str(training_result),
            "training_result_sha256": hashlib.sha256(
                training_result.read_bytes()
            ).hexdigest(),
            "training_manifest_path": str(training_manifest),
            "training_manifest_sha256": hashlib.sha256(
                training_manifest.read_bytes()
            ).hexdigest(),
            "metrics_files": [
                {
                    "path": str(metrics),
                    "sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
                }
            ],
            "metrics_epochs": list(range(1, 11)),
            "process": {
                "pid": 1000 + VARIANT_ORDER.index(variant),
                "state": "completed",
                "exit_code": 0,
                "exit_code_evidence": "launcher_record",
            },
        }
    _write_json(
        root / "training_acceptance.json",
        {
            "schema_version": 1,
            "generated_at_utc": "2026-08-31T00:00:00Z",
            "output_root": "/srv/runs/v1",
            "training_commit": _sha("training-commit")[:40],
            "seed": 3072,
            "status": "PASS",
            "expected_epoch": 10,
            "expected_global_step": 127_960,
            "expected_action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
            "variants": variants,
            "disk": {
                "outputs_bytes": 1_000_000,
                "free_bytes": 2_000_000,
                "min_free_bytes": 1_000_000,
            },
            "warnings": [],
            "errors": [],
        },
    )
    return root


def test_complete_v1_6x3_bundle_generates_archive_without_lewm_loss(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    study = validate_bundle(bundle)
    artifact_dir = tmp_path / "reports/artifacts/v1"
    report = tmp_path / "reports/v1.md"
    paths = write_archive(study, artifact_dir=artifact_dir, report_path=report)

    assert len(paths) == 7
    summary = _read_json(artifact_dir / "summary.json")
    assert summary["study"]["evaluation_count"] == 18
    assert summary["study"]["training_count"] == 6
    assert len(summary["ranking_by_f_plus_g"]) == 6
    assert summary["architecture"]["lewm_frozen"] is True
    assert summary["architecture"]["shared_action_encoder_frozen"] is True
    assert summary["architecture"]["predictor_parameters"] == PREDICTOR_PARAMETERS
    assert (
        summary["architecture"]["action_encoder_state_sha256"]
        == EXPECTED_ACTION_ENCODER_SHA256
    )
    serialized = json.dumps(summary)
    assert "L_LeWM" not in serialized
    assert "lewm_loss" not in serialized.lower()
    assert all(
        method["training"]["provenance"]["peak_cuda_memory_bytes"]["status"]
        == "not_recorded_by_v1_trainer"
        for method in summary["methods"].values()
    )

    with (artifact_dir / "paired_outcomes.csv").open(newline="") as stream:
        outcomes = list(csv.DictReader(stream))
    assert len(outcomes) == 50
    assert len(outcomes[0]) == 7 + 18
    with (artifact_dir / "training_loss_curves.csv").open(newline="") as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 60
    assert {row["train_metric_semantics"] for row in curves} == {
        "method_specific_objective"
    }
    assert {row["validation_metric_semantics"] for row in curves} == {"common_base_td"}
    text = report.read_text()
    assert "L_LeWM" not in text
    assert "frozen LeWM" in text
    assert "G-only" in text and "horizon 1" in text

    write_archive(study, artifact_dir=artifact_dir, report_path=report, check=True)


def test_summary_ranks_only_f_plus_g_not_posthoc_best_mode(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    # Make C excellent only in F-only; the formal combined ranking must ignore it.
    target = bundle / "c/f_only/results.json"
    result = _read_json(target)
    result["metrics"]["episode_successes"] = [True] * 50
    result["metrics"]["success_rate"] = 100.0
    _write_json(target, result)

    ranking = build_summary(validate_bundle(bundle))["ranking_by_f_plus_g"]
    assert ranking[0]["variant"] == "g3"


@pytest.mark.parametrize("mode", SCORE_MODES)
def test_each_score_mode_requires_its_locked_horizon(tmp_path, mode):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / f"c/{mode}/protocol_manifest.json"
    manifest = _read_json(path)
    manifest["protocol"]["planning"]["horizon"] = (
        5 if FORMAL_HORIZON_BY_SCORE_MODE[mode] == 1 else 1
    )
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="horizon"):
        validate_bundle(bundle)


def test_g_only_horizon_one_does_not_break_common_protocol_fingerprint(tmp_path):
    study = validate_bundle(_make_bundle(tmp_path / "bundle"))
    assert all(
        runs["g_only"].planning_horizon == 1
        and runs["f_only"].planning_horizon == 5
        and runs["f_plus_g"].planning_horizon == 5
        for runs in study.evaluations.values()
    )


def test_current_trainer_gaps_require_separate_external_evidence(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    (bundle / "d/execution_evidence.json").unlink()

    with pytest.raises(BundleValidationError, match="execution_evidence"):
        validate_bundle(bundle)


def test_training_acceptance_action_encoder_hash_is_fail_closed(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "training_acceptance.json"
    acceptance = _read_json(path)
    acceptance["variants"]["g2"]["action_encoder_sha256"] = _sha("wrong")
    _write_json(path, acceptance)

    with pytest.raises(BundleValidationError, match="action_encoder_sha256"):
        validate_bundle(bundle)


def test_reaped_process_without_exit_marker_is_preserved_as_warning(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "training_acceptance.json"
    acceptance = _read_json(path)
    acceptance["status"] = "PASS_WITH_WARNINGS"
    acceptance["warnings"] = [
        "launcher did not preserve process exit markers; epoch-10 artifacts are complete"
    ]
    for item in acceptance["variants"].values():
        item["process"]["exit_code"] = None
        item["process"]["exit_code_evidence"] = (
            "unavailable_process_reaped_without_marker"
        )
    _write_json(path, acceptance)

    summary = build_summary(validate_bundle(bundle))
    assert summary["training_acceptance"]["status"] == "PASS_WITH_WARNINGS"
    assert summary["training_acceptance"]["warnings"] == acceptance["warnings"]


def test_external_evidence_cannot_claim_unrecorded_peak_memory(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "d/execution_evidence.json"
    evidence = _read_json(path)
    evidence["trainer_recording_gaps"]["peak_cuda_memory_bytes"] = 8_000_000_000
    _write_json(path, evidence)

    with pytest.raises(BundleValidationError, match="not_recorded_by_v1_trainer"):
        validate_bundle(bundle)


def test_future_trainer_recorded_cuda_fields_are_preserved(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    result_path = bundle / "c/training_result.json"
    manifest_path = bundle / "c/training_manifest.json"
    evidence_path = bundle / "c/execution_evidence.json"
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    evidence = _read_json(evidence_path)
    result["peak_cuda_memory_bytes"] = 8_000_000_000
    manifest["runtime"]["cuda_device"] = "NVIDIA GeForce RTX 4090"
    evidence["trainer_recording_gaps"] = {
        "peak_cuda_memory_bytes": "recorded_by_v1_trainer",
        "runtime.cuda_device": "recorded_by_v1_trainer",
    }
    _write_json(result_path, result)
    _write_json(manifest_path, manifest)
    _write_json(evidence_path, evidence)
    acceptance_path = bundle / "training_acceptance.json"
    acceptance = _read_json(acceptance_path)
    acceptance["variants"]["c"]["training_result_sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    acceptance["variants"]["c"]["training_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _write_json(acceptance_path, acceptance)

    summary = build_summary(validate_bundle(bundle))
    provenance = summary["methods"]["c"]["training"]["provenance"]
    assert provenance["peak_cuda_memory_bytes"] == {
        "status": "recorded_by_v1_trainer",
        "value": 8_000_000_000,
        "source": "training_result.json",
        "execution_evidence_sha256": hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
    }
    assert provenance["runtime.cuda_device"]["status"] == ("recorded_by_v1_trainer")


def test_frozen_lewm_and_action_encoder_semantics_are_fail_closed(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "f/training_manifest.json"
    manifest = _read_json(path)
    manifest["model"]["trainable_lewm_parameters"] = 1
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="trainable_lewm_parameters"):
        validate_bundle(bundle)


def test_complete_locked_training_protocol_rejects_drift(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "g2/training_manifest.json"
    manifest = _read_json(path)
    manifest["protocol"]["joint_objective"]["prefix_slots"] = 4
    manifest["protocol_sha256"] = _canonical_sha256(manifest["protocol"])
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="locked training YAML"):
        validate_bundle(bundle)


def test_action_normalization_file_is_required_and_bound_to_manifest(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "g3/g_only/action_normalization.json"
    stats = _read_json(path)
    stats["mean"][0] = 0.25
    _write_json(path, stats)

    with pytest.raises(BundleValidationError, match="action_normalization"):
        validate_bundle(bundle)


def test_each_method_uses_one_checkpoint_across_three_modes(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    path = bundle / "g1/g_only/protocol_manifest.json"
    manifest = _read_json(path)
    manifest["checkpoint"]["sha256"] = _sha("wrong")
    _write_json(path, manifest)

    with pytest.raises(BundleValidationError, match="same checkpoint"):
        validate_bundle(bundle)


def test_incomplete_bundle_is_rejected_before_reporting(tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    (bundle / "g3/f_plus_g/results.json").unlink()

    with pytest.raises(BundleValidationError, match="missing required file"):
        validate_bundle(bundle)
