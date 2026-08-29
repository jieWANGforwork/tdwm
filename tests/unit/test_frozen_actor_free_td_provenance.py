from __future__ import annotations

import json

import numpy as np
import pytest

from tdwm.training.frozen_actor_free_td import (
    _resolve_bound_frozen_latent_store,
    _verify_completed_pretrained_lewm_run,
)
from tdwm.training.frozen_latent_store import (
    FrozenLatentStoreSpec,
    build_frozen_latent_store,
    file_sha256,
)


def test_frozen_source_requires_epoch10_and_completed_formal_training(tmp_path):
    run_dir = tmp_path / "seed_3072"
    source_cache = run_dir / "checkpoints" / "exports"
    source_cache.mkdir(parents=True)
    (run_dir / "training_result.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "seed": 3072,
                "smoke": False,
                "final_epoch": 10,
                "global_step": 127_960,
            }
        )
    )
    (run_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "seed": 3072,
                "smoke": False,
                "protocol": {"method": "lewm", "training": {"epochs": 10}},
                "training": {
                    "formal_optimizer_steps": 127_960,
                    "configured_optimizer_steps": 127_960,
                },
            }
        )
    )

    provenance = _verify_completed_pretrained_lewm_run(
        source_run_name="epoch_10",
        source_cache=source_cache,
        expected_seed=3072,
        expected_epoch=10,
    )

    assert provenance["source_final_epoch"] == 10
    assert provenance["source_global_step"] == 127_960
    with pytest.raises(ValueError, match="must be epoch_10"):
        _verify_completed_pretrained_lewm_run(
            source_run_name="epoch_01",
            source_cache=source_cache,
            expected_seed=3072,
            expected_epoch=10,
        )


def test_frozen_latent_training_binds_store_and_external_input_hashes(tmp_path):
    checkpoint_sha256 = "a" * 64
    dataset_sha256 = "b" * 64
    dataset_manifest = tmp_path / "cube.lance.manifest.json"
    dataset_manifest.write_text('{"fixture": true}\n')
    normalization = tmp_path / "column_normalization.json"
    normalization.write_text('{"action": {"fixture": true}}\n')
    total_rows = 31
    store_root = tmp_path / "frozen_store"
    spec = FrozenLatentStoreSpec(
        total_rows=total_rows,
        embed_dim=4,
        frame_skip=5,
        history_frames=3,
        action_dim=5,
        pretrained_checkpoint_sha256=checkpoint_sha256,
        dataset_source_sha256=dataset_sha256,
        column_normalization_sha256=file_sha256(normalization),
        git_revision="c" * 40,
    )
    build_frozen_latent_store(
        store_root,
        spec=spec,
        encoded_batches=[
            (
                np.arange(total_rows, dtype=np.int64),
                np.zeros((total_rows, 4), dtype=np.float32),
            )
        ],
        normalized_actions=np.zeros((total_rows, 5), dtype=np.float32),
        episode_ids=np.zeros(total_rows, dtype=np.int64),
        source_metadata={
            "dataset_manifest_sha256": file_sha256(dataset_manifest),
            "column_normalization_path": str(normalization),
            "stable_worldmodel_version": "0.1.1",
            "extraction_precision": "bfloat16",
            "image_preprocessing": {
                "size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "online_cache_parity_audit": {
                "status": "passed_by_construction_for_every_global_row",
                "formal_preprocessing_smoke": {"status": "passed"},
            },
        },
    )
    protocol = {
        "pretrained_world_model": {"checkpoint_sha256": checkpoint_sha256},
        "sequence": {"frame_skip": 5, "history_frames": 3},
        "model": {"embed_dim": 4},
        "runtime": {"stable_worldmodel_version": "0.1.1"},
        "training": {"precision": "bf16-mixed"},
        "image_preprocessing": {
            "size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    dataset_source = {
        "format": "lance",
        "conversion_manifest_path": str(dataset_manifest),
        "conversion_manifest": {"source": {"sha256": dataset_sha256}},
    }

    store, identity = _resolve_bound_frozen_latent_store(
        store_root,
        protocol=protocol,
        dataset_source=dataset_source,
        action_dim=5,
    )

    assert identity["manifest_sha256"] == store.manifest_sha256
    assert identity["dataset_manifest_sha256"] == file_sha256(dataset_manifest)
    assert identity["column_normalization_sha256"] == file_sha256(normalization)
    assert set(identity["input_file_sha256"]) == {
        "latents",
        "action_blocks",
        "episode_ids",
    }
    normalization.write_text('{"action": {"changed": true}}\n')
    with pytest.raises(ValueError, match="column normalization differs"):
        _resolve_bound_frozen_latent_store(
            store_root,
            protocol=protocol,
            dataset_source=dataset_source,
            action_dim=5,
        )
