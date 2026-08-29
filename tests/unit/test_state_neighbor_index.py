from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdwm.training.state_neighbor_index import (
    INDEX_METHOD,
    INDEX_SCHEMA_VERSION,
    StateNeighborActionIndex,
)

SOURCE_HASH = "7" * 64
LATENT_STORE_MANIFEST_SHA256 = "6" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(root: Path, *, same_episode: bool = False) -> Path:
    root.mkdir()
    arrays = {
        "global_rows": np.asarray([10, 20, 30, 40], dtype=np.int64),
        "episode_ids": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "actions": np.arange(4 * 6, dtype=np.float32).reshape(4, 6),
        "neighbor_indices": np.asarray(
            [[1, 2], [0, 2], [0, 1], [1, 2]], dtype=np.int32
        ),
        "neighbor_distances": np.asarray(
            [[0.1, 0.2], [0.1, 0.3], [0.2, 0.3], [0.4, 0.5]],
            dtype=np.float32,
        ),
    }
    if same_episode:
        arrays["episode_ids"][1] = 0
    files = {}
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        files[name] = {"path": path.name, "sha256": _sha256(path)}
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "method": INDEX_METHOD,
        "pretrained_world_model_sha256": SOURCE_HASH,
        "source_bank": {
            "latent_store_manifest_sha256": LATENT_STORE_MANIFEST_SHA256,
        },
        "metric": "squared_l2",
        "retrieval_key": "frozen_current_state_latent",
        "split": "training_only",
        "exclude_exact_transition": True,
        "exclude_same_episode": True,
        "action_block_dim": 6,
        "neighbors_per_anchor": 2,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_lookup_gathers_only_precomputed_neighbor_action_blocks(tmp_path: Path):
    index = StateNeighborActionIndex(
        _write_index(tmp_path / "index"),
        expected_checkpoint_sha256=SOURCE_HASH,
        expected_latent_store_manifest_sha256=LATENT_STORE_MANIFEST_SHA256,
        expected_action_block_dim=6,
        expected_k=2,
    )

    batch = index.lookup(
        torch.tensor([[10, 30]], dtype=torch.int64),
        device="cpu",
        dtype=torch.float32,
    )

    assert batch.actions.shape == (1, 2, 2, 6)
    assert batch.distances.shape == (1, 2, 2)
    assert batch.neighbor_rows.tolist() == [[[20, 30], [10, 20]]]
    assert torch.equal(batch.actions[0, 0, 0], torch.arange(6, 12).float())


def test_lookup_rejects_rows_not_covered_by_training_index(tmp_path: Path):
    index = StateNeighborActionIndex(
        _write_index(tmp_path / "index"),
        expected_checkpoint_sha256=SOURCE_HASH,
        expected_latent_store_manifest_sha256=LATENT_STORE_MANIFEST_SHA256,
        expected_action_block_dim=6,
    )

    with pytest.raises(KeyError, match="99"):
        index.lookup(torch.tensor([99]), device="cpu", dtype=torch.float32)


def test_lookup_rejects_same_episode_neighbor_even_if_manifest_claims_exclusion(
    tmp_path: Path,
):
    index = StateNeighborActionIndex(
        _write_index(tmp_path / "index", same_episode=True),
        expected_checkpoint_sha256=SOURCE_HASH,
        expected_latent_store_manifest_sha256=LATENT_STORE_MANIFEST_SHA256,
        expected_action_block_dim=6,
    )

    with pytest.raises(ValueError, match="same-episode"):
        index.lookup(torch.tensor([10]), device="cpu", dtype=torch.float32)


def test_index_binds_actions_to_exact_pretrained_checkpoint(tmp_path: Path):
    root = _write_index(tmp_path / "index")
    with pytest.raises(ValueError, match="different frozen LeWM"):
        StateNeighborActionIndex(
            root,
            expected_checkpoint_sha256="8" * 64,
            expected_latent_store_manifest_sha256=LATENT_STORE_MANIFEST_SHA256,
            expected_action_block_dim=6,
        )


def test_index_binds_neighbors_to_exact_frozen_latent_store(tmp_path: Path):
    root = _write_index(tmp_path / "index")

    with pytest.raises(ValueError, match="different frozen latent store"):
        StateNeighborActionIndex(
            root,
            expected_checkpoint_sha256=SOURCE_HASH,
            expected_latent_store_manifest_sha256="9" * 64,
            expected_action_block_dim=6,
        )
