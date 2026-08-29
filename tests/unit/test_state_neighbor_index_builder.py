from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdwm.training.frozen_state_bank import (
    FORMAT_NAME as BANK_FORMAT_NAME,
)
from tdwm.training.frozen_state_bank import (
    ROW_MAPPING as BANK_ROW_MAPPING,
)
from tdwm.training.frozen_state_bank import (
    SCHEMA_VERSION as BANK_SCHEMA_VERSION,
)
from tdwm.training.state_neighbor_index import StateNeighborActionIndex
from tdwm.training.state_neighbor_index_builder import (
    ExactNumpySearchBackend,
    build_state_neighbor_index,
    load_frozen_state_bank,
)

CHECKPOINT_SHA256 = "a" * 64
SPLIT_SHA256 = "b" * 64
GIT_REVISION = "c" * 40


class RecordingExactBackend(ExactNumpySearchBackend):
    def __init__(self) -> None:
        super().__init__()
        self.query_sizes: list[int] = []

    def search(self, queries, search_depth):
        self.query_sizes.append(int(queries.shape[0]))
        return super().search(queries, search_depth)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bank(root: Path, *, unsorted: bool = False) -> Path:
    root.mkdir()
    global_rows = np.asarray([10, 11, 20, 21, 30, 40], dtype=np.int64)
    if unsorted:
        global_rows[[0, 1]] = global_rows[[1, 0]]
    arrays = {
        "global_rows": global_rows,
        "episode_ids": np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64),
        "latents": np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0],
             [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        ),
        "actions": np.arange(6 * 25, dtype=np.float32).reshape(6, 25),
    }
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", array, allow_pickle=False)
    file_entries = {
        name: {
            "path": f"{name}.npy",
            "sha256": _sha256(root / f"{name}.npy"),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "size_bytes": (root / f"{name}.npy").stat().st_size,
        }
        for name, array in arrays.items()
    }
    latent_manifest_sha = "d" * 64
    dataset_sha = "e" * 64
    normalization_sha = "f" * 64
    manifest = {
        "schema_version": BANK_SCHEMA_VERSION,
        "format": BANK_FORMAT_NAME,
        "row_mapping": BANK_ROW_MAPPING,
        "pretrained_world_model_sha256": CHECKPOINT_SHA256,
        "dataset_source_sha256": dataset_sha,
        "column_normalization_sha256": normalization_sha,
        "training_split_sha256": SPLIT_SHA256,
        "latent_store_manifest_sha256": latent_manifest_sha,
        "tdwm_git_revision": GIT_REVISION,
        "row_count": 6,
        "latent_dim": 2,
        "action_block_dim": 25,
        "source_latent_store": {
            "path": "/synthetic/latent-store",
            "manifest_path": "/synthetic/latent-store/manifest.json",
            "manifest_sha256": latent_manifest_sha,
            "total_rows": 41,
            "pretrained_checkpoint_sha256": CHECKPOINT_SHA256,
            "dataset_source_sha256": dataset_sha,
            "column_normalization_sha256": normalization_sha,
            "git_revision": GIT_REVISION,
            "embed_dim": 2,
            "frame_skip": 5,
            "history_frames": 3,
            "action_dim": 5,
            "action_block_dim": 25,
            "files": {
                "latents": {
                    "path": "latents.npy",
                    "sha256": "6" * 64,
                    "dtype": "float32",
                    "shape": [41, 2],
                    "size_bytes": 1,
                },
                "action_blocks": {
                    "path": "action_blocks.npy",
                    "sha256": "7" * 64,
                    "dtype": "float32",
                    "shape": [41, 25],
                    "size_bytes": 1,
                },
                "episode_ids": {
                    "path": "episode_ids.npy",
                    "sha256": "8" * 64,
                    "dtype": "int64",
                    "shape": [41],
                    "size_bytes": 1,
                },
            },
        },
        "training_split": {
            "path": "/synthetic/split_indices.npz",
            "file_sha256": "1" * 64,
            "train_samples": 4,
            "validation_samples": 1,
            "train_indices_sha256": SPLIT_SHA256,
            "validation_indices_sha256": "2" * 64,
        },
        "clip_metadata": {
            "clip_count": 5,
            "episode_count": 4,
            "clip_indices_dtype": "int64",
            "clip_indices_shape": [5, 2],
            "clip_indices_sha256": "3" * 64,
            "episode_offsets_dtype": "int64",
            "episode_offsets_shape": [4],
            "episode_offsets_sha256": "4" * 64,
            "episode_lengths_dtype": "int64",
            "episode_lengths_shape": [4],
            "episode_lengths_sha256": "5" * 64,
        },
        "index_semantics": {
            "split": "training_only",
            "split_unit": "sequence_clip_index",
            "anchor_source": "union_of_training_clip_td_current_rows",
            "history_frames": 3,
            "num_steps": 5,
            "frame_skip": 5,
            "first_current_step": 3,
            "last_current_step_inclusive": 3,
            "current_steps": [3],
            "global_row_formula": (
                "episode_offsets[episode] + clip_start + frame_skip * current_step"
            ),
            "deduplication": "sorted_unique_global_rows",
            "retrieval_key": "frozen_current_state_latent",
            "action_anchor": "same_global_row_five_slot_normalized_block",
            "terminal_action_policy": "reject_nonfinite_action_anchor",
        },
        "files": file_entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return root


def _build(bank: Path, output: Path, *, depth: int = 6, chunk: int = 2):
    backend = RecordingExactBackend()
    manifest = build_state_neighbor_index(
        bank_dir=bank,
        output_dir=output,
        pretrained_world_model_sha256=CHECKPOINT_SHA256,
        training_split_sha256=SPLIT_SHA256,
        git_revision=GIT_REVISION,
        neighbors=2,
        search_depth=depth,
        query_chunk_size=chunk,
        backend=backend,
        seed=3072,
        threads=1,
    )
    return manifest, backend


def test_exact_builder_writes_reader_compatible_filtered_graph(tmp_path: Path):
    bank = _write_bank(tmp_path / "bank")
    output = tmp_path / "graph"

    manifest, backend = _build(bank, output)

    assert backend.query_sizes == [2, 2, 2]
    assert manifest["pretrained_world_model_sha256"] == CHECKPOINT_SHA256
    assert manifest["training_split_sha256"] == SPLIT_SHA256
    assert manifest["tdwm_git_revision"] == GIT_REVISION
    assert manifest["construction"]["offline_only"] is True
    assert manifest["construction"]["seed"] == 3072
    assert manifest["construction"]["threads"] == 1
    assert manifest["construction"]["backend"]["name"] == "exact_numpy"
    assert manifest["libraries"]["numpy"] == np.__version__

    for name in (
        "global_rows",
        "episode_ids",
        "actions",
        "neighbor_indices",
        "neighbor_distances",
    ):
        path = output / f"{name}.npy"
        assert path.is_file()
        assert manifest["files"][name]["sha256"] == _sha256(path)

    indices = np.load(output / "neighbor_indices.npy")
    distances = np.load(output / "neighbor_distances.npy")
    # Row 10 excludes itself and row 11 (same episode), leaving rows 20/21.
    assert indices[0].tolist() == [2, 3]
    assert np.allclose(distances[0], [1.0, 1.21])
    episodes = np.load(output / "episode_ids.npy")
    assert np.all(episodes[indices] != episodes[:, None])
    assert np.all(indices != np.arange(indices.shape[0])[:, None])

    reader = StateNeighborActionIndex(
        output,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_latent_store_manifest_sha256="d" * 64,
        expected_action_block_dim=25,
        expected_k=2,
    )
    batch = reader.lookup(
        torch.tensor([10, 20]), device="cpu", dtype=torch.float32
    )
    assert batch.actions.shape == (2, 2, 25)
    assert batch.neighbor_rows.tolist() == [[20, 21], [11, 10]]


def test_build_is_deterministic_for_the_same_bank_seed_and_backend(tmp_path: Path):
    bank = _write_bank(tmp_path / "bank")
    first = tmp_path / "first"
    second = tmp_path / "second"

    _build(bank, first, chunk=1)
    _build(bank, second, chunk=3)

    for name in ("neighbor_indices.npy", "neighbor_distances.npy"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_insufficient_search_depth_reports_anchor_and_recovery(tmp_path: Path):
    bank = _write_bank(tmp_path / "bank")

    with pytest.raises(RuntimeError, match="global row 10.*increase search_depth"):
        _build(bank, tmp_path / "graph", depth=3)

    assert not (tmp_path / "graph").exists()
    assert not list(tmp_path.glob(".graph.tmp.*"))


def test_bank_validation_requires_strictly_sorted_global_rows(tmp_path: Path):
    bank = _write_bank(tmp_path / "bank", unsorted=True)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_frozen_state_bank(bank)


def test_manifest_binds_every_input_bank_file_by_sha256(tmp_path: Path):
    bank = _write_bank(tmp_path / "bank")
    output = tmp_path / "graph"

    manifest, _ = _build(bank, output)

    persisted = json.loads((output / "manifest.json").read_text())
    for name in ("global_rows", "episode_ids", "latents", "actions"):
        source = bank / f"{name}.npy"
        assert persisted["source_bank"]["files"][name]["sha256"] == _sha256(
            source
        )
    assert persisted["source_bank"]["manifest_sha256"] == _sha256(
        bank / "manifest.json"
    )
    assert persisted == manifest


def test_builder_rejects_cli_hashes_that_disagree_with_bank_manifest(
    tmp_path: Path,
):
    bank = _write_bank(tmp_path / "bank")

    with pytest.raises(ValueError, match="active protocol"):
        build_state_neighbor_index(
            bank_dir=bank,
            output_dir=tmp_path / "graph",
            pretrained_world_model_sha256="9" * 64,
            training_split_sha256=SPLIT_SHA256,
            git_revision=GIT_REVISION,
            neighbors=2,
            search_depth=6,
            query_chunk_size=2,
            backend=RecordingExactBackend(),
            seed=3072,
            threads=1,
        )
