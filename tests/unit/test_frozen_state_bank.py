from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdwm.training.frozen_latent_store import (
    FrozenLatentStore,
    FrozenLatentStoreSpec,
    build_frozen_latent_store,
)
from tdwm.training.frozen_state_bank import (
    array_sha256,
    build_frozen_training_state_bank,
    derive_training_anchor_rows,
    load_frozen_training_state_bank,
)

CHECKPOINT_SHA256 = "a" * 64
DATASET_SHA256 = "b" * 64
NORMALIZATION_SHA256 = "c" * 64
GIT_REVISION = "d" * 40


def _make_store(
    root: Path, *, nonfinite_action_row: int | None = None
) -> tuple[FrozenLatentStore, np.ndarray, np.ndarray]:
    total_rows = 80
    latents = np.arange(total_rows * 2, dtype=np.float32).reshape(total_rows, 2)
    actions = np.arange(total_rows * 5, dtype=np.float32).reshape(total_rows, 5)
    if nonfinite_action_row is not None:
        actions[nonfinite_action_row, 0] = np.nan
    episodes = np.repeat(np.arange(2, dtype=np.int64), 40)
    spec = FrozenLatentStoreSpec(
        total_rows=total_rows,
        embed_dim=2,
        frame_skip=5,
        history_frames=3,
        action_dim=5,
        pretrained_checkpoint_sha256=CHECKPOINT_SHA256,
        dataset_source_sha256=DATASET_SHA256,
        column_normalization_sha256=NORMALIZATION_SHA256,
        git_revision=GIT_REVISION,
    )
    build_frozen_latent_store(
        root,
        spec=spec,
        encoded_batches=[(np.arange(total_rows, dtype=np.int64), latents)],
        normalized_actions=actions,
        episode_ids=episodes,
    )
    store = FrozenLatentStore(
        root,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_dataset_source_sha256=DATASET_SHA256,
        expected_column_normalization_sha256=NORMALIZATION_SHA256,
        expected_frame_skip=5,
        expected_history_frames=3,
        expected_embed_dim=2,
        expected_action_dim=5,
    )
    return store, latents, episodes


def _make_split(path: Path) -> tuple[np.ndarray, str]:
    train = np.asarray([0, 1, 2], dtype=np.int64)
    validation = np.asarray([3], dtype=np.int64)
    np.savez_compressed(
        path,
        train_indices=train,
        validation_indices=validation,
    )
    return train, array_sha256(train)


def _build(
    tmp_path: Path, *, nonfinite_action_row: int | None = None
) -> tuple[Path, dict]:
    store, _, _ = _make_store(
        tmp_path / "latent_store",
        nonfinite_action_row=nonfinite_action_row,
    )
    split_path = tmp_path / "split_indices.npz"
    _, split_sha = _make_split(split_path)
    output = tmp_path / "state_bank"
    manifest = build_frozen_training_state_bank(
        output,
        latent_store=store,
        split_indices_path=split_path,
        expected_training_split_sha256=split_sha,
        clip_indices=np.asarray([[0, 0], [0, 5], [1, 0], [1, 5]], dtype=np.int64),
        episode_offsets=np.asarray([0, 40], dtype=np.int64),
        num_steps=6,
        history_frames=3,
        frame_skip=5,
        copy_chunk_rows=2,
    )
    return output, manifest


def test_bank_is_exact_sorted_union_and_seals_source_artifacts(tmp_path: Path):
    output, manifest = _build(tmp_path)

    # Train clips 0 and 1 overlap at row 20; validation-only clip 3 is absent.
    assert np.load(output / "global_rows.npy").tolist() == [15, 20, 25, 55, 60]
    assert np.load(output / "episode_ids.npy").tolist() == [0, 0, 0, 1, 1]
    assert manifest["index_semantics"]["current_steps"] == [3, 4]
    assert manifest["index_semantics"]["last_current_step_inclusive"] == 4
    assert manifest["pretrained_world_model_sha256"] == CHECKPOINT_SHA256
    assert manifest["dataset_source_sha256"] == DATASET_SHA256
    assert manifest["column_normalization_sha256"] == NORMALIZATION_SHA256
    assert (
        manifest["training_split"]["train_indices_sha256"]
        == (manifest["training_split_sha256"])
    )
    assert (
        manifest["source_latent_store"]["manifest_sha256"]
        == (manifest["latent_store_manifest_sha256"])
    )
    assert set(manifest["files"]) == {
        "global_rows",
        "episode_ids",
        "latents",
        "actions",
    }

    bank = load_frozen_training_state_bank(
        output,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_training_split_sha256=manifest["training_split_sha256"],
        expected_dataset_source_sha256=DATASET_SHA256,
        expected_column_normalization_sha256=NORMALIZATION_SHA256,
    )
    assert bank.size == 5
    source_latents = np.arange(80 * 2, dtype=np.float32).reshape(80, 2)
    assert np.array_equal(bank.latents, source_latents[bank.global_rows])
    assert np.isfinite(bank.actions).all()


def test_derivation_rejects_lance_store_episode_misalignment():
    source_episodes = np.repeat(np.arange(2, dtype=np.int64), 40)
    source_episodes[39] = 1

    with pytest.raises(ValueError, match="episode offsets differ"):
        derive_training_anchor_rows(
            train_indices=np.asarray([0], dtype=np.int64),
            clip_indices=np.asarray([[0, 0]], dtype=np.int64),
            episode_offsets=np.asarray([0, 40], dtype=np.int64),
            total_rows=80,
            num_steps=6,
            history_frames=3,
            frame_skip=5,
            source_episode_ids=source_episodes,
        )


def test_build_rejects_split_sha_mismatch_before_publishing(tmp_path: Path):
    store, _, _ = _make_store(tmp_path / "latent_store")
    split_path = tmp_path / "split_indices.npz"
    _make_split(split_path)
    output = tmp_path / "state_bank"

    with pytest.raises(ValueError, match="training split SHA-256 differs"):
        build_frozen_training_state_bank(
            output,
            latent_store=store,
            split_indices_path=split_path,
            expected_training_split_sha256="f" * 64,
            clip_indices=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
            episode_offsets=np.asarray([0, 40], dtype=np.int64),
            num_steps=6,
            history_frames=3,
            frame_skip=5,
        )
    assert not output.exists()


def test_build_rejects_nonfinite_training_action_anchor(tmp_path: Path):
    output = tmp_path / "state_bank"

    with pytest.raises(ValueError, match="action anchors must be finite.*20"):
        _build(tmp_path, nonfinite_action_row=20)
    assert not output.exists()
    assert not list(tmp_path.glob(".state_bank.staging-*"))


def test_reader_rejects_cli_binding_mismatch_and_tampered_array(tmp_path: Path):
    output, manifest = _build(tmp_path)

    with pytest.raises(ValueError, match="active protocol"):
        load_frozen_training_state_bank(
            output,
            expected_checkpoint_sha256="e" * 64,
            expected_training_split_sha256=manifest["training_split_sha256"],
        )

    latent_path = output / "latents.npy"
    with latent_path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="exact SHA-256"):
        load_frozen_training_state_bank(output)
