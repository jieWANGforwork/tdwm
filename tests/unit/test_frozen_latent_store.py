from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from tdwm.training.frozen_latent_store import (
    EncodedRowBatch,
    FrozenLatentClipDataset,
    FrozenLatentStore,
    FrozenLatentStoreBuilder,
    FrozenLatentStoreSpec,
    action_blocks_for_rows,
    build_frozen_latent_store,
    iter_encoded_global_rows_once,
    normalize_actions,
)

CHECKPOINT_SHA256 = "1" * 64
DATASET_SHA256 = "2" * 64
NORMALIZATION_SHA256 = "3" * 64
GIT_REVISION = "4" * 40
TOTAL_ROWS = 62
EMBED_DIM = 4


def _spec() -> FrozenLatentStoreSpec:
    return FrozenLatentStoreSpec(
        total_rows=TOTAL_ROWS,
        embed_dim=EMBED_DIM,
        frame_skip=5,
        history_frames=3,
        action_dim=5,
        pretrained_checkpoint_sha256=CHECKPOINT_SHA256,
        dataset_source_sha256=DATASET_SHA256,
        column_normalization_sha256=NORMALIZATION_SHA256,
        git_revision=GIT_REVISION,
    )


def _fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latents = np.arange(TOTAL_ROWS * EMBED_DIM, dtype=np.float32).reshape(
        TOTAL_ROWS, EMBED_DIM
    )
    raw_actions = np.arange(TOTAL_ROWS * 5, dtype=np.float32).reshape(TOTAL_ROWS, 5)
    # The final dataset row of each episode carries the source terminal NaN.
    raw_actions[[30, 61]] = np.nan
    normalized = normalize_actions(
        raw_actions,
        mean=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
    )
    episodes = np.repeat(np.arange(2, dtype=np.int64), 31)
    return latents, normalized, episodes


def _build_fixture(root: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    latents, actions, episodes = _fixture_arrays()
    # Deliberately write the two global ranges out of order. Row identity, not
    # producer order or clip identity, determines the published position.
    batches = [
        EncodedRowBatch(np.arange(31, 62), latents[31:]),
        EncodedRowBatch(np.arange(0, 31), latents[:31]),
    ]
    build_frozen_latent_store(
        root,
        spec=_spec(),
        encoded_batches=batches,
        normalized_actions=actions,
        episode_ids=episodes,
        source_metadata={"fixture": True},
    )
    return root, latents, actions, episodes


def _open(root: Path) -> FrozenLatentStore:
    return FrozenLatentStore(
        root,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_dataset_source_sha256=DATASET_SHA256,
        expected_column_normalization_sha256=NORMALIZATION_SHA256,
        expected_frame_skip=5,
        expected_history_frames=3,
        expected_embed_dim=EMBED_DIM,
        expected_action_dim=5,
    )


def test_shared_global_rows_are_encoded_once_and_reused_by_clips(tmp_path: Path):
    root, latents, actions, episodes = _build_fixture(tmp_path / "cache")
    store = _open(root)

    cached = store.gather_clips([0, 5], num_steps=4, frame_skip=5)

    assert isinstance(store.latents, np.memmap)
    assert isinstance(store.actions, np.memmap)
    assert cached.latents.shape == (2, 4, EMBED_DIM)
    assert cached.actions.shape == (2, 4, 25)
    assert cached.metadata["global_rows"].tolist() == [
        [0, 5, 10, 15],
        [5, 10, 15, 20],
    ]
    # The overlapping frames come from one globally indexed latent row.
    torch.testing.assert_close(cached.latents[0, 1], cached.latents[1, 0])
    torch.testing.assert_close(cached.latents[0, 2], cached.latents[1, 1])

    online_rows = cached.metadata["global_rows"].numpy()
    online_latents = torch.from_numpy(np.array(latents[online_rows], copy=True))
    online_actions = torch.from_numpy(
        action_blocks_for_rows(
            online_rows.reshape(-1),
            normalized_actions=actions,
            episode_ids=episodes,
        ).reshape(2, 4, 25)
    )
    # Cached E output/action input has the same interface shape and values as
    # one online encoding of those exact rows.
    assert cached.latents.shape == online_latents.shape
    assert cached.actions.shape == online_actions.shape
    torch.testing.assert_close(cached.latents, online_latents)
    torch.testing.assert_close(cached.actions, online_actions, equal_nan=True)


def test_terminal_nan_is_preserved_and_episode_tail_is_nan_padded(tmp_path: Path):
    root, _, _, _ = _build_fixture(tmp_path / "cache")
    store = _open(root)

    # start=11 consumes dense action rows 11..30. The last sampled block starts
    # at 26 and therefore includes terminal row 30 as its fifth action slot.
    cached = store.gather_clips([11], num_steps=4, frame_skip=5)
    last_block = cached.actions[0, -1].reshape(5, 5)
    assert torch.isfinite(last_block[:4]).all()
    assert torch.isnan(last_block[4]).all()
    assert not cached.metadata["action_finite"][0, -1]

    # A row too close to the boundary is still stored, but its block never
    # borrows the next episode's actions.
    tail = np.asarray(store.actions[29]).reshape(5, 5)
    assert np.isfinite(tail[0]).all()
    assert np.isnan(tail[1:]).all()


def test_reader_rejects_checkpoint_mismatch_and_cross_episode_clip(tmp_path: Path):
    root, _, _, _ = _build_fixture(tmp_path / "cache")
    with pytest.raises(ValueError, match="pretrained_checkpoint_sha256"):
        FrozenLatentStore(
            root,
            expected_checkpoint_sha256="9" * 64,
        )

    store = _open(root)
    with pytest.raises(ValueError, match="cross an episode"):
        store.gather_clips([12], num_steps=4, frame_skip=5)
    with pytest.raises(IndexError, match="global row bounds"):
        store.gather_clips([-1], num_steps=4, frame_skip=5)


def test_reader_checks_exact_array_sha256(tmp_path: Path):
    root, _, _, _ = _build_fixture(tmp_path / "cache")
    latent_path = root / "latents.npy"
    with latent_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))

    with pytest.raises(ValueError, match="exact SHA-256"):
        _open(root)


def test_builder_never_publishes_incomplete_or_duplicate_row_coverage(
    tmp_path: Path,
):
    output = tmp_path / "incomplete"
    spec = _spec()
    with pytest.raises(RuntimeError, match="missing rows"):
        with FrozenLatentStoreBuilder(output, spec=spec) as builder:
            builder.add_rows(
                global_rows=[0],
                latents=np.zeros((1, EMBED_DIM), dtype=np.float32),
                action_blocks=np.zeros((1, 25), dtype=np.float32),
                episode_ids=[0],
            )
            builder.finalize()
    assert not output.exists()

    calls: list[list[int]] = []

    def encode_rows(rows: np.ndarray) -> np.ndarray:
        calls.append(rows.tolist())
        return np.zeros((rows.size, EMBED_DIM), dtype=np.float32)

    batches = list(
        iter_encoded_global_rows_once(
            total_rows=7,
            batch_size=3,
            encode_rows=encode_rows,
        )
    )
    assert calls == [[0, 1, 2], [3, 4, 5], [6]]
    assert np.array_equal(
        np.concatenate([batch.global_rows for batch in batches]), np.arange(7)
    )


def test_global_row_image_preprocessing_matches_formal_online_path():
    from scripts.build_frozen_latent_store import _preprocess_frames
    from tdwm.training.gt_lewm_support import preprocess_image_batch

    frames = torch.arange(2 * 3 * 8 * 8, dtype=torch.int64).remainder(256)
    frames = frames.to(torch.uint8).reshape(2, 3, 8, 8)
    mean_values = (0.485, 0.456, 0.406)
    std_values = (0.229, 0.224, 0.225)

    cached_path = _preprocess_frames(
        frames,
        device=torch.device("cpu"),
        image_size=6,
        mean=mean_values,
        std=std_values,
    )
    formal_path = preprocess_image_batch(
        frames.unsqueeze(1),
        mean=torch.tensor(mean_values).reshape(1, 1, 3, 1, 1),
        std=torch.tensor(std_values).reshape(1, 1, 3, 1, 1),
        size=6,
    )[:, 0]

    assert torch.equal(cached_path, formal_path)


def test_clip_dataset_preserves_source_indices_without_returning_pixels(
    tmp_path: Path,
):
    root, latents, _, _ = _build_fixture(tmp_path / "cache")
    store = _open(root)

    class SourceDataset:
        lengths = np.asarray([31, 31], dtype=np.int64)
        offsets = np.asarray([0, 31], dtype=np.int64)
        frameskip = 5
        num_steps = 4
        span = 20
        clip_indices = [(0, 0), (0, 5), (1, 0), (1, 5)]

        def __len__(self):
            return len(self.clip_indices)

        @staticmethod
        def get_dim(name: str) -> int:
            if name != "action":
                raise KeyError(name)
            return 5

    wrapped = FrozenLatentClipDataset(SourceDataset(), store)
    samples = wrapped.__getitems__([1, 2])

    assert len(wrapped) == 4
    assert len(samples) == 2
    assert all("pixels" not in sample for sample in samples)
    assert [sample["_tdwm_global_start"].item() for sample in samples] == [5, 31]
    assert [sample["_tdwm_episode_id"].item() for sample in samples] == [0, 1]
    torch.testing.assert_close(
        samples[0]["_tdwm_frozen_latents"],
        torch.from_numpy(np.array(latents[[5, 10, 15, 20]], copy=True)),
    )
    assert samples[0]["action"].shape == (4, 25)
    assert wrapped[-1]["_tdwm_global_start"].item() == 36

    # The ordinary map-style DataLoader/Subset path must use the wrapper's
    # batch gather. SourceDataset intentionally has no __getitem__, so this
    # also proves that no online/JPEG sample is touched.
    subset = torch.utils.data.Subset(wrapped, [1, 2])
    batch = next(iter(torch.utils.data.DataLoader(subset, batch_size=2, num_workers=0)))
    assert "pixels" not in batch
    assert batch["_tdwm_frozen_latents"].shape == (2, 4, EMBED_DIM)
    assert batch["action"].shape == (2, 4, 25)
    assert batch["_tdwm_global_start"].tolist() == [5, 31]


def test_store_spawn_pickle_reopens_mmaps_without_embedding_array_payload(tmp_path):
    root, latents, actions, episodes = _build_fixture(tmp_path / "cache")
    store = _open(root)

    serialized = pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(serialized)

    assert len(serialized) < 20_000
    assert len(serialized) < latents.nbytes + actions.nbytes + episodes.nbytes
    assert isinstance(restored.latents, np.memmap)
    assert isinstance(restored.actions, np.memmap)
    assert isinstance(restored.episode_ids, np.memmap)
    assert restored.latents.filename is not None
    gathered = restored.gather_clips([0], num_steps=4, frame_skip=5)
    torch.testing.assert_close(
        gathered.latents[0],
        torch.from_numpy(np.array(latents[[0, 5, 10, 15]], copy=True)),
    )


def test_multi_batch_builder_scans_complete_action_array_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    latents, actions, episodes = _fixture_arrays()
    batches = [
        EncodedRowBatch(rows, latents[rows])
        for rows in np.array_split(np.arange(TOTAL_ROWS), 7)
    ]
    original_isinf = np.isinf
    full_action_scans = 0

    def counted_isinf(values):
        nonlocal full_action_scans
        if np.asarray(values).shape == actions.shape:
            full_action_scans += 1
        return original_isinf(values)

    monkeypatch.setattr(np, "isinf", counted_isinf)
    build_frozen_latent_store(
        tmp_path / "cache",
        spec=_spec(),
        encoded_batches=batches,
        normalized_actions=actions,
        episode_ids=episodes,
    )

    assert full_action_scans == 1


def test_cli_action_audit_counts_only_finite_rows_as_normalization_samples():
    from scripts.build_frozen_latent_store import _validate_action_terminal_rows

    _, actions, _ = _fixture_arrays()
    lengths = np.asarray([31, 31], dtype=np.int64)
    offsets = np.asarray([0, 31], dtype=np.int64)

    audit = _validate_action_terminal_rows(
        actions,
        lengths=lengths,
        offsets=offsets,
        normalization_samples=60,
    )
    assert audit["finite_rows"] == 60
    assert audit["terminal_nan_rows"] == 2

    with pytest.raises(ValueError, match="sample count"):
        _validate_action_terminal_rows(
            actions,
            lengths=lengths,
            offsets=offsets,
            normalization_samples=62,
        )
    mixed_nan = actions.copy()
    mixed_nan[3, 0] = np.nan
    with pytest.raises(ValueError, match="entirely finite or entirely NaN"):
        _validate_action_terminal_rows(
            mixed_nan,
            lengths=lengths,
            offsets=offsets,
            normalization_samples=60,
        )
    misplaced_terminal = actions.copy()
    misplaced_terminal[30] = 0.0
    misplaced_terminal[29] = np.nan
    with pytest.raises(ValueError, match="final row"):
        _validate_action_terminal_rows(
            misplaced_terminal,
            lengths=lengths,
            offsets=offsets,
            normalization_samples=60,
        )
