from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tdwm.training.frozen_actor_free_td_v1_data import (
    FrozenActorFreeTDV1C2TransitionDataset,
    FrozenActorFreeTDV1TransitionDataset,
)
from tdwm.training.frozen_latent_store import (
    EncodedRowBatch,
    FrozenLatentClipDataset,
    FrozenLatentStore,
    FrozenLatentStoreSpec,
    build_frozen_latent_store,
    normalize_actions,
)

CHECKPOINT_SHA256 = "1" * 64
DATASET_SHA256 = "2" * 64
NORMALIZATION_SHA256 = "3" * 64
GIT_REVISION = "4" * 40
EPISODE_LENGTH = 65
TOTAL_ROWS = 2 * EPISODE_LENGTH
EMBED_DIM = 192


class _SourceDataset:
    lengths = np.asarray([EPISODE_LENGTH, EPISODE_LENGTH], dtype=np.int64)
    offsets = np.asarray([0, EPISODE_LENGTH], dtype=np.int64)
    frameskip = 5
    num_steps = 6
    span = 30
    clip_indices = [(0, 0), (0, 35), (1, 0), (0, 25)]

    def __len__(self) -> int:
        return len(self.clip_indices)

    @staticmethod
    def get_dim(name: str) -> int:
        if name != "action":
            raise KeyError(name)
        return 5


def _build_store(
    root: Path,
    *,
    nan_action_row: int | None = None,
) -> tuple[FrozenLatentStore, np.ndarray]:
    latents = np.repeat(
        np.arange(TOTAL_ROWS, dtype=np.float32)[:, None],
        EMBED_DIM,
        axis=1,
    )
    raw_actions = np.arange(TOTAL_ROWS * 5, dtype=np.float32).reshape(TOTAL_ROWS, 5)
    if nan_action_row is not None:
        raw_actions[int(nan_action_row)] = np.nan
    normalized = normalize_actions(
        raw_actions,
        mean=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
    )
    episode_ids = np.repeat(np.arange(2, dtype=np.int64), EPISODE_LENGTH)
    spec = FrozenLatentStoreSpec(
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
    build_frozen_latent_store(
        root,
        spec=spec,
        encoded_batches=[EncodedRowBatch(np.arange(TOTAL_ROWS), latents)],
        normalized_actions=normalized,
        episode_ids=episode_ids,
    )
    store = FrozenLatentStore(
        root,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_dataset_source_sha256=DATASET_SHA256,
        expected_column_normalization_sha256=NORMALIZATION_SHA256,
        expected_frame_skip=5,
        expected_history_frames=3,
        expected_embed_dim=EMBED_DIM,
        expected_action_dim=5,
    )
    return store, latents


def _datasets(
    root: Path,
    *,
    clip_indices: list[int],
    nan_action_row: int | None = None,
) -> tuple[
    FrozenActorFreeTDV1TransitionDataset,
    FrozenActorFreeTDV1C2TransitionDataset,
    FrozenLatentStore,
    np.ndarray,
]:
    store, latents = _build_store(root, nan_action_row=nan_action_row)
    clips = FrozenLatentClipDataset(_SourceDataset(), store)
    base = FrozenActorFreeTDV1TransitionDataset(clips, clip_indices)
    c2 = FrozenActorFreeTDV1C2TransitionDataset(clips, clip_indices)
    return base, c2, store, latents


def test_c2_adds_exact_three_state_two_action_and_five_step_context(
    tmp_path: Path,
) -> None:
    base, c2, store, latents = _datasets(
        tmp_path / "cache",
        clip_indices=[0],
    )

    assert c2.global_rows.tolist() == base.global_rows.tolist() == [15, 20]
    item = c2[0]
    assert item["alignment_state_history"].shape == (3, 192)
    assert item["alignment_action_history"].shape == (2, 25)
    assert item["alignment_action_sequence"].shape == (5, 25)
    assert item["alignment_rollout_valid"].shape == ()
    assert item["alignment_rollout_valid"].dtype == torch.bool
    assert item["alignment_rollout_valid"].item() is True
    torch.testing.assert_close(
        item["alignment_state_history"],
        torch.from_numpy(latents[[5, 10, 15]]),
    )
    torch.testing.assert_close(
        item["alignment_action_history"],
        torch.from_numpy(np.array(store.actions[[5, 10]], copy=True)),
    )
    torch.testing.assert_close(
        item["alignment_action_sequence"],
        torch.from_numpy(np.array(store.actions[[15, 20, 25, 30, 35]], copy=True)),
    )


def test_c2_keeps_every_original_transition_field_and_value(
    tmp_path: Path,
) -> None:
    base, c2, _, _ = _datasets(tmp_path / "cache", clip_indices=[0])

    base_item = base[1]
    c2_item = c2[1]
    assert set(c2_item) - set(base_item) == {
        "alignment_state_history",
        "alignment_action_history",
        "alignment_action_sequence",
        "alignment_rollout_valid",
    }
    for key, value in base_item.items():
        assert torch.equal(c2_item[key], value), key


def test_c2_marks_episode_crossing_rollout_invalid_and_zero_fills(
    tmp_path: Path,
) -> None:
    _, c2, _, _ = _datasets(tmp_path / "cache", clip_indices=[1])

    # The logged five-action sequences for rows 50 and 55 reach episode 1,
    # even though both are still legal one-step V1 transition records.
    assert c2.global_rows.tolist() == [50, 55]
    for item in c2.__getitems__([0, 1]):
        assert item["alignment_rollout_valid"].item() is False
        assert torch.count_nonzero(item["alignment_state_history"]).item() == 0
        assert torch.count_nonzero(item["alignment_action_history"]).item() == 0
        assert torch.count_nonzero(item["alignment_action_sequence"]).item() == 0


def test_c2_requires_the_five_block_terminal_state_to_stay_in_episode(
    tmp_path: Path,
) -> None:
    _, c2, _, _ = _datasets(tmp_path / "cache", clip_indices=[3])

    # The five finite blocks from row 40 end at row 65, which is the first row
    # of the next episode.  Checking only their action anchors through row 60
    # would incorrectly accept this teacher rollout.
    assert c2.global_rows.tolist() == [40, 45]
    item = c2[0]
    assert item["alignment_rollout_valid"].item() is False
    assert torch.equal(item["alignment_state_history"], torch.zeros(3, 192))
    assert torch.equal(item["alignment_action_history"], torch.zeros(2, 25))
    assert torch.equal(item["alignment_action_sequence"], torch.zeros(5, 25))


def test_c2_marks_nonfinite_action_context_invalid_and_zero_fills(
    tmp_path: Path,
) -> None:
    _, c2, _, _ = _datasets(
        tmp_path / "cache",
        clip_indices=[0],
        nan_action_row=37,
    )

    # Row 37 occupies a slot of the rollout block anchored at row 35.  The
    # base transition remains legal, but it cannot supervise a five-step C2
    # frozen-F rollout.
    item = c2[0]
    assert item["global_row"].item() == 15
    assert item["alignment_rollout_valid"].item() is False
    assert torch.equal(item["alignment_state_history"], torch.zeros(3, 192))
    assert torch.equal(item["alignment_action_history"], torch.zeros(2, 25))
    assert torch.equal(item["alignment_action_sequence"], torch.zeros(5, 25))


def test_c2_batch_collation_preserves_shapes_and_validity_mask(
    tmp_path: Path,
) -> None:
    _, c2, _, _ = _datasets(tmp_path / "cache", clip_indices=[0, 1])
    batch = next(iter(torch.utils.data.DataLoader(c2, batch_size=4, num_workers=0)))

    assert batch["alignment_state_history"].shape == (4, 3, 192)
    assert batch["alignment_action_history"].shape == (4, 2, 25)
    assert batch["alignment_action_sequence"].shape == (4, 5, 25)
    assert batch["alignment_rollout_valid"].shape == (4,)
    assert batch["alignment_rollout_valid"].dtype == torch.bool
    assert batch["alignment_rollout_valid"].tolist() == [True, True, False, False]
