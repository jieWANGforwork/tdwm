from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tdwm.training.frozen_actor_free_td_v0_data import (
    FrozenActorFreeTDV0TransitionDataset,
    sample_reachable_future_latents_v0,
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
TOTAL_ROWS = 70
EMBED_DIM = 192


class _SourceDataset:
    lengths = np.asarray([35, 35], dtype=np.int64)
    offsets = np.asarray([0, 35], dtype=np.int64)
    frameskip = 5
    num_steps = 6
    span = 30
    clip_indices = [(0, 0), (0, 5), (1, 0), (1, 5)]

    def __len__(self) -> int:
        return len(self.clip_indices)

    @staticmethod
    def get_dim(name: str) -> int:
        if name != "action":
            raise KeyError(name)
        return 5


class _DuplicateSourceDataset(_SourceDataset):
    clip_indices = [*_SourceDataset.clip_indices, (0, 5)]


class _ManyDuplicateSourceDataset(_SourceDataset):
    clip_indices = [(0, 0)] * 20 + [(0, 5)] * 20


def _build_store(root: Path) -> tuple[FrozenLatentStore, np.ndarray]:
    latents = np.repeat(
        np.arange(TOTAL_ROWS, dtype=np.float32)[:, None],
        EMBED_DIM,
        axis=1,
    )
    raw_actions = np.arange(TOTAL_ROWS * 5, dtype=np.float32).reshape(TOTAL_ROWS, 5)
    # Episode 0 carries an explicit terminal NaN.  Episode 1 deliberately has
    # a fully finite final action so the next-next episode boundary remains an
    # independently tested terminal signal.
    raw_actions[34] = np.nan
    normalized = normalize_actions(
        raw_actions,
        mean=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
    )
    episode_ids = np.repeat(np.arange(2, dtype=np.int64), 35)
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


def _transition_dataset(
    root: Path,
) -> tuple[
    FrozenActorFreeTDV0TransitionDataset,
    FrozenLatentStore,
    np.ndarray,
]:
    store, latents = _build_store(root)
    clips = FrozenLatentClipDataset(_SourceDataset(), store)
    transitions = FrozenActorFreeTDV0TransitionDataset(clips, [0, 1])
    return transitions, store, latents


def test_selected_clips_expand_to_single_transition_records(tmp_path: Path) -> None:
    transitions, store, latents = _transition_dataset(tmp_path / "cache")

    # Four expanded records contain row 20 twice.  The replay population keeps
    # one copy, so replacement sampling is uniform over three unique rows.
    assert len(transitions) == 3
    assert transitions.global_rows.tolist() == [15, 20, 25]
    assert transitions.expanded_record_count == 4
    assert transitions.unique_record_count == 3
    assert transitions.duplicate_record_count == 1
    assert transitions.duplicate_global_row_count == 1
    assert transitions.max_multiplicity == 2
    assert transitions.record_multiplicities.tolist() == [1, 2, 1]
    assert transitions.population_diagnostics == {
        "expanded_record_count": 4,
        "unique_record_count": 3,
        "duplicate_record_count": 1,
        "duplicate_global_row_count": 1,
        "max_multiplicity": 2,
    }
    item = transitions[0]
    assert item["state"].shape == (192,)
    assert item["action"].shape == (25,)
    assert item["next_state"].shape == (192,)
    assert item["next_action"].shape == (25,)
    assert item["terminal"].shape == ()
    assert item["terminal"].dtype == torch.bool
    assert item["global_row"].item() == 15
    assert item["episode_id"].item() == 0
    assert item["source_clip_index"].item() == 0
    assert item["clip_position"].item() == 3
    torch.testing.assert_allclose(item["state"], torch.from_numpy(latents[15]))
    torch.testing.assert_allclose(item["next_state"], torch.from_numpy(latents[20]))
    torch.testing.assert_allclose(
        item["action"],
        torch.from_numpy(np.array(store.actions[15], copy=True)),
    )
    # The duplicate row 20 chooses clip 1 because it offers the farther goal
    # bound (30 rather than 25), not whichever clip happened to appear first.
    overlap = transitions[1]
    assert overlap["global_row"].item() == 20
    assert overlap["goal_future_end_row"].item() == 30
    assert overlap["source_clip_index"].item() == 1
    assert overlap["clip_position"].item() == 3
    torch.testing.assert_allclose(
        overlap["action"],
        torch.from_numpy(np.array(store.actions[20], copy=True)),
    )


def test_equal_future_bound_tie_chooses_smallest_source_clip_index(
    tmp_path: Path,
) -> None:
    store, _ = _build_store(tmp_path / "cache")
    clips = FrozenLatentClipDataset(_DuplicateSourceDataset(), store)
    transitions = FrozenActorFreeTDV0TransitionDataset(clips, [4, 1])

    assert transitions.global_rows.tolist() == [20, 25]
    assert transitions.record_multiplicities.tolist() == [2, 2]
    assert transitions.expanded_record_count == 4
    assert transitions.duplicate_record_count == 2
    assert transitions.max_multiplicity == 2
    assert [transitions[index]["source_clip_index"].item() for index in range(2)] == [
        1,
        1,
    ]


def test_many_overlapping_records_vectorize_to_canonical_unique_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _build_store(tmp_path / "cache")
    clips = FrozenLatentClipDataset(_ManyDuplicateSourceDataset(), store)

    def fail_scalar_clip_lookup(*args, **kwargs):
        del args, kwargs
        raise AssertionError("V0 index construction must normalize clips in bulk")

    monkeypatch.setattr(clips, "_clip_start", fail_scalar_clip_lookup)
    transitions = FrozenActorFreeTDV0TransitionDataset(
        clips,
        np.arange(len(clips) - 1, -1, -1),
        index_chunk_clips=7,
    )

    assert transitions.global_rows.tolist() == [15, 20, 25]
    assert transitions.record_multiplicities.tolist() == [20, 40, 20]
    assert transitions.expanded_record_count == 80
    assert transitions.unique_record_count == 3
    assert transitions.duplicate_record_count == 77
    assert transitions.duplicate_global_row_count == 3
    assert transitions.max_multiplicity == 40
    assert transitions.source_clip_indices.tolist() == [0, 20, 20]
    assert transitions.future_end_rows.tolist() == [25, 30, 30]


def test_goal_bounds_stay_in_source_clip_and_stop_at_terminal(tmp_path: Path) -> None:
    transitions, store, _ = _transition_dataset(tmp_path / "cache")

    records = transitions.__getitems__([0, 1, 2])
    assert [record["goal_future_start_row"].item() for record in records] == [
        20,
        25,
        30,
    ]
    assert [record["goal_future_end_row"].item() for record in records] == [
        25,
        30,
        30,
    ]
    assert [record["goal_future_count"].item() for record in records] == [
        2,
        2,
        1,
    ]
    assert [record["terminal"].item() for record in records] == [
        False,
        False,
        True,
    ]

    terminal = records[-1]
    raw_next_action = np.array(store.actions[30], copy=True)
    assert np.isfinite(raw_next_action[:20]).all()
    assert np.isnan(raw_next_action[20:]).all()
    torch.testing.assert_allclose(
        terminal["next_action"][:20], torch.from_numpy(raw_next_action[:20])
    )
    assert torch.equal(terminal["next_action"][20:], torch.zeros(5))
    assert torch.isfinite(terminal["next_action"]).all()


def test_random_sampler_batches_transitions_not_flattened_clips(tmp_path: Path) -> None:
    transitions, _, _ = _transition_dataset(tmp_path / "cache")
    generator = torch.Generator().manual_seed(17)
    sampler = torch.utils.data.RandomSampler(
        transitions,
        replacement=True,
        num_samples=12,
        generator=generator,
    )
    loader = torch.utils.data.DataLoader(
        transitions,
        batch_size=6,
        sampler=sampler,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert batch["state"].shape == (6, 192)
    assert batch["action"].shape == (6, 25)
    assert batch["next_state"].shape == (6, 192)
    assert batch["next_action"].shape == (6, 25)
    assert batch["terminal"].shape == (6,)
    assert set(batch["global_row"].tolist()) <= {15, 20, 25}
    assert len(set(batch["global_row"].tolist())) < 6

    population_sampler = torch.utils.data.RandomSampler(
        transitions,
        replacement=True,
        num_samples=3000,
        generator=torch.Generator().manual_seed(123),
    )
    sampled_rows = transitions.global_rows[np.fromiter(population_sampler, np.int64)]
    counts = {row: int((sampled_rows == row).sum()) for row in (15, 20, 25)}
    assert all(850 <= count <= 1150 for count in counts.values())


def test_reachable_future_sampling_is_matched_and_reproducible(tmp_path: Path) -> None:
    transitions, store, latents = _transition_dataset(tmp_path / "cache")
    items = transitions.__getitems__([0, 1, 2])
    rows = torch.stack([item["global_row"] for item in items])
    ends = torch.stack([item["goal_future_end_row"] for item in items])

    first = sample_reachable_future_latents_v0(
        store,
        rows,
        ends,
        generator=torch.Generator().manual_seed(91),
    )
    second = sample_reachable_future_latents_v0(
        store,
        rows,
        ends,
        generator=torch.Generator().manual_seed(91),
    )

    assert torch.equal(first.global_rows, second.global_rows)
    assert torch.equal(first.offset_steps, second.offset_steps)
    torch.testing.assert_allclose(first.latents, second.latents)
    assert torch.equal(first.global_rows, rows + 5 * first.offset_steps)
    assert bool((first.global_rows <= ends).all())
    assert bool((first.offset_steps >= 1).all())
    expected = torch.from_numpy(latents[first.global_rows.numpy()])
    torch.testing.assert_allclose(first.latents, expected)
    assert np.array_equal(
        store.episode_ids[rows.numpy()],
        store.episode_ids[first.global_rows.numpy()],
    )


def test_validation_future_selection_uses_inclusive_end_without_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions, store, latents = _transition_dataset(tmp_path / "cache")
    items = transitions.__getitems__([0, 1, 2])
    rows = torch.stack([item["global_row"] for item in items])
    ends = torch.stack([item["goal_future_end_row"] for item in items])

    def fail_random_sample(*args, **kwargs):
        del args, kwargs
        raise AssertionError("deterministic validation must not sample RNG")

    monkeypatch.setattr(torch, "rand", fail_random_sample)

    selected = sample_reachable_future_latents_v0(
        store,
        rows,
        ends,
        generator=None,
    )

    assert torch.equal(selected.global_rows, ends)
    assert torch.equal(selected.offset_steps, (ends - rows) // 5)
    torch.testing.assert_allclose(
        selected.latents,
        torch.from_numpy(latents[ends.numpy()]),
    )


def test_transition_reads_never_gather_a_whole_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions, store, _ = _transition_dataset(tmp_path / "cache")

    def fail_gather(*args, **kwargs):
        del args, kwargs
        raise AssertionError("transition access must not gather a complete clip")

    monkeypatch.setattr(store, "gather_clips", fail_gather)
    batch = transitions.__getitems__([0, 2])
    assert [item["global_row"].item() for item in batch] == [15, 25]


def test_finite_next_action_is_terminal_when_next_next_leaves_episode(
    tmp_path: Path,
) -> None:
    store, _ = _build_store(tmp_path / "cache")
    clips = FrozenLatentClipDataset(_SourceDataset(), store)
    transitions = FrozenActorFreeTDV0TransitionDataset(clips, [3])

    assert transitions.global_rows.tolist() == [55, 60]
    boundary = transitions[-1]
    assert boundary["global_row"].item() == 60
    assert boundary["terminal"].item() is True
    assert torch.isfinite(boundary["next_action"]).all()
    torch.testing.assert_allclose(
        boundary["next_action"],
        torch.from_numpy(np.array(store.actions[65], copy=True)),
    )
    assert boundary["goal_future_start_row"].item() == 65
    assert boundary["goal_future_end_row"].item() == 65


def test_transition_dataset_rejects_non_v0_index_contract(tmp_path: Path) -> None:
    store, _ = _build_store(tmp_path / "cache")
    clips = FrozenLatentClipDataset(_SourceDataset(), store)

    with pytest.raises(ValueError, match="first_current_index"):
        FrozenActorFreeTDV0TransitionDataset(
            clips,
            [0],
            first_current_index=2,
        )
    with pytest.raises(IndexError, match="clip_indices"):
        FrozenActorFreeTDV0TransitionDataset(clips, [len(clips)])
    with pytest.raises(ValueError, match="cannot be empty"):
        FrozenActorFreeTDV0TransitionDataset(clips, [])


def test_future_sampler_rejects_cross_episode_or_misaligned_bounds(
    tmp_path: Path,
) -> None:
    _, store, _ = _transition_dataset(tmp_path / "cache")
    generator = torch.Generator().manual_seed(3)

    with pytest.raises(ValueError, match="frame aligned"):
        sample_reachable_future_latents_v0(
            store,
            [20],
            [26],
            generator=generator,
        )
    with pytest.raises(ValueError, match="episode boundary"):
        sample_reachable_future_latents_v0(
            store,
            [30],
            [35],
            generator=generator,
        )
