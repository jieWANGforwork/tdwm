from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.training.eff_data import (
    EffEpisodeReplay,
    EpisodePartition,
    held_out_episode_pairs,
)


def _store():
    ids = np.repeat(np.arange(5, dtype=np.int64), 21)
    latents = np.zeros((len(ids), 192), dtype=np.float32)
    latents[:, 0] = np.tile(np.arange(21), 5)
    latents[:, 1] = ids
    return SimpleNamespace(episode_ids=ids, latents=latents)


def _replay(terminal=None):
    store = _store()
    if terminal is None:
        terminal = np.zeros(len(store.episode_ids), dtype=bool)
    return EffEpisodeReplay(
        store, episodes=(0, 1, 2), stride=5, terminal_at_state=terminal
    )


def test_rp1_episode_partition_is_not_the_old_clip_split():
    split = EpisodePartition.rp1_cube()
    assert split.training == tuple(range(8000))
    assert split.evaluation == tuple(range(8000, 10000))
    assert not set(split.training).intersection(split.evaluation)


def test_episode_leakage_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        EpisodePartition((0, 1), (1, 2))


def test_same_episode_goals_and_backups_are_time_aligned():
    batch = _replay().sample(
        batch_size=100,
        rng=np.random.default_rng(42),
        backup_primitive_steps=10,
        cross_episode_probability=0,
        epsilon=1e-6,
    )
    assert torch.equal(batch.anchor_episodes, batch.goal_episodes)
    assert (batch.goal_rows > batch.anchor_rows).all()
    assert (batch.goal_rows - batch.anchor_rows).remainder(5).eq(0).all()
    assert (batch.bootstrap_rows - batch.anchor_rows <= 10).all()
    assert torch.equal(batch.next_state[:, 0], batch.state[:, 0] + 5)
    torch.testing.assert_close(
        batch.observed_cost, (batch.bootstrap_rows - batch.anchor_rows).float()
    )
    assert batch.goal_reached.any() and (~batch.goal_reached).any()
    assert batch.efficiency_known.all()
    assert batch.anchor_episodes.max() < 3


def test_cross_episode_goals_have_no_fake_path_efficiency_or_direct_label():
    batch = _replay().sample(
        batch_size=200,
        rng=np.random.default_rng(43),
        backup_primitive_steps=10,
        cross_episode_probability=1,
        epsilon=1e-6,
    )
    assert (batch.goal_episodes != batch.anchor_episodes).all()
    assert not batch.efficiency_known.any()
    assert not batch.goal_reached.any()
    assert batch.continuation_valid.all()  # Explicitly declared truncation boundaries.
    assert batch.goal_episodes.max() < 3


def test_window_truncation_and_true_terminal_are_not_confused():
    terminal = np.zeros(105, dtype=bool)
    terminal[[20, 41, 62]] = True
    batch = _replay(terminal).sample(
        batch_size=200,
        rng=np.random.default_rng(43),
        backup_primitive_steps=20,
        cross_episode_probability=1,
        epsilon=1e-6,
    )
    assert not batch.continuation_valid.any()
    assert not batch.goal_reached.any()


def test_sampler_rng_state_is_restorable_exactly():
    replay = _replay()
    a, b = np.random.default_rng(42), np.random.default_rng(1)
    b.bit_generator.state = a.bit_generator.state
    kwargs = dict(
        batch_size=30,
        backup_primitive_steps=10,
        cross_episode_probability=0.3,
        epsilon=1e-6,
    )
    x, y = replay.sample(rng=a, **kwargs), replay.sample(rng=b, **kwargs)
    assert torch.equal(x.anchor_rows, y.anchor_rows)
    assert torch.equal(x.goal_rows, y.goal_rows)


def test_offsets_balanced_before_anchor_sampling():
    batch = _replay().sample(
        batch_size=8000,
        rng=np.random.default_rng(3),
        backup_primitive_steps=10,
        cross_episode_probability=0,
        epsilon=1e-6,
    )
    histogram = torch.bincount(batch.offset_chunks)[1:]
    assert histogram.shape == (4,)
    assert (histogram > 1800).all() and (histogram < 2200).all()


def test_evaluation_pairs_are_shared_reproducible_and_held_out():
    kwargs = dict(
        lengths=np.full(10000, 201),
        episode_ids=tuple(range(8000, 10000)),
        goal_offset=50,
        count=50,
        seed=42,
    )
    x, y = held_out_episode_pairs(**kwargs), held_out_episode_pairs(**kwargs)
    assert x == y
    assert min(x["episode_indices"]) >= 8000
    assert len(set(zip(x["episode_indices"], x["start_steps"]))) == 50
    assert np.array_equal(np.asarray(x["start_steps"]) + 50, x["goal_steps"])


def test_terminal_metadata_cannot_be_silently_assumed():
    with pytest.raises(ValueError, match="terminal"):
        EffEpisodeReplay(
            _store(), episodes=(0,), stride=5, terminal_at_state=np.zeros(105)
        )
