from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.training.eff_data import EffEpisodeReplay, sample_planner_paths
from tdwm.training.effplan_discrete_data import DiscretePlannerSampler, discrete_sampling_identity


def replay(episodes=(0, 1), terminal=None, frames=201):
    ids = np.repeat(np.arange(3), frames)
    states = np.zeros((len(ids), 192), np.float32)
    states[:, 0] = np.tile(np.arange(frames), 3)
    states[:, 1] = ids
    # Nonlinear feature distinguishes selecting a state from averaging states.
    states[:, 2] = states[:, 0] ** 2
    return EffEpisodeReplay(
        SimpleNamespace(episode_ids=ids, latents=states), episodes=episodes,
        stride=5, terminal_at_state=np.zeros(len(ids), bool) if terminal is None else terminal,
    )


def test_import_and_exact_uniform_positions_not_latent_means():
    source = replay()
    b = DiscretePlannerSampler(source).sample(batch_size=600, rng=np.random.default_rng(3072))
    assert b.real_states.shape == (600, 6, 192)
    assert b.state_stride == 5 and b.trajectory_valid.all()
    assert set(b.action_spans.tolist()) == {5, 10, 20}
    for n in (5, 10, 20):
        mask = b.action_spans == n
        expected = torch.tensor([j*n//5 for j in range(6)])
        assert torch.equal(b.node_offsets_blocks[mask], expected.expand(int(mask.sum()), 6))
        assert torch.equal(b.rows[mask]-b.rows[mask, :1], expected.expand(int(mask.sum()), 6)*5)
    np.testing.assert_array_equal(b.real_states.numpy(), source.store.latents[b.rows.numpy()])
    assert torch.equal(b.anchor_episodes, b.goal_episodes)
    assert set(b.anchor_episodes.tolist()) <= {0, 1}
    assert torch.equal(b.real_states[:, :, 1], b.anchor_episodes[:, None].float().expand(-1, 6))


def test_span_distribution_is_uniform_not_weighted_by_number_of_windows():
    b = DiscretePlannerSampler(replay()).sample(batch_size=6000, rng=np.random.default_rng(42))
    for n in (5, 10, 20):
        assert 1800 < int((b.action_spans == n).sum()) < 2200


def test_rng_resume_reproduces_all_rows_and_spans():
    sampler = DiscretePlannerSampler(replay())
    rng = np.random.default_rng(3)
    sampler.sample(batch_size=20, rng=rng)
    checkpoint = copy.deepcopy(rng.bit_generator.state)
    a = sampler.sample(batch_size=30, rng=rng)
    restored = np.random.default_rng(999)
    restored.bit_generator.state = checkpoint
    b = DiscretePlannerSampler(replay()).sample(batch_size=30, rng=restored)
    assert torch.equal(a.rows, b.rows) and torch.equal(a.action_spans, b.action_spans)


def test_terminal_in_unselected_frame_is_not_skipped():
    terminal = np.zeros(603, bool)
    terminal[43] = True  # Not a stride-5 frame, nor necessarily one of six labels.
    source = replay(terminal=terminal)
    b = DiscretePlannerSampler(source).sample(batch_size=600, rng=np.random.default_rng(42))
    for rows in b.rows.tolist():
        assert not source._crosses_terminal(rows[0], rows[-1])


def test_goal_terminal_is_allowed_and_short_episodes_do_not_silently_drop_long_span():
    terminal = np.zeros(303, bool)
    terminal[100] = True
    sampler = DiscretePlannerSampler(replay(episodes=(0,), terminal=terminal, frames=101))
    b = sampler.sample(batch_size=30, rng=np.random.default_rng(2))
    long = b.action_spans == 20
    assert long.any() and b.rows[long, -1].eq(100).all()
    with pytest.raises(ValueError, match="20 action blocks"):
        DiscretePlannerSampler(replay(frames=51))


@pytest.mark.parametrize("spans", [[5], [5, 10], [5, 10, 21], [5, 10, 20, 40], [5.0, 10, 20], None])
def test_only_declared_discrete_lengths_allowed(spans):
    with pytest.raises(ValueError, match="exactly"):
        discrete_sampling_identity(spans)


def test_validation_stays_in_its_own_episode_partition():
    b = DiscretePlannerSampler(replay(episodes=(2,))).sample(batch_size=30, rng=np.random.default_rng(50))
    assert b.anchor_episodes.eq(2).all() and b.goal_episodes.eq(2).all()


def test_original_sampler_remains_five_consecutive_blocks():
    b = sample_planner_paths(replay(), batch_size=30, horizon=5,
                            rng=np.random.default_rng(3), cross_episode_probability=0)
    assert (b.rows[:, 1:]-b.rows[:, :-1]).eq(5).all()
    assert not hasattr(b, "action_spans")
