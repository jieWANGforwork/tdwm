from dataclasses import replace

import numpy as np
import pytest
import torch

from tdwm.training.eff_action_data import EffActionEpisodeData, EffActionSamplingConfig


def config(**kwargs):
    base = EffActionSamplingConfig(
        episode_start=0, episode_stop=2, min_goal_chunks=1, max_goal_chunks=3,
        grid_phase=0, efficiency_threshold=0.9, efficiency_epsilon=1e-6,
        zero_distance_tolerance=1e-7, direct_max_chunks=None,
    )
    return replace(base, **kwargs)


def arrays():
    z = np.zeros((42, 192), dtype=np.float32)
    z[:, 0] = np.tile(np.arange(21), 2)
    z[:, 1] = 1
    a = np.repeat(np.arange(42, dtype=np.float32)[:, None], 25, axis=1)
    a[20] = a[41] = np.nan
    ids = np.repeat(np.arange(2, dtype=np.int64), 21)
    return z, a, ids


def test_direct_sum_uses_each_real_future_state_and_goal():
    z, a, ids = arrays()
    data = EffActionEpisodeData(z, a, ids, config=config())
    batch = data.batch_for_rows(np.array([0, 5]), np.array([15, 10]))
    torch.testing.assert_close(batch["direct_successor_target"][0], torch.tensor(z[[5, 10, 15]].sum(0)))
    torch.testing.assert_close(batch["direct_successor_target"][1], torch.tensor(z[10]))
    assert batch["goal_terminal"].tolist() == [False, True]
    assert torch.isnan(batch["next_raw_action"][1]).all()
    assert batch["next_raw_action"][0, 0].item() == 5
    assert batch["path_length"].tolist() == [15, 5]
    assert batch["net_distance"].tolist() == [15, 5]
    assert batch["direct_branch"].tolist() == [True, True]
    torch.testing.assert_close(torch.linalg.vector_norm(batch["task"], dim=-1), torch.full((2,), 192**0.5))


def test_bent_path_uses_efficiency_to_select_both_targets():
    z, a, ids = arrays()
    z[0, :2], z[5, :2], z[10, :2] = [0, 0], [3, 0], [3, 4]
    data = EffActionEpisodeData(z, a, ids, config=config(direct_max_chunks=2))
    batch = data.batch_for_rows(np.array([0, 21]), np.array([10, 36]))
    assert batch["path_length"][0].item() == 7
    assert batch["net_distance"][0].item() == 5
    assert batch["direct_branch"].tolist() == [False, False]
    assert batch["valid_mask"].all()
    assert abs(batch["efficiency"][0].item() - 5 / 7) < 1e-6


def test_episode_split_future_goal_and_grid_are_enforced():
    data = EffActionEpisodeData(*arrays(), config=config(episode_stop=1))
    for anchors, goals, message in [([15], [25], "cross episodes"), ([21], [26], "split"), ([0], [6], "grid"), ([0], [20], "sampling range")]:
        with pytest.raises(ValueError, match=message):
            data.batch_for_rows(np.array(anchors), np.array(goals))
    batch = data.sample(100, rng=np.random.default_rng(11))
    assert batch["episode_id"].unique().tolist() == [0]
    assert batch["goal_row"].max() <= 20


def test_offsets_balanced_and_rng_resumption_exact():
    data = EffActionEpisodeData(*arrays(), config=config())
    rng = np.random.default_rng(31)
    saved = rng.bit_generator.state
    first = data.sample(12000, rng=rng)
    counts = torch.bincount(first["goal_chunks"])[1:]
    assert torch.all(torch.abs(counts - 4000) < 250)
    rng.bit_generator.state = saved
    again = data.sample(12000, rng=rng)
    for key in ("anchor_row", "goal_row", "direct_successor_target"):
        torch.testing.assert_close(first[key], again[key])


def test_invalid_intermediate_action_and_zero_distance_are_excluded():
    z, a, ids = arrays()
    a[5] = np.nan
    z[15] = z[0]
    data = EffActionEpisodeData(z, a, ids, config=config())
    assert 0 not in data.rows_by_offset[2]
    assert 0 not in data.rows_by_offset[3]
    batch = data.sample(100, rng=np.random.default_rng(5))
    assert batch["valid_mask"].all()
    assert torch.isfinite(batch["raw_action"]).all()
    assert (batch["net_distance"] > 0).all()


def test_unsupported_relabeling_is_rejected():
    with pytest.raises(ValueError, match="Cross-episode"):
        config(goal_source="random_episode")
    with pytest.raises(ValueError, match="No valid"):
        EffActionEpisodeData(*arrays(), config=config(max_goal_chunks=5))
