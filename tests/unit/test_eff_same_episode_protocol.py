"""The no-cross-episode ablation changes exactly one training setting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.training.eff_data import EffEpisodeReplay
from tdwm.training.eff_protocol import load_eff_protocol
from tdwm.training.eff_run import EffRunSettings


CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "experiment"


def test_same_episode_ablation_changes_only_cross_episode_probability():
    baseline = load_eff_protocol(
        CONFIGS / "effplan_cube_stable_p_sparse_v1.yaml", stage="eff_training"
    )
    actual = load_eff_protocol(
        CONFIGS / "effplan_cube_same_episode_v1.yaml", stage="eff_training"
    )
    expected = deepcopy(baseline)
    assert expected["eff_training"]["settings"]["cross_episode_probability"] == .3
    expected["eff_training"]["settings"]["cross_episode_probability"] = 0.
    assert actual == expected
    settings = EffRunSettings(**actual["eff_training"]["settings"])
    assert settings.total_updates == 127960
    assert settings.seed == 3072
    assert settings.v_parameterization == "total_work"
    assert settings.backup_primitive_steps == 50


@pytest.mark.parametrize("seed", [3072, 50, 42])
def test_no_cross_episode_goals_and_ten_block_backup(seed):
    config = load_eff_protocol(
        CONFIGS / "effplan_cube_same_episode_v1.yaml", stage="eff_training"
    )
    settings = EffRunSettings(**config["eff_training"]["settings"])
    ids = np.repeat(np.arange(4), 201)
    states = np.zeros((len(ids), 192), dtype=np.float32)
    states[:, 0] = np.tile(np.arange(201), 4)
    states[:, 1] = ids
    replay = EffEpisodeReplay(
        SimpleNamespace(latents=states, episode_ids=ids),
        episodes=(0, 1, 2), stride=config["data"]["state_stride"],
        terminal_at_state=np.zeros(len(ids), dtype=bool),
    )
    batch = replay.sample(rng=np.random.default_rng(seed), **settings.sample_arguments())
    assert torch.equal(batch.anchor_episodes, batch.goal_episodes)
    assert (batch.goal_rows > batch.anchor_rows).all()
    assert batch.efficiency_known.all()
    direct = batch.offset_chunks <= 10
    assert direct.any() and (~direct).any()
    assert torch.equal(batch.goal_reached, direct)
    assert torch.equal(batch.bootstrap_rows[direct], batch.goal_rows[direct])
    assert ((batch.bootstrap_rows-batch.anchor_rows)[~direct] == 50).all()
    torch.testing.assert_close(batch.observed_cost,
                               (batch.bootstrap_rows-batch.anchor_rows).float())
