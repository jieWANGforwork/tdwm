import numpy as np
import pytest

from tdwm.evaluation.effplan_distance_calibration import distance_quantile


def test_training_only_all_five_step_starts_without_cross_episode_pairs():
    ids = np.repeat([0, 1, 8000], 7)
    states = np.zeros((21, 192))
    states[:7, 0] = np.arange(7)
    states[7:14, 0] = 1000 + np.arange(7)*2
    states[14:, 0] = np.arange(7)*100000
    r = distance_quantile(states, ids, episodes=(0, 1))
    assert r["pair_count"] == 4
    assert r["local_distance_limit"] == np.quantile([5., 5., 10., 10.], 0.95)
    assert r["primitive_lag"] == 5


@pytest.mark.parametrize("fault", ["missing", "short", "nonfinite", "zero"])
def test_invalid_calibration_fails(fault):
    states = np.zeros((7,192)); states[:,0] = np.arange(7)
    ids = np.zeros(7, dtype=int); episodes=(0,)
    if fault == "missing":
        episodes=(1,)
    elif fault == "short":
        states,ids = states[:5],ids[:5]
    elif fault == "nonfinite":
        states[0,0] = np.nan
    else:
        states[:] = 0
    with pytest.raises(ValueError):
        distance_quantile(states,ids,episodes=episodes)
