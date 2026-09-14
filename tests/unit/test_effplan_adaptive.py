import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tdwm.adapters.effplan_adaptive import PlanLengthLimit
from tdwm.methods.effplan_adaptive import adaptive_state_path
from tdwm.methods.effplan_safety import PlannerSafety, PlannerSafetyRuntime


class ZeroP(nn.Module):
    def forward(self, left, candidate, right, value, gradient):
        assert value.shape == (1,) and gradient.shape == (1, 192)
        return torch.zeros_like(candidate)


def path(power, cap):
    start = torch.zeros(1, 192)
    goal = start.clone(); goal[:, 0] = 100
    return adaptive_state_path(
        ZeroP(), start, goal,
        lambda a, b: torch.linalg.vector_norm(a-b, dim=-1).pow(power),
        max_blocks=cap, safety=PlannerSafetyRuntime(PlannerSafety(10, 5)),
    )


@pytest.mark.parametrize("cap", [1, 2, 5, 10, 20, 40])
def test_budget_bound_k_plus_one_and_order(cap):
    states, record = path(2, cap)
    assert states.shape == (1, cap+1, 192)
    assert record['intermediate_nodes']+1 == record['action_blocks'] == cap
    assert states[0, 0, 0] == 0 and states[0, -1, 0] == 100
    assert bool((states[0, 1:, 0] > states[0, :-1, 0]).all())
    assert all(r['relative_gain'] > 1e-6 for r in record['split_attempts'])


def test_no_benefit_no_forced_midpoint_or_five_actions():
    states, record = path(1, 40)
    assert states.shape == (1, 2, 192)
    assert record['action_blocks'] == 1
    assert len(record['split_attempts']) == 1
    assert not record['split_attempts'][0]['accepted']


def test_identical_endpoints_finite_no_duplicate():
    z = torch.zeros(1, 192)
    states, record = adaptive_state_path(
        ZeroP(), z, z, lambda a, b: torch.linalg.vector_norm(a-b, dim=-1),
        max_blocks=40, safety=PlannerSafetyRuntime(PlannerSafety(10, 5)),
    )
    assert torch.isfinite(states).all() and record['action_blocks'] == 1


class CounterEnv(gym.Env):
    def __init__(self, success_at=None):
        self.action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.n = 0
        self.success_at = success_at

    def reset(self, **kwargs):
        self.n = 0
        return np.zeros(1), {}

    def step(self, action):
        self.n += 1
        return np.zeros(1), 0., self.n == self.success_at, False, {}


@pytest.mark.parametrize('success_at', [None, 3, 5])
def test_execution_ends_at_success_or_length_never_padding(success_at):
    env = PlanLengthLimit(CounterEnv(success_at), 50)
    env.reset(); env.set_length(5)
    for _ in range(success_at or 5):
        _, _, term, trunc, _ = env.step(np.zeros(5))
    assert term == (success_at is not None)
    assert trunc == (success_at is None)
    assert env.steps == (success_at or 5)
    with pytest.raises(RuntimeError): env.step(np.ones(5))
    assert env.unwrapped.n == env.steps


def test_length_cannot_exceed_budget_or_be_replaced():
    env = PlanLengthLimit(CounterEnv(), 50); env.reset()
    with pytest.raises(ValueError): env.set_length(51)
    env.set_length(5)
    with pytest.raises(ValueError): env.set_length(10)
