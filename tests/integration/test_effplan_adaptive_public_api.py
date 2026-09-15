"""Exercise adaptive planning against the real public SWM CEM and World APIs."""

import gymnasium as gym
import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from torch import nn

from tdwm.adapters.effplan_adaptive import (
    AdaptiveEffPlanSolver, AdaptiveTrackingCost, PlanExecutionLimits,
)
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import StatePlanner
from tdwm.methods.effplan_safety import PlannerSafety


class ToyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.horizons = []

    def rollout(self, info, actions, history_size):
        assert history_size == 3
        self.horizons.append(actions.shape[-2])
        history = info['emb']
        future = history[..., -1:, :] + torch.nn.functional.pad(actions, (0, 167)).cumsum(-2)
        return dict(predicted_emb=torch.cat((history, future), -2))


def info():
    return dict(pixels=torch.zeros(1, 1, 3, 1, 1), emb=torch.zeros(1, 1, 192),
                goal_emb=torch.ones(1, 1, 192))


@pytest.mark.parametrize('horizon', [1, 2, 10, 20, 40])
def test_variable_tracking_all_actions_through_f(horizon):
    world = ToyWorld()
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).eval().requires_grad_(False)
    model = AdaptiveTrackingCost(world, eff, target=True)
    data = {k: v[:, None].expand(-1, 3, *v.shape[1:]) for k, v in info().items()}
    nodes = torch.zeros(1, 3, horizon+1, 192)
    actions = torch.zeros(1, 3, horizon, 25); actions[..., -1, 0] = 2
    data['effplan_nodes'] = nodes
    cost = model.get_cost(data, actions)
    torch.testing.assert_close(cost, torch.full((1, 3), 4/horizon))
    assert world.horizons == [horizon]


class Counter(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(1, np.float32), 0., False, False, {}


@pytest.mark.parametrize('efficiency_threshold', [None, 0.8])
def test_public_cem_once_all_networks_frozen(monkeypatch, efficiency_threshold):
    import tdwm.adapters.effplan_adaptive as module
    world = ToyWorld()
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).eval().requires_grad_(False)
    p = StatePlanner(hidden_dim=8).eval().requires_grad_(False)
    limits = PlanExecutionLimits(50); limits.wrap(Counter()).reset()
    def fixed(*args, **kwargs):
        if efficiency_threshold is not None:
            assert kwargs['efficiency_threshold'] == efficiency_threshold
            assert 'minimum_relative_gain' not in kwargs
        states = torch.zeros(1, 4, 192); states[:, -1] = 1
        return states, dict(intermediate_nodes=2, action_blocks=3)
    if efficiency_threshold is None:
        monkeypatch.setattr(module, 'adaptive_state_path', fixed)
    else:
        import tdwm.methods.effplan_efficiency as efficiency_module
        monkeypatch.setattr(efficiency_module, 'efficiency_state_path', fixed)
    solver = AdaptiveEffPlanSolver(
        model=AdaptiveTrackingCost(world, eff, target=True), planner=p,
        limits=limits, safety=PlannerSafety(10, 5), budget=50,
        search_iterations=(1, 1), candidates=4, elites=2, device='cpu',
        efficiency_threshold=efficiency_threshold,
    )
    solver.configure(action_space=gym.spaces.Box(-1, 1, (1, 5), np.float32), n_envs=1,
                     config=swm.PlanConfig(10, 10, action_block=5))
    out = solver.solve(info())
    assert out['actions'].shape == (1, 10, 25)
    assert limits.wrappers[0].limit == 15
    assert solver.records[0]['action_blocks'] == 3
    assert solver.artifacts[0]['actions'].shape == (1, 3, 25)
    assert set(world.horizons) == {3}
    assert all(t.grad is None for model in (world, eff, p) for t in model.parameters())
    with pytest.raises(RuntimeError, match='never replan'): solver.solve(info())


def test_public_world_truncation_does_not_count_success_or_step_padding():
    name = 'TDWMAdaptiveCounter-v0'
    if name not in gym.registry: gym.register(name, entry_point=Counter)
    limits = PlanExecutionLimits(50)
    class FixedSolver:
        calls = 0
        action_dim = 25
        n_envs = 2
        horizon = 10
        def configure(self, **kwargs): pass
        def __call__(self, *args, **kwargs): return self.solve(*args, **kwargs)
        def solve(self, *args, **kwargs):
            self.calls += 1
            if self.calls != 1: raise AssertionError('replanned')
            limits.set_lengths([5, 10])
            return {'actions': torch.zeros(2, 10, 25)}
    solver = FixedSolver()
    world = swm.World(name, num_envs=2, max_episode_steps=50, add_pixels=False,
                      pre_wrappers=[limits.wrap])
    try:
        world.set_policy(swm.policy.WorldModelPolicy(solver, swm.PlanConfig(10, 10, action_block=5)))
        metrics = world.evaluate(episodes=2, seed=42, reset_mode='wait')
        assert metrics['success_rate'] == 0
        assert solver.calls == 1
        assert [w.steps for w in limits.wrappers] == [5, 10]
    finally:
        world.close()
