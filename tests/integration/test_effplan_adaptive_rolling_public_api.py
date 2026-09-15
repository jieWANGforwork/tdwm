"""Real public World/WorldModelPolicy execution, with a deterministic search stub."""

import gymnasium as gym
import numpy as np
import pytest
import stable_worldmodel as swm
import torch

import tdwm.adapters.effplan_adaptive_rolling as module
from tdwm.adapters.effplan_adaptive_rolling import AdaptiveRollingPolicy


class Counter(gym.Env):
    def __init__(self, success_at=None):
        self.action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-1000, 1000, (1,), np.float32)
        self.success_at = success_at
        self.n = 0

    def reset(self, **kwargs):
        self.n = 0
        return np.zeros(1, np.float32), {'actual_position': np.float32(0)}

    def step(self, action):
        assert np.isfinite(action).all()
        self.n += 1
        return np.array([self.n], np.float32), 0., self.n == self.success_at, False, {
            'actual_position': np.float32(self.n),
        }


class DecisionStub:
    observed_starts = []
    budgets = []
    expected_threshold = None

    def __init__(self, *, budget, **kwargs):
        assert kwargs.get('efficiency_threshold') == self.expected_threshold
        self.budget = budget
        self.solve_calls = 0
        self.records, self.artifacts = [], []

    def configure(self, *, action_space, n_envs, config):
        self.action_dim, self.n_envs, self.horizon = 25, 1, config.horizon

    def __call__(self, *a, **kw): return self.solve(*a, **kw)

    def solve(self, info_dict, init_action=None):
        self.solve_calls += 1
        assert self.solve_calls == 1
        start = int(info_dict['actual_position'].reshape(-1)[0])
        self.observed_starts.append(start)
        self.budgets.append(self.budget)
        blocks = min(2 if start == 0 else 1, self.budget//5)
        self.policy.cfg = swm.PlanConfig(blocks, blocks, action_block=5, warm_start=False)
        self.records.append(dict(action_blocks=blocks, intermediate_nodes=blocks-1,
                                 planned_primitive_steps=5*blocks))
        self.artifacts.append(dict(real_start=start))
        return {'actions': torch.zeros(1, blocks, 25)}


@pytest.mark.parametrize('budget', [50, 100, 200])
@pytest.mark.parametrize('success_at', [None, 3, 13])
@pytest.mark.parametrize('efficiency_threshold', [None, 0.8])
def test_roll_until_total_budget_or_success_using_real_updated_start(monkeypatch, budget, success_at, efficiency_threshold):
    monkeypatch.setattr(module, 'AdaptiveDecisionSolver', DecisionStub)
    monkeypatch.setattr(DecisionStub, 'expected_threshold', efficiency_threshold)
    DecisionStub.observed_starts = []; DecisionStub.budgets = []
    name = 'TDWMAdaptiveRollingCounter-v0'
    if name not in gym.registry: gym.register(name, entry_point=Counter)
    policy = AdaptiveRollingPolicy(
        model=None, planner=None, safety=None, budget=budget, device='cpu',
        search_iterations=(1,), epsilon=1e-6, dynamics_coefficient=0.1,
        efficiency_threshold=efficiency_threshold,
    )
    world = swm.World(name, num_envs=1, max_episode_steps=budget, add_pixels=False,
                      success_at=success_at)
    try:
        world.set_policy(policy)
        metrics = world.evaluate(episodes=1, seed=42, reset_mode='wait')
        steps = success_at or budget
        assert policy.executed_steps.tolist() == [steps]
        assert metrics['success_rate'] == (100 if success_at else 0)
        expected = [0]+list(range(10, steps, 5))
        assert DecisionStub.observed_starts == expected
        assert DecisionStub.budgets == [budget-s for s in expected]
        rounds = policy.records[0]
        assert sum(r['executed_primitive_steps'] for r in rounds) == steps
        assert all(r['planned_primitive_steps'] <= r['remaining_budget_before'] for r in rounds)
        assert [r['start_primitive_step'] for r in rounds] == expected
        assert all(r['planned_primitive_steps'] == 5*(r['intermediate_nodes']+1) for r in rounds)
    finally:
        world.close()


def test_real_decision_solver_trims_padding_and_updates_public_policy_horizon(monkeypatch):
    from types import SimpleNamespace
    from tdwm.adapters.effplan_adaptive import AdaptiveEffPlanSolver
    from tdwm.methods.effplan import StatePlanner
    from tdwm.methods.effplan_safety import PlannerSafety

    def search(self, info, init_action):
        self.records = [{'action_blocks': 3}]
        # A sentinel beyond the actual plan must never reach the policy buffer.
        actions = torch.zeros(1, 10, 25); actions[:, 3:] = 999
        return {'actions': actions}
    monkeypatch.setattr(AdaptiveEffPlanSolver, 'solve', search)
    solver = module.AdaptiveDecisionSolver(
        model=None, planner=StatePlanner(hidden_dim=8).requires_grad_(False),
        limits=None, safety=PlannerSafety(10, 5), budget=50, device='cpu',
    )
    solver.policy = SimpleNamespace(cfg=swm.PlanConfig(10, 10, action_block=5))
    out = solver.solve({})
    assert out['actions'].shape == (1, 3, 25)
    assert out['actions'].eq(0).all()
    assert solver.policy.cfg.horizon == solver.policy.cfg.receding_horizon == 3


def test_ragged_vector_episodes_do_not_replan_successful_environments(monkeypatch):
    monkeypatch.setattr(module, 'AdaptiveDecisionSolver', DecisionStub)
    name = 'TDWMAdaptiveRollingCounter-v0'
    if name not in gym.registry: gym.register(name, entry_point=Counter)
    made = []
    def vary(env):
        env.unwrapped.success_at = 3 if not made else None
        made.append(env)
        return env
    policy = AdaptiveRollingPolicy(model=None, planner=None, safety=None, budget=50,
                                   device='cpu', search_iterations=(1,), epsilon=1e-6,
                                   dynamics_coefficient=0.1)
    world = swm.World(name, num_envs=2, max_episode_steps=50, add_pixels=False,
                      pre_wrappers=[vary])
    try:
        world.set_policy(policy)
        metrics = world.evaluate(episodes=2, seed=42, reset_mode='wait')
        assert metrics['success_rate'] == 50
        assert policy.executed_steps.tolist() == [3, 50]
        assert len(policy.records[0]) == 1
        assert len(policy.records[1]) == 9
    finally:
        world.close()
