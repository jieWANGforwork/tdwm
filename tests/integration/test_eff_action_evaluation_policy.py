"""Installed SWM CEM/WorldModelPolicy integration without environment or GPU."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tdwm.adapters.eff_action import make_eff_action_policy
from tdwm.evaluation.eff_action import build_eff_action_action_processor


class Encoder(nn.Module):
    input_dim, emb_dim = 25, 192

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, action):
        return torch.nn.functional.pad(action, (0, 167)) + self.anchor


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = Encoder()

    def encode(self, info):
        raise AssertionError(
            "This synthetic test supplies pre-encoded real observations."
        )

    def rollout(self, *args, **kwargs):
        raise AssertionError("Neither EffAction deployment path may roll out F.")


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, state, action_embedding, task):
        return action_embedding + self.anchor


class V(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, psi, task):
        return 1.0 + (psi[..., :25] - 0.4).square().sum(dim=-1) + self.anchor


class GradientPlanner(nn.Module):
    raw_action_dim = 25

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.gradients = []

    def forward(self, state, reference, current, goal, cost, gradient):
        self.gradients.append(gradient.detach().clone())
        return -0.1 * gradient + self.anchor


@pytest.mark.parametrize("method", ["EffAction", "EffActionPlan"])
def test_public_policy_executes_distinct_primitives_and_replans_from_real_state(method):
    import stable_worldmodel as swm

    planning = dict(
        horizon=1,
        receding_horizon=1,
        action_block=5,
        history_len=1,
        warm_start=False,
        planning_seed=42,
        iterations=2,
    )
    if method == "EffAction":
        planning.update(
            candidates=8, elites=3, solver_batch_size=1, initial_variance=1.0
        )
    else:
        planning.update(
            initialization="zeros", initial_std=0.0, lower_bound=-1.0, upper_bound=1.0
        )
    planner = GradientPlanner() if method == "EffActionPlan" else None
    processor, _ = build_eff_action_action_processor(
        dict(mean=[0.2] * 5, scale=[0.5] * 5)
    )
    policy = make_eff_action_policy(
        world_model=WorldModel(),
        successor=G(),
        value=V(),
        planner=planner,
        planning=planning,
        epsilon=1e-6,
        method=method,
        process={"action": processor},
    )
    assert isinstance(policy, swm.policy.WorldModelPolicy)
    assert isinstance(policy.solver, swm.solver.CEMSolver) == (method == "EffAction")
    env = SimpleNamespace(
        num_envs=2,
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2, 5)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(5,)),
    )
    policy.set_env(env)
    original_solve = policy.solver.solve
    calls, plans = [], []

    def recording_solve(info, init_action=None):
        calls.append(info["emb"][:, 0, 0].tolist())
        result = original_solve(info, init_action=init_action)
        plans.append(result["actions"].clone())
        return result

    policy.solver.solve = recording_solve
    actions = []
    for step in range(7):
        info = {
            "emb": np.full((2, 1, 192), float(step), dtype=np.float32),
            "goal_emb": np.full((2, 1, 192), 10.0, dtype=np.float32),
            "terminated": np.array([False, step >= 2]),
        }
        with torch.inference_mode():
            actions.append(policy.get_action(info))
    assert calls == [[0.0, 0.0], [5.0]]
    raw_first_plan = plans[0][0, 0].reshape(5, 5).numpy()
    expected_primitives = processor.inverse_transform(raw_first_plan)
    np.testing.assert_allclose(np.asarray(actions)[:5, 0], expected_primitives)
    assert np.isnan(np.asarray(actions)[2:, 1]).all()
    assert np.isfinite(np.asarray(actions)[:, 0]).all()
    if planner is not None:
        assert len(planner.gradients) == 4
        assert any(bool((gradient != 0).any()) for gradient in planner.gradients)
        assert all(
            parameter.grad is None for parameter in policy.solver.model.parameters()
        )
        assert np.all(np.abs(raw_first_plan) <= 1.0)
