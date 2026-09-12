"""Small CPU integration with the installed stable-worldmodel CEM (no data)."""

from __future__ import annotations

import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from torch import nn

from tdwm.adapters.effplan import (
    EffCEMCost,
    EffPlanSolver,
    EffPlanTrackingCost,
    distribute_cem_iterations,
)
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import StatePlanner


class FrozenWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.seen_actions = []
        self.seen_starts = []

    def rollout(self, info, actions, history_size=None):
        assert history_size == 3
        self.seen_actions.append(actions.detach().clone())
        self.seen_starts.append(info["emb"][..., 0, :].detach().clone())
        history = info["emb"]
        increment = torch.nn.functional.pad(actions, (0, 167))
        future = history[..., -1:, :] + increment.cumsum(dim=2)
        return dict(info, predicted_emb=torch.cat((history, future), dim=2))


def _info(batch=2):
    return {
        "pixels": torch.zeros(batch, 1, 3, 1, 1),
        "emb": torch.zeros(batch, 1, 192),
        "goal_emb": torch.ones(batch, 1, 192),
    }


def _expanded(info, candidates):
    return {
        key: value[:, None].expand(-1, candidates, *value.shape[1:])
        for key, value in info.items()
    }


def _eff():
    return EffModel(g_hidden_dim=16, v_hidden_dim=16).requires_grad_(False).eval()


def test_eff_final_action_reaches_f_before_state_only_g():
    world, eff = FrozenWorld(), _eff()
    cost = EffCEMCost(world, eff, target=True)
    seen = []
    handle = eff.target_g.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach().clone())
    )
    actions = torch.zeros(2, 3, 5, 25)
    actions[..., -1, 0] = 7
    scores = cost.get_cost(_expanded(_info(), 3), actions)
    handle.remove()
    assert scores.shape == (2, 3)
    assert world.seen_actions[-1].shape == (2, 3, 5, 25)
    assert seen[-1][..., 0].eq(7).all()
    assert len(seen) == 1


def test_tracking_keeps_terminal_goal_and_does_not_reset_f_at_middle_nodes():
    world = FrozenWorld()
    cost = EffPlanTrackingCost(world, _eff(), target=True)
    actions = torch.zeros(2, 3, 5, 25)
    info = _expanded(_info(), 3)
    info["effplan_nodes"] = torch.zeros(2, 3, 6, 192)
    info["effplan_nodes"][..., -1, 0] = 5
    scores = cost.get_cost(info, actions)
    torch.testing.assert_close(scores, torch.full((2, 3), 5.0))
    assert len(world.seen_starts) == 1
    assert world.seen_starts[0].eq(0).all()


def test_fixed_budget_distribution_includes_final_search():
    allocation = distribute_cem_iterations(total_iterations=30, searches=9)
    assert allocation == (3, 3, 3, 3, 3, 3, 4, 4, 4)
    assert sum(allocation) == 30
    with pytest.raises(ValueError):
        distribute_cem_iterations(total_iterations=8, searches=9)


def test_public_cem_state_planner_final_search_and_returned_action_feedback():
    from gymnasium.spaces import Box

    world, eff = FrozenWorld(), _eff()
    planner = StatePlanner(hidden_dim=16).requires_grad_(False).eval()
    with torch.no_grad():
        planner.network[-1].bias.fill_(0.01)
    cost = EffPlanTrackingCost(world, eff, target=True)
    solver = EffPlanSolver(
        model=cost,
        planner=planner,
        search_iterations=(1, 1, 1),
        candidates=4,
        elites=2,
        batch_size=2,
        seed=42,
        device="cpu",
        epsilon=1e-6,
        dynamics_coefficient=0.1,
    )
    solver.configure(
        action_space=Box(low=-1.0, high=1.0, shape=(2, 5), dtype=np.float32),
        n_envs=2,
        config=swm.PlanConfig(
            horizon=5, receding_horizon=5, history_len=1, action_block=5
        ),
    )
    output = solver.solve(_info())
    assert output["actions"].shape == (2, 5, 25)
    assert isinstance(solver.inner, swm.solver.CEMSolver)
    assert solver.last_diagnostics["state_updates"] == 2
    assert solver.last_diagnostics["final_search_after_last_state_update"]
    # Search, returned-mean reroll, search, returned-mean reroll, FINAL search.
    assert [a.shape[1] for a in world.seen_actions] == [4, 1, 4, 1, 4]
    assert all(z.eq(0).all() for z in world.seen_starts)
    assert solver.last_diagnostics["candidate_rollouts"] == 12
    assert all(p.grad is None for p in world.parameters())
    assert all(p.grad is None for p in eff.parameters())


def test_effplan_refuses_trainable_world_model():
    world = FrozenWorld().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen LeWM"):
        EffCEMCost(world, _eff(), target=True)
