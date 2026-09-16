"""Real public SWM CEM plus tiny frozen networks; no Cube download."""

import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from torch import nn

from tdwm.adapters.effplan import EffPlanSolver
from tdwm.adapters.effplan_robust import ActionRobustness, RobustEffPlanTrackingCost
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import StatePlanner
from tdwm.methods.effplan_safety import PlannerSafety


class World(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def rollout(self, info, actions, history_size=None):
        assert history_size == 3
        self.calls.append(actions.detach().clone())
        future = info["emb"][..., -1:, :] + nn.functional.pad(actions, (0, 167)).cumsum(2)
        return dict(info, predicted_emb=torch.cat((info["emb"], future), dim=2))


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_risk_uses_public_cem_and_keeps_nominal_refinement_and_final_action(weight):
    world = World()
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).requires_grad_(False).eval()
    planner = StatePlanner(hidden_dim=8).requires_grad_(False).eval()
    cost = RobustEffPlanTrackingCost(
        world, eff, target=True,
        robustness=ActionRobustness(samples=2, shortlist=2, weight=weight),
    )
    solver = EffPlanSolver(
        model=cost, planner=planner, search_iterations=(1, 1), candidates=4,
        elites=2, batch_size=2, seed=42, device="cpu", epsilon=1e-6,
        dynamics_coefficient=0.1, safety=PlannerSafety(10, 5),
    )
    solver.configure(
        action_space=Box(-1, 1, shape=(2, 5), dtype=np.float32), n_envs=2,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, history_len=1, action_block=5),
    )
    info = dict(pixels=torch.zeros(2, 1, 3, 1, 1),
                emb=torch.zeros(2, 1, 192), goal_emb=torch.ones(2, 1, 192))
    result = solver.solve(info)
    assert isinstance(solver.inner, swm.solver.CEMSolver)
    assert result["actions"].shape == (2, 5, 25)
    assert torch.isfinite(result["actions"]).all()
    assert [a.shape[1] for a in world.calls] == (
        [4, 1, 4] if weight == 0 else [4, 2, 2, 1, 4, 2, 2]
    )
    assert solver.last_diagnostics["state_updates"] == 1
    assert solver.last_diagnostics["final_search_after_last_state_update"]
    assert all(a.shape[2] == 5 for a in world.calls)
    assert all(p.grad is None for p in cost.parameters())
