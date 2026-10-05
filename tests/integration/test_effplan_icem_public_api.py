"""Actual installed SWM iCEM, synthetic costs only; no downloads or GPU."""

import json

import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from torch import nn

from tdwm.adapters.effplan import EffPlanSolver, EffPlanTrackingCost
from tdwm.adapters.effplan_icem import (
    EffPlanICEMSettings,
    PrimitiveICEMSolver,
    PrimitiveTrackingCost,
    blocks_to_primitives,
    load_effplan_icem,
    primitives_to_blocks,
)
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import StatePlanner


class QuadraticCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.seen = []

    def get_cost(self, info, actions):
        self.seen.append(actions.detach().clone())
        return (actions - 0.7).square().sum((-1, -2))


def make_solver(cost, *, beta=2.0, samples=16, iterations=3, seed=42, batch_size=2):
    solver = PrimitiveICEMSolver(
        model=cost, settings=EffPlanICEMSettings(noise_beta=beta, n_elite_keep=2),
        num_samples=samples, topk=4, n_steps=iterations, var_scale=1.0,
        device="cpu", seed=seed, batch_size=batch_size,
    )
    solver.configure(
        action_space=Box(-1, 1, (2, 5), dtype=np.float32), n_envs=2,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, history_len=1, action_block=5),
    )
    return solver


def test_import_and_config_manifest(tmp_path):
    path = tmp_path / "icem.json"
    path.write_text(json.dumps(dict(noise_beta=2, alpha=0.1, n_elite_keep=5, return_mean=False)))
    settings = load_effplan_icem(path)
    assert settings == EffPlanICEMSettings()
    assert settings.manifest()["implementation"] == "stable_worldmodel.solver.ICEMSolver"
    assert settings.manifest()["population_decay"] is False
    assert "none" in settings.manifest()["action_clipping"]


@pytest.mark.parametrize("kwargs", [
    {"noise_beta": -1}, {"noise_beta": float("nan")}, {"noise_beta": True},
    {"alpha": -1}, {"alpha": 1}, {"alpha": float("inf")},
    {"n_elite_keep": -1}, {"n_elite_keep": True}, {"return_mean": True},
])
def test_invalid_settings_fail_closed(kwargs):
    with pytest.raises(ValueError):
        EffPlanICEMSettings(**kwargs)


@pytest.mark.parametrize("shape", [(2, 5, 25), (2, 7, 5, 25), (2, 0, 25)])
def test_time_and_dimension_order_are_exactly_preserved(shape):
    actions = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    primitive = blocks_to_primitives(actions)
    torch.testing.assert_close(primitives_to_blocks(primitive), actions, rtol=0, atol=0)
    if actions.numel():
        assert primitive[..., 1, 4].eq(actions[..., 0, 9]).all()
        assert primitive[..., 5, 4].eq(actions[..., 1, 4]).all()


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        blocks_to_primitives(torch.zeros(2, 5, 5))
    with pytest.raises(ValueError):
        primitives_to_blocks(torch.zeros(2, 24, 5))


@pytest.mark.parametrize("override", [
    {"method": "Eff"}, {"action_robustness_path": "risk.json"},
    {"execution_action_bounds": True}, {"adaptive_rolling": True},
    {"offset_window": True}, {"sample_numbers": (1, 2)},
])
def test_formal_entry_rejects_mixed_ablations_before_loading_models(monkeypatch, override):
    import tdwm.evaluation.effplan as evaluation

    monkeypatch.setattr(evaluation, "load_eff_protocol", lambda *args, **kwargs: {})
    args = dict(config_path="unused", dataset_path="unused", lewm_checkpoint="unused",
                selection_path="unused", output_dir="unused", method="EffPlan", device="cpu",
                icem_path="settings.json")
    with pytest.raises(ValueError, match="without other ablations"):
        evaluation.evaluate_effplan(**(args | override))


def test_cost_adapter_is_exact_nominal_score_and_keeps_context():
    original = QuadraticCost()
    primitive = torch.randn(2, 7, 25, 5)
    info = {"pixels": torch.zeros(2, 7, 1, 3, 2, 2)}
    expected = original.get_cost(info, primitives_to_blocks(primitive))
    actual = PrimitiveTrackingCost(original).get_cost(info, primitive)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(original.seen[-1], original.seen[-2], rtol=0, atol=0)


def test_public_solver_best_candidate_elites_budget_and_determinism():
    outputs = []
    for _ in range(2):
        cost = QuadraticCost()
        solver = make_solver(cost)
        output = solver.solve({"dummy": torch.zeros(2, 1)})
        outputs.append(output["actions"])
        assert type(solver.optimizer) is swm.solver.ICEMSolver
        assert (solver.horizon, solver.action_dim, solver.n_envs) == (5, 25, 2)
        assert solver.optimizer.horizon == 25 and solver.optimizer.action_dim == 5
        assert output["actions"].shape == (2, 5, 25)
        assert output["mean"][0].shape == output["var"][0].shape == (2, 5, 25)
        assert len(cost.seen) == 3
        assert sum(a.shape[1] for a in cost.seen) == 16 * 3
        last = cost.seen[-1]
        best = (last - 0.7).square().sum((-1, -2)).argmin(1)
        torch.testing.assert_close(output["actions"], last[torch.arange(2), best], rtol=0, atol=0)
        first_best = (cost.seen[0] - 0.7).square().sum((-1, -2)).argsort(1)[:, :2]
        expected_kept = cost.seen[0][torch.arange(2)[:, None], first_best]
        torch.testing.assert_close(cost.seen[1][:, 1:3], expected_kept, rtol=0, atol=0)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


def test_no_hidden_action_clipping_and_warm_start_roundtrip():
    cost = QuadraticCost()
    solver = make_solver(cost, iterations=1)
    initial = torch.arange(250, dtype=torch.float32).reshape(2, 5, 25) / 10
    solver.solve({"dummy": torch.zeros(2, 1)}, init_action=initial)
    torch.testing.assert_close(cost.seen[0][:, 0], initial, rtol=0, atol=0)
    assert cost.seen[0].abs().max() > 1


def test_correlation_runs_across_all_primitive_steps_including_block_boundaries():
    differences = []
    boundaries = []
    for beta in (0, 2):
        cost = QuadraticCost()
        make_solver(cost, beta=beta, samples=1024, iterations=1).solve({"dummy": torch.zeros(2, 1)})
        primitive = blocks_to_primitives(cost.seen[0][:, 1:])
        delta = primitive.diff(dim=-2).square()
        differences.append(delta.mean())
        # Steps 5->6, 10->11, 15->16 and 20->21 must also be correlated.
        boundaries.append(delta[..., 4::5, :].mean())
    assert differences[1] < 0.4 * differences[0]
    assert boundaries[1] < 0.4 * boundaries[0]


def test_subset_batch_and_iteration_changes_keep_public_contract():
    cost = QuadraticCost()
    solver = make_solver(cost, batch_size=1)
    solver.n_steps = 1
    output = solver.solve({"dummy": torch.zeros(1, 1)}, init_action=torch.zeros(1, 0, 25))
    assert output["actions"].shape == (1, 5, 25)
    assert len(cost.seen) == 1


class FrozenWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.seen = []

    def rollout(self, info, actions, history_size=None):
        assert history_size == 3 and actions.shape[-2:] == (5, 25)
        self.seen.append(actions.detach().clone())
        future = info["emb"][..., -1:, :] + torch.nn.functional.pad(actions, (0, 167)).cumsum(2)
        return dict(info, predicted_emb=torch.cat((info["emb"], future), dim=2))


def test_full_p_icem_feedback_uses_returned_best_actions_without_model_updates():
    world = FrozenWorld()
    eff = EffModel(g_hidden_dim=16, v_hidden_dim=16).requires_grad_(False).eval()
    planner = StatePlanner(hidden_dim=16).requires_grad_(False).eval()
    solver = EffPlanSolver(
        model=EffPlanTrackingCost(world, eff, target=True), planner=planner,
        search_iterations=(2, 1), candidates=8, elites=3, batch_size=2,
        seed=42, device="cpu", epsilon=1e-6, dynamics_coefficient=0.1,
        icem_settings=EffPlanICEMSettings(n_elite_keep=1),
    )
    solver.configure(
        action_space=Box(-1, 1, (2, 5), dtype=np.float32), n_envs=2,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, history_len=1, action_block=5),
    )
    output = solver.solve(dict(pixels=torch.zeros(2, 1, 3, 1, 1),
                               emb=torch.zeros(2, 1, 192), goal_emb=torch.ones(2, 1, 192)))
    assert output["actions"].shape == (2, 5, 25)
    assert [a.shape[1] for a in world.seen] == [8, 8, 1, 8]
    for env in range(2):
        assert any(torch.equal(world.seen[2][env, 0], candidate) for candidate in world.seen[1][env])
    assert solver.last_diagnostics["candidate_rollouts"] == 24
    assert solver.last_diagnostics["state_updates"] == 1
    assert solver.last_diagnostics["final_search_after_last_state_update"]
    assert all(p.grad is None for module in (world, eff, planner) for p in module.parameters())
