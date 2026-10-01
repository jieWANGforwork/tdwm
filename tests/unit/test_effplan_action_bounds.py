"""Regression checks for opt-in action bounds, without data or checkpoints."""

import pytest
import torch

from tdwm.adapters.action_bounds import ActionBox, BoundedCEMCost

MEAN = (0.0108846967805138, -0.0031414329928013945, 0.0026465825646189004,
        0.0004239286647865428, 0.15925256653030173)
SCALE = (0.2894198325506555, 0.39371697495490987, 0.6431365217957304,
         0.3928016202283424, 0.2503073640045173)


def test_bounds_adapter_import():
    from tdwm.adapters.action_bounds import ActionBox, BoundedCEMCost

    assert ActionBox and BoundedCEMCost


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_run_bounds_repeat_for_all_five_primitive_actions(dtype):
    box = ActionBox(MEAN, SCALE)
    actions = torch.full((2, 3, 5, 25), 1e5, dtype=dtype)
    actions[0].neg_()
    box.project_(actions)
    mean, scale = actions.new_tensor(MEAN), actions.new_tensor(SCALE)
    raw = actions.reshape(2, 3, 5, 5, 5) * scale + mean
    assert raw.min() >= -1 and raw.max() <= 1
    torch.testing.assert_close(raw[0], -torch.ones_like(raw[0]))
    torch.testing.assert_close(raw[1], torch.ones_like(raw[1]))
    # Gripper uses execution [-1,1], NOT observed training min/max.
    assert raw[..., 4].max() > 0.99 and raw[..., 4].min() < -0.99


@pytest.mark.parametrize("mean,scale", [([], []), ([0], [0]), ([0], [-1]),
                                       ([float("nan")], [1]), ([0, 1], [1])])
def test_invalid_scaler_rejected(mean, scale):
    with pytest.raises(ValueError):
        ActionBox(tuple(mean), tuple(scale))


def test_bad_action_shape_or_nan_is_not_silently_repaired():
    box = ActionBox(MEAN, SCALE)
    with pytest.raises(ValueError):
        box.project_(torch.zeros(2, 7))
    with pytest.raises(FloatingPointError):
        box.project_(torch.full((1, 25), float("nan")))


def test_projection_accepts_public_solver_inference_tensor():
    with torch.inference_mode():
        actions = torch.full((1, 5, 25), 100.0)
    ActionBox(MEAN, SCALE).project_(actions)
    raw = actions.reshape(-1, 5) * torch.tensor(SCALE) + torch.tensor(MEAN)
    assert raw.min() >= -1 and raw.max() <= 1


class Cost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(()), requires_grad=False))
        self.seen = []

    def get_cost(self, info, actions):
        self.seen.append(actions.clone())
        return -actions.sum(dim=(-1, -2))


def test_projection_is_in_place_before_f_and_elite_fitting():
    cost = Cost()
    bounded = BoundedCEMCost(cost, ActionBox((0.0,), (1.0,)))
    candidates = torch.tensor([[[[-9.0]], [[0.2]], [[8.0]]]])
    identity = candidates.data_ptr()
    scores = bounded.get_cost({}, candidates)
    assert candidates.data_ptr() == identity
    torch.testing.assert_close(cost.seen[0], candidates)
    elites = candidates[:, scores[0].topk(2, largest=False).indices]
    assert elites.mean() < 1  # Would be 4.1 if only the F copy were clipped.
    assert bounded.diagnostics()["projected_candidate_components"] == 2


def public_solver(cost):
    import numpy as np
    import stable_worldmodel as swm
    from gymnasium.spaces import Box

    solver = swm.solver.CEMSolver(model=cost, num_samples=8, topk=3,
                                  n_steps=3, var_scale=10, seed=42, device="cpu")
    solver.configure(action_space=Box(-np.ones((1, 5)), np.ones((1, 5)), dtype=np.float32),
                     n_envs=1, config=swm.PlanConfig(horizon=5, receding_horizon=5,
                                                  action_block=5, history_len=1))
    return solver


def test_installed_public_cem_scores_and_fits_only_bounded_actions():
    box, cost = ActionBox(MEAN, SCALE), Cost()
    solver = public_solver(BoundedCEMCost(cost, box))
    # Include an out-of-bounds warm-start mean; every iteration must project it.
    output = solver.solve({"dummy": torch.zeros(1, 1)}, init_action=torch.full((1, 5, 25), 100.0))
    assert len(cost.seen) == 3
    for actions in [*cost.seen, output["actions"]]:
        raw = actions.reshape(-1, 5) * torch.tensor(SCALE) + torch.tensor(MEAN)
        assert raw.min() >= -1.000001 and raw.max() <= 1.000001


def test_wide_bounds_preserve_public_cem_rng_and_numerical_path():
    original, wrapped = Cost(), Cost()
    wide = ActionBox(MEAN, SCALE, raw_low=-1e6, raw_high=1e6)
    first = public_solver(original).solve({"dummy": torch.zeros(1, 1)})
    second = public_solver(BoundedCEMCost(wrapped, wide)).solve({"dummy": torch.zeros(1, 1)})
    torch.testing.assert_close(first["actions"], second["actions"], atol=0, rtol=0)
    for a, b in zip(original.seen, wrapped.seen, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_effplan_bounds_returned_means_and_refinement_rerolls(monkeypatch):
    import numpy as np
    import stable_worldmodel as swm
    from gymnasium.spaces import Box
    from tdwm.adapters import effplan

    bounds = ActionBox(MEAN, SCALE)

    class PlanningCost(Cost):
        target_critic = None

        def cached_context(self, info, device):
            return dict(emb=torch.zeros(1, 1, 192), goal_emb=torch.ones(1, 1, 192))

        def future_states(self, info, actions):
            raw = actions.reshape(-1, 5) * torch.tensor(SCALE) + torch.tensor(MEAN)
            assert raw.min() >= -1 and raw.max() <= 1
            self.rerolled = True
            return torch.zeros(1, 1, 5, 192)

    monkeypatch.setattr(effplan, "generate_state_path", lambda *a, **k: torch.zeros(1, 6, 192))
    monkeypatch.setattr(effplan, "refine_state_path", lambda p, nodes, *a, **k: nodes)
    model = PlanningCost()
    solver = effplan.EffPlanSolver(
        model=model, planner=torch.nn.Identity(), search_iterations=(1, 1),
        candidates=8, elites=3, batch_size=1, seed=42, device="cpu",
        epsilon=1e-6, dynamics_coefficient=0.1, action_bounds=bounds,
    )
    solver.configure(action_space=Box(-1, 1, shape=(1, 5), dtype=np.float32), n_envs=1,
                     config=swm.PlanConfig(horizon=5, receding_horizon=5,
                                          action_block=5, history_len=1))
    warm = torch.full((1, 5, 25), 100.0)
    output = solver.solve({}, init_action=warm)
    assert (warm == 100).all()  # Caller-owned warm start was not overwritten.
    raw = output["actions"].reshape(-1, 5) * torch.tensor(SCALE) + torch.tensor(MEAN)
    assert raw.min() >= -1 and raw.max() <= 1
    assert model.rerolled and len(model.seen) == 2


def test_subset_preserves_exact_requested_rows_and_original_selection():
    from tdwm.evaluation.effplan import diagnostic_pair_subset

    selection = dict(episodes=50, pairs=dict(episode_indices=list(range(8000, 8050)),
                                           start_steps=list(range(50)),
                                           goal_steps=list(range(100, 150))))
    subset = diagnostic_pair_subset(selection, (49, 50))
    assert subset["pairs"]["episode_indices"] == [8048, 8049]
    assert subset["pairs"]["start_steps"] == [48, 49]
    assert subset["episodes"] == 2 and selection["episodes"] == 50
    assert diagnostic_pair_subset(selection, None) is selection
    for bad in [(), (0,), (51,), (49, 49), (True,)]:
        with pytest.raises(ValueError):
            diagnostic_pair_subset(selection, bad)
