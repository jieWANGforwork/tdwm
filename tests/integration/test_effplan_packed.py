import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_effplan_variable_training import trainer, batch_with_lengths, draw
from tdwm.methods.effplan_packed import PackedPaths, packed_generate, packed_refine
from tdwm.methods.effplan_variable import generate_variable_state_path
from tdwm.methods.effplan import refine_state_path
from tdwm.methods.effplan_safety import PlannerSafety
from tdwm.training.effplan_packed_runtime import PackedEffPlanTrainer
from tdwm.training.effplan_variable_data import VariablePlannerBatch


def fast_like(old):
    return PackedEffPlanTrainer(planner=copy.deepcopy(old.planner), eff=old.eff,
        settings=old.settings, identity=old.identity, device="cpu", tracking_model=old.tracking_model)


@pytest.mark.parametrize("phase", ["generation", "refinement"])
@pytest.mark.parametrize("safe", [False, True])
def test_equal_losses_gradients_and_parameter_updates(phase, safe):
    settings = {} if phase=="generation" else dict(calibration_interval=10, search_iterations=(1,1,1,1))
    if safe:
        settings["safety"] = PlannerSafety(state_gradient_max_norm=10, state_delta_max_norm=5)
    old, _ = trainer(phase, **settings)
    torch.nn.init.normal_(old.planner.network[-1].weight, std=0.001)
    fast = fast_like(old)
    batch = batch_with_lengths(3, 6, 10, 10, 21, 41)
    before = copy.deepcopy(old.eff.state_dict())
    a, _ = old._aggregate(batch, validation=False)
    b, _ = fast._aggregate(batch, validation=False)
    for key in ("total_loss", "trajectory_loss", "efficiency_loss", "dynamics_loss"):
        assert a[key] == pytest.approx(b[key], rel=1e-4, abs=1e-5)
    for p, q in zip(old.planner.parameters(), fast.planner.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=2e-5)
    a, b = old.step(batch), fast.step(batch)
    for p, q in zip(old.planner.parameters(), fast.planner.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-3, atol=2e-5)
    assert a["global_step"] == b["global_step"] == 1
    assert all(p.grad is None for p in fast.eff.parameters())
    assert all(torch.equal(v, before[k]) for k,v in old.eff.state_dict().items())


def test_packed_intermediates_equal_unpacked_generation_and_refinement():
    old, _ = trainer("refinement", calibration_interval=10)
    batch = batch_with_lengths(3, 6, 10, 21, 41)
    layout = PackedPaths.from_samples(batch.samples, "cpu")
    nodes = packed_generate(old.planner, layout, old.value, epsilon=1e-6, safety=None)
    improved = packed_refine(old.planner, nodes, layout, old.value, epsilon=1e-6, safety=None)
    offset = 0
    for sample in batch.samples:
        states = sample.real_states
        single = generate_variable_state_path(old.planner, states[:,0], states[:,-1],
            old.value, horizon=states.shape[1]-1, epsilon=1e-6)
        torch.testing.assert_close(nodes[offset:offset+states.shape[1]], single[0])
        refined = refine_state_path(old.planner, single, old.value, predicted_future=None,
                                    epsilon=1e-6, dynamics_coefficient=0)
        torch.testing.assert_close(improved[offset:offset+states.shape[1]], refined[0])
        offset += states.shape[1]


def test_calibration_paths_draws_and_budget_remain_identical():
    old, world = trainer("refinement", calibration_interval=10, calibration_batch_size=2)
    fast = fast_like(old)
    old.global_step = fast.global_step = 9
    batch = batch_with_lengths(3, 6, 10)
    a = old.step(batch)
    old_calls = world.horizons.copy()
    world.horizons.clear()
    b = fast.step(batch)
    assert a["total_loss"] == pytest.approx(b["total_loss"])
    assert world.horizons == old_calls
    assert a["cem_candidate_rollouts"] == b["cem_candidate_rollouts"]
    assert torch.equal(old.solver.torch_gen.get_state(), fast.solver.torch_gen.get_state())


def test_resume_old_checkpoint_retains_optimizer_step_and_sampling_rng(tmp_path):
    old, _ = trainer()
    old.step(batch_with_lengths(6,10))
    old.save(tmp_path/"old.pt", epoch=0)
    fast = fast_like(old)
    fast.resume(tmp_path/"old.pt")
    assert fast.global_step == old.global_step == 1
    a, b = draw(old.rng), draw(fast.rng)
    for p, q in zip(a.samples, b.samples):
        assert torch.equal(p.rows, q.rows)
    for key, value in old.optimizer.state_dict()["state"].items():
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(value[field], fast.optimizer.state_dict()["state"][key][field])
    fast.step(a)
    fast.save(tmp_path/"fast.pt", epoch=0)
    old.resume(tmp_path/"fast.pt")
    assert old.global_step == fast.global_step == 2


def test_boundary_only_batch_does_not_invent_gradients():
    source = draw(np.random.default_rng(4), size=256)
    sample = next(s for s in source.samples if s.rows.shape[1] == 2)
    old, _ = trainer()
    fast = fast_like(old)
    before = copy.deepcopy(fast.planner.state_dict())
    metrics = fast.step(VariablePlannerBatch((sample,)))
    assert metrics["total_loss"] == 0
    assert all(torch.equal(v, before[k]) for k,v in fast.planner.state_dict().items())


def test_two_state_samples_retain_equal_path_weight():
    source = draw(np.random.default_rng(4), size=256)
    sample = next(s for s in source.samples if s.rows.shape[1] == 2)
    other = batch_with_lengths(3,10)
    batch = VariablePlannerBatch((sample, *other.samples))
    old, _ = trainer()
    fast = fast_like(old)
    assert fast.validate(batch)["total_loss"] == pytest.approx(old.validate(batch)["total_loss"], rel=1e-5)
