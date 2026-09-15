"""Variable P is additive; exercise both phases through public SWM CEM."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.adapters.effplan_adaptive import AdaptiveTrackingCost
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import StatePlanner, generate_state_path
from tdwm.training.eff_data import EffEpisodeReplay, sample_planner_paths
from tdwm.training.effplan_runtime import EffPlanTrainSettings, load_effplan_planner
from tdwm.training.effplan_variable_data import VariablePlannerBatch, sample_variable_planner_paths
from tdwm.training.effplan_variable_runtime import VariableEffPlanTrainer
from tdwm.training.effplan_variable_run import SAMPLING


class TinyWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.horizons = []

    def rollout(self, info, actions, history_size):
        self.horizons.append(actions.shape[-2])
        assert history_size == 3
        future = info["emb"][..., -1:, :] + torch.nn.functional.pad(actions, (0, 167)).cumsum(2)
        return dict(info, predicted_emb=torch.cat((info["emb"], future), dim=2))


def replay():
    rng = np.random.default_rng(19)
    store = SimpleNamespace(episode_ids=np.repeat([0, 1], 201),
                            latents=rng.normal(size=(402, 192)).astype(np.float32))
    return EffEpisodeReplay(store, episodes=(0, 1), stride=5,
                            terminal_at_state=np.zeros(402, bool))


def draw(rng, size=16):
    return sample_variable_planner_paths(replay(), batch_size=size, rng=rng,
                                         backup_primitive_steps=50, epsilon=1e-6)


def batch_with_lengths(*lengths):
    return VariablePlannerBatch(tuple(sample_planner_paths(
        replay(), batch_size=1, horizon=length-1, rng=np.random.default_rng(length),
        cross_episode_probability=0,
    ) for length in lengths))


def trainer(phase="generation", **overrides):
    settings = EffPlanTrainSettings(
        phase=phase, seed=3072, learning_rate=1e-3, weight_decay=0.001,
        gradient_clip=1, epsilon=1e-6, target_readout=True,
        trajectory_coefficient=1, efficiency_coefficient=0 if phase=="generation" else 0.01,
        dynamics_coefficient=0 if phase=="generation" else 0.1,
        search_iterations=() if phase=="generation" else (1,),
        cem_candidates=4, cem_elites=2, cem_batch_size=2, supervision="final",
    )
    settings = replace(settings, **overrides)
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).eval().requires_grad_(False)
    world = TinyWorld()
    tracking = None if phase=="generation" else AdaptiveTrackingCost(world, eff, target=True)
    t = VariableEffPlanTrainer(planner=StatePlanner(hidden_dim=8), eff=eff,
        settings=settings, identity={"planner_sampling": SAMPLING}, device="cpu",
        tracking_model=tracking)
    return t, world


def test_exact_same_draws_as_v_and_all_intermediate_rows_present():
    rng = np.random.default_rng(3072)
    ref = replay().sample(batch_size=512, rng=rng, backup_primitive_steps=50,
                          cross_episode_probability=0, epsilon=1e-6)
    batch = draw(np.random.default_rng(3072), 512)
    for i, sample in enumerate(batch.samples):
        assert sample.rows[0, 0] == ref.anchor_rows[i]
        assert sample.rows[0, -1] == ref.goal_rows[i]
        assert sample.anchor_episodes[0] == ref.anchor_episodes[i]
        assert sample.rows.shape[1] == ref.offset_chunks[i] + 1
        assert torch.diff(sample.rows).eq(5).all()
    lengths = {s.rows.shape[1] for s in batch.samples}
    assert 2 in lengths and 41 in lengths and len(lengths) == 40
    assert sum(len(g.real_states) for g in batch.groups()) == 512


def test_ten_states_mean_nine_actions_eight_generated_nodes():
    t, world = trainer("refinement")
    batch = batch_with_lengths(10)
    path = generate_state_path(t.planner, batch.samples[0].real_states[:, 0],
        batch.samples[0].real_states[:, -1], t.value, horizon=9, epsilon=1e-6)
    assert path.shape == (1, 10, 192) and path[:, 1:-1].shape[1] == 8
    result = t.step(batch)
    assert result["sampled_states_mean"] == 10
    assert world.horizons and set(world.horizons) == {9}


@pytest.mark.parametrize("phase", ["generation", "refinement"])
def test_ragged_batch_one_optimizer_update_only_p_changes(phase):
    t, world = trainer(phase)
    previous = copy.deepcopy(t.planner.state_dict())
    eff = copy.deepcopy(t.eff.state_dict())
    result = t.step(batch_with_lengths(3, 6, 10))
    assert result["global_step"] == 1 and result["optimized_paths"] == 3
    assert any(not torch.equal(v, previous[k]) for k, v in t.planner.state_dict().items())
    assert all(torch.equal(v, eff[k]) for k, v in t.eff.state_dict().items())
    assert all(p.grad is None for p in t.eff.parameters())
    assert world.fixed.grad is None
    if phase == "refinement":
        assert set(world.horizons) == {2, 5, 9}
    else:
        assert world.horizons == []


def test_two_state_boundary_has_no_fake_node_or_nan_or_weight_decay():
    batch = draw(np.random.default_rng(1), 256)
    sample = next(s for s in batch.samples if s.rows.shape[1] == 2)
    t, _ = trainer()
    initial = copy.deepcopy(t.planner.state_dict())
    metrics = t.step(VariablePlannerBatch((sample,)))
    assert metrics["total_loss"] == 0 and metrics["no_interior_paths"] == 1
    assert metrics["optimized_paths"] == 0
    assert all(torch.equal(v, initial[k]) for k, v in t.planner.state_dict().items())


def test_bucket_weighting_is_per_path_not_per_length_bucket():
    t, _ = trainer()
    batch = batch_with_lengths(3, 6, 6, 10)
    expected = sum(t.validate(VariablePlannerBatch((s,)))["total_loss"] for s in batch.samples)/4
    assert t.validate(batch)["total_loss"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("phase", ["generation", "refinement"])
def test_resume_rng_weights_optimizer_and_old_evaluation_loader(tmp_path, phase):
    t, _ = trainer(phase)
    b = batch_with_lengths(3, 6)
    t.step(b)
    t.save(tmp_path/"p.pt", epoch=0)
    restored, _ = trainer(phase)
    restored.eff.load_state_dict(t.eff.state_dict())
    restored.resume(tmp_path/"p.pt")
    assert np.array_equal(t.rng.integers(1000, size=8), restored.rng.integers(1000, size=8))
    before_rng = None if t.solver is None else t.solver.torch_gen.get_state().clone()
    t.validate(b)
    if before_rng is not None:
        assert torch.equal(before_rng, t.solver.torch_gen.get_state())
    assert t.step(b)["total_loss"] == pytest.approx(restored.step(b)["total_loss"], rel=1e-6)
    for k, value in t.planner.state_dict().items():
        torch.testing.assert_close(value, restored.planner.state_dict()[k], rtol=0, atol=0)
    model, _ = load_effplan_planner(tmp_path/"p.pt", expected_identity=t.identity,
        expected_global_step=1, expected_phase=phase, device="cpu")
    assert isinstance(model, StatePlanner)


def test_sparse_calibration_uses_only_prefix_of_iid_paths():
    t, world = trainer("refinement", calibration_interval=2, calibration_batch_size=2)
    batch = batch_with_lengths(3, 6, 10)
    first = t.step(batch)
    assert not first["calibration_update"] and world.horizons == []
    second = t.step(batch)
    assert second["selected_paths"] == 2 and set(world.horizons) == {2, 5}


def test_terminal_crossing_is_not_a_midpoint_label():
    r = replay()
    r.terminal[:] = True
    r.terminal_prefix = np.concatenate(([0], np.cumsum(r.terminal)))
    with pytest.raises(ValueError, match="terminal"):
        sample_variable_planner_paths(r, batch_size=1, rng=np.random.default_rng(1),
                                     backup_primitive_steps=50, epsilon=1e-6)


def test_variable_entry_refuses_old_fixed_configuration_before_loading_data(tmp_path):
    from tdwm.training.effplan_variable_run import run_variable_effplan_training
    with pytest.raises(ValueError, match="sampling configuration"):
        run_variable_effplan_training(
            config_path="configs/experiment/effplan_cube_same_episode_extra_work_v1.yaml",
            phase="generation", latent_store=tmp_path, terminal_metadata=tmp_path,
            eff_checkpoint=tmp_path, eff_manifest=tmp_path, lewm_checkpoint=None,
            output_dir=tmp_path/"out", device="cpu",
        )


def test_length_41_retains_39_interior_states_and_backpropagates():
    t, _ = trainer()
    metrics = t.step(batch_with_lengths(41))
    assert metrics["sampled_states_max"] == 41
    assert metrics["gradient_norm"] > 0 and np.isfinite(metrics["total_loss"])


@pytest.mark.parametrize("horizon", [1, 2, 5, 9, 20, 40])
def test_batched_tree_matches_original_values_and_p_gradients(horizon):
    from tdwm.methods.effplan_variable import generate_variable_state_path
    t, _ = trainer()
    torch.nn.init.normal_(t.planner.network[-1].weight, std=0.001)
    start, goal = torch.randn(2, 192), torch.randn(2, 192)
    old = generate_state_path(t.planner, start, goal, t.value, horizon=horizon, epsilon=1e-6)
    new = generate_variable_state_path(t.planner, start, goal, t.value, horizon=horizon, epsilon=1e-6)
    torch.testing.assert_close(old, new, atol=2e-5, rtol=2e-5)
    if horizon > 1:
        a = torch.autograd.grad(old.square().mean(), tuple(t.planner.parameters()))
        b = torch.autograd.grad(new.square().mean(), tuple(t.planner.parameters()))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)
