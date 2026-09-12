from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from torch import nn

from tdwm.adapters.effplan import EffPlanTrackingCost
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import StatePlanner
from tdwm.training.eff_data import EffEpisodeReplay, sample_planner_paths
from tdwm.training.effplan_runtime import (
    EffPlanTrainer,
    EffPlanTrainSettings,
    load_effplan_planner,
)


class TinyFrozenWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.seen_starts = []
        self.seen_samples = []

    def rollout(self, info, actions, history_size):
        assert history_size == 3
        self.seen_starts.append(info["emb"][..., 0, :].clone())
        self.seen_samples.append(actions.shape[1])
        increment = torch.nn.functional.pad(actions, (0, 167))
        future = info["emb"][..., -1:, :] + increment.cumsum(2)
        return dict(info, predicted_emb=torch.cat((info["emb"], future), dim=2))


def replay():
    states = np.random.default_rng(9).normal(size=(62, 192)).astype(np.float32)
    store = SimpleNamespace(latents=states, episode_ids=np.repeat([0, 1], 31))
    return EffEpisodeReplay(
        store, episodes=(0, 1), stride=5, terminal_at_state=np.zeros(62, bool)
    )


def batch(cross=0):
    return sample_planner_paths(
        replay(),
        batch_size=2,
        horizon=5,
        rng=np.random.default_rng(3),
        cross_episode_probability=cross,
    )


def trainer(phase="generation", supervision="final", planner=None):
    cfg = EffPlanTrainSettings(
        phase=phase,
        seed=3072,
        learning_rate=1e-3,
        weight_decay=0.001,
        gradient_clip=1,
        epsilon=1e-6,
        target_readout=True,
        trajectory_coefficient=1,
        efficiency_coefficient=0 if phase == "generation" else 0.01,
        dynamics_coefficient=0 if phase == "generation" else 0.1,
        search_iterations=() if phase == "generation" else (1, 1),
        cem_candidates=4,
        cem_elites=2,
        cem_batch_size=2,
        supervision=supervision,
    )
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).eval().requires_grad_(False)
    world = TinyFrozenWorld()
    model = (
        None if phase == "generation" else EffPlanTrackingCost(world, eff, target=True)
    )
    t = EffPlanTrainer(
        planner=StatePlanner(hidden_dim=8) if planner is None else planner,
        eff=eff,
        settings=cfg,
        identity={"eff_sha": "test-frozen-eff"},
        device="cpu",
        tracking_model=model,
    )
    return t, world


def test_planner_label_time_positions_are_not_resampled_for_the_goal():
    b = batch()
    assert b.real_states.shape == (2, 6, 192)
    assert torch.diff(b.rows, dim=1).eq(5).all()
    assert (b.rows[:, -1] - b.rows[:, 0]).eq(25).all()
    assert b.trajectory_valid.all() and b.state_stride == 5


def test_cross_episode_goals_never_get_fake_midpoint_labels():
    b = batch(cross=1)
    assert not b.trajectory_valid.any()
    assert (b.goal_episodes != b.anchor_episodes).all()
    t, _ = trainer()
    with pytest.raises(ValueError, match="connected midpoint"):
        t.step(b)


def test_generation_updates_only_p_and_never_calls_f_or_cem():
    t, world = trainer()
    original_eff = copy.deepcopy(t.eff.state_dict())
    original_p = copy.deepcopy(t.planner.state_dict())
    result = t.step(batch())
    assert result["global_step"] == 1 and result["cem_candidate_rollouts"] == 0
    assert not world.seen_samples and t.solver is None
    assert result["dynamics_loss"] == 0
    assert any(
        not torch.equal(value, original_p[key])
        for key, value in t.planner.state_dict().items()
    )
    assert all(
        torch.equal(value, original_eff[key])
        for key, value in t.eff.state_dict().items()
    )
    assert all(p.grad is None for p in t.eff.parameters())


@pytest.mark.parametrize("supervision", ["final", "mean_rounds"])
def test_refinement_has_real_swm_cem_and_only_p_gradients(supervision):
    t, world = trainer("refinement", supervision)
    b = batch()
    result = t.step(b)
    assert isinstance(t.solver, swm.solver.CEMSolver)
    assert world.seen_samples == [4, 1, 4, 1]
    assert result["cem_candidate_rollouts"] == 2 * 4 * 2
    for anchor in world.seen_starts:
        torch.testing.assert_close(anchor, b.real_states[:, None, 0].expand_as(anchor))
    owned = {id(p) for group in t.optimizer.param_groups for p in group["params"]}
    assert owned == {id(p) for p in t.planner.parameters()}
    assert all(p.grad is None for p in t.eff.parameters())
    assert all(p.grad is None for p in world.parameters())
    assert result["gradient_norm"] > 0


def test_refinement_can_mask_cross_episode_trajectory_fit():
    t, _ = trainer("refinement")
    result = t.step(batch(cross=1))
    assert result["trajectory_loss"] == 0 and result["trajectory_labelled_paths"] == 0
    assert result["dynamics_loss"] > 0


@pytest.mark.parametrize("phase", ["generation", "refinement"])
def test_planner_resume_restores_cem_and_optimizer_exactly(tmp_path, phase):
    torch.manual_seed(1)
    t, _ = trainer(phase)
    t.step(batch())
    path = tmp_path / "p.pt"
    t.save(path, epoch=1)
    expected = t.step(batch())
    expected_state = copy.deepcopy(t.planner.state_dict())
    restored, _ = trainer(phase)
    # Frozen Eff is a separate immutable artifact, not saved into the P payload.
    restored.eff.load_state_dict(t.eff.state_dict())
    assert restored.resume(path) == 1
    assert restored.step(batch()) == expected
    for key, value in restored.planner.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)


def test_planner_validation_does_not_change_search_rng_or_parameters():
    t, _ = trainer("refinement")
    before = copy.deepcopy(t.planner.state_dict())
    rng = t.solver.torch_gen.get_state().clone()
    result = t.validate(batch())
    assert torch.equal(t.solver.torch_gen.get_state(), rng)
    assert result["total_loss"] > 0 and t.global_step == 0
    assert all(
        torch.equal(value, before[key]) for key, value in t.planner.state_dict().items()
    )


def test_deployment_loader_rejects_wrong_phase_and_updates(tmp_path):
    t, _ = trainer()
    t.step(batch())
    path = tmp_path / "p.pt"
    t.save(path, epoch=1)
    p, _ = load_effplan_planner(
        path,
        expected_identity=t.identity,
        expected_global_step=1,
        expected_phase="generation",
        device="cpu",
    )
    assert all(not x.requires_grad for x in p.parameters())
    with pytest.raises(ValueError, match="phase"):
        load_effplan_planner(
            path,
            expected_identity=t.identity,
            expected_global_step=1,
            expected_phase="refinement",
            device="cpu",
        )


def test_ambiguous_generation_losses_are_rejected():
    t, _ = trainer()
    with pytest.raises(ValueError, match="trajectory loss only"):
        replace(t.settings, efficiency_coefficient=1)
