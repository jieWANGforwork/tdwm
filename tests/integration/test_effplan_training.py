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


def trainer(phase="generation", supervision="final", planner=None, safety=None,
            v_parameterization="total_work", calibration_interval=1,
            calibration_batch_size=None):
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
        safety=safety,
        calibration_interval=calibration_interval,
        calibration_batch_size=calibration_batch_size,
    )
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8,
                   v_parameterization=v_parameterization).eval().requires_grad_(False)
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


def test_sparse_updates_skip_f_and_calibrate_only_subset():
    from tdwm.methods.effplan_safety import PlannerSafety
    t, world = trainer("refinement", safety=PlannerSafety(10, 5),
                       calibration_interval=3, calibration_batch_size=1)
    before = copy.deepcopy(t.planner.state_dict())
    for _ in range(2):
        result = t.step(batch())
        assert not result["calibration_update"]
        assert result["cem_candidate_rollouts"] == 0
        assert result["dynamics_loss"] == 0
        assert result["trajectory_labelled_paths"] == 2
    assert not world.seen_samples
    assert any(not torch.equal(v, before[k]) for k,v in t.planner.state_dict().items())
    result = t.step(batch())
    assert result["calibration_update"] and result["cem_candidate_rollouts"] == 8
    assert result["trajectory_labelled_paths"] == 1
    assert result["dynamics_loss"] > 0
    assert all(p.grad is None for p in t.eff.parameters())
    assert all(p.grad is None for p in world.parameters())


def test_sparse_branch_preserves_optimizer_rng_and_step_then_resumes(tmp_path):
    from tdwm.training.effplan_runtime import planner_settings_payload
    from tdwm.training.eff_runtime import canonical_sha256
    dense, _ = trainer("refinement")
    dense.identity = {"source": "fixed", "settings_sha256": canonical_sha256(planner_settings_payload(dense.settings))}
    dense.step(batch())
    parent = tmp_path / "dense.pt"
    dense.save(parent, epoch=0)
    sparse, _ = trainer("refinement", calibration_interval=3, calibration_batch_size=1)
    sparse.identity = {"source": "fixed", "settings_sha256": canonical_sha256(planner_settings_payload(sparse.settings))}
    sparse.eff.load_state_dict(dense.eff.state_dict())
    with pytest.raises(ValueError, match="identity"):
        sparse.resume(parent)
    sparse.resume(parent, schedule_branch=True)
    assert sparse.global_step == 1
    assert sparse.schedule_transition["switch_after_update"] == 1
    assert sparse.rng.bit_generator.state == dense.rng.bit_generator.state
    assert torch.equal(sparse.solver.torch_gen.get_state(), dense.solver.torch_gen.get_state())
    for p, q in zip(sparse.planner.parameters(), dense.planner.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(sparse.optimizer.state[p]["exp_avg"], dense.optimizer.state[q]["exp_avg"], rtol=0, atol=0)
    assert not sparse.step(batch())["calibration_update"]
    child = tmp_path / "sparse.pt"
    sparse.save(child, epoch=0)
    expected = sparse.step(batch())
    restored, _ = trainer("refinement", calibration_interval=3, calibration_batch_size=1)
    restored.identity = sparse.identity
    restored.eff.load_state_dict(sparse.eff.state_dict())
    restored.resume(child)
    assert restored.step(batch()) == expected
    assert restored.schedule_transition == sparse.schedule_transition
    # An unrelated loss change cannot be smuggled into a schedule handoff.
    changed, _ = trainer("refinement", calibration_interval=3)
    changed.settings = replace(changed.settings, efficiency_coefficient=.7)
    changed.identity = {"source": "fixed", "settings_sha256": canonical_sha256(planner_settings_payload(changed.settings))}
    with pytest.raises(ValueError, match="identity"):
        changed.resume(parent, schedule_branch=True)


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


@pytest.mark.parametrize("mode", ["total_work", "extra_work"])
def test_stable_refinement_25_updates_with_public_cem_resume_and_frozen_gvf(tmp_path, mode):
    from tdwm.methods.effplan_safety import PlannerSafety

    torch.manual_seed(3072)
    safe = PlannerSafety(10, 5)
    t, world = trainer("refinement", safety=safe, v_parameterization=mode)
    # Reproduce the hazardous readout without changing P's loss definition.
    with torch.no_grad():
        t.eff.target_v.network[-1].weight.zero_()
        t.eff.target_v.network[-1].bias.fill_(-200)
    eff_before = copy.deepcopy(t.eff.state_dict())
    p_before = copy.deepcopy(t.planner.state_dict())
    for _ in range(25):
        result = t.step(batch())
        assert result["safety/protected_efficiency/max"] <= 1.000001
        assert result["safety/candidate_gradient_capped_norm/max"] <= 10.00001
        assert result["safety/state_delta_capped_norm/max"] <= 5.00001
        assert np.isfinite(result["gradient_norm"])
    assert any(
        not torch.equal(v, p_before[k]) for k, v in t.planner.state_dict().items()
    )
    assert all(torch.equal(v, eff_before[k]) for k, v in t.eff.state_dict().items())
    assert all(p.grad is None for p in world.parameters())
    assert all(p.grad is None for p in t.eff.parameters())
    path = tmp_path / "stable.pt"
    t.save(path, epoch=0)
    expected = t.step(batch())
    restored, _ = trainer("refinement", safety=safe, v_parameterization=mode)
    restored.eff.load_state_dict(t.eff.state_dict())
    restored.resume(path)
    assert restored.step(batch()) == expected
    _, payload = load_effplan_planner(
        path,
        expected_identity=t.identity,
        expected_global_step=25,
        expected_phase="refinement",
        device="cpu",
    )
    assert payload["settings"]["safety"]["state_delta_max_norm"] == 5
    legacy, _ = trainer("refinement")
    with pytest.raises(ValueError, match="settings"):
        legacy.resume(path)


@pytest.mark.parametrize("phase", ["generation", "refinement"])
def test_safety_run_pause_resume_keeps_full_schedule_and_completes(
    tmp_path, monkeypatch, phase
):
    import json
    from pathlib import Path
    import yaml
    from tdwm.training import effplan_run as run
    from tdwm.training.eff_run import EffRunSettings, eff_settings_payload
    from tdwm.training.eff_runtime import canonical_sha256

    config = yaml.safe_load(
        Path("configs/experiment/effplan_cube_stable_p_v1.yaml").read_text()
    )
    config[f"planner_{phase}"]["run"].update(
        epochs=1,
        updates_per_epoch=4,
        planner_hidden_dim=8,
        batch_size=2,
        validation_batches=1,
        checkpoint_every_updates=1000,
    )
    if phase == "refinement":
        config["planner_refinement"]["settings"].update(
            search_iterations=[1, 1], cem_candidates=4, cem_elites=2, cem_batch_size=2,
        )
    conf = tmp_path / "protocol.yaml"
    conf.write_text(yaml.safe_dump(config))
    eff_settings = EffRunSettings(**config["eff_training"]["settings"])
    meta = tmp_path / "eff.json"
    meta.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_updates": eff_settings.total_updates,
                "identity": {
                    "settings_sha256": canonical_sha256(
                        eff_settings_payload(eff_settings)
                    )
                },
            }
        )
    )
    eff_file = tmp_path / "eff.pt"
    eff_file.write_bytes(b"test fixture only")
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).eval().requires_grad_(False)
    monkeypatch.setattr(
        run, "load_eff_replays", lambda **_: (replay(), replay(), {"fixture": True})
    )
    monkeypatch.setattr(run, "load_eff_model", lambda *_, **__: (eff, {}))
    monkeypatch.setattr(
        run, "load_frozen_world_model", lambda *_, **__: (TinyFrozenWorld(), None)
    )
    args = dict(
        config_path=conf,
        phase=phase,
        latent_store="fixture",
        terminal_metadata="fixture",
        eff_checkpoint=eff_file,
        eff_manifest=meta,
        lewm_checkpoint="fixture",
        device="cpu",
    )
    paused = run.run_effplan_training(
        **args, output_dir=tmp_path / "paused", stop_after_updates=2
    )
    assert paused["status"] == "paused" and paused["completed_updates"] == 2
    assert not (tmp_path / "paused/planner_manifest.json").exists()
    done = run.run_effplan_training(
        **args, output_dir=tmp_path / "paused", resume=tmp_path / "paused/last.pt"
    )
    assert done["status"] == "complete" and done["completed_updates"] == 4
    torch.manual_seed(98765)  # An unrelated prior RNG state must not change P.
    direct = run.run_effplan_training(**args, output_dir=tmp_path / "direct")
    a = torch.load(done["last_recoverable_checkpoint"], weights_only=False)
    b = torch.load(direct["last_recoverable_checkpoint"], weights_only=False)
    for key, value in a["planner"].items():
        torch.testing.assert_close(value, b["planner"][key], rtol=0, atol=0)
    if phase == "refinement":
        assert a["optimizer"]["param_groups"][0]["lr"] == 0
        assert a["settings"]["safety"]["geometric_lower_bound"]
