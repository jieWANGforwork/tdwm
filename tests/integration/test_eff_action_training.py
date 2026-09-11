"""Synthetic CPU integration checks for EffAction training and exact recovery."""

import copy
import importlib
import random

import numpy as np
import pytest
import torch
from torch import nn

from tdwm.training.eff_action_data import EffActionEpisodeData, EffActionSamplingConfig
from tdwm.training.eff_action_runtime import (
    EffActionTrainer,
    eff_action_learning_rate_factor,
    validate_eff_action_training_config,
)


def test_eff_action_training_import_smoke():
    assert importlib.import_module("tdwm.training.eff_action_runtime")


PROVENANCE = {
    "encoder_sha256": "a" * 64,
    "latent_manifest": "b" * 64,
    "dataset": "synthetic",
}


class ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(901)
            self.linear = nn.Linear(25, 192)

    def forward(self, action):
        return self.linear(action)


def _config():
    optimizer = {"lr": 0.0003, "weight_decay": 0.01, "betas": [0.9, 0.95], "eps": 1e-8}
    stage = {
        "steps": 3,
        "warmup_steps": 1,
        "min_lr_ratio": 0.1,
        "gradient_clip_norm": 2.0,
    }
    return {
        "seed": 67,
        "precision": "float32",
        "g": {"hidden_dim": 8, "hidden_layers": 1, "embedding_layers": 2},
        "v": {"hidden_dim": 8, "hidden_layers": 1, "output_activation": "softplus"},
        "planner": {
            "raw_action_dim": 25,
            "hidden_dim": 12,
            "hidden_layers": 2,
            "iterations": 2,
            "epsilon": 0.1,
            "lower_bound": -1.0,
            "upper_bound": 1.0,
            "initialization": {"distribution": "normal", "mean": 0.0, "std": 0.1},
            "lambda_traj": 1.0,
            "lambda_eff": 0.3,
        },
        "gv": {
            **stage,
            "g_optimizer": copy.deepcopy(optimizer),
            "v_optimizer": copy.deepcopy(optimizer),
            "ema_decay": 0.9,
        },
        "stage1": {**stage, "optimizer": copy.deepcopy(optimizer)},
        "stage2": {
            **stage,
            "optimizer": copy.deepcopy(optimizer),
            "optimizer_transition": "carry",
        },
    }


def _data():
    rng = np.random.default_rng(913)
    latents = rng.normal(size=(40, 192)).astype(np.float32)
    actions = (0.1 * rng.normal(size=(40, 25))).astype(np.float32)
    ids = np.repeat(np.arange(2), 20)
    config = EffActionSamplingConfig(
        episode_start=0,
        episode_stop=2,
        min_goal_chunks=1,
        max_goal_chunks=3,
        grid_phase=0,
        efficiency_threshold=0.8,
        efficiency_epsilon=0.001,
        zero_distance_tolerance=1e-6,
        direct_max_chunks=None,
    )
    return EffActionEpisodeData(latents, actions, ids, config=config)


def _step(trainer, data):
    return trainer.train_step(data.sample(4, rng=trainer.sample_rng))


def _reach_stage(trainer, data, stage):
    for next_stage in ("stage1", "stage2"):
        if trainer.stage == stage:
            return
        while not trainer.is_stage_complete:
            _step(trainer, data)
        trainer.begin_planner_stage(int(next_stage[-1]))


def _equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _equal(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("stage", ["gv", "stage1", "stage2"])
def test_resume_reproduces_sampling_rng_and_exact_next_optimizer_update(
    tmp_path, stage
):
    config, data = _config(), _data()
    trainer = EffActionTrainer(ActionEncoder(), config, PROVENANCE, "cpu")
    _reach_stage(trainer, data, stage)
    _step(trainer, data)
    path = trainer.save_checkpoint(tmp_path / f"{stage}.pt")
    expected_rng = (random.random(), float(np.random.random()), torch.rand(4))
    expected_batch = data.sample(4, rng=trainer.sample_rng)
    expected_metrics = trainer.train_step(expected_batch)
    expected_state = copy.deepcopy(trainer.checkpoint_state())
    restored = EffActionTrainer.load_checkpoint(
        path,
        action_encoder=ActionEncoder(),
        config=config,
        provenance=PROVENANCE,
        device="cpu",
    )
    actual_rng = (random.random(), float(np.random.random()), torch.rand(4))
    _equal(expected_rng, actual_rng)
    actual_batch = data.sample(4, rng=restored.sample_rng)
    _equal(expected_batch, actual_batch)
    actual_metrics = restored.train_step(actual_batch)
    _equal(expected_metrics, actual_metrics)
    _equal(expected_state, restored.checkpoint_state())


@pytest.mark.parametrize("stage", ["gv", "stage1", "stage2"])
def test_validation_preserves_rng_parameters_optimizer_gradients_and_modes(stage):
    trainer, data = (
        EffActionTrainer(ActionEncoder(), _config(), PROVENANCE, "cpu"),
        _data(),
    )
    _reach_stage(trainer, data, stage)
    _step(trainer, data)
    modules = [
        trainer.g,
        trainer.v,
        trainer.target_g,
        trainer.target_v,
        trainer.action_encoder,
    ]
    if trainer.planner is not None:
        modules.append(trainer.planner)
    modes = [part.training for module in modules for part in module.modules()]
    gradients = [
        parameter.grad.clone() if parameter.grad is not None else None
        for module in modules
        for parameter in module.parameters()
    ]
    batch = data.sample(4, rng=np.random.default_rng(107))
    before = copy.deepcopy(trainer.checkpoint_state())
    first = trainer.evaluate_batch(batch)
    second = trainer.evaluate_batch(batch)
    assert first == second
    assert 0 <= first["direct_frac"] <= 1 and 0 <= first["terminal_frac"] <= 1
    _equal(before, trainer.checkpoint_state())
    assert modes == [part.training for module in modules for part in module.modules()]
    after_gradients = [
        parameter.grad for module in modules for parameter in module.parameters()
    ]
    _equal(gradients, after_gradients)


def test_planner_stages_freeze_gv_targets_encoder_and_keep_separate_stage1_deployment(
    tmp_path,
):
    config, data = _config(), _data()
    config["planner"]["lambda_traj"] = 0
    trainer = EffActionTrainer(ActionEncoder(), config, PROVENANCE, "cpu")
    assert trainer.planner is None
    initial = trainer.checkpoint_state()
    assert initial["models"]["planner"] is None
    assert not initial["planner_initialized"] and not initial["planner_trained"]
    assert set(initial["models"]) == {"g", "v", "target_g", "target_v", "planner"}
    with pytest.raises(RuntimeError, match="Complete gv"):
        trainer.begin_planner_stage(1)
    _reach_stage(trainer, data, "stage1")
    fixed = copy.deepcopy(
        {
            key: trainer.checkpoint_state()["models"][key]
            for key in ("g", "v", "target_g", "target_v")
        }
    )
    encoder = copy.deepcopy(trainer.action_encoder.state_dict())
    assert not trainer.checkpoint_state()["planner_trained"]
    while not trainer.is_stage_complete:
        _step(trainer, data)
    path = trainer.save_checkpoint(tmp_path / "planner-stage1.pt")
    stage1_bytes = path.read_bytes()
    moments = copy.deepcopy(trainer.planner_optimizer.state_dict()["state"])
    trainer.begin_planner_stage(2)
    _equal(moments, trainer.planner_optimizer.state_dict()["state"])
    before_p = copy.deepcopy(trainer.planner.state_dict())
    metrics = _step(trainer, data)
    assert metrics["gradient_norm_planner"] > 0
    assert any(
        not torch.equal(value, before_p[key])
        for key, value in trainer.planner.state_dict().items()
    )
    assert metrics["loss"] == pytest.approx(
        config["planner"]["lambda_eff"] * metrics["efficiency_loss"]
    )
    for key in fixed:
        _equal(fixed[key], getattr(trainer, key).state_dict())
    _equal(encoder, trainer.action_encoder.state_dict())
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for module in (
            trainer.g,
            trainer.v,
            trainer.target_g,
            trainer.target_v,
            trainer.action_encoder,
        )
        for parameter in module.parameters()
    )
    assert path.read_bytes() == stage1_bytes
    restored_bc = EffActionTrainer.load_checkpoint(
        path,
        action_encoder=ActionEncoder(),
        config=config,
        provenance=PROVENANCE,
        device="cpu",
    )
    assert restored_bc.stage == "stage1" and restored_bc.is_stage_complete
    assert restored_bc.checkpoint_state()["planner_trained"]


def test_strict_config_provenance_and_counter_checks(tmp_path):
    config = _config()
    trainer = EffActionTrainer(ActionEncoder(), config, PROVENANCE, "cpu")
    path = trainer.save_checkpoint(tmp_path / "resume.pt")
    altered = copy.deepcopy(config)
    altered["gv"]["steps"] += 1
    with pytest.raises(ValueError, match="config"):
        EffActionTrainer.load_checkpoint(
            path,
            action_encoder=ActionEncoder(),
            config=altered,
            provenance=PROVENANCE,
            device="cpu",
        )
    with pytest.raises(ValueError, match="provenance"):
        EffActionTrainer.load_checkpoint(
            path,
            action_encoder=ActionEncoder(),
            config=config,
            provenance={**PROVENANCE, "encoder_sha256": "different"},
            device="cpu",
        )
    state = trainer.checkpoint_state()
    state["counters"]["stage1"] = 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="future stage"):
        EffActionTrainer.load_checkpoint(
            path,
            action_encoder=ActionEncoder(),
            config=config,
            provenance=PROVENANCE,
            device="cpu",
        )
    incomplete = copy.deepcopy(config)
    incomplete["stage1"]["steps"] = None
    with pytest.raises(ValueError, match="steps"):
        validate_eff_action_training_config(incomplete)


def test_failed_atomic_save_preserves_previous_checkpoint_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    trainer = EffActionTrainer(ActionEncoder(), _config(), PROVENANCE, "cpu")
    path = trainer.save_checkpoint(tmp_path / "resume.pt")
    previous = path.read_bytes()

    def fail_save(value, handle):
        handle.write(b"partial")
        raise OSError("simulated storage interruption")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="storage interruption"):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == previous
    assert sorted(item.name for item in tmp_path.iterdir()) == ["resume.pt"]


def test_scheduler_and_explicit_optimizer_reset():
    factors = [
        eff_action_learning_rate_factor(
            step, total_steps=5, warmup_steps=2, min_lr_ratio=0.1
        )
        for step in range(5)
    ]
    assert factors == pytest.approx([0.5, 1.0, 1.0, 0.55, 0.1])
    config, data = _config(), _data()
    config["stage2"]["optimizer_transition"] = "reset"
    trainer = EffActionTrainer(ActionEncoder(), config, PROVENANCE, "cpu")
    _reach_stage(trainer, data, "stage1")
    while not trainer.is_stage_complete:
        _step(trainer, data)
    assert trainer.planner_optimizer.state
    before = copy.deepcopy(trainer.planner.state_dict())
    trainer.begin_planner_stage(2)
    assert not trainer.planner_optimizer.state
    _equal(before, trainer.planner.state_dict())


def test_bfloat16_runtime_can_train_validate_and_advance_all_three_stages():
    config, data = _config(), _data()
    config["precision"] = "bfloat16"
    for stage in ("gv", "stage1", "stage2"):
        config[stage]["steps"] = 1
    trainer = EffActionTrainer(ActionEncoder(), config, PROVENANCE, "cpu")
    for index, stage in enumerate(("gv", "stage1", "stage2")):
        if index:
            trainer.begin_planner_stage(index)
        metrics = _step(trainer, data)
        assert metrics["stage"] == stage and np.isfinite(metrics["loss"])
        assert np.isfinite(
            trainer.evaluate_batch(data.sample(4, rng=np.random.default_rng(123)))[
                "loss"
            ]
        )
        assert trainer.is_stage_complete
    with pytest.raises(RuntimeError, match="already complete"):
        _step(trainer, data)
