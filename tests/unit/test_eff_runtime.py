from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.methods.eff import EffModel, efficiency_weights
from tdwm.training.eff_data import EffEpisodeReplay
from tdwm.training.eff_runtime import EffTrainer, load_eff_model, loss_inputs


def replay():
    rng = np.random.default_rng(12)
    states = rng.standard_normal((42, 192)).astype(np.float32)
    store = SimpleNamespace(latents=states, episode_ids=np.repeat([0, 1], 21))
    return EffEpisodeReplay(
        store, episodes=(0, 1), stride=5, terminal_at_state=np.zeros(42, bool)
    )


def batch(rng=None):
    return replay().sample(
        batch_size=8,
        rng=np.random.default_rng(3) if rng is None else rng,
        backup_primitive_steps=10,
        cross_episode_probability=0.3,
        epsilon=1e-6,
    )


def trainer(**overrides):
    arguments = dict(
        identity={"source_sha256": "test-source", "training_episodes": [0, 1]},
        seed=3072,
        learning_rate=1e-3,
        weight_decay=0.001,
        gamma_g=0.95,
        beta=5,
        critic_coefficient=1,
        ema_rate=0.005,
        gradient_clip=1,
        include_goal_boundary=True,
        device="cpu",
    )
    arguments.update(overrides)
    return EffTrainer(EffModel(g_hidden_dim=16, v_hidden_dim=16), **arguments)


def test_boundary_supervision_is_same_path_and_does_not_zero_g():
    b = batch()
    inputs = loss_inputs(
        b,
        device=torch.device("cpu"),
        beta=5,
        gamma_g=0.95,
        critic_coefficient=1,
        include_goal_boundary=True,
    )
    n = len(b.state)
    torch.testing.assert_close(inputs["state"][n:], b.goal)
    assert inputs["goal_reached"][n:].all()
    assert not inputs["vector_valid"][n:].any()
    assert inputs["observed_cost"][n:].eq(0).all()
    assert torch.equal(inputs["path_ids"][:n], inputs["path_ids"][n:])
    assert len(inputs["path_weights"]) == n


def test_optimizer_only_owns_online_g_and_v():
    t = trainer()
    actual = {id(p) for group in t.optimizer.param_groups for p in group["params"]}
    assert actual == {id(p) for p in t.model.online_parameters()}
    assert not actual.intersection(id(p) for p in t.model.target_g.parameters())
    result = t.step(batch())
    assert result["global_step"] == 1
    assert all(p.grad is None for p in t.model.target_g.parameters())
    assert all(p.grad is None for p in t.model.target_v.parameters())


def test_checkpoint_resume_reproduces_next_sample_and_update(tmp_path):
    torch.manual_seed(4)
    original = trainer()
    original.step(batch(original.rng))
    path = tmp_path / "checkpoint.pt"
    checksum = original.save(path, epoch=1)
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()
    next_original = batch(original.rng)
    expected_metrics = original.step(next_original)
    expected_state = copy.deepcopy(original.model.state_dict())
    resumed = trainer()
    assert resumed.resume(path) == 1
    next_resumed = batch(resumed.rng)
    torch.testing.assert_close(next_resumed.state, next_original.state, rtol=0, atol=0)
    assert resumed.step(next_resumed) == expected_metrics
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)


def test_validation_preserves_rng_parameters_and_mode():
    t = trainer()
    t.model.eval()
    state = copy.deepcopy(t.model.state_dict())
    numpy_state = copy.deepcopy(t.rng.bit_generator.state)
    torch_state = torch.get_rng_state().clone()
    metrics = t.validate(batch())
    assert "unweighted_v_movement_loss" in metrics
    assert not t.model.training and not t.model.target_g.training
    assert t.rng.bit_generator.state == numpy_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    for name, value in t.model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_resume_identity_mismatch_rejected_without_mutation(tmp_path):
    t = trainer()
    path = tmp_path / "eff.pt"
    t.save(path, epoch=0)
    changed = trainer(identity={"training_episodes": [1, 2]})
    before = copy.deepcopy(changed.model.state_dict())
    with pytest.raises(ValueError, match="identity"):
        changed.resume(path)
    for name, value in changed.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_eval_restore_freezes_parameters_but_preserves_input_gradients(tmp_path):
    t = trainer()
    t.step(batch())
    path = tmp_path / "eff.pt"
    t.save(path, epoch=1)
    model, payload = load_eff_model(
        path, expected_identity=t.identity, expected_global_step=1, device="cpu"
    )
    z = torch.randn(2, 192, requires_grad=True)
    goal = torch.randn(2, 192, requires_grad=True)
    model.value(z, goal, target=True).sum().backward()
    assert z.grad is not None and goal.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="updates"):
        load_eff_model(
            path, expected_identity=t.identity, expected_global_step=2, device="cpu"
        )
    assert payload["global_step"] == 1


@pytest.mark.parametrize(
    "settings",
    [{"ema_rate": 0}, {"beta": -1}, {"gamma_g": 2}, {"critic_coefficient": -1}],
)
def test_invalid_loss_config_fails_before_any_optimizer_update(settings):
    with pytest.raises(ValueError):
        trainer(**settings)


def test_efficiency_weights_stay_positive_at_extreme_beta():
    weights = efficiency_weights(
        torch.tensor([0.0, 0.2, 1.0]),
        torch.ones(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
        beta=1e200,
    )
    assert torch.isfinite(weights).all() and (weights > 0).all()
    torch.testing.assert_close(weights.sum(), torch.tensor(3.0))
