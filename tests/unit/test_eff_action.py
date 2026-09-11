"""CPU-only checks for the undiscounted EffAction objective."""

import importlib

import pytest
import torch
from torch import nn

from tdwm.methods.eff_action import (
    EffActionSuccessor,
    EffActionValue,
    build_eff_action_loss,
    eff_action_cost,
    ema_update_eff_action,
    encode_eff_action_blocks,
)


def test_eff_action_import_smoke():
    assert importlib.import_module("tdwm.methods.eff_action")
    assert importlib.import_module("tdwm.methods.eff_action_plan")


class ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(25, 192)
        self.seen = []

    def forward(self, action):
        assert action.ndim == 3 and action.shape[-2:] == (1, 25)
        assert torch.isfinite(action).all(), "Unused next actions were read."
        self.seen.append(action.detach().clone())
        return self.linear(action)


class ConstantSuccessor(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(float(constant)))
        self.calls = 0

    def forward(self, state, action_embedding, task):
        self.calls += 1
        assert torch.isfinite(state).all()
        assert torch.isfinite(action_embedding).all()
        assert torch.isfinite(task).all()
        return torch.ones_like(state) * self.constant


class ConstantValue(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(float(constant)))
        self.calls = 0

    def forward(self, psi, task):
        self.calls += 1
        return torch.ones_like(psi[..., 0]) * self.constant


def _heads():
    torch.manual_seed(13)
    g = EffActionSuccessor(hidden_dim=8, hidden_layers=1, embedding_layers=2)
    v = EffActionValue(hidden_dim=8, hidden_layers=1, output_activation="softplus")
    e = ActionEncoder().requires_grad_(False).eval()
    return g, v, g.make_target(), v.make_target(), e


def _batch(shape=(4,)):
    torch.manual_seed(31)
    return dict(
        state=torch.randn(*shape, 192),
        raw_action=torch.randn(*shape, 25),
        task=torch.randn(*shape, 192),
        goal=torch.randn(*shape, 192),
        next_state=torch.randn(*shape, 192),
        next_raw_action=torch.randn(*shape, 25),
        direct_successor_target=torch.randn(*shape, 192),
        direct_branch=torch.zeros(shape, dtype=torch.bool),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        goal_terminal=torch.zeros(shape, dtype=torch.bool),
    )


@pytest.mark.parametrize("shape", [(3,), (2, 3), ()])
def test_heads_and_cost_preserve_arbitrary_batch_axes(shape):
    g, v, _, _, e = _heads()
    batch = _batch(shape)
    encoded = encode_eff_action_blocks(
        e, batch["raw_action"], batch["state"], retain_action_grad=False
    )
    psi = g(batch["state"], encoded, batch["task"])
    assert psi.shape == shape + (192,)
    assert v(psi, batch["task"]).shape == shape
    cost = eff_action_cost(
        g,
        v,
        e,
        state=batch["state"],
        raw_action=batch["raw_action"],
        task=batch["task"],
        goal=batch["goal"],
        epsilon=0.1,
    )
    assert cost.shape == shape
    assert torch.all(cost <= 0)
    result = build_eff_action_loss(g, v, g.make_target(), v.make_target(), e, **batch)
    assert result.g_prediction.shape == shape + (192,)
    assert result.v_prediction.shape == shape
    assert result.g_per_example_loss.shape == shape
    assert result.g_loss.ndim == result.v_loss.ndim == 0


def test_exact_undiscounted_targets_and_vector_norm_loss_with_masked_rows():
    g = ConstantSuccessor(0)
    v = ConstantValue(0)
    target_g = ConstantSuccessor(5).requires_grad_(False).eval()
    target_v = ConstantValue(7).requires_grad_(False).eval()
    e = ActionEncoder().requires_grad_(False).eval()
    batch = _batch()
    batch["state"].zero_()
    batch["goal"].zero_()
    batch["goal"][0, 0] = 3
    batch["next_state"][:] = 2
    batch["next_state"][2] = 4
    batch["direct_successor_target"][:] = 9
    batch["direct_branch"] = torch.tensor([True, False, False, False])
    batch["goal_terminal"] = torch.tensor([False, False, True, False])
    batch["valid_mask"] = torch.tensor([True, True, True, False])
    # Direct branch has no next-state requirement; terminal has no next action.
    batch["next_state"][0] = torch.nan
    batch["next_raw_action"][[0, 2, 3]] = torch.nan
    for name in ("state", "goal", "raw_action", "task", "next_state"):
        batch[name][3] = torch.nan
    result = build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    assert torch.equal(result.g_target[:, 0], torch.tensor([9.0, 7.0, 4.0, 0.0]))
    expected_v = torch.tensor([3.0, 2 * 192**0.5 + 7, 4 * 192**0.5, 0.0])
    assert torch.allclose(result.v_target, expected_v)
    assert result.g_loss.item() == pytest.approx(192 * (9**2 + 7**2 + 4**2) / 3)
    assert result.g_direct_loss.item() == pytest.approx(192 * 9**2 / 3)
    assert result.v_loss.item() == pytest.approx(expected_v.square().sum().item() / 3)
    assert target_g.calls == target_v.calls == 1
    assert e.seen[0].shape[0] == 3
    assert e.seen[1].shape[0] == 1


def test_terminal_only_does_not_require_or_read_any_bootstrap_action():
    g, v, target_g, target_v, e = _heads()
    batch = _batch()
    batch["goal_terminal"].fill_(True)
    batch["next_raw_action"] = None
    batch["direct_successor_target"] = None
    result = build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    assert torch.equal(result.g_target, batch["next_state"])
    assert torch.equal(
        result.v_target,
        torch.linalg.vector_norm(batch["next_state"] - batch["state"], dim=-1),
    )
    assert len(e.seen) == 1


def test_direct_only_does_not_require_any_next_transition():
    g, v, target_g, target_v, e = _heads()
    batch = _batch()
    batch["direct_branch"].fill_(True)
    batch["next_state"] = None
    batch["next_raw_action"] = None
    result = build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    assert torch.equal(result.g_target, batch["direct_successor_target"])
    assert torch.equal(
        result.v_target,
        torch.linalg.vector_norm(batch["goal"] - batch["state"], dim=-1),
    )
    assert result.g_td_loss.item() == result.v_td_loss.item() == 0
    assert len(e.seen) == 1


def test_targets_and_frozen_embeddings_detach_all_input_gradients():
    g, v, target_g, target_v, e = _heads()
    batch = _batch()
    batch["direct_branch"][0] = True
    inputs = [tensor for tensor in batch.values() if tensor.is_floating_point()]
    for tensor in inputs:
        tensor.requires_grad_(True)
    result = build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    result.loss.backward()
    assert not result.g_target.requires_grad
    assert not result.v_target.requires_grad
    assert all(tensor.grad is None for tensor in inputs)
    assert any(parameter.grad is not None for parameter in g.parameters())
    assert any(parameter.grad is not None for parameter in v.parameters())
    assert all(parameter.grad is None for parameter in target_g.parameters())
    assert all(parameter.grad is None for parameter in target_v.parameters())
    assert all(parameter.grad is None for parameter in e.parameters())


def test_v_loss_cannot_update_g_or_the_action_encoder():
    g, v, target_g, target_v, e = _heads()
    result = build_eff_action_loss(g, v, target_g, target_v, e, **_batch())
    result.v_loss.backward()
    assert all(parameter.grad is None for parameter in g.parameters())
    assert all(parameter.grad is None for parameter in e.parameters())
    assert any(parameter.grad is not None for parameter in v.parameters())


def test_bfloat16_autocast_keeps_target_and_loss_arithmetic_in_float32():
    g, v, target_g, target_v, e = _heads()
    batch = _batch()
    batch["direct_branch"][0] = True
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    assert result.g_target.dtype == result.v_target.dtype == torch.float32
    assert result.g_loss.dtype == result.v_loss.dtype == torch.float32
    result.loss.backward()
    assert all(parameter.grad is None for parameter in e.parameters())
    assert any(parameter.grad is not None for parameter in g.parameters())
    assert any(parameter.grad is not None for parameter in v.parameters())


@pytest.mark.parametrize("activation", ["softplus", "relu", "square"])
def test_value_has_only_successor_and_task_inputs_and_nonnegative_outputs(activation):
    v = EffActionValue(hidden_dim=8, hidden_layers=2, output_activation=activation)
    assert v.network[0].in_features == 384
    assert v.network[-1].out_features == 1
    assert torch.all(v(torch.randn(5, 192), torch.randn(5, 192)) >= 0)


def test_action_helper_preserves_action_gradient_only_when_requested():
    e = ActionEncoder().requires_grad_(False).eval()
    action = torch.randn(2, 25, requires_grad=True)
    state = torch.randn(2, 192)
    detached = encode_eff_action_blocks(e, action, state, retain_action_grad=False)
    live = encode_eff_action_blocks(e, action, state, retain_action_grad=True)
    assert torch.equal(live, detached)
    assert not detached.requires_grad
    live.square().sum().backward()
    assert action.grad is not None and action.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in e.parameters())


def test_cost_is_negative_geometric_efficiency_and_zero_at_goal():
    g = ConstantSuccessor(3)
    v = ConstantValue(7)
    e = ActionEncoder().requires_grad_(False).eval()
    batch = _batch()
    cost = eff_action_cost(
        g,
        v,
        e,
        state=batch["state"],
        raw_action=batch["raw_action"],
        task=batch["task"],
        goal=batch["goal"],
        epsilon=0.5,
    )
    expected = -torch.linalg.vector_norm(batch["goal"] - batch["state"], dim=-1) / 7.5
    assert torch.equal(cost, expected)
    at_goal = eff_action_cost(
        g,
        v,
        e,
        state=batch["state"],
        raw_action=batch["raw_action"],
        task=batch["task"],
        goal=batch["state"],
        epsilon=0.5,
    )
    assert torch.equal(at_goal, torch.zeros(4))


def test_mask_shapes_nonfinite_and_unfrozen_targets_fail_closed():
    g, v, target_g, target_v, e = _heads()
    batch = _batch()
    batch["raw_action"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    batch = _batch()
    batch["direct_branch"] = batch["direct_branch"].float()
    with pytest.raises(TypeError, match="boolean"):
        build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    batch = _batch()
    batch["valid_mask"] = torch.ones(4, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    batch = _batch()
    batch["valid_mask"].zero_()
    with pytest.raises(ValueError, match="at least one"):
        build_eff_action_loss(g, v, target_g, target_v, e, **batch)
    target_g.train()
    with pytest.raises(ValueError, match="eval mode"):
        build_eff_action_loss(g, v, target_g, target_v, e, **_batch())
    target_g.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        build_eff_action_loss(g, v, target_g, target_v, e, **_batch())


def test_ema_is_explicit_online_rate_and_core_checkpoint_roundtrip(tmp_path):
    g, v, target_g, target_v, _ = _heads()
    with torch.no_grad():
        for parameter in target_g.parameters():
            parameter.fill_(2)
        for parameter in g.parameters():
            parameter.fill_(10)
    ema_update_eff_action(target_g, g, rate=0.25)
    assert all(torch.equal(p, torch.full_like(p, 4)) for p in target_g.parameters())
    assert not target_g.training and all(
        not p.requires_grad for p in target_g.parameters()
    )
    path = tmp_path / "heads.pt"
    torch.save(
        {
            "g": g.state_dict(),
            "v": v.state_dict(),
            "tg": target_g.state_dict(),
            "tv": target_v.state_dict(),
        },
        path,
    )
    restored = _heads()[:4]
    saved = torch.load(path, weights_only=True)
    for model, key in zip(restored, ("g", "v", "tg", "tv"), strict=True):
        model.load_state_dict(saved[key], strict=True)
        assert all(
            torch.equal(tensor, saved[key][name])
            for name, tensor in model.state_dict().items()
        )
    with pytest.raises(ValueError, match="rate"):
        ema_update_eff_action(target_g, g, rate=float("nan"))
