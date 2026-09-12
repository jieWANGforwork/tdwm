from __future__ import annotations

import inspect

import pytest
import torch

from tdwm.methods.eff import (
    EffCritic,
    EffModel,
    EffSuccessor,
    eff_loss,
    efficiency_weights,
    movement_target,
    successor_target,
    trajectory_efficiency,
    weighted_path_loss,
)


def test_interfaces_and_output_shapes():
    assert list(inspect.signature(EffSuccessor.forward).parameters) == [
        "self",
        "state",
        "task",
    ]
    assert list(inspect.signature(EffCritic.forward).parameters) == [
        "self",
        "successor",
        "task",
    ]
    g, v = EffSuccessor(hidden_dim=16), EffCritic(hidden_dim=16)
    state, task = torch.randn(2, 3, 192), torch.randn(2, 3, 192)
    successor = g(state, task)
    assert successor.shape == state.shape
    assert v(successor, task).shape == (2, 3)
    assert (v(successor, task) >= 0).all()
    assert not any("action" in key for key in g.state_dict())
    with pytest.raises(TypeError):
        g(state, task, torch.randn(2, 25))


def test_successor_target_uses_real_next_once_and_stops_terminal_tail():
    real_next = torch.full((2, 192), 3.0, requires_grad=True)
    future = torch.full((2, 192), 10.0, requires_grad=True)
    target = successor_target(real_next, future, torch.tensor([False, True]), gamma=0.5)
    assert torch.equal(target[0], torch.full((192,), 8.0))
    assert torch.equal(target[1], real_next[1])
    assert not target.requires_grad


@pytest.mark.parametrize("gamma", [-1.0, 1.1, float("nan")])
def test_invalid_discount_rejected(gamma):
    with pytest.raises(ValueError):
        successor_target(
            torch.zeros(2, 192),
            torch.zeros(2, 192),
            torch.zeros(2, dtype=torch.bool),
            gamma=gamma,
        )


def test_movement_direct_td_and_failed_terminal_are_distinct():
    target, valid = movement_target(
        torch.tensor([7.0, 7.0, 7.0, 0.0], requires_grad=True),
        torch.tensor([99.0, 11.0, 11.0, 99.0], requires_grad=True),
        torch.tensor([True, False, False, True]),
        torch.tensor([True, True, False, False]),
    )
    assert target.tolist() == [7, 18, 0, 0]
    assert valid.tolist() == [True, True, False, True]
    assert not target.requires_grad


def test_efficiency_is_ratio_of_sums_not_sum_of_ratios_or_squared_distance():
    states = torch.zeros(3, 4, 192)
    states[0, :, 0] = torch.tensor([0, 1, 2, 3])
    states[1, :, 0] = torch.tensor([0, 3, 1, 3])
    states[2, :, 0] = torch.tensor([0, 1, 200, 300])
    eta, valid = trajectory_efficiency(
        states.requires_grad_(), torch.tensor([3, 3, 1]), epsilon=1e-8
    )
    torch.testing.assert_close(eta, torch.tensor([1, 3 / 7, 1]))
    assert valid.all() and not eta.requires_grad


def test_stationary_path_is_not_labeled_as_high_efficiency():
    eta, valid = trajectory_efficiency(
        torch.zeros(2, 3, 192), torch.tensor([2, 0]), epsilon=1e-8
    )
    assert not valid.any()
    assert eta.tolist() == [0, 0]


def test_weights_prioritize_within_group_and_keep_unknown_at_one():
    eta = torch.tensor([0.8, 0.2, 0.1, 0.1, float("nan")], requires_grad=True)
    weights = efficiency_weights(
        eta,
        torch.tensor([True, True, True, True, False]),
        torch.tensor([0, 0, 1, 1, 0]),
        beta=5,
    )
    assert weights[0] > weights[1] > 0
    torch.testing.assert_close(weights[:2].sum(), torch.tensor(2.0))
    assert weights[2:].tolist() == [1, 1, 1]
    assert not weights.requires_grad


def test_beta_zero_recovers_equal_weights():
    weights = efficiency_weights(
        torch.tensor([0.1, 0.9]),
        torch.tensor([True, True]),
        torch.tensor([0, 0]),
        beta=0,
    )
    assert weights.tolist() == [1, 1]


@pytest.mark.parametrize("beta", [-1, float("nan"), float("inf")])
def test_invalid_efficiency_beta_is_rejected(beta):
    with pytest.raises(ValueError):
        efficiency_weights(
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            beta=beta,
        )


def test_weight_is_outside_loss_not_target_and_gradient_is_scaled():
    errors = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
    weights = torch.tensor([3.0, 1.0], requires_grad=True)
    loss = weighted_path_loss(
        errors.square(),
        torch.tensor([True, True, True]),
        torch.tensor([0, 1, 1]),
        weights,
    )
    loss.backward()
    # Path 1 has two observations, but is not given twice the path weight.
    torch.testing.assert_close(errors.grad, torch.tensor([3.0, 0.5, 0.5]))
    assert weights.grad is None
    assert loss.item() == 4


def test_all_invalid_critic_batch_is_differentiable_zero():
    losses = torch.ones(2, requires_grad=True)
    result = weighted_path_loss(
        losses,
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([0, 1]),
        torch.ones(2),
    )
    result.backward()
    assert result.item() == 0
    assert losses.grad.tolist() == [0, 0]


def _loss_inputs():
    return dict(
        state=torch.randn(3, 192, requires_grad=True),
        next_state=torch.randn(3, 192, requires_grad=True),
        bootstrap_state=torch.randn(3, 192, requires_grad=True),
        goal=torch.randn(3, 192, requires_grad=True),
        terminal_after_transition=torch.tensor([False, False, True]),
        observed_cost=torch.tensor([1.0, 2.0, 0.0], requires_grad=True),
        goal_reached=torch.tensor([True, False, True]),
        continuation_valid=torch.tensor([True, True, False]),
        vector_valid=torch.tensor([True, True, False]),
        path_ids=torch.tensor([0, 1, 2]),
        path_weights=torch.tensor([1.5, 0.5, 1.0], requires_grad=True),
        gamma_g=0.95,
        critic_coefficient=1.0,
    )


def test_critic_training_does_not_update_g_or_real_latents_or_ema():
    model = EffModel(g_hidden_dim=16, v_hidden_dim=16)
    inputs = _loss_inputs()
    losses = eff_loss(model, **inputs)
    losses.critic.backward()
    assert all(p.grad is None for p in model.g.parameters())
    assert any(p.grad is not None for p in model.v.parameters())
    assert all(p.grad is None for p in model.target_g.parameters())
    assert all(p.grad is None for p in model.target_v.parameters())
    for key in [
        "state",
        "next_state",
        "bootstrap_state",
        "goal",
        "observed_cost",
        "path_weights",
    ]:
        assert inputs[key].grad is None
    assert not losses.vector_target.requires_grad
    assert not losses.critic_target.requires_grad


def test_vector_training_updates_only_online_g():
    model = EffModel(g_hidden_dim=16, v_hidden_dim=16)
    eff_loss(model, **_loss_inputs()).vector.backward()
    assert any(p.grad is not None for p in model.g.parameters())
    assert all(p.grad is None for p in model.v.parameters())
    assert all(p.grad is None for p in model.target_g.parameters())


def test_optimizer_contains_only_g_and_v_not_target_or_world_model():
    model = EffModel(g_hidden_dim=16, v_hidden_dim=16)
    optimizer = torch.optim.AdamW(model.online_parameters(), lr=1e-4)
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for module in [model.g, model.v] for p in module.parameters()}
    assert actual == expected
    assert not actual.intersection(id(p) for p in model.target_g.parameters())
    assert not actual.intersection(id(p) for p in model.target_v.parameters())


def test_ema_rate_and_target_eval_state():
    model = EffModel(g_hidden_dim=16, v_hidden_dim=16)
    before = next(model.target_g.parameters()).detach().clone()
    with torch.no_grad():
        next(model.g.parameters()).add_(1)
    model.update_targets(rate=0.1)
    torch.testing.assert_close(next(model.target_g.parameters()), before + 0.1)
    model.train()
    assert not model.target_g.training and not model.target_v.training


def test_frozen_value_still_has_candidate_and_goal_input_gradients():
    torch.manual_seed(0)
    model = EffModel(g_hidden_dim=32, v_hidden_dim=32).requires_grad_(False).eval()
    candidate = torch.randn(4, 192, requires_grad=True)
    goal = torch.randn(4, 192, requires_grad=True)
    model.value(candidate, goal, target=True).sum().backward()
    assert candidate.grad.abs().sum() > 0
    assert goal.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_state_dict_roundtrip_includes_both_ema_networks(tmp_path):
    torch.manual_seed(0)
    model = EffModel(g_hidden_dim=16, v_hidden_dim=16)
    path = tmp_path / "eff.pt"
    torch.save(model.state_dict(), path)
    restored = EffModel(g_hidden_dim=16, v_hidden_dim=16)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    state, goal = torch.randn(2, 192), torch.randn(2, 192)
    torch.testing.assert_close(model.value(state, goal), restored.value(state, goal))
    torch.testing.assert_close(
        model.value(state, goal, target=True), restored.value(state, goal, target=True)
    )
