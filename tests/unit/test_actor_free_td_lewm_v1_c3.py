from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1_c3 import (
    RP1StateValueV1C3,
    V1C3Config,
    build_rp1_td_loss_v1_c3,
    discounted_primitive_cost_v1_c3,
    ema_update_target_v1_c3,
    expectile_huber_td_loss_v1_c3,
    rp1_block_window_v1_c3,
    rp1_temporal_td_target_v1_c3,
)


def _critic(*, hidden_dim: int = 16, embedding_dim: int = 8) -> RP1StateValueV1C3:
    return RP1StateValueV1C3(
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        depth=2,
    )


def test_default_v1_c3_config_and_mrn_architecture_are_locked():
    config = V1C3Config()
    critic = RP1StateValueV1C3()

    assert config.backup_blocks == 10
    assert (
        config.hidden_dim,
        config.embedding_dim,
        config.depth,
        config.gamma,
        config.expectile_tau,
        config.ema_rate,
    ) == (256, 128, 2, 0.98, 0.03, 0.005)
    linear = [module for module in critic.head if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (192, 256),
        (256, 256),
        (256, 256),
    ]
    assert sum(parameter.numel() for parameter in critic.parameters()) == 180_992
    with pytest.raises(ValueError, match="exactly 192"):
        RP1StateValueV1C3(state_dim=191)


def test_mrn_value_is_nonnegative_directed_and_exactly_zero_at_goal():
    torch.manual_seed(7)
    critic = _critic()
    state = torch.randn(6, 192)
    goal = torch.randn(6, 192)

    state_u, state_v = critic.components(state)
    goal_u, goal_v = critic.components(goal)
    expected = torch.linalg.vector_norm(
        state_u.float() - goal_u.float(), dim=-1
    ) + torch.relu(goal_v.float() - state_v.float()).amax(dim=-1)
    value = critic(state, goal)

    assert torch.equal(value, expected)
    assert torch.all(value >= 0.0)
    assert torch.equal(critic(goal, goal), torch.zeros(6))
    assert not torch.equal(critic(state, goal), critic(goal, state))


def test_block_window_uses_five_primitive_steps_and_caps_backup_at_fifty():
    window = rp1_block_window_v1_c3(
        torch.tensor([0, 10, 12]),
        torch.tensor([3, 10, 13]),
    )

    assert torch.equal(window.delta_primitive, torch.tensor([0, 50, 60]))
    assert torch.equal(window.n_eff_primitive, torch.tensor([15, 50, 50]))
    assert torch.equal(window.exact_mask, torch.tensor([True, True, False]))
    with pytest.raises(ValueError, match="same episode"):
        rp1_block_window_v1_c3(torch.tensor([4]), torch.tensor([3]))


def test_discounted_cost_and_bootstrap_exponents_use_primitive_steps():
    steps = torch.tensor([5, 50])
    assert torch.equal(
        discounted_primitive_cost_v1_c3(steps, gamma=1.0),
        torch.tensor([5.0, 50.0]),
    )
    gamma = 0.98
    expected_cost = torch.tensor(
        [(1.0 - gamma**5) / (1.0 - gamma), (1.0 - gamma**50) / (1.0 - gamma)]
    )
    assert torch.allclose(
        discounted_primitive_cost_v1_c3(steps, gamma=gamma),
        expected_cost,
    )

    gamma_one = rp1_temporal_td_target_v1_c3(
        torch.tensor([55]),
        torch.tensor([50]),
        torch.tensor([10.0]),
        gamma=1.0,
    )
    discounted = rp1_temporal_td_target_v1_c3(
        torch.tensor([55]),
        torch.tensor([50]),
        torch.tensor([10.0]),
        gamma=gamma,
    )
    assert gamma_one.target.item() == 60.0
    assert discounted.target.item() == pytest.approx(
        (1.0 - gamma**50) / (1.0 - gamma) + gamma**50 * 10.0,
        rel=1e-6,
    )


def test_target_uses_exact_delta_at_and_inside_backup_boundary():
    result = rp1_temporal_td_target_v1_c3(
        torch.tensor([45, 50, 55]),
        torch.tensor([50, 50, 50]),
        torch.tensor([999.0, 999.0, 10.0]),
        gamma=1.0,
    )

    assert torch.equal(result.exact_mask, torch.tensor([True, True, False]))
    assert torch.equal(result.target, torch.tensor([45.0, 50.0, 60.0]))
    assert not result.target.requires_grad


def test_expectile_huber_weights_cost_overestimates_more_than_underestimates():
    output = expectile_huber_td_loss_v1_c3(
        torch.tensor([2.0, 0.0], requires_grad=True),
        torch.tensor([1.0, 1.0], requires_grad=True),
        tau=0.03,
        huber_beta=1.0,
    )

    assert torch.equal(output.residual, torch.tensor([1.0, -1.0]))
    assert torch.equal(output.weight, torch.tensor([0.97, 0.03]))
    assert torch.allclose(output.per_example_loss, torch.tensor([0.485, 0.015]))
    assert output.loss.item() == pytest.approx(0.25)
    assert output.per_example_loss[0] > output.per_example_loss[1]


def test_td_batch_detaches_frozen_latents_and_updates_only_online_critic():
    torch.manual_seed(19)
    critic = _critic()
    target = critic.make_target()
    anchor = torch.randn(4, 192, requires_grad=True)
    successor = torch.randn(4, 192, requires_grad=True)
    goal = torch.randn(4, 192, requires_grad=True)

    batch = build_rp1_td_loss_v1_c3(
        critic,
        target,
        anchor,
        successor,
        goal,
        torch.tensor([5, 50, 55, 70]),
        torch.tensor([50, 50, 50, 50]),
        gamma=0.98,
        tau=0.03,
        huber_beta=1.0,
    )
    batch.loss.backward()

    assert batch.prediction.shape == batch.target.shape == (4,)
    assert torch.equal(batch.exact_mask, torch.tensor([True, True, False, False]))
    assert not batch.target.requires_grad
    assert all(value.grad is None for value in (anchor, successor, goal))
    assert any(parameter.grad is not None for parameter in critic.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())
    assert math.isfinite(batch.loss.item())


def test_make_target_and_ema_use_online_rate_and_keep_target_frozen():
    critic = _critic(hidden_dim=8, embedding_dim=4)
    target = critic.make_target()
    assert target is not critic
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())

    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        for parameter in critic.parameters():
            parameter.fill_(10.0)
    ema_update_target_v1_c3(target, critic, rate=0.1)

    assert all(
        torch.allclose(parameter, torch.ones_like(parameter))
        for parameter in target.parameters()
    )
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_v1_c3_rejects_bad_shapes_units_nonfinite_inputs_and_wrong_target_state():
    critic = _critic()
    target = critic.make_target()
    with pytest.raises(ValueError, match="shape"):
        critic(torch.randn(2, 191), torch.randn(2, 191))
    bad = torch.randn(2, 192)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        critic(bad, torch.randn(2, 192))
    with pytest.raises(ValueError, match="aligned"):
        rp1_temporal_td_target_v1_c3(
            torch.tensor([6]), torch.tensor([50]), torch.tensor([1.0])
        )
    with pytest.raises(ValueError, match="backup window"):
        rp1_temporal_td_target_v1_c3(
            torch.tensor([55]), torch.tensor([55]), torch.tensor([1.0])
        )

    target.train()
    with pytest.raises(ValueError, match="eval mode"):
        build_rp1_td_loss_v1_c3(
            critic,
            target,
            torch.randn(2, 192),
            torch.randn(2, 192),
            torch.randn(2, 192),
            torch.tensor([5, 10]),
            torch.tensor([50, 50]),
        )
