from __future__ import annotations

import inspect
import math
import types

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v0 import (
    ActorFreeTDJEPAPredictorV0,
    build_tdjepa_td_batch_v0,
    ema_update_target_v0,
    project_tasks_to_sphere_v0,
    sample_mixed_tasks_v0,
    tdjepa_goal_score_v0,
    tdjepa_successor_td_target_v0,
)


def _predictor(*, hidden_dim: int = 16) -> ActorFreeTDJEPAPredictorV0:
    return ActorFreeTDJEPAPredictorV0(hidden_dim=hidden_dim)


def test_v0_predictor_has_locked_lewm_dimensions_and_tdjepa_branches():
    predictor = _predictor()
    signature = inspect.signature(ActorFreeTDJEPAPredictorV0.forward)

    output = predictor(
        torch.randn(3, 4, 192),
        torch.randn(3, 4, 25),
        torch.randn(3, 4, 192),
    )

    assert list(signature.parameters) == ["self", "state", "action", "task"]
    assert output.shape == (3, 4, 192)
    assert predictor.state_dim == 192
    assert predictor.action_dim == 25
    assert predictor.task_dim == 192
    assert predictor.output_dim == 192
    assert predictor.embedding_layers == 2
    assert hasattr(predictor, "embed_task")
    assert hasattr(predictor, "embed_state_action")
    assert hasattr(predictor, "output")
    assert not hasattr(predictor, "num_parallel")
    assert not hasattr(predictor, "heads")
    assert not hasattr(predictor, "action_encoder")
    assert not hasattr(predictor, "actor")
    assert not hasattr(predictor, "policy")
    assert not hasattr(predictor, "history_size")


def test_default_v0_predictor_matches_the_tdjepa_forward_map_widths():
    predictor = ActorFreeTDJEPAPredictorV0()

    assert isinstance(predictor.embed_task[0], nn.Linear)
    assert (predictor.embed_task[0].in_features, predictor.embed_task[0].out_features) == (
        384,
        256,
    )
    assert isinstance(predictor.embed_task[1], nn.LayerNorm)
    assert isinstance(predictor.embed_task[2], nn.Tanh)
    assert isinstance(predictor.embed_task[3], nn.Linear)
    assert (predictor.embed_task[3].in_features, predictor.embed_task[3].out_features) == (
        256,
        128,
    )
    assert isinstance(predictor.embed_task[4], nn.ReLU)

    assert isinstance(predictor.embed_state_action[0], nn.Linear)
    assert (
        predictor.embed_state_action[0].in_features,
        predictor.embed_state_action[0].out_features,
    ) == (217, 256)
    assert isinstance(predictor.output[0], nn.Linear)
    assert (predictor.output[0].in_features, predictor.output[0].out_features) == (
        256,
        256,
    )
    assert isinstance(predictor.output[1], nn.ReLU)
    assert isinstance(predictor.output[2], nn.Linear)
    assert (predictor.output[2].in_features, predictor.output[2].out_features) == (
        256,
        192,
    )


def test_mixed_tasks_are_reproducible_bernoulli_draws_on_the_task_sphere():
    goals = torch.arange(1, 1 + 40 * 192, dtype=torch.float32).reshape(40, 192)
    first = sample_mixed_tasks_v0(
        goals,
        generator=torch.Generator().manual_seed(91),
    )
    second = sample_mixed_tasks_v0(
        goals,
        generator=torch.Generator().manual_seed(91),
    )

    assert first.task.shape == (40, 192)
    assert first.goal_mask.shape == (40,)
    assert first.goal_mask.dtype == torch.bool
    assert first.goal_count == int(first.goal_mask.sum())
    assert first.goal_count + first.random_count == 40
    assert torch.equal(first.goal_mask, second.goal_mask)
    assert torch.equal(first.task, second.task)
    expected_norm = torch.full((40,), math.sqrt(192))
    assert torch.allclose(torch.norm(first.task, dim=-1), expected_norm)
    assert torch.allclose(
        first.task[first.goal_mask],
        project_tasks_to_sphere_v0(goals)[first.goal_mask],
    )
    assert not first.task.requires_grad


def test_mixed_tasks_probability_is_applied_per_transition():
    goals = torch.randn(3, 5, 192)
    all_random = sample_mixed_tasks_v0(
        goals,
        goal_probability=0.0,
        generator=torch.Generator().manual_seed(3),
    )
    all_goal = sample_mixed_tasks_v0(
        goals,
        goal_probability=1.0,
        generator=torch.Generator().manual_seed(3),
    )

    assert not bool(all_random.goal_mask.any())
    assert bool(all_goal.goal_mask.all())


def test_td_target_uses_single_target_prediction_and_terminal_mask():
    next_state = torch.full((3, 192), 2.0, requires_grad=True)
    target_next = torch.full((3, 192), 4.0, requires_grad=True)
    terminal = torch.tensor([False, True, False])

    result = tdjepa_successor_td_target_v0(
        next_state,
        target_next,
        gamma=0.5,
        terminal=terminal,
    )

    expected = torch.full((3, 192), 4.0)
    expected[1].fill_(2.0)
    assert torch.equal(result, expected)
    assert not result.requires_grad


def test_td_batch_detaches_frozen_inputs_and_updates_only_online_predictor():
    torch.manual_seed(17)
    online = _predictor()
    target = online.make_target()
    state = torch.randn(5, 192, requires_grad=True)
    action = torch.randn(5, 25, requires_grad=True)
    task = torch.randn(5, 192, requires_grad=True)
    next_state = torch.randn(5, 192, requires_grad=True)
    next_action = torch.randn(5, 25, requires_grad=True)

    batch = build_tdjepa_td_batch_v0(
        online,
        target,
        state,
        action,
        task,
        next_state,
        next_action,
        gamma=0.97,
        terminal=torch.tensor([False, False, True, False, False]),
    )
    batch.td_loss.backward()

    assert batch.prediction.shape == (5, 192)
    assert batch.target.shape == (5, 192)
    assert batch.per_transition_td_loss.shape == (5,)
    assert torch.isfinite(batch.td_loss)
    assert not batch.target.requires_grad
    assert all(
        value.grad is None for value in (state, action, task, next_state, next_action)
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in online.parameters()
    )
    assert all(parameter.grad is None for parameter in target.parameters())


@pytest.mark.skipif(
    not hasattr(torch, "autocast"), reason="torch.autocast requires modern PyTorch"
)
def test_bfloat16_autocast_uses_float32_td_and_goal_score_accumulation():
    """The formal bf16 policy must not change targets, reductions, or gradients."""

    torch.manual_seed(23)
    online = _predictor()
    target = online.make_target()
    state = torch.randn(4, 192)
    action = torch.randn(4, 25)
    task = torch.randn(4, 192)
    next_state = torch.randn(4, 192)
    next_action = torch.randn(4, 25)
    terminal = torch.tensor([False, True, False, False])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        batch = build_tdjepa_td_batch_v0(
            online,
            target,
            state,
            action,
            task,
            next_state,
            next_action,
            gamma=0.97,
            terminal=terminal,
        )
        goal_score = tdjepa_goal_score_v0(batch.prediction, batch.task)
        loss = batch.td_loss - 0.01 * goal_score.mean()

    assert batch.prediction.dtype == torch.bfloat16
    assert batch.target.dtype == torch.float32
    assert batch.per_transition_td_loss.dtype == torch.float32
    assert batch.td_loss.dtype == torch.float32
    assert goal_score.dtype == torch.float32
    assert not batch.target.requires_grad
    assert torch.equal(
        goal_score,
        (batch.prediction.float() * batch.task.float()).sum(dim=-1),
    )
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected_next_prediction = target(next_state, next_action, task)
    expected_target = next_state + 0.97 * (~terminal).float().unsqueeze(-1) * (
        expected_next_prediction.float()
    )
    assert torch.equal(batch.target, expected_target)
    assert torch.equal(
        batch.per_transition_td_loss,
        (batch.prediction.float() - expected_target).square().sum(dim=-1),
    )

    score_gradient = torch.autograd.grad(
        goal_score.mean(), batch.prediction, retain_graph=True
    )[0]
    assert score_gradient.dtype == torch.bfloat16
    assert torch.count_nonzero(score_gradient) > 0

    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in online.parameters()
    )
    assert all(parameter.grad is None for parameter in target.parameters())


def test_td_batch_bootstrap_queries_the_dataset_next_action():
    online = _predictor()
    target = online.make_target()
    current_action = torch.zeros(3, 25)
    dataset_next_action = torch.arange(75, dtype=torch.float32).reshape(3, 25)
    recorded: dict[str, torch.Tensor] = {}

    def record_target_forward(self, state, action, task):
        recorded["action"] = action.detach().clone()
        return state.new_zeros(state.shape)

    target.forward = types.MethodType(record_target_forward, target)
    build_tdjepa_td_batch_v0(
        online,
        target,
        torch.randn(3, 192),
        current_action,
        torch.randn(3, 192),
        torch.randn(3, 192),
        dataset_next_action,
        gamma=0.99,
    )

    assert torch.equal(recorded["action"], dataset_next_action)


def test_td_loss_sums_features_then_means_transitions():
    online = _predictor()
    target = online.make_target()

    def zero_forward(self, state, action, task):
        del self, action, task
        return torch.zeros_like(state)

    online.forward = types.MethodType(zero_forward, online)
    target.forward = types.MethodType(zero_forward, target)
    next_state = torch.stack((torch.ones(192), torch.full((192,), 2.0)))
    batch = build_tdjepa_td_batch_v0(
        online,
        target,
        torch.zeros(2, 192),
        torch.zeros(2, 25),
        torch.ones(2, 192),
        next_state,
        torch.zeros(2, 25),
        gamma=0.95,
    )

    assert torch.equal(batch.per_transition_td_loss, torch.tensor([192.0, 768.0]))
    assert batch.td_loss == 480.0


def test_single_predictor_goal_score_is_direct_dot_product():
    task = torch.zeros(2, 192)
    task[:, 0] = torch.tensor([2.0, 3.0])
    prediction = torch.zeros(2, 192)
    prediction[:, 0] = torch.tensor([1.0, 5.0])

    score = tdjepa_goal_score_v0(prediction, task)

    assert torch.equal(score, torch.tensor([2.0, 15.0]))


def test_ema_target_builder_and_update_use_the_previous_target_weight():
    online = _predictor(hidden_dim=8)
    target = online.make_target()
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())

    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        for parameter in online.parameters():
            parameter.fill_(10.0)
    ema_update_target_v0(target, online, decay=0.9)

    assert all(
        torch.allclose(parameter, torch.ones_like(parameter))
        for parameter in target.parameters()
    )
    assert all(not parameter.requires_grad for parameter in target.parameters())


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda model: model(
                torch.randn(2, 191),
                torch.randn(2, 25),
                torch.randn(2, 192),
            ),
            "dimension 192",
        ),
        (
            lambda model: model(
                torch.randn(2, 192),
                torch.randn(2, 24),
                torch.randn(2, 192),
            ),
            "dimension 25",
        ),
        (
            lambda model: model(
                torch.randn(2, 192),
                torch.randn(3, 25),
                torch.randn(2, 192),
            ),
            "leading axes",
        ),
        (
            lambda model: model(
                torch.full((2, 192), float("nan")),
                torch.randn(2, 25),
                torch.randn(2, 192),
            ),
            "finite",
        ),
    ],
)
def test_v0_predictor_rejects_wrong_or_nonfinite_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call(_predictor())


def test_zero_goal_and_trainable_target_are_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        sample_mixed_tasks_v0(torch.zeros(2, 192))

    online = _predictor()
    trainable_target = _predictor()
    with pytest.raises(ValueError, match="must be frozen"):
        build_tdjepa_td_batch_v0(
            online,
            trainable_target,
            torch.randn(2, 192),
            torch.randn(2, 25),
            torch.randn(2, 192),
            torch.randn(2, 192),
            torch.randn(2, 25),
            gamma=0.99,
        )
