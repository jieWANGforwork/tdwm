from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1_c4 import (
    ActorFreeTDJEPAPredictorV1C4,
    build_two_branch_td_loss_v1_c4,
    ema_update_target_v1_c4,
    predict_frozen_lewm_aligned_state_v1_c4,
    successor_td_target_v1_c4,
)


class _RecordingActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192, bias=False)
        self.seen: list[torch.Tensor] = []
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[:25, :25].copy_(torch.eye(25))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        self.seen.append(action.detach().clone())
        return self.projection(action)


class _RecordingFrozenWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))
        self.action_encoder = _RecordingActionEncoder()
        self.seen_state: torch.Tensor | None = None
        self.seen_action: torch.Tensor | None = None

    def predict(
        self, state_history: torch.Tensor, action_embedding: torch.Tensor
    ) -> torch.Tensor:
        self.seen_state = state_history.detach().clone()
        self.seen_action = action_embedding.detach().clone()
        return state_history + action_embedding


class _BFloat16FrozenWorld(_RecordingFrozenWorld):
    def predict(
        self, state_history: torch.Tensor, action_embedding: torch.Tensor
    ) -> torch.Tensor:
        return super().predict(state_history, action_embedding).to(torch.bfloat16)


def _frozen_world() -> _RecordingFrozenWorld:
    return _RecordingFrozenWorld().requires_grad_(False).eval()


def _predictor(hidden_dim: int = 16) -> ActorFreeTDJEPAPredictorV1C4:
    return ActorFreeTDJEPAPredictorV1C4(hidden_dim=hidden_dim)


def test_c4_predictor_interface_is_strictly_state_task_and_outputs_192() -> None:
    predictor = _predictor()
    signature = inspect.signature(ActorFreeTDJEPAPredictorV1C4.forward)

    output = predictor(torch.randn(2, 3, 192), torch.randn(2, 3, 192))

    assert list(signature.parameters) == ["self", "state", "task"]
    assert output.shape == (2, 3, 192)
    assert predictor.state_dim == predictor.task_dim == predictor.output_dim == 192
    assert all("action" not in name for name, _ in predictor.named_parameters())
    with pytest.raises(TypeError):
        predictor(torch.randn(2, 192), torch.randn(2, 25), torch.randn(2, 192))


def test_frozen_f_predicts_the_same_time_as_real_c4_state_and_detaches() -> None:
    world = _frozen_world()
    state_history = torch.zeros(2, 3, 192, requires_grad=True)
    previous_actions = torch.zeros(2, 2, 25, requires_grad=True)
    predecessor_action = torch.zeros(2, 25)
    predecessor_action[:, 0] = torch.tensor([2.0, 3.0])
    predecessor_action.requires_grad_()

    prediction = predict_frozen_lewm_aligned_state_v1_c4(
        world, state_history, previous_actions, predecessor_action
    )

    assert prediction.shape == (2, 192)
    assert not prediction.requires_grad
    assert world.seen_state is not None
    assert world.seen_state.shape == (2, 3, 192)
    assert world.action_encoder.seen[0].shape == (6, 1, 25)
    assert world.seen_action is not None
    assert prediction[:, 0].tolist() == [2.0, 3.0]
    assert state_history.grad is None
    assert previous_actions.grad is None
    assert predecessor_action.grad is None
    assert all(parameter.grad is None for parameter in world.parameters())

    changed_action = predecessor_action.detach().clone()
    changed_action[:, 0].add_(5.0)
    changed = predict_frozen_lewm_aligned_state_v1_c4(
        world, state_history.detach(), previous_actions.detach(), changed_action
    )
    assert not torch.equal(prediction, changed)


def test_frozen_f_prediction_restores_the_latent_store_dtype() -> None:
    world = _BFloat16FrozenWorld().requires_grad_(False).eval()
    state_history = torch.randn(2, 3, 192, dtype=torch.float32)
    previous_actions = torch.randn(2, 2, 25, dtype=torch.float32)
    predecessor_action = torch.randn(2, 25, dtype=torch.float32)

    prediction = predict_frozen_lewm_aligned_state_v1_c4(
        world,
        state_history,
        previous_actions,
        predecessor_action,
    )

    assert prediction.dtype == state_history.dtype
    assert prediction.device == state_history.device
    assert not prediction.requires_grad


def test_c4_target_includes_current_once_bootstraps_next_and_masks_terminal() -> None:
    target = _predictor()
    target.requires_grad_(False).eval()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        target.output[-1].bias.fill_(4.0)
    current = torch.stack((torch.full((192,), 2.0), torch.full((192,), 3.0)))
    next_state = torch.stack((torch.full((192,), 7.0), torch.full((192,), 9.0)))
    task = torch.ones(2, 192)
    seen: list[torch.Tensor] = []
    target.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach().clone())
    )

    result = successor_td_target_v1_c4(
        target,
        current.requires_grad_(),
        next_state.requires_grad_(),
        task.requires_grad_(),
        gamma=0.5,
        terminal=torch.tensor([False, True]),
    )

    assert torch.equal(seen[0], next_state.detach())
    assert torch.equal(result[0], torch.full((192,), 4.0))
    assert torch.equal(result[1], torch.full((192,), 3.0))
    assert not result.requires_grad
    # If current had accidentally been passed to G_bar, the recorded input and
    # the first numerical result above would both differ.


def test_two_branch_loss_shares_target_masks_goal_loss_and_only_updates_online() -> None:
    torch.manual_seed(19)
    online = _predictor()
    target = online.make_target()
    real = torch.randn(3, 192, requires_grad=True)
    predicted = torch.randn(3, 192, requires_grad=True)
    next_state = torch.randn(3, 192, requires_grad=True)
    task = torch.randn(3, 192, requires_grad=True)
    goal_mask = torch.tensor([True, False, True])

    output = build_two_branch_td_loss_v1_c4(
        online,
        target,
        real,
        predicted,
        next_state,
        task,
        goal_mask,
        gamma=0.95,
        terminal=torch.tensor([False, False, True]),
        goal_projection_weight=1.0,
    )
    output.total_loss.backward()

    assert output.target.shape == output.real.prediction.shape == (3, 192)
    assert output.predicted.prediction.shape == (3, 192)
    assert not output.target.requires_grad
    assert output.goal_indices.tolist() == [0, 2]
    assert torch.allclose(
        output.real.goal_loss,
        output.real.score_residual[[0, 2]].square().mean(),
    )
    assert torch.allclose(
        output.total_loss,
        0.5 * (output.real.loss + output.predicted.loss),
    )
    assert all(value.grad is None for value in (real, predicted, next_state, task))
    assert all(parameter.grad is None for parameter in target.parameters())
    assert any(parameter.grad is not None for parameter in online.parameters())


def test_random_only_batch_has_vector_td_but_zero_goal_projection() -> None:
    online = _predictor()
    target = online.make_target()
    batch = 2
    output = build_two_branch_td_loss_v1_c4(
        online,
        target,
        torch.randn(batch, 192),
        torch.randn(batch, 192),
        torch.randn(batch, 192),
        torch.randn(batch, 192),
        torch.zeros(batch, dtype=torch.bool),
        gamma=0.95,
    )

    assert output.goal_indices.numel() == 0
    assert output.real.goal_loss.item() == 0.0
    assert output.predicted.goal_loss.item() == 0.0
    assert output.real.vector_loss.item() > 0.0
    assert output.predicted.vector_loss.item() > 0.0


def test_c4_ema_updates_target_by_decay_and_keeps_it_frozen() -> None:
    online = _predictor()
    target = online.make_target()
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(2.0)
        for parameter in target.parameters():
            parameter.fill_(-2.0)
    online_before = {
        name: parameter.detach().clone()
        for name, parameter in online.named_parameters()
    }
    target_before = {
        name: parameter.detach().clone()
        for name, parameter in target.named_parameters()
    }
    target.train()

    ema_update_target_v1_c4(target, online, decay=0.75)

    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())
    for name, parameter in target.named_parameters():
        expected = 0.75 * target_before[name] + 0.25 * online_before[name]
        torch.testing.assert_close(parameter, expected)
    for name, parameter in online.named_parameters():
        torch.testing.assert_close(parameter, online_before[name])
