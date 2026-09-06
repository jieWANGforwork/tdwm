from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1_c4 import (
    ActorFreeTDJEPAPredictorV1C4,
    build_td_loss_v1_c4,
    ema_update_target_v1_c4,
    predict_frozen_lewm_ghost_next_state_v1_c4,
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


def test_frozen_f_predicts_detached_ghost_next_state_from_current_action() -> None:
    world = _frozen_world()
    state_history = torch.zeros(2, 3, 192, requires_grad=True)
    previous_actions = torch.zeros(2, 2, 25, requires_grad=True)
    current_action = torch.zeros(2, 25)
    current_action[:, 0] = torch.tensor([2.0, 3.0])
    current_action.requires_grad_()

    prediction = predict_frozen_lewm_ghost_next_state_v1_c4(
        world, state_history, previous_actions, current_action
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
    assert current_action.grad is None
    assert all(parameter.grad is None for parameter in world.parameters())

    changed_action = current_action.detach().clone()
    changed_action[:, 0].add_(5.0)
    changed = predict_frozen_lewm_ghost_next_state_v1_c4(
        world, state_history.detach(), previous_actions.detach(), changed_action
    )
    assert not torch.equal(prediction, changed)


def test_frozen_f_prediction_restores_the_latent_store_dtype() -> None:
    world = _BFloat16FrozenWorld().requires_grad_(False).eval()
    state_history = torch.randn(2, 3, 192, dtype=torch.float32)
    previous_actions = torch.randn(2, 2, 25, dtype=torch.float32)
    current_action = torch.randn(2, 25, dtype=torch.float32)

    prediction = predict_frozen_lewm_ghost_next_state_v1_c4(
        world,
        state_history,
        previous_actions,
        current_action,
    )

    assert prediction.dtype == state_history.dtype
    assert prediction.device == state_history.device
    assert not prediction.requires_grad


def test_c4_target_uses_real_next_once_bootstraps_ghost_and_masks_terminal() -> None:
    target = _predictor()
    target.requires_grad_(False).eval()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        target.output[-1].bias.fill_(4.0)
    real_next = torch.stack((torch.full((192,), 2.0), torch.full((192,), 3.0)))
    ghost_next_next = torch.stack(
        (torch.full((192,), 7.0), torch.full((192,), 9.0))
    )
    task = torch.ones(2, 192)
    seen: list[torch.Tensor] = []
    target.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach().clone())
    )

    result = successor_td_target_v1_c4(
        target,
        real_next.requires_grad_(),
        ghost_next_next.requires_grad_(),
        task.requires_grad_(),
        gamma=0.5,
        terminal=torch.tensor([False, True]),
    )

    assert seen[0].shape == (1, 192)
    assert torch.equal(seen[0], ghost_next_next[:1].detach())
    assert torch.equal(result[0], torch.full((192,), 4.0))
    assert torch.equal(result[1], torch.full((192,), 3.0))
    assert not result.requires_grad
    # If real z_(i+1) had accidentally been passed to G_bar, the recorded input
    # and the first numerical result above would both differ.


def test_single_branch_loss_uses_post_action_ghosts_and_real_immediate() -> None:
    torch.manual_seed(19)
    online = _predictor()
    target = online.make_target()
    online_ghost = torch.randn(3, 192, requires_grad=True)
    real_next = torch.randn(3, 192, requires_grad=True)
    target_ghost = torch.randn(3, 192, requires_grad=True)
    task = torch.randn(3, 192, requires_grad=True)
    goal_mask = torch.tensor([True, False, True])

    online_inputs: list[torch.Tensor] = []
    target_inputs: list[torch.Tensor] = []
    online.register_forward_pre_hook(
        lambda _module, inputs: online_inputs.append(inputs[0].detach().clone())
    )
    target.register_forward_pre_hook(
        lambda _module, inputs: target_inputs.append(inputs[0].detach().clone())
    )

    output = build_td_loss_v1_c4(
        online,
        target,
        online_ghost,
        real_next,
        target_ghost,
        task,
        goal_mask,
        gamma=0.95,
        terminal=torch.tensor([False, False, True]),
        goal_projection_weight=1.0,
    )
    output.total_loss.backward()

    assert output.target.shape == output.prediction.shape == (3, 192)
    assert torch.equal(online_inputs[0], online_ghost.detach())
    assert torch.equal(target_inputs[0], target_ghost[:2].detach())
    assert not output.target.requires_grad
    assert output.goal_indices.tolist() == [0, 2]
    assert torch.allclose(
        output.goal_loss,
        output.score_residual[[0, 2]].square().mean(),
    )
    assert torch.allclose(
        output.total_loss,
        output.vector_loss + output.goal_loss,
    )
    assert all(
        value.grad is None
        for value in (online_ghost, real_next, target_ghost, task)
    )
    assert all(parameter.grad is None for parameter in target.parameters())
    assert any(parameter.grad is not None for parameter in online.parameters())


def test_random_only_batch_has_vector_td_but_zero_goal_projection() -> None:
    online = _predictor()
    target = online.make_target()
    batch = 2
    output = build_td_loss_v1_c4(
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
    assert output.goal_loss.item() == 0.0
    assert output.vector_loss.item() > 0.0


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
