from __future__ import annotations

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1
from tdwm.methods.actor_free_td_lewm_v1_c2 import (
    first_q_alignment_v1_c2_loss,
    rollout_frozen_lewm_candidates_v1_c2,
    sample_first_q_candidates_v1_c2,
)


class _ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            self.projection.weight[:25, :25].copy_(torch.eye(25))

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.projection(raw_action)


class _RecordingFrozenWorld(nn.Module):
    def __init__(self, terminal_latent: torch.Tensor) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_encoder = _ActionEncoder()
        self.register_buffer("terminal_latent", terminal_latent.clone())
        self.seen_info: dict[str, torch.Tensor] | None = None
        self.seen_actions: torch.Tensor | None = None
        self.seen_history_size: int | None = None

    def rollout(
        self,
        info: dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self.seen_info = {key: value.detach().clone() for key, value in info.items()}
        self.seen_actions = action_sequence.detach().clone()
        self.seen_history_size = history_size
        if history_size is None:
            raise AssertionError("C2 must pass the explicit three-state history size.")
        batch, candidates, action_count, _ = action_sequence.shape
        horizon = action_count - (history_size - 1)
        predicted = action_sequence.new_zeros(
            batch,
            candidates,
            history_size + horizon,
            192,
        )
        predicted[:, :, :history_size, :] = info["emb"]
        predicted[:, :, -1, :] = self.terminal_latent.to(action_sequence)
        return {"predicted_emb": predicted}


def _frozen_world(terminal_latent: torch.Tensor) -> _RecordingFrozenWorld:
    return _RecordingFrozenWorld(terminal_latent).requires_grad_(False).eval()


def _alignment_inputs(
    *,
    batch: int = 2,
    candidates: int = 5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cpu").manual_seed(913)
    state_history = torch.randn(batch, 3, 192, generator=generator)
    action_history = torch.randn(batch, 2, 25, generator=generator)
    candidate_actions = torch.randn(batch, candidates, 5, 25, generator=generator)
    candidate_actions[:, 0].zero_()
    goal_latent = torch.randn(batch, 192, generator=generator)
    task = torch.randn(batch, 192, generator=generator)
    return state_history, action_history, candidate_actions, goal_latent, task


def test_cem_candidate_sampling_has_forced_mean_candidate_shape_and_seed() -> None:
    reference = torch.ones(3, 192, dtype=torch.float64)
    first_generator = torch.Generator(device="cpu").manual_seed(370_009)
    second_generator = torch.Generator(device="cpu").manual_seed(370_009)

    first = sample_first_q_candidates_v1_c2(
        reference,
        candidate_count=5,
        rollout_horizon=5,
        initial_variance=1.0,
        generator=first_generator,
    )
    second = sample_first_q_candidates_v1_c2(
        reference,
        candidate_count=5,
        rollout_horizon=5,
        initial_variance=1.0,
        generator=second_generator,
    )

    assert first.shape == (3, 5, 5, 25)
    assert first.dtype == reference.dtype
    assert first.device == reference.device
    assert torch.count_nonzero(first[:, 0]) == 0
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[:, 1:]) > 0
    assert not first.requires_grad


def test_three_state_two_action_five_candidate_rollout_is_axis_aligned() -> None:
    batch, candidates, horizon = 2, 5, 5
    state_history = torch.arange(batch * 3 * 192, dtype=torch.float32).reshape(
        batch, 3, 192
    )
    action_history = torch.arange(batch * 2 * 25, dtype=torch.float32).reshape(
        batch, 2, 25
    )
    candidate_actions = torch.arange(
        batch * candidates * horizon * 25,
        dtype=torch.float32,
    ).reshape(batch, candidates, horizon, 25)
    terminal = torch.arange(batch * candidates * 192, dtype=torch.float32).reshape(
        batch, candidates, 192
    )
    world = _frozen_world(terminal)

    result = rollout_frozen_lewm_candidates_v1_c2(
        world,
        state_history,
        action_history,
        candidate_actions,
    )

    assert result.shape == (batch, candidates, 192)
    assert torch.equal(result, terminal)
    assert not result.requires_grad
    assert world.seen_history_size == 3
    assert world.seen_info is not None
    assert world.seen_actions is not None
    assert world.seen_info["emb"].shape == (batch, candidates, 3, 192)
    assert world.seen_info["pixels"].shape == (batch, candidates, 3, 1)
    assert torch.equal(
        world.seen_info["emb"],
        state_history.unsqueeze(1).expand(batch, candidates, 3, 192),
    )
    assert world.seen_actions.shape == (batch, candidates, 7, 25)
    assert torch.equal(
        world.seen_actions[:, :, :2],
        action_history.unsqueeze(1).expand(batch, candidates, 2, 25),
    )
    assert torch.equal(world.seen_actions[:, :, 2:], candidate_actions)


def test_alignment_teacher_is_stop_gradient_and_only_g_receives_gradients() -> None:
    torch.manual_seed(29)
    batch, candidates = 2, 5
    state, action_history, actions, goal, task = _alignment_inputs(
        batch=batch,
        candidates=candidates,
    )
    terminal = torch.randn(batch, candidates, 192)
    world = _frozen_world(terminal)
    predictor = ActorFreeTDJEPAPredictorV1(
        hidden_dim=32,
        hidden_layers=0,
        embedding_layers=2,
    )
    state.requires_grad_()
    action_history.requires_grad_()
    actions.requires_grad_()
    goal.requires_grad_()
    task.requires_grad_()

    output = first_q_alignment_v1_c2_loss(
        predictor,
        world,
        state,
        action_history,
        actions,
        goal,
        task,
        torch.ones(batch, dtype=torch.bool),
        torch.ones(batch, dtype=torch.bool),
        teacher_temperature=1.0,
        student_temperature=1.0,
        standardization_epsilon=1e-6,
    )
    output.loss.backward()

    assert output.loss.requires_grad
    assert not output.teacher_cost.requires_grad
    assert not output.teacher_probability.requires_grad
    assert all(parameter.grad is None for parameter in world.parameters())
    predictor_gradients = [parameter.grad for parameter in predictor.parameters()]
    assert all(gradient is not None for gradient in predictor_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in predictor_gradients)
    assert state.grad is None
    assert action_history.grad is None
    assert actions.grad is None
    assert goal.grad is None
    assert task.grad is None


def test_teacher_terminal_goal_cost_is_summed_mse() -> None:
    batch, candidates = 1, 5
    state, action_history, actions, goal, task = _alignment_inputs(
        batch=batch,
        candidates=candidates,
    )
    goal.zero_()
    terminal = torch.zeros(batch, candidates, 192)
    terminal[0, 0].fill_(2.0)
    terminal[0, 1, :2] = torch.tensor([3.0, 4.0])
    terminal[0, 2, 9] = -5.0
    terminal[0, 3, :3] = 1.0
    world = _frozen_world(terminal)
    predictor = ActorFreeTDJEPAPredictorV1(hidden_dim=32, hidden_layers=0)

    output = first_q_alignment_v1_c2_loss(
        predictor,
        world,
        state,
        action_history,
        actions,
        goal,
        task,
        torch.ones(batch, dtype=torch.bool),
        torch.ones(batch, dtype=torch.bool),
        teacher_temperature=1.0,
        student_temperature=1.0,
        standardization_epsilon=1e-6,
    )

    expected = (terminal - goal.unsqueeze(1)).square().sum(dim=-1)
    assert torch.equal(output.teacher_cost, expected)
    assert output.teacher_cost.tolist() == [[768.0, 25.0, 25.0, 3.0, 0.0]]


def test_no_eligible_examples_return_graph_connected_zero_loss() -> None:
    torch.manual_seed(31)
    batch, candidates = 2, 5
    state, action_history, actions, goal, task = _alignment_inputs(
        batch=batch,
        candidates=candidates,
    )
    terminal = torch.randn(batch, candidates, 192)
    world = _frozen_world(terminal)
    predictor = ActorFreeTDJEPAPredictorV1(hidden_dim=32, hidden_layers=0)

    output = first_q_alignment_v1_c2_loss(
        predictor,
        world,
        state,
        action_history,
        actions,
        goal,
        task,
        torch.zeros(batch, dtype=torch.bool),
        torch.ones(batch, dtype=torch.bool),
        teacher_temperature=1.0,
        student_temperature=1.0,
        standardization_epsilon=1e-6,
    )

    assert output.loss.item() == 0.0
    assert output.loss.requires_grad
    assert not bool(output.eligible_mask.any())
    output.loss.backward()
    gradients = [parameter.grad for parameter in predictor.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
