from __future__ import annotations

import pytest
import torch
from torch import nn

from tdwm.methods.frozen_td_common import (
    build_frozen_real_td_batch,
    gather_hindsight_goals,
    per_transition_vector_td_mse,
    successor_goal_score,
)
from tdwm.methods.goal_projected_td import goal_projected_td_loss
from tdwm.methods.goal_value_weighted_td import teacher_goal_weighted_td_loss
from tdwm.methods.same_future_goal_advantage import (
    same_future_goal_advantage_td_loss,
)
from tdwm.methods.successor_geometry import successor_goal_cost


class AlignmentSuccessor(nn.Module):
    def __init__(
        self, *, embed_dim: int = 1, action_dim: int = 1, history_size: int = 2
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.history_size = history_size
        self.output_dim = embed_dim + 2
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, latent_history, previous_actions, current_action):
        output = latent_history.new_zeros(*latent_history.shape[:-2], self.output_dim)
        output[..., 0] = latent_history[..., -1, 0] * self.scale
        output[..., 1] = current_action[..., 0] * self.scale
        output[..., 2] = self.scale
        return output


def test_build_frozen_real_td_batch_has_exact_dataset_next_action_alignment():
    successor = AlignmentSuccessor()
    target_successor = AlignmentSuccessor()
    real_latents = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    real_latents.requires_grad_()
    real_ema_latents = (10.0 + torch.arange(6, dtype=torch.float32)).reshape(1, 6, 1)
    real_ema_latents.requires_grad_()
    actions = (100.0 + torch.arange(6, dtype=torch.float32)).reshape(1, 6, 1)
    actions.requires_grad_()
    terminals = torch.tensor([[0, 0, 0, 1, 0, 0]])
    gamma = 0.75

    batch = build_frozen_real_td_batch(
        successor,
        target_successor,
        real_latents,
        real_ema_latents,
        actions,
        gamma=gamma,
        terminals=terminals,
    )

    # With history_size=2, current states are indices 2, 3, and 4.
    assert torch.equal(
        batch.current_history[..., 0],
        torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]]),
    )
    assert torch.equal(
        batch.previous_actions[..., 0],
        torch.tensor([[[101.0], [102.0], [103.0]]]),
    )
    assert torch.equal(
        batch.current_actions[..., 0], torch.tensor([[102.0, 103.0, 104.0]])
    )
    assert torch.equal(batch.aligned_terminal, torch.tensor([[False, True, False]]))

    next_latent = torch.tensor([[[13.0], [14.0], [15.0]]])
    bootstrap = torch.tensor(
        [[[13.0, 103.0, 1.0], [14.0, 104.0, 1.0], [15.0, 105.0, 1.0]]]
    )
    lifted = torch.cat(
        (next_latent, next_latent.square(), torch.ones_like(next_latent)), dim=-1
    )
    continuation = torch.tensor([[[1.0], [0.0], [1.0]]])
    expected_target = (1.0 - gamma) * lifted + gamma * continuation * bootstrap
    assert torch.allclose(batch.target, expected_target)
    assert not batch.target.requires_grad
    assert not batch.current_history.requires_grad
    assert not batch.previous_actions.requires_grad
    assert not batch.current_actions.requires_grad

    per_transition_vector_td_mse(batch.prediction, batch.target).mean().backward()
    assert successor.scale.grad is not None
    assert target_successor.scale.grad is None
    assert real_latents.grad is None
    assert real_ema_latents.grad is None
    assert actions.grad is None


def test_build_frozen_real_td_batch_allows_explicit_later_first_current_index():
    successor = AlignmentSuccessor()
    target_successor = AlignmentSuccessor()
    latents = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    actions = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)

    batch = build_frozen_real_td_batch(
        successor,
        target_successor,
        latents,
        latents,
        actions,
        gamma=0.9,
        first_current_index=4,
    )

    assert batch.prediction.shape == (1, 2, 3)
    assert torch.equal(
        batch.current_history[..., 0], torch.tensor([[[3.0, 4.0], [4.0, 5.0]]])
    )
    assert torch.equal(batch.current_actions[..., 0], torch.tensor([[4.0, 5.0]]))


def test_build_frozen_real_td_batch_rejects_invalid_alignment():
    successor = AlignmentSuccessor()
    target_successor = AlignmentSuccessor(history_size=3)
    latents = torch.zeros(2, 5, 1)
    actions = torch.zeros(2, 5, 1)

    with pytest.raises(ValueError, match="share dimensions"):
        build_frozen_real_td_batch(
            successor,
            target_successor,
            latents,
            latents,
            actions,
            gamma=0.9,
        )

    with pytest.raises(ValueError, match="binary"):
        build_frozen_real_td_batch(
            successor,
            successor,
            latents,
            latents,
            actions,
            gamma=0.9,
            terminals=torch.full((2, 5), 0.5),
        )


def test_gather_hindsight_goals_returns_detached_aligned_latents():
    latents = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    latents.requires_grad_()
    terminals = torch.tensor(
        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.int64
    )
    offsets = torch.tensor([[2, 1, 1], [3, 2, 1]])

    goals = gather_hindsight_goals(latents, terminals, 2, offsets)

    expected_indices = torch.tensor([[4, 4, 5], [5, 5, 5]])
    expected = latents.detach().gather(
        1, expected_indices.unsqueeze(-1).expand(2, 3, 2)
    )
    assert goals.shape == (2, 3, 2)
    assert torch.equal(goals, expected)
    assert not goals.requires_grad


def test_gather_hindsight_goals_rejects_offset_crossing_terminal():
    latents = torch.zeros(1, 6, 2)
    terminals = torch.tensor([[0, 0, 0, 1, 0, 0]])

    with pytest.raises(ValueError, match="without crossing a terminal"):
        gather_hindsight_goals(
            latents,
            terminals,
            2,
            torch.tensor([[2, 1, 1]]),
        )


def test_gather_hindsight_goals_rejects_non_integer_or_wrong_shape_offsets():
    latents = torch.zeros(1, 5, 2)
    terminals = torch.zeros(1, 5)

    with pytest.raises(ValueError, match="integer"):
        gather_hindsight_goals(latents, terminals, 2, torch.tensor([[1.5, 1.0]]))
    with pytest.raises(ValueError, match="goal_offsets must have shape"):
        gather_hindsight_goals(latents, terminals, 2, torch.ones(1, 3))


def test_per_transition_vector_td_mse_preserves_leading_axes_and_detaches_target():
    prediction = torch.tensor(
        [[[1.0, 3.0], [2.0, -2.0]], [[0.0, 4.0], [5.0, 1.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction, requires_grad=True)

    losses = per_transition_vector_td_mse(prediction, target)

    assert losses.shape == (2, 2)
    assert torch.allclose(losses, prediction.square().mean(dim=-1))
    losses.sum().backward()
    assert prediction.grad is not None
    assert target.grad is None


def test_per_transition_vector_td_mse_is_fp32_for_low_precision_inputs():
    prediction = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    target = torch.zeros_like(prediction)

    loss = per_transition_vector_td_mse(prediction, target)

    assert loss.dtype == torch.float32
    assert torch.equal(loss, torch.tensor([2.5]))


def test_successor_goal_score_is_negative_shared_goal_cost_and_detaches_goal():
    torch.manual_seed(4)
    successor = torch.randn(3, 7, requires_grad=True)
    goal = torch.randn(3, 5, requires_grad=True)

    score = successor_goal_score(successor, goal)

    assert torch.allclose(score, -successor_goal_cost(successor, goal))
    score.sum().backward()
    assert successor.grad is not None
    assert goal.grad is None


def test_goal_projected_td_loss_matches_manual_projection_and_gradient_boundary():
    prediction = torch.tensor(
        [[0.3, -0.5, 0.8, 1.2], [1.0, 0.2, -0.1, 0.4]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[-0.2, 0.4, 0.3, 0.9], [0.7, -0.6, 0.2, 1.1]],
        requires_grad=True,
    )
    goal = torch.tensor([[0.5, -0.4], [-0.2, 0.8]], requires_grad=True)

    output = goal_projected_td_loss(prediction, target, goal)
    detached_goal = goal.detach()
    expected_prediction = -successor_goal_cost(prediction, detached_goal)
    expected_target = -successor_goal_cost(target.detach(), detached_goal)
    expected_per_transition = (expected_prediction - expected_target).square()

    assert torch.allclose(output.per_transition_loss, expected_per_transition)
    assert torch.allclose(output.loss, expected_per_transition.mean())
    assert not output.target_score.requires_grad
    assert not output.residual_diagnostics.mean.requires_grad
    output.loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) > 0
    assert target.grad is None
    assert goal.grad is None


def test_teacher_goal_weighted_td_uses_global_mean_one_softmax_weights():
    prediction = torch.zeros(2, 2, 3, requires_grad=True)
    target = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    goal = torch.tensor([[[0.0], [0.5]], [[1.0], [1.5]]], requires_grad=True)
    temperature = 0.7

    output = teacher_goal_weighted_td_loss(
        prediction, target, goal, temperature=temperature
    )
    expected_score = -successor_goal_cost(target.detach(), goal.detach())
    expected_weights = torch.softmax(expected_score.flatten() / temperature, dim=0)
    expected_weights = (expected_weights * expected_weights.numel()).reshape(2, 2)

    assert output.weights.shape == (2, 2)
    assert torch.allclose(output.teacher_score, expected_score)
    assert torch.allclose(output.weights, expected_weights)
    assert torch.allclose(output.weights.mean(), torch.ones(()), atol=1e-6)
    assert torch.allclose(
        output.loss,
        (expected_weights * prediction.detach().sub(target).square().mean(-1)).mean(),
    )
    assert not output.weights.requires_grad
    output.loss.backward()
    assert prediction.grad is not None
    assert target.grad is None
    assert goal.grad is None


def test_teacher_goal_weight_clip_is_applied_then_renormalized():
    prediction = torch.zeros(4, 3)
    target = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 10.0, 0.0], [0.0, 20.0, 0.0]]
    )
    goal = torch.zeros(4, 1)

    output = teacher_goal_weighted_td_loss(
        prediction,
        target,
        goal,
        temperature=0.1,
        weight_clip=(0.2, 1.5),
    )

    assert torch.allclose(output.weights.mean(), torch.ones(()), atol=1e-6)
    assert output.weight_diagnostics.clipped_fraction > 0
    assert output.weight_diagnostics.effective_sample_size <= 4
    assert torch.allclose(
        output.weight_diagnostics.normalized_weights.mean,
        torch.ones(()),
        atol=1e-6,
    )


def test_same_future_goal_advantage_uses_all_batch_goals_including_positive():
    # One-dimensional goals imply three-dimensional successor feature vectors.
    prediction = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.2, 0.0], [1.0, 0.5, 0.0]],
        requires_grad=True,
    )
    goals = torch.tensor([[0.0], [0.5], [1.0]], requires_grad=True)

    output = same_future_goal_advantage_td_loss(
        prediction, target, goals, temperature=0.5
    )
    expected_matrix = -successor_goal_cost(
        target.detach()[:, None, :], goals.detach()[None, :, :]
    )
    expected_positive = expected_matrix.diagonal()
    expected_advantage = expected_positive - expected_matrix.mean(dim=1)
    expected_weights = torch.softmax(expected_advantage / 0.5, dim=0) * 3

    assert output.score_matrix.shape == (3, 3)
    assert torch.allclose(output.score_matrix, expected_matrix)
    assert torch.allclose(output.positive_score, expected_positive)
    assert torch.allclose(output.all_goal_mean_score, expected_matrix.mean(dim=1))
    assert torch.allclose(output.advantage, expected_advantage)
    assert torch.allclose(output.weights, expected_weights)
    assert torch.allclose(output.weights.mean(), torch.ones(()), atol=1e-6)
    assert not output.advantage.requires_grad
    assert not output.weights.requires_grad

    output.loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) > 0
    assert target.grad is None
    assert goals.grad is None


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (
            lambda: per_transition_vector_td_mse(torch.zeros(2, 3), torch.zeros(2, 4)),
            "identical shapes",
        ),
        (
            lambda: goal_projected_td_loss(
                torch.zeros(2, 5), torch.zeros(2, 5), torch.zeros(2, 2)
            ),
            "goal must have shape",
        ),
        (
            lambda: teacher_goal_weighted_td_loss(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.zeros(2, 1),
                temperature=0.0,
            ),
            "temperature",
        ),
        (
            lambda: teacher_goal_weighted_td_loss(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.zeros(2, 1),
                temperature=1.0,
                weight_clip=(2.0, 1.0),
            ),
            "weight_clip",
        ),
        (
            lambda: same_future_goal_advantage_td_loss(
                torch.zeros(1, 3),
                torch.zeros(1, 3),
                torch.zeros(1, 1),
                temperature=1.0,
            ),
            "batch size at least 2",
        ),
        (
            lambda: same_future_goal_advantage_td_loss(
                torch.zeros(2, 2, 3),
                torch.zeros(2, 2, 3),
                torch.zeros(2, 1),
                temperature=1.0,
            ),
            r"shape \(batch, features\)",
        ),
    ],
)
def test_frozen_advantage_losses_reject_invalid_inputs(operation, match):
    with pytest.raises((TypeError, ValueError), match=match):
        operation()


def test_frozen_advantage_losses_reject_nonfinite_values():
    prediction = torch.tensor([[float("nan"), 0.0, 0.0]])
    target = torch.zeros_like(prediction)

    with pytest.raises(ValueError, match="finite"):
        per_transition_vector_td_mse(prediction, target)
