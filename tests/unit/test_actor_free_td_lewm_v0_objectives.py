from __future__ import annotations

import pytest
import torch

from tdwm.methods.actor_free_td_lewm_v0_objectives import (
    goal_projected_v0_loss,
    goal_value_weighted_v0_loss,
    neighbor_action_advantage_v0_loss,
    prefix_marginal_advantage_v0_loss,
    prefix_mean_advantage_v0_loss,
    same_future_goal_advantage_v0_loss,
)


def _mixed_mask() -> torch.Tensor:
    return torch.tensor([True, False, True, False])


def test_c_adds_projection_only_for_goal_derived_transitions():
    online = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]],
        requires_grad=True,
    )
    td_target = torch.tensor(
        [[0.5, 0.0], [8.0, 0.0], [1.0, 2.0], [9.0, 0.0]],
        requires_grad=True,
    )
    tasks = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 0.0]],
        requires_grad=True,
    )
    per_td = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = goal_projected_v0_loss(
        online,
        td_target,
        tasks,
        _mixed_mask(),
        per_td,
        projection_coefficient=0.25,
    )

    prediction_score = (online.detach() * tasks.detach()).sum(dim=-1)
    target_score = (td_target.detach() * tasks.detach()).sum(dim=-1)
    residual = prediction_score - target_score
    expected_projection = residual[torch.tensor([0, 2])].square().mean()
    expected_loss = per_td.detach().mean() + 0.25 * expected_projection
    assert torch.allclose(output.prediction_score, prediction_score)
    assert torch.allclose(output.target_score, target_score)
    assert torch.allclose(output.projection_loss, expected_projection)
    assert torch.allclose(output.loss, expected_loss)

    output.loss.backward()
    assert online.grad is not None
    assert torch.count_nonzero(online.grad[0]) > 0
    assert torch.count_nonzero(online.grad[2]) > 0
    assert torch.count_nonzero(online.grad[1]) == 0
    assert torch.count_nonzero(online.grad[3]) == 0
    assert td_target.grad is None
    assert tasks.grad is None
    assert torch.allclose(per_td.grad, torch.full((4,), 0.25))


def test_c_empty_goal_subset_is_exact_base_td_fallback():
    online = torch.randn(3, 4, requires_grad=True)
    td_target = torch.randn(3, 4)
    tasks = torch.randn(3, 4)
    per_td = torch.tensor([1.0, 2.0, 6.0], requires_grad=True)

    output = goal_projected_v0_loss(
        online,
        td_target,
        tasks,
        torch.zeros(3, dtype=torch.bool),
        per_td,
        projection_coefficient=10.0,
    )

    assert torch.equal(output.goal_indices, torch.empty(0, dtype=torch.long))
    assert output.projection_loss == 0
    assert torch.allclose(output.loss, per_td.mean())


def test_d_weights_only_goal_subset_and_keeps_random_weights_at_one():
    td_target = torch.tensor(
        [[2.0, 0.0], [100.0, 0.0], [-1.0, 0.0], [-100.0, 0.0]],
        requires_grad=True,
    )
    tasks = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        requires_grad=True,
    )
    per_td = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = goal_value_weighted_v0_loss(
        td_target, tasks, _mixed_mask(), per_td, temperature=0.5
    )

    selected_scores = torch.tensor([2.0, -1.0])
    selected_weights = torch.softmax(selected_scores / 0.5, dim=0) * 2
    expected_weights = torch.ones(4)
    expected_weights[torch.tensor([0, 2])] = selected_weights
    expected_weights = expected_weights / expected_weights.mean()
    assert output.used_weighting
    assert torch.allclose(output.weights, expected_weights)
    assert torch.allclose(output.weights.mean(), torch.ones(()), atol=1e-7)
    assert torch.allclose(output.weights[~_mixed_mask()], torch.ones(2), atol=1e-7)
    assert torch.allclose(output.loss, (expected_weights * per_td.detach()).mean())

    output.loss.backward()
    assert td_target.grad is None
    assert tasks.grad is None
    assert torch.allclose(per_td.grad, expected_weights / 4)


def test_f_uses_fixed_td_targets_and_only_goal_subset_goals():
    td_target = torch.tensor(
        [[99.0, 99.0], [1.0, 2.0], [3.0, 4.0], [-99.0, -99.0]],
        requires_grad=True,
    )
    tasks = torch.tensor(
        [[8.0, 8.0], [2.0, 0.0], [0.0, 1.0], [-8.0, -8.0]],
        requires_grad=True,
    )
    mask = torch.tensor([False, True, True, False])
    per_td = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = same_future_goal_advantage_v0_loss(
        td_target, tasks, mask, per_td, temperature=0.75
    )

    expected_matrix = td_target.detach()[1:3] @ tasks.detach()[1:3].T
    expected_positive = expected_matrix.diagonal()
    expected_advantage = expected_positive - expected_matrix.mean(dim=1)
    subset_weights = torch.softmax(expected_advantage / 0.75, dim=0) * 2
    expected_weights = torch.tensor([1.0, subset_weights[0], subset_weights[1], 1.0])
    expected_weights = expected_weights / expected_weights.mean()
    assert torch.allclose(output.score_matrix, expected_matrix)
    assert torch.allclose(output.positive_score, expected_positive)
    assert torch.allclose(output.advantage, expected_advantage)
    assert torch.allclose(output.weights, expected_weights)
    assert not output.score_matrix.requires_grad
    assert not output.advantage.requires_grad

    output.loss.backward()
    assert td_target.grad is None
    assert tasks.grad is None
    assert torch.allclose(per_td.grad, expected_weights / 4)


def test_g1_uses_distance_attention_and_detaches_all_candidate_scores():
    positive = torch.tensor([3.0, 50.0, 1.0, -50.0], requires_grad=True)
    neighbors = torch.tensor(
        [[1.0, 2.0], [8.0, 9.0], [0.0, 4.0], [-8.0, -9.0]],
        requires_grad=True,
    )
    distances = torch.tensor(
        [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [1.0, 1.0]],
        requires_grad=True,
    )
    per_td = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = neighbor_action_advantage_v0_loss(
        positive,
        neighbors,
        distances,
        _mixed_mask(),
        per_td,
        neighbor_temperature=0.5,
        weight_temperature=1.25,
    )

    expected_attention = torch.softmax(-distances.detach()[[0, 2]] / 0.5, dim=-1)
    expected_advantage = positive.detach()[[0, 2]] - (
        expected_attention * neighbors.detach()[[0, 2]]
    ).sum(dim=-1)
    subset_weights = torch.softmax(expected_advantage / 1.25, dim=0) * 2
    expected_weights = torch.tensor([subset_weights[0], 1.0, subset_weights[1], 1.0])
    expected_weights = expected_weights / expected_weights.mean()
    assert torch.allclose(output.neighbor_attention, expected_attention)
    assert torch.allclose(output.advantage, expected_advantage)
    assert torch.allclose(output.weights, expected_weights)

    output.loss.backward()
    assert positive.grad is None
    assert neighbors.grad is None
    assert distances.grad is None
    assert torch.allclose(per_td.grad, expected_weights / 4)


def test_g2_is_q5_minus_mean_of_all_five_prefix_scores():
    prefix_scores = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 5.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [-9.0, -9.0, -9.0, -9.0, -9.0],
        ],
        requires_grad=True,
    )
    per_td = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = prefix_mean_advantage_v0_loss(
        prefix_scores, _mixed_mask(), per_td, temperature=0.8
    )

    selected = prefix_scores.detach()[[0, 2]]
    expected_advantage = selected[:, -1] - selected.mean(dim=-1)
    assert torch.allclose(output.advantage, expected_advantage)
    assert torch.allclose(output.full_score, selected[:, -1])
    assert torch.allclose(output.prefix_mean_score, selected.mean(dim=-1))
    assert torch.allclose(output.weights[~_mixed_mask()], torch.ones(2))
    assert torch.allclose(output.weights.mean(), torch.ones(()), atol=1e-7)
    output.loss.backward()
    assert prefix_scores.grad is None
    assert per_td.grad is not None


def test_g3_is_mean_of_four_adjacent_prefix_marginals():
    prefix_scores = torch.tensor(
        [
            [0.0, 1.0, 3.0, 6.0, 10.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [10.0, 6.0, 3.0, 1.0, 0.0],
            [-9.0, -9.0, -9.0, -9.0, -9.0],
        ],
        requires_grad=True,
    )
    per_td = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = prefix_marginal_advantage_v0_loss(
        prefix_scores, _mixed_mask(), per_td, temperature=0.8
    )

    selected = prefix_scores.detach()[[0, 2]]
    expected_marginals = selected[:, 1:] - selected[:, :-1]
    expected_advantage = expected_marginals.mean(dim=-1)
    assert torch.allclose(output.marginal_scores, expected_marginals)
    assert torch.allclose(output.advantage, expected_advantage)
    assert torch.allclose(
        output.advantage, (selected[:, -1] - selected[:, 0]) / 4
    )
    assert torch.allclose(output.weights[~_mixed_mask()], torch.ones(2))
    assert torch.allclose(output.weights.mean(), torch.ones(()), atol=1e-7)
    output.loss.backward()
    assert prefix_scores.grad is None
    assert per_td.grad is not None


@pytest.mark.parametrize("goal_count", [0, 1])
@pytest.mark.parametrize("variant", ["d", "f", "g1", "g2", "g3"])
def test_weighting_objectives_fall_back_to_base_td_for_small_goal_subset(
    goal_count: int,
    variant: str,
):
    mask = torch.zeros(4, dtype=torch.bool)
    mask[:goal_count] = True
    per_td = torch.tensor([1.0, 2.0, 3.0, 8.0], requires_grad=True)
    td_target = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    tasks = torch.flip(td_target, dims=(0,))
    prefix_scores = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    if variant == "d":
        output = goal_value_weighted_v0_loss(
            td_target, tasks, mask, per_td, temperature=1.0
        )
    elif variant == "f":
        output = same_future_goal_advantage_v0_loss(
            td_target, tasks, mask, per_td, temperature=1.0
        )
    elif variant == "g1":
        output = neighbor_action_advantage_v0_loss(
            torch.arange(4, dtype=torch.float32),
            torch.ones(4, 2),
            torch.ones(4, 2),
            mask,
            per_td,
            neighbor_temperature=1.0,
            weight_temperature=1.0,
        )
    elif variant == "g2":
        output = prefix_mean_advantage_v0_loss(
            prefix_scores, mask, per_td, temperature=1.0
        )
    else:
        output = prefix_marginal_advantage_v0_loss(
            prefix_scores, mask, per_td, temperature=1.0
        )

    assert not output.used_weighting
    assert torch.equal(output.weights, torch.ones(4))
    assert torch.allclose(output.loss, per_td.mean())
    output.loss.backward()
    assert torch.allclose(per_td.grad, torch.full((4,), 0.25))


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (
            lambda: goal_value_weighted_v0_loss(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.tensor([True, False]),
                torch.ones(2),
                temperature=0.0,
            ),
            "temperature",
        ),
        (
            lambda: prefix_mean_advantage_v0_loss(
                torch.zeros(2, 4),
                torch.tensor([True, False]),
                torch.ones(2),
                temperature=1.0,
            ),
            "shape",
        ),
        (
            lambda: neighbor_action_advantage_v0_loss(
                torch.zeros(2),
                torch.zeros(2, 2),
                -torch.ones(2, 2),
                torch.tensor([True, False]),
                torch.ones(2),
                neighbor_temperature=1.0,
                weight_temperature=1.0,
            ),
            "negative",
        ),
    ],
)
def test_v0_objectives_reject_invalid_inputs(operation, match):
    with pytest.raises((TypeError, ValueError), match=match):
        operation()
