from __future__ import annotations

import pytest
import torch
from torch import nn

from tdwm.methods.state_neighbor_advantage import (
    state_neighbor_advantage_weighted_td_loss,
)


class ActionCostSuccessor(nn.Module):
    """Expose the first action coordinate as a zero-goal successor cost."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, real_history, previous_actions, current_action):
        del previous_actions
        output = current_action.new_zeros(
            current_action.shape[:-1] + (real_history.shape[-1] + 2,)
        )
        output[..., -2] = self.scale * current_action[..., 0]
        return output


def _inputs():
    history = torch.zeros(2, 2, 3, requires_grad=True)
    previous = torch.zeros(2, 1, 6, requires_grad=True)
    positive = torch.zeros(2, 6, requires_grad=True)
    with torch.no_grad():
        positive[:, 0] = torch.tensor([-4.0, 2.0])
    neighbors = torch.zeros(2, 2, 6, requires_grad=True)
    with torch.no_grad():
        neighbors[:, :, 0] = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    goals = torch.zeros(2, 3, requires_grad=True)
    distances = torch.zeros(2, 2, requires_grad=True)
    return history, previous, positive, neighbors, goals, distances


def test_g1_scores_only_current_actions_and_detaches_advantage_weights():
    successor = ActionCostSuccessor()
    history, previous, positive, neighbors, goals, distances = _inputs()
    td_coefficients = torch.tensor([1.0, 3.0])
    real_td = successor.scale.square() * td_coefficients

    output = state_neighbor_advantage_weighted_td_loss(
        successor,
        history,
        previous,
        positive,
        neighbors,
        goals,
        distances,
        real_td,
        neighbor_temperature=1.0,
        weight_temperature=1.0,
    )

    assert torch.allclose(output.neighbor_attention, torch.full((2, 2), 0.5))
    assert torch.allclose(output.positive_scores, torch.tensor([4.0, -2.0]))
    assert torch.allclose(
        output.neighbor_scores, torch.tensor([[0.0, -2.0], [-1.0, -3.0]])
    )
    assert torch.allclose(output.advantage, torch.tensor([5.0, 0.0]))
    assert torch.allclose(output.weights.mean(), torch.tensor(1.0))
    assert not output.weights.requires_grad
    assert not output.advantage.requires_grad

    output.loss.backward()

    expected_gradient = (output.weights * (2.0 * td_coefficients)).mean()
    assert torch.allclose(successor.scale.grad, expected_gradient)
    # Counterfactual scoring is a detached weight calculation.  Only the
    # externally supplied real-data TD loss contributes gradients.
    for tensor in (history, previous, positive, neighbors, goals, distances):
        assert tensor.grad is None


def test_g1_neighbor_distance_softmax_and_large_advantages_stay_finite():
    successor = ActionCostSuccessor()
    history, previous, positive, neighbors, goals, distances = _inputs()
    with torch.no_grad():
        distances.copy_(torch.tensor([[0.0, 2.0], [3.0, 0.0]]))
        positive[:, 0] *= 100_000.0

    output = state_neighbor_advantage_weighted_td_loss(
        successor,
        history,
        previous,
        positive,
        neighbors,
        goals,
        distances,
        torch.tensor([1.0, 2.0]),
        neighbor_temperature=0.5,
        weight_temperature=0.01,
    )

    assert torch.allclose(
        output.neighbor_attention,
        torch.softmax(-distances.detach() / 0.5, dim=-1),
    )
    assert torch.isfinite(output.weights).all()
    assert torch.isfinite(output.loss)
    assert torch.allclose(output.weights.mean(), torch.tensor(1.0))


def test_counterfactual_successor_scoring_runs_without_autograd_graph():
    successor = ActionCostSuccessor()
    history, previous, positive, neighbors, goals, distances = _inputs()
    real_td = torch.ones(2, requires_grad=True)

    output = state_neighbor_advantage_weighted_td_loss(
        successor,
        history,
        previous,
        positive,
        neighbors,
        goals,
        distances,
        real_td,
        neighbor_temperature=1.0,
        weight_temperature=1.0,
    )
    output.loss.backward()

    assert successor.scale.grad is None
    assert real_td.grad is not None


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("positive", torch.zeros(2, 5), "previous_actions must have shape"),
        ("neighbors", torch.zeros(2, 0, 6), "at least one action"),
        ("goals", torch.zeros(2, 4), "goals must have shape"),
        ("distances", torch.zeros(2, 3), "distances must match"),
        ("td", torch.zeros(2, 1), "one scalar loss"),
    ],
)
def test_g1_rejects_misaligned_shapes(field, replacement, message):
    successor = ActionCostSuccessor()
    history, previous, positive, neighbors, goals, distances = _inputs()
    values = {
        "history": history,
        "previous": previous,
        "positive": positive,
        "neighbors": neighbors,
        "goals": goals,
        "distances": distances,
        "td": torch.ones(2),
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=message):
        state_neighbor_advantage_weighted_td_loss(
            successor,
            values["history"],
            values["previous"],
            values["positive"],
            values["neighbors"],
            values["goals"],
            values["distances"],
            values["td"],
            neighbor_temperature=1.0,
            weight_temperature=1.0,
        )


def test_g1_rejects_invalid_temperatures_and_distances():
    successor = ActionCostSuccessor()
    history, previous, positive, neighbors, goals, distances = _inputs()

    with pytest.raises(ValueError, match="neighbor_temperature"):
        state_neighbor_advantage_weighted_td_loss(
            successor,
            history,
            previous,
            positive,
            neighbors,
            goals,
            distances,
            torch.ones(2),
            neighbor_temperature=0.0,
            weight_temperature=1.0,
        )

    bad_distances = distances.detach().clone()
    bad_distances[0, 0] = -1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        state_neighbor_advantage_weighted_td_loss(
            successor,
            history,
            previous,
            positive,
            neighbors,
            goals,
            bad_distances,
            torch.ones(2),
            neighbor_temperature=1.0,
            weight_temperature=1.0,
        )
