from __future__ import annotations

import pytest
import torch
from torch import nn

from tdwm.methods.action_prefix_advantage_common import build_zero_mean_action_prefixes
from tdwm.methods.action_prefix_marginal_advantage import (
    action_prefix_marginal_advantage_td_loss,
)
from tdwm.methods.action_prefix_mean_advantage import (
    action_prefix_mean_advantage_td_loss,
)


class ActionSumCostSuccessor(nn.Module):
    """Expose the sum of normalized action coordinates as zero-goal cost."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.grad_enabled: list[bool] = []

    def forward(self, real_history, previous_actions, current_action):
        del previous_actions
        self.grad_enabled.append(torch.is_grad_enabled())
        output = current_action.new_zeros(
            current_action.shape[:-1] + (real_history.shape[-1] + 2,)
        )
        output[..., -2] = self.scale * current_action.sum(dim=-1)
        return output


def _inputs():
    history = torch.zeros(2, 2, 3, requires_grad=True)
    previous = torch.zeros(2, 1, 25, requires_grad=True)
    full_actions = torch.zeros(2, 25, requires_grad=True)
    with torch.no_grad():
        full_actions[0, ::5] = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        full_actions[1, ::5] = 1.0
    goals = torch.zeros(2, 3, requires_grad=True)
    return history, previous, full_actions, goals


def test_zero_mean_prefix_builder_retains_recorded_prefix_and_zeros_suffix():
    actions = torch.arange(1.0, 26.0).reshape(1, 25)

    prefixes = build_zero_mean_action_prefixes(actions).reshape(1, 5, 5, 5)

    assert prefixes.shape == (1, 5, 5, 5)
    action_slots = actions.reshape(1, 5, 5)
    for prefix_index in range(5):
        assert torch.equal(
            prefixes[:, prefix_index, : prefix_index + 1],
            action_slots[:, : prefix_index + 1],
        )
        assert torch.count_nonzero(prefixes[:, prefix_index, prefix_index + 1 :]) == 0


def test_g2_prefix_mean_formula_weights_only_full_action_td():
    successor = ActionSumCostSuccessor()
    history, previous, full_actions, goals = _inputs()
    td_coefficients = torch.tensor([1.0, 3.0])
    real_td = successor.scale.square() * td_coefficients

    output = action_prefix_mean_advantage_td_loss(
        successor,
        history,
        previous,
        full_actions,
        goals,
        real_td,
        weight_temperature=1.0,
    )

    expected_scores = torch.tensor(
        [[1.0, 3.0, 6.0, 10.0, 15.0], [-1.0, -2.0, -3.0, -4.0, -5.0]]
    )
    assert torch.allclose(output.prefix_scores, expected_scores)
    assert torch.allclose(output.full_score, torch.tensor([15.0, -5.0]))
    assert torch.allclose(output.prefix_mean_score, torch.tensor([7.0, -3.0]))
    assert torch.allclose(output.advantage, torch.tensor([8.0, -2.0]))
    assert torch.allclose(output.weights.mean(), torch.tensor(1.0))
    assert not output.advantage.requires_grad
    assert not output.weights.requires_grad
    assert successor.grad_enabled == [False]

    output.loss.backward()

    expected_gradient = (output.weights * (2.0 * td_coefficients)).mean()
    assert torch.allclose(successor.scale.grad, expected_gradient)
    for tensor in (history, previous, full_actions, goals):
        assert tensor.grad is None


def test_g3_uses_mean_of_all_four_adjacent_prefix_marginals():
    successor = ActionSumCostSuccessor()
    history, previous, full_actions, goals = _inputs()
    real_td = torch.tensor([2.0, 4.0], requires_grad=True)

    output = action_prefix_marginal_advantage_td_loss(
        successor,
        history,
        previous,
        full_actions,
        goals,
        real_td,
        weight_temperature=0.5,
    )

    assert torch.allclose(
        output.marginal_scores,
        torch.tensor([[2.0, 3.0, 4.0, 5.0], [-1.0, -1.0, -1.0, -1.0]]),
    )
    assert torch.allclose(output.advantage, torch.tensor([3.5, -1.0]))
    assert torch.allclose(
        output.advantage,
        (output.prefix_scores[:, -1] - output.prefix_scores[:, 0]) / 4.0,
    )
    assert torch.allclose(output.weights.mean(), torch.tensor(1.0))
    assert not output.marginal_scores.requires_grad
    assert successor.grad_enabled == [False]

    output.loss.backward()

    assert successor.scale.grad is None
    assert real_td.grad is not None


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.zeros(2, 20), "require a 25D Cube action block"),
        (torch.zeros(2, 24), "require a 25D Cube action block"),
        (torch.zeros(3, 25), "full_actions must have shape"),
    ],
)
def test_prefix_objectives_reject_misaligned_actions(replacement, message):
    successor = ActionSumCostSuccessor()
    history, previous, _, goals = _inputs()

    with pytest.raises(ValueError, match=message):
        action_prefix_mean_advantage_td_loss(
            successor,
            history,
            previous,
            replacement,
            goals,
            torch.ones(replacement.shape[0]),
            weight_temperature=1.0,
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_prefix_objectives_reject_invalid_weight_temperature(temperature):
    successor = ActionSumCostSuccessor()
    history, previous, full_actions, goals = _inputs()

    with pytest.raises(ValueError, match="temperature"):
        action_prefix_marginal_advantage_td_loss(
            successor,
            history,
            previous,
            full_actions,
            goals,
            torch.ones(2),
            weight_temperature=temperature,
        )
