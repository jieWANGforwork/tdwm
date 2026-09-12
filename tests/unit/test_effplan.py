from __future__ import annotations

import pytest
import torch

from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import (
    StatePlanner,
    binary_midpoint_order,
    dynamics_consistency,
    generate_state_path,
    path_efficiency,
    planner_loss,
    refine_state_path,
    state_feedback,
)


def _distance(start, goal):
    return torch.linalg.vector_norm(goal - start, dim=-1)


@pytest.mark.parametrize("horizon", [1, 2, 3, 5, 8, 10])
def test_binary_subdivision_generates_every_interior_once(horizon):
    available = {0, horizon}
    order = binary_midpoint_order(horizon)
    for left, mid, right in order:
        assert left in available and right in available
        assert mid == (left + right) // 2 and mid not in available
        available.add(mid)
    assert available == set(range(horizon + 1))
    assert len(order) == horizon - 1


def test_generation_has_no_action_or_midpoint_label_input():
    planner = StatePlanner(hidden_dim=16)
    left, right = torch.randn(3, 192), torch.randn(3, 192)
    path = generate_state_path(planner, left, right, _distance, horizon=5, epsilon=1e-6)
    assert path.shape == (3, 6, 192)
    torch.testing.assert_close(path[:, 0], left)
    torch.testing.assert_close(path[:, -1], right)
    torch.testing.assert_close(path[:, 2], (left + right) / 2)
    assert planner.network[0].in_features == 769
    assert planner.network[-1].out_features == 192


def test_efficiency_sums_costs_before_dividing():
    path = torch.zeros(1, 4, 192)
    path[0, :, 0] = torch.tensor([0.0, 3.0, 1.0, 3.0])
    torch.testing.assert_close(
        path_efficiency(path, _distance, epsilon=1e-8), torch.tensor([3 / 7])
    )


def test_dynamics_reference_is_detached_and_time_aligned():
    path = torch.zeros(1, 4, 192, requires_grad=True)
    future = torch.ones(1, 3, 192, requires_grad=True)
    loss = dynamics_consistency(path, future)
    loss.sum().backward()
    assert future.grad is None
    assert torch.equal(path.grad[:, 0], torch.zeros(1, 192))
    torch.testing.assert_close(path.grad[:, 1:], torch.full((1, 3, 192), -2 / 3))
    with pytest.raises(ValueError):
        dynamics_consistency(path, future[:, :2])


def test_feedback_is_gradient_of_candidate_not_anchor_and_batch_invariant():
    path = torch.zeros(1, 3, 192)
    path[0, 1, :2] = torch.tensor([1.0, 1.0])
    path[0, 2, 0] = 2.0
    _, grad = state_feedback(path, _distance, epsilon=1e-6)
    _, repeated = state_feedback(path.repeat(4, 1, 1), _distance, epsilon=1e-6)
    assert grad[0, 1, 1] > 0  # Gradient descent straightens the detour.
    torch.testing.assert_close(repeated, grad.repeat(4, 1, 1))
    assert not grad.requires_grad


def test_feedback_operates_inside_inference_mode():
    model = EffModel(g_hidden_dim=16, v_hidden_dim=16).requires_grad_(False).eval()
    with torch.inference_mode():
        path = torch.randn(2, 4, 192)
        value, gradient = state_feedback(path, model.value, epsilon=1e-6)
    assert value.shape == (2,)
    assert gradient.shape == (2, 4, 192)


def test_refinement_preserves_endpoints_and_time_positions():
    planner = StatePlanner(hidden_dim=16)
    with torch.no_grad():
        planner.network[-1].bias.fill_(0.1)
    path = torch.randn(2, 6, 192)
    changed = refine_state_path(
        planner,
        path,
        _distance,
        predicted_future=torch.randn(2, 5, 192),
        epsilon=1e-6,
        dynamics_coefficient=0.1,
    )
    assert changed.shape == path.shape
    torch.testing.assert_close(changed[:, 0], path[:, 0])
    torch.testing.assert_close(changed[:, -1], path[:, -1])
    torch.testing.assert_close(changed[:, 1:-1], path[:, 1:-1] + 0.1)


def test_phase_one_learns_real_midpoints_without_action_or_cem():
    planner = StatePlanner(hidden_dim=16)
    real = torch.randn(2, 6, 192)
    path = generate_state_path(
        planner, real[:, 0], real[:, -1], _distance, horizon=5, epsilon=1e-6
    )
    losses = planner_loss(
        path,
        real,
        _distance,
        predicted_future=None,
        epsilon=1e-6,
        trajectory_coefficient=1,
        efficiency_coefficient=0,
        dynamics_coefficient=0,
    )
    losses.total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters()
    )
    torch.testing.assert_close(losses.total, losses.trajectory)


def test_phase_two_updates_only_p_but_uses_differentiable_gv():
    torch.manual_seed(0)
    eff = EffModel(g_hidden_dim=16, v_hidden_dim=16).requires_grad_(False).eval()
    planner = StatePlanner(hidden_dim=16)
    real = torch.randn(2, 6, 192)
    path = generate_state_path(
        planner, real[:, 0], real[:, -1], eff.value, horizon=5, epsilon=1e-6
    )
    reference = torch.randn(2, 5, 192, requires_grad=True)
    path = refine_state_path(
        planner,
        path,
        eff.value,
        predicted_future=reference,
        epsilon=1e-6,
        dynamics_coefficient=0.1,
    )
    losses = planner_loss(
        path,
        real,
        eff.value,
        predicted_future=reference,
        epsilon=1e-6,
        trajectory_coefficient=1,
        efficiency_coefficient=1,
        dynamics_coefficient=0.1,
    )
    losses.total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters()
    )
    assert all(p.grad is None for p in eff.parameters())
    assert reference.grad is None


def test_planner_feedback_inputs_detach_but_candidate_does_not():
    planner = StatePlanner(hidden_dim=16)
    with torch.no_grad():
        planner.network[-1].weight.normal_()
    candidate = torch.randn(2, 192, requires_grad=True)
    value = torch.randn(2, requires_grad=True)
    gradient = torch.randn(2, 192, requires_grad=True)
    planner(candidate, candidate, candidate, value, gradient).sum().backward()
    assert candidate.grad is not None
    assert value.grad is None and gradient.grad is None
