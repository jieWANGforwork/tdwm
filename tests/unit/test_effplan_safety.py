import math

import pytest
import torch

from tdwm.methods.effplan import (
    StatePlanner,
    generate_state_path,
    path_efficiency,
    planner_loss,
    refine_state_path,
    state_feedback,
)
from tdwm.methods.effplan_safety import (
    PlannerSafety,
    PlannerSafetyRuntime,
    limit_vector_norm,
)


def safety():
    return PlannerSafetyRuntime(PlannerSafety(10.0, 5.0))


def zero_value(a, b):
    return (a.sum(-1) + b.sum(-1)) * 0


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_invalid_safety_limits(limit):
    with pytest.raises(ValueError):
        PlannerSafety(limit, 5)
    with pytest.raises(ValueError):
        PlannerSafety(10, limit)


@pytest.mark.parametrize("magnitude", [0.0, 0.1, 1e8, 1e30])
def test_norm_limit_is_per_node_overflow_safe_and_differentiable(magnitude):
    raw = torch.full((2, 3, 192), magnitude, requires_grad=True)
    output = limit_vector_norm(raw, 5)
    assert torch.isfinite(output).all()
    assert torch.linalg.vector_norm(output.double(), dim=-1).max() <= 5.000001
    output.sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_rejected_not_silently_clipped(bad):
    raw = torch.full((1, 192), bad)
    with pytest.raises(FloatingPointError):
        limit_vector_norm(raw, 5)


def test_geometric_lower_bound_retains_values_above_bound_and_path_ratio():
    nodes = torch.zeros(1, 4, 192)
    nodes[0, :, 0] = torch.tensor([0.0, 3.0, 1.0, 3.0])
    def value(a, b):
        return a[..., 0] * 0 + torch.tensor([[1.0, 4.0, 0.0]])
    runtime = safety()
    result = path_efficiency(nodes, value, epsilon=1e-6, safety=runtime)
    torch.testing.assert_close(result, torch.tensor([3 / 9]))
    assert runtime.metrics()[
        "safety/value_below_distance_fraction/mean"
    ] == pytest.approx(2 / 3)
    legacy = path_efficiency(nodes, value, epsilon=1e-6)
    torch.testing.assert_close(legacy, torch.tensor([3 / 5]))


@pytest.mark.parametrize("degenerate", [False, True])
def test_near_zero_v_and_coincident_states_have_finite_bounded_feedback(degenerate):
    nodes = torch.zeros(2, 6, 192) if degenerate else torch.randn(2, 6, 192)
    runtime = safety()
    eta = path_efficiency(nodes, zero_value, epsilon=1e-6, safety=runtime)
    objective, gradient = state_feedback(
        nodes, zero_value, epsilon=1e-6, safety=runtime
    )
    assert (eta >= 0).all() and (eta <= 1.000001).all()
    if degenerate:
        assert eta.eq(0).all()
    assert torch.isfinite(objective).all() and torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient, dim=-1).max() <= 10.00001
    assert not gradient.requires_grad


def test_gradient_cap_preserves_direction_and_batch_independence():
    nodes = torch.randn(1, 6, 192)
    future = torch.full((1, 5, 192), 1e6)
    kwargs = dict(epsilon=1e-6, predicted_future=future, dynamics_coefficient=0.1)
    _, raw = state_feedback(
        nodes, zero_value, safety=PlannerSafetyRuntime(PlannerSafety(1e20, 5)), **kwargs
    )
    _, safe = state_feedback(nodes, zero_value, safety=safety(), **kwargs)
    torch.testing.assert_close(
        safe[:, 1:] / 10, torch.nn.functional.normalize(raw[:, 1:], dim=-1)
    )
    _, repeated = state_feedback(
        nodes.repeat(3, 1, 1),
        zero_value,
        epsilon=1e-6,
        predicted_future=future.repeat(3, 1, 1),
        dynamics_coefficient=0.1,
        safety=safety(),
    )
    torch.testing.assert_close(repeated, safe.repeat(3, 1, 1))


def test_generation_and_refinement_cap_deltas_preserve_ends_and_backpropagate():
    torch.manual_seed(3)
    p = StatePlanner(hidden_dim=8)
    with torch.no_grad():
        p.network[-1].bias.fill_(1000)
    start, goal = torch.randn(2, 192), torch.randn(2, 192)
    runtime = safety()
    nodes = generate_state_path(
        p, start, goal, zero_value, horizon=5, epsilon=1e-6, safety=runtime
    )
    before = nodes.detach().clone()
    future = torch.randn(2, 5, 192, requires_grad=True)
    changed = refine_state_path(
        p,
        nodes,
        zero_value,
        predicted_future=future,
        epsilon=1e-6,
        dynamics_coefficient=0.1,
        safety=runtime,
    )
    torch.testing.assert_close(changed[:, 0], start)
    torch.testing.assert_close(changed[:, -1], goal)
    assert (
        torch.linalg.vector_norm(changed[:, 1:-1] - before[:, 1:-1], dim=-1).max()
        <= 5.00001
    )
    loss = planner_loss(
        changed,
        torch.randn_like(changed),
        zero_value,
        predicted_future=future,
        epsilon=1e-6,
        trajectory_coefficient=1,
        efficiency_coefficient=0.1,
        dynamics_coefficient=0.1,
        safety=runtime,
    )
    loss.total.backward()
    assert future.grad is None
    assert all(
        t.grad is not None and torch.isfinite(t.grad).all() for t in p.parameters()
    )
    assert any(t.grad.abs().sum() > 0 for t in p.parameters())
    assert all(math.isfinite(v) for v in runtime.metrics().values())
    assert runtime.metrics()["safety/state_delta_cap_fraction/mean"] == 1


def test_protected_feedback_operates_in_inference_mode():
    with torch.inference_mode():
        nodes = torch.randn(2, 6, 192)
        _, gradient = state_feedback(nodes, zero_value, epsilon=1e-6, safety=safety())
    assert torch.isfinite(gradient).all()
