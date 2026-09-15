import math

import pytest
import torch
from torch import nn

from tdwm.methods.effplan_efficiency import efficiency_state_path, validate_efficiency_threshold
from tdwm.methods.effplan_safety import PlannerSafety, PlannerSafetyRuntime


class MidpointP(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, left, candidate, right, value, gradient):
        self.calls += 1
        assert torch.isfinite(gradient).all()
        return torch.zeros_like(candidate)


def run(value, *, threshold=0.8, cap=8, planner=None, goal_value=8.):
    start = torch.zeros(1, 192, requires_grad=True)
    goal = start.detach().clone(); goal[0, 0] = goal_value
    p = MidpointP() if planner is None else planner
    path, record = efficiency_state_path(
        p, start, goal, value, max_blocks=cap,
        safety=PlannerSafetyRuntime(PlannerSafety(10, 5)),
        efficiency_threshold=threshold,
    )
    assert not path.requires_grad and start.grad is None
    return path, record, p


def distance(a, b):
    return torch.linalg.vector_norm(b-a, dim=-1)


def test_high_efficiency_stops_before_calling_p():
    states, r, p = run(distance)
    assert states.shape == (1, 2, 192) and p.calls == 0
    assert r['split_attempts'][0]['stop_reason'] == 'efficiency_sufficient'


@pytest.mark.parametrize('side', ['left', 'right', 'both'])
def test_children_decide_independently_and_can_both_continue(side):
    def value(a, b):
        d = distance(a, b)
        need = d > 6
        if side in ('left', 'both'):
            need = need | ((a[..., 0] < 1) & (d > 3))
        if side in ('right', 'both'):
            need = need | ((a[..., 0] >= 4) & (d > 3))
        return d * torch.where(need, 2., 1.)
    states, r, _ = run(value)
    branches = {x['branch'] for x in r['split_attempts'] if x['accepted']}
    expected = {'', 'L', 'R'} if side == 'both' else {'', 'L' if side == 'left' else 'R'}
    assert branches == expected
    assert r['action_blocks'] == len(expected)+1
    assert (states[:, 1:, 0] > states[:, :-1, 0]).all()


def test_increased_split_work_is_not_rejected():
    # Parent=18; children=10+10=20. Both children STILL need subdivision.
    value = lambda a, b: 2*distance(a, b)+2
    states, r, _ = run(value, cap=4)
    assert r['split_attempts'][0]['accepted']
    assert {x['branch'] for x in r['split_attempts'] if x['accepted']} == {'', 'L', 'R'}
    assert states.shape == (1, 5, 192)
    assert r['stop_reason'] == 'budget_cap'


@pytest.mark.parametrize('cap', [1, 2, 10, 20, 40])
def test_budget_cap_never_reported_as_efficiency_success(cap):
    states, r, _ = run(lambda a,b: 2*distance(a,b), cap=cap)
    assert r['action_blocks'] == cap and r['intermediate_nodes'] == cap-1
    assert states.shape == (1, cap+1, 192)
    assert r['stop_reason'] == 'budget_cap'
    assert all(x['stop_reason'] == 'budget_cap' for x in r['split_attempts'] if not x['accepted'])


def test_threshold_equality_stops():
    value = lambda a,b: 2*distance(a,b)
    _, probe, _ = run(value, threshold=0.4)
    eta = probe['split_attempts'][0]['efficiency']
    _, r, p = run(value, threshold=eta)
    assert r['action_blocks'] == 1 and p.calls == 0


def test_zero_length_segment_does_not_recurse_or_evaluate_undefined_efficiency():
    def forbidden(*args):
        raise AssertionError('zero segment must stop before V')
    states, r, p = run(forbidden, goal_value=0)
    assert r['action_blocks'] == 1 and p.calls == 0
    assert r['split_attempts'][0]['stop_reason'] == 'degenerate_segment'
    assert torch.isfinite(states).all()


def test_duplicate_midpoint_guard_is_not_an_efficiency_stop():
    class DuplicateP(MidpointP):
        def forward(self, left, candidate, right, value, gradient):
            return left-candidate
    _, r, _ = run(lambda a,b: 2*distance(a,b), planner=DuplicateP())
    assert r['action_blocks'] == 1
    assert r['split_attempts'][0]['stop_reason'] == 'duplicate_midpoint'


def test_geometric_floor_prevents_low_v_from_producing_efficiency_above_one():
    _, r, _ = run(lambda a,b: distance(a,b)*0)
    root = r['split_attempts'][0]
    assert root['raw_work'] == 0 and root['predicted_work'] == 8
    assert 0 < root['efficiency'] <= 1 and not root['accepted']


@pytest.mark.parametrize('threshold', [0, 1, -0.1, 1.1, math.nan, math.inf, True])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='efficiency_threshold'):
        validate_efficiency_threshold(threshold)


def test_nonfinite_v_fails_instead_of_silently_stopping():
    with pytest.raises(FloatingPointError):
        run(lambda a,b: distance(a,b)*math.nan)
