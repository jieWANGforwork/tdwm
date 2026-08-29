from __future__ import annotations

import pytest

from tdwm.methods import (
    goal_projected_td,
    goal_value_weighted_td,
    same_future_goal_advantage,
    state_neighbor_advantage,
)


@pytest.mark.parametrize(
    ("module", "variant"),
    [
        (goal_projected_td, "goal_projected_td"),
        (goal_value_weighted_td, "goal_value_weighted_td"),
        (same_future_goal_advantage, "same_future_goal_advantage"),
        (state_neighbor_advantage, "neighbor_action_advantage"),
    ],
)
def test_each_frozen_method_has_an_independent_versioned_module(module, variant):
    assert module.VARIANT == variant
    assert module.OBJECTIVE_VERSION == 1
