"""Compatibility exports for the independently implemented C, D, and F methods.

New code should import each objective from its method-specific module. This
aggregate remains so callers written while the methods were prototyped in one
file continue to work during the integration split.
"""

from tdwm.methods.frozen_td_common import (
    DistributionDiagnostics,
    FrozenRealTDBatch,
    SuccessorModule,
    WeightDiagnostics,
    build_frozen_real_td_batch,
    gather_hindsight_goals,
    per_transition_vector_td_mse,
    successor_goal_score,
)
from tdwm.methods.goal_projected_td import (
    GoalProjectedTDOutput,
    goal_projected_td_loss,
)
from tdwm.methods.goal_value_weighted_td import (
    GoalWeightedTDOutput,
    teacher_goal_weighted_td_loss,
)
from tdwm.methods.same_future_goal_advantage import (
    SameFutureGoalAdvantageTDOutput,
    same_future_goal_advantage_td_loss,
)

__all__ = [
    "DistributionDiagnostics",
    "FrozenRealTDBatch",
    "GoalProjectedTDOutput",
    "GoalWeightedTDOutput",
    "SameFutureGoalAdvantageTDOutput",
    "SuccessorModule",
    "WeightDiagnostics",
    "build_frozen_real_td_batch",
    "gather_hindsight_goals",
    "goal_projected_td_loss",
    "per_transition_vector_td_mse",
    "same_future_goal_advantage_td_loss",
    "successor_goal_score",
    "teacher_goal_weighted_td_loss",
]
