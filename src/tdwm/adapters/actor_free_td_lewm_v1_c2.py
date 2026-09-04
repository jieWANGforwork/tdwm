"""Deployment adapter for Actor-Free TD-LeWM V1 C2."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tdwm.adapters.frozen_actor_free_td_v1_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    OBJECTIVE_VERSION,
    FrozenActorFreeTDV1MethodSpec,
    load_frozen_actor_free_td_v1_checkpoint,
    make_frozen_actor_free_td_v1_policy,
    require_exact_values,
)

METHOD = "actor_free_td_lewm_v1_c2"
VARIANT = "c2"

FIRST_Q_ALIGNMENT_LOCK = {
    "version": 1,
    "loss": "teacher_student_cross_entropy",
    "weight": 1.0,
    "teacher": "frozen_lewm_terminal_goal_cost",
    "teacher_gradient": "stop_gradient",
    "teacher_cost_reducer": "final_predicted_latent_summed_mse",
    "student": "first_action_goal_projection",
    "student_gradient": "online_predictor_only",
    "candidate_source": "cem_initial_gaussian_no_actor",
    "candidate_count": 16,
    "rollout_horizon": 5,
    "initial_mean": 0.0,
    "initial_variance": 1.0,
    "force_first_candidate_to_mean": True,
    "goal_population": "goal_derived_tasks_with_valid_five_block_context",
    "max_goal_examples_per_batch": 8,
    "subset_selection": "first_valid_in_random_replay_batch",
    "score_standardization": "population_z_score_over_candidates",
    "standardization_epsilon": 1.0e-6,
    "teacher_temperature": 1.0,
    "student_temperature": 1.0,
    "candidate_sampling_seed_offset": 370009,
}


def _validate_method_config(config: Mapping[str, Any]) -> None:
    objective = config["joint_objective"]
    require_exact_values(
        objective,
        {
            "objective": "goal_projected_td_plus_first_q_alignment",
            "goal_signal": "matched_future_latent",
            "goal_projection_weight": 1.0,
            "goal_projection_target": "detached_td_target_projection",
            "goal_projection_prediction_gradient": "online_predictor",
            "projection_population": "goal_derived_tasks_only",
        },
        label="predictor_config.joint_objective",
    )
    alignment = objective.get("first_q_alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError(
            "predictor_config.joint_objective.first_q_alignment must be a mapping."
        )
    if set(alignment) != set(FIRST_Q_ALIGNMENT_LOCK):
        raise ValueError(
            "predictor_config.joint_objective.first_q_alignment must contain "
            "exactly the locked C2 fields."
        )
    require_exact_values(
        alignment,
        FIRST_Q_ALIGNMENT_LOCK,
        label="predictor_config.joint_objective.first_q_alignment",
    )
    for key, expected in FIRST_Q_ALIGNMENT_LOCK.items():
        if type(alignment[key]) is not type(expected):
            raise ValueError(
                "predictor_config.joint_objective.first_q_alignment."
                f"{key} must use the locked {type(expected).__name__} type."
            )


METHOD_SPEC = FrozenActorFreeTDV1MethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM V1 C2",
    objective_keys=(
        "objective",
        "goal_signal",
        "goal_projection_weight",
        "goal_projection_target",
        "goal_projection_prediction_gradient",
        "projection_population",
        "first_q_alignment",
    ),
    validate_method_config=_validate_method_config,
)


def load_actor_free_td_lewm_v1_c2_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_frozen_actor_free_td_v1_checkpoint(
        checkpoint_path, spec=METHOD_SPEC, map_location=map_location
    )


def make_actor_free_td_lewm_v1_c2_policy(**kwargs):
    return make_frozen_actor_free_td_v1_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FIRST_Q_ALIGNMENT_LOCK",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_v1_c2_checkpoint",
    "make_actor_free_td_lewm_v1_c2_policy",
]
