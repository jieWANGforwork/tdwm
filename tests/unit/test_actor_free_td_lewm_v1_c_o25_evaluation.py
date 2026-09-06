from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.evaluation.actor_free_td_lewm_v1_c import (
    configure_actor_free_td_lewm_v1_c_evaluation_mode,
    load_actor_free_td_lewm_v1_c_evaluation_protocol,
    validate_actor_free_td_lewm_v1_c_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    actor_free_td_v1_output_directory_name,
)

O25_CONFIG = Path(
    "configs/experiment/actor_free_td_lewm_v1_c_cube_checkpoint_o25.yaml"
)
O50_CONFIG = Path(
    "configs/experiment/actor_free_td_lewm_v1_c_cube_checkpoint_o50.yaml"
)
SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "f_plus_g_first_q2",
    "g_only_f_rollout_mean",
)


def test_v1_c_o25_inherits_the_audited_o50_protocol_envelope() -> None:
    protocol = load_actor_free_td_lewm_v1_c_evaluation_protocol(O25_CONFIG)

    assert protocol["id"].endswith("checkpoint_o25")
    assert protocol["evaluation"]["episodes"] == 50
    assert protocol["evaluation"]["goal_offset"] == 25
    assert protocol["planning"]["episode_budget"] == 50
    assert protocol["planning"]["horizon"] == 5
    assert protocol["planning"]["receding_horizon"] == 5
    assert protocol["planning"]["executed_environment_steps_before_replanning"] == 25
    assert actor_free_td_v1_output_directory_name(
        protocol,
        smoke=False,
        pilot=False,
        score_mode="f_only",
    ) == "actor_free_td_lewm_v1_c_cube_o25_f_only_formal"


@pytest.mark.parametrize("score_mode", SCORE_MODES)
def test_v1_c_o25_six_modes_use_the_exact_mode_specific_cadence(
    score_mode: str,
) -> None:
    formal = load_actor_free_td_lewm_v1_c_evaluation_protocol(O25_CONFIG)
    configured = configure_actor_free_td_lewm_v1_c_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=score_mode,
        g_first_weight=0.25 if "first" in score_mode else None,
    )

    expected = 1 if score_mode == "g_only" else 5
    assert configured["planning"]["horizon"] == expected
    assert configured["planning"]["receding_horizon"] == expected
    assert configured["planning"][
        "executed_environment_steps_before_replanning"
    ] == expected * 5
    assert configured["inference_objective"]["replanning"] == (
        "every_action_block" if expected == 1 else "every_five_action_blocks"
    )
    validate_actor_free_td_lewm_v1_c_evaluation_protocol(configured)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evaluation", "episodes", 49),
        ("evaluation", "goal_offset", 26),
        ("planning", "episode_budget", 100),
        ("planning", "receding_horizon", 1),
        ("planning", "executed_environment_steps_before_replanning", 5),
        ("evaluation", "episodes", 50.0),
        ("planning", "receding_horizon", 5.0),
        ("planning", "episode_budget", True),
    ],
)
def test_v1_c_o25_rejects_protocol_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    protocol = load_actor_free_td_lewm_v1_c_evaluation_protocol(O25_CONFIG)
    changed = deepcopy(protocol)
    changed[section][field] = value

    with pytest.raises(ValueError):
        validate_actor_free_td_lewm_v1_c_evaluation_protocol(changed)


def test_v1_c_o50_numerical_planning_contract_is_unchanged() -> None:
    formal = load_actor_free_td_lewm_v1_c_evaluation_protocol(O50_CONFIG)
    assert formal["evaluation"] == {
        "episodes": 50,
        "goal_offset": 50,
        "start_goal_source": "same_dataset_episode",
    }
    for score_mode in SCORE_MODES:
        configured = configure_actor_free_td_lewm_v1_c_evaluation_mode(
            formal,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
            g_first_weight=0.25 if "first" in score_mode else None,
        )
        assert configured["planning"]["episode_budget"] == 100
        assert configured["planning"]["receding_horizon"] == 1
        assert configured["planning"]["horizon"] == (
            1 if score_mode == "g_only" else 5
        )
        assert configured["inference_objective"]["replanning"] == (
            "every_action_block"
        )
        validate_actor_free_td_lewm_v1_c_evaluation_protocol(configured)
