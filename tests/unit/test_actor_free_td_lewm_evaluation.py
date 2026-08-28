from __future__ import annotations

from copy import deepcopy

import pytest

from tdwm.evaluation.actor_free_td_lewm import (
    CHECKPOINT_SEMANTICS,
    FORMAL_O50_PLANNING,
    _validate_checkpoint,
    configure_actor_free_td_evaluation_mode,
    load_actor_free_td_evaluation_protocol,
    validate_actor_free_td_evaluation_protocol,
)

CONFIGS = {
    "parallel_real": (
        "configs/experiment/"
        "actor_free_td_lewm_parallel_real_cube_checkpoint_o50.yaml"
    ),
    "serial_decoupled": (
        "configs/experiment/"
        "actor_free_td_lewm_serial_decoupled_cube_checkpoint_o50.yaml"
    ),
    "serial_coupled": (
        "configs/experiment/"
        "actor_free_td_lewm_serial_coupled_cube_checkpoint_o50.yaml"
    ),
    "hybrid": (
        "configs/experiment/"
        "actor_free_td_lewm_hybrid_cube_checkpoint_o50.yaml"
    ),
}


@pytest.mark.parametrize(("variant", "path"), CONFIGS.items())
def test_all_variants_share_one_actor_free_cube_o50_protocol(variant, path):
    protocol = load_actor_free_td_evaluation_protocol(path)

    assert protocol["variant"] == variant
    assert protocol["successor"]["goal_conditioning"] == "none"
    assert protocol["successor"]["actor"] == "none"
    assert protocol["successor"]["td_bootstrap"] is True
    assert protocol["evaluation"]["episodes"] == 50
    assert protocol["evaluation"]["goal_offset"] == 50
    for key, expected in FORMAL_O50_PLANNING.items():
        assert protocol["planning"][key] == expected
    assert protocol["inference_objective"]["goal_usage"] == (
        "planning_linear_readout_only"
    )
    assert protocol["inference_objective"]["goal_enters_successor_head"] is False
    assert protocol["inference_objective"]["learned_actor"] is False


def _checkpoint_for(protocol, *, variant=None):
    successor_config = {
        "embed_dim": protocol["model"]["embed_dim"],
        "action_dim": 25,
        "history_size": protocol["successor"]["history_size"],
        "hidden_dim": protocol["successor"]["hidden_dim"],
        "gamma": protocol["successor"]["gamma"],
        "variant": variant or protocol["variant"],
        "feature_basis": protocol["successor"]["feature_basis"],
        **CHECKPOINT_SEMANTICS,
    }
    payload = {
        "method": "actor_free_td_lewm",
        "variant": variant or protocol["variant"],
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "world_model_state_dict": {},
        "successor_state_dict": {},
        "world_model_config": {"_target_": "fake.WorldModel"},
        "successor_config": successor_config,
    }
    return successor_config, payload


def test_joint_checkpoint_identity_is_locked_to_the_selected_variant():
    protocol = load_actor_free_td_evaluation_protocol(
        CONFIGS["parallel_real"]
    )
    config, payload = _checkpoint_for(protocol)
    _validate_checkpoint(
        payload=payload, successor_config=config, protocol=protocol
    )

    wrong_config, wrong_payload = _checkpoint_for(
        protocol, variant="serial_coupled"
    )
    with pytest.raises(ValueError, match="variant differs"):
        _validate_checkpoint(
            payload=wrong_payload,
            successor_config=wrong_config,
            protocol=protocol,
        )


def test_joint_checkpoint_must_contain_both_models_and_instantiation_config():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["hybrid"])
    config, payload = _checkpoint_for(protocol)
    del payload["world_model_state_dict"]

    with pytest.raises(ValueError, match="world_model_state_dict"):
        _validate_checkpoint(
            payload=payload, successor_config=config, protocol=protocol
        )


def test_protocol_rejects_goal_conditioning_and_a_learned_actor():
    protocol = load_actor_free_td_evaluation_protocol(
        CONFIGS["serial_coupled"]
    )
    goal_conditioned = deepcopy(protocol)
    goal_conditioned["inference_objective"]["goal_enters_successor_head"] = True
    with pytest.raises(ValueError, match="goal-free successor head"):
        validate_actor_free_td_evaluation_protocol(goal_conditioned)

    actor = deepcopy(protocol)
    actor["successor"]["actor"] = "policy_network"
    with pytest.raises(ValueError, match="successor.actor"):
        validate_actor_free_td_evaluation_protocol(actor)


def test_smoke_and_pilot_budgets_do_not_mutate_the_formal_protocol():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["hybrid"])
    smoke = configure_actor_free_td_evaluation_mode(
        protocol, smoke=True, pilot=False
    )
    pilot = configure_actor_free_td_evaluation_mode(
        protocol, smoke=False, pilot=True
    )

    assert protocol["evaluation"]["episodes"] == 50
    assert protocol["planning"]["candidates"] == 300
    assert smoke["evaluation"]["episodes"] == 1
    assert smoke["planning"]["candidates"] == 8
    assert pilot["evaluation"]["episodes"] == 10
    assert pilot["planning"]["candidates"] == 128
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_actor_free_td_evaluation_mode(
            protocol, smoke=True, pilot=True
        )
