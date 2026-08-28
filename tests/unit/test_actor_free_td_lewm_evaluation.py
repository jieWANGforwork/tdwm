from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.evaluation.actor_free_td_lewm import (
    CHECKPOINT_SEMANTICS,
    DIRECT_CRITIC_SEMANTICS,
    FORMAL_O50_PLANNING,
    _validate_checkpoint,
    configure_actor_free_td_evaluation_mode,
    load_actor_free_td_evaluation_protocol,
    validate_actor_free_td_evaluation_protocol,
)

_CLI_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "evaluate_actor_free_td_lewm.py"
)
_CLI_SPEC = importlib.util.spec_from_file_location(
    "tdwm_evaluate_actor_free_td_lewm_cli", _CLI_PATH
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
evaluation_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(evaluation_cli)

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
        "configs/experiment/" "actor_free_td_lewm_hybrid_cube_checkpoint_o50.yaml"
    ),
    "goal_hybrid": (
        "configs/experiment/" "actor_free_td_lewm_goal_hybrid_cube_checkpoint_o50.yaml"
    ),
    "imaginary_hybrid": (
        "configs/experiment/"
        "actor_free_td_lewm_imaginary_hybrid_cube_checkpoint_o50.yaml"
    ),
}
DIRECT_CONFIG = (
    "configs/experiment/"
    "actor_free_td_lewm_direct_goal_hybrid_cube_checkpoint_o50.yaml"
)


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
    expected_goal_usage = (
        "training_goal_readout_and_planning_linear_readout"
        if variant == "goal_hybrid"
        else "planning_linear_readout_only"
    )
    assert protocol["inference_objective"]["goal_usage"] == expected_goal_usage
    assert protocol["inference_objective"]["goal_enters_successor_head"] is False
    assert protocol["inference_objective"]["learned_actor"] is False
    assert protocol["inference_objective"]["score_mode"] == "f_plus_g"


def _checkpoint_for(protocol, *, variant=None):
    successor_config = {
        "embed_dim": protocol["model"]["embed_dim"],
        "action_dim": 25,
        "history_size": protocol["successor"]["history_size"],
        "hidden_dim": protocol["successor"]["hidden_dim"],
        "gamma": protocol["successor"]["gamma"],
        "variant": variant or protocol["variant"],
        "objective_version": protocol["successor"]["objective_version"],
        "feature_basis": protocol["successor"]["feature_basis"],
        **CHECKPOINT_SEMANTICS,
    }
    if protocol["variant"] == "goal_hybrid":
        successor_config.update(
            {
                "goal_readout_training": True,
                "goal_source": protocol["successor"]["goal_source"],
                "goal_offset_weighting": protocol["successor"]["goal_offset_weighting"],
                "goal_terminal_condition": protocol["successor"][
                    "goal_terminal_condition"
                ],
                "goal_readout_branches": protocol["successor"]["goal_readout_branches"],
                "goal_readout_precision": protocol["successor"][
                    "goal_readout_precision"
                ],
                "goal_cost": protocol["successor"]["goal_cost"],
                "goal_enters_successor_head": False,
                "predicted_goal_td_weight": 1.0,
                "real_goal_td_weight": 1.0,
            }
        )
    if protocol["variant"] == "imaginary_hybrid":
        successor_config.update(
            {
                "immediate_feature_source": protocol["successor"][
                    "immediate_feature_source"
                ],
                "bootstrap_state_source": protocol["successor"][
                    "bootstrap_state_source"
                ],
                "imaginary_horizon": protocol["successor"]["imaginary_horizon"],
                "imaginary_predictor_gradient": protocol["successor"][
                    "imaginary_predictor_gradient"
                ],
            }
        )
    payload = {
        "method": "actor_free_td_lewm",
        "variant": variant or protocol["variant"],
        "objective_version": protocol["successor"]["objective_version"],
        "deployment_checkpoint_version": 1,
        "world_model_state_dict": {},
        "successor_state_dict": {},
        "world_model_config": {"_target_": "fake.WorldModel"},
        "successor_config": successor_config,
    }
    return successor_config, payload


def test_joint_checkpoint_identity_is_locked_to_the_selected_variant():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["parallel_real"])
    config, payload = _checkpoint_for(protocol)
    _validate_checkpoint(payload=payload, successor_config=config, protocol=protocol)

    wrong_config, wrong_payload = _checkpoint_for(protocol, variant="serial_coupled")
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


def test_goal_hybrid_checkpoint_requires_version_two_and_goal_metadata():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["goal_hybrid"])
    config, payload = _checkpoint_for(protocol)
    _validate_checkpoint(payload=payload, successor_config=config, protocol=protocol)

    wrong_version = deepcopy(payload)
    wrong_version["objective_version"] = 1
    with pytest.raises(ValueError, match="objective_version differs"):
        _validate_checkpoint(
            payload=wrong_version,
            successor_config=config,
            protocol=protocol,
        )

    missing_metadata = deepcopy(config)
    del missing_metadata["goal_terminal_condition"]
    with pytest.raises(ValueError, match="goal_terminal_condition"):
        _validate_checkpoint(
            payload=payload,
            successor_config=missing_metadata,
            protocol=protocol,
        )


def test_imaginary_hybrid_checkpoint_requires_version_three_and_metadata():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["imaginary_hybrid"])
    config, payload = _checkpoint_for(protocol)
    _validate_checkpoint(payload=payload, successor_config=config, protocol=protocol)

    wrong_version = deepcopy(payload)
    wrong_version["objective_version"] = 1
    with pytest.raises(ValueError, match="objective_version differs"):
        _validate_checkpoint(
            payload=wrong_version,
            successor_config=config,
            protocol=protocol,
        )

    missing_metadata = deepcopy(config)
    del missing_metadata["bootstrap_state_source"]
    with pytest.raises(ValueError, match="bootstrap_state_source"):
        _validate_checkpoint(
            payload=payload,
            successor_config=missing_metadata,
            protocol=protocol,
        )


def test_protocol_rejects_goal_conditioning_and_a_learned_actor():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["serial_coupled"])
    goal_conditioned = deepcopy(protocol)
    goal_conditioned["inference_objective"]["goal_enters_successor_head"] = True
    with pytest.raises(ValueError, match="goal-free successor head"):
        validate_actor_free_td_evaluation_protocol(goal_conditioned)

    actor = deepcopy(protocol)
    actor["successor"]["actor"] = "policy_network"
    with pytest.raises(ValueError, match="successor.actor"):
        validate_actor_free_td_evaluation_protocol(actor)


def test_direct_goal_critic_o50_protocol_and_checkpoint_are_not_sf_factorized():
    protocol = load_actor_free_td_evaluation_protocol(DIRECT_CONFIG)
    critic = protocol["critic"]

    assert protocol["variant"] == "direct_goal_hybrid"
    assert critic["objective_version"] == 3
    assert critic["goal_conditioning"] == "direct_latent_input"
    assert "successor" not in protocol
    assert protocol["inference_objective"]["goal_enters_critic_head"] is True
    assert protocol["inference_objective"]["score_mode"] == "f_plus_c"
    for key, expected in FORMAL_O50_PLANNING.items():
        assert protocol["planning"][key] == expected

    critic_config = {
        "embed_dim": protocol["model"]["embed_dim"],
        "action_dim": 25,
        "history_size": critic["history_size"],
        "hidden_dim": critic["hidden_dim"],
        "gamma": critic["gamma"],
        "variant": protocol["variant"],
        "objective_version": critic["objective_version"],
        "goal_source": critic["goal_source"],
        "goal_offset_weighting": critic["goal_offset_weighting"],
        "goal_terminal_condition": critic["goal_terminal_condition"],
        "td_branches": critic["td_branches"],
        "goal_cost": critic["goal_cost"],
        "goal_enters_critic_head": True,
        "predicted_context_detach": False,
        "predicted_critic_td_weight": 1.0,
        "real_critic_td_weight": 1.0,
        **DIRECT_CRITIC_SEMANTICS,
    }
    payload = {
        "method": "actor_free_td_lewm",
        "variant": "direct_goal_hybrid",
        "objective_version": 3,
        "deployment_checkpoint_version": 1,
        "world_model_state_dict": {},
        "critic_state_dict": {},
        "world_model_config": {"_target_": "fake.WorldModel"},
        "critic_config": critic_config,
    }
    _validate_checkpoint(
        payload=payload,
        successor_config=critic_config,
        protocol=protocol,
    )


def test_smoke_and_pilot_budgets_do_not_mutate_the_formal_protocol():
    protocol = load_actor_free_td_evaluation_protocol(CONFIGS["hybrid"])
    smoke = configure_actor_free_td_evaluation_mode(protocol, smoke=True, pilot=False)
    pilot = configure_actor_free_td_evaluation_mode(protocol, smoke=False, pilot=True)

    assert protocol["evaluation"]["episodes"] == 50
    assert protocol["planning"]["candidates"] == 300
    assert smoke["evaluation"]["episodes"] == 1
    assert smoke["planning"]["candidates"] == 8
    assert pilot["evaluation"]["episodes"] == 10
    assert pilot["planning"]["candidates"] == 128
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_actor_free_td_evaluation_mode(protocol, smoke=True, pilot=True)


def test_missing_score_mode_defaults_to_legacy_combined_score():
    successor = load_actor_free_td_evaluation_protocol(CONFIGS["hybrid"])
    del successor["inference_objective"]["score_mode"]
    validate_actor_free_td_evaluation_protocol(successor)
    successor_configured = configure_actor_free_td_evaluation_mode(
        successor,
        smoke=False,
        pilot=False,
    )
    assert successor_configured["inference_objective"]["score_mode"] == "f_plus_g"

    direct = load_actor_free_td_evaluation_protocol(DIRECT_CONFIG)
    del direct["inference_objective"]["score_mode"]
    validate_actor_free_td_evaluation_protocol(direct)
    direct_configured = configure_actor_free_td_evaluation_mode(
        direct,
        smoke=False,
        pilot=False,
    )
    assert direct_configured["inference_objective"]["score_mode"] == "f_plus_c"


def test_score_mode_override_is_validated_and_recorded_in_runtime_protocol():
    successor = load_actor_free_td_evaluation_protocol(CONFIGS["hybrid"])
    configured = configure_actor_free_td_evaluation_mode(
        successor,
        smoke=False,
        pilot=False,
        score_mode="g_only",
    )
    assert successor["inference_objective"]["score_mode"] == "f_plus_g"
    assert configured["inference_objective"]["score_mode"] == "g_only"

    direct = load_actor_free_td_evaluation_protocol(DIRECT_CONFIG)
    configured_direct = configure_actor_free_td_evaluation_mode(
        direct,
        smoke=False,
        pilot=False,
        score_mode="c_only",
    )
    assert configured_direct["inference_objective"]["score_mode"] == "c_only"

    with pytest.raises(ValueError, match="incompatible"):
        configure_actor_free_td_evaluation_mode(
            successor,
            smoke=False,
            pilot=False,
            score_mode="c_only",
        )
    with pytest.raises(ValueError, match="incompatible"):
        configure_actor_free_td_evaluation_mode(
            direct,
            smoke=False,
            pilot=False,
            score_mode="g_only",
        )


def test_cli_accepts_score_mode_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_actor_free_td_lewm.py",
            "--checkpoint-path",
            "checkpoint.pt",
            "--score-mode",
            "g_only",
        ],
    )

    assert evaluation_cli.parse_args().score_mode == "g_only"
