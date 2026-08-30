from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    actor_free_td_v2_output_directory_name,
    configure_actor_free_td_v2_evaluation_mode,
    validate_actor_free_td_v2_evaluation_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def _module(variant: str):
    return importlib.import_module(f"tdwm.evaluation.actor_free_td_lewm_v2_{variant}")


def _config(variant: str) -> Path:
    return (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_{variant}_cube_checkpoint_o50.yaml"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_v2_o50_protocol_resolves_its_training_overlay(variant: str) -> None:
    module = _module(variant)
    protocol = getattr(
        module, f"load_actor_free_td_lewm_v2_{variant}_evaluation_protocol"
    )(_config(variant))

    assert protocol["method"] == f"actor_free_td_lewm_v2_{variant}"
    assert protocol["method_family"] == "actor_free_td_lewm_v2"
    assert protocol["variant"] == variant
    assert protocol["implementation_version"] == "v2"
    assert protocol["stage"] == "planner_evaluation"
    assert protocol["source_v1"]["method"] == f"actor_free_td_lewm_v1_{variant}"
    assert protocol["predictor"]["state_parameterization"] == (
        "coupled_online_lewm_latent"
    )
    assert protocol["predictor"]["action_processing"] == (
        "online_shared_lewm_action_encoder"
    )
    assert protocol["joint_objective"]["per_transition_td_reduction"] == (
        "feature_sum"
    )
    assert protocol["joint_objective"]["batch_td_reduction"] == "transition_mean"
    assert protocol["source_artifacts"]["split_file_sha256"] == (
        "4594afb3603b4258431ff9076c82acbe3ddcaccb277940b825a99017ce83d830"
    )
    assert protocol["inference_objective"]["deployed_world_model"] == (
        "online_v2_world_model"
    )
    assert protocol["inference_objective"]["target_modules_used_at_evaluation"] is False


@pytest.mark.parametrize("variant", VARIANTS)
def test_v2_scripts_and_all_three_score_modes_exist(variant: str) -> None:
    module = _module(variant)
    protocol = getattr(
        module, f"load_actor_free_td_lewm_v2_{variant}_evaluation_protocol"
    )(_config(variant))
    assert (ROOT / "scripts" / f"evaluate_actor_free_td_lewm_v2_{variant}.py").is_file()
    for score_mode, horizon in (("f_only", 5), ("g_only", 1), ("f_plus_g", 5)):
        configured = configure_actor_free_td_v2_evaluation_mode(
            protocol,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
        )
        assert configured["planning"]["horizon"] == horizon
        assert configured["inference_objective"]["score_mode"] == score_mode


def test_v2_protocol_rejects_v1_identity_and_target_inference() -> None:
    module = _module("c")
    protocol = module.load_actor_free_td_lewm_v2_c_evaluation_protocol(_config("c"))

    v1 = deepcopy(protocol)
    v1["implementation_version"] = "v1"
    with pytest.raises(ValueError, match="implementation_version"):
        validate_actor_free_td_v2_evaluation_protocol(v1, spec=module.METHOD_SPEC)

    target = deepcopy(protocol)
    target["inference_objective"]["target_modules_used_at_evaluation"] = True
    with pytest.raises(ValueError, match="target_modules_used_at_evaluation"):
        validate_actor_free_td_v2_evaluation_protocol(target, spec=module.METHOD_SPEC)

    artifacts = deepcopy(protocol)
    artifacts["source_artifacts"]["column_normalization_sha256"] = "0" * 64
    with pytest.raises(
        ValueError, match="source_artifacts.column_normalization_sha256"
    ):
        validate_actor_free_td_v2_evaluation_protocol(
            artifacts, spec=module.METHOD_SPEC
        )


def test_v2_output_directory_names_never_collide_with_v1() -> None:
    module = _module("g3")
    protocol = module.load_actor_free_td_lewm_v2_g3_evaluation_protocol(_config("g3"))
    assert actor_free_td_v2_output_directory_name(
        protocol,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g",
    ) == "actor_free_td_lewm_v2_g3_cube_o50_f_plus_g_formal"
