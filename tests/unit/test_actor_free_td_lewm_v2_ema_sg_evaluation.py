from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_common import (
    actor_free_td_v2_ema_sg_output_directory_name,
    configure_actor_free_td_v2_ema_sg_evaluation_mode,
    validate_actor_free_td_v2_ema_sg_evaluation_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def _module(variant: str):
    return importlib.import_module(
        f"tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_{variant}"
    )


def _config(variant: str) -> Path:
    return (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_checkpoint_o50.yaml"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_ema_sg_o50_protocol_has_a_separate_strict_identity(
    variant: str,
) -> None:
    module = _module(variant)
    protocol = getattr(
        module,
        f"load_actor_free_td_lewm_v2_ema_sg_{variant}_evaluation_protocol",
    )(_config(variant))

    assert protocol["method"] == f"actor_free_td_lewm_v2_ema_sg_{variant}"
    assert protocol["method_family"] == "actor_free_td_lewm_v2_ema_sg"
    assert protocol["implementation_version"] == "v2_ema_sg"
    assert protocol["stage"] == "planner_evaluation"
    assert protocol["initialization"] == "corresponding_v1_deployment_finetune"
    assert protocol["joint_objective"]["local_prediction"] == (
        "ema_target_lewm_one_step_mse"
    )
    assert protocol["joint_objective"]["local_prediction_target"] == (
        "ema_world_model_next_latent"
    )
    assert protocol["joint_objective"]["local_prediction_target_gradient"] == (
        "stop_gradient"
    )
    assert protocol["inference_objective"]["deployed_world_model"] == (
        "online_v2_ema_sg_world_model"
    )
    assert protocol["inference_objective"]["deployed_predictor"] == (
        "online_v2_ema_sg_predictor"
    )
    assert (
        ROOT
        / "scripts"
        / f"evaluate_actor_free_td_lewm_v2_ema_sg_{variant}.py"
    ).is_file()

    for score_mode, horizon in (("f_only", 5), ("g_only", 1), ("f_plus_g", 5)):
        configured = configure_actor_free_td_v2_ema_sg_evaluation_mode(
            protocol,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
        )
        assert configured["planning"]["horizon"] == horizon


def test_ema_sg_protocol_rejects_original_v2_target_contract() -> None:
    module = _module("c")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        _config("c")
    )
    original = deepcopy(protocol)
    original["joint_objective"]["local_prediction"] = (
        "original_lewm_one_step_mse"
    )
    with pytest.raises(ValueError, match="local_prediction"):
        validate_actor_free_td_v2_ema_sg_evaluation_protocol(
            original,
            spec=module.METHOD_SPEC,
        )


def test_ema_sg_output_names_do_not_collide_with_v2() -> None:
    module = _module("g3")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_g3_evaluation_protocol(
        _config("g3")
    )
    assert actor_free_td_v2_ema_sg_output_directory_name(
        protocol,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g",
    ) == "actor_free_td_lewm_v2_ema_sg_g3_cube_o50_f_plus_g_formal"
