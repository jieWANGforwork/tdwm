from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.adapters.actor_free_td_lewm_v0_c import METHOD_SPEC as C_SPEC
from tdwm.adapters.actor_free_td_lewm_v0_d import METHOD_SPEC as D_SPEC
from tdwm.adapters.actor_free_td_lewm_v0_f import METHOD_SPEC as F_SPEC
from tdwm.adapters.actor_free_td_lewm_v0_g1 import METHOD_SPEC as G1_SPEC
from tdwm.adapters.actor_free_td_lewm_v0_g2 import METHOD_SPEC as G2_SPEC
from tdwm.adapters.actor_free_td_lewm_v0_g3 import METHOD_SPEC as G3_SPEC
from tdwm.adapters.frozen_actor_free_td_v0_common import OBJECTIVE_VERSION
from tdwm.evaluation.actor_free_td_lewm_v0_c import (
    load_actor_free_td_lewm_v0_c_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v0_d import (
    load_actor_free_td_lewm_v0_d_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v0_f import (
    load_actor_free_td_lewm_v0_f_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v0_g1 import (
    load_actor_free_td_lewm_v0_g1_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v0_g2 import (
    load_actor_free_td_lewm_v0_g2_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v0_g3 import (
    load_actor_free_td_lewm_v0_g3_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v0_common import (
    actor_free_td_v0_output_directory_name,
    configure_frozen_actor_free_td_v0_evaluation_mode,
    validate_frozen_actor_free_td_v0_checkpoint_protocol,
    validate_frozen_actor_free_td_v0_evaluation_protocol,
)

CONFIG_ROOT = Path("configs/experiment")
SOURCE_SHA = "0ce38860a672c4a304d6921c6f07158977bb1d2c8f0eed8a002bb7c89502b579"
METHOD_CASES = [
    (
        "c",
        C_SPEC,
        load_actor_free_td_lewm_v0_c_evaluation_protocol,
    ),
    (
        "d",
        D_SPEC,
        load_actor_free_td_lewm_v0_d_evaluation_protocol,
    ),
    (
        "f",
        F_SPEC,
        load_actor_free_td_lewm_v0_f_evaluation_protocol,
    ),
    (
        "g1",
        G1_SPEC,
        load_actor_free_td_lewm_v0_g1_evaluation_protocol,
    ),
    (
        "g2",
        G2_SPEC,
        load_actor_free_td_lewm_v0_g2_evaluation_protocol,
    ),
    (
        "g3",
        G3_SPEC,
        load_actor_free_td_lewm_v0_g3_evaluation_protocol,
    ),
]


def _config_path(variant: str) -> Path:
    return CONFIG_ROOT / f"actor_free_td_lewm_v0_{variant}_cube_checkpoint_o50.yaml"


@pytest.mark.parametrize(("variant", "spec", "loader"), METHOD_CASES)
def test_each_v0_o50_protocol_loads_as_a_single_symmetric_predictor(
    variant,
    spec,
    loader,
):
    protocol = loader(_config_path(variant))

    assert OBJECTIVE_VERSION == 0
    assert protocol["method"] == spec.method
    assert protocol["variant"] == spec.variant
    assert protocol["predictor"]["objective_version"] == 0
    assert protocol["predictor"]["num_parallel"] == 1
    assert protocol["predictor"]["state_parameterization"] == (
        "symmetric_shared_frozen_lewm_latent"
    )
    assert protocol["predictor"]["action_processing"] == (
        "normalized_raw_lewm_action_block"
    )
    assert protocol["predictor"]["shared_lewm_action_encoder"] is False
    assert protocol["predictor"]["goal_conditioning"] == "task_input"


@pytest.mark.parametrize(("variant", "spec", "loader"), METHOD_CASES)
def test_all_v0_methods_share_the_same_three_evaluation_modes(
    variant,
    spec,
    loader,
):
    protocol = loader(_config_path(variant))

    for score_mode, expected_horizon in (
        ("f_only", 5),
        ("g_only", 1),
        ("f_plus_g", 5),
    ):
        configured = configure_frozen_actor_free_td_v0_evaluation_mode(
            protocol,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
        )
        assert configured["inference_objective"]["score_mode"] == score_mode
        assert configured["planning"]["horizon"] == expected_horizon
        # Training auxiliaries remain metadata and never become planner options.
        assert "neighbor_retrieval" not in configured["planning"]
        assert "prefix_scoring" not in configured["planning"]


def test_formal_v0_protocol_enforces_mode_specific_horizons():
    protocol = load_actor_free_td_lewm_v0_c_evaluation_protocol(_config_path("c"))

    invalid_g_only = deepcopy(protocol)
    invalid_g_only["inference_objective"]["score_mode"] = "g_only"
    with pytest.raises(ValueError, match=r"g_only.*planning\.horizon=1"):
        validate_frozen_actor_free_td_v0_evaluation_protocol(
            invalid_g_only,
            spec=C_SPEC,
        )

    valid_g_only = deepcopy(invalid_g_only)
    valid_g_only["planning"]["horizon"] = 1
    validate_frozen_actor_free_td_v0_evaluation_protocol(
        valid_g_only,
        spec=C_SPEC,
    )

    invalid_f_only = deepcopy(valid_g_only)
    invalid_f_only["inference_objective"]["score_mode"] = "f_only"
    with pytest.raises(ValueError, match=r"f_only.*planning\.horizon=5"):
        validate_frozen_actor_free_td_v0_evaluation_protocol(
            invalid_f_only,
            spec=C_SPEC,
        )


def test_v0_protocol_rejects_parallel_predictors_and_eval_time_candidates():
    protocol = load_actor_free_td_lewm_v0_g1_evaluation_protocol(_config_path("g1"))

    parallel = deepcopy(protocol)
    parallel["predictor"]["num_parallel"] = 2
    with pytest.raises(ValueError, match=r"predictor\.num_parallel must be 1"):
        validate_frozen_actor_free_td_v0_evaluation_protocol(
            parallel,
            spec=G1_SPEC,
        )

    retrieval = deepcopy(protocol)
    retrieval["inference_objective"]["neighbor_retrieval"] = True
    with pytest.raises(ValueError, match="training-only"):
        validate_frozen_actor_free_td_v0_evaluation_protocol(
            retrieval,
            spec=G1_SPEC,
        )


def test_v0_checkpoint_binding_rejects_a_second_predictor():
    protocol = load_actor_free_td_lewm_v0_f_evaluation_protocol(_config_path("f"))
    predictor_config = {
        "method": F_SPEC.method,
        "method_family": "actor_free_td_lewm_v0",
        "variant": F_SPEC.variant,
        "implementation_version": "v0",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        **deepcopy(protocol["predictor"]),
        "task_sampling": deepcopy(protocol["task_sampling"]),
        "joint_objective": deepcopy(protocol["joint_objective"]),
        "pretrained_world_model": deepcopy(protocol["pretrained_world_model"]),
    }
    payload = {
        "method": F_SPEC.method,
        "method_family": "actor_free_td_lewm_v0",
        "variant": F_SPEC.variant,
        "implementation_version": "v0",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "pretrained_world_model_provenance": {
            "source_checkpoint_sha256": SOURCE_SHA,
        },
    }

    validate_frozen_actor_free_td_v0_checkpoint_protocol(
        payload=payload,
        predictor_config=predictor_config,
        protocol=protocol,
        spec=F_SPEC,
    )

    parallel = deepcopy(predictor_config)
    parallel["num_parallel"] = 2
    with pytest.raises(ValueError, match="num_parallel"):
        validate_frozen_actor_free_td_v0_checkpoint_protocol(
            payload=payload,
            predictor_config=parallel,
            protocol=protocol,
            spec=F_SPEC,
        )


def test_v0_output_names_separate_score_and_run_modes():
    protocol = load_actor_free_td_lewm_v0_g3_evaluation_protocol(_config_path("g3"))

    assert (
        actor_free_td_v0_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="g_only",
        )
        == "actor_free_td_lewm_v0_g3_cube_o50_g_only_formal"
    )
    assert (
        actor_free_td_v0_output_directory_name(
            protocol,
            smoke=True,
            pilot=False,
            score_mode="f_plus_g",
        )
        == "actor_free_td_lewm_v0_g3_cube_o50_f_plus_g_smoke"
    )


@pytest.mark.parametrize("variant", ["c", "d", "f", "g1", "g2", "g3"])
def test_v0_evaluation_script_exists_for_every_method(variant):
    assert Path(f"scripts/evaluate_actor_free_td_lewm_v0_{variant}.py").is_file()
