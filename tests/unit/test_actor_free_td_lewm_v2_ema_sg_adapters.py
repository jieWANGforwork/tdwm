from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.adapters.actor_free_td_lewm_v2_c import METHOD_SPEC as V2_C_SPEC
from tdwm.adapters.actor_free_td_lewm_v2_common import (
    validate_actor_free_td_v2_payload,
)
from tdwm.adapters.actor_free_td_lewm_v2_ema_sg_c import (
    METHOD_SPEC as EMA_SG_C_SPEC,
)
from tdwm.adapters.actor_free_td_lewm_v2_ema_sg_common import (
    ActorFreeTDLeWMV2EMASG,
)
from tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_c import (
    load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol,
    validate_actor_free_td_lewm_v2_ema_sg_c_checkpoint_protocol,
)
from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1


def _payload() -> dict:
    protocol = load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        Path(
            "configs/experiment/"
            "actor_free_td_lewm_v2_ema_sg_c_cube_checkpoint_o50.yaml"
        )
    )
    predictor = ActorFreeTDJEPAPredictorV1()
    return {
        "method": EMA_SG_C_SPEC.method,
        "method_family": EMA_SG_C_SPEC.method_family,
        "variant": "c",
        "implementation_version": EMA_SG_C_SPEC.implementation_version,
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "world_model_state_dict": {},
        "target_world_model_state_dict": {},
        "world_model_config": {
            "action_encoder": {"input_dim": 25, "emb_dim": 192},
        },
        "predictor_state_dict": predictor.state_dict(),
        "target_predictor_state_dict": predictor.make_target().state_dict(),
        "predictor_config": {
            "method": EMA_SG_C_SPEC.method,
            "method_family": EMA_SG_C_SPEC.method_family,
            "variant": "c",
            "implementation_version": EMA_SG_C_SPEC.implementation_version,
            "objective_version": 0,
            "deployment_checkpoint_version": 1,
            **deepcopy(protocol["predictor"]),
            "task_sampling": deepcopy(protocol["task_sampling"]),
            "joint_objective": deepcopy(protocol["joint_objective"]),
            "source_v1": deepcopy(protocol["source_v1"]),
            "source_artifacts": deepcopy(protocol["source_artifacts"]),
        },
        "source_v1_provenance": {
            "checkpoint_path": "/external/v1-c.pt",
            "checkpoint_sha256": protocol["source_v1"]["checkpoint_sha256"],
            "source_epoch": 10,
            "source_global_step": 127_960,
            "optimizer_state_loaded": False,
            "target_world_initialization": "copy_of_v1_online_world_model",
        },
    }


def test_ema_sg_payload_strictly_locks_its_identity_and_local_target() -> None:
    payload = _payload()
    config = validate_actor_free_td_v2_payload(payload, spec=EMA_SG_C_SPEC)

    assert config["method_family"] == "actor_free_td_lewm_v2_ema_sg"
    assert config["implementation_version"] == "v2_ema_sg"
    assert config["joint_objective"]["local_prediction"] == (
        "ema_target_lewm_one_step_mse"
    )
    assert config["joint_objective"]["local_prediction_target"] == (
        "ema_world_model_next_latent"
    )
    assert config["joint_objective"]["local_prediction_target_gradient"] == (
        "stop_gradient"
    )
    validate_actor_free_td_lewm_v2_ema_sg_c_checkpoint_protocol(
        payload=payload,
        predictor_config=payload["predictor_config"],
        protocol=load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
            Path(
                "configs/experiment/"
                "actor_free_td_lewm_v2_ema_sg_c_cube_checkpoint_o50.yaml"
            )
        ),
    )

    online_target = deepcopy(payload)
    online_target["predictor_config"]["joint_objective"]["local_prediction"] = (
        "original_lewm_one_step_mse"
    )
    with pytest.raises(ValueError, match="local_prediction"):
        validate_actor_free_td_v2_payload(online_target, spec=EMA_SG_C_SPEC)

    with pytest.raises(ValueError, match="method"):
        validate_actor_free_td_v2_payload(payload, spec=V2_C_SPEC)


def test_ema_sg_has_an_explicit_deployment_wrapper_identity() -> None:
    assert ActorFreeTDLeWMV2EMASG.method_family == (
        "actor_free_td_lewm_v2_ema_sg"
    )
    assert ActorFreeTDLeWMV2EMASG.implementation_version == "v2_ema_sg"
