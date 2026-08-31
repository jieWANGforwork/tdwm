from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v2_c import (
    METHOD_SPEC,
    load_actor_free_td_lewm_v2_c_checkpoint,
)
from tdwm.adapters.actor_free_td_lewm_v2_common import (
    ActorFreeTDLeWMV2,
    validate_actor_free_td_v2_payload,
)
from tdwm.adapters.actor_free_td_lewm_v2_g1 import METHOD_SPEC as G1_METHOD_SPEC
from tdwm.evaluation.actor_free_td_lewm_v2_c import (
    load_actor_free_td_lewm_v2_c_evaluation_protocol,
    validate_actor_free_td_lewm_v2_c_checkpoint_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v2_common import (
    load_actor_free_td_v2_evaluation_protocol,
)
from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1


class TinyActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class TinyWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_encoder = TinyActionEncoder()


def _payload() -> dict:
    protocol = load_actor_free_td_lewm_v2_c_evaluation_protocol(
        Path("configs/experiment/actor_free_td_lewm_v2_c_cube_checkpoint_o50.yaml")
    )
    online_world = TinyWorld()
    target_world = TinyWorld()
    predictor = ActorFreeTDJEPAPredictorV1()
    return {
        "method": METHOD_SPEC.method,
        "method_family": "actor_free_td_lewm_v2",
        "variant": METHOD_SPEC.variant,
        "implementation_version": "v2",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "world_model_state_dict": online_world.state_dict(),
        "target_world_model_state_dict": target_world.state_dict(),
        "world_model_config": {
            "_target_": "tests.TinyWorld",
            "action_encoder": {"input_dim": 25, "emb_dim": 192},
        },
        "predictor_state_dict": predictor.state_dict(),
        "target_predictor_state_dict": predictor.make_target().state_dict(),
        "predictor_config": {
            "method": METHOD_SPEC.method,
            "method_family": "actor_free_td_lewm_v2",
            "variant": METHOD_SPEC.variant,
            "implementation_version": "v2",
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


def test_v2_payload_is_explicitly_v2_and_requires_both_ema_pairs() -> None:
    payload = _payload()
    config = validate_actor_free_td_v2_payload(payload, spec=METHOD_SPEC)

    assert config["implementation_version"] == "v2"
    assert config["state_parameterization"] == "coupled_online_lewm_latent"
    assert config["source_v1"]["method"] == "actor_free_td_lewm_v1_c"
    assert config["joint_objective"]["per_transition_td_reduction"] == (
        "feature_sum"
    )
    assert config["joint_objective"]["batch_td_reduction"] == "transition_mean"
    assert config["source_artifacts"]["split_file_sha256"] == (
        "4594afb3603b4258431ff9076c82acbe3ddcaccb277940b825a99017ce83d830"
    )

    v1 = deepcopy(payload)
    v1["method_family"] = "actor_free_td_lewm_v1"
    with pytest.raises(ValueError, match="method_family"):
        validate_actor_free_td_v2_payload(v1, spec=METHOD_SPEC)

    for key in ("target_world_model_state_dict", "target_predictor_state_dict"):
        missing = deepcopy(payload)
        missing.pop(key)
        with pytest.raises(ValueError, match=key):
            validate_actor_free_td_v2_payload(missing, spec=METHOD_SPEC)

    wrong_artifacts = deepcopy(payload)
    wrong_artifacts["predictor_config"]["source_artifacts"][
        "split_file_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="source_artifacts.split_file_sha256"):
        validate_actor_free_td_v2_payload(wrong_artifacts, spec=METHOD_SPEC)


def test_v2_loader_uses_online_modules_but_strictly_audits_targets(tmp_path) -> None:
    payload = _payload()
    checkpoint = tmp_path / "v2-c.pt"
    torch.save(payload, checkpoint)

    worlds = [TinyWorld(), TinyWorld()]
    with patch("hydra.utils.instantiate", side_effect=worlds):
        online_world, online_g, config, restored = (
            load_actor_free_td_lewm_v2_c_checkpoint(checkpoint)
        )

    assert isinstance(online_world, TinyWorld)
    assert isinstance(online_g, ActorFreeTDJEPAPredictorV1)
    assert config["method_family"] == "actor_free_td_lewm_v2"
    assert restored["implementation_version"] == "v2"
    assert not online_world.training
    assert not online_g.training
    assert not any(parameter.requires_grad for parameter in online_world.parameters())
    assert not any(parameter.requires_grad for parameter in online_g.parameters())

    malformed = deepcopy(payload)
    malformed["target_world_model_state_dict"] = {"bad": torch.zeros(1)}
    torch.save(malformed, checkpoint)
    with patch("hydra.utils.instantiate", side_effect=[TinyWorld(), TinyWorld()]):
        with pytest.raises(RuntimeError):
            load_actor_free_td_lewm_v2_c_checkpoint(checkpoint)


def test_v2_checkpoint_source_artifacts_must_exactly_match_protocol() -> None:
    payload = _payload()
    protocol = load_actor_free_td_lewm_v2_c_evaluation_protocol(
        Path("configs/experiment/actor_free_td_lewm_v2_c_cube_checkpoint_o50.yaml")
    )
    validate_actor_free_td_lewm_v2_c_checkpoint_protocol(
        payload=payload,
        predictor_config=payload["predictor_config"],
        protocol=protocol,
    )

    changed = deepcopy(payload["predictor_config"])
    changed["source_artifacts"]["train_indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source_artifacts differs"):
        validate_actor_free_td_lewm_v2_c_checkpoint_protocol(
            payload=payload,
            predictor_config=changed,
            protocol=protocol,
        )


def test_v2_intermediate_o50_checkpoint_is_strictly_epoch_bound() -> None:
    payload = _payload()
    payload["epoch"] = 3
    payload["global_step"] = 38_388
    protocol = load_actor_free_td_lewm_v2_c_evaluation_protocol(
        Path("configs/experiment/actor_free_td_lewm_v2_c_cube_checkpoint_o50.yaml")
    )

    validate_actor_free_td_lewm_v2_c_checkpoint_protocol(
        payload=payload,
        predictor_config=payload["predictor_config"],
        protocol=protocol,
        require_formal_completion=False,
        expected_checkpoint_epoch=3,
    )

    payload["global_step"] += 1
    with pytest.raises(ValueError, match="global_step"):
        validate_actor_free_td_lewm_v2_c_checkpoint_protocol(
            payload=payload,
            predictor_config=payload["predictor_config"],
            protocol=protocol,
            require_formal_completion=False,
            expected_checkpoint_epoch=3,
        )


def test_v2_g1_adapter_locks_the_archived_neighbor_count() -> None:
    protocol = load_actor_free_td_v2_evaluation_protocol(
        Path("configs/experiment/actor_free_td_lewm_v2_g1_cube_checkpoint_o50.yaml"),
        spec=G1_METHOD_SPEC,
    )
    objective = deepcopy(protocol["joint_objective"])
    G1_METHOD_SPEC.validate_method_config({"joint_objective": objective})

    objective["neighbors_per_anchor"] = 4
    with pytest.raises(ValueError, match="neighbors_per_anchor"):
        G1_METHOD_SPEC.validate_method_config({"joint_objective": objective})


def test_v2_has_an_explicit_deployment_wrapper_not_a_v1_identity() -> None:
    assert ActorFreeTDLeWMV2.method_family == "actor_free_td_lewm_v2"
    assert ActorFreeTDLeWMV2.implementation_version == "v2"
