from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
SERVER_PRETRAINED_SHA256 = (
    "198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257"
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_training_entry_loads_its_resolved_protocol(variant: str) -> None:
    module = importlib.import_module(
        f"tdwm.training.actor_free_td_lewm_v1_{variant}"
    )
    path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v1_{variant}_cube_train.yaml"
    )

    protocol = getattr(
        module, f"load_actor_free_td_lewm_v1_{variant}_training_protocol"
    )(path)

    assert module.METHOD == f"actor_free_td_lewm_v1_{variant}"
    assert module.VARIANT == variant
    assert module.SPEC.variant == variant
    assert protocol["method"] == module.METHOD
    assert protocol["predictor"]["objective_version"] == 0
    assert protocol["predictor"]["num_parallel"] == 1
    assert protocol["predictor"]["state_parameterization"].startswith("symmetric")
    assert protocol["predictor"]["raw_action_dim"] == 25
    assert protocol["predictor"]["action_dim"] == 192
    assert protocol["predictor"]["action_embedding_dim"] == 192
    assert protocol["predictor"]["action_processing"] == (
        "frozen_shared_lewm_action_encoder"
    )
    assert protocol["predictor"]["shared_lewm_action_encoder"] is True
    assert protocol["predictor"]["action_encoder_trainable"] is False
    assert protocol["predictor"]["action_encoder_source"] == (
        "world_model.action_encoder"
    )
    assert protocol["task_sampling"]["sampling"] == "per_transition_bernoulli"
    assert "exact_half" not in protocol["task_sampling"]
    assert (
        protocol["pretrained_world_model"]["checkpoint_sha256"]
        == SERVER_PRETRAINED_SHA256
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_o50_protocol_is_bound_to_matching_training_method(variant: str) -> None:
    training_path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v1_{variant}_cube_train.yaml"
    )
    evaluation_path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v1_{variant}_cube_checkpoint_o50.yaml"
    )
    training = yaml.safe_load(training_path.read_text())
    evaluation = yaml.safe_load(evaluation_path.read_text())

    assert evaluation["method"] == training["method"]
    assert evaluation["variant"] == training["variant"]
    assert evaluation["pretrained_world_model"] == training["pretrained_world_model"]
    assert (
        evaluation["pretrained_world_model"]["checkpoint_sha256"]
        == SERVER_PRETRAINED_SHA256
    )
    assert evaluation["predictor"] == training["predictor"]
    assert evaluation["predictor"]["objective_version"] == 0
    assert evaluation["predictor"]["num_parallel"] == 1
    assert evaluation["predictor"]["bootstrap_action"] == "dataset_next_action"
    assert evaluation["world"]["env_name"] == "swm/OGBCube-v0"
    assert evaluation["task_sampling"] == training["task_sampling"]
    assert evaluation["joint_objective"] == training["joint_objective"]
    assert evaluation["planning"]["horizon"] == 5
    assert evaluation["inference_objective"]["score_mode"] == "f_plus_g"
    assert evaluation["inference_objective"][
        "training_only_auxiliary_used_at_evaluation"
    ] is False


def test_only_v1_g1_training_entry_requires_neighbor_index() -> None:
    for variant in VARIANTS:
        module = importlib.import_module(
            f"tdwm.training.actor_free_td_lewm_v1_{variant}"
        )
        assert module.SPEC.requires_neighbor_index is (variant == "g1")


def test_v1_training_protocol_rejects_non_task_conditioned_predictor() -> None:
    module = importlib.import_module("tdwm.training.actor_free_td_lewm_v1_c")
    path = (
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c_cube_train.yaml"
    )
    protocol = deepcopy(yaml.safe_load(path.read_text()))
    protocol["predictor"]["goal_conditioning"] = "none"

    with pytest.raises(ValueError, match="predictor.goal_conditioning"):
        module.validate_actor_free_td_lewm_v1_c_training_protocol(protocol)


def test_v1_training_protocol_rejects_alternate_epoch_factorization() -> None:
    module = importlib.import_module("tdwm.training.actor_free_td_lewm_v1_c")
    path = (
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c_cube_train.yaml"
    )
    protocol = deepcopy(yaml.safe_load(path.read_text()))
    protocol["training"].update(
        epochs=20,
        scheduler_epochs=20,
        optimizer_steps_per_epoch=6_398,
    )

    with pytest.raises(ValueError, match="training.epochs"):
        module.validate_actor_free_td_lewm_v1_c_training_protocol(protocol)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("training", "precision", "32-true"),
        ("training", "gradient_clip_norm", 0.5),
        ("training", "checkpoint_every_epochs", 2),
        ("optimizer", "type", "SGD"),
        ("scheduler", "type", "constant"),
        ("scheduler", "interval", "epoch"),
    ),
)
def test_v1_training_protocol_rejects_runtime_contract_drift(
    section: str,
    key: str,
    value,
) -> None:
    module = importlib.import_module("tdwm.training.actor_free_td_lewm_v1_c")
    path = (
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c_cube_train.yaml"
    )
    protocol = deepcopy(yaml.safe_load(path.read_text()))
    protocol[section][key] = value

    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        module.validate_actor_free_td_lewm_v1_c_training_protocol(protocol)
