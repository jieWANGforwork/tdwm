from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


@pytest.mark.parametrize("variant", VARIANTS)
def test_v0_training_entry_loads_its_resolved_protocol(variant: str) -> None:
    module = importlib.import_module(
        f"tdwm.training.actor_free_td_lewm_v0_{variant}"
    )
    path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v0_{variant}_cube_train.yaml"
    )

    protocol = getattr(
        module, f"load_actor_free_td_lewm_v0_{variant}_training_protocol"
    )(path)

    assert module.METHOD == f"actor_free_td_lewm_v0_{variant}"
    assert module.VARIANT == variant
    assert module.SPEC.variant == variant
    assert protocol["method"] == module.METHOD
    assert protocol["predictor"]["objective_version"] == 0
    assert protocol["predictor"]["num_parallel"] == 1
    assert protocol["predictor"]["state_parameterization"].startswith("symmetric")
    assert protocol["predictor"]["shared_lewm_action_encoder"] is False
    assert protocol["task_sampling"]["sampling"] == "per_transition_bernoulli"
    assert "exact_half" not in protocol["task_sampling"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_v0_o50_protocol_is_bound_to_matching_training_method(variant: str) -> None:
    training_path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v0_{variant}_cube_train.yaml"
    )
    evaluation_path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v0_{variant}_cube_checkpoint_o50.yaml"
    )
    training = yaml.safe_load(training_path.read_text())
    evaluation = yaml.safe_load(evaluation_path.read_text())

    assert evaluation["method"] == training["method"]
    assert evaluation["variant"] == training["variant"]
    assert evaluation["pretrained_world_model"] == training["pretrained_world_model"]
    assert evaluation["predictor"] == training["predictor"]
    assert evaluation["predictor"]["objective_version"] == 0
    assert evaluation["predictor"]["num_parallel"] == 1
    assert evaluation["predictor"]["bootstrap_action"] == "dataset_next_action"
    assert evaluation["task_sampling"] == training["task_sampling"]
    assert evaluation["joint_objective"] == training["joint_objective"]
    assert evaluation["planning"]["horizon"] == 5
    assert evaluation["inference_objective"]["score_mode"] == "f_plus_g"
    assert evaluation["inference_objective"][
        "training_only_auxiliary_used_at_evaluation"
    ] is False


def test_only_v0_g1_training_entry_requires_neighbor_index() -> None:
    for variant in VARIANTS:
        module = importlib.import_module(
            f"tdwm.training.actor_free_td_lewm_v0_{variant}"
        )
        assert module.SPEC.requires_neighbor_index is (variant == "g1")
