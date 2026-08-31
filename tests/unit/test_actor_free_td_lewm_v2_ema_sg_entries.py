from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tdwm.training.actor_free_td_lewm_v2_cli import (
    build_actor_free_td_lewm_v2_parser,
    run_actor_free_td_lewm_v2_cli,
)
from tdwm.training.actor_free_td_lewm_v2_ema_sg import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    INITIALIZATION,
    INITIALIZATION_CONTRACT,
    LOCAL_PREDICTION,
    LOCAL_PREDICTION_TARGET,
    LOCAL_PREDICTION_TARGET_GRADIENT,
    METHOD_FAMILY,
    STAGE,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
V1_REVISION = "3c4e62ef2ab72387536433f27ef11bce75477e7e"
V1_CHECKPOINT_SHA256 = {
    "c": "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3",
    "d": "3115fffeb83ba6ae7e0c272913fe7a1ba16d42953b2185f6a3f7b168899d819a",
    "f": "b4de1b511075d763194ad1e332d127cbe390553738162f3a402ef8847bb74fd0",
    "g1": "c224d18fcd8390247f115239c4b2db013479a062438cca92003674c739f3e24b",
    "g2": "1c290f91772b42fdf6824d92832c6fff4e2d8ca3ea08089ff1a41016ea1c2ebe",
    "g3": "b279a85b1dd0816bd5fb9724da490810d470755880639297aa13699c86c2d8fb",
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_ema_sg_entry_resolves_separate_strict_protocol(variant: str) -> None:
    module = importlib.import_module(
        f"tdwm.training.actor_free_td_lewm_v2_ema_sg_{variant}"
    )
    path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_train.yaml"
    )
    protocol = getattr(
        module,
        f"load_actor_free_td_lewm_v2_ema_sg_{variant}_training_protocol",
    )(path)

    assert module.METHOD == f"{METHOD_FAMILY}_{variant}"
    assert module.VARIANT == variant
    assert module.SPEC.method_family == METHOD_FAMILY
    assert module.SPEC.implementation_version == IMPLEMENTATION_VERSION
    assert module.SPEC.stage == STAGE
    assert module.SPEC.initialization == INITIALIZATION
    assert module.SPEC.deployment_checkpoint_version == (DEPLOYMENT_CHECKPOINT_VERSION)
    assert module.SPEC.requires_neighbor_index is (variant == "g1")
    assert module.SPEC.local_prediction == LOCAL_PREDICTION
    assert module.SPEC.local_prediction_target == LOCAL_PREDICTION_TARGET
    assert module.SPEC.local_prediction_target_gradient == (
        LOCAL_PREDICTION_TARGET_GRADIENT
    )

    assert protocol["method_family"] == METHOD_FAMILY
    assert protocol["implementation_version"] == IMPLEMENTATION_VERSION
    assert protocol["stage"] == STAGE
    assert protocol["initialization"] == INITIALIZATION
    assert protocol["seeds"] == [3072]
    assert protocol["source_v1"]["method"] == f"actor_free_td_lewm_v1_{variant}"
    assert protocol["source_v1"]["checkpoint_sha256"] == (V1_CHECKPOINT_SHA256[variant])
    assert protocol["source_v1"]["source_code_revision"] == V1_REVISION
    assert protocol["source_v1"]["optimizer_state"] == "reset"

    objective = protocol["joint_objective"]
    assert objective["local_prediction"] == LOCAL_PREDICTION
    assert objective["local_prediction_target"] == LOCAL_PREDICTION_TARGET
    assert objective["local_prediction_target_gradient"] == (
        LOCAL_PREDICTION_TARGET_GRADIENT
    )
    assert objective["local_prediction_weight"] == 1.0
    assert objective["predicted_context_detach"] is False
    assert objective["hybrid_reduction"] == "sum"
    assert objective["real_td_weight"] == 1.0
    assert objective["predicted_td_weight"] == 1.0

    assert protocol["optimizer"] == {
        "type": "AdamW",
        "world_model_learning_rate": 5e-5,
        "predictor_learning_rate": 1e-4,
        "initialize_state": "fresh",
        "weight_decay": 0.001,
    }
    assert protocol["predictor"]["target_ema_decay"] == 0.995
    assert protocol["predictor"]["target_world_ema_decay"] == 0.995
    assert protocol["training"]["epochs"] == 10
    assert protocol["training"]["optimizer_steps_per_epoch"] == 12_796
    assert protocol["split"]["implementation"] == "prebuilt_exact_indices"

    initialization_contract = protocol["initialization_contract"]
    assert initialization_contract == INITIALIZATION_CONTRACT
    assert module.SPEC.initialization_contract == INITIALIZATION_CONTRACT


def test_ema_sg_overlays_preserve_all_six_v1_method_objectives() -> None:
    expected = {
        "c": "goal_projected_td",
        "d": "goal_value_weighted_td",
        "f": "same_future_different_goal_advantage",
        "g1": "neighbor_action_advantage",
        "g2": "prefix_mean_advantage",
        "g3": "prefix_marginal_advantage",
    }
    for variant, objective in expected.items():
        path = (
            ROOT
            / "configs"
            / "experiment"
            / f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_train.yaml"
        )
        overlay = yaml.safe_load(path.read_text())
        assert (
            overlay["extends"] == "actor_free_td_lewm_v2_ema_sg_common_cube_train.yaml"
        )
        assert overlay["joint_objective"]["objective"] == objective
        assert (
            overlay["source_v1"]["checkpoint_sha256"] == (V1_CHECKPOINT_SHA256[variant])
        )


def test_ema_sg_training_protocol_rejects_initialization_contract_drift() -> None:
    module = importlib.import_module("tdwm.training.actor_free_td_lewm_v2_ema_sg_c")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_c_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v2_ema_sg_c_cube_train.yaml"
    )
    changed = deepcopy(protocol)
    changed["initialization_contract"]["optimizer_state"] = "resume"

    with pytest.raises(ValueError, match="initialization_contract"):
        module.validate_actor_free_td_lewm_v2_ema_sg_c_training_protocol(changed)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ema_sg_has_separate_training_script_and_g1_neighbor_contract(
    variant: str,
) -> None:
    path = ROOT / "scripts" / f"train_actor_free_td_lewm_v2_ema_sg_{variant}.py"
    source = path.read_text()
    assert "run_actor_free_td_lewm_v2_cli(" in source
    assert f'method_label="V2-EMA-SG {variant.upper()}"' in source
    assert f"actor_free_td_lewm_v2_ema_sg_{variant}" in source
    assert ("requires_neighbor_index=True" in source) is (variant == "g1")


def test_ema_sg_cli_only_accepts_matching_v1_initial_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_actor_free_td_lewm_v2_parser(
        method_label="V2-EMA-SG C",
        requires_neighbor_index=False,
    )
    destinations = {action.dest for action in parser._actions}
    assert "initial_v1_checkpoint" in destinations
    assert "initial_v2_checkpoint" not in destinations

    captured: dict[str, object] = {}

    def load_protocol(path: str | Path) -> dict[str, object]:
        captured["loaded"] = path
        return {"method": "actor_free_td_lewm_v2_ema_sg_c"}

    def train(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setenv("TDWM_RUN_ROOT", str(tmp_path))
    result = run_actor_free_td_lewm_v2_cli(
        method_label="V2-EMA-SG C",
        requires_neighbor_index=False,
        load_protocol=load_protocol,
        train=train,
        argv=[
            "--config",
            "ema-sg-c.yaml",
            "--dataset",
            "cube.lance",
            "--seed",
            "3072",
            "--initial-v1-checkpoint",
            "v1-c.pt",
            "--split-indices",
            "split_indices.npz",
            "--smoke",
            "--max-steps",
            "1",
            "--skip-validation",
        ],
    )

    assert result == {"status": "ok"}
    assert captured["initial_v1_checkpoint_path"] == "v1-c.pt"
    assert captured["split_indices_path"] == "split_indices.npz"
    assert captured["seed"] == 3072
    assert captured["resume"] == "auto"
    assert Path(captured["output_dir"]) == (
        tmp_path / "actor_free_td_lewm_v2_ema_sg_c_cube_finetuning"
    )


def test_original_v2_protocol_keeps_its_online_target_identity() -> None:
    module = importlib.import_module("tdwm.training.actor_free_td_lewm_v2_c")
    path = ROOT / "configs" / "experiment" / "actor_free_td_lewm_v2_c_cube_train.yaml"
    protocol = module.load_actor_free_td_lewm_v2_c_training_protocol(path)

    assert protocol["method_family"] == "actor_free_td_lewm_v2"
    assert protocol["implementation_version"] == "v2"
    assert protocol["stage"] == "coupled_hybrid_finetuning"
    assert (
        protocol["joint_objective"]["local_prediction"] == "original_lewm_one_step_mse"
    )
