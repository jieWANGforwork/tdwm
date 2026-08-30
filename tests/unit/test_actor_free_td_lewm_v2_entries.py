from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from tdwm.training.actor_free_td_lewm_v2_cli import run_actor_free_td_lewm_v2_cli

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
def test_v2_entry_resolves_common_coupled_hybrid_protocol(variant: str) -> None:
    module = importlib.import_module(f"tdwm.training.actor_free_td_lewm_v2_{variant}")
    path = (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_{variant}_cube_train.yaml"
    )
    protocol = getattr(
        module, f"load_actor_free_td_lewm_v2_{variant}_training_protocol"
    )(path)

    assert module.METHOD == f"actor_free_td_lewm_v2_{variant}"
    assert module.VARIANT == variant
    assert module.SPEC.requires_neighbor_index is (variant == "g1")
    assert protocol["method_family"] == "actor_free_td_lewm_v2"
    assert protocol["implementation_version"] == "v2"
    assert protocol["stage"] == "coupled_hybrid_finetuning"
    assert protocol["initialization"] == "corresponding_v1_deployment_finetune"
    assert protocol["seeds"] == [3072]
    assert protocol["sequence"]["num_steps"] == 19
    assert protocol["sequence"]["history_frames"] == 3
    assert protocol["loader"]["sampling_unit"] == "sequence_clip"
    assert protocol["world_model"]["online"]["full_lewm_trainable"] is True
    assert protocol["world_model"]["target"]["tracks_action_encoder"] is True
    assert protocol["predictor"]["action_processing"] == (
        "online_shared_lewm_action_encoder"
    )
    assert protocol["predictor"]["raw_action_dim"] == 25
    assert protocol["predictor"]["action_embedding_dim"] == 192
    assert protocol["predictor"]["state_dim"] == 192
    assert protocol["predictor"]["task_dim"] == 192
    assert protocol["joint_objective"]["predicted_context_detach"] is False
    assert protocol["joint_objective"]["hybrid_reduction"] == "sum"
    assert protocol["loss"]["sigreg"]["weight"] == 0.09
    assert protocol["optimizer"]["world_model_learning_rate"] == 5e-5
    assert protocol["optimizer"]["predictor_learning_rate"] == 1e-4
    assert protocol["training"]["epochs"] == 10
    assert protocol["training"]["optimizer_steps_per_epoch"] == 12_796
    assert protocol["source_v1"]["method"] == (f"actor_free_td_lewm_v1_{variant}")
    assert protocol["source_v1"]["checkpoint_sha256"] == (V1_CHECKPOINT_SHA256[variant])
    assert protocol["source_v1"]["source_code_revision"] == V1_REVISION


def test_v2_overlays_are_thin_and_preserve_v1_joint_objectives() -> None:
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
            / f"actor_free_td_lewm_v2_{variant}_cube_train.yaml"
        )
        overlay = yaml.safe_load(path.read_text())
        assert overlay["extends"] == "actor_free_td_lewm_v2_common_cube_train.yaml"
        assert overlay["joint_objective"]["objective"] == objective


@pytest.mark.parametrize("requires_neighbor_index", (False, True))
def test_v2_cli_dispatches_checkpoint_and_bound_artifacts(
    requires_neighbor_index: bool,
) -> None:
    captured: dict[str, object] = {}

    def load_protocol(path: str | Path) -> dict[str, object]:
        captured["loaded"] = path
        return {"method": "actor_free_td_lewm_v2_g1"}

    def train(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    argv = [
        "--config",
        "v2.yaml",
        "--dataset",
        "cube.lance",
        "--output-dir",
        "v2-output",
        "--seed",
        "3072",
        "--initial-v1-checkpoint",
        "v1.pt",
        "--split-indices",
        "split_indices.npz",
        "--smoke",
        "--max-steps",
        "1",
        "--skip-validation",
    ]
    if requires_neighbor_index:
        argv.extend(("--neighbor-index", "neighbors"))

    result = run_actor_free_td_lewm_v2_cli(
        method_label="V2 test",
        requires_neighbor_index=requires_neighbor_index,
        load_protocol=load_protocol,
        train=train,
        argv=argv,
    )

    assert result == {"status": "ok"}
    assert captured["initial_v1_checkpoint_path"] == "v1.pt"
    assert captured["split_indices_path"] == "split_indices.npz"
    assert captured["seed"] == 3072
    assert captured["smoke"] is True
    assert captured.get("neighbor_index_path") == (
        "neighbors" if requires_neighbor_index else None
    )
