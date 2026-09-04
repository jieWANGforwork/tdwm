from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tdwm.training.actor_free_td_lewm_v1_c3 import (
    FORMAL_OPTIMIZER_STEPS,
    METHOD,
    _build_v1_c3_training_module,
    _deployment_payload,
    _validation_summary,
    load_actor_free_td_lewm_v1_c3_training_protocol,
    sample_v1_c3_context,
    validate_actor_free_td_lewm_v1_c3_training_protocol,
)
from tdwm.training.frozen_actor_free_td_v1 import _state_dict_sha256
from tdwm.training.frozen_latent_store import (
    EncodedRowBatch,
    FrozenLatentStore,
    FrozenLatentStoreSpec,
    build_frozen_latent_store,
    normalize_actions,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment/actor_free_td_lewm_v1_c3_cube_train.yaml"


def _store(tmp_path: Path) -> FrozenLatentStore:
    total_rows = 162
    episode_ids = np.repeat(np.arange(2, dtype=np.int64), 81)
    latents = np.arange(total_rows * 192, dtype=np.float32).reshape(total_rows, 192)
    raw_actions = np.zeros((total_rows, 5), dtype=np.float32)
    raw_actions[[80, 161]] = np.nan
    normalized = normalize_actions(
        raw_actions,
        mean=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
    )
    root = tmp_path / "store"
    spec = FrozenLatentStoreSpec(
        total_rows=total_rows,
        embed_dim=192,
        frame_skip=5,
        history_frames=3,
        action_dim=5,
        pretrained_checkpoint_sha256="1" * 64,
        dataset_source_sha256="2" * 64,
        column_normalization_sha256="3" * 64,
        git_revision="4" * 40,
    )
    build_frozen_latent_store(
        root,
        spec=spec,
        encoded_batches=[EncodedRowBatch(np.arange(total_rows), latents)],
        normalized_actions=normalized,
        episode_ids=episode_ids,
        source_metadata={"fixture": True},
    )
    return FrozenLatentStore(
        root,
        expected_checkpoint_sha256="1" * 64,
        expected_dataset_source_sha256="2" * 64,
        expected_column_normalization_sha256="3" * 64,
        expected_frame_skip=5,
        expected_history_frames=3,
        expected_embed_dim=192,
        expected_action_dim=5,
    )


def _small_protocol() -> dict:
    protocol = load_actor_free_td_lewm_v1_c3_training_protocol(CONFIG)
    protocol["state_critic"]["hidden_dim"] = 16
    protocol["state_critic"]["embedding_dim"] = 8
    return protocol


def test_c3_protocol_locks_frozen_parent_rp1_units_and_budget():
    protocol = load_actor_free_td_lewm_v1_c3_training_protocol(CONFIG)

    assert protocol["method"] == METHOD
    assert protocol["source_v1_c"]["source_epoch"] == 10
    assert protocol["source_v1_c"]["parameter_state"] == (
        "strict_all_model_parameters_frozen"
    )
    assert protocol["state_critic"]["action_input"] == "none"
    assert protocol["state_critic"]["actor"] == "none"
    assert protocol["state_critic"]["block_primitive_steps"] == 5
    assert protocol["state_critic"]["backup_horizon_primitive_steps"] == 50
    assert protocol["training"]["total_optimizer_steps"] == FORMAL_OPTIMIZER_STEPS
    assert protocol["loader"]["batch_size"] == 1024

    changed = deepcopy(protocol)
    changed["goal_sampling"]["cross_episode_probability"] = 0.3
    with pytest.raises(ValueError, match="cross_episode_probability"):
        validate_actor_free_td_lewm_v1_c3_training_protocol(changed)

    changed = deepcopy(protocol)
    changed["source_v1_c"]["parameter_state"] = "online_g_only"
    with pytest.raises(ValueError, match="parameter_state"):
        validate_actor_free_td_lewm_v1_c3_training_protocol(changed)


def test_context_sampling_keeps_rows_in_episode_and_converts_to_primitive_steps(
    tmp_path: Path,
):
    store = _store(tmp_path)
    rows = torch.tensor([0, 81], dtype=torch.int64)
    ends = torch.tensor([75, 156], dtype=torch.int64)
    context = sample_v1_c3_context(
        store,
        rows,
        ends,
        backup_horizon_primitive_steps=50,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
    )

    assert torch.equal(context.n_eff_primitive, torch.tensor([50, 50]))
    assert torch.all(context.delta_primitive.remainder(5) == 0)
    assert torch.all((context.delta_primitive >= 5) & (context.delta_primitive <= 75))
    assert torch.equal(context.successor_rows, torch.tensor([50, 131]))
    assert torch.equal(
        torch.from_numpy(store.episode_ids[context.goal_rows.numpy()]),
        torch.tensor([0, 1]),
    )
    torch.testing.assert_close(
        context.goal,
        torch.from_numpy(np.array(store.latents[context.goal_rows.numpy()], copy=True)),
    )


def test_training_module_optimizes_only_state_v_and_ema_is_frozen(tmp_path: Path):
    store = _store(tmp_path)
    module = _build_v1_c3_training_module(
        _small_protocol(),
        store=store,
        data_generator=torch.Generator().manual_seed(1),
        goal_generator=torch.Generator().manual_seed(2),
        validation_goal_generator=torch.Generator().manual_seed(3),
    )
    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in module.critic.parameters()}
    assert optimized.isdisjoint(
        {id(parameter) for parameter in module.target_critic.parameters()}
    )
    assert all(parameter.requires_grad for parameter in module.critic.parameters())
    assert all(
        not parameter.requires_grad for parameter in module.target_critic.parameters()
    )

    module.log_dict = lambda *_args, **_kwargs: None
    batch = {
        "state": torch.from_numpy(np.array(store.latents[[0, 81]], copy=True)),
        "global_row": torch.tensor([0, 81], dtype=torch.int64),
        "goal_future_end_row": torch.tensor([75, 156], dtype=torch.int64),
    }
    loss = module._forward_loss(batch, "train")
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in module.critic.parameters())
    assert all(
        parameter.grad is None for parameter in module.target_critic.parameters()
    )


def test_c3_lightning_checkpoint_restores_all_sampling_rng_streams(tmp_path: Path):
    store = _store(tmp_path)
    source = _build_v1_c3_training_module(
        _small_protocol(),
        store=store,
        data_generator=torch.Generator().manual_seed(11),
        goal_generator=torch.Generator().manual_seed(12),
        validation_goal_generator=torch.Generator().manual_seed(13),
    )
    torch.rand(7, generator=source.data_generator)
    torch.rand(8, generator=source.goal_generator)
    checkpoint: dict[str, object] = {}
    source.on_save_checkpoint(checkpoint)
    expected_data = torch.rand(4, generator=source.data_generator)
    expected_goal = torch.rand(4, generator=source.goal_generator)

    resumed = _build_v1_c3_training_module(
        _small_protocol(),
        store=store,
        data_generator=torch.Generator().manual_seed(99),
        goal_generator=torch.Generator().manual_seed(99),
        validation_goal_generator=torch.Generator().manual_seed(99),
    )
    resumed.on_load_checkpoint(checkpoint)
    assert torch.equal(torch.rand(4, generator=resumed.data_generator), expected_data)
    assert torch.equal(torch.rand(4, generator=resumed.goal_generator), expected_goal)


def test_deployment_payload_preserves_all_parent_states_byte_exact(tmp_path: Path):
    store = _store(tmp_path)
    protocol = _small_protocol()
    module = _build_v1_c3_training_module(
        protocol,
        store=store,
        data_generator=torch.Generator().manual_seed(1),
        goal_generator=torch.Generator().manual_seed(2),
        validation_goal_generator=torch.Generator().manual_seed(3),
    )
    parent = {
        "world_model_state_dict": {"w": torch.randn(3)},
        "world_model_config": {"_target_": "fixture.World"},
        "predictor_state_dict": {"g": torch.randn(4)},
        "target_predictor_state_dict": {"g": torch.randn(4)},
        "predictor_config": {"method": "actor_free_td_lewm_v1_c"},
        "pretrained_world_model_provenance": {"frozen": True},
    }
    provenance = {
        "world_model_state_sha256": _state_dict_sha256(
            parent["world_model_state_dict"]
        ),
        "online_g_state_sha256": _state_dict_sha256(
            parent["predictor_state_dict"]
        ),
        "target_g_state_sha256": _state_dict_sha256(
            parent["target_predictor_state_dict"]
        ),
    }
    payload = _deployment_payload(
        module,
        protocol=protocol,
        parent_payload=parent,
        parent_provenance=provenance,
        epoch=12,
        global_step=12_000,
    )

    assert payload["source_v1_c_provenance"] == protocol["source_v1_c"]
    assert payload["global_step"] == 12_000
    for output_key, parent_key in (
        ("world_model_state_dict", "world_model_state_dict"),
        ("predictor_state_dict", "predictor_state_dict"),
        ("target_predictor_state_dict", "target_predictor_state_dict"),
    ):
        assert _state_dict_sha256(payload[output_key]) == _state_dict_sha256(
            parent[parent_key]
        )


def test_validation_summary_reports_required_offline_metrics():
    prediction = np.asarray([5.0, 12.0, 24.0, 51.0])
    mc_target = np.asarray([5.0, 10.0, 25.0, 50.0])
    td_target = np.asarray([5.0, 11.0, 25.0, 49.0])
    identity = np.zeros(4)
    summary = _validation_summary(
        prediction,
        mc_target,
        td_target,
        identity,
        bins=[0, 10, 25, 50, 75],
    )

    assert summary["finite_fraction"] == 1.0
    assert summary["goal_identity_max"] == 0.0
    assert summary["spearman"] == pytest.approx(1.0)
    assert summary["monotonic_ranking_accuracy"] == 1.0
    assert summary["mc_mse"] == pytest.approx(1.5)
    assert summary["mc_mae"] == pytest.approx(1.0)
    assert summary["delta_binned_calibration"]
