from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from tdwm.training.actor_free_td_lewm_v1_c4 import (
    METHOD,
    _build_v1_c4_training_module,
    _deployment_payload,
    _validate_resume_manifest,
    load_actor_free_td_lewm_v1_c4_training_protocol,
    validate_actor_free_td_lewm_v1_c4_training_protocol,
)
from tdwm.training.frozen_actor_free_td_v1 import _state_dict_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "actor_free_td_lewm_v1_c4_cube_train.yaml"
)


class _RecordingActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)
        self.seen: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            self.projection.weight[:25, :25].copy_(torch.eye(25))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        self.seen.append(actions.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.projection(actions)


class _FrozenWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation_encoder = nn.Linear(2, 2)
        self.action_encoder = _RecordingActionEncoder()
        self.forward_bias = nn.Parameter(torch.full((192,), 0.125))
        self.encode_calls = 0
        self.predict_calls = 0
        self.predict_grad_enabled: list[bool] = []

    def encode(self, _data):
        self.encode_calls += 1
        raise AssertionError("C4 frozen-cache training must not call visual encode.")

    def predict(
        self, state_history: torch.Tensor, action_embedding: torch.Tensor
    ) -> torch.Tensor:
        self.predict_calls += 1
        self.predict_grad_enabled.append(torch.is_grad_enabled())
        return state_history + action_embedding + self.forward_bias


def _protocol() -> dict:
    protocol = load_actor_free_td_lewm_v1_c4_training_protocol(CONFIG)
    protocol["g"]["hidden_dim"] = 16
    protocol["training"]["epochs"] = 1
    protocol["training"]["scheduler_epochs"] = 1
    protocol["training"]["optimizer_steps_per_epoch"] = 1
    return protocol


def _module():
    generators = [torch.Generator().manual_seed(seed) for seed in range(10, 15)]
    world = _FrozenWorld()
    module = _build_v1_c4_training_module(
        world,
        _protocol(),
        total_steps=1,
        data_generator=generators[0],
        goal_generator=generators[1],
        task_generator=generators[2],
        validation_goal_generator=generators[3],
        validation_task_generator=generators[4],
        latent_store=None,
    )
    return module, world, generators


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    terminal = torch.tensor(
        [index == batch_size - 1 for index in range(batch_size)], dtype=torch.bool
    )
    return {
        "state": torch.randn(batch_size, 192),
        "next_state": torch.randn(batch_size, 192),
        "action": torch.randn(batch_size, 25),
        "next_action": torch.randn(batch_size, 25),
        "terminal": terminal.clone(),
        "global_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 100,
        "goal_future_end_row": (
            torch.arange(batch_size, dtype=torch.int64) * 5 + 105
        ),
        "_tdwm_matched_goal": torch.randn(batch_size, 192),
        "c4_real_state": torch.randn(batch_size, 192),
        "c4_bootstrap_next_state": torch.randn(batch_size, 192),
        "c4_f_state_history": torch.randn(batch_size, 3, 192),
        "c4_f_previous_actions": torch.randn(batch_size, 2, 25),
        "c4_current_global_row": (
            torch.arange(batch_size, dtype=torch.int64) * 5 + 105
        ),
        "c4_bootstrap_global_row": (
            torch.arange(batch_size, dtype=torch.int64) * 5 + 110
        ),
        "c4_terminal": terminal,
    }


def test_c4_formal_protocol_is_v1_c_paired_and_has_no_g_action_input() -> None:
    protocol = load_actor_free_td_lewm_v1_c4_training_protocol(CONFIG)

    assert protocol["method"] == METHOD
    assert protocol["seeds"] == [3072]
    assert protocol["training"]["epochs"] == 10
    assert protocol["training"]["optimizer_steps_per_epoch"] == 12_796
    assert 10 * 12_796 == 127_960
    assert protocol["optimizer"]["world_model_learning_rate"] == 0.0
    assert protocol["joint_objective"]["goal_projection_weight"] == 1.0
    assert protocol["g"]["action_input"] == "none"
    assert not {
        "raw_action_dim",
        "action_dim",
        "action_embedding_dim",
    }.intersection(protocol["g"])

    changed = deepcopy(protocol)
    changed["g"]["action_dim"] = 192
    with pytest.raises(ValueError, match="must not declare action inputs"):
        validate_actor_free_td_lewm_v1_c4_training_protocol(changed)


def test_c4_module_freezes_all_lewm_and_optimizer_is_exactly_online_g(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, world, _generators = _module()
    module.train()
    assert not world.training
    assert not world.action_encoder.training
    assert not module.target_g.training
    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in module.online_g.parameters()}
    assert optimized.isdisjoint({id(parameter) for parameter in world.parameters()})
    assert optimized.isdisjoint({id(parameter) for parameter in module.target_g.parameters()})

    captured: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        module,
        "log_dict",
        lambda values, **_kwargs: captured.update(values),
    )
    loss = module._forward_loss(_batch(), "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert world.encode_calls == 0
    assert world.predict_calls == 1
    assert len(world.action_encoder.seen) == 1
    assert world.action_encoder.grad_enabled == [False]
    assert world.predict_grad_enabled == [False]
    assert all(parameter.grad is None for parameter in world.parameters())
    assert all(parameter.grad is None for parameter in module.target_g.parameters())
    assert any(parameter.grad is not None for parameter in module.online_g.parameters())
    for name in (
        "real_vector_loss",
        "real_goal_loss",
        "predicted_vector_loss",
        "predicted_goal_loss",
        "c4_total_loss",
    ):
        assert f"train/{name}" in captured
        assert torch.isfinite(captured[f"train/{name}"])
    torch.testing.assert_close(captured["train/loss"], captured["train/c4_total_loss"])


def test_c4_module_rejects_terminal_mapping_drift() -> None:
    module, _world, _generators = _module()
    batch = _batch()
    batch["c4_terminal"] = ~batch["terminal"]

    with pytest.raises(RuntimeError, match="terminal mapping"):
        module._forward_loss(batch, "train")


def test_c4_lightning_rng_state_round_trip() -> None:
    module, _world, generators = _module()
    checkpoint: dict = {}
    module.on_save_checkpoint(checkpoint)
    expected = [torch.rand(5, generator=generator) for generator in generators]
    for generator in generators:
        torch.rand(9, generator=generator)

    module.on_load_checkpoint(checkpoint)

    for generator, values in zip(generators, expected, strict=True):
        assert torch.equal(torch.rand(5, generator=generator), values)


def test_c4_deployment_payload_has_independent_schema_and_frozen_f_hash() -> None:
    module, world, _generators = _module()
    frozen_hash = _state_dict_sha256(world.state_dict())
    payload = _deployment_payload(
        module,
        protocol=_protocol(),
        model_config={"_target_": "tests.FakeWorld"},
        initialization_info={"source_checkpoint_sha256": "a" * 64},
        frozen_world_model_sha256=frozen_hash,
        epoch=10,
        global_step=127_960,
    )

    assert payload["method"] == METHOD
    assert payload["variant"] == "c4"
    assert payload["epoch"] == 10
    assert payload["global_step"] == 127_960
    assert set(("online_g_state_dict", "target_g_state_dict", "g_config")) <= set(
        payload
    )
    assert "predictor_state_dict" not in payload
    assert "action_dim" not in payload["g_config"]
    assert payload["frozen_world_model_state_sha256"] == frozen_hash

    with torch.no_grad():
        world.forward_bias.add_(1.0)
    with pytest.raises(RuntimeError, match="LeWM parameters changed"):
        _deployment_payload(
            module,
            protocol=_protocol(),
            model_config={},
            initialization_info={},
            frozen_world_model_sha256=frozen_hash,
            epoch=1,
            global_step=1,
        )


def test_c4_resume_manifest_binds_protocol_split_store_and_lewm() -> None:
    split = {
        "train_indices_sha256": "train",
        "validation_indices_sha256": "validation",
    }
    store = {"manifest_sha256": "store"}
    previous = {
        "method": METHOD,
        "variant": "c4",
        "protocol_sha256": "protocol",
        "seed": 3072,
        "dataset": {"split": deepcopy(split)},
        "frozen_latent_store": deepcopy(store),
        "model": {
            "initialization": {"source_checkpoint_sha256": "checkpoint"}
        },
    }
    kwargs = {
        "protocol_sha256": "protocol",
        "seed": 3072,
        "split_manifest": split,
        "source_checkpoint_sha256": "checkpoint",
        "store_info": store,
    }

    _validate_resume_manifest(previous, **kwargs)

    changed = deepcopy(previous)
    changed["dataset"]["split"]["train_indices_sha256"] = "other"
    with pytest.raises(RuntimeError, match="resume split changed"):
        _validate_resume_manifest(changed, **kwargs)
