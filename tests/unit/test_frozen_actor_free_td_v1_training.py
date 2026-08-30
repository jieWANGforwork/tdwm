from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.methods.action_prefix_advantage_common import (
    build_zero_mean_action_prefixes,
)
from tdwm.training.frozen_actor_free_td_v1 import (
    OBJECTIVE_VERSION,
    V1_SPECS,
    _build_v1_training_module,
    _checkpoint_result_fields,
    _cuda_runtime_provenance,
    _deployment_checkpoint_path,
    _deployment_payload,
    _record_peak_cuda_memory,
    _reset_peak_cuda_memory,
    _validate_v1_resume_manifest,
    load_actor_free_td_lewm_v1_training_protocol,
    validate_actor_free_td_lewm_v1_training_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def _protocol(variant: str) -> dict:
    return load_actor_free_td_lewm_v1_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v1_{variant}_cube_train.yaml",
        spec=V1_SPECS[variant],
    )


class _RecordingActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)
        self.seen: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.fill_(0.25)
            self.projection.weight[:25, :25].copy_(torch.eye(25))

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        self.seen.append(raw_action.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        output = torch.tanh(self.projection(raw_action))
        self.outputs.append(output.detach().clone())
        return output


class _FrozenWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))
        self.action_encoder = _RecordingActionEncoder()
        self.encode_calls = 0

    def encode(self, data):
        del data
        self.encode_calls += 1
        raise AssertionError("V1 frozen-cache training must not call encode().")


class _NeighborIndex:
    def __init__(self, actions: torch.Tensor | None = None) -> None:
        self.actions = actions
        self.returned_actions: torch.Tensor | None = None

    def lookup(self, global_rows, *, device, dtype):
        count = int(global_rows.numel())
        actions = self.actions
        if actions is None:
            actions = torch.randn(count, 2, 25)
        actions = actions.to(device=device, dtype=dtype)
        self.returned_actions = actions.detach().clone()
        return SimpleNamespace(
            actions=actions,
            distances=torch.ones(count, 2, device=device, dtype=dtype),
            neighbor_rows=torch.zeros(count, 2, device=device, dtype=torch.int64),
        )


def _small_protocol(variant: str) -> dict:
    protocol = _protocol(variant)
    protocol["sequence"]["num_steps"] = 7
    protocol["predictor"]["hidden_dim"] = 32
    protocol["training"]["epochs"] = 1
    protocol["training"]["scheduler_epochs"] = 1
    protocol["training"]["optimizer_steps_per_epoch"] = 1
    return protocol


def _training_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(batch_size, 192),
        "action": torch.randn(batch_size, 25),
        "next_state": torch.randn(batch_size, 192),
        "next_action": torch.randn(batch_size, 25),
        "terminal": torch.zeros(batch_size, dtype=torch.bool),
        "global_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 100,
        "goal_future_end_row": (torch.arange(batch_size, dtype=torch.int64) * 5 + 105),
        "_tdwm_matched_goal": torch.randn(batch_size, 192),
    }


def test_v1_cuda_provenance_uses_one_explicit_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, torch.device] = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def get_device_name(device: torch.device) -> str:
        seen["name"] = device
        return "Audit GPU"

    def reset_peak_memory_stats(device: torch.device) -> None:
        seen["reset"] = device

    def max_memory_allocated(device: torch.device) -> int:
        seen["peak"] = device
        return 12_345_678

    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_memory_allocated)

    device, runtime = _cuda_runtime_provenance()
    result: dict = {}
    _reset_peak_cuda_memory(device)
    _record_peak_cuda_memory(result, device)

    expected_device = torch.device("cuda", 3)
    assert device == expected_device
    assert runtime == {"cuda_device": "Audit GPU"}
    assert result == {"peak_cuda_memory_bytes": 12_345_678}
    assert seen == {
        "name": expected_device,
        "reset": expected_device,
        "peak": expected_device,
    }


def test_v1_cuda_provenance_omits_gpu_fields_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda *_args, **_kwargs: pytest.fail("CPU provenance must not reset CUDA"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda *_args, **_kwargs: pytest.fail("CPU provenance must not query CUDA"),
    )

    device, runtime = _cuda_runtime_provenance()
    result: dict = {}
    _reset_peak_cuda_memory(device)
    _record_peak_cuda_memory(result, device)

    assert device is None
    assert runtime == {}
    assert result == {}


def _resume_fixture() -> tuple[dict, dict, dict, dict]:
    split = {
        "train_indices_sha256": "train-split",
        "validation_indices_sha256": "validation-split",
    }
    initialization = {
        "source_checkpoint_sha256": "checkpoint",
        "source_training_result_sha256": "training-result",
        "source_training_manifest_sha256": "training-manifest",
        "source_final_epoch": 10,
        "source_global_step": 127960,
    }
    store = {
        "manifest_sha256": "latent-manifest",
        "pretrained_checkpoint_sha256": "checkpoint",
        "dataset_source_sha256": "dataset",
        "dataset_manifest_sha256": "dataset-manifest",
        "column_normalization_sha256": "normalization",
        "input_file_sha256": "input",
    }
    previous = {
        "method": V1_SPECS["c"].method,
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c",
        "implementation_version": "v1",
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": 1,
        "protocol_sha256": "protocol",
        "seed": 42,
        "dataset": {"split": deepcopy(split)},
        "model": {"initialization": deepcopy(initialization)},
        "frozen_latent_store": deepcopy(store),
        "neighbor_index": None,
    }
    return previous, split, initialization, store


def test_v1_resume_manifest_is_bound_to_all_training_artifacts() -> None:
    previous, split, initialization, store = _resume_fixture()
    arguments = {
        "spec": V1_SPECS["c"],
        "protocol_sha256": "protocol",
        "seed": 42,
        "split_manifest": split,
        "initialization_info": initialization,
        "frozen_latent_store_info": store,
        "neighbor_index_info": None,
    }

    _validate_v1_resume_manifest(previous, **arguments)

    incompatible = deepcopy(previous)
    incompatible["dataset"]["split"]["validation_indices_sha256"] = "other"
    with pytest.raises(RuntimeError, match="incompatible V1 run"):
        _validate_v1_resume_manifest(incompatible, **arguments)

    incompatible = deepcopy(previous)
    incompatible["model"]["initialization"]["source_global_step"] += 1
    with pytest.raises(RuntimeError, match="incompatible V1 run"):
        _validate_v1_resume_manifest(incompatible, **arguments)

    incompatible = deepcopy(previous)
    incompatible["frozen_latent_store"]["column_normalization_sha256"] = "other"
    with pytest.raises(RuntimeError, match="incompatible V1 run"):
        _validate_v1_resume_manifest(incompatible, **arguments)


def test_v1_g1_resume_manifest_is_bound_to_neighbor_index() -> None:
    previous, split, initialization, store = _resume_fixture()
    previous.update(method=V1_SPECS["g1"].method, variant="g1")
    previous["neighbor_index"] = {"manifest_sha256": "neighbors"}
    arguments = {
        "spec": V1_SPECS["g1"],
        "protocol_sha256": "protocol",
        "seed": 42,
        "split_manifest": split,
        "initialization_info": initialization,
        "frozen_latent_store_info": store,
        "neighbor_index_info": {"manifest_sha256": "neighbors"},
    }

    _validate_v1_resume_manifest(previous, **arguments)

    arguments["neighbor_index_info"] = {"manifest_sha256": "other"}
    with pytest.raises(RuntimeError, match="incompatible V1 run"):
        _validate_v1_resume_manifest(previous, **arguments)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_variants_train_only_the_single_predictor_from_frozen_inputs(
    variant: str,
) -> None:
    protocol = _small_protocol(variant)
    world = _FrozenWorld()
    data_generator = torch.Generator().manual_seed(11)
    goal_generator = torch.Generator().manual_seed(12)
    task_generator = torch.Generator().manual_seed(13)
    module = _build_v1_training_module(
        world,
        protocol,
        total_steps=1,
        spec=V1_SPECS[variant],
        data_generator=data_generator,
        goal_generator=goal_generator,
        task_generator=task_generator,
        neighbor_index=_NeighborIndex() if variant == "g1" else None,
    )
    module.train()
    assert not world.training
    assert not world.action_encoder.training
    assert not module.target_predictor.training
    optimizer = module.configure_optimizers()["optimizer"]
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == {
        id(parameter) for parameter in module.predictor.parameters()
    }
    assert optimizer_parameter_ids.isdisjoint(
        {id(parameter) for parameter in world.action_encoder.parameters()}
    )
    encoder_before = {
        key: value.detach().clone()
        for key, value in world.action_encoder.state_dict().items()
    }
    batch = _training_batch(batch_size=8)

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert world.encode_calls == 0
    assert len(world.action_encoder.seen) == (3 if variant in {"g1", "g2", "g3"} else 2)
    assert torch.equal(world.action_encoder.seen[0], batch["action"].reshape(-1, 1, 25))
    assert torch.equal(
        world.action_encoder.seen[1], batch["next_action"].reshape(-1, 1, 25)
    )
    assert all(enabled is False for enabled in world.action_encoder.grad_enabled)
    assert all(parameter.grad is None for parameter in world.parameters())
    assert all(
        parameter.grad is None for parameter in module.target_predictor.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in module.predictor.parameters()
    )
    assert not hasattr(module.predictor, "heads")
    assert not hasattr(module.predictor, "num_parallel")
    optimizer.step()
    assert all(
        torch.equal(value, encoder_before[key])
        for key, value in world.action_encoder.state_dict().items()
    )


def test_current_and_dataset_next_actions_share_one_frozen_encoder() -> None:
    world = _FrozenWorld()
    module = _build_v1_training_module(
        world,
        _small_protocol("c"),
        total_steps=1,
        spec=V1_SPECS["c"],
        data_generator=torch.Generator().manual_seed(1),
        goal_generator=torch.Generator().manual_seed(2),
        task_generator=torch.Generator().manual_seed(3),
        neighbor_index=None,
    )
    batch = _training_batch(batch_size=3)
    batch["action"].fill_(1.0)
    batch["next_action"].fill_(2.0)
    online_actions: list[torch.Tensor] = []
    target_actions: list[torch.Tensor] = []

    module.predictor.register_forward_pre_hook(
        lambda _module, inputs: online_actions.append(inputs[1].detach().clone())
    )
    module.target_predictor.register_forward_pre_hook(
        lambda _module, inputs: target_actions.append(inputs[1].detach().clone())
    )
    module._forward_loss(batch, "train")

    assert world.action_encoder is module.model.action_encoder
    assert len(world.action_encoder.seen) == 2
    assert torch.equal(world.action_encoder.seen[0], batch["action"].reshape(-1, 1, 25))
    assert torch.equal(
        world.action_encoder.seen[1], batch["next_action"].reshape(-1, 1, 25)
    )
    assert torch.equal(online_actions[0], world.action_encoder.outputs[0].squeeze(1))
    assert torch.equal(target_actions[0], world.action_encoder.outputs[1].squeeze(1))
    assert online_actions[0].shape == target_actions[0].shape == (3, 192)
    assert not torch.equal(online_actions[0], target_actions[0])


def test_g1_encodes_retrieved_raw_neighbor_blocks_before_scoring() -> None:
    batch_size, candidates = 3, 2
    neighbors = torch.arange(batch_size * candidates * 25, dtype=torch.float32).reshape(
        batch_size, candidates, 25
    )
    index = _NeighborIndex(neighbors)
    world = _FrozenWorld()
    module = _build_v1_training_module(
        world,
        _small_protocol("g1"),
        total_steps=1,
        spec=V1_SPECS["g1"],
        data_generator=torch.Generator().manual_seed(4),
        goal_generator=torch.Generator().manual_seed(5),
        task_generator=torch.Generator().manual_seed(6),
        neighbor_index=index,
    )
    predictor_actions: list[torch.Tensor] = []
    module.predictor.register_forward_pre_hook(
        lambda _module, inputs: predictor_actions.append(inputs[1].detach().clone())
    )
    module._forward_loss(_training_batch(batch_size), "train")

    assert index.returned_actions is not None
    assert torch.equal(world.action_encoder.seen[-1], neighbors.reshape(-1, 1, 25))
    assert predictor_actions[-1].shape == (batch_size, candidates, 192)
    assert torch.equal(
        predictor_actions[-1],
        world.action_encoder.outputs[-1].reshape(batch_size, candidates, 192),
    )


@pytest.mark.parametrize("variant", ("g2", "g3"))
def test_prefix_variants_build_raw_zero_suffixes_before_nonlinear_encoding(
    variant: str,
) -> None:
    world = _FrozenWorld()
    module = _build_v1_training_module(
        world,
        _small_protocol(variant),
        total_steps=1,
        spec=V1_SPECS[variant],
        data_generator=torch.Generator().manual_seed(7),
        goal_generator=torch.Generator().manual_seed(8),
        task_generator=torch.Generator().manual_seed(9),
        neighbor_index=None,
    )
    batch = _training_batch(batch_size=2)
    batch["action"] = torch.arange(50, dtype=torch.float32).reshape(2, 25)
    predictor_actions: list[torch.Tensor] = []
    module.predictor.register_forward_pre_hook(
        lambda _module, inputs: predictor_actions.append(inputs[1].detach().clone())
    )
    module._forward_loss(batch, "train")

    raw_prefixes = build_zero_mean_action_prefixes(batch["action"])
    assert torch.equal(world.action_encoder.seen[-1], raw_prefixes.reshape(-1, 1, 25))
    assert predictor_actions[-1].shape == (2, 5, 192)
    assert torch.equal(
        predictor_actions[-1],
        world.action_encoder.outputs[-1].reshape(2, 5, 192),
    )


def test_v1_checkpoint_round_trip_records_training_and_validation_rng_streams() -> None:
    protocol = _small_protocol("c")
    generators = [torch.Generator().manual_seed(seed) for seed in (21, 22, 23, 24, 25)]
    module = _build_v1_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V1_SPECS["c"],
        data_generator=generators[0],
        goal_generator=generators[1],
        task_generator=generators[2],
        validation_goal_generator=generators[3],
        validation_task_generator=generators[4],
        neighbor_index=None,
    )
    checkpoint: dict = {}
    module.on_save_checkpoint(checkpoint)
    expected = [torch.rand(4, generator=generator) for generator in generators]
    for generator in generators:
        torch.rand(9, generator=generator)

    module.on_load_checkpoint(checkpoint)

    assert len(generators) == len(expected)
    for generator, expected_values in zip(generators, expected):
        assert torch.equal(torch.rand(4, generator=generator), expected_values)


def test_validation_goal_and_task_streams_reset_to_one_fixed_epoch_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _small_protocol("c")
    task_generator = torch.Generator().manual_seed(32)
    validation_goal_generator = torch.Generator().manual_seed(33)
    validation_task_generator = torch.Generator().manual_seed(34)
    module = _build_v1_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V1_SPECS["c"],
        data_generator=torch.Generator().manual_seed(30),
        goal_generator=torch.Generator().manual_seed(31),
        task_generator=task_generator,
        validation_goal_generator=validation_goal_generator,
        validation_task_generator=validation_task_generator,
        neighbor_index=None,
        latent_store=object(),
    )
    batch_size = 4
    batch = {
        "state": torch.randn(batch_size, 192),
        "action": torch.randn(batch_size, 25),
        "next_state": torch.randn(batch_size, 192),
        "next_action": torch.randn(batch_size, 25),
        "terminal": torch.zeros(batch_size, dtype=torch.bool),
        "global_row": torch.arange(batch_size, dtype=torch.int64) * 5,
        "goal_future_end_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 5,
    }
    seen_goal_generators: list[torch.Generator] = []

    def sample_validation_goals(
        store, global_rows, future_end_rows, *, generator, device
    ):
        del store, future_end_rows
        seen_goal_generators.append(generator)
        draws = torch.rand(len(global_rows), generator=generator)
        return SimpleNamespace(latents=draws[:, None].expand(-1, 192).to(device=device))

    monkeypatch.setattr(
        "tdwm.training.frozen_actor_free_td_v1.sample_reachable_future_latents_v1",
        sample_validation_goals,
    )
    training_state = task_generator.get_state().clone()
    validation_goal_epoch_state = validation_goal_generator.get_state().clone()
    validation_task_epoch_state = validation_task_generator.get_state().clone()

    module.on_validation_epoch_start()
    first_loss = module._forward_loss(batch, "validation").detach()
    first_goal_end_state = validation_goal_generator.get_state().clone()
    first_task_end_state = validation_task_generator.get_state().clone()

    # Mimic both validation streams having advanced during a complete epoch,
    # then require the lifecycle hook to recreate the identical population.
    torch.rand(7, generator=validation_goal_generator)
    torch.rand(7, generator=validation_task_generator)
    module.on_validation_epoch_start()
    assert torch.equal(
        validation_goal_generator.get_state(), validation_goal_epoch_state
    )
    assert torch.equal(
        validation_task_generator.get_state(), validation_task_epoch_state
    )
    second_loss = module._forward_loss(batch, "validation").detach()

    assert torch.equal(task_generator.get_state(), training_state)
    assert seen_goal_generators == [validation_goal_generator] * 2
    assert torch.equal(validation_goal_generator.get_state(), first_goal_end_state)
    assert torch.equal(validation_task_generator.get_state(), first_task_end_state)
    torch.testing.assert_allclose(second_loss, first_loss)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_train_on_method_loss_but_validate_on_common_base_td(
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _small_protocol(variant)
    module = _build_v1_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V1_SPECS[variant],
        data_generator=torch.Generator().manual_seed(40),
        goal_generator=torch.Generator().manual_seed(41),
        task_generator=torch.Generator().manual_seed(42),
        validation_goal_generator=torch.Generator().manual_seed(43),
        validation_task_generator=torch.Generator().manual_seed(44),
        neighbor_index=_NeighborIndex() if variant == "g1" else None,
    )
    batch_size = 4
    batch = {
        "state": torch.randn(batch_size, 192),
        "action": torch.randn(batch_size, 25),
        "next_state": torch.randn(batch_size, 192),
        "next_action": torch.randn(batch_size, 25),
        "terminal": torch.zeros(batch_size, dtype=torch.bool),
        "global_row": torch.arange(batch_size, dtype=torch.int64) * 5,
        "goal_future_end_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 5,
        "_tdwm_matched_goal": torch.randn(batch_size, 192),
    }
    observed_base: dict[str, torch.Tensor] = {}

    def method_loss(td_batch, *args, stage, **kwargs):
        del args, kwargs
        observed_base[stage] = td_batch.td_loss.detach()
        return td_batch.td_loss + 17.0, {}

    monkeypatch.setattr(module, "_method_loss", method_loss)
    train_loss = module._forward_loss(batch, "train")
    module.on_validation_epoch_start()
    validation_loss = module._forward_loss(batch, "validation")

    torch.testing.assert_allclose(train_loss, observed_base["train"] + 17.0)
    torch.testing.assert_allclose(validation_loss, observed_base["validation"])


def test_g1_validation_does_not_query_incomplete_training_anchor_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _small_protocol("g1")
    module = _build_v1_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V1_SPECS["g1"],
        data_generator=torch.Generator().manual_seed(50),
        goal_generator=torch.Generator().manual_seed(51),
        task_generator=torch.Generator().manual_seed(52),
        neighbor_index=_NeighborIndex(),
    )

    def fail_neighbor_query(*args, **kwargs):
        del args, kwargs
        raise AssertionError("validation must not query a train-anchor-only index")

    monkeypatch.setattr(module, "_score_neighbors", fail_neighbor_query)
    per_td = torch.tensor([1.0, 2.0, 4.0])
    td_batch = SimpleNamespace(td_loss=per_td.mean(), per_transition_td_loss=per_td)
    loss, metrics = module._method_loss(
        td_batch,
        torch.randn(3, 192),
        torch.randn(3, 25),
        torch.randn(3, 192),
        torch.randn(3, 192),
        torch.ones(3, dtype=torch.bool),
        torch.tensor([10, 20, 30]),
        stage="validation",
    )

    torch.testing.assert_allclose(loss, td_batch.td_loss)
    assert metrics["validation/neighbor_objective_available"].item() == 0.0


def test_v1_deployment_contains_one_online_and_one_ema_target_predictor() -> None:
    protocol = _protocol("f")
    module = _build_v1_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V1_SPECS["f"],
        data_generator=torch.Generator().manual_seed(60),
        goal_generator=torch.Generator().manual_seed(61),
        task_generator=torch.Generator().manual_seed(62),
        neighbor_index=None,
    )
    payload = _deployment_payload(
        module,
        protocol=protocol,
        spec=V1_SPECS["f"],
        model_config={
            "_target_": "example.WorldModel",
            "action_encoder": {"input_dim": 25, "emb_dim": 192},
        },
        initialization_info={"frozen": True},
        epoch=1,
        global_step=2,
    )

    assert payload["objective_version"] == OBJECTIVE_VERSION == 0
    assert payload["predictor_config"]["num_parallel"] == 1
    assert payload["predictor_config"]["state_dim"] == 192
    assert payload["predictor_config"]["raw_action_dim"] == 25
    assert payload["predictor_config"]["action_dim"] == 192
    assert payload["predictor_config"]["action_embedding_dim"] == 192
    assert payload["predictor_config"]["shared_lewm_action_encoder"] is True
    assert payload["predictor_config"]["action_encoder_trainable"] is False
    assert payload["predictor_config"]["action_encoder_source"] == (
        "world_model.action_encoder"
    )
    assert payload["predictor_config"]["task_dim"] == 192
    assert payload["predictor_config"]["output_dim"] == 192
    assert "predictor_state_dict" in payload
    assert "target_predictor_state_dict" in payload
    assert "successor_state_dict" not in payload
    assert "actor_state_dict" not in payload
    assert "action_encoder_state_dict" not in payload
    assert any(
        key.startswith("action_encoder.") for key in payload["world_model_state_dict"]
    )
    assert all("action_encoder" not in key for key in payload["predictor_state_dict"])
    assert all(
        "action_encoder" not in key for key in payload["target_predictor_state_dict"]
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_formal_result_uses_the_epoch_10_deployment_export(
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint_path = _deployment_checkpoint_path(
        tmp_path / f"seed_{3072}",
        spec=V1_SPECS[variant],
        epoch=10,
    )

    assert checkpoint_path == (
        tmp_path
        / "seed_3072"
        / "checkpoints"
        / f"actor_free_td_lewm_v1_{variant}"
        / variant
        / "epoch_10.pt"
    )
    assert checkpoint_path.name != "last.ckpt"

    result_fields = _checkpoint_result_fields(
        tmp_path / "seed_3072",
        spec=V1_SPECS[variant],
        deployment_epoch=10,
    )
    assert result_fields["deployment_checkpoint"] == str(checkpoint_path)
    assert result_fields["last_checkpoint"].endswith("checkpoints/lightning/last.ckpt")
    assert result_fields["deployment_checkpoint"] != result_fields["last_checkpoint"]


def test_v1_protocol_rejects_parallel_heads_and_exact_half_sampling() -> None:
    protocol = _protocol("c")
    parallel = deepcopy(protocol)
    parallel["predictor"]["num_parallel"] = 2
    with pytest.raises(ValueError, match="num_parallel"):
        validate_actor_free_td_lewm_v1_training_protocol(parallel, spec=V1_SPECS["c"])

    exact_half = deepcopy(protocol)
    exact_half["task_sampling"]["sampling"] = "exact_half"
    with pytest.raises(ValueError, match="sampling"):
        validate_actor_free_td_lewm_v1_training_protocol(exact_half, spec=V1_SPECS["c"])


def test_v1_protocol_locks_transition_minibatches() -> None:
    protocol = _protocol("f")
    assert protocol["loader"]["batch_size"] == 256
    assert protocol["loader"]["sampling_unit"] == "transition"
    assert protocol["loader"]["train_sampling"] == "random_with_replacement"
    assert protocol["loader"]["transition_population"] == (
        "unique_legal_td_rows_from_exact_clip_split"
    )
    assert protocol["task_sampling"]["mix_unit"] == "transition_minibatch"
    assert protocol["context"] == {
        "g_state_frames": 1,
        "lewm_rollout_history_frames": 3,
    }

    clip_batch = deepcopy(protocol)
    clip_batch["loader"]["sampling_unit"] = "sequence_clip"
    with pytest.raises(ValueError, match="sampling_unit"):
        validate_actor_free_td_lewm_v1_training_protocol(clip_batch, spec=V1_SPECS["f"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_protocol_locks_each_variant_objective_semantics(variant: str) -> None:
    protocol = _protocol(variant)
    changed = deepcopy(protocol)
    changed["joint_objective"]["objective"] = "different_objective"
    with pytest.raises(ValueError, match="joint_objective.objective"):
        validate_actor_free_td_lewm_v1_training_protocol(
            changed, spec=V1_SPECS[variant]
        )


def test_v1_protocol_locks_default_tdjepa_forward_map_widths() -> None:
    protocol = _protocol("c")
    changed = deepcopy(protocol)
    changed["predictor"]["hidden_dim"] = 512
    with pytest.raises(ValueError, match="predictor.hidden_dim"):
        validate_actor_free_td_lewm_v1_training_protocol(changed, spec=V1_SPECS["c"])
