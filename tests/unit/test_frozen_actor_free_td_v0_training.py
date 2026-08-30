from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.training.frozen_actor_free_td_v0 import (
    OBJECTIVE_VERSION,
    V0_SPECS,
    _build_v0_training_module,
    _checkpoint_result_fields,
    _deployment_checkpoint_path,
    _deployment_payload,
    load_actor_free_td_lewm_v0_training_protocol,
    validate_actor_free_td_lewm_v0_training_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def _protocol(variant: str) -> dict:
    return load_actor_free_td_lewm_v0_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v0_{variant}_cube_train.yaml",
        spec=V0_SPECS[variant],
    )


class _FrozenWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))
        self.encode_calls = 0

    def encode(self, data):
        del data
        self.encode_calls += 1
        raise AssertionError("V0 frozen-cache training must not call encode().")


class _NeighborIndex:
    def lookup(self, global_rows, *, device, dtype):
        count = int(global_rows.numel())
        return SimpleNamespace(
            actions=torch.randn(count, 2, 25, device=device, dtype=dtype),
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_v0_variants_train_only_the_single_predictor_from_frozen_inputs(
    variant: str,
) -> None:
    protocol = _small_protocol(variant)
    world = _FrozenWorld()
    data_generator = torch.Generator().manual_seed(11)
    goal_generator = torch.Generator().manual_seed(12)
    task_generator = torch.Generator().manual_seed(13)
    module = _build_v0_training_module(
        world,
        protocol,
        total_steps=1,
        spec=V0_SPECS[variant],
        data_generator=data_generator,
        goal_generator=goal_generator,
        task_generator=task_generator,
        neighbor_index=_NeighborIndex() if variant == "g1" else None,
    )
    batch_size = 8
    batch = {
        "state": torch.randn(batch_size, 192),
        "action": torch.randn(batch_size, 25),
        "next_state": torch.randn(batch_size, 192),
        "next_action": torch.randn(batch_size, 25),
        "terminal": torch.zeros(batch_size, dtype=torch.bool),
        "global_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 100,
        "goal_future_end_row": (
            torch.arange(batch_size, dtype=torch.int64) * 5 + 105
        ),
        "_tdwm_matched_goal": torch.randn(batch_size, 192),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert world.encode_calls == 0
    assert all(parameter.grad is None for parameter in world.parameters())
    assert all(
        parameter.grad is None for parameter in module.target_predictor.parameters()
    )
    assert any(parameter.grad is not None for parameter in module.predictor.parameters())
    assert not hasattr(module.predictor, "heads")
    assert not hasattr(module.predictor, "num_parallel")


def test_v0_checkpoint_round_trip_records_training_and_validation_rng_streams() -> None:
    protocol = _small_protocol("c")
    generators = [
        torch.Generator().manual_seed(seed) for seed in (21, 22, 23, 24, 25)
    ]
    module = _build_v0_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V0_SPECS["c"],
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
    module = _build_v0_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V0_SPECS["c"],
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
        return SimpleNamespace(
            latents=draws[:, None].expand(-1, 192).to(device=device)
        )

    monkeypatch.setattr(
        "tdwm.training.frozen_actor_free_td_v0.sample_reachable_future_latents_v0",
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
    module = _build_v0_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V0_SPECS[variant],
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
    module = _build_v0_training_module(
        _FrozenWorld(),
        protocol,
        total_steps=1,
        spec=V0_SPECS["g1"],
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
        torch.ones(3, dtype=torch.bool),
        torch.tensor([10, 20, 30]),
        stage="validation",
    )

    torch.testing.assert_allclose(loss, td_batch.td_loss)
    assert metrics["validation/neighbor_objective_available"].item() == 0.0


def test_v0_deployment_contains_one_online_and_one_ema_target_predictor() -> None:
    protocol = _protocol("f")
    module = SimpleNamespace(
        model=nn.Linear(2, 3),
        predictor=nn.Linear(3, 4),
        target_predictor=nn.Linear(3, 4),
    )
    payload = _deployment_payload(
        module,
        protocol=protocol,
        spec=V0_SPECS["f"],
        model_config={"_target_": "example.WorldModel"},
        initialization_info={"frozen": True},
        epoch=1,
        global_step=2,
    )

    assert payload["objective_version"] == OBJECTIVE_VERSION == 0
    assert payload["predictor_config"]["num_parallel"] == 1
    assert payload["predictor_config"]["state_dim"] == 192
    assert payload["predictor_config"]["action_dim"] == 25
    assert payload["predictor_config"]["task_dim"] == 192
    assert payload["predictor_config"]["output_dim"] == 192
    assert "predictor_state_dict" in payload
    assert "target_predictor_state_dict" in payload
    assert "successor_state_dict" not in payload
    assert "actor_state_dict" not in payload


@pytest.mark.parametrize("variant", VARIANTS)
def test_v0_formal_result_uses_the_epoch_10_deployment_export(
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint_path = _deployment_checkpoint_path(
        tmp_path / f"seed_{3072}",
        spec=V0_SPECS[variant],
        epoch=10,
    )

    assert checkpoint_path == (
        tmp_path
        / "seed_3072"
        / "checkpoints"
        / f"actor_free_td_lewm_v0_{variant}"
        / variant
        / "epoch_10.pt"
    )
    assert checkpoint_path.name != "last.ckpt"

    result_fields = _checkpoint_result_fields(
        tmp_path / "seed_3072",
        spec=V0_SPECS[variant],
        deployment_epoch=10,
    )
    assert result_fields["deployment_checkpoint"] == str(checkpoint_path)
    assert result_fields["last_checkpoint"].endswith(
        "checkpoints/lightning/last.ckpt"
    )
    assert result_fields["deployment_checkpoint"] != result_fields["last_checkpoint"]


def test_v0_protocol_rejects_parallel_heads_and_exact_half_sampling() -> None:
    protocol = _protocol("c")
    parallel = deepcopy(protocol)
    parallel["predictor"]["num_parallel"] = 2
    with pytest.raises(ValueError, match="num_parallel"):
        validate_actor_free_td_lewm_v0_training_protocol(
            parallel, spec=V0_SPECS["c"]
        )

    exact_half = deepcopy(protocol)
    exact_half["task_sampling"]["sampling"] = "exact_half"
    with pytest.raises(ValueError, match="sampling"):
        validate_actor_free_td_lewm_v0_training_protocol(
            exact_half, spec=V0_SPECS["c"]
        )


def test_v0_protocol_locks_transition_minibatches() -> None:
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
        validate_actor_free_td_lewm_v0_training_protocol(
            clip_batch, spec=V0_SPECS["f"]
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_v0_protocol_locks_each_variant_objective_semantics(variant: str) -> None:
    protocol = _protocol(variant)
    changed = deepcopy(protocol)
    changed["joint_objective"]["objective"] = "different_objective"
    with pytest.raises(ValueError, match="joint_objective.objective"):
        validate_actor_free_td_lewm_v0_training_protocol(
            changed, spec=V0_SPECS[variant]
        )


def test_v0_protocol_locks_default_tdjepa_forward_map_widths() -> None:
    protocol = _protocol("c")
    changed = deepcopy(protocol)
    changed["predictor"]["hidden_dim"] = 512
    with pytest.raises(ValueError, match="predictor.hidden_dim"):
        validate_actor_free_td_lewm_v0_training_protocol(
            changed, spec=V0_SPECS["c"]
        )
