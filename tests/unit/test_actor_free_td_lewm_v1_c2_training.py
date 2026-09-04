from __future__ import annotations

import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1_c2 import (
    sample_first_q_candidates_v1_c2,
)
from tdwm.training.actor_free_td_lewm_v1_c import (
    load_actor_free_td_lewm_v1_c_training_protocol,
)
from tdwm.training.actor_free_td_lewm_v1_c2 import (
    METHOD,
    SPEC,
    VARIANT,
    load_actor_free_td_lewm_v1_c2_training_protocol,
    validate_actor_free_td_lewm_v1_c2_training_protocol,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli
from tdwm.training.frozen_actor_free_td_v1 import (
    C2_METHOD,
    C2_SPEC,
    C2_VARIANT,
    V1_SPECS,
    _build_v1_training_module,
    _deployment_payload,
    _load_v1_c_parent_payload,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs" / "experiment" / "actor_free_td_lewm_v1_c2_cube_train.yaml"
)
SCRIPT_PATH = ROOT / "scripts" / "train_actor_free_td_lewm_v1_c2.py"


class _ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(action))


class _FrozenWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.world_parameter = nn.Parameter(torch.ones(()))
        self.action_encoder = _ActionEncoder()


def _small_protocol() -> dict:
    protocol = load_actor_free_td_lewm_v1_c2_training_protocol(CONFIG_PATH)
    protocol["predictor"]["hidden_dim"] = 32
    protocol["training"]["epochs"] = 1
    protocol["training"]["scheduler_epochs"] = 1
    protocol["training"]["optimizer_steps_per_epoch"] = 1
    return protocol


def _build_module(*, candidate_seed: int = 104):
    return _build_v1_training_module(
        _FrozenWorld(),
        _small_protocol(),
        total_steps=1,
        spec=C2_SPEC,
        data_generator=torch.Generator().manual_seed(100),
        goal_generator=torch.Generator().manual_seed(101),
        task_generator=torch.Generator().manual_seed(102),
        candidate_generator=torch.Generator().manual_seed(candidate_seed),
        validation_goal_generator=torch.Generator().manual_seed(105),
        validation_task_generator=torch.Generator().manual_seed(106),
        neighbor_index=None,
    )


def _training_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(batch_size, 192),
        "action": torch.randn(batch_size, 25),
        "next_state": torch.randn(batch_size, 192),
        "next_action": torch.randn(batch_size, 25),
        "terminal": torch.zeros(batch_size, dtype=torch.bool),
        "global_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 100,
        "goal_future_end_row": torch.arange(batch_size, dtype=torch.int64) * 5 + 105,
        "_tdwm_matched_goal": torch.ones(batch_size, 192),
        "alignment_state_history": torch.randn(batch_size, 3, 192),
        "alignment_action_history": torch.randn(batch_size, 2, 25),
        "alignment_action_sequence": torch.randn(batch_size, 5, 25),
        "alignment_rollout_valid": torch.ones(batch_size, dtype=torch.bool),
    }


def test_c2_training_protocol_loads_and_rejects_parent_or_alignment_drift() -> None:
    protocol = load_actor_free_td_lewm_v1_c2_training_protocol(CONFIG_PATH)

    assert protocol["method"] == C2_METHOD
    assert protocol["variant"] == C2_VARIANT
    assert protocol["initialization"] == "v1_c_deployment_parameter_initialization"
    assert protocol["source_v1_c"]["method"] == "actor_free_td_lewm_v1_c"
    assert protocol["source_v1_c"]["parameter_state"] == ("strict_all_model_parameters")
    assert protocol["joint_objective"]["objective"] == (
        "goal_projected_td_plus_first_q_alignment"
    )

    changed_parent = deepcopy(protocol)
    changed_parent["source_v1_c"]["parameter_state"] = "predictor_only"
    with pytest.raises(ValueError, match="source_v1_c.parameter_state"):
        validate_actor_free_td_lewm_v1_c2_training_protocol(changed_parent)

    changed_alignment = deepcopy(protocol)
    changed_alignment["joint_objective"]["first_q_alignment"][
        "teacher_cost_reducer"
    ] = "mean_squared_error"
    with pytest.raises(ValueError, match="teacher_cost_reducer"):
        validate_actor_free_td_lewm_v1_c2_training_protocol(changed_alignment)


def test_c2_spec_and_cli_forward_the_required_v1_c_parent() -> None:
    assert METHOD == C2_METHOD == "actor_free_td_lewm_v1_c2"
    assert VARIANT == C2_VARIANT == "c2"
    assert SPEC is C2_SPEC
    assert SPEC.requires_neighbor_index is False

    script_spec = importlib.util.spec_from_file_location("v1_c2_train_cli", SCRIPT_PATH)
    assert script_spec is not None and script_spec.loader is not None
    script = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script)
    captured_entry: dict[str, object] = {}
    script.run_frozen_actor_free_td_cli = lambda **kwargs: captured_entry.update(kwargs)
    script.main()
    assert captured_entry["requires_v1_c_checkpoint"] is True
    assert captured_entry["requires_neighbor_index"] is False
    assert captured_entry["load_protocol"] is (
        script.load_actor_free_td_lewm_v1_c2_training_protocol
    )
    assert captured_entry["train"] is script.train_actor_free_td_lewm_v1_c2

    captured_train: dict[str, object] = {}

    def load_protocol(path: str | Path) -> dict[str, object]:
        assert path == "c2.yaml"
        return {"method": C2_METHOD}

    def train(**kwargs: object) -> dict[str, object]:
        captured_train.update(kwargs)
        return {"status": "ok"}

    result = run_frozen_actor_free_td_cli(
        method_label="V1 C2",
        requires_neighbor_index=False,
        requires_v1_c_checkpoint=True,
        load_protocol=load_protocol,
        train=train,
        argv=[
            "--config",
            "c2.yaml",
            "--dataset",
            "cube.lance",
            "--output-dir",
            "c2-output",
            "--seed",
            "3072",
            "--initial-world-model-checkpoint",
            "lewm.pt",
            "--initial-v1-c-checkpoint",
            "v1-c.pt",
            "--frozen-latent-store",
            "frozen-store",
            "--split-indices",
            "split_indices.npz",
            "--smoke",
            "--max-steps",
            "1",
            "--skip-validation",
        ],
    )

    assert result == {"status": "ok"}
    assert captured_train["initial_world_model_checkpoint_path"] == "lewm.pt"
    assert captured_train["initial_v1_c_checkpoint_path"] == "v1-c.pt"
    assert captured_train["seed"] == 3072


def test_c2_training_module_freezes_everything_except_g() -> None:
    module = _build_module()
    module.train()

    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    predictor_parameters = {
        id(parameter) for parameter in module.predictor.parameters()
    }
    target_parameters = {
        id(parameter) for parameter in module.target_predictor.parameters()
    }
    world_parameters = {id(parameter) for parameter in module.model.parameters()}

    assert optimized == predictor_parameters
    assert optimized.isdisjoint(target_parameters | world_parameters)
    assert all(parameter.requires_grad for parameter in module.predictor.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in module.target_predictor.parameters()
    )
    assert all(not parameter.requires_grad for parameter in module.model.parameters())
    assert module.predictor.training
    assert not module.target_predictor.training
    assert not module.model.training
    assert not module.model.action_encoder.training


def test_c2_training_loss_is_original_c_loss_plus_first_q_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _build_module()
    batch = _training_batch()
    logged: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        module,
        "log_dict",
        lambda metrics, **_kwargs: logged.update(metrics),
    )

    def c_loss(td_batch, *_args, stage: str, **_kwargs):
        assert stage == "train"
        loss = td_batch.prediction.sum() * 0.0 + 2.25
        return loss, {"train/c_loss_sentinel": loss.detach()}

    def alignment_loss(*_args, **_kwargs):
        loss = next(module.predictor.parameters()).sum() * 0.0 + 0.75
        return loss, {"train/alignment_sentinel": loss.detach()}

    monkeypatch.setattr(module, "_method_loss", c_loss)
    monkeypatch.setattr(module, "_first_q_alignment_loss", alignment_loss)

    loss = module._forward_loss(batch, "train")
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(3.0))
    torch.testing.assert_close(logged["train/c2_base_c_loss"], torch.tensor(2.25))
    torch.testing.assert_close(
        logged["train/first_q_alignment_loss"], torch.tensor(0.75)
    )
    torch.testing.assert_close(logged["train/method_td_loss"], torch.tensor(3.0))
    assert any(
        parameter.grad is not None for parameter in module.predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in module.model.parameters())
    assert all(
        parameter.grad is None for parameter in module.target_predictor.parameters()
    )


def test_c2_checkpoint_restores_the_dedicated_candidate_rng_stream() -> None:
    source = _build_module(candidate_seed=701)
    reference = torch.zeros(2, 192)

    sample_first_q_candidates_v1_c2(
        reference,
        candidate_count=5,
        rollout_horizon=5,
        initial_variance=1.0,
        generator=source.candidate_generator,
    )
    checkpoint: dict[str, object] = {}
    source.on_save_checkpoint(checkpoint)
    expected = sample_first_q_candidates_v1_c2(
        reference,
        candidate_count=5,
        rollout_horizon=5,
        initial_variance=1.0,
        generator=source.candidate_generator,
    )

    resumed = _build_module(candidate_seed=999)
    resumed.on_load_checkpoint(checkpoint)
    actual = sample_first_q_candidates_v1_c2(
        reference,
        candidate_count=5,
        rollout_horizon=5,
        initial_variance=1.0,
        generator=resumed.candidate_generator,
    )

    assert "v1_c2_candidate_generator_state" in checkpoint
    assert torch.equal(actual, expected)


def test_c2_parent_loader_validates_and_hashes_all_three_model_states(
    tmp_path: Path,
) -> None:
    parent_protocol = load_actor_free_td_lewm_v1_c_training_protocol(
        ROOT / "configs" / "experiment" / "actor_free_td_lewm_v1_c_cube_train.yaml"
    )
    parent_module = _build_v1_training_module(
        _FrozenWorld(),
        parent_protocol,
        total_steps=1,
        spec=V1_SPECS["c"],
        data_generator=torch.Generator().manual_seed(1),
        goal_generator=torch.Generator().manual_seed(2),
        task_generator=torch.Generator().manual_seed(3),
        neighbor_index=None,
    )
    payload = _deployment_payload(
        parent_module,
        protocol=parent_protocol,
        spec=V1_SPECS["c"],
        model_config={
            "_target_": "unused.in.payload.validation",
            "action_encoder": {"input_dim": 25, "emb_dim": 192},
        },
        initialization_info={"source_checkpoint_sha256": "0" * 64},
        epoch=10,
        global_step=127960,
    )
    checkpoint = tmp_path / "v1-c-epoch-10.pt"
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    c2_protocol = load_actor_free_td_lewm_v1_c2_training_protocol(CONFIG_PATH)
    c2_protocol["source_v1_c"]["checkpoint_sha256"] = digest

    restored, provenance = _load_v1_c_parent_payload(
        checkpoint,
        protocol=c2_protocol,
    )

    assert restored.keys() >= {
        "world_model_state_dict",
        "predictor_state_dict",
        "target_predictor_state_dict",
    }
    assert provenance["parent_checkpoint_sha256"] == digest
    assert provenance["parent_epoch"] == 10
    assert provenance["parent_global_step"] == 127960
    for key in (
        "world_model_state_sha256",
        "online_predictor_state_sha256",
        "target_predictor_state_sha256",
    ):
        assert len(provenance[key]) == 64
