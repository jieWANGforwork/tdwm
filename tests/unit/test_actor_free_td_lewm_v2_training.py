from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1
from tdwm.training.actor_free_td_lewm_v2 import (
    V1_SOURCE_GLOBAL_STEP,
    V1_SOURCE_SHA256,
    V2_SPECS,
    V2Initialization,
    _build_v2_training_module,
    build_hybrid_tdjepa_td_batch_v2,
    load_actor_free_td_lewm_v2_training_protocol,
    load_v2_initialization,
    sample_matched_future_goals_v2,
    validate_actor_free_td_lewm_v2_training_protocol,
)
from tdwm.training.frozen_actor_free_td_v1 import (
    V1_SPECS,
    load_actor_free_td_lewm_v1_training_protocol,
)
from tdwm.training.frozen_actor_free_td_v1 import (
    _predictor_config as _v1_predictor_config,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
TEST_PROTOCOL_SHA256 = "1" * 64
TEST_V2_START_REVISION = "2" * 40
TEST_NEIGHBOR_SHA256 = "3" * 64


def _v2_protocol(variant: str = "c") -> dict:
    return load_actor_free_td_lewm_v2_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_{variant}_cube_train.yaml",
        spec=V2_SPECS[variant],
    )


def _v1_payload(variant: str = "c") -> dict:
    protocol = load_actor_free_td_lewm_v1_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v1_{variant}_cube_train.yaml",
        spec=V1_SPECS[variant],
    )
    predictor = ActorFreeTDJEPAPredictorV1()
    target = predictor.make_target()
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.fill_(0.01)
        for parameter in target.parameters():
            parameter.fill_(0.02)
    return {
        "method": V1_SPECS[variant].method,
        "method_family": "actor_free_td_lewm_v1",
        "variant": variant,
        "implementation_version": "v1",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": V1_SOURCE_GLOBAL_STEP,
        "world_model_state_dict": {"source": torch.tensor(1.0)},
        "world_model_config": {
            "_target_": "example.WorldModel",
            "action_encoder": {"input_dim": 25, "emb_dim": 192},
        },
        "predictor_state_dict": predictor.state_dict(),
        "target_predictor_state_dict": target.state_dict(),
        "predictor_config": _v1_predictor_config(
            protocol,
            spec=V1_SPECS[variant],
        ),
        # A deployment artifact may be accompanied by optimizer history, but
        # V2 initialization deliberately consumes model parameters only.
        "optimizer_states": [{"state": {"must_not_resume": True}}],
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_v2_protocols_lock_the_coupled_finetune_contract(variant: str) -> None:
    protocol = _v2_protocol(variant)

    assert protocol["method"] == V2_SPECS[variant].method
    assert protocol["source_v1"]["method"] == V1_SPECS[variant].method
    assert protocol["source_v1"]["checkpoint_sha256"] == (V1_SOURCE_SHA256[variant])
    assert protocol["seeds"] == [3072]
    assert protocol["world_model"]["online"]["full_lewm_trainable"] is True
    assert protocol["world_model"]["target"]["tracks_full_world_model"] is True
    assert protocol["predictor"]["action_processing"] == (
        "online_shared_lewm_action_encoder"
    )
    assert protocol["joint_objective"]["predicted_context_detach"] is False
    assert protocol["joint_objective"]["hybrid_reduction"] == "sum"
    assert protocol["optimizer"]["initialize_state"] == "fresh"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("seeds",), [0, 42, 3072], "only archived matching V1 seed"),
        (
            ("predictor", "action_processing"),
            "raw_action_concat",
            "action_processing",
        ),
        (
            ("joint_objective", "predicted_context_detach"),
            True,
            "predicted_context_detach",
        ),
        (
            ("joint_objective", "hybrid_reduction"),
            "mean",
            "hybrid_reduction",
        ),
        (("optimizer", "initialize_state"), "resume", "optimizer.initialize_state"),
        (
            ("training", "optimizer_steps_per_epoch"),
            1,
            "exactly 127960 new updates",
        ),
    ),
)
def test_v2_protocol_validation_fails_closed_on_stage_drift(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    protocol = _v2_protocol("c")
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_actor_free_td_lewm_v2_training_protocol(
            protocol,
            spec=V2_SPECS["c"],
        )


def test_v2_protocol_inheritance_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n")
    second.write_text("extends: first.yaml\n")

    with pytest.raises(ValueError, match="inheritance contains a cycle"):
        load_actor_free_td_lewm_v2_training_protocol(
            first,
            spec=V2_SPECS["c"],
        )


def test_v2_initialization_accepts_only_matching_completed_v1_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _v1_payload("c")
    checkpoint = tmp_path / "v1-c.pt"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(
        "tdwm.training.actor_free_td_lewm_v2._file_sha256",
        lambda _path: V1_SOURCE_SHA256["c"],
    )

    initialization = load_v2_initialization(
        checkpoint,
        spec=V2_SPECS["c"],
        protocol=_v2_protocol("c"),
    )

    for field in ("predictor_state_dict", "target_predictor_state_dict"):
        assert initialization.payload[field].keys() == payload[field].keys()
        for key, value in payload[field].items():
            torch.testing.assert_close(initialization.payload[field][key], value)
    assert initialization.payload["optimizer_states"] == (payload["optimizer_states"])
    assert initialization.checkpoint_sha256 == V1_SOURCE_SHA256["c"]
    assert initialization.predictor_config["variant"] == "c"

    incomplete = deepcopy(payload)
    incomplete["global_step"] -= 1
    torch.save(incomplete, checkpoint)
    with pytest.raises(ValueError, match="completed 127960-step V1"):
        load_v2_initialization(
            checkpoint,
            spec=V2_SPECS["c"],
            protocol=_v2_protocol("c"),
        )

    wrong_variant = _v1_payload("d")
    torch.save(wrong_variant, checkpoint)
    with pytest.raises(ValueError, match="checkpoint.method"):
        load_v2_initialization(
            checkpoint,
            spec=V2_SPECS["c"],
            protocol=_v2_protocol("c"),
        )


def test_v2_initialization_checks_locked_sha_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "not-a-checkpoint.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    monkeypatch.setattr(
        "tdwm.training.actor_free_td_lewm_v2._file_sha256",
        lambda _path: "0" * 64,
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "hash mismatch must fail before deserialization"
        ),
    )

    with pytest.raises(ValueError, match="locked SHA-256"):
        load_v2_initialization(
            checkpoint,
            spec=V2_SPECS["c"],
            protocol=_v2_protocol("c"),
        )


def test_matched_future_goals_are_ema_detached_and_transition_aligned() -> None:
    latents = torch.arange(2 * 19, dtype=torch.float32).reshape(2, 19, 1)
    latents = latents.expand(-1, -1, 192).clone().requires_grad_(True)
    terminals = torch.zeros(2, 19, dtype=torch.bool)

    goals, offsets = sample_matched_future_goals_v2(
        latents,
        terminals,
        first_current_index=3,
        generator=torch.Generator().manual_seed(71),
    )

    current = torch.arange(3, 18).unsqueeze(0)
    expected_indices = current + offsets.cpu()
    expected = latents.detach().gather(
        1,
        expected_indices.unsqueeze(-1).expand(2, 15, 192),
    )
    assert goals.shape == (2, 15, 192)
    assert offsets.shape == (2, 15)
    assert torch.equal(goals, expected)
    assert not goals.requires_grad
    assert not offsets.requires_grad


class _ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(action))


class _TrainableWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = nn.Linear(3, 192)
        self.action_encoder = _ActionEncoder()
        self.forward_model = nn.Linear(384, 192)

    def encode(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_features = data["pixels"].mean(dim=(-1, -2))
        return {
            "emb": torch.tanh(self.visual_encoder(image_features)),
            "act_emb": self.action_encoder(data["action"]),
        }

    def predict(
        self,
        history: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            self.forward_model(torch.cat((history, action_embedding), dim=-1))
        )


class _DifferentiableSIGReg(nn.Module):
    def __init__(self, *, knots: int, num_proj: int) -> None:
        super().__init__()
        self.knots = knots
        self.num_proj = num_proj

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.square().mean()


class _NeighborIndex:
    def lookup(
        self,
        global_rows: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SimpleNamespace:
        count = int(global_rows.numel())
        return SimpleNamespace(
            actions=torch.randn(count, 2, 25, device=device, dtype=dtype),
            distances=torch.ones(count, 2, device=device, dtype=torch.float32),
            neighbor_rows=torch.zeros(count, 2, device=device, dtype=torch.int64),
        )


def _v2_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    variant: str = "c",
    protocol_sha256: str = TEST_PROTOCOL_SHA256,
    v2_start_revision: str = TEST_V2_START_REVISION,
    neighbor_index_manifest_sha256: str | None = None,
):
    import stable_worldmodel as swm

    monkeypatch.setattr(swm.wm, "SIGReg", _DifferentiableSIGReg)
    payload = _v1_payload(variant)
    initialization = V2Initialization(
        payload=payload,
        checkpoint_path=f"/locked/v1-{variant}.pt",
        checkpoint_sha256=V1_SOURCE_SHA256[variant],
        predictor_config=payload["predictor_config"],
    )
    if variant == "g1" and neighbor_index_manifest_sha256 is None:
        neighbor_index_manifest_sha256 = TEST_NEIGHBOR_SHA256
    return _build_v2_training_module(
        _TrainableWorld(),
        initialization,
        _v2_protocol(variant),
        total_steps=20,
        spec=V2_SPECS[variant],
        data_generator=torch.Generator().manual_seed(1),
        goal_generator=torch.Generator().manual_seed(2),
        task_generator=torch.Generator().manual_seed(3),
        validation_goal_generator=torch.Generator().manual_seed(4),
        validation_task_generator=torch.Generator().manual_seed(5),
        neighbor_index=_NeighborIndex() if variant == "g1" else None,
        protocol_sha256=protocol_sha256,
        v2_start_revision=v2_start_revision,
        neighbor_index_manifest_sha256=neighbor_index_manifest_sha256,
        device_image_preprocessing=False,
    )


def test_v2_module_restores_both_v1_gs_but_starts_a_fresh_joint_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _v2_module(monkeypatch)
    source = _v1_payload("c")

    for key, value in module.predictor.state_dict().items():
        torch.testing.assert_close(value, source["predictor_state_dict"][key])
    for key, value in module.target_predictor.state_dict().items():
        torch.testing.assert_close(
            value,
            source["target_predictor_state_dict"][key],
        )
    for key, value in module.model.state_dict().items():
        torch.testing.assert_close(value, module.target_model.state_dict()[key])

    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {
        *(id(parameter) for parameter in module.model.parameters()),
        *(id(parameter) for parameter in module.predictor.parameters()),
    }
    assert optimizer.state == {}
    assert optimized.isdisjoint(
        {id(parameter) for parameter in module.target_model.parameters()}
    )
    assert optimized.isdisjoint(
        {id(parameter) for parameter in module.target_predictor.parameters()}
    )
    assert all(parameter.requires_grad for parameter in module.model.parameters())
    assert all(parameter.requires_grad for parameter in module.predictor.parameters())
    assert not any(
        parameter.requires_grad for parameter in module.target_model.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in module.target_predictor.parameters()
    )
    module.train()
    assert module.model.training
    assert module.predictor.training
    assert not module.target_model.training
    assert not module.target_predictor.training


def test_v2_lightning_checkpoint_embeds_and_requires_exact_resume_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _v2_module(monkeypatch)
    checkpoint: dict[str, object] = {}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["v2_resume_identity"] == {
        "schema_version": 1,
        "method": "actor_free_td_lewm_v2_c",
        "method_family": "actor_free_td_lewm_v2",
        "variant": "c",
        "implementation_version": "v2",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "protocol_sha256": TEST_PROTOCOL_SHA256,
        "source_v1_sha256": V1_SOURCE_SHA256["c"],
        "v2_start_revision": TEST_V2_START_REVISION,
        "neighbor_index_manifest_sha256": None,
    }
    module.on_load_checkpoint(checkpoint)

    missing = dict(checkpoint)
    missing.pop("v2_resume_identity")
    with pytest.raises(RuntimeError, match="missing its embedded identity"):
        module.on_load_checkpoint(missing)


def test_v2_resume_rejects_another_methods_shape_compatible_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _v2_module(monkeypatch, variant="d")
    destination = _v2_module(monkeypatch, variant="c")
    checkpoint: dict[str, object] = {}
    source.on_save_checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="checkpoint method differs"):
        destination.on_load_checkpoint(checkpoint)


def test_v2_resume_rejects_a_replaced_variant_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _v2_module(monkeypatch)
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    changed = dict(checkpoint)
    changed_identity = dict(checkpoint["v2_resume_identity"])
    changed_identity["variant"] = "d"
    changed["v2_resume_identity"] = changed_identity

    with pytest.raises(RuntimeError, match="checkpoint variant differs"):
        module.on_load_checkpoint(changed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("protocol_sha256", "4" * 64),
        ("source_v1_sha256", "5" * 64),
        ("v2_start_revision", "6" * 40),
    ),
)
def test_v2_resume_rejects_protocol_source_or_revision_replacement(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    module = _v2_module(monkeypatch)
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    changed = dict(checkpoint)
    changed_identity = dict(checkpoint["v2_resume_identity"])
    changed_identity[field] = replacement
    changed["v2_resume_identity"] = changed_identity

    with pytest.raises(RuntimeError, match=rf"checkpoint {field} differs"):
        module.on_load_checkpoint(changed)


def test_v2_g1_resume_binds_the_neighbor_index_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _v2_module(monkeypatch, variant="g1")
    destination = _v2_module(
        monkeypatch,
        variant="g1",
        neighbor_index_manifest_sha256="7" * 64,
    )
    checkpoint: dict[str, object] = {}
    source.on_save_checkpoint(checkpoint)

    with pytest.raises(
        RuntimeError,
        match="checkpoint neighbor_index_manifest_sha256 differs",
    ):
        destination.on_load_checkpoint(checkpoint)


def test_v2_full_loss_backpropagates_through_online_lewm_f_ea_and_shared_g_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(79)
    module = _v2_module(monkeypatch)
    module.train()
    batch = {
        "pixels": torch.randn(2, 19, 3, 2, 2),
        "action": torch.randn(2, 19, 25),
        "_tdwm_global_start": torch.tensor([100, 500], dtype=torch.int64),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert module.model.visual_encoder.weight.grad is not None
    assert torch.count_nonzero(module.model.visual_encoder.weight.grad) > 0
    assert module.model.action_encoder.projection.weight.grad is not None
    assert torch.count_nonzero(module.model.action_encoder.projection.weight.grad) > 0
    assert module.model.forward_model.weight.grad is not None
    assert torch.count_nonzero(module.model.forward_model.weight.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in module.predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in module.target_model.parameters())
    assert all(
        parameter.grad is None for parameter in module.target_predictor.parameters()
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_v2_all_variants_align_g_context_under_cpu_bfloat16_autocast(
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(83)
    module = _v2_module(monkeypatch, variant=variant)
    module.train()
    batch = {
        "pixels": torch.randn(2, 19, 3, 2, 2),
        "action": torch.randn(2, 19, 25),
        "_tdwm_global_start": torch.tensor([100, 500], dtype=torch.int64),
    }
    observed: dict[str, torch.Tensor] = {}
    original = build_hybrid_tdjepa_td_batch_v2
    original_goal_sampler = sample_matched_future_goals_v2

    def sample_float32_goals(*args, **kwargs):
        goals, offsets = original_goal_sampler(*args, **kwargs)
        return goals.float(), offsets

    def record_context(*args, **kwargs):
        names = (
            "real_state",
            "predicted_state",
            "action_embedding",
            "task",
            "ema_next_state",
            "target_next_action_embedding",
        )
        observed.update(zip(names, args[2:8], strict=True))
        result = original(*args, **kwargs)
        observed["td_target"] = result.target
        return result

    monkeypatch.setattr(
        "tdwm.training.actor_free_td_lewm_v2.build_hybrid_tdjepa_td_batch_v2",
        record_context,
    )
    monkeypatch.setattr(
        "tdwm.training.actor_free_td_lewm_v2.sample_matched_future_goals_v2",
        sample_float32_goals,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = module._forward_loss(batch, "train")

    assert torch.isfinite(loss)
    for name in (
        "real_state",
        "predicted_state",
        "action_embedding",
        "task",
        "ema_next_state",
        "target_next_action_embedding",
    ):
        assert observed[name].dtype == torch.bfloat16
        assert observed[name].device == observed["real_state"].device
    assert not observed["task"].requires_grad
    assert observed["td_target"].dtype == torch.float32
    assert not observed["td_target"].requires_grad
