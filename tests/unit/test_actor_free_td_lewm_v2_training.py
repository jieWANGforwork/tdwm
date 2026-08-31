from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1
from tdwm.training.actor_free_td_lewm_v2 import (
    EMA_LOCAL_PREDICTION,
    EMA_LOCAL_PREDICTION_TARGET,
    EMA_LOCAL_PREDICTION_TARGET_GRADIENT,
    V1_SOURCE_GLOBAL_STEP,
    V1_SOURCE_SHA256,
    V2_SPECS,
    ActorFreeTDLeWMV2Spec,
    V2Initialization,
    _build_v2_training_module,
    _deployment_payload,
    _local_prediction_objective_v2,
    _predictor_config,
    _validate_v2_resume_manifest,
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


def _ema_sg_spec(variant: str = "c") -> ActorFreeTDLeWMV2Spec:
    family = "actor_free_td_lewm_v2_ema_sg"
    return ActorFreeTDLeWMV2Spec(
        method=f"{family}_{variant}",
        variant=variant,
        requires_neighbor_index=variant == "g1",
        method_family=family,
        implementation_version="v2_ema_sg",
        stage="coupled_hybrid_ema_target_finetuning",
        local_prediction=EMA_LOCAL_PREDICTION,
        local_prediction_target=EMA_LOCAL_PREDICTION_TARGET,
        local_prediction_target_gradient=(
            EMA_LOCAL_PREDICTION_TARGET_GRADIENT
        ),
    )


def _v2_protocol(variant: str = "c") -> dict:
    return load_actor_free_td_lewm_v2_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_{variant}_cube_train.yaml",
        spec=V2_SPECS[variant],
    )


def _ema_sg_protocol(variant: str = "c") -> dict:
    spec = _ema_sg_spec(variant)
    protocol = _v2_protocol(variant)
    protocol.update(
        {
            "method": spec.method,
            "method_family": spec.method_family,
            "implementation_version": spec.implementation_version,
            "stage": spec.stage,
            "initialization": spec.initialization,
        }
    )
    protocol["joint_objective"].update(
        {
            "local_prediction": spec.local_prediction,
            "local_prediction_target": spec.local_prediction_target,
            "local_prediction_target_gradient": (
                spec.local_prediction_target_gradient
            ),
        }
    )
    return protocol


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


def test_v2_ema_sg_protocol_locks_all_three_local_target_fields() -> None:
    spec = _ema_sg_spec()
    protocol = _ema_sg_protocol()

    validate_actor_free_td_lewm_v2_training_protocol(protocol, spec=spec)
    for key in (
        "local_prediction_target",
        "local_prediction_target_gradient",
    ):
        drifted = deepcopy(protocol)
        drifted["joint_objective"].pop(key)
        with pytest.raises(ValueError, match=key):
            validate_actor_free_td_lewm_v2_training_protocol(
                drifted,
                spec=spec,
            )


def test_original_v2_protocol_keeps_its_legacy_online_target_contract() -> None:
    protocol = _v2_protocol("c")

    assert "local_prediction_target" not in protocol["joint_objective"]
    assert "local_prediction_target_gradient" not in protocol["joint_objective"]
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


def test_original_v2_local_mse_keeps_online_target_gradient_and_diagnostics() -> None:
    prediction = torch.tensor([1.0, 4.0], requires_grad=True)
    online_target = torch.tensor([3.0, 2.0], requires_grad=True)
    ema_target = torch.tensor([2.0, 8.0], requires_grad=True)

    objective = _local_prediction_objective_v2(
        prediction,
        online_target,
        ema_target,
        spec=V2_SPECS["c"],
    )

    torch.testing.assert_close(objective.loss, torch.tensor(4.0))
    torch.testing.assert_close(
        objective.online_reference_mse,
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        objective.online_ema_latent_drift,
        torch.tensor(18.5),
    )
    assert not objective.online_reference_mse.requires_grad
    assert not objective.online_ema_latent_drift.requires_grad

    objective.loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 2.0]))
    torch.testing.assert_close(online_target.grad, torch.tensor([2.0, -2.0]))
    assert ema_target.grad is None


def test_ema_sg_local_mse_uses_detached_ema_target_with_exact_diagnostics() -> None:
    prediction = torch.tensor([1.0, 4.0], requires_grad=True)
    online_target = torch.tensor([3.0, 2.0], requires_grad=True)
    ema_target = torch.tensor([2.0, 8.0], requires_grad=True)

    objective = _local_prediction_objective_v2(
        prediction,
        online_target,
        ema_target,
        spec=_ema_sg_spec(),
    )

    torch.testing.assert_close(objective.loss, torch.tensor(8.5))
    torch.testing.assert_close(
        objective.online_reference_mse,
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        objective.online_ema_latent_drift,
        torch.tensor(18.5),
    )
    assert not objective.online_reference_mse.requires_grad
    assert not objective.online_ema_latent_drift.requires_grad

    objective.loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([-1.0, -4.0]))
    assert online_target.grad is None
    assert ema_target.grad is None


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


def _v2_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: ActorFreeTDLeWMV2Spec | None = None,
    protocol: dict | None = None,
):
    import stable_worldmodel as swm

    monkeypatch.setattr(swm.wm, "SIGReg", _DifferentiableSIGReg)
    spec = V2_SPECS["c"] if spec is None else spec
    protocol = _v2_protocol("c") if protocol is None else protocol
    payload = _v1_payload("c")
    initialization = V2Initialization(
        payload=payload,
        checkpoint_path="/locked/v1-c.pt",
        checkpoint_sha256=V1_SOURCE_SHA256["c"],
        predictor_config=payload["predictor_config"],
    )
    return _build_v2_training_module(
        _TrainableWorld(),
        initialization,
        protocol,
        total_steps=20,
        spec=spec,
        data_generator=torch.Generator().manual_seed(1),
        goal_generator=torch.Generator().manual_seed(2),
        task_generator=torch.Generator().manual_seed(3),
        validation_goal_generator=torch.Generator().manual_seed(4),
        validation_task_generator=torch.Generator().manual_seed(5),
        neighbor_index=None,
        device_image_preprocessing=False,
    )


def test_ema_sg_payload_and_resume_identity_come_from_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _ema_sg_spec()
    protocol = _ema_sg_protocol()
    module = _v2_module(monkeypatch)
    source = _v1_payload("c")
    initialization = V2Initialization(
        payload=source,
        checkpoint_path="/locked/v1-c.pt",
        checkpoint_sha256=V1_SOURCE_SHA256["c"],
        predictor_config=source["predictor_config"],
    )

    predictor_config = _predictor_config(protocol, spec=spec)
    payload = _deployment_payload(
        module,
        protocol=protocol,
        spec=spec,
        world_model_config=source["world_model_config"],
        initialization=initialization,
        epoch=1,
        global_step=10,
    )
    for value in (predictor_config, payload):
        assert value["method"] == spec.method
        assert value["method_family"] == spec.method_family
        assert value["implementation_version"] == spec.implementation_version
        assert value["stage"] == spec.stage
        assert value["initialization"] == spec.initialization
    assert predictor_config["joint_objective"]["local_prediction"] == (
        EMA_LOCAL_PREDICTION
    )

    split = {
        "train_indices_sha256": "train",
        "validation_indices_sha256": "validation",
    }
    manifest = {
        "method": spec.method,
        "method_family": spec.method_family,
        "variant": spec.variant,
        "implementation_version": spec.implementation_version,
        "objective_version": spec.objective_version,
        "deployment_checkpoint_version": spec.deployment_checkpoint_version,
        "stage": spec.stage,
        "initialization": spec.initialization,
        "protocol_sha256": "protocol",
        "seed": 3072,
        "source_v1": {"checkpoint_sha256": V1_SOURCE_SHA256["c"]},
        "dataset": {"split": split},
        "neighbor_index": None,
    }
    _validate_v2_resume_manifest(
        manifest,
        spec=spec,
        protocol_sha256="protocol",
        seed=3072,
        split_manifest=split,
        initialization=initialization,
        neighbor_info=None,
    )
    manifest["method_family"] = "actor_free_td_lewm_v2"
    with pytest.raises(RuntimeError, match="incompatible V2 run"):
        _validate_v2_resume_manifest(
            manifest,
            spec=spec,
            protocol_sha256="protocol",
            seed=3072,
            split_manifest=split,
            initialization=initialization,
            neighbor_info=None,
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


def test_ema_sg_module_uses_shifted_ema_latents_but_keeps_online_prediction_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(83)
    protocol = _ema_sg_protocol()
    protocol["loss"]["sigreg"]["weight"] = 0.0
    module = _v2_module(
        monkeypatch,
        spec=_ema_sg_spec(),
        protocol=protocol,
    )
    module.train()
    with torch.no_grad():
        module.target_model.visual_encoder.bias.add_(0.75)
    monkeypatch.setattr(module, "_auxiliary_scale", lambda: 0.0)
    captured: dict[str, torch.Tensor] = {}

    def capture_metrics(metrics: dict[str, torch.Tensor], **_kwargs: object) -> None:
        captured.update(metrics)

    monkeypatch.setattr(module, "log_dict", capture_metrics)
    batch = {
        "pixels": torch.randn(2, 19, 3, 2, 2),
        "action": torch.randn(2, 19, 25),
        "_tdwm_global_start": torch.tensor([100, 500], dtype=torch.int64),
    }
    with torch.no_grad():
        online = module.model.encode(batch)
        ema = module.target_model.encode(batch)
        histories = torch.cat(
            [online["emb"][:, start : start + 3] for start in range(16)],
            dim=0,
        )
        actions = torch.cat(
            [online["act_emb"][:, start : start + 3] for start in range(16)],
            dim=0,
        )
        prediction = module.model.predict(histories, actions)
        online_targets = torch.cat(
            [online["emb"][:, start + 1 : start + 4] for start in range(16)],
            dim=0,
        )
        ema_targets = torch.cat(
            [ema["emb"][:, start + 1 : start + 4] for start in range(16)],
            dim=0,
        )
        expected_loss = (prediction - ema_targets).square().mean()
        expected_online_reference = (
            prediction - online_targets
        ).square().mean()
        expected_drift = (online_targets - ema_targets).square().mean()

    loss = module._forward_loss(batch, "train")

    torch.testing.assert_close(loss.detach(), expected_loss)
    torch.testing.assert_close(
        captured["train/prediction_online_reference_mse"],
        expected_online_reference,
    )
    torch.testing.assert_close(
        captured["train/online_ema_latent_drift"],
        expected_drift,
    )
    assert not captured["train/prediction_online_reference_mse"].requires_grad
    assert not captured["train/online_ema_latent_drift"].requires_grad

    loss.backward()
    assert torch.count_nonzero(module.model.visual_encoder.weight.grad) > 0
    assert torch.count_nonzero(module.model.action_encoder.projection.weight.grad) > 0
    assert torch.count_nonzero(module.model.forward_model.weight.grad) > 0
    assert all(parameter.grad is None for parameter in module.target_model.parameters())


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
