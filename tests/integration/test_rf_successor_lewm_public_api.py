from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.adapters.rf_successor_lewm import RewardFreeSuccessorLeWM
from tdwm.methods.rf_successor_lewm import ActionPrefixSuccessorHead
from tdwm.training.rf_successor_lewm import (
    _build_training_module,
    load_rf_successor_training_protocol,
)

swm = pytest.importorskip("stable_worldmodel")


class TinyEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, embed_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        pooled = pixels.mean(dim=(-2, -1))
        token = self.projection(pooled).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=token)


class TinyPredictor(nn.Module):
    num_frames = 3

    def forward(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def test_public_lewm_0_1_1_accepts_masked_one_frame_planning_history():
    torch.manual_seed(5)
    embed_dim = 4
    action_dim = 2
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(action_dim, embed_dim),
    )
    successor = ActionPrefixSuccessorHead(
        embed_dim=embed_dim,
        action_dim=action_dim,
        history_size=3,
        hidden_dim=8,
        masked_history=True,
    )
    method = RewardFreeSuccessorLeWM(
        world_model,
        successor,
        max_horizon=4,
        successor_weight=1.0,
        terminal_weight=0.25,
    )
    info = {
        "pixels": torch.randn(2, 3, 1, 3, 8, 8),
        "goal": torch.randn(2, 3, 1, 3, 8, 8),
    }
    candidates = torch.randn(2, 3, 4, action_dim)

    cost = method.get_cost(info, candidates)

    assert cost.shape == (2, 3)
    assert torch.isfinite(cost).all()


def test_joint_training_loss_backpropagates_through_public_lewm():
    torch.manual_seed(6)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    world_model.predictor.num_frames = 2
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_lewm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"]["hidden_dim"] = 8
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    assert module.successor.masked_history is True
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in module.model.parameters())
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )
    assert all(
        parameter.grad is None for parameter in module.target_model.parameters()
    )


def test_s_only_training_uses_public_encoder_without_lewm_dynamics_loss():
    torch.manual_seed(7)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_sequence_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"]["hidden_dim"] = 8
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert not hasattr(module, "target_model")
    assert module.model.encoder.projection.weight.grad is not None
    assert module.model.action_encoder.weight.grad is None
    assert module.model.action_encoder.weight.requires_grad is False
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )


def test_ema_manifold_training_keeps_target_encoder_frozen():
    torch.manual_seed(11)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_ema_manifold_prefix_successor_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"].update(
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=12,
        predictor_depth=1,
        predictor_mlp_dim=16,
        fusion_dim=12,
        dropout=0.0,
        max_horizon=2,
    )
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert module.use_ema_target is True
    assert module.model.encoder.projection.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in module.target_model.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )


def test_anchored_e2e_training_updates_student_but_not_frozen_teacher():
    torch.manual_seed(12)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
        projector=nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_anchored_e2e_manifold_prefix_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"].update(
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=12,
        predictor_depth=1,
        predictor_mlp_dim=16,
        fusion_dim=12,
        dropout=0.0,
        max_horizon=2,
    )
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    logged = {}
    module.log_dict = lambda metrics, *args, **kwargs: logged.update(metrics)
    module.train()
    teacher_before = {
        key: value.detach().clone()
        for key, value in module.target_model.state_dict().items()
    }
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()
    module.on_train_batch_end(None, None, 0)
    optimizer = module.configure_optimizers()["optimizer"]

    assert torch.isfinite(loss)
    assert module.use_frozen_teacher is True
    assert module.use_ema_target is False
    assert module.model.training is False
    assert logged["train/geometry_anchor_loss"].item() == 0.0
    assert module.model.encoder.projection.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in module.target_model.parameters()
    )
    assert all(
        torch.equal(value, teacher_before[key])
        for key, value in module.target_model.state_dict().items()
    )
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )
    assert len(optimizer.param_groups) == 2


def test_manifold_head_refinement_freezes_geometry_but_updates_head():
    torch.manual_seed(13)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_manifold_prefix_successor_wm_cube_head_refine.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"].update(
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=12,
        predictor_depth=1,
        predictor_mlp_dim=16,
        fusion_dim=12,
        dropout=0.0,
        max_horizon=2,
    )
    protocol["loss"]["sigreg"].update(knots=3, num_projections=4)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    module._freeze_world_model()
    module.train()
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert module.model.training is False
    assert all(
        not parameter.requires_grad for parameter in module.model.parameters()
    )
    assert all(parameter.grad is None for parameter in module.model.parameters())
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )


def test_frozen_pretrained_manifold_trains_only_the_prefix_head():
    torch.manual_seed(17)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/"
        "rf_frozen_manifold_prefix_successor_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"].update(
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=12,
        predictor_depth=1,
        predictor_mlp_dim=16,
        fusion_dim=12,
        dropout=0.0,
        max_horizon=2,
    )
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    module.log_dict = lambda *args, **kwargs: None
    module.train()
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()
    optimizer = module.configure_optimizers()["optimizer"]

    assert torch.isfinite(loss)
    assert module.sigreg is None
    assert module.model.training is False
    assert all(
        not parameter.requires_grad for parameter in module.model.parameters()
    )
    assert all(parameter.grad is None for parameter in module.model.parameters())
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )
    assert len(optimizer.param_groups) == 1


def test_frozen_pretrained_residual_trains_only_the_zero_safe_head():
    torch.manual_seed(31)
    world_model = swm.wm.LeWM(
        encoder=TinyEncoder(embed_dim=4),
        predictor=TinyPredictor(),
        action_encoder=nn.Linear(2, 4),
    )
    world_model.predictor.num_frames = 2
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_frozen_residual_prefix_wm_cube_train.yaml"
    )
    protocol["sequence"].update(
        history_frames=2,
        rollout_horizon=2,
        num_steps=5,
    )
    protocol["model"]["embed_dim"] = 4
    protocol["successor"].update(
        prefix_depth=1,
        prefix_heads=2,
        prefix_mlp_dim=12,
        predictor_depth=1,
        predictor_mlp_dim=16,
        fusion_dim=12,
        dropout=0.0,
        max_horizon=2,
    )
    module = _build_training_module(
        world_model,
        protocol,
        total_steps=2,
        action_block_dim=2,
        device_image_preprocessing=False,
    )
    logged = {}
    module.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    module.train()
    batch = {
        "pixels": torch.randn(2, 5, 3, 8, 8),
        "action": torch.randn(2, 5, 2),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()
    optimizer = module.configure_optimizers()["optimizer"]

    assert torch.isfinite(loss)
    assert module.sigreg is None
    assert module.model.training is False
    assert all(
        not parameter.requires_grad for parameter in module.model.parameters()
    )
    assert all(parameter.grad is None for parameter in module.model.parameters())
    assert module.successor.correction_out.weight.grad is not None
    assert len(optimizer.param_groups) == 1
    assert torch.allclose(
        logged["train/latent_mse_h1"],
        logged["train/base_latent_mse_h1"],
    )
