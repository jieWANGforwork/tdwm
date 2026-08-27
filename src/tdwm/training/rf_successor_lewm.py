"""Joint multi-horizon training for reward-free Successor-LeWM."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.rf_successor_lewm import (
    ActionPrefixMomentHead,
    ActionPrefixSuccessorHead,
    LeWMResidualTransformerHead,
    ManifoldTransformerMomentHead,
    balanced_successor_mse,
    finite_horizon_successor_targets,
    left_pad_latent_history,
    manifold_sequence_objective,
    moment_sequence_objective,
    residual_manifold_sequence_objective,
    successor_recurrence_residual,
    successor_sequence_objective,
)
from tdwm.training.block_sampler import BlockShuffleBatchSampler
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.gt_lewm_support import (
    LeWMTransform,
    build_metrics_logger,
    build_model_config,
    compile_world_model,
    fit_column_stats,
    preprocess_image_batch,
    resolve_train_batch_limit,
    save_split,
    write_json,
)
from tdwm.training.joint_td_gt_lewm import rollout_from_latents
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision

METHOD = "rf_successor_lewm"
S_ONLY_METHOD = "rf_successor_sequence_wm"
BALANCED_SEQUENCE_METHOD = "rf_balanced_successor_sequence_wm"
EMA_BALANCED_SEQUENCE_METHOD = "rf_ema_balanced_successor_sequence_wm"
DIRECT_MOMENT_METHOD = "rf_direct_moment_sequence_wm"
E2E_MOMENT_METHOD = "rf_e2e_moment_sequence_wm"
MANIFOLD_PREFIX_METHOD = "rf_manifold_prefix_successor_wm"
EMA_MANIFOLD_PREFIX_METHOD = "rf_ema_manifold_prefix_successor_wm"
FROZEN_MANIFOLD_PREFIX_METHOD = "rf_frozen_manifold_prefix_successor_wm"
FROZEN_RESIDUAL_PREFIX_METHOD = "rf_frozen_residual_prefix_wm"
ANCHORED_E2E_MANIFOLD_PREFIX_METHOD = "rf_anchored_e2e_manifold_prefix_wm"
FROZEN_PRETRAINED_METHODS = frozenset(
    (FROZEN_MANIFOLD_PREFIX_METHOD, FROZEN_RESIDUAL_PREFIX_METHOD)
)
PRETRAINED_METHODS = frozenset(
    (*FROZEN_PRETRAINED_METHODS, ANCHORED_E2E_MANIFOLD_PREFIX_METHOD)
)
DIRECT_MOMENT_METHODS = frozenset((DIRECT_MOMENT_METHOD, E2E_MOMENT_METHOD))
MANIFOLD_PREFIX_METHODS = frozenset(
    (
        MANIFOLD_PREFIX_METHOD,
        EMA_MANIFOLD_PREFIX_METHOD,
        FROZEN_MANIFOLD_PREFIX_METHOD,
        FROZEN_RESIDUAL_PREFIX_METHOD,
        ANCHORED_E2E_MANIFOLD_PREFIX_METHOD,
    )
)
SEQUENCE_METHODS = frozenset(
    (
        S_ONLY_METHOD,
        BALANCED_SEQUENCE_METHOD,
        EMA_BALANCED_SEQUENCE_METHOD,
        DIRECT_MOMENT_METHOD,
        E2E_MOMENT_METHOD,
        MANIFOLD_PREFIX_METHOD,
        EMA_MANIFOLD_PREFIX_METHOD,
        FROZEN_MANIFOLD_PREFIX_METHOD,
        FROZEN_RESIDUAL_PREFIX_METHOD,
        ANCHORED_E2E_MANIFOLD_PREFIX_METHOD,
    )
)
SUPPORTED_METHODS = frozenset((METHOD, *SEQUENCE_METHODS))


def load_rf_successor_training_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_rf_successor_training_protocol(protocol)
    return protocol


def validate_rf_successor_training_protocol(protocol: dict[str, Any]) -> None:
    method = protocol.get("method")
    if protocol.get("schema_version") != 1 or method not in SUPPORTED_METHODS:
        raise ValueError("This trainer only accepts supported reward-free schema 1 methods.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "full_training":
        raise ValueError("RF-Successor-LeWM training is locked to full Cube training.")
    training = protocol.get("training", {})
    freeze_after_epoch = training.get("freeze_world_model_after_epoch")
    freeze_from_start = training.get("freeze_world_model_from_start", False)
    if not isinstance(freeze_from_start, bool):
        raise TypeError("training.freeze_world_model_from_start must be boolean.")
    if method in FROZEN_PRETRAINED_METHODS:
        expected_initialization = "frozen_pretrained_lewm"
    elif method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        expected_initialization = "anchored_pretrained_lewm"
    elif freeze_after_epoch is not None:
        expected_initialization = "resume_same_objective_checkpoint"
    else:
        expected_initialization = "random_from_scratch"
    if protocol.get("initialization") != expected_initialization:
        raise ValueError(
            "RF-Successor-LeWM initialization differs from its training stage."
        )
    if freeze_from_start != (method in FROZEN_PRETRAINED_METHODS):
        raise ValueError(
            "Only frozen-pretrained methods freeze the world model from start."
        )
    if freeze_after_epoch is not None and method != MANIFOLD_PREFIX_METHOD:
        raise ValueError("Head-only refinement is locked to the online manifold model.")
    pretrained = protocol.get("pretrained_world_model")
    if method in PRETRAINED_METHODS:
        if not isinstance(pretrained, dict):
            raise ValueError("The pretrained method requires source metadata.")
        if pretrained.get("source_method") != "lewm":
            raise ValueError("The pretrained source must be LeWM.")
        source_hash = pretrained.get("checkpoint_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError("The pretrained LeWM checkpoint hash is invalid.")
        if int(pretrained.get("source_epoch", 0)) <= 0:
            raise ValueError("The pretrained LeWM source epoch is invalid.")
        if method in FROZEN_PRETRAINED_METHODS and pretrained.get("frozen") is not True:
            raise ValueError("The frozen method requires a frozen LeWM source.")
        if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD and (
            pretrained.get("student_frozen") is not False
            or pretrained.get("teacher_frozen") is not True
        ):
            raise ValueError(
                "Anchored end-to-end training requires a trainable student and "
                "frozen teacher."
            )
    elif pretrained is not None:
        raise ValueError("Only pretrained methods accept source metadata.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("RF-Successor-LeWM requires stable-worldmodel 0.1.1.")

    sequence = protocol.get("sequence", {})
    history = int(sequence.get("history_frames", 0))
    horizon = int(sequence.get("rollout_horizon", 0))
    num_steps = int(sequence.get("num_steps", 0))
    if min(history, horizon) <= 0 or num_steps < history + horizon:
        raise ValueError("The clip must cover history plus the rollout horizon.")
    if sequence.get("prediction_frames") != 1:
        raise ValueError("The public LeWM model configuration predicts one frame.")
    if int(sequence.get("frame_skip", 0)) <= 0:
        raise ValueError("sequence.frame_skip must be positive.")

    objective = protocol.get("joint_objective", {})
    if method == METHOD:
        expected = {
            "local_prediction": "original_lewm_one_step_mse",
            "multi_step_prediction": "open_loop_latent_mse_all_horizons",
            "successor": "direct_mc_all_prefix_horizons",
            "consistency": "successor_increment_equals_rollout_feature",
            "target_encoder": "ema_world_model",
            "goal_conditioning": "none",
            "policy": "none",
            "bootstrap": "none",
        }
        for key, value in expected.items():
            if objective.get(key) != value:
                raise ValueError(f"joint_objective.{key} must be {value!r}.")
        if objective.get("local_prediction_weight") != 1.0:
            raise ValueError("The original local LeWM prediction weight remains one.")
        for key in (
            "multi_step_prediction_weight",
            "successor_weight",
            "recurrence_weight",
        ):
            if float(objective.get(key, -1.0)) < 0.0:
                raise ValueError(f"joint_objective.{key} cannot be negative.")
        if float(objective.get("successor_weight", 0.0)) <= 0.0:
            raise ValueError("The direct successor supervision must remain active.")
    else:
        if method == FROZEN_RESIDUAL_PREFIX_METHOD:
            expected = {
                "primitive_prediction": "lewm_residual_future_latent_sequence",
                "single_step": "horizon_one_residual_corrected_latent",
                "multi_step_prediction": "all_horizon_residual_corrected_latents",
                "consistency": "exact_base_plus_residual_successor_cumsum",
                "target_encoder": "frozen_pretrained",
                "base_predictor": "frozen_pretrained_lewm_rollout",
                "goal_conditioning": "none",
                "policy": "none",
                "bootstrap": "none",
            }
            predictive_weight_key = "latent_sequence_weight"
        elif method in MANIFOLD_PREFIX_METHODS:
            expected = {
                "primitive_prediction": "future_latent_sequence",
                "single_step": "horizon_one_latent",
                "multi_step_prediction": "all_horizon_latents",
                "consistency": "exact_manifold_successor_cumsum",
                "target_encoder": (
                    "frozen_teacher_stop_gradient"
                    if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD
                    else (
                        "ema_stop_gradient"
                        if method == EMA_MANIFOLD_PREFIX_METHOD
                        else (
                            "frozen_pretrained"
                            if method in FROZEN_PRETRAINED_METHODS
                            else "online_end_to_end"
                        )
                    )
                ),
                "goal_conditioning": "none",
                "policy": "none",
                "bootstrap": "none",
            }
            if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
                expected.update(
                    {
                        "goal_encoder": "frozen_pretrained_teacher",
                        "geometry_anchor": "student_to_frozen_teacher_mse",
                    }
                )
            predictive_weight_key = "latent_sequence_weight"
        elif method in DIRECT_MOMENT_METHODS:
            expected = {
                "primitive_prediction": "future_moment_sequence",
                "single_step": "horizon_one_moment",
                "multi_step_prediction": "all_horizon_moments",
                "consistency": "architectural_discounted_cumsum",
                "target_encoder": (
                    "online_stop_gradient"
                    if method == DIRECT_MOMENT_METHOD
                    else "online_end_to_end"
                ),
                "goal_conditioning": "none",
                "policy": "none",
                "bootstrap": "none",
            }
            predictive_weight_key = "moment_sequence_weight"
        else:
            expected = {
                "primitive_prediction": "successor_sequence",
                "single_step": "horizon_one_successor",
                "multi_step_prediction": "recovered_from_successor_increments",
                "consistency": "architectural_discounted_cumsum",
                "target_encoder": (
                    "ema_stop_gradient"
                    if method == EMA_BALANCED_SEQUENCE_METHOD
                    else "online_end_to_end"
                ),
                "goal_conditioning": "none",
                "policy": "none",
                "bootstrap": "none",
            }
            predictive_weight_key = "successor_sequence_weight"
        for key, value in expected.items():
            if objective.get(key) != value:
                raise ValueError(f"joint_objective.{key} must be {value!r}.")
        if float(objective.get(predictive_weight_key, -1.0)) != 1.0:
            raise ValueError("The sequence method has one unit-weight predictive loss.")

    successor = protocol.get("successor", {})
    locked = {
        "objective_version": {
            METHOD: 12,
            S_ONLY_METHOD: 2,
            BALANCED_SEQUENCE_METHOD: 3,
            EMA_BALANCED_SEQUENCE_METHOD: 4,
            DIRECT_MOMENT_METHOD: 5,
            E2E_MOMENT_METHOD: 6,
            MANIFOLD_PREFIX_METHOD: 7,
            EMA_MANIFOLD_PREFIX_METHOD: 8,
            FROZEN_MANIFOLD_PREFIX_METHOD: 9,
            FROZEN_RESIDUAL_PREFIX_METHOD: 10,
            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD: 11,
        }[method],
        "architecture": {
            METHOD: "masked_history_causal_gru_action_prefix",
            S_ONLY_METHOD: "causal_gru_successor_increments",
            BALANCED_SEQUENCE_METHOD: "causal_gru_successor_increments",
            EMA_BALANCED_SEQUENCE_METHOD: "causal_gru_successor_increments",
            DIRECT_MOMENT_METHOD: "causal_gru_successor_increments",
            E2E_MOMENT_METHOD: "causal_gru_successor_increments",
            MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
            EMA_MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
            FROZEN_MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
            FROZEN_RESIDUAL_PREFIX_METHOD: "causal_transformer_lewm_residual",
            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
        }[method],
        "feature_basis": "augmented_latent_squared_distance",
        "horizon_normalization": "discounted_prefix_mean",
        "target": {
            METHOD: "direct_monte_carlo",
            S_ONLY_METHOD: "online_direct_monte_carlo",
            BALANCED_SEQUENCE_METHOD: "online_direct_monte_carlo",
            EMA_BALANCED_SEQUENCE_METHOD: "ema_direct_monte_carlo",
            DIRECT_MOMENT_METHOD: "online_stop_gradient_direct_moments",
            E2E_MOMENT_METHOD: "online_end_to_end_direct_moments",
            MANIFOLD_PREFIX_METHOD: "online_end_to_end_latents",
            EMA_MANIFOLD_PREFIX_METHOD: "ema_stop_gradient_latents",
            FROZEN_MANIFOLD_PREFIX_METHOD: "frozen_pretrained_latents",
            FROZEN_RESIDUAL_PREFIX_METHOD: "frozen_pretrained_residual_latents",
            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD: "frozen_teacher_latents",
        }[method],
        "action_conditioning": "causal_prefix",
        "goal_conditioning": "none",
        "continuation_policy": "none",
        "td_bootstrap": False,
    }
    if method == METHOD:
        locked.update(
            {
                "history_padding": "left_zero",
                "history_masking": "explicit_validity",
                "history_supervision": "all_prefix_lengths",
            }
        )
    if method in SEQUENCE_METHODS:
        if method == FROZEN_RESIDUAL_PREFIX_METHOD:
            locked["latent_recovery"] = "base_plus_residual_manifold_latents"
        else:
            locked["latent_recovery"] = (
                "direct_manifold_latents"
                if method in MANIFOLD_PREFIX_METHODS
                else "exact_adjacent_successor_difference"
            )
    if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        locked["goal_encoder"] = "frozen_pretrained_teacher"
    if method in {
        BALANCED_SEQUENCE_METHOD,
        EMA_BALANCED_SEQUENCE_METHOD,
        DIRECT_MOMENT_METHOD,
        E2E_MOMENT_METHOD,
    }:
        locked["feature_group_reduction"] = "group_sum"
    for key, value in locked.items():
        if successor.get(key) != value:
            raise ValueError(f"successor.{key} must be {value!r}.")
    if int(successor.get("max_horizon", 0)) != horizon:
        raise ValueError("successor.max_horizon must equal the rollout horizon.")
    if method in MANIFOLD_PREFIX_METHODS:
        architecture_dimensions = (
            "prefix_depth",
            "prefix_heads",
            "prefix_mlp_dim",
            "predictor_depth",
            "predictor_mlp_dim",
            "fusion_dim",
        )
        if min(int(successor.get(key, 0)) for key in architecture_dimensions) <= 0:
            raise ValueError("Manifold-prefix architecture dimensions must be positive.")
        if int(protocol["model"]["embed_dim"]) % int(successor["prefix_heads"]):
            raise ValueError("model.embed_dim must be divisible by prefix_heads.")
        if not 0.0 <= float(successor.get("dropout", -1.0)) < 1.0:
            raise ValueError("successor.dropout must lie in [0, 1).")
        if method in PRETRAINED_METHODS and successor.get(
            "pretrained_world_model_sha256"
        ) != protocol["pretrained_world_model"]["checkpoint_sha256"]:
            raise ValueError("The pretrained LeWM source hashes must match.")
    elif int(successor.get("hidden_dim", 0)) <= 0:
        raise ValueError("successor.hidden_dim must be positive.")
    gamma = float(successor.get("gamma", -1.0))
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("successor.gamma must lie in [0, 1].")
    if method in SEQUENCE_METHODS and gamma == 0.0:
        raise ValueError("The S-only method requires gamma > 0 for latent recovery.")
    if method == METHOD:
        if not 0.0 <= float(successor.get("target_world_ema_decay", -1.0)) < 1.0:
            raise ValueError("target_world_ema_decay must lie in [0, 1).")
        if not 0.0 <= float(successor.get("loss_warmup_fraction", -1.0)) < 1.0:
            raise ValueError("loss_warmup_fraction must lie in [0, 1).")
    if method in {
        EMA_BALANCED_SEQUENCE_METHOD,
        EMA_MANIFOLD_PREFIX_METHOD,
    } and not 0.0 <= float(
        successor.get("target_world_ema_decay", -1.0)
    ) < 1.0:
        raise ValueError("target_world_ema_decay must lie in [0, 1).")
    geometry_anchor = protocol.get("loss", {}).get("geometry_anchor")
    if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        if not isinstance(geometry_anchor, dict):
            raise ValueError("Anchored end-to-end training requires a geometry anchor.")
        if (
            geometry_anchor.get("target") != "frozen_pretrained_teacher"
            or geometry_anchor.get("metric") != "coordinate_mse"
            or float(geometry_anchor.get("weight", 0.0)) <= 0.0
        ):
            raise ValueError("The frozen-teacher geometry anchor is invalid.")
    elif geometry_anchor is not None:
        raise ValueError("Only anchored end-to-end training accepts a geometry anchor.")
    planning_weight = float(successor.get("planning_weight", -1.0))
    terminal_weight = float(successor.get("terminal_weight", -1.0))
    if min(planning_weight, terminal_weight) < 0.0:
        raise ValueError("Successor planning weights cannot be negative.")
    if planning_weight + terminal_weight <= 0.0:
        raise ValueError("At least one planning cost must be active.")
    if method in SEQUENCE_METHODS and (
        planning_weight != 1.0 or terminal_weight != 0.0
    ):
        raise ValueError("The S-only primary planner must use only the successor score.")

    split = protocol.get("split", {})
    if split.get("unit") != "sequence_clip":
        raise ValueError("RF-Successor-LeWM splits sequence clips.")
    if not 0.0 < float(split.get("train_fraction", 0.0)) < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not math.isclose(
        float(split["train_fraction"]) + float(split.get("validation_fraction", 0.0)),
        1.0,
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    loader = protocol.get("loader", {})
    if int(loader.get("batch_size", 0)) <= 0 or int(loader.get("workers", -1)) < 0:
        raise ValueError("Loader batch size and workers are invalid.")
    if int(loader.get("prefetch_factor", 0)) <= 0:
        raise ValueError("loader.prefetch_factor must be positive.")
    if not isinstance(loader.get("device_image_preprocessing"), bool):
        raise TypeError("loader.device_image_preprocessing must be boolean.")
    if not isinstance(loader.get("episode_streaming"), bool):
        raise TypeError("loader.episode_streaming must be boolean.")
    if not 1 <= int(loader.get("minimum_unique_episodes_per_batch", 0)) <= int(
        loader["batch_size"]
    ):
        raise ValueError("minimum_unique_episodes_per_batch is invalid.")
    local_windows = num_steps - history
    effective_batch = int(loader["batch_size"]) * local_windows
    configured_batch = int(protocol.get("loss", {}).get("sigreg", {}).get(
        "effective_batch_size", 0
    ))
    if effective_batch != configured_batch:
        raise ValueError("The overlapping-window SIGReg effective batch size changed.")

    if training.get("epochs") != training.get("scheduler_epochs"):
        raise ValueError("Scheduler and trainer epochs must match.")
    if min(
        int(training.get("epochs", 0)),
        int(training.get("optimizer_steps_per_epoch", 0)),
    ) <= 0:
        raise ValueError("The formal optimizer budget must be positive.")
    if training.get("model_compile") is not False:
        raise ValueError("The first RF-Successor-LeWM protocol keeps compilation off.")
    if freeze_after_epoch is not None:
        stop_after_epoch = training.get("stop_after_epoch")
        if int(freeze_after_epoch) < 1 or int(stop_after_epoch or 0) != int(
            freeze_after_epoch
        ) + 1:
            raise ValueError(
                "Head-only refinement must run exactly one epoch after freezing."
            )
        if int(stop_after_epoch) > int(training["epochs"]):
            raise ValueError("Head-only refinement exceeds the formal epoch budget.")
    elif freeze_from_start:
        if int(training.get("stop_after_epoch", 0)) != 1:
            raise ValueError("The first frozen-head screen must run exactly one epoch.")
        if float(protocol["loss"]["sigreg"]["weight"]) != 0.0:
            raise ValueError("A frozen encoder must not carry an inactive SIGReg loss.")
    elif training.get("stop_after_epoch") is not None:
        raise ValueError("stop_after_epoch requires an explicit freezing stage.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("The scheduler must step per optimizer update.")
    if not protocol.get("seeds"):
        raise ValueError("At least one training seed is required.")

    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("The Lance conversion must use stable-worldmodel 0.1.1.")
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("The audited RF-Successor-LeWM input is JPEG quality 100.")
    if lance.get("source", {}).get("sha256") != dataset.get(
        "optimized_layout", {}
    ).get("sha256"):
        raise ValueError("The Lance source differs from the audited Cube layout.")


@dataclass(frozen=True)
class MultiHorizonWindows:
    """Batched windows starting at every valid point in a training clip."""

    history: torch.Tensor
    rollout_actions: torch.Tensor
    action_prefix: torch.Tensor
    target_future: torch.Tensor
    count_per_clip: int


@dataclass(frozen=True)
class HistoryContextBatch:
    """The same decision point conditioned on every valid history suffix."""

    padded_history: torch.Tensor
    history_mask: torch.Tensor
    action_prefix: torch.Tensor
    target_future: torch.Tensor
    contexts_per_window: int


def build_multi_horizon_windows(
    online_latents: torch.Tensor,
    target_latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_size: int,
    horizon: int,
) -> MultiHorizonWindows:
    """Align LeWM histories, candidate actions, and direct future targets."""

    if online_latents.ndim != 3 or target_latents.shape != online_latents.shape:
        raise ValueError("Online and target latents must have shape (batch, time, dim).")
    if actions.ndim != 3 or actions.shape[:2] != online_latents.shape[:2]:
        raise ValueError("Actions must share the latent batch and time axes.")
    count = online_latents.shape[1] - history_size - horizon + 1
    if count <= 0:
        raise ValueError("The clip contains no complete multi-horizon window.")
    histories = torch.cat(
        [online_latents[:, start : start + history_size] for start in range(count)],
        dim=0,
    )
    rollout_actions = torch.cat(
        [
            actions[:, start : start + history_size + horizon - 1]
            for start in range(count)
        ],
        dim=0,
    )
    action_prefix = torch.cat(
        [
            actions[
                :,
                start + history_size - 1 : start + history_size + horizon - 1,
            ]
            for start in range(count)
        ],
        dim=0,
    )
    target_future = torch.cat(
        [
            target_latents[
                :, start + history_size : start + history_size + horizon
            ]
            for start in range(count)
        ],
        dim=0,
    )
    return MultiHorizonWindows(
        history=histories,
        rollout_actions=rollout_actions,
        action_prefix=action_prefix,
        target_future=target_future,
        count_per_clip=count,
    )


def build_history_context_batch(
    windows: MultiHorizonWindows,
) -> HistoryContextBatch:
    """Build H=1..Hmax contexts ending at each window's current latent."""

    if windows.history.ndim != 3 or windows.history.shape[-2] <= 0:
        raise ValueError("windows.history must have shape (batch, time, dim).")
    history_size = int(windows.history.shape[-2])

    padded_histories = []
    history_masks = []
    for length in range(1, history_size + 1):
        actual = windows.history[..., -length:, :]
        padded, mask = left_pad_latent_history(
            actual,
            history_size=history_size,
        )
        padded_histories.append(padded)
        history_masks.append(mask)

    return HistoryContextBatch(
        padded_history=torch.cat(padded_histories, dim=0),
        history_mask=torch.cat(history_masks, dim=0),
        action_prefix=torch.cat([windows.action_prefix] * history_size, dim=0),
        target_future=torch.cat([windows.target_future] * history_size, dim=0),
        contexts_per_window=history_size,
    )


@torch.no_grad()
def ema_update_world_model(
    target: torch.nn.Module,
    source: torch.nn.Module,
    *,
    decay: float,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(
        target.buffers(), source.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            target_buffer.copy_(source_buffer)


def _encode_online_and_target(
    online_model: Any,
    target_model: Any,
    encoder_input: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode independent mappings so the target pass cannot overwrite online data."""

    online = online_model.encode(dict(encoder_input))
    online_embeddings = online["emb"]
    online_action_embeddings = online["act_emb"]
    with torch.no_grad():
        target_embeddings = target_model.encode(dict(encoder_input))["emb"]
    return online_embeddings, online_action_embeddings, target_embeddings


def _build_joint_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    action_block_dim: int,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class RFSuccessorLeWMTrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            self.target_model = copy.deepcopy(world_model).requires_grad_(False)
            self.target_model.eval()
            self.device_image_preprocessing = device_image_preprocessing
            if device_image_preprocessing:
                image = protocol["image_preprocessing"]
                self.register_buffer(
                    "image_mean",
                    torch.tensor(image["mean"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "image_std",
                    torch.tensor(image["std"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
            sigreg = protocol["loss"]["sigreg"]
            self.sigreg = swm.wm.SIGReg(
                knots=sigreg["knots"], num_proj=sigreg["num_projections"]
            )
            successor = protocol["successor"]
            self.successor = ActionPrefixSuccessorHead(
                embed_dim=int(protocol["model"]["embed_dim"]),
                action_dim=action_block_dim,
                history_size=int(protocol["sequence"]["history_frames"]),
                hidden_dim=int(successor["hidden_dim"]),
                masked_history=True,
            )
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.horizon = int(protocol["sequence"]["rollout_horizon"])
            self.gamma = float(successor["gamma"])
            self.target_world_ema_decay = float(
                successor["target_world_ema_decay"]
            )
            self.auxiliary_warmup_steps = int(
                float(successor["loss_warmup_fraction"]) * total_steps
            )

        def train(self, mode: bool = True):
            super().train(mode)
            self.target_model.eval()
            return self

        def _auxiliary_scale(self) -> float:
            if self.auxiliary_warmup_steps <= 0:
                return 1.0
            return min(
                1.0,
                float(self.global_step + 1) / float(self.auxiliary_warmup_steps),
            )

        def _preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
            if not self.device_image_preprocessing:
                return pixels
            return preprocess_image_batch(
                pixels,
                mean=self.image_mean,
                std=self.image_std,
                size=protocol["image_preprocessing"]["size"],
            )

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.pop("_tdwm_episode_id", None)
            cache_bytes = batch.pop("_tdwm_cache_bytes", None)
            pixels = self._preprocess(batch["pixels"])
            actions = torch.nan_to_num(batch["action"], 0.0)
            online_input = {**batch, "pixels": pixels, "action": actions}
            embeddings, online_action_embeddings, target_embeddings = (
                _encode_online_and_target(
                    self.model,
                    self.target_model,
                    online_input,
                )
            )
            expected_steps = int(protocol["sequence"]["num_steps"])
            if embeddings.shape[1] != expected_steps:
                raise RuntimeError("The encoded clip has an unexpected length.")

            local_count = embeddings.shape[1] - self.history_size
            local_histories = torch.cat(
                [
                    embeddings[:, start : start + self.history_size]
                    for start in range(local_count)
                ],
                dim=0,
            )
            local_actions = torch.cat(
                [
                    online_action_embeddings[
                        :, start : start + self.history_size
                    ]
                    for start in range(local_count)
                ],
                dim=0,
            )
            local_targets = torch.cat(
                [
                    embeddings[:, start + 1 : start + self.history_size + 1]
                    for start in range(local_count)
                ],
                dim=0,
            )
            local_prediction = self.model.predict(local_histories, local_actions)
            local_prediction_loss = (
                local_prediction - local_targets
            ).square().mean()
            local_sequences = torch.cat(
                [
                    embeddings[:, start : start + self.history_size + 1]
                    for start in range(local_count)
                ],
                dim=0,
            )
            sigreg_loss = self.sigreg(local_sequences.transpose(0, 1))

            windows = build_multi_horizon_windows(
                embeddings,
                target_embeddings,
                actions,
                history_size=self.history_size,
                horizon=self.horizon,
            )
            contexts = build_history_context_batch(windows)
            pred_proj_was_training = self.model.pred_proj.training
            self.model.pred_proj.eval()
            try:
                rollout = rollout_from_latents(
                    self.model,
                    windows.history,
                    windows.rollout_actions,
                    history_size=self.history_size,
                )
            finally:
                self.model.pred_proj.train(pred_proj_was_training)
            if rollout.shape[-2] < self.history_size + self.horizon:
                raise RuntimeError("LeWM rollout did not cover the training horizon.")
            predicted_future = rollout[
                ..., self.history_size : self.history_size + self.horizon, :
            ]

            successor_prediction = self.successor(
                contexts.padded_history,
                contexts.action_prefix,
                history_mask=contexts.history_mask,
            )
            successor_target = finite_horizon_successor_targets(
                contexts.target_future.detach(), gamma=self.gamma
            )
            successor_loss = balanced_successor_mse(
                successor_prediction, successor_target
            )
            full_history_count = int(windows.history.shape[0])
            recurrence = successor_recurrence_residual(
                successor_prediction[-full_history_count:],
                predicted_future,
                gamma=self.gamma,
            )
            recurrence_loss = balanced_successor_mse(
                recurrence, torch.zeros_like(recurrence)
            )
            detached_future = windows.target_future.detach()
            latent_error = predicted_future - detached_future
            successor_error = successor_prediction - successor_target
            latent_loss = latent_error.square().mean()
            latent_mse_by_horizon = latent_error.square().mean(dim=(0, 2))
            successor_mse_by_horizon = successor_error.square().mean(dim=(0, 2))
            recurrence_mse_by_horizon = recurrence.square().mean(dim=(0, 2))

            auxiliary_scale = self._auxiliary_scale()
            objective = protocol["joint_objective"]
            auxiliary_loss = (
                float(objective["multi_step_prediction_weight"]) * latent_loss
                + float(objective["successor_weight"]) * successor_loss
                + float(objective["recurrence_weight"]) * recurrence_loss
            )
            loss = (
                local_prediction_loss
                + float(protocol["loss"]["sigreg"]["weight"]) * sigreg_loss
                + auxiliary_scale * auxiliary_loss
            )
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/local_prediction_loss": local_prediction_loss.detach(),
                f"{stage}/sigreg_loss": sigreg_loss.detach(),
                f"{stage}/multi_step_latent_loss": latent_loss.detach(),
                f"{stage}/successor_loss": successor_loss.detach(),
                f"{stage}/recurrence_loss": recurrence_loss.detach(),
                f"{stage}/latent_mse_h1": latent_mse_by_horizon[0].detach(),
                f"{stage}/latent_mse_hK": latent_mse_by_horizon[-1].detach(),
                f"{stage}/successor_mse_h1": (
                    successor_mse_by_horizon[0].detach()
                ),
                f"{stage}/successor_mse_hK": (
                    successor_mse_by_horizon[-1].detach()
                ),
                f"{stage}/recurrence_mse_h1": (
                    recurrence_mse_by_horizon[0].detach()
                ),
                f"{stage}/recurrence_mse_hK": (
                    recurrence_mse_by_horizon[-1].detach()
                ),
                f"{stage}/auxiliary_weight_scale": loss.new_tensor(auxiliary_scale),
                f"{stage}/multi_horizon_windows": loss.new_tensor(
                    float(windows.count_per_clip * batch_size)
                ),
                f"{stage}/multi_history_contexts": loss.new_tensor(
                    float(
                        windows.count_per_clip
                        * batch_size
                        * contexts.contexts_per_window
                    )
                ),
            }
            if episode_ids is not None:
                metrics[f"{stage}/unique_episodes_per_batch"] = loss.new_tensor(
                    float(torch.unique(episode_ids).numel())
                )
            if cache_bytes is not None:
                metrics[f"{stage}/compressed_cache_gib"] = loss.new_tensor(
                    float(cache_bytes) / 1024**3
                )
            self.log_dict(
                metrics,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage == "validation",
                sync_dist=False,
                batch_size=batch_size,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "validation")

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            del outputs, batch, batch_idx
            ema_update_world_model(
                self.target_model,
                self.model,
                decay=self.target_world_ema_decay,
            )

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(self.model.parameters()),
                        "lr": optimizer_cfg["world_model_learning_rate"],
                    },
                    {
                        "params": list(self.successor.parameters()),
                        "lr": optimizer_cfg["successor_learning_rate"],
                    },
                ],
                weight_decay=optimizer_cfg["weight_decay"],
            )
            warmup_steps = max(
                1, int(float(protocol["scheduler"]["warmup_fraction"]) * total_steps)
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return RFSuccessorLeWMTrainingModule()


def _build_successor_sequence_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    action_block_dim: int,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class RFSuccessorSequenceTrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            for name in ("predictor", "action_encoder", "pred_proj"):
                module = getattr(self.model, name, None)
                if module is not None:
                    module.requires_grad_(False)
            self.use_ema_target = protocol["method"] in {
                EMA_BALANCED_SEQUENCE_METHOD,
                EMA_MANIFOLD_PREFIX_METHOD,
            }
            self.use_frozen_teacher = (
                protocol["method"] == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD
            )
            if self.use_ema_target or self.use_frozen_teacher:
                self.target_model = copy.deepcopy(self.model).requires_grad_(False)
                self.target_model.eval()
            if self.use_ema_target:
                self.target_world_ema_decay = float(
                    protocol["successor"]["target_world_ema_decay"]
                )
            self.device_image_preprocessing = device_image_preprocessing
            if device_image_preprocessing:
                image = protocol["image_preprocessing"]
                self.register_buffer(
                    "image_mean",
                    torch.tensor(image["mean"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "image_std",
                    torch.tensor(image["std"], dtype=torch.float32).reshape(
                        1, 1, 3, 1, 1
                    ),
                    persistent=False,
                )
            sigreg = protocol["loss"]["sigreg"]
            self.sigreg_weight = float(sigreg["weight"])
            self.sigreg = (
                swm.wm.SIGReg(
                    knots=sigreg["knots"], num_proj=sigreg["num_projections"]
                )
                if self.sigreg_weight > 0.0
                else None
            )
            successor = protocol["successor"]
            head_dimensions = {
                "embed_dim": int(protocol["model"]["embed_dim"]),
                "action_dim": action_block_dim,
                "history_size": int(protocol["sequence"]["history_frames"]),
                "gamma": float(successor["gamma"]),
            }
            if protocol["method"] == FROZEN_RESIDUAL_PREFIX_METHOD:
                self.successor = LeWMResidualTransformerHead(
                    **head_dimensions,
                    prefix_depth=int(successor["prefix_depth"]),
                    prefix_heads=int(successor["prefix_heads"]),
                    prefix_mlp_dim=int(successor["prefix_mlp_dim"]),
                    predictor_depth=int(successor["predictor_depth"]),
                    predictor_mlp_dim=int(successor["predictor_mlp_dim"]),
                    fusion_dim=int(successor["fusion_dim"]),
                    dropout=float(successor["dropout"]),
                )
            elif protocol["method"] in MANIFOLD_PREFIX_METHODS:
                self.successor = ManifoldTransformerMomentHead(
                    **head_dimensions,
                    prefix_depth=int(successor["prefix_depth"]),
                    prefix_heads=int(successor["prefix_heads"]),
                    prefix_mlp_dim=int(successor["prefix_mlp_dim"]),
                    predictor_depth=int(successor["predictor_depth"]),
                    predictor_mlp_dim=int(successor["predictor_mlp_dim"]),
                    fusion_dim=int(successor["fusion_dim"]),
                    dropout=float(successor["dropout"]),
                )
            else:
                self.successor = ActionPrefixMomentHead(
                    **head_dimensions,
                    hidden_dim=int(successor["hidden_dim"]),
                )
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.horizon = int(protocol["sequence"]["rollout_horizon"])
            self.gamma = float(successor["gamma"])
            geometry_anchor = protocol["loss"].get("geometry_anchor")
            self.geometry_anchor_weight = (
                float(geometry_anchor["weight"])
                if geometry_anchor is not None
                else 0.0
            )
            self.freeze_world_model_after_epoch = protocol["training"].get(
                "freeze_world_model_after_epoch"
            )
            self.world_model_frozen = False
            if protocol["training"].get("freeze_world_model_from_start", False):
                self._freeze_world_model()

        def train(self, mode: bool = True):
            super().train(mode)
            if self.world_model_frozen or self.use_frozen_teacher:
                self.model.eval()
            target_model = getattr(self, "target_model", None)
            if target_model is not None:
                target_model.eval()
            return self

        def _freeze_world_model(self) -> None:
            self.model.requires_grad_(False)
            self.model.eval()
            self.world_model_frozen = True

        def on_train_epoch_start(self) -> None:
            freeze_epoch = self.freeze_world_model_after_epoch
            if freeze_epoch is not None and self.current_epoch >= int(freeze_epoch):
                self._freeze_world_model()

        def _preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
            if not self.device_image_preprocessing:
                return pixels
            return preprocess_image_batch(
                pixels,
                mean=self.image_mean,
                std=self.image_std,
                size=protocol["image_preprocessing"]["size"],
            )

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            if self.use_frozen_teacher:
                # Gradients remain enabled in eval mode; this only keeps the
                # pretrained BatchNorm coordinate system deterministic.
                self.model.eval()
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.pop("_tdwm_episode_id", None)
            cache_bytes = batch.pop("_tdwm_cache_bytes", None)
            pixels = self._preprocess(batch["pixels"])
            actions = torch.nan_to_num(batch["action"], 0.0)
            encoder_input = {
                key: value for key, value in batch.items() if key != "action"
            }
            encoder_input["pixels"] = pixels
            if self.world_model_frozen:
                with torch.no_grad():
                    embeddings = self.model.encode(encoder_input)["emb"]
            else:
                embeddings = self.model.encode(encoder_input)["emb"]
            expected_steps = int(protocol["sequence"]["num_steps"])
            if embeddings.shape[1] != expected_steps:
                raise RuntimeError("The encoded clip has an unexpected length.")

            target_embeddings = embeddings
            target_model = getattr(self, "target_model", None)
            if target_model is not None:
                with torch.no_grad():
                    target_embeddings = target_model.encode(encoder_input)["emb"]

            windows = build_multi_horizon_windows(
                embeddings,
                target_embeddings,
                actions,
                history_size=self.history_size,
                horizon=self.horizon,
            )
            vector_reduction = protocol["successor"].get(
                "feature_group_reduction", "coordinate_mean"
            )
            if protocol["method"] == FROZEN_RESIDUAL_PREFIX_METHOD:
                with torch.no_grad():
                    rollout = rollout_from_latents(
                        self.model,
                        windows.history,
                        windows.rollout_actions,
                        history_size=self.history_size,
                    )
                    if rollout.shape[-2] < self.history_size + self.horizon:
                        raise RuntimeError(
                            "Frozen LeWM rollout did not cover the training horizon."
                        )
                    base_future = rollout[
                        ...,
                        self.history_size : self.history_size + self.horizon,
                        :,
                    ]
                output = residual_manifold_sequence_objective(
                    self.successor,
                    windows.history,
                    windows.action_prefix,
                    base_future,
                    windows.target_future,
                    gamma=self.gamma,
                )
                predictive_loss = output.latent_loss
                predictive_metric = "latent_sequence_loss"
            elif protocol["method"] in MANIFOLD_PREFIX_METHODS:
                output = manifold_sequence_objective(
                    self.successor,
                    windows.history,
                    windows.action_prefix,
                    windows.target_future,
                    gamma=self.gamma,
                    detach_target=(
                        protocol["method"]
                        in {
                            EMA_MANIFOLD_PREFIX_METHOD,
                            FROZEN_MANIFOLD_PREFIX_METHOD,
                            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD,
                        }
                    ),
                )
                predictive_loss = output.latent_loss
                predictive_metric = "latent_sequence_loss"
            elif protocol["method"] in DIRECT_MOMENT_METHODS:
                output = moment_sequence_objective(
                    self.successor,
                    windows.history,
                    windows.action_prefix,
                    windows.target_future,
                    gamma=self.gamma,
                    vector_reduction=vector_reduction,
                    detach_target=protocol["method"] == DIRECT_MOMENT_METHOD,
                )
                predictive_loss = output.moment_loss
                predictive_metric = "moment_sequence_loss"
            else:
                output = successor_sequence_objective(
                    self.successor,
                    windows.history,
                    windows.action_prefix,
                    windows.target_future,
                    gamma=self.gamma,
                    vector_reduction=vector_reduction,
                )
                predictive_loss = output.successor_loss
                predictive_metric = "successor_sequence_loss"

            if self.sigreg is None:
                sigreg_loss = predictive_loss.new_zeros(())
            else:
                local_count = embeddings.shape[1] - self.history_size
                sigreg_sequences = torch.cat(
                    [
                        embeddings[:, start : start + self.history_size + 1]
                        for start in range(local_count)
                    ],
                    dim=0,
                )
                sigreg_loss = self.sigreg(sigreg_sequences.transpose(0, 1))
            if self.use_frozen_teacher:
                geometry_anchor_loss = (
                    embeddings - target_embeddings.detach()
                ).square().mean()
            else:
                geometry_anchor_loss = predictive_loss.new_zeros(())
            loss = (
                predictive_loss
                + self.sigreg_weight * sigreg_loss
                + self.geometry_anchor_weight * geometry_anchor_loss
            )
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/{predictive_metric}": predictive_loss.detach(),
                f"{stage}/sigreg_loss": sigreg_loss.detach(),
                f"{stage}/geometry_anchor_loss": geometry_anchor_loss.detach(),
                f"{stage}/geometry_anchor_rms": (
                    geometry_anchor_loss.sqrt().detach()
                ),
                f"{stage}/successor_mse_h1": (
                    output.successor_mse_by_horizon[0].detach()
                ),
                f"{stage}/successor_mse_hK": (
                    output.successor_mse_by_horizon[-1].detach()
                ),
                f"{stage}/recovered_latent_mse_h1": (
                    output.recovered_latent_mse_by_horizon[0].detach()
                ),
                f"{stage}/recovered_latent_mse_hK": (
                    output.recovered_latent_mse_by_horizon[-1].detach()
                ),
                f"{stage}/multi_horizon_windows": loss.new_tensor(
                    float(windows.count_per_clip * batch_size)
                ),
            }
            if protocol["method"] in DIRECT_MOMENT_METHODS:
                metrics[f"{stage}/moment_mse_h1"] = (
                    output.moment_mse_by_horizon[0].detach()
                )
                metrics[f"{stage}/moment_mse_hK"] = (
                    output.moment_mse_by_horizon[-1].detach()
                )
            elif protocol["method"] in MANIFOLD_PREFIX_METHODS:
                metrics[f"{stage}/latent_mse_h1"] = (
                    output.latent_mse_by_horizon[0].detach()
                )
                metrics[f"{stage}/latent_mse_hK"] = (
                    output.latent_mse_by_horizon[-1].detach()
                )
                if protocol["method"] == FROZEN_RESIDUAL_PREFIX_METHOD:
                    metrics[f"{stage}/base_latent_mse_h1"] = (
                        output.base_latent_mse_by_horizon[0].detach()
                    )
                    metrics[f"{stage}/base_latent_mse_hK"] = (
                        output.base_latent_mse_by_horizon[-1].detach()
                    )
                    metrics[f"{stage}/correction_rms"] = (
                        output.correction.square().mean().sqrt().detach()
                    )
            if episode_ids is not None:
                metrics[f"{stage}/unique_episodes_per_batch"] = loss.new_tensor(
                    float(torch.unique(episode_ids).numel())
                )
            if cache_bytes is not None:
                metrics[f"{stage}/compressed_cache_gib"] = loss.new_tensor(
                    float(cache_bytes) / 1024**3
                )
            self.log_dict(
                metrics,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage == "validation",
                sync_dist=False,
                batch_size=batch_size,
            )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "validation")

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            del outputs, batch, batch_idx
            target_model = getattr(self, "target_model", None)
            if target_model is not None and self.use_ema_target:
                ema_update_world_model(
                    target_model,
                    self.model,
                    decay=self.target_world_ema_decay,
                )

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            model_parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            parameter_groups = []
            if model_parameters:
                parameter_groups.append(
                    {
                        "params": model_parameters,
                        "lr": optimizer_cfg["world_model_learning_rate"],
                    }
                )
            parameter_groups.append(
                {
                    "params": list(self.successor.parameters()),
                    "lr": optimizer_cfg["successor_learning_rate"],
                }
            )
            optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=optimizer_cfg["weight_decay"],
            )
            warmup_steps = max(
                1, int(float(protocol["scheduler"]["warmup_fraction"]) * total_steps)
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return RFSuccessorSequenceTrainingModule()


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    action_block_dim: int,
    device_image_preprocessing: bool,
):
    builder = (
        _build_successor_sequence_training_module
        if protocol["method"] in SEQUENCE_METHODS
        else _build_joint_training_module
    )
    return builder(
        world_model,
        protocol,
        total_steps,
        action_block_dim=action_block_dim,
        device_image_preprocessing=device_image_preprocessing,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local_pretrained_lewm_export(
    checkpoint_path: str | Path,
) -> tuple[str, Path, Path]:
    requested = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir = requested if requested.is_dir() else requested.parent
    weights = sorted(checkpoint_dir.glob("*.pt"))
    if len(weights) != 1:
        raise FileNotFoundError(
            "A pretrained LeWM export must contain exactly one .pt file."
        )
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "A pretrained LeWM export must use the public "
            "<cache_dir>/checkpoints/<run_name> layout."
        )
    return checkpoint_dir.name, weights[0], checkpoint_dir.parent.parent


def _successor_config(
    protocol: dict[str, Any],
    *,
    action_block_dim: int,
    base_export_run_name: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    successor = protocol["successor"]
    config = {
        "objective_version": successor["objective_version"],
        "architecture": successor["architecture"],
        "embed_dim": protocol["model"]["embed_dim"],
        "action_dim": action_block_dim,
        "history_size": protocol["sequence"]["history_frames"],
        "max_horizon": successor["max_horizon"],
        "gamma": successor["gamma"],
        "feature_basis": successor["feature_basis"],
        "horizon_normalization": successor["horizon_normalization"],
        "target": successor["target"],
        "action_conditioning": successor["action_conditioning"],
        "goal_conditioning": successor["goal_conditioning"],
        "continuation_policy": successor["continuation_policy"],
        "td_bootstrap": successor["td_bootstrap"],
        "planning_weight": successor["planning_weight"],
        "terminal_weight": successor["terminal_weight"],
        "clamp_successor_cost": successor["clamp_successor_cost"],
        "base_export_run_name": base_export_run_name,
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }
    if protocol["method"] == METHOD:
        for key in (
            "history_padding",
            "history_masking",
            "history_supervision",
        ):
            config[key] = successor[key]
    if protocol["method"] in MANIFOLD_PREFIX_METHODS:
        for key in (
            "prefix_depth",
            "prefix_heads",
            "prefix_mlp_dim",
            "predictor_depth",
            "predictor_mlp_dim",
            "fusion_dim",
            "dropout",
        ):
            config[key] = successor[key]
    else:
        config["hidden_dim"] = successor["hidden_dim"]
    if "target_world_ema_decay" in successor:
        config["target_world_ema_decay"] = successor["target_world_ema_decay"]
    if "latent_recovery" in successor:
        config["latent_recovery"] = successor["latent_recovery"]
    if "feature_group_reduction" in successor:
        config["feature_group_reduction"] = successor[
            "feature_group_reduction"
        ]
    if protocol["method"] in PRETRAINED_METHODS:
        config["pretrained_world_model_sha256"] = protocol[
            "pretrained_world_model"
        ]["checkpoint_sha256"]
    if protocol["method"] == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        config["goal_encoder"] = successor["goal_encoder"]
        config["geometry_anchor_weight"] = protocol["loss"]["geometry_anchor"][
            "weight"
        ]
    return config


def _build_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
    action_block_dim: int,
    initialization_info: dict[str, Any] | None = None,
):
    import lightning as pl
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class RFSuccessorExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = trainer.current_epoch + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            base_run_name = f"epoch_{epoch:02d}"
            export_root = run_dir / "checkpoints" / "exports"
            swm.wm.save_pretrained(
                pl_module.model,
                run_name=base_run_name,
                config=export_config,
                cache_dir=str(export_root),
            )
            base_dir = export_root / "checkpoints" / base_run_name
            base_weights = sorted(base_dir.glob("*.pt"))
            if len(base_weights) != 1:
                raise RuntimeError(
                    "Stable World Model export did not contain exactly one weight file."
                )
            base_hash = _file_sha256(base_weights[0])
            method = protocol["method"]
            deployment_dir = run_dir / "checkpoints" / method
            deployment_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "method": method,
                "objective_version": protocol["successor"]["objective_version"],
                "deployment_checkpoint_version": 1,
                "epoch": epoch,
                "global_step": int(trainer.global_step),
                "world_model_state_dict": pl_module.model.state_dict(),
                "successor_state_dict": pl_module.successor.state_dict(),
                "world_model_config": model_config,
                "initialization": initialization_info,
                "successor_config": _successor_config(
                    protocol,
                    action_block_dim=action_block_dim,
                    base_export_run_name=base_run_name,
                    base_checkpoint_sha256=base_hash,
                ),
            }
            if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
                payload["target_world_model_state_dict"] = (
                    pl_module.target_model.state_dict()
                )
            torch.save(payload, deployment_dir / f"epoch_{epoch:02d}.pt")

    return RFSuccessorExportCallback()


def _build_generator_callback(generator: torch.Generator, *, method: str = METHOD):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return f"tdwm_{method}_dataloader_generator"

        def state_dict(self) -> dict[str, Any]:
            return {"generator_state": generator.get_state()}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            generator.set_state(state_dict["generator_state"])

    return DataLoaderGeneratorCallback()


def _build_episode_epoch_callback(dataset: EpisodeStreamingBatchDataset):
    import lightning as pl

    class EpisodeStreamingEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            del pl_module
            dataset.set_epoch(int(trainer.current_epoch))

    return EpisodeStreamingEpochCallback()


def train_rf_successor_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
    initial_world_model_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a reward-free successor method independently of the baselines."""

    protocol = load_rf_successor_training_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
    if (
        protocol.get("training", {}).get("freeze_world_model_after_epoch")
        is not None
        and resume != "required"
    ):
        raise ValueError("Head-only refinement requires an explicit checkpoint resume.")
    pretrained = protocol["method"] in PRETRAINED_METHODS
    if pretrained != (initial_world_model_checkpoint_path is not None):
        raise ValueError(
            "A pretrained method requires exactly one initial LeWM checkpoint."
        )
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / (f"seed_{seed}_smoke" if smoke else f"seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    compatibility = prepare_cloud_runtime() or {}
    import hydra
    import lightning as pl
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import ModelCheckpoint

    package_version = importlib.metadata.version("stable-worldmodel")
    expected_version = protocol["runtime"]["stable_worldmodel_version"]
    if package_version != expected_version:
        raise RuntimeError(
            f"Expected stable-worldmodel {expected_version}, found {package_version}."
        )
    pl.seed_everything(seed, workers=True)

    sequence = protocol["sequence"]
    dataset_cfg = protocol["dataset"]
    loader_cfg = protocol["loader"]
    device_preprocessing = bool(loader_cfg["device_image_preprocessing"])
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=sequence["num_steps"],
        frameskip=sequence["frame_skip"],
        keys_to_load=list(dataset_cfg["keys_to_load"]),
        keys_to_cache=list(dataset_cfg["keys_to_cache"]),
        keys_to_merge=dict(dataset_cfg["keys_to_merge"]),
    )
    if len(dataset.lengths) != dataset_cfg["expected_episodes"]:
        raise ValueError("Dataset episode count differs from the protocol.")
    if int(np.asarray(dataset.lengths).sum()) != dataset_cfg["expected_transitions"]:
        raise ValueError("Dataset transition count differs from the protocol.")

    statistics = fit_column_stats(
        dataset,
        list(protocol["normalization"]["columns"]),
        output_dir / "column_normalization.json",
    )
    dataset.transform = LeWMTransform(
        image=protocol["image_preprocessing"],
        columns=statistics,
        preprocess_images=not device_preprocessing,
    )
    if dataset_source["format"] == "lance":
        dataset = StrideAwareLanceDataset(dataset)

    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        [protocol["split"]["train_fraction"], protocol["split"]["validation_fraction"]],
        generator=generator,
    )
    split_manifest = save_split(
        run_dir,
        np.asarray(train_set.indices, dtype=np.int64),
        np.asarray(validation_set.indices, dtype=np.int64),
    )

    episode_train_dataset = None
    use_episode_streaming = bool(loader_cfg["episode_streaming"]) and not smoke
    if use_episode_streaming:
        if not isinstance(dataset, StrideAwareLanceDataset):
            raise ValueError("Episode streaming requires the audited Lance dataset.")
        episode_train_dataset = EpisodeStreamingBatchDataset(
            dataset,
            train_set.indices,
            batch_size=loader_cfg["batch_size"],
            active_episodes=loader_cfg["episode_pool_size"],
            read_episodes=loader_cfg["episode_read_size"],
            cache_bytes=loader_cfg["episode_cache_bytes"],
            prefetch_blocks=loader_cfg["episode_prefetch_blocks"],
            seed=seed,
            drop_last=loader_cfg["train_drop_last"],
            min_unique_episodes=loader_cfg["minimum_unique_episodes_per_batch"],
        )
        train_loader = torch.utils.data.DataLoader(
            episode_train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=loader_cfg["pin_memory"],
        )
    else:
        workers = 0 if smoke else int(loader_cfg["workers"])
        train_kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": loader_cfg["pin_memory"],
        }
        if workers:
            train_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": loader_cfg["prefetch_factor"],
                }
            )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=loader_cfg["batch_size"],
            shuffle=loader_cfg["train_shuffle"],
            drop_last=loader_cfg["train_drop_last"],
            generator=generator,
            **train_kwargs,
        )

    validation_workers = 0 if smoke else int(loader_cfg["validation_workers"])
    validation_kwargs: dict[str, Any] = {
        "num_workers": validation_workers,
        "pin_memory": loader_cfg["pin_memory"],
    }
    if validation_workers:
        validation_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": loader_cfg["prefetch_factor"],
            }
        )
    if loader_cfg["validation_locality"]:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_sampler=BlockShuffleBatchSampler(
                validation_set.indices,
                batch_size=loader_cfg["batch_size"],
                block_size=loader_cfg["block_size"],
                drop_last=loader_cfg["validation_drop_last"],
                shuffle_batches_within_block=False,
                shuffle_blocks=False,
            ),
            **validation_kwargs,
        )
    else:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=loader_cfg["batch_size"],
            shuffle=loader_cfg["validation_shuffle"],
            drop_last=loader_cfg["validation_drop_last"],
            **validation_kwargs,
        )

    action_dim = int(dataset.get_dim("action"))
    action_block_dim = int(sequence["frame_skip"]) * action_dim
    model_config = build_model_config(protocol, action_dim)
    initialization_info = None
    if pretrained:
        source_name, source_file, source_cache = (
            _resolve_local_pretrained_lewm_export(
                initial_world_model_checkpoint_path
            )
        )
        source_hash = _file_sha256(source_file)
        expected_hash = protocol["pretrained_world_model"]["checkpoint_sha256"]
        if source_hash != expected_hash:
            raise ValueError("The pretrained LeWM checkpoint hash differs from protocol.")
        world_model = swm.wm.load_pretrained(
            source_name,
            cache_dir=str(source_cache),
        )
        initialization_info = {
            "strategy": protocol["initialization"],
            "source_method": "lewm",
            "source_seed": protocol["pretrained_world_model"]["source_seed"],
            "source_epoch": protocol["pretrained_world_model"]["source_epoch"],
            "source_run_name": source_name,
            "source_checkpoint_path": str(source_file),
            "source_checkpoint_sha256": source_hash,
        }
        if protocol["method"] in FROZEN_PRETRAINED_METHODS:
            initialization_info["frozen"] = True
        else:
            initialization_info.update(
                {
                    "student_frozen": False,
                    "teacher_frozen": True,
                }
            )
    else:
        world_model = hydra.utils.instantiate(model_config)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} LeWM parameters, found {parameter_count}."
        )
    if protocol["training"]["model_compile"]:
        compile_world_model(
            world_model, mode=protocol["training"]["model_compile_mode"]
        )

    available_epoch_steps = len(train_loader)
    formal_epoch_steps = int(protocol["training"]["optimizer_steps_per_epoch"])
    if formal_epoch_steps > available_epoch_steps:
        raise ValueError("optimizer_steps_per_epoch exceeds available batches.")
    formal_steps = int(protocol["training"]["scheduler_epochs"]) * formal_epoch_steps
    train_limit = resolve_train_batch_limit(
        smoke=smoke,
        max_steps=max_steps,
        train_loader_length=available_epoch_steps,
    )
    if not smoke and max_steps is None:
        train_limit = formal_epoch_steps
    if smoke:
        # The second smoke invocation resumes into a second two-update epoch.
        total_steps = 2 * int(train_limit)
    else:
        total_steps = formal_steps
    if max_steps is not None:
        total_steps = int(train_limit)
    module = _build_training_module(
        world_model,
        protocol,
        total_steps,
        action_block_dim=action_block_dim,
        device_image_preprocessing=device_preprocessing,
    )

    checkpoint_dir = run_dir / "checkpoints" / "lightning"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=protocol["training"]["checkpoint_every_epochs"],
        save_last=True,
        save_top_k=-1,
    )
    callbacks = [
        checkpoint_callback,
        _build_export_callback(
            run_dir,
            model_config,
            protocol,
            action_block_dim,
            initialization_info,
        ),
        _build_generator_callback(generator, method=protocol["method"]),
    ]
    if episode_train_dataset is not None:
        callbacks.append(_build_episode_epoch_callback(episode_train_dataset))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = (
        protocol["training"]["precision"] if accelerator == "gpu" else "32-true"
    )
    if smoke:
        epochs = 2 if resume == "required" else 1
    elif max_steps is not None:
        epochs = 1
    elif protocol["training"].get("stop_after_epoch") is not None:
        epochs = int(protocol["training"]["stop_after_epoch"])
    else:
        epochs = protocol["training"]["epochs"]
    with patch(
        "lightning.pytorch.trainer.connectors.callback_connector."
        "_load_external_callbacks",
        return_value=[],
    ):
        trainer = pl.Trainer(
            default_root_dir=run_dir,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            max_epochs=epochs,
            gradient_clip_val=protocol["training"]["gradient_clip_norm"],
            limit_train_batches=train_limit,
            limit_val_batches=0.0 if smoke or skip_validation else 1.0,
            num_sanity_val_steps=0,
            logger=build_metrics_logger(run_dir, protocol["logging"]),
            callbacks=callbacks,
            log_every_n_steps=1 if smoke else 50,
        )

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {last_checkpoint}")
    checkpoint_path = None
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify the objective version for resume.")
        with manifest_path.open() as stream:
            previous = json.load(stream)
        previous_protocol = previous.get("protocol", {})
        if previous_protocol.get("method") != protocol["method"] or previous_protocol.get(
            "successor", {}
        ).get("objective_version") != protocol["successor"]["objective_version"]:
            raise RuntimeError("Refusing to resume an incompatible objective.")
        checkpoint_path = str(last_checkpoint)

    runtime = {
        "stable_worldmodel": package_version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdwm_git_revision": _git_revision(),
        "compatibility_adapter": compatibility,
    }
    if torch.cuda.is_available():
        runtime["cuda_device"] = torch.cuda.get_device_name(0)
    write_json(
        run_dir / "training_manifest.json",
        {
            "method": protocol["method"],
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "dataset": {
                **dataset_source,
                "sequence_samples": len(dataset),
                "split": split_manifest,
            },
            "model": {
                "config": model_config,
                "initialization": initialization_info,
                "lewm_parameters": parameter_count,
                "trainable_lewm_parameters": sum(
                    parameter.numel()
                    for parameter in module.model.parameters()
                    if parameter.requires_grad
                ),
                "successor_parameters": sum(
                    parameter.numel() for parameter in module.successor.parameters()
                ),
                "action_block_dim": action_block_dim,
            },
            "training": {
                "formal_optimizer_steps": formal_steps,
                "optimizer_steps_per_epoch": formal_epoch_steps,
                "available_batches_per_epoch": available_epoch_steps,
                "configured_optimizer_steps": total_steps,
                "resume_mode": resume,
                "resumed_from": checkpoint_path,
                "episode_streaming": use_episode_streaming,
                "validation_batches": len(validation_loader),
                "validation_skipped": smoke or skip_validation,
            },
            "runtime": runtime,
        },
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=checkpoint_path,
    )
    result = {
        "run_dir": str(run_dir),
        "seed": seed,
        "last_checkpoint": str(last_checkpoint),
        "final_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
    }
    if torch.cuda.is_available():
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "ANCHORED_E2E_MANIFOLD_PREFIX_METHOD",
    "METHOD",
    "BALANCED_SEQUENCE_METHOD",
    "DIRECT_MOMENT_METHOD",
    "DIRECT_MOMENT_METHODS",
    "E2E_MOMENT_METHOD",
    "EMA_BALANCED_SEQUENCE_METHOD",
    "EMA_MANIFOLD_PREFIX_METHOD",
    "FROZEN_MANIFOLD_PREFIX_METHOD",
    "FROZEN_PRETRAINED_METHODS",
    "FROZEN_RESIDUAL_PREFIX_METHOD",
    "MANIFOLD_PREFIX_METHOD",
    "MANIFOLD_PREFIX_METHODS",
    "PRETRAINED_METHODS",
    "SEQUENCE_METHODS",
    "S_ONLY_METHOD",
    "HistoryContextBatch",
    "MultiHorizonWindows",
    "build_history_context_batch",
    "build_multi_horizon_windows",
    "ema_update_world_model",
    "load_rf_successor_training_protocol",
    "train_rf_successor_lewm",
    "validate_rf_successor_training_protocol",
]
