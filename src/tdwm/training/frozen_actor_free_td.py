"""Shared frozen-latent training runtime for standalone C/D/F/G1/G2/G3 methods.

The legacy Actor-Free TD-LeWM trainer intentionally does not import this module.
Each new method supplies a fixed :class:`FrozenActorFreeTDSpec`; this file owns
only the common data, optimization, provenance, resume, and export machinery.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.actor_free_td_lewm import (
    ActorFreeSuccessorHead,
    actor_free_goal_future_offset_limits,
    actor_free_td_objective,
    ema_update,
    sample_actor_free_goal_offsets,
)
from tdwm.methods.direct_goal_critic_lewm import (
    DirectGoalCriticHead,
    direct_goal_critic_td_objective,
)
from tdwm.methods.frozen_td_common import (
    FrozenRealTDBatch,
    build_frozen_real_td_batch,
    gather_hindsight_goals,
    per_transition_vector_td_mse,
)
from tdwm.training.block_sampler import BlockShuffleBatchSampler
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.frozen_latent_store import (
    CUBE_ACTION_DIM,
    FrozenLatentClipDataset,
    FrozenLatentStore,
)
from tdwm.training.frozen_state_bank import load_split_indices
from tdwm.training.gt_lewm_support import (
    LeWMTransform,
    build_metrics_logger,
    build_model_config,
    compile_world_model,
    fit_column_stats,
    preprocess_image_batch,
    resolve_train_batch_limit,
    write_json,
)
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision
from tdwm.training.rf_successor_lewm import (
    DECODED_FRAME_STORE_ENV,
    _prepare_decoded_frame_store,
)
from tdwm.training.state_neighbor_index import StateNeighborActionIndex

METHOD_FAMILY = "actor_free_td_lewm"
OBJECTIVE_VERSION = 1
GOAL_OBJECTIVE_VERSION = 2
DIRECT_GOAL_OBJECTIVE_VERSION = 3
IMAGINARY_OBJECTIVE_VERSION = 3
DEPLOYMENT_CHECKPOINT_VERSION = 1
FORMAL_OPTIMIZER_UPDATES = 127_960
GOAL_VARIANT = "goal_hybrid"
DIRECT_GOAL_VARIANT = "direct_goal_hybrid"
IMAGINARY_VARIANT = "imaginary_hybrid"
GOAL_PROJECTED_VARIANT = "goal_projected_td"
GOAL_VALUE_WEIGHTED_VARIANT = "goal_value_weighted_td"
SAME_FUTURE_GOAL_ADVANTAGE_VARIANT = "same_future_goal_advantage"
NEIGHBOR_ACTION_ADVANTAGE_VARIANT = "neighbor_action_advantage"
PREFIX_MEAN_ACTION_ADVANTAGE_VARIANT = "prefix_mean_action_advantage"
PREFIX_MARGINAL_ACTION_ADVANTAGE_VARIANT = "prefix_marginal_action_advantage"
FROZEN_PRETRAINED_VARIANTS = frozenset(
    {
        GOAL_PROJECTED_VARIANT,
        GOAL_VALUE_WEIGHTED_VARIANT,
        SAME_FUTURE_GOAL_ADVANTAGE_VARIANT,
        NEIGHBOR_ACTION_ADVANTAGE_VARIANT,
        PREFIX_MEAN_ACTION_ADVANTAGE_VARIANT,
        PREFIX_MARGINAL_ACTION_ADVANTAGE_VARIANT,
    }
)
NEIGHBOR_INDEX_VARIANTS = frozenset({NEIGHBOR_ACTION_ADVANTAGE_VARIANT})
FROZEN_OBJECTIVE_VERSIONS = {
    GOAL_PROJECTED_VARIANT: 1,
    GOAL_VALUE_WEIGHTED_VARIANT: 1,
    SAME_FUTURE_GOAL_ADVANTAGE_VARIANT: 1,
    NEIGHBOR_ACTION_ADVANTAGE_VARIANT: 1,
    PREFIX_MEAN_ACTION_ADVANTAGE_VARIANT: 1,
    PREFIX_MARGINAL_ACTION_ADVANTAGE_VARIANT: 1,
}
HINDSIGHT_GOAL_VARIANTS = frozenset(
    {GOAL_VARIANT, DIRECT_GOAL_VARIANT, *FROZEN_PRETRAINED_VARIANTS}
)


@dataclass(frozen=True)
class FrozenTDContext:
    """Method-owned loss inputs produced by the shared frozen trainer."""

    protocol: dict[str, Any]
    successor: ActorFreeSuccessorHead
    td_batch: FrozenRealTDBatch
    goals: torch.Tensor
    per_transition_td: torch.Tensor
    base_td_loss: torch.Tensor
    batch: dict[str, Any]
    stage: str
    batch_size: int
    neighbor_index: StateNeighborActionIndex | None


@dataclass(frozen=True)
class FrozenTDLoss:
    """One standalone method's scalar loss and method-specific metrics."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FrozenActorFreeTDSpec:
    """Fixed identity and behavior injected by one standalone method module."""

    method: str
    variant: str
    objective_version: int
    requires_neighbor_index: bool
    validate_objective: Callable[[dict[str, Any]], None]
    compute_loss: Callable[[FrozenTDContext], FrozenTDLoss]
    successor_config_fields: Callable[[dict[str, Any]], dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.method.startswith(f"{METHOD_FAMILY}_"):
            raise ValueError("Standalone method names must preserve the method family.")
        if self.variant not in FROZEN_PRETRAINED_VARIANTS:
            raise ValueError(f"Unsupported frozen variant: {self.variant!r}.")
        if self.objective_version <= 0:
            raise ValueError("objective_version must be positive.")
        if self.requires_neighbor_index != (
            self.variant == NEIGHBOR_ACTION_ADVANTAGE_VARIANT
        ):
            raise ValueError("Only the standalone G1 method may require neighbors.")


def objective_version_for_variant(variant: str) -> int:
    """Return the locked semantic objective version for a TD variant."""

    if variant == GOAL_VARIANT:
        return GOAL_OBJECTIVE_VERSION
    if variant == DIRECT_GOAL_VARIANT:
        return DIRECT_GOAL_OBJECTIVE_VERSION
    if variant == IMAGINARY_VARIANT:
        return IMAGINARY_OBJECTIVE_VERSION
    if variant in FROZEN_OBJECTIVE_VERSIONS:
        return FROZEN_OBJECTIVE_VERSIONS[variant]
    return OBJECTIVE_VERSION


def load_frozen_actor_free_td_training_protocol(
    path: str | Path,
    *,
    spec: FrozenActorFreeTDSpec,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_frozen_actor_free_td_training_protocol(protocol, spec=spec)
    return protocol


def validate_frozen_actor_free_td_training_protocol(
    protocol: dict[str, Any],
    *,
    spec: FrozenActorFreeTDSpec,
) -> None:
    """Reject protocol drift for one fixed standalone frozen method."""

    if protocol.get("schema_version") != 1:
        raise ValueError("Frozen Actor-Free TD-LeWM requires schema version 1.")
    if protocol.get("method") != spec.method:
        raise ValueError(f"Protocol method must be {spec.method!r}.")
    if protocol.get("method_family") != METHOD_FAMILY:
        raise ValueError(f"Protocol method_family must be {METHOD_FAMILY!r}.")
    variant = protocol.get("variant")
    if variant != spec.variant:
        raise ValueError(f"Protocol variant must be {spec.variant!r}.")
    frozen_pretrained = True
    if (
        protocol.get("environment") != "cube"
        or protocol.get("stage") != "full_training"
        or protocol.get("initialization") != "frozen_pretrained_lewm"
    ):
        raise ValueError("Frozen Actor-Free TD-LeWM is locked to full Cube training.")
    pretrained = protocol.get("pretrained_world_model")
    if frozen_pretrained:
        if not isinstance(pretrained, dict):
            raise ValueError("Frozen successor variants require pretrained metadata.")
        locked_pretrained = {
            "source_method": "lewm",
            "source_seed": 3072,
            "source_epoch": 10,
            "frozen": True,
        }
        for key, expected in locked_pretrained.items():
            if pretrained.get(key) != expected:
                raise ValueError(f"pretrained_world_model.{key} must be {expected!r}.")
        source_hash = pretrained.get("checkpoint_sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError(
                "pretrained_world_model.checkpoint_sha256 must be lowercase SHA-256."
            )
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Actor-Free TD-LeWM requires stable-worldmodel 0.1.1.")
    if protocol.get("seeds") != [0, 42, 3072]:
        raise ValueError("All variants use the locked seeds [0, 42, 3072].")

    sequence = protocol.get("sequence", {})
    history = int(sequence.get("history_frames", 0))
    num_steps = int(sequence.get("num_steps", 0))
    if history <= 0 or num_steps <= history + 1:
        raise ValueError("The clip must leave TD pairs after a complete history.")
    if sequence.get("prediction_frames") != 1:
        raise ValueError("The retained LeWM objective predicts one frame.")
    if int(sequence.get("frame_skip", 0)) <= 0:
        raise ValueError("sequence.frame_skip must be positive.")

    objective = protocol.get("joint_objective", {})
    if frozen_pretrained:
        expected_objective: dict[str, Any] = {
            "target_encoder": "frozen_pretrained",
            "bootstrap_action": "dataset_next_action",
            "terminal_mask": "next_action_nan_invalid",
            "actor": "none",
            "reward": "none",
            "local_prediction_weight": 0.0,
        }
    else:
        expected_objective = {
            "local_prediction": "original_lewm_one_step_mse",
            "regularization": "original_lewm_sigreg",
            "target_encoder": "ema_world_model",
            "bootstrap_action": "dataset_next_action",
            "terminal_mask": "next_action_nan_invalid",
            "actor": "none",
            "reward": "none",
            "local_prediction_weight": 1.0,
        }
    if variant == DIRECT_GOAL_VARIANT:
        expected_objective.update(
            {
                "td_target": "ema_next_goal_cost_plus_ema_critic_dataset_next_action",
                "goal_conditioning": "direct_critic_input",
                "predicted_context_detach": False,
                "real_critic_td_weight": 1.0,
                "predicted_critic_td_weight": 1.0,
            }
        )
    elif frozen_pretrained:
        expected_weights = (1.0, 0.0, True, 0.0, 0.0)
        expected_objective.update(
            {
                "td_target": (
                    "frozen_next_latent_plus_ema_successor_dataset_next_action"
                ),
                "goal_conditioning": "none",
                "real_td_weight": expected_weights[0],
                "predicted_td_weight": expected_weights[1],
                "predicted_context_detach": expected_weights[2],
            }
        )
    else:
        expected_weights = {
            "parallel_real": (1.0, 0.0, False, 0.0, 0.0),
            "serial_decoupled": (0.0, 1.0, True, 0.0, 0.0),
            "serial_coupled": (0.0, 1.0, False, 0.0, 0.0),
            "hybrid": (1.0, 1.0, False, 0.0, 0.0),
            "goal_hybrid": (1.0, 1.0, False, 1.0, 1.0),
            "imaginary_hybrid": (1.0, 1.0, False, 0.0, 0.0),
        }[variant]
        expected_td_target = (
            "real_ema_next_feature_plus_ema_successor_imagined_next_history_"
            "dataset_next_action"
            if variant == IMAGINARY_VARIANT
            else "ema_next_latent_plus_ema_successor_dataset_next_action"
        )
        expected_objective.update(
            {
                "td_target": expected_td_target,
                "goal_conditioning": "none",
                "real_td_weight": expected_weights[0],
                "predicted_td_weight": expected_weights[1],
                "predicted_context_detach": expected_weights[2],
            }
        )
    for key, expected in expected_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")

    if variant != DIRECT_GOAL_VARIANT:
        for key, expected in {
            "real_goal_td_weight": expected_weights[3],
            "predicted_goal_td_weight": expected_weights[4],
        }.items():
            if float(objective.get(key, 0.0)) != expected:
                raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    if frozen_pretrained:
        frozen_objective = {
            "world_model_gradient": "stop_gradient",
            "td_branches": ["real_context"],
            "goal_source": "uniform_reachable_future_frozen_latent_same_clip",
            "goal_validation": "max_reachable_future_frozen_latent_same_clip",
            "goal_score": "negative_successor_cost",
            "goal_sampling_seed_offset": 1,
        }
        for key, expected in frozen_objective.items():
            if objective.get(key) != expected:
                raise ValueError(f"joint_objective.{key} must be {expected!r}.")
        spec.validate_objective(objective)
    if variant == GOAL_VARIANT:
        goal_objective = {
            "goal_readout_td": "hindsight_future_ema_latent_bellman_cost",
            "goal_enters_successor_head": False,
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "goal_validation": "exact_conditional_uniform_future_expectation",
            "goal_terminal": "dataset_terminal_or_next_state_is_goal",
            "goal_cost": "normalized_discounted_latent_mse",
            "goal_readout_precision": "float32",
            "goal_sampling_seed_offset": 1,
        }
        for key, expected in goal_objective.items():
            if objective.get(key) != expected:
                raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    if variant == IMAGINARY_VARIANT:
        imaginary_objective = {
            "imaginary_transition_model": "ema_lewm_predictor",
            "imaginary_immediate_feature": "real_ema_next_latent",
            "imaginary_bootstrap_history": (
                "shift_real_ema_history_append_ema_predicted_next_latent"
            ),
            "imaginary_horizon": 1,
            "imaginary_target_gradient": "stop_gradient",
        }
        for key, expected in imaginary_objective.items():
            if objective.get(key) != expected:
                raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    if variant == DIRECT_GOAL_VARIANT:
        direct_objective = {
            "direct_goal_td": "hindsight_future_ema_latent_bellman_cost",
            "goal_enters_critic_head": True,
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "goal_validation": "exact_conditional_uniform_future_expectation",
            "goal_terminal": "dataset_terminal_or_next_state_is_goal",
            "goal_cost": "normalized_discounted_latent_mse",
            "goal_sampling_seed_offset": 1,
        }
        for key, expected in direct_objective.items():
            if objective.get(key) != expected:
                raise ValueError(f"joint_objective.{key} must be {expected!r}.")

        head = protocol.get("critic", {})
        locked_head = {
            "objective_version": DIRECT_GOAL_OBJECTIVE_VERSION,
            "architecture": "direct_goal_critic_head",
            "action_conditioning": "dataset_current_action",
            "bootstrap_action": "dataset_next_action",
            "terminal_source": "next_action_nan_invalid",
            "goal_conditioning": "direct_latent_input",
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "goal_offset_weighting": "uniform_per_transition",
            "goal_terminal_condition": "dataset_terminal_or_next_state_is_goal",
            "td_branches": ["real_context", "predicted_context"],
            "actor": "none",
            "reward": "none",
        }
        section = "critic"
        target_head_decay_key = "target_critic_ema_decay"
        clamp_key = "clamp_critic_cost"
    else:
        head = protocol.get("successor", {})
        locked_head = {
            "objective_version": spec.objective_version,
            "architecture": "actor_free_successor_head",
            "feature_basis": "augmented_latent_squared_distance",
            "action_conditioning": "dataset_current_action",
            "bootstrap_action": "dataset_next_action",
            "terminal_source": "next_action_nan_invalid",
            "goal_conditioning": "none",
            "actor": "none",
            "reward": "none",
        }
        if variant == GOAL_VARIANT:
            locked_head.update(
                {
                    "goal_readout_training": True,
                    "goal_source": "uniform_reachable_future_ema_latent_same_clip",
                    "goal_offset_weighting": "uniform_per_transition",
                    "goal_terminal_condition": (
                        "dataset_terminal_or_next_state_is_goal"
                    ),
                    "goal_readout_branches": [
                        "real_context",
                        "predicted_context",
                    ],
                    "goal_readout_precision": "float32",
                }
            )
        if variant == IMAGINARY_VARIANT:
            locked_head.update(
                {
                    "immediate_feature_source": "real_ema_next_latent",
                    "bootstrap_state_source": (
                        "ema_lewm_predicted_next_from_real_ema_history"
                    ),
                    "imaginary_horizon": 1,
                    "imaginary_predictor_gradient": "target_ema_stop_gradient",
                }
            )
        if frozen_pretrained:
            locked_head.update(
                {
                    "training_branches": ["real_context"],
                    "pretrained_world_model_frozen": True,
                }
            )
        section = "successor"
        target_head_decay_key = "target_successor_ema_decay"
        clamp_key = "clamp_successor_cost"

    for key, expected in locked_head.items():
        if head.get(key) != expected:
            raise ValueError(f"{section}.{key} must be {expected!r}.")
    if int(head.get("hidden_dim", 0)) <= 0:
        raise ValueError(f"{section}.hidden_dim must be positive.")
    if not 0.0 <= float(head.get("gamma", -1.0)) < 1.0:
        raise ValueError(f"{section}.gamma must lie in [0, 1).")
    for key in ("target_world_ema_decay", target_head_decay_key):
        if not 0.0 <= float(head.get(key, -1.0)) < 1.0:
            raise ValueError(f"{section}.{key} must lie in [0, 1).")
    if frozen_pretrained and float(head["target_world_ema_decay"]) != 0.0:
        raise ValueError(
            "Frozen variants require successor.target_world_ema_decay = 0.0."
        )
    if not 0.0 <= float(head.get("loss_warmup_fraction", -1.0)) < 1.0:
        raise ValueError(f"{section}.loss_warmup_fraction must lie in [0, 1).")
    if float(head.get("planning_weight", -1.0)) < 0.0:
        raise ValueError(f"{section}.planning_weight cannot be negative.")
    if head.get(clamp_key) is not True:
        raise ValueError(f"{section}.{clamp_key} must be true.")

    loss = protocol.get("loss", {})
    sigreg = loss.get("sigreg", {})
    if frozen_pretrained:
        if loss.get("prediction") != "mse" or float(sigreg.get("weight", -1.0)) != 0.0:
            raise ValueError("Frozen variants require zero SIGReg.")
    elif loss.get("prediction") != "mse" or float(sigreg.get("weight", -1.0)) != 0.09:
        raise ValueError("The original LeWM prediction + 0.09 SIGReg loss is fixed.")
    local_windows = num_steps - history
    effective_batch = (
        int(protocol.get("loader", {}).get("batch_size", 0)) * local_windows
    )
    if int(sigreg.get("effective_batch_size", 0)) != effective_batch:
        raise ValueError("The overlapping-window SIGReg effective batch changed.")
    if min(int(sigreg.get("knots", 0)), int(sigreg.get("num_projections", 0))) <= 0:
        raise ValueError("SIGReg settings must be positive.")

    split = protocol.get("split", {})
    if split.get("unit") != "sequence_clip":
        raise ValueError("Actor-Free TD-LeWM splits sequence clips.")
    if not 0.0 < float(split.get("train_fraction", 0.0)) < 1.0 or not math.isclose(
        float(split.get("train_fraction", 0.0))
        + float(split.get("validation_fraction", 0.0)),
        1.0,
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    loader = protocol.get("loader", {})
    if int(loader.get("batch_size", 0)) <= 0 or int(loader.get("workers", -1)) < 0:
        raise ValueError("Loader batch size and workers are invalid.")
    if int(loader.get("prefetch_factor", 0)) <= 0:
        raise ValueError("loader.prefetch_factor must be positive.")
    for key in ("device_image_preprocessing", "episode_streaming"):
        if not isinstance(loader.get(key), bool):
            raise TypeError(f"loader.{key} must be boolean.")
    if (
        not 1
        <= int(loader.get("minimum_unique_episodes_per_batch", 0))
        <= int(loader["batch_size"])
    ):
        raise ValueError("minimum_unique_episodes_per_batch is invalid.")

    training = protocol.get("training", {})
    epochs = int(training.get("epochs", 0))
    epoch_steps = int(training.get("optimizer_steps_per_epoch", 0))
    if epochs != int(training.get("scheduler_epochs", -1)):
        raise ValueError("Scheduler and trainer epochs must match.")
    if epochs * epoch_steps != FORMAL_OPTIMIZER_UPDATES:
        raise ValueError("Every variant must receive exactly 127960 updates.")
    if training.get("model_compile") is not False:
        raise ValueError("The locked comparison keeps model compilation disabled.")
    if protocol.get("scheduler", {}).get("interval") != "optimizer_step":
        raise ValueError("The scheduler must step per optimizer update.")
    optimizer = protocol.get("optimizer", {})
    head_learning_rate_key = (
        "critic_learning_rate"
        if variant == DIRECT_GOAL_VARIANT
        else "successor_learning_rate"
    )
    if frozen_pretrained:
        world_learning_rate = float(optimizer.get("world_model_learning_rate", -1.0))
        head_learning_rate = float(optimizer.get(head_learning_rate_key, 0.0))
        if world_learning_rate != 0.0 or head_learning_rate <= 0.0:
            raise ValueError(
                "Frozen variants require zero world-model learning rate and a "
                "positive successor learning rate."
            )
    elif (
        min(
            float(optimizer.get("world_model_learning_rate", 0.0)),
            float(optimizer.get(head_learning_rate_key, 0.0)),
        )
        <= 0.0
    ):
        raise ValueError("Both optimizer learning rates must be positive.")

    dataset = protocol.get("dataset", {})
    lance = dataset.get("lance", {})
    if lance.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("The Lance conversion must use stable-worldmodel 0.1.1.")
    if lance.get("image_codec") != "jpeg" or lance.get("jpeg_quality") != 100:
        raise ValueError("The audited Actor-Free input is JPEG quality 100.")
    if lance.get("source", {}).get("sha256") != dataset.get("optimized_layout", {}).get(
        "sha256"
    ):
        raise ValueError("The Lance source differs from the audited Cube layout.")


@dataclass(frozen=True)
class ActorFreeTDInputs:
    """Clean actions, terminal flags and teacher-forced predicted contexts."""

    actions: torch.Tensor
    terminals: torch.Tensor
    predicted_context: torch.Tensor
    one_step_predictions: torch.Tensor


def build_actor_free_td_inputs(
    real_latents: torch.Tensor,
    raw_actions: torch.Tensor,
    local_prediction: torch.Tensor,
    *,
    history_size: int,
) -> ActorFreeTDInputs:
    """Align local LeWM predictions and TD terminals without losing NaN evidence.

    The last token of local window ``start`` predicts latent
    ``start + history_size``.  It therefore replaces indices ``history_size:``.
    A transition at current time ``t`` is terminal exactly when its recorded
    bootstrap action ``action[t + 1]`` is non-finite.
    """

    if real_latents.ndim != 3 or raw_actions.ndim != 3:
        raise ValueError("Latents and actions must have shape (batch, time, dim).")
    if real_latents.shape[:2] != raw_actions.shape[:2]:
        raise ValueError("Latents and actions must share batch and time axes.")
    batch, time, embed_dim = real_latents.shape
    local_count = time - int(history_size)
    expected = (local_count * batch, int(history_size), embed_dim)
    if local_count <= 0 or local_prediction.shape != expected:
        raise ValueError(
            f"local_prediction must have shape {expected}, found {tuple(local_prediction.shape)}."
        )
    one_step = (
        local_prediction[:, -1].reshape(local_count, batch, embed_dim).transpose(0, 1)
    )
    predicted_context = torch.cat((real_latents[:, :history_size], one_step), dim=1)
    action_valid = torch.isfinite(raw_actions).all(dim=-1)
    terminals = torch.zeros_like(action_valid)
    terminals[:, :-1] = ~action_valid[:, 1:]
    return ActorFreeTDInputs(
        actions=torch.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0),
        terminals=terminals,
        predicted_context=predicted_context,
        one_step_predictions=one_step,
    )


def build_imaginary_ema_next_latents(
    target_local_prediction: torch.Tensor,
    *,
    batch_size: int,
    num_steps: int,
    history_size: int,
) -> torch.Tensor:
    """Align EMA LeWM window predictions with TD next-state indices."""

    local_count = int(num_steps) - int(history_size)
    if target_local_prediction.ndim != 3 or local_count <= 1:
        raise ValueError("EMA local predictions must contain all TD next states.")
    expected_prefix = (local_count * int(batch_size), int(history_size))
    if target_local_prediction.shape[:2] != expected_prefix:
        raise ValueError(
            "EMA local predictions must have start-major shape "
            f"{expected_prefix + (target_local_prediction.shape[-1],)}."
        )
    embed_dim = target_local_prediction.shape[-1]
    one_step = (
        target_local_prediction[:, -1]
        .reshape(local_count, int(batch_size), embed_dim)
        .transpose(0, 1)
    )
    return one_step[:, 1:]


@dataclass(frozen=True)
class ActorFreeTrainingSchedule:
    total_scheduler_steps: int
    max_epochs: int


def load_bound_training_split(
    split_indices_path: str | Path,
    *,
    dataset_size: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the exact external split shared by training and derived artifacts."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    fractions = (float(train_fraction), float(validation_fraction))
    if any(not 0.0 < fraction < 1.0 for fraction in fractions) or not math.isclose(
        sum(fractions), 1.0
    ):
        raise ValueError("Training and validation fractions must sum to one.")

    train, validation, metadata = load_split_indices(split_indices_path)
    partition = np.concatenate((train, validation))
    if (
        partition.size != int(dataset_size)
        or np.any(partition < 0)
        or np.any(partition >= int(dataset_size))
        or np.unique(partition).size != int(dataset_size)
    ):
        raise ValueError(
            "The supplied split_indices.npz must partition every dataset clip "
            "exactly once."
        )

    expected_sizes = [math.floor(fraction * dataset_size) for fraction in fractions]
    for index in range(dataset_size - sum(expected_sizes)):
        expected_sizes[index % len(expected_sizes)] += 1
    if [int(train.size), int(validation.size)] != expected_sizes:
        raise ValueError(
            "The supplied split sizes differ from the protocol fractions: "
            f"expected {expected_sizes}, found {[int(train.size), int(validation.size)]}."
        )
    return (
        train,
        validation,
        {
            **metadata,
            "binding": "externally_supplied_exact_artifact",
        },
    )


def resolve_actor_free_training_schedule(
    protocol: dict[str, Any],
    *,
    smoke: bool,
    resume: str,
    max_steps: int | None,
    train_limit: int | float,
) -> ActorFreeTrainingSchedule:
    """Keep a two-call smoke resume inside one fixed four-update schedule."""

    formal_steps = int(protocol["training"]["scheduler_epochs"]) * int(
        protocol["training"]["optimizer_steps_per_epoch"]
    )
    if smoke:
        return ActorFreeTrainingSchedule(
            total_scheduler_steps=2 * int(train_limit),
            max_epochs=2 if resume == "required" else 1,
        )
    if max_steps is not None:
        return ActorFreeTrainingSchedule(
            total_scheduler_steps=int(train_limit), max_epochs=1
        )
    return ActorFreeTrainingSchedule(
        total_scheduler_steps=formal_steps,
        max_epochs=int(protocol["training"]["epochs"]),
    )


def _encode_online_and_target(
    online_model: Any,
    target_model: Any,
    encoder_input: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    online = online_model.encode(dict(encoder_input))
    with torch.no_grad():
        target = target_model.encode(dict(encoder_input))
    return online["emb"], online["act_emb"], target["emb"], target["act_emb"]


def frozen_weight_metrics(stage: str, diagnostics: Any) -> dict[str, torch.Tensor]:
    """Convert shared detached-weight diagnostics to stable metric names."""

    return {
        f"{stage}/weight_signal_mean": diagnostics.signal.mean,
        f"{stage}/weight_signal_std": diagnostics.signal.std,
        f"{stage}/weight_mean": diagnostics.normalized_weights.mean,
        f"{stage}/weight_std": diagnostics.normalized_weights.std,
        f"{stage}/weight_min": diagnostics.normalized_weights.minimum,
        f"{stage}/weight_max": diagnostics.normalized_weights.maximum,
        f"{stage}/weight_clipped_fraction": diagnostics.clipped_fraction,
        f"{stage}/weight_effective_sample_size": diagnostics.effective_sample_size,
        f"{stage}/weight_effective_sample_fraction": (
            diagnostics.effective_sample_fraction
        ),
    }


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    spec: FrozenActorFreeTDSpec,
    action_block_dim: int,
    device_image_preprocessing: bool,
    goal_generator: torch.Generator | None = None,
    neighbor_index: StateNeighborActionIndex | None = None,
):
    import lightning as pl
    import stable_worldmodel as swm

    class ActorFreeTDLeWMTrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            self.target_model = copy.deepcopy(world_model).requires_grad_(False)
            self.target_model.eval()
            self.objective_spec = spec
            self.variant = str(protocol["variant"])
            self.frozen_world_model = self.variant in FROZEN_PRETRAINED_VARIANTS
            if self.frozen_world_model:
                self.model.requires_grad_(False)
                self.model.eval()
            self.is_direct_goal_critic = self.variant == DIRECT_GOAL_VARIANT
            if self.frozen_world_model:
                self.neighbor_index = neighbor_index
                if self.objective_spec.requires_neighbor_index != (
                    self.neighbor_index is not None
                ):
                    raise ValueError(
                        "Neighbor-action variants require exactly one validated "
                        "state-neighbor index."
                    )
            head_cfg = (
                protocol["critic"]
                if self.is_direct_goal_critic
                else protocol["successor"]
            )
            head_type = (
                DirectGoalCriticHead
                if self.is_direct_goal_critic
                else ActorFreeSuccessorHead
            )
            head = head_type(
                embed_dim=int(protocol["model"]["embed_dim"]),
                action_dim=action_block_dim,
                history_size=int(protocol["sequence"]["history_frames"]),
                hidden_dim=int(head_cfg["hidden_dim"]),
            )
            target_head = head.make_target()
            target_head.eval()
            if self.is_direct_goal_critic:
                self.critic = head
                self.target_critic = target_head
            else:
                self.successor = head
                self.target_successor = target_head
            self.goal_generator = goal_generator
            if self.variant in HINDSIGHT_GOAL_VARIANTS and self.goal_generator is None:
                raise ValueError(f"{self.variant} requires a dedicated goal generator.")
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.gamma = float(head_cfg["gamma"])
            self.target_world_ema_decay = float(head_cfg["target_world_ema_decay"])
            self.target_head_ema_decay = float(
                head_cfg[
                    "target_critic_ema_decay"
                    if self.is_direct_goal_critic
                    else "target_successor_ema_decay"
                ]
            )
            self.auxiliary_warmup_steps = int(
                float(head_cfg["loss_warmup_fraction"]) * total_steps
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
            if self.frozen_world_model:
                self.sigreg = None
            else:
                self.sigreg = swm.wm.SIGReg(
                    knots=int(sigreg["knots"]),
                    num_proj=int(sigreg["num_projections"]),
                )

        def train(self, mode: bool = True):
            super().train(mode)
            if self.frozen_world_model:
                self.model.eval()
            self.target_model.eval()
            self._target_head().eval()
            return self

        def _online_head(self):
            return self.critic if self.is_direct_goal_critic else self.successor

        def _target_head(self):
            return (
                self.target_critic
                if self.is_direct_goal_critic
                else self.target_successor
            )

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
                size=int(protocol["image_preprocessing"]["size"]),
            )

        def _forward_frozen_loss(
            self,
            batch: dict[str, Any],
            embeddings: torch.Tensor,
            raw_actions: torch.Tensor,
            *,
            stage: str,
            batch_size: int,
            episode_ids: torch.Tensor | None,
            cache_bytes: torch.Tensor | None,
        ) -> torch.Tensor:
            """Run the real-only objective while E+F remain inference-only."""

            action_valid = torch.isfinite(raw_actions).all(dim=-1)
            terminals = torch.zeros_like(action_valid)
            terminals[:, :-1] = ~action_valid[:, 1:]
            actions = torch.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)
            td_batch = build_frozen_real_td_batch(
                self.successor,
                self.target_successor,
                embeddings,
                embeddings,
                actions,
                gamma=self.gamma,
                terminals=terminals,
                first_current_index=self.history_size,
            )
            offset_limits = actor_free_goal_future_offset_limits(
                terminals, first_current_index=self.history_size
            )
            if bool(torch.any(offset_limits < 1)):
                raise RuntimeError(
                    "Frozen-objective clips must not contain transitions after an "
                    "episode terminal."
                )
            if stage == "train":
                assert self.goal_generator is not None
                goal_offsets = sample_actor_free_goal_offsets(
                    terminals,
                    first_current_index=self.history_size,
                    generator=self.goal_generator,
                )
            else:
                # Validation is deterministic and stresses the longest reachable
                # temporal dependency for every transition.
                goal_offsets = offset_limits
            goals = gather_hindsight_goals(
                embeddings,
                terminals,
                self.history_size,
                goal_offsets,
            )
            per_transition_td = per_transition_vector_td_mse(
                td_batch.prediction, td_batch.target
            )
            base_td_loss = per_transition_td.mean()
            zero = base_td_loss.new_zeros(())
            metrics: dict[str, torch.Tensor] = {
                f"{stage}/real_td_loss": base_td_loss.detach(),
                f"{stage}/predicted_td_loss": zero,
                f"{stage}/prediction_loss": zero,
                f"{stage}/sigreg_loss": zero,
                f"{stage}/td_prediction_mean": (
                    td_batch.prediction.detach().float().mean()
                ),
                f"{stage}/td_target_mean": td_batch.target.float().mean(),
                f"{stage}/terminal_fraction": (
                    td_batch.aligned_terminal.float().mean()
                ),
                f"{stage}/td_pairs": base_td_loss.new_tensor(
                    float(per_transition_td.numel())
                ),
                f"{stage}/goal_offset_mean": goal_offsets.float().mean(),
                f"{stage}/goal_offset_max": goal_offsets.float().max(),
            }

            method_output = self.objective_spec.compute_loss(
                FrozenTDContext(
                    protocol=protocol,
                    successor=self.successor,
                    td_batch=td_batch,
                    goals=goals,
                    per_transition_td=per_transition_td,
                    base_td_loss=base_td_loss,
                    batch=batch,
                    stage=stage,
                    batch_size=batch_size,
                    neighbor_index=self.neighbor_index,
                )
            )
            weighted_auxiliary = method_output.loss
            if weighted_auxiliary.ndim != 0:
                raise RuntimeError("A frozen TD objective must return one scalar loss.")
            metrics.update(method_output.metrics)
            auxiliary_scale = self._auxiliary_scale()
            loss = auxiliary_scale * weighted_auxiliary
            metrics.update(
                {
                    f"{stage}/loss": loss.detach(),
                    f"{stage}/td_loss": weighted_auxiliary.detach(),
                    f"{stage}/successor_td_loss": weighted_auxiliary.detach(),
                    f"{stage}/td_weight_scale": loss.new_tensor(auxiliary_scale),
                }
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

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            if self.frozen_world_model:
                episode_ids = batch.get("_tdwm_episode_id")
                cache_bytes = batch.get("_tdwm_cache_bytes")
                raw_actions = batch["action"]
                if raw_actions.ndim != 3:
                    raise RuntimeError("Actions must be flattened frame-skip blocks.")
                if "pixels" in batch:
                    raise RuntimeError(
                        "Frozen variants must consume the audited latent store, "
                        "not pixel batches."
                    )
                embeddings = batch.get("_tdwm_frozen_latents")
                if not isinstance(embeddings, torch.Tensor):
                    raise RuntimeError(
                        "Frozen variants require _tdwm_frozen_latents from the "
                        "audited latent store."
                    )
                if embeddings.ndim != 3:
                    raise RuntimeError(
                        "Frozen latent clips must have shape [batch, steps, embed]."
                    )
                expected_steps = int(protocol["sequence"]["num_steps"])
                expected_embed_dim = int(protocol["model"]["embed_dim"])
                if tuple(embeddings.shape[1:]) != (
                    expected_steps,
                    expected_embed_dim,
                ):
                    raise RuntimeError(
                        "The frozen latent clip has an unexpected shape."
                    )
                if not embeddings.is_floating_point() or not bool(
                    torch.isfinite(embeddings).all()
                ):
                    raise RuntimeError("Frozen latent clips must be finite floats.")
                if raw_actions.shape[:2] != embeddings.shape[:2]:
                    raise RuntimeError(
                        "Frozen actions and latent clips have different shapes."
                    )
                return self._forward_frozen_loss(
                    batch,
                    embeddings,
                    raw_actions,
                    stage=stage,
                    batch_size=int(embeddings.shape[0]),
                    episode_ids=episode_ids,
                    cache_bytes=cache_bytes,
                )
            batch_size = int(batch["pixels"].shape[0])
            episode_ids = batch.get("_tdwm_episode_id")
            cache_bytes = batch.get("_tdwm_cache_bytes")
            pixels = self._preprocess(batch["pixels"])
            raw_actions = batch["action"]
            if raw_actions.ndim != 3:
                raise RuntimeError("Actions must be flattened frame-skip blocks.")
            cleaned_actions = torch.nan_to_num(
                raw_actions, nan=0.0, posinf=0.0, neginf=0.0
            )
            encoder_input = {
                key: value
                for key, value in batch.items()
                if not key.startswith("_tdwm_") and key not in {"pixels", "action"}
            }
            encoder_input.update({"pixels": pixels, "action": cleaned_actions})
            (
                embeddings,
                action_embeddings,
                target_embeddings,
                target_action_embeddings,
            ) = _encode_online_and_target(self.model, self.target_model, encoder_input)
            expected_steps = int(protocol["sequence"]["num_steps"])
            if embeddings.shape[1] != expected_steps:
                raise RuntimeError("The encoded clip has an unexpected length.")

            local_count = expected_steps - self.history_size
            local_histories = torch.cat(
                [
                    embeddings[:, start : start + self.history_size]
                    for start in range(local_count)
                ],
                dim=0,
            )
            local_actions = torch.cat(
                [
                    action_embeddings[:, start : start + self.history_size]
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
            prediction_loss = (local_prediction - local_targets).square().mean()
            if self.sigreg is None:
                sigreg_loss = prediction_loss.new_zeros(())
            else:
                sigreg_sequences = torch.cat(
                    [
                        embeddings[:, start : start + self.history_size + 1]
                        for start in range(local_count)
                    ],
                    dim=0,
                )
                sigreg_loss = self.sigreg(sigreg_sequences.transpose(0, 1))

            td_inputs = build_actor_free_td_inputs(
                embeddings,
                raw_actions,
                local_prediction,
                history_size=self.history_size,
            )
            imagined_ema_next_latents = None
            if self.variant == IMAGINARY_VARIANT:
                target_local_histories = torch.cat(
                    [
                        target_embeddings[:, start : start + self.history_size]
                        for start in range(local_count)
                    ],
                    dim=0,
                )
                target_local_actions = torch.cat(
                    [
                        target_action_embeddings[:, start : start + self.history_size]
                        for start in range(local_count)
                    ],
                    dim=0,
                )
                with torch.no_grad():
                    target_local_prediction = self.target_model.predict(
                        target_local_histories, target_local_actions
                    )
                    imagined_ema_next_latents = build_imaginary_ema_next_latents(
                        target_local_prediction,
                        batch_size=batch_size,
                        num_steps=expected_steps,
                        history_size=self.history_size,
                    )
            goal_offsets = None
            if self.variant in HINDSIGHT_GOAL_VARIANTS and stage == "train":
                assert self.goal_generator is not None
                goal_offsets = sample_actor_free_goal_offsets(
                    td_inputs.terminals,
                    first_current_index=self.history_size,
                    generator=self.goal_generator,
                )
            objective = protocol["joint_objective"]
            if self.is_direct_goal_critic:
                critic_output = direct_goal_critic_td_objective(
                    self.critic,
                    self.target_critic,
                    embeddings,
                    td_inputs.predicted_context,
                    target_embeddings,
                    td_inputs.actions,
                    gamma=self.gamma,
                    terminals=td_inputs.terminals,
                    first_current_index=self.history_size,
                    goal_offsets=goal_offsets,
                )
                weighted_auxiliary = (
                    float(objective["predicted_critic_td_weight"])
                    * critic_output.predicted_td_loss
                    + float(objective["real_critic_td_weight"])
                    * critic_output.real_td_loss
                )
                auxiliary_metrics = {
                    f"{stage}/critic_td_loss": weighted_auxiliary.detach(),
                    f"{stage}/predicted_critic_td_loss": (
                        critic_output.predicted_td_loss.detach()
                    ),
                    f"{stage}/real_critic_td_loss": (
                        critic_output.real_td_loss.detach()
                    ),
                    f"{stage}/critic_prediction_mean": (
                        critic_output.prediction_mean.detach()
                    ),
                    f"{stage}/critic_target_mean": critic_output.target_mean.detach(),
                    f"{stage}/critic_terminal_fraction": (
                        critic_output.terminal_fraction.detach()
                    ),
                    f"{stage}/critic_negative_prediction_fraction": (
                        critic_output.negative_prediction_fraction.detach()
                    ),
                    f"{stage}/critic_pairs": critic_output.pair_count.detach().to(
                        prediction_loss
                    ),
                }
            else:
                td_output = actor_free_td_objective(
                    self.successor,
                    self.target_successor,
                    embeddings,
                    td_inputs.predicted_context,
                    target_embeddings,
                    td_inputs.actions,
                    gamma=self.gamma,
                    variant=self.variant,
                    terminals=td_inputs.terminals,
                    first_current_index=self.history_size,
                    goal_offsets=goal_offsets,
                    imagined_ema_next_latents=imagined_ema_next_latents,
                )
                real_td_loss = (
                    td_output.real_td_loss
                    if td_output.real_td_loss is not None
                    else prediction_loss.new_zeros(())
                )
                weighted_td = (
                    float(objective["predicted_td_weight"])
                    * td_output.predicted_td_loss
                    + float(objective["real_td_weight"]) * real_td_loss
                )
                real_goal_td_loss = (
                    td_output.real_goal_td_loss
                    if td_output.real_goal_td_loss is not None
                    else prediction_loss.new_zeros((), dtype=torch.float32)
                )
                weighted_goal_td = (
                    float(objective.get("predicted_goal_td_weight", 0.0))
                    * td_output.predicted_goal_td_loss
                    + float(objective.get("real_goal_td_weight", 0.0))
                    * real_goal_td_loss
                )
                weighted_auxiliary = weighted_td + weighted_goal_td
                auxiliary_metrics = {
                    f"{stage}/successor_td_loss": weighted_td.detach(),
                    f"{stage}/predicted_td_loss": (
                        td_output.predicted_td_loss.detach()
                    ),
                    f"{stage}/real_td_loss": real_td_loss.detach(),
                    f"{stage}/goal_td_loss": weighted_goal_td.detach(),
                    f"{stage}/predicted_goal_td_loss": (
                        td_output.predicted_goal_td_loss.detach()
                    ),
                    f"{stage}/real_goal_td_loss": real_goal_td_loss.detach(),
                    f"{stage}/goal_prediction_mean": (
                        td_output.goal_prediction_mean.detach()
                    ),
                    f"{stage}/goal_target_mean": td_output.goal_target_mean.detach(),
                    f"{stage}/goal_terminal_fraction": (
                        td_output.goal_terminal_fraction.detach()
                    ),
                    f"{stage}/goal_negative_prediction_fraction": (
                        td_output.goal_negative_prediction_fraction.detach()
                    ),
                    f"{stage}/goal_pairs": td_output.goal_pair_count.detach().to(
                        prediction_loss
                    ),
                    f"{stage}/td_prediction_mean": (td_output.prediction_mean.detach()),
                    f"{stage}/td_target_mean": td_output.target_mean.detach(),
                    f"{stage}/terminal_fraction": (
                        td_output.terminal_fraction.detach()
                    ),
                    f"{stage}/td_pairs": prediction_loss.new_tensor(
                        float(td_output.pair_count)
                    ),
                    f"{stage}/imaginary_next_mse": (
                        td_output.imaginary_next_mse.detach()
                    ),
                }
            auxiliary_scale = self._auxiliary_scale()
            loss = (
                prediction_loss
                + float(protocol["loss"]["sigreg"]["weight"]) * sigreg_loss
                + auxiliary_scale * weighted_auxiliary
            )
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/prediction_loss": prediction_loss.detach(),
                f"{stage}/sigreg_loss": sigreg_loss.detach(),
                f"{stage}/td_loss": weighted_auxiliary.detach(),
                f"{stage}/td_weight_scale": loss.new_tensor(auxiliary_scale),
                **auxiliary_metrics,
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
            if not self.frozen_world_model:
                ema_update(
                    self.target_model,
                    self.model,
                    decay=self.target_world_ema_decay,
                )
            ema_update(
                self._target_head(),
                self._online_head(),
                decay=self.target_head_ema_decay,
            )

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            if self.frozen_world_model:
                if any(
                    parameter.requires_grad for parameter in self.model.parameters()
                ):
                    raise RuntimeError("The pretrained world model must remain frozen.")
                parameter_groups = [
                    {
                        "params": list(self._online_head().parameters()),
                        "lr": float(optimizer_cfg["successor_learning_rate"]),
                    }
                ]
            else:
                parameter_groups = [
                    {
                        "params": list(self.model.parameters()),
                        "lr": float(optimizer_cfg["world_model_learning_rate"]),
                    },
                    {
                        "params": list(self._online_head().parameters()),
                        "lr": float(
                            optimizer_cfg[
                                "critic_learning_rate"
                                if self.is_direct_goal_critic
                                else "successor_learning_rate"
                            ]
                        ),
                    },
                ]
            optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=float(optimizer_cfg["weight_decay"]),
            )
            warmup_steps = max(
                1,
                int(float(protocol["scheduler"]["warmup_fraction"]) * total_steps),
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(float(progress), 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return ActorFreeTDLeWMTrainingModule()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bound_frozen_latent_store(
    artifact_dir: str | Path,
    *,
    protocol: dict[str, Any],
    dataset_source: dict[str, Any],
    action_dim: int,
) -> tuple[FrozenLatentStore, dict[str, Any]]:
    """Load one exact frozen store and revalidate each external input binding."""

    if dataset_source.get("format") != "lance":
        raise ValueError("Frozen latent training requires the audited Lance dataset.")
    conversion = dataset_source.get("conversion_manifest")
    if not isinstance(conversion, dict):
        raise ValueError("The Lance conversion manifest is missing.")
    source_sha256 = conversion.get("source", {}).get("sha256")
    if not isinstance(source_sha256, str):
        raise ValueError("The Lance conversion manifest is missing source.sha256.")
    conversion_path_value = dataset_source.get("conversion_manifest_path")
    if not isinstance(conversion_path_value, str) or not conversion_path_value:
        raise ValueError("The Lance conversion-manifest path is missing.")
    conversion_path = Path(conversion_path_value).expanduser().resolve()
    conversion_sha256 = _file_sha256(conversion_path)

    expected_checkpoint_sha256 = protocol["pretrained_world_model"]["checkpoint_sha256"]
    store = FrozenLatentStore(
        artifact_dir,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_dataset_source_sha256=source_sha256,
        expected_frame_skip=int(protocol["sequence"]["frame_skip"]),
        expected_history_frames=int(protocol["sequence"]["history_frames"]),
        expected_embed_dim=int(protocol["model"]["embed_dim"]),
        expected_action_dim=int(action_dim),
    )
    source_metadata = store.manifest.get("source_metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError("Frozen latent store source_metadata is missing.")
    if source_metadata.get("dataset_manifest_sha256") != conversion_sha256:
        raise ValueError(
            "Frozen latent store was built from a different Lance manifest."
        )

    normalization_path_value = source_metadata.get("column_normalization_path")
    if not isinstance(normalization_path_value, str) or not normalization_path_value:
        raise ValueError(
            "Frozen latent store does not bind its column-normalization input."
        )
    normalization_path = Path(normalization_path_value).expanduser().resolve()
    normalization_sha256 = _file_sha256(normalization_path)
    if normalization_sha256 != store.manifest["column_normalization_sha256"]:
        raise ValueError(
            "Frozen latent store column normalization differs from its source file."
        )

    expected_stable_version = str(protocol["runtime"]["stable_worldmodel_version"])
    if source_metadata.get("stable_worldmodel_version") != expected_stable_version:
        raise ValueError(
            "Frozen latent store Stable World Model version differs from protocol."
        )
    image_metadata = source_metadata.get("image_preprocessing")
    expected_image = protocol["image_preprocessing"]
    if not isinstance(image_metadata, dict) or (
        int(image_metadata.get("size", -1)) != int(expected_image["size"])
        or image_metadata.get("mean") != list(expected_image["mean"])
        or image_metadata.get("std") != list(expected_image["std"])
    ):
        raise ValueError(
            "Frozen latent store image preprocessing differs from protocol."
        )
    precision_mapping = {"bf16-mixed": "bfloat16", "32-true": "float32"}
    expected_extraction_precision = precision_mapping.get(
        str(protocol["training"]["precision"])
    )
    if (
        expected_extraction_precision is None
        or source_metadata.get("extraction_precision") != expected_extraction_precision
    ):
        raise ValueError(
            "Frozen latent extraction precision differs from training protocol."
        )

    parity = source_metadata.get("online_cache_parity_audit")
    formal_parity = (
        parity.get("formal_preprocessing_smoke", {}) if isinstance(parity, dict) else {}
    )
    if (
        not isinstance(parity, dict)
        or parity.get("status") != "passed_by_construction_for_every_global_row"
        or formal_parity.get("status") != "passed"
    ):
        raise ValueError(
            "Frozen latent store is missing the formal online/cache parity audit."
        )

    files = store.manifest.get("files", {})
    input_file_sha256 = {
        name: files[name]["sha256"]
        for name in ("latents", "action_blocks", "episode_ids")
    }
    identity = {
        "path": str(store.root),
        "manifest_path": str(store.manifest_path),
        "manifest_sha256": store.manifest_sha256,
        "pretrained_checkpoint_sha256": store.manifest["pretrained_checkpoint_sha256"],
        "dataset_source_sha256": source_sha256,
        "dataset_manifest_path": str(conversion_path),
        "dataset_manifest_sha256": conversion_sha256,
        "column_normalization_path": str(normalization_path),
        "column_normalization_sha256": normalization_sha256,
        "input_file_sha256": input_file_sha256,
        "total_rows": store.total_rows,
        "embed_dim": store.embed_dim,
        "frame_skip": store.frame_skip,
        "history_frames": store.history_frames,
        "action_dim": store.action_dim,
        "action_block_dim": store.action_block_dim,
        "git_revision": store.manifest["git_revision"],
        "stable_worldmodel_version": expected_stable_version,
        "extraction_precision": expected_extraction_precision,
        "image_preprocessing": image_metadata,
        "online_cache_parity_audit": parity,
    }
    return store, identity


def _canonical_protocol_sha256(protocol: dict[str, Any]) -> str:
    """Hash the complete parsed protocol with a stable JSON serialization."""

    encoded = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_local_pretrained_lewm_export(
    checkpoint_path: str | Path,
) -> tuple[str, Path, Path]:
    """Resolve one public ``save_pretrained`` LeWM export without guessing."""

    requested = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir = requested if requested.is_dir() else requested.parent
    weights = sorted(checkpoint_dir.glob("*.pt"))
    if len(weights) != 1:
        raise FileNotFoundError(
            "A pretrained LeWM export must contain exactly one .pt file."
        )
    if not requested.is_dir() and requested != weights[0]:
        raise ValueError("The requested checkpoint is not the export weight file.")
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "A pretrained LeWM export must use the public "
            "<cache_dir>/checkpoints/<run_name> layout."
        )
    return checkpoint_dir.name, weights[0], checkpoint_dir.parent.parent


def _verify_completed_pretrained_lewm_run(
    *,
    source_run_name: str,
    source_cache: Path,
    expected_seed: int,
    expected_epoch: int,
) -> dict[str, Any]:
    """Bind a public epoch export to its completed formal LeWM training run."""

    expected_run_name = f"epoch_{int(expected_epoch):02d}"
    if source_run_name != expected_run_name:
        raise ValueError(
            f"The frozen LeWM source must be {expected_run_name}, not "
            f"{source_run_name}."
        )
    run_dir = source_cache.parent.parent.resolve()
    result_path = run_dir / "training_result.json"
    manifest_path = run_dir / "training_manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "The pretrained export must remain beside its training_result.json "
            "and training_manifest.json."
        )
    with result_path.open() as stream:
        result = json.load(stream)
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    formal_steps = FORMAL_OPTIMIZER_UPDATES
    if (
        int(result.get("seed", -1)) != int(expected_seed)
        or result.get("smoke") is not False
        or int(result.get("final_epoch", -1)) != int(expected_epoch)
        or int(result.get("global_step", -1)) != formal_steps
    ):
        raise ValueError(
            "The pretrained LeWM training result is not the completed formal "
            f"epoch-{expected_epoch}/step-{formal_steps} run."
        )
    manifest_training = manifest.get("training", {})
    manifest_protocol = manifest.get("protocol", {})
    if (
        int(manifest.get("seed", -1)) != int(expected_seed)
        or manifest.get("smoke") is not False
        or manifest_protocol.get("method") != "lewm"
        or int(manifest_protocol.get("training", {}).get("epochs", -1))
        != int(expected_epoch)
        or int(manifest_training.get("formal_optimizer_steps", -1)) != formal_steps
        or int(manifest_training.get("configured_optimizer_steps", -1)) != formal_steps
    ):
        raise ValueError(
            "The pretrained LeWM manifest is not the locked completed formal run."
        )
    result_run_dir = Path(str(result.get("run_dir", ""))).expanduser().resolve()
    if result_run_dir != run_dir:
        raise ValueError("The pretrained training result names a different run dir.")
    return {
        "source_training_run_dir": str(run_dir),
        "source_training_result_path": str(result_path),
        "source_training_result_sha256": _file_sha256(result_path),
        "source_training_manifest_path": str(manifest_path),
        "source_training_manifest_sha256": _file_sha256(manifest_path),
        "source_final_epoch": int(result["final_epoch"]),
        "source_global_step": int(result["global_step"]),
    }


def _successor_config(
    protocol: dict[str, Any],
    *,
    spec: FrozenActorFreeTDSpec,
    action_block_dim: int,
    base_export_run_name: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    successor = protocol["successor"]
    pretrained = protocol["pretrained_world_model"]
    objective = protocol["joint_objective"]
    if base_checkpoint_sha256 != pretrained["checkpoint_sha256"]:
        raise RuntimeError(
            "Frozen deployment base SHA differs from pretrained protocol SHA."
        )
    config = {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "objective_version": spec.objective_version,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "architecture": successor["architecture"],
        "embed_dim": int(protocol["model"]["embed_dim"]),
        "action_dim": int(action_block_dim),
        "history_size": int(protocol["sequence"]["history_frames"]),
        "hidden_dim": int(successor["hidden_dim"]),
        "gamma": float(successor["gamma"]),
        "feature_basis": successor["feature_basis"],
        "action_conditioning": successor["action_conditioning"],
        "bootstrap_action": successor["bootstrap_action"],
        "terminal_source": successor["terminal_source"],
        "goal_conditioning": successor["goal_conditioning"],
        "actor": successor["actor"],
        "reward": successor["reward"],
        "predicted_context_detach": objective["predicted_context_detach"],
        "target_world_ema_decay": float(successor["target_world_ema_decay"]),
        "target_successor_ema_decay": float(successor["target_successor_ema_decay"]),
        "planning_weight": float(successor["planning_weight"]),
        "terminal_weight": float(successor["terminal_weight"]),
        "clamp_successor_cost": bool(successor["clamp_successor_cost"]),
        "base_export_run_name": base_export_run_name,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "pretrained_world_model_frozen": True,
        "pretrained_world_model_source_method": pretrained["source_method"],
        "pretrained_world_model_source_seed": int(pretrained["source_seed"]),
        "pretrained_world_model_source_epoch": int(pretrained["source_epoch"]),
        "pretrained_world_model_sha256": pretrained["checkpoint_sha256"],
        "training_branches": successor["training_branches"],
        "world_model_gradient": objective["world_model_gradient"],
        "td_branches": objective["td_branches"],
        "goal_source": objective["goal_source"],
        "goal_validation": objective["goal_validation"],
        "goal_score": objective["goal_score"],
        "real_td_weight": float(objective["real_td_weight"]),
        "predicted_td_weight": float(objective["predicted_td_weight"]),
    }
    config.update(spec.successor_config_fields(protocol))
    return config


def _deployment_payload(
    module: Any,
    *,
    protocol: dict[str, Any],
    spec: FrozenActorFreeTDSpec,
    model_config: dict[str, Any],
    action_block_dim: int,
    epoch: int,
    global_step: int,
    base_export_run_name: str,
    base_checkpoint_sha256: str,
    initialization_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one self-contained and version-locked deployment checkpoint."""

    if initialization_info is None:
        raise RuntimeError("Frozen deployment is missing source provenance.")
    return {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "objective_version": spec.objective_version,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "world_model_state_dict": module.model.state_dict(),
        "target_world_model_state_dict": module.target_model.state_dict(),
        "world_model_config": model_config,
        "pretrained_world_model_provenance": copy.deepcopy(initialization_info),
        "successor_state_dict": module.successor.state_dict(),
        "target_successor_state_dict": module.target_successor.state_dict(),
        "successor_config": _successor_config(
            protocol,
            spec=spec,
            action_block_dim=action_block_dim,
            base_export_run_name=base_export_run_name,
            base_checkpoint_sha256=base_checkpoint_sha256,
        ),
    }


def _build_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
    spec: FrozenActorFreeTDSpec,
    action_block_dim: int,
    initialization_info: dict[str, Any] | None = None,
):
    import lightning as pl

    class FrozenActorFreeTDExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = int(trainer.current_epoch) + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            if initialization_info is None:
                raise RuntimeError("Frozen export is missing source provenance.")
            base_run_name = str(initialization_info["source_run_name"])
            base_checkpoint_sha256 = str(
                initialization_info["source_checkpoint_sha256"]
            )
            deployment_dir = run_dir / "checkpoints" / spec.method / spec.variant
            deployment_dir.mkdir(parents=True, exist_ok=True)
            payload = _deployment_payload(
                pl_module,
                protocol=protocol,
                spec=spec,
                model_config=model_config,
                action_block_dim=action_block_dim,
                epoch=epoch,
                global_step=int(trainer.global_step),
                base_export_run_name=base_run_name,
                base_checkpoint_sha256=base_checkpoint_sha256,
                initialization_info=initialization_info,
            )
            torch.save(payload, deployment_dir / f"epoch_{epoch:02d}.pt")

    return FrozenActorFreeTDExportCallback()


def _build_generator_callback(
    generator: torch.Generator,
    *,
    method: str,
    variant: str,
    goal_generator: torch.Generator | None = None,
):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return f"tdwm_{method}_{variant}_dataloader_generator"

        def state_dict(self) -> dict[str, Any]:
            state = {"generator_state": generator.get_state()}
            if goal_generator is not None:
                state["goal_generator_state"] = goal_generator.get_state()
            return state

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            generator.set_state(state_dict["generator_state"])
            if goal_generator is not None:
                if "goal_generator_state" not in state_dict:
                    raise RuntimeError(
                        "Goal-TD checkpoint is missing its goal sampler RNG state."
                    )
                goal_generator.set_state(state_dict["goal_generator_state"])

    return DataLoaderGeneratorCallback()


def _build_episode_epoch_callback(dataset: EpisodeStreamingBatchDataset):
    import lightning as pl

    class EpisodeStreamingEpochCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module) -> None:
            del pl_module
            dataset.set_epoch(int(trainer.current_epoch))

    return EpisodeStreamingEpochCallback()


def train_frozen_actor_free_td(
    *,
    spec: FrozenActorFreeTDSpec,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
    initial_world_model_checkpoint_path: str | Path | None = None,
    frozen_latent_store_path: str | Path | None = None,
    split_indices_path: str | Path | None = None,
    neighbor_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train one fixed standalone actor-free method from frozen Cube latents."""

    protocol = load_frozen_actor_free_td_training_protocol(
        protocol_path,
        spec=spec,
    )
    frozen_pretrained = True
    protocol_sha256 = _canonical_protocol_sha256(protocol)
    if initial_world_model_checkpoint_path is None:
        raise ValueError(
            "Frozen successor training requires one initial LeWM checkpoint."
        )
    if frozen_latent_store_path is None:
        raise ValueError("Frozen successor training requires --frozen-latent-store.")
    if split_indices_path is None:
        raise ValueError(
            "Frozen successor training requires an existing --split-indices artifact."
        )
    neighbor_variant = spec.requires_neighbor_index
    if neighbor_variant != (neighbor_index_path is not None):
        raise ValueError(
            "Neighbor-action variants require --neighbor-index; every other "
            "variant rejects it."
        )
    resolved_neighbor_index = None
    neighbor_index_info = None
    if neighbor_index_path is not None:
        resolved_neighbor_index = Path(neighbor_index_path).expanduser().resolve()
        if not resolved_neighbor_index.is_dir():
            raise FileNotFoundError(
                f"Neighbor index directory not found: {resolved_neighbor_index}"
            )
    resolved_frozen_latent_store = None
    if frozen_latent_store_path is not None:
        resolved_frozen_latent_store = (
            Path(frozen_latent_store_path).expanduser().resolve()
        )
        if not resolved_frozen_latent_store.is_dir():
            raise FileNotFoundError(
                "Frozen latent store directory not found: "
                f"{resolved_frozen_latent_store}"
            )
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if not smoke and max_steps is not None:
        raise ValueError("max_steps is smoke-only.")
    if not smoke and skip_validation:
        raise ValueError("skip_validation is smoke-only.")

    resolved_split_indices = Path(split_indices_path).expanduser().resolve()
    if not resolved_split_indices.is_file():
        raise FileNotFoundError(
            f"Split-indices artifact not found: {resolved_split_indices}"
        )

    dataset_path = Path(dataset_path).expanduser().resolve()
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    if (neighbor_variant or frozen_pretrained) and dataset_source["format"] != "lance":
        raise ValueError(
            "Frozen and neighbor-action variants require the audited Lance "
            "dataset with stable global row identities."
        )
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
    if frozen_pretrained:
        # Only episode/clip metadata are needed from Lance. Actual samples come
        # exclusively from FrozenLatentClipDataset below; do not cache a
        # second in-process copy of the complete action column.
        keys_to_load = ["action"]
        keys_to_cache = []
        keys_to_merge: dict[str, Any] = {}
        device_preprocessing = False
    else:
        keys_to_load = list(dataset_cfg["keys_to_load"])
        keys_to_cache = list(dataset_cfg["keys_to_cache"])
        keys_to_merge = dict(dataset_cfg["keys_to_merge"])
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=int(sequence["num_steps"]),
        frameskip=int(sequence["frame_skip"]),
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        keys_to_merge=keys_to_merge,
    )
    if len(dataset.lengths) != int(dataset_cfg["expected_episodes"]):
        raise ValueError("Dataset episode count differs from the protocol.")
    if int(np.asarray(dataset.lengths).sum()) != int(
        dataset_cfg["expected_transitions"]
    ):
        raise ValueError("Dataset transition count differs from the protocol.")

    frozen_latent_store = None
    frozen_latent_store_info = None
    decoded_frame_store_metadata = None
    if frozen_pretrained:
        action_dim = CUBE_ACTION_DIM
        assert resolved_frozen_latent_store is not None
        (
            frozen_latent_store,
            frozen_latent_store_info,
        ) = _resolve_bound_frozen_latent_store(
            resolved_frozen_latent_store,
            protocol=protocol,
            dataset_source=dataset_source,
            action_dim=action_dim,
        )
        dataset = FrozenLatentClipDataset(dataset, frozen_latent_store)
    else:
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
    if not frozen_pretrained and dataset_source["format"] == "lance":
        (
            decoded_frame_store,
            decoded_frame_store_metadata,
        ) = _prepare_decoded_frame_store(protocol, dataset_source, dataset)
        dataset = StrideAwareLanceDataset(
            dataset,
            decoded_frame_store=decoded_frame_store,
        )
    elif not frozen_pretrained and os.environ.get(DECODED_FRAME_STORE_ENV) is not None:
        raise ValueError(
            f"{DECODED_FRAME_STORE_ENV} is only supported for Lance datasets."
        )

    generator = torch.Generator().manual_seed(seed)
    goal_generator = None
    if protocol["variant"] in HINDSIGHT_GOAL_VARIANTS:
        goal_generator = torch.Generator().manual_seed(
            seed + int(protocol["joint_objective"]["goal_sampling_seed_offset"])
        )
    train_indices, validation_indices, split_manifest = load_bound_training_split(
        resolved_split_indices,
        dataset_size=len(dataset),
        train_fraction=float(protocol["split"]["train_fraction"]),
        validation_fraction=float(protocol["split"]["validation_fraction"]),
    )
    train_set = torch.utils.data.Subset(dataset, train_indices.tolist())
    validation_set = torch.utils.data.Subset(dataset, validation_indices.tolist())

    episode_train_dataset = None
    use_episode_streaming = (
        bool(loader_cfg["episode_streaming"]) and not smoke and not frozen_pretrained
    )
    if use_episode_streaming:
        if not isinstance(dataset, StrideAwareLanceDataset):
            raise ValueError("Episode streaming requires the audited Lance dataset.")
        episode_train_dataset = EpisodeStreamingBatchDataset(
            dataset,
            train_set.indices,
            batch_size=int(loader_cfg["batch_size"]),
            active_episodes=int(loader_cfg["episode_pool_size"]),
            read_episodes=int(loader_cfg["episode_read_size"]),
            cache_bytes=int(loader_cfg["episode_cache_bytes"]),
            prefetch_blocks=int(loader_cfg["episode_prefetch_blocks"]),
            seed=seed,
            drop_last=bool(loader_cfg["train_drop_last"]),
            min_unique_episodes=int(loader_cfg["minimum_unique_episodes_per_batch"]),
        )
        train_loader = torch.utils.data.DataLoader(
            episode_train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=bool(loader_cfg["pin_memory"]),
        )
    else:
        workers = 0 if smoke else int(loader_cfg["workers"])
        train_kwargs: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": bool(loader_cfg["pin_memory"]),
        }
        if workers:
            train_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": int(loader_cfg["prefetch_factor"]),
                }
            )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(loader_cfg["batch_size"]),
            shuffle=bool(loader_cfg["train_shuffle"]),
            drop_last=bool(loader_cfg["train_drop_last"]),
            generator=generator,
            **train_kwargs,
        )

    validation_workers = 0 if smoke else int(loader_cfg["validation_workers"])
    validation_kwargs: dict[str, Any] = {
        "num_workers": validation_workers,
        "pin_memory": bool(loader_cfg["pin_memory"]),
    }
    if validation_workers:
        validation_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(loader_cfg["prefetch_factor"]),
            }
        )
    if bool(loader_cfg["validation_locality"]):
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_sampler=BlockShuffleBatchSampler(
                validation_set.indices,
                batch_size=int(loader_cfg["batch_size"]),
                block_size=int(loader_cfg["block_size"]),
                drop_last=bool(loader_cfg["validation_drop_last"]),
                shuffle_batches_within_block=False,
                shuffle_blocks=False,
            ),
            **validation_kwargs,
        )
    else:
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=int(loader_cfg["batch_size"]),
            shuffle=bool(loader_cfg["validation_shuffle"]),
            drop_last=bool(loader_cfg["validation_drop_last"]),
            **validation_kwargs,
        )

    if not frozen_pretrained:
        action_dim = int(dataset.get_dim("action"))
    action_block_dim = int(sequence["frame_skip"]) * action_dim
    neighbor_index = None
    if resolved_neighbor_index is not None:
        neighbor_index = StateNeighborActionIndex(
            resolved_neighbor_index,
            expected_checkpoint_sha256=protocol["pretrained_world_model"][
                "checkpoint_sha256"
            ],
            expected_latent_store_manifest_sha256=frozen_latent_store_info[
                "manifest_sha256"
            ],
            expected_action_block_dim=action_block_dim,
            expected_k=int(protocol["joint_objective"]["neighbors_per_anchor"]),
        )
        index_split_hash = neighbor_index.manifest.get("training_split_sha256")
        if index_split_hash != split_manifest["train_indices_sha256"]:
            raise ValueError(
                "The state-neighbor index was built for a different training split."
            )
        neighbor_index_info = {
            "path": str(neighbor_index.root),
            "manifest_path": str(neighbor_index.manifest_path),
            "manifest_sha256": _file_sha256(neighbor_index.manifest_path),
            "manifest": neighbor_index.manifest,
            "validation_objective": {
                "neighbor_objective_available": False,
                "reported_loss": "unweighted_real_vector_td",
                "reason": (
                    "The immutable candidate graph contains training anchors only; "
                    "validation rows are neither inserted nor searched online."
                ),
            },
        }
    model_config = build_model_config(protocol, action_dim)
    initialization_info = None
    if frozen_pretrained:
        assert initial_world_model_checkpoint_path is not None
        source_name, source_file, source_cache = _resolve_local_pretrained_lewm_export(
            initial_world_model_checkpoint_path
        )
        source_hash = _file_sha256(source_file)
        expected_hash = protocol["pretrained_world_model"]["checkpoint_sha256"]
        if source_hash != expected_hash:
            raise ValueError(
                "The pretrained LeWM checkpoint SHA-256 differs from protocol."
            )
        assert frozen_latent_store_info is not None
        if frozen_latent_store_info["pretrained_checkpoint_sha256"] != source_hash:
            raise ValueError(
                "Frozen latent store was encoded by a different LeWM checkpoint."
            )
        source_training_provenance = _verify_completed_pretrained_lewm_run(
            source_run_name=source_name,
            source_cache=source_cache,
            expected_seed=int(protocol["pretrained_world_model"]["source_seed"]),
            expected_epoch=int(protocol["pretrained_world_model"]["source_epoch"]),
        )
        world_model = swm.wm.load_pretrained(
            source_name,
            cache_dir=str(source_cache),
        )
        world_model.requires_grad_(False)
        world_model.eval()
        initialization_info = {
            "strategy": "frozen_pretrained_lewm",
            "source_method": "lewm",
            "source_seed": int(protocol["pretrained_world_model"]["source_seed"]),
            "source_epoch": int(protocol["pretrained_world_model"]["source_epoch"]),
            "source_run_name": source_name,
            "source_checkpoint_path": str(source_file),
            "source_checkpoint_sha256": source_hash,
            "frozen": True,
            **source_training_provenance,
        }
    else:
        world_model = hydra.utils.instantiate(model_config)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != int(expected_parameters):
        raise ValueError(
            f"Expected {expected_parameters} LeWM parameters, found {parameter_count}."
        )
    if protocol["training"]["model_compile"]:
        compile_world_model(
            world_model, mode=str(protocol["training"]["model_compile_mode"])
        )

    available_epoch_steps = len(train_loader)
    formal_epoch_steps = int(protocol["training"]["optimizer_steps_per_epoch"])
    if formal_epoch_steps > available_epoch_steps:
        raise ValueError("optimizer_steps_per_epoch exceeds available batches.")
    train_limit = resolve_train_batch_limit(
        smoke=smoke,
        max_steps=max_steps,
        train_loader_length=available_epoch_steps,
    )
    if not smoke and max_steps is None:
        train_limit = formal_epoch_steps
    schedule = resolve_actor_free_training_schedule(
        protocol,
        smoke=smoke,
        resume=resume,
        max_steps=max_steps,
        train_limit=train_limit,
    )
    module = _build_training_module(
        world_model,
        protocol,
        schedule.total_scheduler_steps,
        spec=spec,
        action_block_dim=action_block_dim,
        device_image_preprocessing=device_preprocessing,
        goal_generator=goal_generator,
        neighbor_index=neighbor_index,
    )

    checkpoint_dir = run_dir / "checkpoints" / "lightning"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=int(protocol["training"]["checkpoint_every_epochs"]),
        save_last=True,
        save_top_k=-1,
    )
    callbacks = [
        checkpoint_callback,
        _build_export_callback(
            run_dir,
            model_config,
            protocol,
            spec,
            action_block_dim,
            initialization_info=initialization_info,
        ),
        _build_generator_callback(
            generator,
            method=spec.method,
            variant=str(protocol["variant"]),
            goal_generator=goal_generator,
        ),
    ]
    if episode_train_dataset is not None:
        callbacks.append(_build_episode_epoch_callback(episode_train_dataset))

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {last_checkpoint}")
    checkpoint_path = None
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify objective metadata for resume.")
        with manifest_path.open() as stream:
            previous = json.load(stream)
        previous_protocol = previous.get("protocol", {})
        compatible = (
            previous_protocol.get("method") == spec.method
            and previous_protocol.get("method_family") == METHOD_FAMILY
            and previous_protocol.get("variant") == spec.variant
            and previous_protocol.get("successor", {}).get("objective_version")
            == spec.objective_version
            and previous.get("deployment_checkpoint_version")
            == DEPLOYMENT_CHECKPOINT_VERSION
        )
        if frozen_pretrained:
            assert protocol_sha256 is not None
            assert initialization_info is not None
            assert frozen_latent_store_info is not None
            compatible = (
                compatible
                and previous.get("protocol_sha256") == protocol_sha256
                and previous.get("seed") == seed
                and previous.get("dataset", {})
                .get("split", {})
                .get("train_indices_sha256")
                == split_manifest["train_indices_sha256"]
                and previous.get("dataset", {})
                .get("split", {})
                .get("validation_indices_sha256")
                == split_manifest["validation_indices_sha256"]
            )
            previous_initialization = previous.get("model", {}).get(
                "initialization", {}
            )
            compatible = compatible and all(
                previous_initialization.get(key) == initialization_info.get(key)
                for key in (
                    "source_checkpoint_sha256",
                    "source_training_result_sha256",
                    "source_training_manifest_sha256",
                    "source_final_epoch",
                    "source_global_step",
                )
            )
            previous_store = previous.get("frozen_latent_store", {})
            compatible = compatible and all(
                previous_store.get(key) == frozen_latent_store_info.get(key)
                for key in (
                    "manifest_sha256",
                    "pretrained_checkpoint_sha256",
                    "dataset_source_sha256",
                    "dataset_manifest_sha256",
                    "column_normalization_sha256",
                    "input_file_sha256",
                )
            )
        if neighbor_index_info is not None:
            compatible = (
                compatible
                and previous.get("neighbor_index", {}).get("manifest_sha256")
                == neighbor_index_info["manifest_sha256"]
            )
        if not compatible:
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
    dataset_manifest = {
        **dataset_source,
        "sequence_samples": len(dataset),
        "split": split_manifest,
    }
    if decoded_frame_store_metadata is not None:
        dataset_manifest["decoded_frame_store"] = decoded_frame_store_metadata
    training_manifest = {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "objective_version": spec.objective_version,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "seed": seed,
        "dataset": dataset_manifest,
        "model": {
            "config": model_config,
            "lewm_parameters": parameter_count,
            "successor_parameters": sum(
                parameter.numel() for parameter in module._online_head().parameters()
            ),
            "action_block_dim": action_block_dim,
        },
        "training": {
            "formal_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
            "optimizer_steps_per_epoch": formal_epoch_steps,
            "available_batches_per_epoch": available_epoch_steps,
            "configured_optimizer_steps": schedule.total_scheduler_steps,
            "resume_mode": resume,
            "resumed_from": checkpoint_path,
            "episode_streaming": use_episode_streaming,
            "validation_batches": len(validation_loader),
            "validation_skipped": smoke or skip_validation,
        },
        "runtime": runtime,
    }
    if frozen_pretrained:
        assert protocol_sha256 is not None
        training_manifest["protocol_sha256"] = protocol_sha256
        training_manifest["model"].update(
            {
                "initialization": initialization_info,
                "trainable_lewm_parameters": sum(
                    parameter.numel()
                    for parameter in module.model.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        training_manifest["frozen_latent_store"] = frozen_latent_store_info
        training_manifest["neighbor_index"] = neighbor_index_info
        training_manifest["training"].update(
            {
                "data_source": "frozen_latent_store",
                "image_reads_during_training": False,
                "world_model_encode_during_training": False,
            }
        )
    write_json(run_dir / "training_manifest.json", training_manifest)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = protocol["training"]["precision"] if accelerator == "gpu" else "32-true"
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
            max_epochs=schedule.max_epochs,
            gradient_clip_val=float(protocol["training"]["gradient_clip_norm"]),
            limit_train_batches=train_limit,
            limit_val_batches=0.0 if smoke or skip_validation else 1.0,
            num_sanity_val_steps=0,
            logger=build_metrics_logger(run_dir, protocol["logging"]),
            callbacks=callbacks,
            log_every_n_steps=1 if smoke else 50,
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
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "run_dir": str(run_dir),
        "seed": seed,
        "last_checkpoint": str(last_checkpoint),
        "final_epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
    }
    if frozen_pretrained:
        assert protocol_sha256 is not None
        assert initialization_info is not None
        assert frozen_latent_store_info is not None
        result["protocol_sha256"] = protocol_sha256
        result["pretrained_world_model_sha256"] = initialization_info[
            "source_checkpoint_sha256"
        ]
        result["frozen_latent_store_manifest_sha256"] = frozen_latent_store_info[
            "manifest_sha256"
        ]
    if neighbor_index_info is not None:
        result["neighbor_index_manifest_sha256"] = neighbor_index_info[
            "manifest_sha256"
        ]
    if torch.cuda.is_available():
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "ActorFreeTDInputs",
    "ActorFreeTrainingSchedule",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_OPTIMIZER_UPDATES",
    "FrozenActorFreeTDSpec",
    "FrozenTDContext",
    "FrozenTDLoss",
    "FROZEN_OBJECTIVE_VERSIONS",
    "FROZEN_PRETRAINED_VARIANTS",
    "GOAL_PROJECTED_VARIANT",
    "GOAL_VALUE_WEIGHTED_VARIANT",
    "METHOD_FAMILY",
    "NEIGHBOR_ACTION_ADVANTAGE_VARIANT",
    "PREFIX_MARGINAL_ACTION_ADVANTAGE_VARIANT",
    "PREFIX_MEAN_ACTION_ADVANTAGE_VARIANT",
    "SAME_FUTURE_GOAL_ADVANTAGE_VARIANT",
    "_canonical_protocol_sha256",
    "_build_training_module",
    "_resolve_local_pretrained_lewm_export",
    "_verify_completed_pretrained_lewm_run",
    "frozen_weight_metrics",
    "load_bound_training_split",
    "load_frozen_actor_free_td_training_protocol",
    "resolve_actor_free_training_schedule",
    "train_frozen_actor_free_td",
    "validate_frozen_actor_free_td_training_protocol",
]
