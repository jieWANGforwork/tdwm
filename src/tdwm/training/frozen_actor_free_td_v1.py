"""Standalone frozen-LeWM training runtime for Actor-Free TD-JEPA V1.

V1 is deliberately separate from the V-1 frozen successor runtime.  The
pretrained LeWM encoder/predictor remains frozen and supplies audited 192D
latents plus normalized 25D raw action blocks.  Only the online, goal-
conditioned TD-JEPA predictor is optimized; its target copy is updated by EMA.
"""

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
from tdwm.methods.action_prefix_advantage_common import (
    build_zero_mean_action_prefixes,
)
from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_ACTION_DIM,
    V1_ACTION_EMBEDDING_DIM,
    V1_OUTPUT_DIM,
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    V1_TASK_DIM,
    ActorFreeTDJEPAPredictorV1,
    build_tdjepa_td_batch_v1,
    ema_update_target_v1,
    encode_frozen_action_blocks_v1,
    sample_mixed_tasks_v1,
    tdjepa_goal_score_v1,
    validate_frozen_lewm_action_encoder_v1,
)
from tdwm.methods.actor_free_td_lewm_v1_objectives import (
    OBJECTIVE_VERSION,
    goal_projected_v1_loss,
    goal_value_weighted_v1_loss,
    neighbor_action_advantage_v1_loss,
    prefix_marginal_advantage_v1_loss,
    prefix_mean_advantage_v1_loss,
    same_future_goal_advantage_v1_loss,
)
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.frozen_actor_free_td import (
    FORMAL_OPTIMIZER_UPDATES,
    _file_sha256,
    _resolve_bound_frozen_latent_store,
    _resolve_local_pretrained_lewm_export,
    _verify_completed_pretrained_lewm_run,
    load_bound_training_split,
    resolve_actor_free_training_schedule,
)
from tdwm.training.frozen_actor_free_td_v1_data import (
    FrozenActorFreeTDV1TransitionDataset,
    sample_reachable_future_latents_v1,
)
from tdwm.training.frozen_latent_store import (
    CUBE_ACTION_DIM,
    FrozenLatentClipDataset,
)
from tdwm.training.gt_lewm_support import (
    build_metrics_logger,
    build_model_config,
    resolve_train_batch_limit,
    write_json,
)
from tdwm.training.lewm import _git_revision
from tdwm.training.state_neighbor_index import StateNeighborActionIndex

METHOD_FAMILY = "actor_free_td_lewm_v1"
IMPLEMENTATION_VERSION = "v1"
DEPLOYMENT_CHECKPOINT_VERSION = 1
SUPPORTED_VARIANTS = frozenset({"c", "d", "f", "g1", "g2", "g3"})


@dataclass(frozen=True)
class ActorFreeTDLeWMV1Spec:
    """Identity of one independently runnable V1 method."""

    method: str
    variant: str
    requires_neighbor_index: bool = False

    def __post_init__(self) -> None:
        if self.method != f"{METHOD_FAMILY}_{self.variant}":
            raise ValueError("V1 method names must end in their exact variant.")
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported V1 variant: {self.variant!r}.")
        if self.requires_neighbor_index != (self.variant == "g1"):
            raise ValueError("Only V1 G1 uses the training neighbor index.")


V1_SPECS = {
    variant: ActorFreeTDLeWMV1Spec(
        method=f"{METHOD_FAMILY}_{variant}",
        variant=variant,
        requires_neighbor_index=variant == "g1",
    )
    for variant in sorted(SUPPORTED_VARIANTS)
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cuda_runtime_provenance() -> tuple[torch.device | None, dict[str, str]]:
    """Return the exact CUDA device used by one future V1 training run."""

    if not torch.cuda.is_available():
        return None, {}
    device = torch.device("cuda", torch.cuda.current_device())
    return device, {"cuda_device": torch.cuda.get_device_name(device)}


def _reset_peak_cuda_memory(device: torch.device | None) -> None:
    """Start peak-allocation accounting without affecting CPU runs."""

    if device is not None:
        torch.cuda.reset_peak_memory_stats(device)


def _record_peak_cuda_memory(
    result: dict[str, Any],
    device: torch.device | None,
) -> None:
    """Attach the observed CUDA allocation peak using the project schema."""

    if device is not None:
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))


def _validate_v1_resume_manifest(
    previous: dict[str, Any],
    *,
    spec: ActorFreeTDLeWMV1Spec,
    protocol_sha256: str,
    seed: int,
    split_manifest: dict[str, Any],
    initialization_info: dict[str, Any],
    frozen_latent_store_info: dict[str, Any],
    neighbor_index_info: dict[str, Any] | None,
) -> None:
    """Reject resume checkpoints that are not bound to this exact V1 run."""

    compatible = (
        previous.get("method") == spec.method
        and previous.get("method_family") == METHOD_FAMILY
        and previous.get("variant") == spec.variant
        and previous.get("implementation_version") == IMPLEMENTATION_VERSION
        and previous.get("objective_version") == OBJECTIVE_VERSION
        and previous.get("deployment_checkpoint_version")
        == DEPLOYMENT_CHECKPOINT_VERSION
        and previous.get("protocol_sha256") == protocol_sha256
        and previous.get("seed") == seed
        and previous.get("dataset", {}).get("split", {}).get("train_indices_sha256")
        == split_manifest.get("train_indices_sha256")
        and previous.get("dataset", {})
        .get("split", {})
        .get("validation_indices_sha256")
        == split_manifest.get("validation_indices_sha256")
    )
    previous_initialization = previous.get("model", {}).get("initialization", {})
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
    previous_neighbor = previous.get("neighbor_index")
    if neighbor_index_info is None:
        compatible = compatible and previous_neighbor is None
    else:
        compatible = (
            compatible
            and isinstance(previous_neighbor, dict)
            and previous_neighbor.get("manifest_sha256")
            == neighbor_index_info.get("manifest_sha256")
        )
    if not compatible:
        raise RuntimeError("Refusing to resume an incompatible V1 run.")


def load_actor_free_td_lewm_v1_training_protocol(
    path: str | Path,
    *,
    spec: ActorFreeTDLeWMV1Spec,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict):
        raise ValueError("V1 training protocol must contain a mapping.")
    validate_actor_free_td_lewm_v1_training_protocol(protocol, spec=spec)
    return protocol


def _positive_float(mapping: dict[str, Any], key: str, *, section: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{section}.{key} must be finite and positive.") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{section}.{key} must be finite and positive.")
    return value


def validate_actor_free_td_lewm_v1_training_protocol(
    protocol: dict[str, Any],
    *,
    spec: ActorFreeTDLeWMV1Spec,
) -> None:
    """Validate the V1 architecture and experiment boundary without V-1 locks."""

    exact = {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION_VERSION,
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "environment": "cube",
        "stage": "full_training",
        "initialization": "frozen_pretrained_lewm",
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol.{key} must be {expected!r}.")
    if protocol.get("seeds") != [0, 42, 3072]:
        raise ValueError("V1 uses the comparison seeds [0, 42, 3072].")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("V1 requires stable-worldmodel 0.1.1.")

    pretrained = protocol.get("pretrained_world_model", {})
    for key, expected in {
        "source_method": "lewm",
        "source_seed": 3072,
        "source_epoch": 10,
        "frozen": True,
    }.items():
        if pretrained.get(key) != expected:
            raise ValueError(f"pretrained_world_model.{key} must be {expected!r}.")
    source_hash = pretrained.get("checkpoint_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("pretrained checkpoint_sha256 must be lowercase SHA-256.")

    sequence = protocol.get("sequence", {})
    if (
        int(sequence.get("frame_skip", 0)) != 5
        or int(sequence.get("history_frames", 0)) != 3
        or int(sequence.get("num_steps", 0)) <= 4
        or int(sequence.get("prediction_frames", 0)) != 1
    ):
        raise ValueError("V1 Cube alignment requires frame_skip=5 and data offset 3.")
    context = protocol.get("context", {})
    for key, expected in {
        "g_state_frames": 1,
        "lewm_rollout_history_frames": 3,
    }.items():
        if context.get(key) != expected:
            raise ValueError(f"context.{key} must be {expected!r}.")
    if int(protocol.get("model", {}).get("embed_dim", 0)) != V1_STATE_DIM:
        raise ValueError("V1 uses the baseline LeWM 192D latent.")

    predictor = protocol.get("predictor", {})
    locked_predictor = {
        "objective_version": OBJECTIVE_VERSION,
        "architecture": "td_jepa_forward_map_v1",
        "state_dim": V1_STATE_DIM,
        "raw_action_dim": V1_RAW_ACTION_DIM,
        "action_dim": V1_ACTION_DIM,
        "action_embedding_dim": V1_ACTION_EMBEDDING_DIM,
        "task_dim": V1_TASK_DIM,
        "output_dim": V1_OUTPUT_DIM,
        "hidden_dim": 256,
        "hidden_layers": 1,
        "embedding_layers": 2,
        "num_parallel": 1,
        "action_processing": "frozen_shared_lewm_action_encoder",
        "shared_lewm_action_encoder": True,
        "action_encoder_trainable": False,
        "action_encoder_source": "world_model.action_encoder",
        "state_parameterization": "symmetric_shared_frozen_lewm_latent",
        "goal_conditioning": "task_input",
        "bootstrap_action": "dataset_next_action",
        "actor": "none",
        "reward": "none",
        "loss_warmup_fraction": 0.0,
    }
    for key, expected in locked_predictor.items():
        if predictor.get(key) != expected:
            raise ValueError(f"predictor.{key} must be {expected!r}.")
    gamma = float(predictor.get("gamma", -1.0))
    decay = float(predictor.get("target_ema_decay", -1.0))
    if not 0.0 <= gamma < 1.0 or not 0.0 <= decay < 1.0:
        raise ValueError("predictor gamma and target_ema_decay must lie in [0,1).")

    tasks = protocol.get("task_sampling", {})
    task_lock = {
        "goal_probability": 0.5,
        "sampling": "per_transition_bernoulli",
        "random_source": "isotropic_gaussian_sphere",
        "goal_source": "uniform_reachable_future_frozen_latent_same_clip",
        "normalization": "sqrt_dim_l2_sphere",
        "mix_unit": "transition_minibatch",
    }
    for key, expected in task_lock.items():
        if tasks.get(key) != expected:
            raise ValueError(f"task_sampling.{key} must be {expected!r}.")

    objective = protocol.get("joint_objective", {})
    objective_lock = {
        "base_td_population": "all_transitions",
        "random_task_weight": 1.0,
        "goal_subset": "goal_derived_tasks_only",
        "final_weight_normalization": "mean_one_over_all_transitions",
        "weight_gradient": "stop_gradient",
        "candidate_td_targets": "none",
    }
    for key, expected in objective_lock.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    variant_objective_locks = {
        "c": {
            "objective": "goal_projected_td",
            "goal_signal": "matched_future_latent",
            "goal_projection_target": "detached_td_target_projection",
            "goal_projection_prediction_gradient": "online_predictor",
            "projection_population": "goal_derived_tasks_only",
        },
        "d": {
            "objective": "goal_value_weighted_td",
            "score_source": "detached_td_target",
            "weight_clip": None,
            "goal_subset_weighting": "softmax_mean_one",
        },
        "f": {
            "objective": "same_future_different_goal_advantage",
            "score_source": "detached_td_target",
            "baseline": "all_goal_derived_tasks_in_batch",
            "positive": "matching_transition_goal",
            "weight_clip": None,
            "goal_subset_weighting": "softmax_mean_one",
        },
        "g1": {
            "objective": "neighbor_action_advantage",
            "score_source": "detached_online_predictor",
            "candidate_source": ("other_episode_frozen_latent_knn_real_action_blocks"),
            "goal_subset_weighting": "softmax_mean_one",
        },
        "g2": {
            "objective": "prefix_mean_advantage",
            "score_source": "detached_online_predictor",
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "advantage_reducer": "full_score_minus_all_prefix_mean",
            "goal_subset_weighting": "softmax_mean_one",
        },
        "g3": {
            "objective": "prefix_marginal_advantage",
            "score_source": "detached_online_predictor",
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "advantage_reducer": "mean_adjacent_prefix_score_deltas",
            "goal_subset_weighting": "softmax_mean_one",
        },
    }
    for key, expected in variant_objective_locks[spec.variant].items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")
    if spec.variant == "c":
        _positive_float(objective, "goal_projection_weight", section="joint_objective")
    else:
        _positive_float(objective, "weight_temperature", section="joint_objective")
    if spec.variant == "g1":
        _positive_float(objective, "neighbor_temperature", section="joint_objective")
        if int(objective.get("neighbors_per_anchor", 0)) <= 0:
            raise ValueError("G1 neighbors_per_anchor must be positive.")
    if spec.variant in {"g2", "g3"}:
        for key, expected in {
            "prefix_slots": 5,
            "suffix_fill": "normalized_zero_mean_action",
        }.items():
            if objective.get(key) != expected:
                raise ValueError(f"joint_objective.{key} must be {expected!r}.")

    loader = protocol.get("loader", {})
    loader_lock = {
        "batch_size": 256,
        "sampling_unit": "transition",
        "transition_population": ("unique_legal_td_rows_from_exact_clip_split"),
        "train_sampling": "random_with_replacement",
        "validation_sampling": "sequential_without_replacement",
        "frozen_latent_mmap": True,
        "train_drop_last": True,
        "validation_drop_last": False,
    }
    for key, expected in loader_lock.items():
        if loader.get(key) != expected:
            raise ValueError(f"loader.{key} must be {expected!r}.")
    if int(loader.get("workers", -1)) < 0:
        raise ValueError("loader batch_size/workers are invalid.")
    training = protocol.get("training", {})
    training_lock = {
        "epochs": 10,
        "scheduler_epochs": 10,
        "optimizer_steps_per_epoch": 12_796,
        "precision": "bf16-mixed",
        "model_compile": False,
        "gradient_clip_norm": 1.0,
        "checkpoint_every_epochs": 1,
        "resume": True,
    }
    for key, expected in training_lock.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}.")
    if training["epochs"] * training["optimizer_steps_per_epoch"] != (
        FORMAL_OPTIMIZER_UPDATES
    ):
        raise ValueError("V1 formal runs require exactly 127960 optimizer updates.")
    optimizer = protocol.get("optimizer", {})
    if optimizer.get("type") != "AdamW":
        raise ValueError("optimizer.type must be 'AdamW'.")
    if float(optimizer.get("world_model_learning_rate", -1.0)) != 0.0:
        raise ValueError("The V1 LeWM world model is frozen.")
    _positive_float(optimizer, "predictor_learning_rate", section="optimizer")
    try:
        weight_decay = float(optimizer["weight_decay"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "optimizer.weight_decay must be finite and non-negative."
        ) from error
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be finite and non-negative.")
    scheduler = protocol.get("scheduler", {})
    for key, expected in {
        "type": "linear_warmup_cosine_annealing",
        "interval": "optimizer_step",
    }.items():
        if scheduler.get(key) != expected:
            raise ValueError(f"scheduler.{key} must be {expected!r}.")
    try:
        warmup_fraction = float(scheduler["warmup_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scheduler.warmup_fraction must lie in [0,1).") from error
    if not math.isfinite(warmup_fraction) or not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("scheduler.warmup_fraction must lie in [0,1).")


def _mean_or_zero(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.mean() if value.numel() else reference.new_zeros(())


def _build_v1_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    spec: ActorFreeTDLeWMV1Spec,
    data_generator: torch.Generator,
    goal_generator: torch.Generator,
    task_generator: torch.Generator,
    validation_goal_generator: torch.Generator | None = None,
    validation_task_generator: torch.Generator | None = None,
    neighbor_index: StateNeighborActionIndex | None,
    latent_store: Any | None = None,
):
    import lightning as pl

    class ActorFreeTDLeWMV1TrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model.requires_grad_(False)
            self.model.eval()
            if not isinstance(
                getattr(self.model, "action_encoder", None), torch.nn.Module
            ):
                raise ValueError("V1 requires world_model.action_encoder.")
            validate_frozen_lewm_action_encoder_v1(self.model.action_encoder)
            cfg = protocol["predictor"]
            self.predictor = ActorFreeTDJEPAPredictorV1(
                hidden_dim=int(cfg["hidden_dim"]),
                hidden_layers=int(cfg["hidden_layers"]),
                embedding_layers=int(cfg["embedding_layers"]),
            )
            self.target_predictor = self.predictor.make_target()
            self.variant = spec.variant
            self.data_generator = data_generator
            self.goal_generator = goal_generator
            self.task_generator = task_generator
            self.validation_goal_generator = (
                validation_goal_generator
                if validation_goal_generator is not None
                else torch.Generator().manual_seed(0)
            )
            self.validation_task_generator = (
                validation_task_generator
                if validation_task_generator is not None
                else torch.Generator().manual_seed(0)
            )
            self._validation_goal_epoch_state = (
                self.validation_goal_generator.get_state().clone()
            )
            self._validation_task_epoch_state = (
                self.validation_task_generator.get_state().clone()
            )
            self.neighbor_index = neighbor_index
            self.latent_store = latent_store
            if spec.requires_neighbor_index != (neighbor_index is not None):
                raise ValueError("Only V1 G1 accepts a neighbor index.")
            self.gamma = float(cfg["gamma"])
            self.target_ema_decay = float(cfg["target_ema_decay"])

        def train(self, mode: bool = True):
            super().train(mode)
            self.model.eval()
            self.target_predictor.eval()
            return self

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            checkpoint["v1_data_generator_state"] = self.data_generator.get_state()
            checkpoint["v1_goal_generator_state"] = self.goal_generator.get_state()
            checkpoint["v1_task_generator_state"] = self.task_generator.get_state()
            checkpoint["v1_validation_goal_generator_state"] = (
                self.validation_goal_generator.get_state()
            )
            checkpoint["v1_validation_task_generator_state"] = (
                self.validation_task_generator.get_state()
            )
            checkpoint["v1_validation_goal_epoch_state"] = (
                self._validation_goal_epoch_state.clone()
            )
            checkpoint["v1_validation_task_epoch_state"] = (
                self._validation_task_epoch_state.clone()
            )

        def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            data_state = checkpoint.get("v1_data_generator_state")
            goal_state = checkpoint.get("v1_goal_generator_state")
            task_state = checkpoint.get("v1_task_generator_state")
            validation_goal_state = checkpoint.get("v1_validation_goal_generator_state")
            validation_task_state = checkpoint.get("v1_validation_task_generator_state")
            validation_goal_epoch_state = checkpoint.get(
                "v1_validation_goal_epoch_state"
            )
            validation_task_epoch_state = checkpoint.get(
                "v1_validation_task_epoch_state"
            )
            if (
                data_state is None
                or goal_state is None
                or task_state is None
                or validation_goal_state is None
                or validation_task_state is None
                or validation_goal_epoch_state is None
                or validation_task_epoch_state is None
            ):
                raise RuntimeError("V1 resume checkpoint is missing RNG state.")
            self.data_generator.set_state(data_state.cpu())
            self.goal_generator.set_state(goal_state.cpu())
            self.task_generator.set_state(task_state.cpu())
            self.validation_goal_generator.set_state(validation_goal_state.cpu())
            self.validation_task_generator.set_state(validation_task_state.cpu())
            self._validation_goal_epoch_state = (
                validation_goal_epoch_state.cpu().clone()
            )
            self._validation_task_epoch_state = (
                validation_task_epoch_state.cpu().clone()
            )

        def on_validation_epoch_start(self) -> None:
            # Validation follows the configured uniform-future/Bernoulli task
            # distributions, but evaluates exactly the same sampled population
            # at every epoch so curves do not include advancing-RNG noise.
            self.validation_goal_generator.set_state(
                self._validation_goal_epoch_state.clone()
            )
            self.validation_task_generator.set_state(
                self._validation_task_epoch_state.clone()
            )

        def _score_neighbors(
            self,
            state: torch.Tensor,
            raw_action: torch.Tensor,
            action_embedding: torch.Tensor,
            task: torch.Tensor,
            global_rows: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.neighbor_index is None:
                raise RuntimeError("V1 G1 is missing its neighbor index.")
            neighbors = self.neighbor_index.lookup(
                global_rows,
                device=raw_action.device,
                dtype=raw_action.dtype,
            )
            with torch.no_grad():
                positive = tdjepa_goal_score_v1(
                    self.predictor(state, action_embedding, task), task
                )
                count, candidates = neighbors.actions.shape[:2]
                neighbor_state = state.unsqueeze(1).expand(-1, candidates, -1)
                neighbor_task = task.unsqueeze(1).expand(-1, candidates, -1)
                neighbor_action_embedding = encode_frozen_action_blocks_v1(
                    self.model.action_encoder,
                    neighbors.actions,
                    reference=neighbor_state,
                )
                neighbor_scores = tdjepa_goal_score_v1(
                    self.predictor(
                        neighbor_state,
                        neighbor_action_embedding,
                        neighbor_task,
                    ),
                    neighbor_task,
                )
            if positive.shape != (count,):
                raise RuntimeError("V1 G1 positive score alignment failed.")
            return positive, neighbor_scores, neighbors.distances

        def _score_prefixes(
            self,
            state: torch.Tensor,
            raw_action: torch.Tensor,
            task: torch.Tensor,
        ) -> torch.Tensor:
            prefixes = build_zero_mean_action_prefixes(raw_action.detach())
            prefix_state = state.unsqueeze(1).expand(-1, 5, -1)
            prefix_task = task.unsqueeze(1).expand(-1, 5, -1)
            with torch.no_grad():
                prefix_action_embedding = encode_frozen_action_blocks_v1(
                    self.model.action_encoder,
                    prefixes,
                    reference=prefix_state,
                )
                return tdjepa_goal_score_v1(
                    self.predictor(
                        prefix_state,
                        prefix_action_embedding,
                        prefix_task,
                    ),
                    prefix_task,
                )

        def _method_loss(
            self,
            td_batch: Any,
            state: torch.Tensor,
            raw_action: torch.Tensor,
            action_embedding: torch.Tensor,
            task: torch.Tensor,
            goal_mask: torch.Tensor,
            global_rows: torch.Tensor,
            *,
            stage: str,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            objective = protocol["joint_objective"]
            per_td = td_batch.per_transition_td_loss
            if self.variant == "c":
                output = goal_projected_v1_loss(
                    td_batch.prediction,
                    td_batch.target,
                    task,
                    goal_mask,
                    per_td,
                    projection_coefficient=float(objective["goal_projection_weight"]),
                )
                metrics = {
                    f"{stage}/goal_projection_loss": output.projection_loss.detach(),
                    f"{stage}/goal_score_residual_mean": _mean_or_zero(
                        output.score_residual.index_select(0, output.goal_indices),
                        output.loss,
                    ).detach(),
                }
                return output.loss, metrics
            if self.variant == "d":
                output = goal_value_weighted_v1_loss(
                    td_batch.target,
                    task,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
                signal = output.target_score.index_select(0, output.goal_indices)
                metrics = {
                    f"{stage}/target_goal_score_mean": _mean_or_zero(
                        signal, output.loss
                    ),
                    f"{stage}/weight_mean": output.weights.mean(),
                    f"{stage}/weight_std": output.weights.std(unbiased=False),
                }
                return output.loss, metrics
            if self.variant == "f":
                output = same_future_goal_advantage_v1_loss(
                    td_batch.target,
                    task,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
                metrics = {
                    f"{stage}/same_future_advantage_mean": _mean_or_zero(
                        output.advantage, output.loss
                    ),
                    f"{stage}/weight_mean": output.weights.mean(),
                    f"{stage}/weight_std": output.weights.std(unbiased=False),
                }
                return output.loss, metrics
            if self.variant == "g1":
                if stage != "train":
                    return td_batch.td_loss, {
                        f"{stage}/neighbor_objective_available": per_td.new_zeros(())
                    }
                positive, neighbor_scores, distances = self._score_neighbors(
                    state,
                    raw_action,
                    action_embedding,
                    task,
                    global_rows,
                )
                output = neighbor_action_advantage_v1_loss(
                    positive,
                    neighbor_scores,
                    distances,
                    goal_mask,
                    per_td,
                    neighbor_temperature=float(objective["neighbor_temperature"]),
                    weight_temperature=float(objective["weight_temperature"]),
                )
                metrics = {
                    f"{stage}/neighbor_objective_available": per_td.new_ones(()),
                    f"{stage}/neighbor_advantage_mean": _mean_or_zero(
                        output.advantage, output.loss
                    ),
                    f"{stage}/weight_mean": output.weights.mean(),
                }
                return output.loss, metrics
            prefix_scores = self._score_prefixes(state, raw_action, task)
            if self.variant == "g2":
                output = prefix_mean_advantage_v1_loss(
                    prefix_scores,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
                metric_name = "prefix_mean_advantage_mean"
            else:
                output = prefix_marginal_advantage_v1_loss(
                    prefix_scores,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
                metric_name = "prefix_marginal_advantage_mean"
            return output.loss, {
                f"{stage}/{metric_name}": _mean_or_zero(output.advantage, output.loss),
                f"{stage}/weight_mean": output.weights.mean(),
                f"{stage}/weight_std": output.weights.std(unbiased=False),
            }

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            names_and_dims = {
                "state": V1_STATE_DIM,
                "action": V1_RAW_ACTION_DIM,
                "next_state": V1_STATE_DIM,
                "next_action": V1_RAW_ACTION_DIM,
            }
            tensors: dict[str, torch.Tensor] = {}
            batch_size: int | None = None
            for name, dimension in names_and_dims.items():
                value = batch.get(name)
                if not isinstance(value, torch.Tensor) or value.ndim != 2:
                    raise RuntimeError(
                        f"V1 transition field {name!r} must have shape [B,{dimension}]."
                    )
                if value.shape[-1] != dimension or not bool(
                    torch.isfinite(value).all()
                ):
                    raise RuntimeError(
                        f"V1 transition field {name!r} is malformed or non-finite."
                    )
                if batch_size is None:
                    batch_size = int(value.shape[0])
                elif value.shape[0] != batch_size:
                    raise RuntimeError(
                        "V1 transition fields have different batch sizes."
                    )
                tensors[name] = value
            assert batch_size is not None
            terminal = batch.get("terminal")
            global_rows = batch.get("global_row")
            future_end_rows = batch.get("goal_future_end_row")
            if (
                not isinstance(terminal, torch.Tensor)
                or terminal.shape != (batch_size,)
                or terminal.dtype != torch.bool
            ):
                raise RuntimeError("V1 terminal must be a boolean [B] tensor.")
            for name, value in {
                "global_row": global_rows,
                "goal_future_end_row": future_end_rows,
            }.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (batch_size,)
                    or value.is_floating_point()
                    or value.is_complex()
                ):
                    raise RuntimeError(f"V1 {name} must be an integer [B] tensor.")
            if self.latent_store is None:
                matched_goals = batch.get("_tdwm_matched_goal")
                if not isinstance(
                    matched_goals, torch.Tensor
                ) or matched_goals.shape != (batch_size, V1_TASK_DIM):
                    raise RuntimeError(
                        "V1 requires its frozen latent store to sample real goals."
                    )
                matched_goals = matched_goals.to(
                    device=tensors["state"].device,
                    dtype=tensors["state"].dtype,
                )
            else:
                matched_goals = sample_reachable_future_latents_v1(
                    self.latent_store,
                    global_rows,
                    future_end_rows,
                    generator=(
                        self.goal_generator
                        if stage == "train"
                        else self.validation_goal_generator
                    ),
                    device=tensors["state"].device,
                ).latents.to(dtype=tensors["state"].dtype)
            mixed = sample_mixed_tasks_v1(
                matched_goals,
                goal_probability=float(protocol["task_sampling"]["goal_probability"]),
                generator=(
                    self.task_generator
                    if stage == "train"
                    else self.validation_task_generator
                ),
            )
            state = tensors["state"]
            action = tensors["action"]
            next_state = tensors["next_state"]
            next_action = tensors["next_action"]
            task = mixed.task
            goal_mask = mixed.goal_mask
            action_embedding = encode_frozen_action_blocks_v1(
                self.model.action_encoder,
                action,
                reference=state,
            )
            next_action_embedding = encode_frozen_action_blocks_v1(
                self.model.action_encoder,
                next_action,
                reference=next_state,
            )
            td_batch = build_tdjepa_td_batch_v1(
                self.predictor,
                self.target_predictor,
                state,
                action_embedding,
                task,
                next_state,
                next_action_embedding,
                gamma=self.gamma,
                terminal=terminal,
            )

            method_loss, method_metrics = self._method_loss(
                td_batch,
                state,
                action,
                action_embedding,
                task,
                goal_mask,
                global_rows,
                stage=stage,
            )
            # The common base TD is the primary validation metric for all six
            # variants.  Method-specific objectives remain diagnostics where
            # they are evaluable; G1 cannot query every fixed validation anchor
            # from its immutable training-anchor-only neighbor artifact.
            loss = td_batch.td_loss if stage == "validation" else method_loss
            metrics: dict[str, torch.Tensor] = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/base_td_loss": td_batch.td_loss.detach(),
                f"{stage}/method_td_loss": method_loss.detach(),
                f"{stage}/goal_task_fraction": goal_mask.float().mean(),
                f"{stage}/random_task_fraction": (~goal_mask).float().mean(),
                f"{stage}/terminal_fraction": terminal.float().mean(),
                f"{stage}/td_pairs": loss.new_tensor(float(state.shape[0])),
                f"{stage}/td_prediction_mean": td_batch.prediction.detach().mean(),
                f"{stage}/td_target_mean": td_batch.target.mean(),
                f"{stage}/action_embedding_mean": action_embedding.mean(),
                **{key: value.detach() for key, value in method_metrics.items()},
            }
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
            ema_update_target_v1(
                self.target_predictor,
                self.predictor,
                decay=self.target_ema_decay,
            )

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=float(optimizer_cfg["predictor_learning_rate"]),
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
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return ActorFreeTDLeWMV1TrainingModule()


def _predictor_config(
    protocol: dict[str, Any],
    *,
    spec: ActorFreeTDLeWMV1Spec,
) -> dict[str, Any]:
    predictor = protocol["predictor"]
    return {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        **copy.deepcopy(predictor),
        "task_sampling": copy.deepcopy(protocol["task_sampling"]),
        "joint_objective": copy.deepcopy(protocol["joint_objective"]),
        "pretrained_world_model": copy.deepcopy(protocol["pretrained_world_model"]),
    }


def _deployment_payload(
    module: Any,
    *,
    protocol: dict[str, Any],
    spec: ActorFreeTDLeWMV1Spec,
    model_config: dict[str, Any],
    initialization_info: dict[str, Any],
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    return {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "world_model_state_dict": module.model.state_dict(),
        "world_model_config": model_config,
        "pretrained_world_model_provenance": copy.deepcopy(initialization_info),
        "predictor_state_dict": module.predictor.state_dict(),
        "target_predictor_state_dict": module.target_predictor.state_dict(),
        "predictor_config": _predictor_config(protocol, spec=spec),
    }


def _deployment_checkpoint_path(
    run_dir: Path,
    *,
    spec: ActorFreeTDLeWMV1Spec,
    epoch: int,
) -> Path:
    return (
        run_dir
        / "checkpoints"
        / spec.method
        / spec.variant
        / f"epoch_{int(epoch):02d}.pt"
    )


def _checkpoint_result_fields(
    run_dir: Path,
    *,
    spec: ActorFreeTDLeWMV1Spec,
    deployment_epoch: int,
) -> dict[str, str]:
    return {
        "last_checkpoint": str(run_dir / "checkpoints" / "lightning" / "last.ckpt"),
        "deployment_checkpoint": str(
            _deployment_checkpoint_path(
                run_dir,
                spec=spec,
                epoch=deployment_epoch,
            )
        ),
    }


def _build_export_callback(
    run_dir: Path,
    *,
    protocol: dict[str, Any],
    spec: ActorFreeTDLeWMV1Spec,
    model_config: dict[str, Any],
    initialization_info: dict[str, Any],
):
    import lightning as pl

    class V1ExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = int(trainer.current_epoch) + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            checkpoint_path = _deployment_checkpoint_path(
                run_dir,
                spec=spec,
                epoch=epoch,
            )
            destination = checkpoint_path.parent
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(
                _deployment_payload(
                    pl_module,
                    protocol=protocol,
                    spec=spec,
                    model_config=model_config,
                    initialization_info=initialization_info,
                    epoch=epoch,
                    global_step=int(trainer.global_step),
                ),
                checkpoint_path,
            )

    return V1ExportCallback()


def train_actor_free_td_lewm_v1(
    *,
    spec: ActorFreeTDLeWMV1Spec,
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
    """Train one V1 method from the same audited frozen LeWM artifacts."""

    protocol = load_actor_free_td_lewm_v1_training_protocol(protocol_path, spec=spec)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be auto, never, or required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if not smoke and (max_steps is not None or skip_validation):
        raise ValueError("max_steps and skip_validation are smoke-only.")
    if initial_world_model_checkpoint_path is None:
        raise ValueError("V1 requires --initial-world-model-checkpoint.")
    if frozen_latent_store_path is None or split_indices_path is None:
        raise ValueError("V1 requires frozen latent store and split indices.")
    if spec.requires_neighbor_index != (neighbor_index_path is not None):
        raise ValueError("Only V1 G1 requires --neighbor-index.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / (f"seed_{seed}_smoke" if smoke else f"seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(split_indices_path).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    compatibility = prepare_cloud_runtime() or {}
    import lightning as pl
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import ModelCheckpoint

    version = importlib.metadata.version("stable-worldmodel")
    if version != protocol["runtime"]["stable_worldmodel_version"]:
        raise RuntimeError(f"Expected stable-worldmodel 0.1.1, found {version}.")
    pl.seed_everything(seed, workers=True)
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    if dataset_source["format"] != "lance":
        raise ValueError("V1 frozen training requires the audited Lance dataset.")
    sequence = protocol["sequence"]
    dataset_cfg = protocol["dataset"]
    source_dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=int(sequence["num_steps"]),
        frameskip=int(sequence["frame_skip"]),
        keys_to_load=["action"],
        keys_to_cache=[],
        keys_to_merge={},
    )
    if len(source_dataset.lengths) != int(dataset_cfg["expected_episodes"]):
        raise ValueError("Dataset episode count differs from V1 protocol.")
    if int(np.asarray(source_dataset.lengths).sum()) != int(
        dataset_cfg["expected_transitions"]
    ):
        raise ValueError("Dataset transition count differs from V1 protocol.")

    store, store_info = _resolve_bound_frozen_latent_store(
        frozen_latent_store_path,
        protocol=protocol,
        dataset_source=dataset_source,
        action_dim=CUBE_ACTION_DIM,
    )
    clip_dataset = FrozenLatentClipDataset(source_dataset, store)
    train_indices, validation_indices, split_manifest = load_bound_training_split(
        split_path,
        dataset_size=len(clip_dataset),
        train_fraction=float(protocol["split"]["train_fraction"]),
        validation_fraction=float(protocol["split"]["validation_fraction"]),
    )
    train_set = FrozenActorFreeTDV1TransitionDataset(
        clip_dataset,
        train_indices,
        first_current_index=int(protocol["sequence"]["history_frames"]),
    )
    validation_set = FrozenActorFreeTDV1TransitionDataset(
        clip_dataset,
        validation_indices,
        first_current_index=int(protocol["sequence"]["history_frames"]),
    )
    cross_split_transition_overlap = int(
        np.intersect1d(
            train_set.global_rows,
            validation_set.global_rows,
            assume_unique=True,
        ).size
    )
    loader = protocol["loader"]
    formal_epoch_steps = int(protocol["training"]["optimizer_steps_per_epoch"])
    transition_batch_size = int(loader["batch_size"])
    workers = 0 if smoke else int(loader["workers"])
    train_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(loader["pin_memory"]),
    }
    if workers:
        train_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(loader["prefetch_factor"]),
        )
    data_generator = torch.Generator().manual_seed(seed)
    train_sampler = torch.utils.data.RandomSampler(
        train_set,
        replacement=True,
        num_samples=formal_epoch_steps * transition_batch_size,
        generator=data_generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=transition_batch_size,
        sampler=train_sampler,
        drop_last=bool(loader["train_drop_last"]),
        **train_kwargs,
    )
    validation_workers = 0 if smoke else int(loader["validation_workers"])
    validation_kwargs: dict[str, Any] = {
        "num_workers": validation_workers,
        "pin_memory": bool(loader["pin_memory"]),
    }
    if validation_workers:
        validation_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(loader["prefetch_factor"]),
        )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=transition_batch_size,
        shuffle=False,
        drop_last=bool(loader["validation_drop_last"]),
        **validation_kwargs,
    )

    neighbor_index = None
    neighbor_info = None
    if neighbor_index_path is not None:
        neighbor_index = StateNeighborActionIndex(
            neighbor_index_path,
            expected_checkpoint_sha256=protocol["pretrained_world_model"][
                "checkpoint_sha256"
            ],
            expected_latent_store_manifest_sha256=store_info["manifest_sha256"],
            expected_action_block_dim=V1_RAW_ACTION_DIM,
            expected_k=int(protocol["joint_objective"]["neighbors_per_anchor"]),
        )
        if (
            neighbor_index.manifest.get("training_split_sha256")
            != split_manifest["train_indices_sha256"]
        ):
            raise ValueError("V1 G1 neighbor index uses another training split.")
        neighbor_info = {
            "path": str(Path(neighbor_index_path).expanduser().resolve()),
            "manifest_sha256": _file_sha256(neighbor_index.manifest_path),
        }

    source_name, source_file, source_cache = _resolve_local_pretrained_lewm_export(
        initial_world_model_checkpoint_path
    )
    source_hash = _file_sha256(source_file)
    if source_hash != protocol["pretrained_world_model"]["checkpoint_sha256"]:
        raise ValueError("V1 pretrained checkpoint SHA differs from protocol.")
    if source_hash != store_info["pretrained_checkpoint_sha256"]:
        raise ValueError("V1 latent store was encoded by another LeWM checkpoint.")
    source_training = _verify_completed_pretrained_lewm_run(
        source_run_name=source_name,
        source_cache=source_cache,
        expected_seed=int(protocol["pretrained_world_model"]["source_seed"]),
        expected_epoch=int(protocol["pretrained_world_model"]["source_epoch"]),
    )
    world_model = swm.wm.load_pretrained(source_name, cache_dir=str(source_cache))
    world_model.requires_grad_(False).eval()
    initialization_info = {
        "strategy": "frozen_pretrained_lewm",
        "source_method": "lewm",
        "source_seed": int(protocol["pretrained_world_model"]["source_seed"]),
        "source_epoch": int(protocol["pretrained_world_model"]["source_epoch"]),
        "source_run_name": source_name,
        "source_checkpoint_path": str(source_file),
        "source_checkpoint_sha256": source_hash,
        "frozen": True,
        **source_training,
    }
    model_config = build_model_config(protocol, CUBE_ACTION_DIM)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != int(expected_parameters):
        raise ValueError("Loaded LeWM parameter count differs from protocol.")

    if formal_epoch_steps != len(train_loader):
        raise RuntimeError(
            "V1 transition sampler must yield exactly optimizer_steps_per_epoch."
        )
    train_limit = resolve_train_batch_limit(
        smoke=smoke,
        max_steps=max_steps,
        train_loader_length=len(train_loader),
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
    goal_generator = torch.Generator().manual_seed(
        seed + int(protocol["task_sampling"]["goal_sampling_seed_offset"])
    )
    task_generator = torch.Generator().manual_seed(
        seed + int(protocol["task_sampling"]["task_sampling_seed_offset"])
    )
    validation_goal_generator = torch.Generator().manual_seed(
        seed + int(protocol["task_sampling"]["goal_sampling_seed_offset"]) + 1
    )
    validation_task_generator = torch.Generator().manual_seed(
        seed + int(protocol["task_sampling"]["task_sampling_seed_offset"]) + 1
    )
    module = _build_v1_training_module(
        world_model,
        protocol,
        schedule.total_scheduler_steps,
        spec=spec,
        data_generator=data_generator,
        goal_generator=goal_generator,
        task_generator=task_generator,
        validation_goal_generator=validation_goal_generator,
        validation_task_generator=validation_task_generator,
        neighbor_index=neighbor_index,
        latent_store=store,
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
            protocol=protocol,
            spec=spec,
            model_config=model_config,
            initialization_info=initialization_info,
        ),
    ]
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)
    checkpoint_path: str | None = None
    protocol_hash = _canonical_sha256(protocol)
    manifest_path = run_dir / "training_manifest.json"
    if resume != "never" and last_checkpoint.is_file():
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify V1 resume without its manifest.")
        previous = json.loads(manifest_path.read_text())
        _validate_v1_resume_manifest(
            previous,
            spec=spec,
            protocol_sha256=protocol_hash,
            seed=seed,
            split_manifest=split_manifest,
            initialization_info=initialization_info,
            frozen_latent_store_info=store_info,
            neighbor_index_info=neighbor_info,
        )
        checkpoint_path = str(last_checkpoint)

    cuda_device, cuda_runtime = _cuda_runtime_provenance()
    runtime = {
        "stable_worldmodel": version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdwm_git_revision": _git_revision(),
        "compatibility_adapter": compatibility,
        **cuda_runtime,
    }
    manifest = {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": protocol_hash,
        "seed": seed,
        "dataset": {
            **dataset_source,
            "sequence_samples": len(clip_dataset),
            "train_transition_population": {
                "size": len(train_set),
                **train_set.population_diagnostics,
            },
            "validation_transition_population": {
                "size": len(validation_set),
                **validation_set.population_diagnostics,
            },
            "cross_split_transition_overlap": cross_split_transition_overlap,
            "cross_split_overlap_source": "inherited_sequence_clip_split",
            "split": split_manifest,
        },
        "frozen_latent_store": store_info,
        "neighbor_index": neighbor_info,
        "model": {
            "config": model_config,
            "initialization": initialization_info,
            "lewm_parameters": parameter_count,
            "trainable_lewm_parameters": 0,
            "predictor_parameters": sum(
                parameter.numel() for parameter in module.predictor.parameters()
            ),
        },
        "training": {
            "formal_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
            "optimizer_steps_per_epoch": formal_epoch_steps,
            "configured_optimizer_steps": schedule.total_scheduler_steps,
            "available_batches_per_epoch": len(train_loader),
            "validation_batches": len(validation_loader),
            "validation_skipped": smoke or skip_validation,
            "resumed_from": checkpoint_path,
            "data_source": "frozen_latent_store",
            "sampling_unit": "transition",
            "train_sampling": "random_with_replacement",
            "validation_goal_sampling": "uniform_reachable_future_fixed_per_epoch",
            "validation_task_sampling": "bernoulli_mixture_fixed_per_epoch",
            "validation_primary_objective": "common_base_td_all_variants",
            "transition_batch_size": transition_batch_size,
            "world_model_visual_encode_during_training": False,
            "shared_action_encoder_forward_during_training": True,
        },
        "runtime": runtime,
    }
    write_json(manifest_path, manifest)

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
    _reset_peak_cuda_memory(cuda_device)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=checkpoint_path,
    )
    deployment_checkpoint = _deployment_checkpoint_path(
        run_dir,
        spec=spec,
        epoch=schedule.max_epochs,
    )
    if not deployment_checkpoint.is_file():
        raise RuntimeError(
            "The completed V1 run did not produce its expected deployment "
            f"checkpoint: {deployment_checkpoint}"
        )
    result = {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "run_dir": str(run_dir),
        "seed": seed,
        **_checkpoint_result_fields(
            run_dir,
            spec=spec,
            deployment_epoch=schedule.max_epochs,
        ),
        "final_epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "protocol_sha256": protocol_hash,
        "pretrained_world_model_sha256": source_hash,
        "frozen_latent_store_manifest_sha256": store_info["manifest_sha256"],
    }
    if neighbor_info is not None:
        result["neighbor_index_manifest_sha256"] = neighbor_info["manifest_sha256"]
    _record_peak_cuda_memory(result, cuda_device)
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "ActorFreeTDLeWMV1Spec",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "SUPPORTED_VARIANTS",
    "V1_SPECS",
    "_build_v1_training_module",
    "load_actor_free_td_lewm_v1_training_protocol",
    "train_actor_free_td_lewm_v1",
    "validate_actor_free_td_lewm_v1_training_protocol",
]
