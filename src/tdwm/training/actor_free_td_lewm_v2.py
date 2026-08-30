"""Coupled-Hybrid fine-tuning for Actor-Free TD-JEPA V2.

V2 is deliberately a new stage rather than a V1 resume.  Every method starts
from its matching V1 deployment checkpoint, restores the pretrained LeWM and
the online/EMA TD-JEPA predictor, resets the optimizer, and then jointly
fine-tunes the online LeWM and the same predictor.  The real-state and
teacher-forced predicted-state TD branches share one predictor and one EMA
target.  There is no actor, reward model, or action loss.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.adapters.frozen_actor_free_td_v1_common import (
    validate_frozen_actor_free_td_v1_payload,
)
from tdwm.methods.action_prefix_advantage_common import (
    build_zero_mean_action_prefixes,
)
from tdwm.methods.actor_free_td_lewm import (
    actor_free_goal_future_offset_limits,
    ema_update,
    sample_actor_free_goal_offsets,
)
from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_ACTION_EMBEDDING_DIM,
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    V1_TASK_DIM,
    ActorFreeTDJEPAPredictorV1,
    sample_mixed_tasks_v1,
    tdjepa_goal_score_v1,
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
from tdwm.methods.actor_free_td_lewm_v2 import (
    HybridTDJEPATDBatchV2,
    build_hybrid_tdjepa_td_batch_v2,
    encode_trainable_action_blocks_v2,
)
from tdwm.training.actor_free_td_lewm import (
    _build_episode_epoch_callback,
    build_actor_free_td_inputs,
    resolve_actor_free_training_schedule,
)
from tdwm.training.block_sampler import BlockShuffleBatchSampler
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.frozen_actor_free_td import (
    FORMAL_OPTIMIZER_UPDATES,
    _file_sha256,
    load_bound_training_split,
)
from tdwm.training.gt_lewm_support import (
    LeWMTransform,
    build_metrics_logger,
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

METHOD_FAMILY = "actor_free_td_lewm_v2"
IMPLEMENTATION_VERSION = "v2"
DEPLOYMENT_CHECKPOINT_VERSION = 1
SUPPORTED_VARIANTS = frozenset({"c", "d", "f", "g1", "g2", "g3"})
V1_SOURCE_FAMILY = "actor_free_td_lewm_v1"
V1_SOURCE_IMPLEMENTATION = "v1"
V1_SOURCE_EPOCH = 10
V1_SOURCE_GLOBAL_STEP = FORMAL_OPTIMIZER_UPDATES
V1_SOURCE_CODE_REVISION = "3c4e62ef2ab72387536433f27ef11bce75477e7e"
V1_SOURCE_SHA256 = {
    "c": "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3",
    "d": "3115fffeb83ba6ae7e0c272913fe7a1ba16d42953b2185f6a3f7b168899d819a",
    "f": "b4de1b511075d763194ad1e332d127cbe390553738162f3a402ef8847bb74fd0",
    "g1": "c224d18fcd8390247f115239c4b2db013479a062438cca92003674c739f3e24b",
    "g2": "1c290f91772b42fdf6824d92832c6fff4e2d8ca3ea08089ff1a41016ea1c2ebe",
    "g3": "b279a85b1dd0816bd5fb9724da490810d470755880639297aa13699c86c2d8fb",
}
V2_RESUME_IDENTITY_KEY = "v2_resume_identity"


@dataclass(frozen=True)
class ActorFreeTDLeWMV2Spec:
    """Identity of one independently trained V2 coupled method."""

    method: str
    variant: str
    requires_neighbor_index: bool = False

    def __post_init__(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported V2 variant: {self.variant!r}.")
        if self.method != f"{METHOD_FAMILY}_{self.variant}":
            raise ValueError("V2 method names must end in their exact variant.")
        if self.requires_neighbor_index != (self.variant == "g1"):
            raise ValueError("Only V2 G1 requires a neighbor index.")


V2_SPECS = {
    variant: ActorFreeTDLeWMV2Spec(
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


def _is_lower_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _build_v2_resume_identity(
    *,
    spec: ActorFreeTDLeWMV2Spec,
    protocol_sha256: str,
    source_v1_sha256: str,
    v2_start_revision: str,
    neighbor_index_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Build the exact identity embedded in every resumable V2 checkpoint."""

    if not _is_lower_hex(protocol_sha256, length=64):
        raise ValueError("V2 resume protocol_sha256 must be lowercase SHA-256.")
    if not _is_lower_hex(source_v1_sha256, length=64):
        raise ValueError("V2 resume source_v1_sha256 must be lowercase SHA-256.")
    if not _is_lower_hex(v2_start_revision, length=40):
        raise ValueError("V2 start revision must be a full lowercase Git revision.")
    if spec.requires_neighbor_index:
        if not _is_lower_hex(neighbor_index_manifest_sha256, length=64):
            raise ValueError(
                "V2 G1 resume identity requires its neighbor-index SHA-256."
            )
    elif neighbor_index_manifest_sha256 is not None:
        raise ValueError("Only V2 G1 may bind a neighbor-index resume identity.")
    return {
        "schema_version": 1,
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "protocol_sha256": protocol_sha256,
        "source_v1_sha256": source_v1_sha256,
        "v2_start_revision": v2_start_revision,
        "neighbor_index_manifest_sha256": neighbor_index_manifest_sha256,
    }


def _validate_v2_resume_checkpoint_identity(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    actual_value = checkpoint.get(V2_RESUME_IDENTITY_KEY)
    if not isinstance(actual_value, Mapping):
        raise RuntimeError("V2 resume checkpoint is missing its embedded identity.")
    actual = dict(actual_value)
    if set(actual) != set(expected):
        raise RuntimeError("V2 resume checkpoint identity fields are incompatible.")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise RuntimeError(
                f"V2 resume checkpoint {key} differs from the current run."
            )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_protocol_mapping(path: Path, *, seen: frozenset[Path]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        raise ValueError("V2 protocol inheritance contains a cycle.")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("V2 protocol must contain a mapping.")
    current = dict(value)
    parent = current.get("extends")
    if parent is None:
        return current
    if not isinstance(parent, str) or not parent:
        raise ValueError("protocol.extends must be a non-empty relative path.")
    parent_path = (resolved.parent / parent).resolve()
    base = _load_protocol_mapping(parent_path, seen=seen | {resolved})
    return _deep_merge(base, current)


def load_actor_free_td_lewm_v2_training_protocol(
    path: str | Path,
    *,
    spec: ActorFreeTDLeWMV2Spec,
) -> dict[str, Any]:
    protocol = _load_protocol_mapping(Path(path), seen=frozenset())
    validate_actor_free_td_lewm_v2_training_protocol(protocol, spec=spec)
    return protocol


def _require_exact(
    mapping: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, expected_value in expected.items():
        if mapping.get(key) != expected_value:
            raise ValueError(f"{label}.{key} must be {expected_value!r}.")


def _positive_float(mapping: Mapping[str, Any], key: str, *, label: str) -> float:
    try:
        value = float(mapping.get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.{key} must be finite and positive.") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label}.{key} must be finite and positive.")
    return value


def validate_actor_free_td_lewm_v2_training_protocol(
    protocol: dict[str, Any],
    *,
    spec: ActorFreeTDLeWMV2Spec,
) -> None:
    """Fail closed on changes to the agreed V2 coupled fine-tune contract."""

    _require_exact(
        protocol,
        {
            "schema_version": 1,
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "environment": "cube",
            "stage": "coupled_hybrid_finetuning",
            "initialization": "corresponding_v1_deployment_finetune",
        },
        label="protocol",
    )
    if protocol.get("seeds") != [3072]:
        raise ValueError("V2 is locked to the only archived matching V1 seed, 3072.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("V2 requires stable-worldmodel 0.1.1.")
    _require_exact(
        protocol.get("pretrained_world_model", {}),
        {
            "source_method": "lewm",
            "source_seed": 3072,
            "source_epoch": 10,
            "checkpoint_sha256": (
                "198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257"
            ),
            "initialization_source": "v1_embedded_state",
            "online_trainable": True,
        },
        label="pretrained_world_model",
    )

    source = protocol.get("source_v1", {})
    _require_exact(
        source,
        {
            "method": f"{V1_SOURCE_FAMILY}_{spec.variant}",
            "method_family": V1_SOURCE_FAMILY,
            "variant": spec.variant,
            "implementation_version": V1_SOURCE_IMPLEMENTATION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": 1,
            "source_seed": 3072,
            "source_epoch": V1_SOURCE_EPOCH,
            "source_global_step": V1_SOURCE_GLOBAL_STEP,
            "checkpoint_sha256": V1_SOURCE_SHA256[spec.variant],
            "source_code_revision": V1_SOURCE_CODE_REVISION,
            "optimizer_state": "reset",
        },
        label="source_v1",
    )

    sequence = protocol.get("sequence", {})
    _require_exact(
        sequence,
        {
            "frame_skip": 5,
            "history_frames": 3,
            "prediction_frames": 1,
            "num_steps": 19,
        },
        label="sequence",
    )
    model = protocol.get("model", {})
    _require_exact(
        model,
        {
            "parameters": 18_034_628,
            "encoder_size": "tiny",
            "patch_size": 14,
            "embed_dim": V1_STATE_DIM,
            "predictor_depth": 6,
            "predictor_heads": 16,
            "predictor_mlp_dim": 2048,
            "predictor_dim_head": 64,
            "predictor_dropout": 0.1,
            "predictor_embedding_dropout": 0.0,
            "projector_hidden_dim": 2048,
        },
        label="model",
    )
    world = protocol.get("world_model", {})
    _require_exact(
        world.get("online", {}),
        {
            "source": "corresponding_v1_embedded_pretrained_lewm",
            "full_lewm_trainable": True,
            "visual_encoder_trainable": True,
            "predictor_trainable": True,
            "projector_trainable": True,
            "action_encoder_trainable": True,
        },
        label="world_model.online",
    )
    _require_exact(
        world.get("target", {}),
        {
            "type": "exponential_moving_average",
            "trainable": False,
            "tracks_full_world_model": True,
            "tracks_action_encoder": True,
        },
        label="world_model.target",
    )
    predictor = protocol.get("predictor", {})
    _require_exact(
        predictor,
        {
            "objective_version": OBJECTIVE_VERSION,
            "architecture": "td_jepa_forward_map_v1",
            "state_dim": V1_STATE_DIM,
            "raw_action_dim": V1_RAW_ACTION_DIM,
            "action_dim": V1_ACTION_EMBEDDING_DIM,
            "action_embedding_dim": V1_ACTION_EMBEDDING_DIM,
            "task_dim": V1_TASK_DIM,
            "output_dim": V1_STATE_DIM,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "embedding_layers": 2,
            "num_parallel": 1,
            "action_processing": "online_shared_lewm_action_encoder",
            "shared_lewm_action_encoder": True,
            "action_encoder_trainable": True,
            "action_encoder_source": "world_model.action_encoder",
            "state_parameterization": "coupled_online_lewm_latent",
            "goal_conditioning": "task_input",
            "bootstrap_action": "ema_dataset_next_action_embedding",
            "actor": "none",
            "reward": "none",
            "gamma": 0.95,
            "target_ema_decay": 0.995,
            "target_world_ema_decay": 0.995,
            "loss_warmup_fraction": 0.05,
        },
        label="predictor",
    )
    gamma = float(predictor.get("gamma", -1.0))
    if not 0.0 <= gamma < 1.0:
        raise ValueError("predictor.gamma must lie in [0, 1).")
    if float(predictor.get("loss_warmup_fraction", -1.0)) != 0.05:
        raise ValueError("V2 retains the old coupled-Hybrid 5% TD warmup.")
    for key in ("target_ema_decay", "target_world_ema_decay"):
        decay = float(predictor.get(key, -1.0))
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"predictor.{key} must lie in [0, 1).")

    tasks = protocol.get("task_sampling", {})
    _require_exact(
        tasks,
        {
            "sampling": "per_transition_bernoulli",
            "goal_probability": 0.5,
            "random_source": "isotropic_gaussian_sphere",
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "flattened_transition_minibatch",
        },
        label="task_sampling",
    )
    objective = protocol.get("joint_objective", {})
    _require_exact(
        objective,
        {
            "local_prediction": "original_lewm_one_step_mse",
            "regularization": "original_lewm_sigreg",
            "target_encoder": "ema_world_model",
            "td_target": "ema_next_latent_plus_ema_predictor_dataset_next_action",
            "bootstrap_action": "ema_world_model_action_encoder",
            "real_td_weight": 1.0,
            "predicted_td_weight": 1.0,
            "predicted_context_detach": False,
            "hybrid_reduction": "sum",
            "per_transition_td_reduction": "feature_sum",
            "batch_td_reduction": "transition_mean",
            "base_td_population": "all_transitions",
            "random_task_weight": 1.0,
            "goal_subset": "goal_derived_tasks_only",
            "final_weight_normalization": "mean_one_over_all_transitions",
            "weight_gradient": "stop_gradient",
            "candidate_td_targets": "none",
            "actor": "none",
            "reward": "none",
            "local_prediction_weight": 1.0,
        },
        label="joint_objective",
    )
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
            "candidate_source": "other_episode_frozen_latent_knn_real_action_blocks",
            "candidate_action_processing": "online_shared_lewm_action_encoder",
            "goal_subset_weighting": "softmax_mean_one",
        },
        "g2": {
            "objective": "prefix_mean_advantage",
            "score_source": "detached_online_predictor",
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "candidate_action_processing": "online_shared_lewm_action_encoder",
            "advantage_reducer": "full_score_minus_all_prefix_mean",
            "goal_subset_weighting": "softmax_mean_one",
        },
        "g3": {
            "objective": "prefix_marginal_advantage",
            "score_source": "detached_online_predictor",
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "candidate_action_processing": "online_shared_lewm_action_encoder",
            "advantage_reducer": "mean_adjacent_prefix_score_deltas",
            "goal_subset_weighting": "softmax_mean_one",
        },
    }
    _require_exact(
        objective,
        variant_objective_locks[spec.variant],
        label="joint_objective",
    )
    if spec.variant == "c":
        _positive_float(
            objective,
            "goal_projection_weight",
            label="joint_objective",
        )
    else:
        _positive_float(objective, "weight_temperature", label="joint_objective")
    if spec.variant == "g1":
        _positive_float(objective, "neighbor_temperature", label="joint_objective")
        if int(objective.get("neighbors_per_anchor", 0)) != 8:
            raise ValueError("joint_objective.neighbors_per_anchor must be 8.")
    if spec.variant in {"g2", "g3"}:
        _require_exact(
            objective,
            {
                "prefix_slots": 5,
                "suffix_fill": "normalized_zero_mean_action",
            },
            label="joint_objective",
        )

    loss = protocol.get("loss", {})
    if loss.get("prediction") != "mse":
        raise ValueError("V2 retains the original LeWM MSE.")
    sigreg = loss.get("sigreg", {})
    _require_exact(
        sigreg,
        {
            "weight": 0.09,
            "knots": 17,
            "num_projections": 1024,
            "effective_batch_size": 128,
        },
        label="loss.sigreg",
    )

    loader = protocol.get("loader", {})
    if int(loader.get("batch_size", 0)) <= 0:
        raise ValueError("loader.batch_size must be positive.")
    if loader.get("sampling_unit") != "sequence_clip":
        raise ValueError("V2 requires live sequence clips for coupled SIGReg.")
    training = protocol.get("training", {})
    epochs = int(training.get("epochs", 0))
    steps = int(training.get("optimizer_steps_per_epoch", 0))
    if epochs != int(training.get("scheduler_epochs", -1)):
        raise ValueError("V2 scheduler and trainer epochs must match.")
    if epochs * steps != FORMAL_OPTIMIZER_UPDATES:
        raise ValueError("V2 formal runs require exactly 127960 new updates.")
    _require_exact(
        training,
        {
            "epochs": 10,
            "scheduler_epochs": 10,
            "optimizer_steps_per_epoch": 12_796,
            "precision": "bf16-mixed",
            "model_compile": False,
            "model_compile_mode": "reduce-overhead",
            "gradient_clip_norm": 1.0,
            "checkpoint_every_epochs": 1,
            "resume": True,
        },
        label="training",
    )
    optimizer = protocol.get("optimizer", {})
    _require_exact(
        optimizer,
        {
            "type": "AdamW",
            "world_model_learning_rate": 5e-5,
            "predictor_learning_rate": 1e-4,
            "initialize_state": "fresh",
            "weight_decay": 0.001,
        },
        label="optimizer",
    )
    _require_exact(
        protocol.get("scheduler", {}),
        {
            "type": "linear_warmup_cosine_annealing",
            "interval": "optimizer_step",
            "warmup_fraction": 0.01,
        },
        label="scheduler",
    )

    _require_exact(
        protocol.get("source_artifacts", {}),
        {
            "split_file_sha256": (
                "4594afb3603b4258431ff9076c82acbe3ddcaccb277940b825a99017ce83d830"
            ),
            "train_indices_sha256": (
                "a1665554b6f5dc1c4aa37768cd7008fdc96f6a55ec5e8e12d9a93afa99880561"
            ),
            "validation_indices_sha256": (
                "e5aed8baa556f3f868ed471c511488df2117332837303ba958df278b34a61a6c"
            ),
            "column_normalization_sha256": (
                "7fd14e6a72841a36abd8f1d4aedf4f17f4f71ca508cacefe331e989664954818"
            ),
            "frozen_latent_store_manifest_sha256": (
                "fc80bcc4187a7fd98ff7bbfcfa1d5a4c3a76b467af2f5f22fed601855c573c7e"
            ),
            "g1_neighbor_index_manifest_sha256": (
                "3b2d785790d86c4c45bc10f1cf706f9fc186a02071fb4f8b586eca75a2af76f2"
            ),
        },
        label="source_artifacts",
    )


@dataclass(frozen=True)
class V2Initialization:
    payload: dict[str, Any]
    checkpoint_path: str
    checkpoint_sha256: str
    predictor_config: dict[str, Any]


def load_v2_initialization(
    checkpoint_path: str | Path,
    *,
    spec: ActorFreeTDLeWMV2Spec,
    protocol: dict[str, Any],
) -> V2Initialization:
    """Validate a matching V1 deployment checkpoint for V2 initialization."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_hash = _file_sha256(path)
    expected_hash = str(protocol["source_v1"]["checkpoint_sha256"])
    if checkpoint_hash != expected_hash:
        raise ValueError("V2 source V1 checkpoint failed its locked SHA-256.")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("V1 deployment checkpoint must contain a mapping.")
    payload = dict(value)
    module = importlib.import_module(
        f"tdwm.adapters.actor_free_td_lewm_v1_{spec.variant}"
    )
    source_spec = module.METHOD_SPEC
    predictor_config = validate_frozen_actor_free_td_v1_payload(
        payload, spec=source_spec
    )
    if int(payload.get("epoch", -1)) != V1_SOURCE_EPOCH:
        raise ValueError("V2 requires the archived V1 epoch-10 deployment.")
    if int(payload.get("global_step", -1)) != V1_SOURCE_GLOBAL_STEP:
        raise ValueError("V2 requires the completed 127960-step V1 deployment.")
    return V2Initialization(
        payload=payload,
        checkpoint_path=str(path),
        checkpoint_sha256=checkpoint_hash,
        predictor_config=predictor_config,
    )


def sample_matched_future_goals_v2(
    ema_latents: torch.Tensor,
    terminals: torch.Tensor,
    *,
    first_current_index: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one reachable EMA future latent for every aligned TD pair."""

    if ema_latents.ndim != 3 or ema_latents.shape[-1] != V1_STATE_DIM:
        raise ValueError("ema_latents must have shape [B,T,192].")
    if terminals.shape != ema_latents.shape[:2]:
        raise ValueError("terminals must match the EMA latent batch/time axes.")
    limits = actor_free_goal_future_offset_limits(
        terminals, first_current_index=first_current_index
    )
    offsets = sample_actor_free_goal_offsets(
        terminals,
        first_current_index=first_current_index,
        generator=generator,
    )
    if bool((limits <= 0).any()):
        raise ValueError("V2 clips contain an unreachable aligned TD transition.")
    batch, time, feature_dim = ema_latents.shape
    current_count = time - int(first_current_index) - 1
    current = torch.arange(
        int(first_current_index),
        int(first_current_index) + current_count,
        device=ema_latents.device,
        dtype=torch.int64,
    ).unsqueeze(0)
    indices = current + offsets.to(device=ema_latents.device, dtype=torch.int64)
    goals = ema_latents.detach().gather(
        1, indices.unsqueeze(-1).expand(batch, current_count, feature_dim)
    )
    return goals.detach(), offsets.detach()


def _mean_or_zero(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.mean() if value.numel() else reference.new_zeros(())


def _build_v2_training_module(
    world_model: Any,
    initialization: V2Initialization,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    spec: ActorFreeTDLeWMV2Spec,
    data_generator: torch.Generator,
    goal_generator: torch.Generator,
    task_generator: torch.Generator,
    validation_goal_generator: torch.Generator,
    validation_task_generator: torch.Generator,
    neighbor_index: StateNeighborActionIndex | None,
    protocol_sha256: str,
    v2_start_revision: str,
    neighbor_index_manifest_sha256: str | None,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class ActorFreeTDLeWMV2TrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            # V2 is fully coupled: online visual/action encoders and LeWM F are
            # trainable.  Their EMA copy is the only world-model target.
            self.model = world_model.requires_grad_(True)
            self.target_model = copy.deepcopy(world_model).requires_grad_(False)
            self.target_model.eval()
            cfg = protocol["predictor"]
            self.predictor = ActorFreeTDJEPAPredictorV1(
                hidden_dim=int(cfg["hidden_dim"]),
                hidden_layers=int(cfg["hidden_layers"]),
                embedding_layers=int(cfg["embedding_layers"]),
            )
            self.predictor.load_state_dict(
                initialization.payload["predictor_state_dict"], strict=True
            )
            self.target_predictor = self.predictor.make_target()
            self.target_predictor.load_state_dict(
                initialization.payload["target_predictor_state_dict"], strict=True
            )
            self.target_predictor.requires_grad_(False).eval()
            if not isinstance(
                getattr(self.model, "action_encoder", None), torch.nn.Module
            ) or not isinstance(
                getattr(self.target_model, "action_encoder", None), torch.nn.Module
            ):
                raise ValueError("V2 requires online and EMA LeWM action encoders.")

            self.variant = spec.variant
            self.neighbor_index = neighbor_index
            if spec.requires_neighbor_index != (neighbor_index is not None):
                raise ValueError("Only V2 G1 accepts a neighbor index.")
            self._v2_resume_identity = _build_v2_resume_identity(
                spec=spec,
                protocol_sha256=protocol_sha256,
                source_v1_sha256=initialization.checkpoint_sha256,
                v2_start_revision=v2_start_revision,
                neighbor_index_manifest_sha256=(neighbor_index_manifest_sha256),
            )
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.frame_skip = int(protocol["sequence"]["frame_skip"])
            self.gamma = float(cfg["gamma"])
            self.target_predictor_ema_decay = float(cfg["target_ema_decay"])
            self.target_world_ema_decay = float(cfg["target_world_ema_decay"])
            self.auxiliary_warmup_steps = int(
                float(cfg["loss_warmup_fraction"]) * int(total_steps)
            )
            self.data_generator = data_generator
            self.goal_generator = goal_generator
            self.task_generator = task_generator
            self.validation_goal_generator = validation_goal_generator
            self.validation_task_generator = validation_task_generator
            self._validation_goal_epoch_state = (
                validation_goal_generator.get_state().clone()
            )
            self._validation_task_epoch_state = (
                validation_task_generator.get_state().clone()
            )
            self.device_image_preprocessing = bool(device_image_preprocessing)
            if self.device_image_preprocessing:
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
                knots=int(sigreg["knots"]),
                num_proj=int(sigreg["num_projections"]),
            )

        def train(self, mode: bool = True):
            super().train(mode)
            self.target_model.eval()
            self.target_predictor.eval()
            return self

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            checkpoint[V2_RESUME_IDENTITY_KEY] = copy.deepcopy(self._v2_resume_identity)
            checkpoint["v2_data_generator_state"] = self.data_generator.get_state()
            checkpoint["v2_goal_generator_state"] = self.goal_generator.get_state()
            checkpoint["v2_task_generator_state"] = self.task_generator.get_state()
            checkpoint["v2_validation_goal_generator_state"] = (
                self.validation_goal_generator.get_state()
            )
            checkpoint["v2_validation_task_generator_state"] = (
                self.validation_task_generator.get_state()
            )
            checkpoint["v2_validation_goal_epoch_state"] = (
                self._validation_goal_epoch_state.clone()
            )
            checkpoint["v2_validation_task_epoch_state"] = (
                self._validation_task_epoch_state.clone()
            )

        def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            _validate_v2_resume_checkpoint_identity(
                checkpoint,
                self._v2_resume_identity,
            )
            keys = (
                "v2_data_generator_state",
                "v2_goal_generator_state",
                "v2_task_generator_state",
                "v2_validation_goal_generator_state",
                "v2_validation_task_generator_state",
                "v2_validation_goal_epoch_state",
                "v2_validation_task_epoch_state",
            )
            if any(checkpoint.get(key) is None for key in keys):
                raise RuntimeError("V2 resume checkpoint is missing RNG state.")
            self.data_generator.set_state(checkpoint["v2_data_generator_state"].cpu())
            self.goal_generator.set_state(checkpoint["v2_goal_generator_state"].cpu())
            self.task_generator.set_state(checkpoint["v2_task_generator_state"].cpu())
            self.validation_goal_generator.set_state(
                checkpoint["v2_validation_goal_generator_state"].cpu()
            )
            self.validation_task_generator.set_state(
                checkpoint["v2_validation_task_generator_state"].cpu()
            )
            self._validation_goal_epoch_state = (
                checkpoint["v2_validation_goal_epoch_state"].cpu().clone()
            )
            self._validation_task_epoch_state = (
                checkpoint["v2_validation_task_epoch_state"].cpu().clone()
            )

        def on_validation_epoch_start(self) -> None:
            self.validation_goal_generator.set_state(
                self._validation_goal_epoch_state.clone()
            )
            self.validation_task_generator.set_state(
                self._validation_task_epoch_state.clone()
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

        def _auxiliary_scale(self) -> float:
            if self.auxiliary_warmup_steps <= 0:
                return 1.0
            return min(
                1.0,
                float(self.global_step + 1) / float(self.auxiliary_warmup_steps),
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
                raise RuntimeError("V2 G1 is missing its neighbor index.")
            neighbors = self.neighbor_index.lookup(
                global_rows,
                device=state.device,
                dtype=raw_action.dtype,
            )
            with torch.no_grad():
                positive = tdjepa_goal_score_v1(
                    self.predictor(state, action_embedding, task), task
                )
                count, candidates = neighbors.actions.shape[:2]
                candidate_state = state.unsqueeze(1).expand(-1, candidates, -1)
                candidate_task = task.unsqueeze(1).expand(-1, candidates, -1)
                candidate_action_embedding = encode_trainable_action_blocks_v2(
                    self.model.action_encoder,
                    neighbors.actions,
                    reference=candidate_state,
                )
                neighbor_scores = tdjepa_goal_score_v1(
                    self.predictor(
                        candidate_state,
                        candidate_action_embedding,
                        candidate_task,
                    ),
                    candidate_task,
                )
            if positive.shape != (count,):
                raise RuntimeError("V2 G1 positive score alignment failed.")
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
                prefix_action_embedding = encode_trainable_action_blocks_v2(
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

        def _branch_method_loss(
            self,
            *,
            branch: str,
            prediction: torch.Tensor,
            state: torch.Tensor,
            raw_action: torch.Tensor,
            action_embedding: torch.Tensor,
            task: torch.Tensor,
            goal_mask: torch.Tensor,
            global_rows: torch.Tensor,
            target: torch.Tensor,
            per_td: torch.Tensor,
            stage: str,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            objective = protocol["joint_objective"]
            prefix = f"{stage}/{branch}"
            if self.variant == "c":
                output = goal_projected_v1_loss(
                    prediction,
                    target,
                    task,
                    goal_mask,
                    per_td,
                    projection_coefficient=float(objective["goal_projection_weight"]),
                )
                return output.loss, {
                    f"{prefix}_goal_projection_loss": output.projection_loss.detach(),
                    f"{prefix}_goal_score_residual_mean": _mean_or_zero(
                        output.score_residual.index_select(0, output.goal_indices),
                        output.loss,
                    ).detach(),
                }
            if self.variant == "d":
                output = goal_value_weighted_v1_loss(
                    target,
                    task,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
                return output.loss, {
                    f"{prefix}_weight_mean": output.weights.mean(),
                    f"{prefix}_weight_std": output.weights.std(unbiased=False),
                }
            if self.variant == "f":
                output = same_future_goal_advantage_v1_loss(
                    target,
                    task,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
                return output.loss, {
                    f"{prefix}_advantage_mean": _mean_or_zero(
                        output.advantage, output.loss
                    ),
                    f"{prefix}_weight_mean": output.weights.mean(),
                    f"{prefix}_weight_std": output.weights.std(unbiased=False),
                }
            if self.variant == "g1":
                if stage != "train":
                    return per_td.mean(), {
                        f"{prefix}_neighbor_objective_available": per_td.new_zeros(())
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
                return output.loss, {
                    f"{prefix}_neighbor_objective_available": per_td.new_ones(()),
                    f"{prefix}_advantage_mean": _mean_or_zero(
                        output.advantage, output.loss
                    ),
                    f"{prefix}_weight_mean": output.weights.mean(),
                }
            prefix_scores = self._score_prefixes(state, raw_action, task)
            if self.variant == "g2":
                output = prefix_mean_advantage_v1_loss(
                    prefix_scores,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
            else:
                output = prefix_marginal_advantage_v1_loss(
                    prefix_scores,
                    goal_mask,
                    per_td,
                    temperature=float(objective["weight_temperature"]),
                )
            return output.loss, {
                f"{prefix}_advantage_mean": _mean_or_zero(
                    output.advantage, output.loss
                ),
                f"{prefix}_weight_mean": output.weights.mean(),
                f"{prefix}_weight_std": output.weights.std(unbiased=False),
            }

        def _method_loss(
            self,
            td_batch: HybridTDJEPATDBatchV2,
            *,
            real_state: torch.Tensor,
            predicted_state: torch.Tensor,
            raw_action: torch.Tensor,
            action_embedding: torch.Tensor,
            task: torch.Tensor,
            goal_mask: torch.Tensor,
            global_rows: torch.Tensor,
            stage: str,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            real_loss, real_metrics = self._branch_method_loss(
                branch="real",
                prediction=td_batch.real_prediction,
                state=real_state,
                raw_action=raw_action,
                action_embedding=action_embedding,
                task=task,
                goal_mask=goal_mask,
                global_rows=global_rows,
                target=td_batch.target,
                per_td=td_batch.real_per_transition_td_loss,
                stage=stage,
            )
            predicted_loss, predicted_metrics = self._branch_method_loss(
                branch="predicted",
                prediction=td_batch.predicted_prediction,
                state=predicted_state,
                raw_action=raw_action,
                action_embedding=action_embedding,
                task=task,
                goal_mask=goal_mask,
                global_rows=global_rows,
                target=td_batch.target,
                per_td=td_batch.predicted_per_transition_td_loss,
                stage=stage,
            )
            return real_loss + predicted_loss, {
                **real_metrics,
                **predicted_metrics,
                f"{stage}/real_method_td_loss": real_loss.detach(),
                f"{stage}/predicted_method_td_loss": predicted_loss.detach(),
            }

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            pixels = batch.get("pixels")
            raw_actions = batch.get("action")
            if not isinstance(pixels, torch.Tensor) or pixels.ndim != 5:
                raise RuntimeError("V2 pixels must have shape [B,T,C,H,W].")
            if (
                not isinstance(raw_actions, torch.Tensor)
                or raw_actions.ndim != 3
                or raw_actions.shape[-1] != V1_RAW_ACTION_DIM
            ):
                raise RuntimeError("V2 actions must have shape [B,T,25].")
            batch_size = int(pixels.shape[0])
            expected_steps = int(protocol["sequence"]["num_steps"])
            if pixels.shape[1] != expected_steps or raw_actions.shape[:2] != (
                batch_size,
                expected_steps,
            ):
                raise RuntimeError("V2 clip length differs from the protocol.")
            global_starts = batch.get("_tdwm_global_start")
            if (
                not isinstance(global_starts, torch.Tensor)
                or global_starts.shape != (batch_size,)
                or global_starts.is_floating_point()
                or global_starts.is_complex()
            ):
                raise RuntimeError("V2 batches require integer global clip starts.")

            pixels = self._preprocess(pixels)
            cleaned_actions = torch.nan_to_num(
                raw_actions, nan=0.0, posinf=0.0, neginf=0.0
            )
            encoder_input = {
                key: value
                for key, value in batch.items()
                if not key.startswith("_tdwm_") and key not in {"pixels", "action"}
            }
            encoder_input.update({"pixels": pixels, "action": cleaned_actions})
            online_output = self.model.encode(dict(encoder_input))
            with torch.no_grad():
                target_output = self.target_model.encode(dict(encoder_input))
            embeddings = online_output["emb"]
            action_embeddings = online_output["act_emb"]
            target_embeddings = target_output["emb"].detach()
            target_action_embeddings = target_output["act_emb"].detach()
            expected_latent_shape = (batch_size, expected_steps, V1_STATE_DIM)
            if embeddings.shape != expected_latent_shape:
                raise RuntimeError("V2 online LeWM returned misaligned latents.")
            if target_embeddings.shape != expected_latent_shape:
                raise RuntimeError("V2 EMA LeWM returned misaligned latents.")
            if action_embeddings.shape != expected_latent_shape:
                raise RuntimeError("V2 online EA returned misaligned embeddings.")
            if target_action_embeddings.shape != expected_latent_shape:
                raise RuntimeError("V2 EMA EA returned misaligned embeddings.")

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
            if local_prediction.shape != local_targets.shape:
                raise RuntimeError("V2 LeWM local prediction is misaligned.")
            prediction_loss = (local_prediction - local_targets).square().mean()
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
            current_slice = slice(self.history_size, expected_steps - 1)
            next_slice = slice(self.history_size + 1, expected_steps)
            real_state = embeddings[:, current_slice].reshape(-1, V1_STATE_DIM)
            predicted_state = (
                td_inputs.predicted_context[:, current_slice]
                .reshape(-1, V1_STATE_DIM)
                .to(real_state)
            )
            raw_action = td_inputs.actions[:, current_slice].reshape(
                -1, V1_RAW_ACTION_DIM
            )
            action_embedding = (
                action_embeddings[:, current_slice]
                .reshape(-1, V1_ACTION_EMBEDDING_DIM)
                .to(real_state)
            )
            ema_next_state = (
                target_embeddings[:, next_slice]
                .reshape(-1, V1_STATE_DIM)
                .to(real_state)
            )
            ema_next_action_embedding = (
                target_action_embeddings[:, next_slice]
                .reshape(-1, V1_ACTION_EMBEDDING_DIM)
                .to(real_state)
            )
            terminal = td_inputs.terminals[:, current_slice].reshape(-1)
            goal_source = (
                self.goal_generator
                if stage == "train"
                else self.validation_goal_generator
            )
            matched_goals, goal_offsets = sample_matched_future_goals_v2(
                target_embeddings,
                td_inputs.terminals,
                first_current_index=self.history_size,
                generator=goal_source,
            )
            matched_goals = matched_goals.to(real_state)
            task_source = (
                self.task_generator
                if stage == "train"
                else self.validation_task_generator
            )
            mixed = sample_mixed_tasks_v1(
                matched_goals.reshape(-1, V1_TASK_DIM),
                goal_probability=float(protocol["task_sampling"]["goal_probability"]),
                generator=task_source,
            )
            task = mixed.task.to(real_state)
            goal_mask = mixed.goal_mask.to(device=real_state.device)
            transition_positions = torch.arange(
                self.history_size,
                expected_steps - 1,
                device=global_starts.device,
                dtype=torch.int64,
            )
            global_rows = (
                global_starts.to(dtype=torch.int64).unsqueeze(1)
                + self.frame_skip * transition_positions.unsqueeze(0)
            ).reshape(-1)
            td_batch = build_hybrid_tdjepa_td_batch_v2(
                self.predictor,
                self.target_predictor,
                real_state,
                predicted_state,
                action_embedding,
                task,
                ema_next_state,
                ema_next_action_embedding,
                gamma=self.gamma,
                terminal=terminal,
            )
            method_loss, method_metrics = self._method_loss(
                td_batch,
                real_state=real_state,
                predicted_state=predicted_state,
                raw_action=raw_action,
                action_embedding=action_embedding,
                task=task,
                goal_mask=goal_mask,
                global_rows=global_rows,
                stage=stage,
            )
            auxiliary_scale = self._auxiliary_scale()
            # The old Hybrid uses a sum, not an average, for its two branches.
            # V2 keeps that coupled topology while replacing the old G/objective
            # with the corresponding V1 C--G3 predictor and loss.
            base_hybrid_td = td_batch.hybrid_td_loss
            auxiliary = method_loss if stage == "train" else base_hybrid_td
            loss = (
                float(protocol["joint_objective"]["local_prediction_weight"])
                * prediction_loss
                + float(protocol["loss"]["sigreg"]["weight"]) * sigreg_loss
                + auxiliary_scale * auxiliary
            )
            metrics: dict[str, torch.Tensor] = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/prediction_loss": prediction_loss.detach(),
                f"{stage}/sigreg_loss": sigreg_loss.detach(),
                f"{stage}/base_hybrid_td_loss": base_hybrid_td.detach(),
                f"{stage}/method_hybrid_td_loss": method_loss.detach(),
                f"{stage}/real_base_td_loss": td_batch.real_td_loss.detach(),
                f"{stage}/predicted_base_td_loss": (
                    td_batch.predicted_td_loss.detach()
                ),
                f"{stage}/td_weight_scale": loss.new_tensor(auxiliary_scale),
                f"{stage}/goal_task_fraction": goal_mask.float().mean(),
                f"{stage}/random_task_fraction": (~goal_mask).float().mean(),
                f"{stage}/terminal_fraction": terminal.float().mean(),
                f"{stage}/td_pairs": loss.new_tensor(float(real_state.shape[0])),
                f"{stage}/goal_offset_mean": goal_offsets.float().mean(),
                f"{stage}/online_action_embedding_mean": action_embedding.mean(),
                f"{stage}/ema_next_action_embedding_mean": (
                    ema_next_action_embedding.mean()
                ),
                **{key: value.detach() for key, value in method_metrics.items()},
            }
            episode_ids = batch.get("_tdwm_episode_id")
            if isinstance(episode_ids, torch.Tensor):
                metrics[f"{stage}/unique_episodes_per_batch"] = loss.new_tensor(
                    float(torch.unique(episode_ids).numel())
                )
            cache_bytes = batch.get("_tdwm_cache_bytes")
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
            ema_update(
                self.target_model,
                self.model,
                decay=self.target_world_ema_decay,
            )
            ema_update(
                self.target_predictor,
                self.predictor,
                decay=self.target_predictor_ema_decay,
            )

        def configure_optimizers(self):
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(self.model.parameters()),
                        "lr": float(optimizer_cfg["world_model_learning_rate"]),
                    },
                    {
                        "params": list(self.predictor.parameters()),
                        "lr": float(optimizer_cfg["predictor_learning_rate"]),
                    },
                ],
                weight_decay=float(optimizer_cfg["weight_decay"]),
            )
            warmup_steps = max(
                1,
                int(float(protocol["scheduler"]["warmup_fraction"]) * int(total_steps)),
            )

            def learning_rate_scale(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(
                    1, int(total_steps) - warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * min(float(progress), 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return ActorFreeTDLeWMV2TrainingModule()


def _predictor_config(
    protocol: dict[str, Any], *, spec: ActorFreeTDLeWMV2Spec
) -> dict[str, Any]:
    return {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        **copy.deepcopy(protocol["predictor"]),
        "task_sampling": copy.deepcopy(protocol["task_sampling"]),
        "joint_objective": copy.deepcopy(protocol["joint_objective"]),
        "source_v1": copy.deepcopy(protocol["source_v1"]),
        "source_artifacts": copy.deepcopy(protocol["source_artifacts"]),
    }


def _deployment_payload(
    module: Any,
    *,
    protocol: dict[str, Any],
    spec: ActorFreeTDLeWMV2Spec,
    world_model_config: Mapping[str, Any],
    initialization: V2Initialization,
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
        "target_world_model_state_dict": module.target_model.state_dict(),
        "world_model_config": copy.deepcopy(dict(world_model_config)),
        "predictor_state_dict": module.predictor.state_dict(),
        "target_predictor_state_dict": module.target_predictor.state_dict(),
        "predictor_config": _predictor_config(protocol, spec=spec),
        "source_v1_provenance": {
            "checkpoint_path": initialization.checkpoint_path,
            "checkpoint_sha256": initialization.checkpoint_sha256,
            "source_epoch": int(initialization.payload["epoch"]),
            "source_global_step": int(initialization.payload["global_step"]),
            "optimizer_state_loaded": False,
            "target_world_initialization": "copy_of_v1_online_world_model",
        },
    }


def _deployment_checkpoint_path(
    run_dir: Path, *, spec: ActorFreeTDLeWMV2Spec, epoch: int
) -> Path:
    return (
        run_dir
        / "checkpoints"
        / spec.method
        / spec.variant
        / f"epoch_{int(epoch):02d}.pt"
    )


def _build_export_callback(
    run_dir: Path,
    *,
    protocol: dict[str, Any],
    spec: ActorFreeTDLeWMV2Spec,
    world_model_config: Mapping[str, Any],
    initialization: V2Initialization,
):
    import lightning as pl

    class V2ExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = int(trainer.current_epoch) + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            checkpoint_path = _deployment_checkpoint_path(
                run_dir, spec=spec, epoch=epoch
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _deployment_payload(
                    pl_module,
                    protocol=protocol,
                    spec=spec,
                    world_model_config=world_model_config,
                    initialization=initialization,
                    epoch=epoch,
                    global_step=int(trainer.global_step),
                ),
                checkpoint_path,
            )

    return V2ExportCallback()


def _validate_v2_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    spec: ActorFreeTDLeWMV2Spec,
    protocol_sha256: str,
    seed: int,
    split_manifest: Mapping[str, Any],
    initialization: V2Initialization,
    neighbor_info: Mapping[str, Any] | None,
    v2_start_revision: str,
) -> None:
    compatible = (
        manifest.get("method") == spec.method
        and manifest.get("method_family") == METHOD_FAMILY
        and manifest.get("variant") == spec.variant
        and manifest.get("implementation_version") == IMPLEMENTATION_VERSION
        and manifest.get("objective_version") == OBJECTIVE_VERSION
        and manifest.get("deployment_checkpoint_version")
        == DEPLOYMENT_CHECKPOINT_VERSION
        and manifest.get("protocol_sha256") == protocol_sha256
        and manifest.get("seed") == seed
        and manifest.get("source_v1", {}).get("checkpoint_sha256")
        == initialization.checkpoint_sha256
        and manifest.get("runtime", {}).get("tdwm_git_revision") == v2_start_revision
        and manifest.get("dataset", {}).get("split", {}).get("train_indices_sha256")
        == split_manifest.get("train_indices_sha256")
        and manifest.get("dataset", {})
        .get("split", {})
        .get("validation_indices_sha256")
        == split_manifest.get("validation_indices_sha256")
    )
    previous_neighbor = manifest.get("neighbor_index")
    if neighbor_info is None:
        compatible = compatible and previous_neighbor is None
    else:
        compatible = compatible and isinstance(previous_neighbor, Mapping)
        compatible = compatible and previous_neighbor.get("manifest_sha256") == (
            neighbor_info.get("manifest_sha256")
        )
    if not compatible:
        raise RuntimeError("Refusing to resume an incompatible V2 run.")


def _load_neighbor_index(
    path: str | Path,
    *,
    protocol: dict[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[StateNeighborActionIndex, dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    source_artifacts = protocol["source_artifacts"]
    manifest_sha256 = _file_sha256(manifest_path)
    if manifest_sha256 != source_artifacts["g1_neighbor_index_manifest_sha256"]:
        raise ValueError("V2 G1 neighbor index differs from the archived V1 index.")
    source_bank = manifest.get("source_bank", {})
    latent_store_hash = source_bank.get("latent_store_manifest_sha256")
    expected_latent_store_hash = source_artifacts["frozen_latent_store_manifest_sha256"]
    if latent_store_hash != expected_latent_store_hash:
        raise ValueError("V2 G1 neighbor index uses another frozen latent store.")
    index = StateNeighborActionIndex(
        root,
        expected_checkpoint_sha256=str(
            protocol["pretrained_world_model"]["checkpoint_sha256"]
        ),
        expected_latent_store_manifest_sha256=expected_latent_store_hash,
        expected_action_block_dim=V1_RAW_ACTION_DIM,
        expected_k=int(protocol["joint_objective"]["neighbors_per_anchor"]),
    )
    if index.manifest.get("training_split_sha256") != split_manifest.get(
        "train_indices_sha256"
    ):
        raise ValueError("V2 G1 neighbor index uses another training split.")
    return index, {
        "path": str(root),
        "manifest_sha256": manifest_sha256,
        "latent_store_manifest_sha256": latent_store_hash,
    }


def train_actor_free_td_lewm_v2(
    *,
    spec: ActorFreeTDLeWMV2Spec,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
    initial_v1_checkpoint_path: str | Path | None = None,
    split_indices_path: str | Path | None = None,
    neighbor_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune one C--G3 method with the full coupled Hybrid dataflow."""

    protocol = load_actor_free_td_lewm_v2_training_protocol(protocol_path, spec=spec)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be auto, never, or required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if not smoke and (max_steps is not None or skip_validation):
        raise ValueError("max_steps and skip_validation are smoke-only.")
    if initial_v1_checkpoint_path is None:
        raise ValueError("V2 requires --initial-v1-checkpoint.")
    if split_indices_path is None:
        raise ValueError("V2 requires the exact V1 --split-indices artifact.")
    if spec.requires_neighbor_index != (neighbor_index_path is not None):
        raise ValueError("Only V2 G1 requires --neighbor-index.")

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = output_dir / (f"seed_{seed}_smoke" if smoke else f"seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(split_indices_path).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    initialization = load_v2_initialization(
        initial_v1_checkpoint_path,
        spec=spec,
        protocol=protocol,
    )

    compatibility = prepare_cloud_runtime() or {}
    import hydra
    import lightning as pl
    import stable_worldmodel as swm
    from lightning.pytorch.callbacks import ModelCheckpoint
    from omegaconf import OmegaConf

    package_version = importlib.metadata.version("stable-worldmodel")
    if package_version != protocol["runtime"]["stable_worldmodel_version"]:
        raise RuntimeError(
            f"Expected stable-worldmodel 0.1.1, found {package_version}."
        )
    pl.seed_everything(seed, workers=True)
    dataset_source = validate_cube_training_dataset(dataset_path, protocol["dataset"])
    if dataset_source["format"] != "lance":
        raise ValueError("V2 coupled training requires the audited Lance clip loader.")
    sequence = protocol["sequence"]
    dataset_cfg = protocol["dataset"]
    loader_cfg = protocol["loader"]
    device_preprocessing = bool(loader_cfg["device_image_preprocessing"])
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        transform=None,
        num_steps=int(sequence["num_steps"]),
        frameskip=int(sequence["frame_skip"]),
        keys_to_load=list(dataset_cfg["keys_to_load"]),
        keys_to_cache=list(dataset_cfg["keys_to_cache"]),
        keys_to_merge=dict(dataset_cfg["keys_to_merge"]),
    )
    if len(dataset.lengths) != int(dataset_cfg["expected_episodes"]):
        raise ValueError("Dataset episode count differs from the V2 protocol.")
    if int(np.asarray(dataset.lengths).sum()) != int(
        dataset_cfg["expected_transitions"]
    ):
        raise ValueError("Dataset transition count differs from the V2 protocol.")
    normalization_path = output_dir / "column_normalization.json"
    statistics = fit_column_stats(
        dataset,
        list(protocol["normalization"]["columns"]),
        normalization_path,
    )
    normalization_sha256 = _file_sha256(normalization_path)
    if (
        normalization_sha256
        != protocol["source_artifacts"]["column_normalization_sha256"]
    ):
        raise ValueError(
            "V2 column normalization differs from the archived V1 normalization."
        )
    dataset.transform = LeWMTransform(
        image=protocol["image_preprocessing"],
        columns=statistics,
        preprocess_images=not device_preprocessing,
    )
    decoded_frame_store_metadata = None
    if dataset_source["format"] == "lance":
        decoded_frame_store, decoded_frame_store_metadata = (
            _prepare_decoded_frame_store(protocol, dataset_source, dataset)
        )
        dataset = StrideAwareLanceDataset(
            dataset, decoded_frame_store=decoded_frame_store
        )
    elif os.environ.get(DECODED_FRAME_STORE_ENV) is not None:
        raise ValueError(
            f"{DECODED_FRAME_STORE_ENV} is only supported for Lance datasets."
        )

    train_indices, validation_indices, split_manifest = load_bound_training_split(
        split_path,
        dataset_size=len(dataset),
        train_fraction=float(protocol["split"]["train_fraction"]),
        validation_fraction=float(protocol["split"]["validation_fraction"]),
    )
    source_artifacts = protocol["source_artifacts"]
    expected_split = {
        "file_sha256": source_artifacts["split_file_sha256"],
        "train_indices_sha256": source_artifacts["train_indices_sha256"],
        "validation_indices_sha256": source_artifacts["validation_indices_sha256"],
    }
    _require_exact(split_manifest, expected_split, label="split_indices")
    train_set = torch.utils.data.Subset(dataset, train_indices.tolist())
    validation_set = torch.utils.data.Subset(dataset, validation_indices.tolist())
    data_generator = torch.Generator().manual_seed(seed)
    episode_train_dataset = None
    use_episode_streaming = bool(loader_cfg["episode_streaming"]) and not smoke
    if use_episode_streaming:
        if not isinstance(dataset, StrideAwareLanceDataset):
            raise ValueError("V2 episode streaming requires audited Lance data.")
        episode_train_dataset = EpisodeStreamingBatchDataset(
            dataset,
            train_indices,
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
                persistent_workers=True,
                prefetch_factor=int(loader_cfg["prefetch_factor"]),
            )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(loader_cfg["batch_size"]),
            shuffle=bool(loader_cfg["train_shuffle"]),
            drop_last=bool(loader_cfg["train_drop_last"]),
            generator=data_generator,
            **train_kwargs,
        )
    validation_workers = 0 if smoke else int(loader_cfg["validation_workers"])
    validation_kwargs: dict[str, Any] = {
        "num_workers": validation_workers,
        "pin_memory": bool(loader_cfg["pin_memory"]),
    }
    if validation_workers:
        validation_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(loader_cfg["prefetch_factor"]),
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
            shuffle=False,
            drop_last=bool(loader_cfg["validation_drop_last"]),
            **validation_kwargs,
        )

    neighbor_index = None
    neighbor_info = None
    if neighbor_index_path is not None:
        neighbor_index, neighbor_info = _load_neighbor_index(
            neighbor_index_path,
            protocol=protocol,
            split_manifest=split_manifest,
        )

    world_model_config_value = initialization.payload["world_model_config"]
    if not isinstance(world_model_config_value, Mapping):
        raise ValueError("V1 source world_model_config must be a mapping.")
    world_model_config = copy.deepcopy(dict(world_model_config_value))
    world_model = hydra.utils.instantiate(OmegaConf.create(world_model_config))
    world_model.load_state_dict(
        initialization.payload["world_model_state_dict"], strict=True
    )
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != int(expected_parameters):
        raise ValueError("V2 source LeWM parameter count differs from protocol.")

    available_epoch_steps = len(train_loader)
    formal_epoch_steps = int(protocol["training"]["optimizer_steps_per_epoch"])
    if formal_epoch_steps > available_epoch_steps:
        raise ValueError("optimizer_steps_per_epoch exceeds available V2 batches.")
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
    protocol_hash = _canonical_sha256(protocol)
    v2_start_revision = _git_revision()
    if not _is_lower_hex(v2_start_revision, length=40):
        raise RuntimeError(
            "V2 training requires a full Git revision for resumable identity."
        )
    neighbor_manifest_sha256 = (
        str(neighbor_info["manifest_sha256"]) if neighbor_info is not None else None
    )
    module = _build_v2_training_module(
        world_model,
        initialization,
        protocol,
        schedule.total_scheduler_steps,
        spec=spec,
        data_generator=data_generator,
        goal_generator=goal_generator,
        task_generator=task_generator,
        validation_goal_generator=validation_goal_generator,
        validation_task_generator=validation_task_generator,
        neighbor_index=neighbor_index,
        protocol_sha256=protocol_hash,
        v2_start_revision=v2_start_revision,
        neighbor_index_manifest_sha256=neighbor_manifest_sha256,
        device_image_preprocessing=device_preprocessing,
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
            world_model_config=world_model_config,
            initialization=initialization,
        ),
    ]
    if episode_train_dataset is not None:
        callbacks.append(_build_episode_epoch_callback(episode_train_dataset))

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)
    checkpoint_path: str | None = None
    if resume != "never" and last_checkpoint.is_file():
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify V2 resume without its manifest.")
        with manifest_path.open() as stream:
            previous_manifest = json.load(stream)
        _validate_v2_resume_manifest(
            previous_manifest,
            spec=spec,
            protocol_sha256=protocol_hash,
            seed=seed,
            split_manifest=split_manifest,
            initialization=initialization,
            neighbor_info=neighbor_info,
            v2_start_revision=v2_start_revision,
        )
        checkpoint_path = str(last_checkpoint)

    runtime = {
        "stable_worldmodel": package_version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdwm_git_revision": v2_start_revision,
        "compatibility_adapter": compatibility,
    }
    if torch.cuda.is_available():
        runtime["cuda_device"] = torch.cuda.get_device_name(0)
    dataset_manifest: dict[str, Any] = {
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
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).expanduser().resolve()),
        "protocol_sha256": protocol_hash,
        "seed": seed,
        "source_v1": {
            **copy.deepcopy(protocol["source_v1"]),
            "checkpoint_path": initialization.checkpoint_path,
            "checkpoint_sha256": initialization.checkpoint_sha256,
            "optimizer_state_loaded": False,
        },
        "source_artifacts": {
            **copy.deepcopy(protocol["source_artifacts"]),
            "column_normalization_path": str(normalization_path),
            "column_normalization_sha256": normalization_sha256,
        },
        "dataset": dataset_manifest,
        "neighbor_index": neighbor_info,
        "model": {
            "world_model_parameters": parameter_count,
            "predictor_parameters": sum(
                parameter.numel() for parameter in module.predictor.parameters()
            ),
            "online_world_model_trainable": True,
            "online_action_encoder_trainable": True,
            "target_world_model_trainable": False,
            "target_predictor_trainable": False,
        },
        "training": {
            "formal_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
            "optimizer_steps_per_epoch": formal_epoch_steps,
            "available_batches_per_epoch": available_epoch_steps,
            "configured_optimizer_steps": schedule.total_scheduler_steps,
            "optimizer_initialized_fresh": True,
            "resume_mode": resume,
            "resumed_from": checkpoint_path,
            "episode_streaming": use_episode_streaming,
            "validation_batches": len(validation_loader),
            "validation_skipped": smoke or skip_validation,
        },
        "runtime": runtime,
    }
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
    deployment_checkpoint = _deployment_checkpoint_path(
        run_dir, spec=spec, epoch=schedule.max_epochs
    )
    if not deployment_checkpoint.is_file():
        raise RuntimeError(
            "The completed V2 run did not produce its expected deployment "
            f"checkpoint: {deployment_checkpoint}"
        )
    result = {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "run_dir": str(run_dir),
        "seed": seed,
        "last_checkpoint": str(last_checkpoint),
        "deployment_checkpoint": str(deployment_checkpoint),
        "source_v1_checkpoint_sha256": initialization.checkpoint_sha256,
        "final_epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
    }
    if torch.cuda.is_available():
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "ActorFreeTDLeWMV2Spec",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD_FAMILY",
    "SUPPORTED_VARIANTS",
    "V1_SOURCE_SHA256",
    "V2Initialization",
    "V2_SPECS",
    "load_actor_free_td_lewm_v2_training_protocol",
    "load_v2_initialization",
    "sample_matched_future_goals_v2",
    "train_actor_free_td_lewm_v2",
    "validate_actor_free_td_lewm_v2_training_protocol",
]
