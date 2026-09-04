"""Frozen-parent training runtime for the V1-C3 RP1 state-value critic.

V1-C3 starts from the audited V1-C epoch-10 deployment artifact.  The LeWM
encoder, LeWM dynamics/action encoder, online G, and target G are immutable
checkpoint payloads throughout this run.  Training reads only their cached
LeWM latents and optimizes a new state-goal temporal cost ``V(z, goal)`` plus
its EMA target; there is no actor and no action input to the critic.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.adapters.actor_free_td_lewm_v1_c import (
    load_actor_free_td_lewm_v1_c_checkpoint,
)
from tdwm.methods.actor_free_td_lewm_v1_c3 import (
    RP1StateValueV1C3,
    ema_update_target_v1_c3,
    expectile_huber_td_loss_v1_c3,
    rp1_temporal_td_target_v1_c3,
)
from tdwm.training.cube_data import validate_cube_training_dataset
from tdwm.training.frozen_actor_free_td import (
    _file_sha256,
    _resolve_bound_frozen_latent_store,
    load_bound_training_split,
)
from tdwm.training.frozen_actor_free_td_v1 import (
    _canonical_sha256,
    _state_dict_sha256,
)
from tdwm.training.frozen_actor_free_td_v1_data import (
    FrozenActorFreeTDV1TransitionDataset,
)
from tdwm.training.frozen_latent_store import (
    CUBE_ACTION_DIM,
    FrozenLatentClipDataset,
    FrozenLatentStore,
)
from tdwm.training.gt_lewm_support import build_metrics_logger, write_json
from tdwm.training.lewm import _git_revision

METHOD = "actor_free_td_lewm_v1_c3"
METHOD_FAMILY = "actor_free_td_lewm_v1"
VARIANT = "c3"
IMPLEMENTATION_VERSION = "v1"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1
STATE_DIM = 192
BLOCK_PRIMITIVE_STEPS = 5
FORMAL_OPTIMIZER_STEPS = 12_000


@dataclass(frozen=True)
class V1C3SampledContext:
    """One hindsight goal and one RP1 backup successor per replay anchor."""

    goal: torch.Tensor
    successor: torch.Tensor
    delta_primitive: torch.Tensor
    n_eff_primitive: torch.Tensor
    goal_rows: torch.Tensor
    successor_rows: torch.Tensor


def load_actor_free_td_lewm_v1_c3_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict):
        raise ValueError("V1-C3 training protocol must be a mapping.")
    validate_actor_free_td_lewm_v1_c3_training_protocol(protocol)
    return protocol


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _require_exact(
    mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"{label}.{key} must be {value!r}.")


def validate_actor_free_td_lewm_v1_c3_training_protocol(
    protocol: dict[str, Any],
) -> None:
    """Fail closed on every method-defining V1-C3 choice."""

    _require_exact(
        protocol,
        {
            "schema_version": 1,
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "variant": VARIANT,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "environment": "cube",
            "stage": "full_training",
            "initialization": (
                "v1_c_deployment_frozen_parent_plus_new_state_value"
            ),
            "seeds": [3072],
        },
        "protocol",
    )
    runtime = protocol.get("runtime", {})
    _require_exact(
        runtime,
        {
            "stable_worldmodel_version": "0.1.1",
            "import": "import stable_worldmodel as swm",
        },
        "runtime",
    )
    pretrained = protocol.get("pretrained_world_model", {})
    _require_exact(
        pretrained,
        {
            "source_method": "lewm",
            "source_seed": 3072,
            "source_epoch": 10,
            "checkpoint_sha256": (
                "198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257"
            ),
            "frozen": True,
        },
        "pretrained_world_model",
    )
    source = protocol.get("source_v1_c", {})
    _require_exact(
        source,
        {
            "method": "actor_free_td_lewm_v1_c",
            "method_family": METHOD_FAMILY,
            "variant": "c",
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": 0,
            "deployment_checkpoint_version": 1,
            "source_seed": 3072,
            "source_epoch": 10,
            "source_global_step": 127960,
            "parameter_state": "strict_all_model_parameters_frozen",
            "optimizer_state": "not_loaded",
            "scheduler_state": "not_loaded",
            "epoch_and_global_step": "reset",
        },
        "source_v1_c",
    )
    parent_hash = source.get("checkpoint_sha256")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise ValueError("source_v1_c.checkpoint_sha256 must be a SHA-256 string.")

    _require_exact(
        protocol.get("sequence", {}),
        {
            "frame_skip": BLOCK_PRIMITIVE_STEPS,
            "history_frames": 3,
            "prediction_frames": 1,
            "num_steps": 19,
        },
        "sequence",
    )
    if int(protocol.get("model", {}).get("embed_dim", 0)) != STATE_DIM:
        raise ValueError("V1-C3 requires the 192D V1-C latent space.")

    critic = protocol.get("state_critic", {})
    _require_exact(
        critic,
        {
            "architecture": "rp1_mrn_quasimetric",
            "input": "current_and_goal_frozen_lewm_latents",
            "state_dim": STATE_DIM,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "depth": 2,
            "output": "nonnegative_primitive_temporal_cost_to_go",
            "action_input": "none",
            "actor": "none",
            "goal_identity_value": "exact_zero_by_architecture",
            "block_primitive_steps": BLOCK_PRIMITIVE_STEPS,
            "backup_horizon_primitive_steps": 50,
            "huber_beta_source": "local_prelock_paper_does_not_report",
            "target_ema_source": (
                "nearest_documented_rp1_co_trained_critic_value"
            ),
        },
        "state_critic",
    )
    gamma = _finite_number(
        critic.get("gamma_per_primitive_step"),
        label="state_critic.gamma_per_primitive_step",
    )
    tau = _finite_number(critic.get("expectile"), label="state_critic.expectile")
    beta = _finite_number(critic.get("huber_beta"), label="state_critic.huber_beta")
    eta = _finite_number(
        critic.get("target_ema_update_rate"),
        label="state_critic.target_ema_update_rate",
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("state_critic gamma must lie in [0,1].")
    if not 0.0 < tau < 0.5:
        raise ValueError("Cost critic expectile must lie strictly between 0 and 0.5.")
    if beta <= 0.0 or not 0.0 < eta <= 1.0:
        raise ValueError("Huber beta and EMA update rate must be positive.")

    _require_exact(
        protocol.get("goal_sampling", {}),
        {
            "source": "same_episode_reachable_future_frozen_latent",
            "distribution": "uniform_temporal_offset_over_available_horizon",
            "cross_episode_probability": 0.0,
            "cross_episode_reason": (
                "omitted_because_paper_does_not_specify_unreachable_label"
            ),
            "goal_sampling_seed_offset": 470003,
        },
        "goal_sampling",
    )
    _require_exact(
        protocol.get("objective", {}),
        {
            "name": "rp1_state_value_n_step_expectile_huber_td",
            "exact_goal_branch": "delta_if_goal_inside_n_step_backup",
            "bootstrap_branch": (
                "discounted_primitive_step_cost_plus_ema_state_value"
            ),
            "target_gradient": "stop_gradient",
            "trainable_modules": ["state_critic"],
            "frozen_modules": [
                "lewm_encoder",
                "lewm_predictor",
                "lewm_action_encoder",
                "online_g",
                "target_g",
            ],
        },
        "objective",
    )
    loader = protocol.get("loader", {})
    _require_exact(
        loader,
        {
            "batch_size": 1024,
            "sampling_unit": "block_boundary_anchor",
            "transition_population": (
                "unique_legal_td_rows_from_exact_v1_c_clip_split"
            ),
            "train_sampling": "random_with_replacement",
            "validation_sampling": "sequential_without_replacement",
            "frozen_latent_mmap": True,
            "train_drop_last": True,
            "validation_drop_last": False,
        },
        "loader",
    )
    if int(loader.get("workers", -1)) < 0 or int(
        loader.get("validation_workers", -1)
    ) < 0:
        raise ValueError("V1-C3 loader worker counts cannot be negative.")
    if int(loader.get("prefetch_factor", 0)) <= 0:
        raise ValueError("V1-C3 prefetch_factor must be positive.")

    training = protocol.get("training", {})
    _require_exact(
        training,
        {
            "epochs": 12,
            "scheduler_epochs": 12,
            "optimizer_steps_per_epoch": 1000,
            "total_optimizer_steps": FORMAL_OPTIMIZER_STEPS,
            "precision": "bf16-mixed",
            "model_compile": False,
            "gradient_clip_norm": 1.0,
            "checkpoint_every_epochs": 1,
            "resume": True,
        },
        "training",
    )
    if training["epochs"] * training["optimizer_steps_per_epoch"] != (
        FORMAL_OPTIMIZER_STEPS
    ):
        raise ValueError("V1-C3 must perform exactly 12,000 optimizer steps.")
    optimizer = protocol.get("optimizer", {})
    _require_exact(
        optimizer,
        {
            "type": "AdamW",
            "learning_rate_source": (
                "local_prelock_from_documented_rp1_critic_schedule"
            ),
        },
        "optimizer",
    )
    initial_lr = _finite_number(
        optimizer.get("critic_learning_rate"),
        label="optimizer.critic_learning_rate",
    )
    final_lr = _finite_number(
        optimizer.get("final_learning_rate"),
        label="optimizer.final_learning_rate",
    )
    weight_decay = _finite_number(
        optimizer.get("weight_decay"), label="optimizer.weight_decay"
    )
    if initial_lr <= 0.0 or not 0.0 < final_lr <= initial_lr or weight_decay < 0.0:
        raise ValueError("V1-C3 optimizer values are invalid.")
    _require_exact(
        protocol.get("scheduler", {}),
        {"type": "cosine_annealing", "interval": "optimizer_step"},
        "scheduler",
    )
    bins = protocol.get("validation", {}).get("delta_bins_primitive_steps")
    if bins != [0, 10, 25, 50, 75]:
        raise ValueError("V1-C3 validation delta bins changed.")


def _as_integer_numpy(values: Any, *, label: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device="cpu")
        if tensor.is_floating_point() or tensor.is_complex():
            raise TypeError(f"{label} must be an integer tensor.")
        array = tensor.numpy()
    else:
        array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{label} must be a one-dimensional integer vector.")
    return np.asarray(array, dtype=np.int64)


def sample_v1_c3_context(
    store: FrozenLatentStore,
    global_rows: Sequence[int] | np.ndarray | torch.Tensor,
    future_end_rows: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    backup_horizon_primitive_steps: int,
    generator: torch.Generator,
    device: str | torch.device,
) -> V1C3SampledContext:
    """Sample same-episode goals and a 50-primitive-step RP1 successor.

    Frozen-store row spacing is five primitive actions.  Every sampled block
    offset is converted to primitive time before it enters either the exact
    target or the discount exponent.
    """

    if not isinstance(store, FrozenLatentStore):
        raise TypeError("store must be a validated FrozenLatentStore.")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit CPU torch.Generator.")
    if torch.device(generator.device).type != "cpu":
        raise ValueError("V1-C3 goal sampling uses an explicit CPU RNG.")
    rows = _as_integer_numpy(global_rows, label="global_rows")
    ends = _as_integer_numpy(future_end_rows, label="future_end_rows")
    if rows.size == 0 or rows.shape != ends.shape:
        raise ValueError("V1-C3 goal bounds must be nonempty aligned vectors.")
    frame_skip = int(store.frame_skip)
    if frame_skip != BLOCK_PRIMITIVE_STEPS:
        raise ValueError("V1-C3 requires five primitive steps per cached block.")
    if (
        isinstance(backup_horizon_primitive_steps, bool)
        or backup_horizon_primitive_steps <= 0
        or backup_horizon_primitive_steps % frame_skip
    ):
        raise ValueError("Backup horizon must be a positive whole number of blocks.")
    differences = ends - rows
    if np.any(rows < 0) or np.any(ends >= store.total_rows):
        raise IndexError("V1-C3 row bounds exceed the frozen store.")
    if np.any(differences < frame_skip) or np.any(differences % frame_skip):
        raise ValueError("V1-C3 future bounds must be positive and block aligned.")
    store._assert_immutable()
    if np.any(store.episode_ids[rows] != store.episode_ids[ends]):
        raise ValueError("V1-C3 future bounds cross an episode boundary.")

    maximum_blocks = differences // frame_skip
    uniform = torch.rand(rows.size, generator=generator, device="cpu").numpy()
    delta_blocks = np.floor(uniform * maximum_blocks).astype(np.int64) + 1
    backup_blocks = int(backup_horizon_primitive_steps) // frame_skip
    n_eff_blocks = np.minimum(maximum_blocks, backup_blocks)
    goal_rows = rows + frame_skip * delta_blocks
    successor_rows = rows + frame_skip * n_eff_blocks
    if np.any(store.episode_ids[goal_rows] != store.episode_ids[rows]) or np.any(
        store.episode_ids[successor_rows] != store.episode_ids[rows]
    ):
        raise RuntimeError("V1-C3 sampled context crossed an episode boundary.")
    goals = torch.from_numpy(
        np.array(store.latents[goal_rows], dtype=np.float32, copy=True)
    )
    successors = torch.from_numpy(
        np.array(store.latents[successor_rows], dtype=np.float32, copy=True)
    )
    target_device = torch.device(device)
    return V1C3SampledContext(
        goal=goals.to(device=target_device, non_blocking=True),
        successor=successors.to(device=target_device, non_blocking=True),
        delta_primitive=torch.from_numpy(delta_blocks * frame_skip).to(
            device=target_device, non_blocking=True
        ),
        n_eff_primitive=torch.from_numpy(n_eff_blocks * frame_skip).to(
            device=target_device, non_blocking=True
        ),
        goal_rows=torch.from_numpy(goal_rows).to(
            device=target_device, non_blocking=True
        ),
        successor_rows=torch.from_numpy(successor_rows).to(
            device=target_device, non_blocking=True
        ),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or prediction.shape != target.shape:
        return float("nan")
    first = _average_ranks(prediction)
    second = _average_ranks(target)
    if first.std() == 0.0 or second.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _monotonic_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    order = np.argsort(target, kind="mergesort")
    target_diff = np.diff(target[order])
    valid = target_diff > 0
    if not np.any(valid):
        return float("nan")
    prediction_diff = np.diff(prediction[order])
    return float(np.mean(prediction_diff[valid] >= 0.0))


def _validation_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    td_target: np.ndarray,
    goal_identity: np.ndarray,
    *,
    bins: Sequence[int],
) -> dict[str, Any]:
    residual = prediction - td_target
    finite = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & np.isfinite(td_target)
        & np.isfinite(goal_identity)
    )
    summary: dict[str, Any] = {
        "samples": int(prediction.size),
        "finite_fraction": float(np.mean(finite)),
        "mc_mse": float(np.mean(np.square(prediction - target))),
        "mc_mae": float(np.mean(np.abs(prediction - target))),
        "spearman": _spearman(prediction, target),
        "td_residual_mean": float(np.mean(residual)),
        "td_residual_mae": float(np.mean(np.abs(residual))),
        "goal_identity_mean": float(np.mean(goal_identity)),
        "goal_identity_max": float(np.max(goal_identity)),
        "monotonic_ranking_accuracy": _monotonic_accuracy(prediction, target),
    }
    calibration: list[dict[str, Any]] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        include_upper = upper == bins[-1]
        mask = (target >= lower) & (
            (target <= upper) if include_upper else (target < upper)
        )
        if np.any(mask):
            calibration.append(
                {
                    "lower_inclusive": int(lower),
                    "upper": int(upper),
                    "upper_inclusive": include_upper,
                    "samples": int(mask.sum()),
                    "prediction_mean": float(prediction[mask].mean()),
                    "target_mean": float(target[mask].mean()),
                    "mae": float(np.abs(prediction[mask] - target[mask]).mean()),
                }
            )
    summary["delta_binned_calibration"] = calibration
    return summary


def _load_parent_v1_c(
    checkpoint_path: str | Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    actual_hash = _file_sha256(path)
    source = protocol["source_v1_c"]
    if actual_hash != source["checkpoint_sha256"]:
        raise ValueError("V1-C3 parent checkpoint SHA differs from the protocol.")
    world_model, predictor, predictor_config, payload = (
        load_actor_free_td_lewm_v1_c_checkpoint(path, map_location="cpu")
    )
    _require_exact(
        payload,
        {
            "method": source["method"],
            "method_family": source["method_family"],
            "variant": source["variant"],
            "implementation_version": source["implementation_version"],
            "objective_version": source["objective_version"],
            "deployment_checkpoint_version": source["deployment_checkpoint_version"],
            "epoch": source["source_epoch"],
            "global_step": source["source_global_step"],
        },
        "parent_checkpoint",
    )
    if any(parameter.requires_grad for parameter in world_model.parameters()):
        raise RuntimeError("Loaded V1-C world model was not frozen.")
    if any(parameter.requires_grad for parameter in predictor.parameters()):
        raise RuntimeError("Loaded V1-C online G was not frozen.")
    state_hashes = {
        "world_model_state_sha256": _state_dict_sha256(
            payload["world_model_state_dict"]
        ),
        "online_g_state_sha256": _state_dict_sha256(payload["predictor_state_dict"]),
        "target_g_state_sha256": _state_dict_sha256(
            payload["target_predictor_state_dict"]
        ),
    }
    provenance = {
        "strategy": "strict_frozen_v1_c_epoch_10_parent",
        "parent_checkpoint_path": str(path),
        "parent_checkpoint_sha256": actual_hash,
        "parent_method": payload["method"],
        "parent_epoch": int(payload["epoch"]),
        "parent_global_step": int(payload["global_step"]),
        "predictor_config_sha256": _canonical_sha256(predictor_config),
        **state_hashes,
    }
    del world_model, predictor
    return payload, provenance


def _assert_parent_payload_immutable(
    payload: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    checks = {
        "world_model_state_sha256": _state_dict_sha256(
            payload["world_model_state_dict"]
        ),
        "online_g_state_sha256": _state_dict_sha256(payload["predictor_state_dict"]),
        "target_g_state_sha256": _state_dict_sha256(
            payload["target_predictor_state_dict"]
        ),
    }
    for key, actual in checks.items():
        if actual != provenance[key]:
            raise RuntimeError(f"Frozen V1-C parent changed: {key}.")


def _build_v1_c3_training_module(
    protocol: dict[str, Any],
    *,
    store: FrozenLatentStore,
    data_generator: torch.Generator,
    goal_generator: torch.Generator,
    validation_goal_generator: torch.Generator,
    validation_metrics_path: Path | None = None,
):
    import lightning as pl

    class V1C3TrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            config = protocol["state_critic"]
            self.critic = RP1StateValueV1C3(
                state_dim=int(config["state_dim"]),
                hidden_dim=int(config["hidden_dim"]),
                embedding_dim=int(config["embedding_dim"]),
                depth=int(config["depth"]),
            )
            self.target_critic = self.critic.make_target()
            self.store = store
            self.data_generator = data_generator
            self.goal_generator = goal_generator
            self.validation_goal_generator = validation_goal_generator
            self._validation_goal_epoch_state = (
                validation_goal_generator.get_state().clone()
            )
            self.validation_metrics_path = validation_metrics_path
            self.validation_history: list[dict[str, Any]] = []
            if (
                validation_metrics_path is not None
                and validation_metrics_path.is_file()
            ):
                previous_metrics = json.loads(validation_metrics_path.read_text())
                previous_epochs = previous_metrics.get("epochs", [])
                if not isinstance(previous_epochs, list):
                    raise RuntimeError("V1-C3 validation history is malformed.")
                self.validation_history = list(previous_epochs)
            self._validation_batches: list[tuple[torch.Tensor, ...]] = []
            self.gamma = float(config["gamma_per_primitive_step"])
            self.expectile = float(config["expectile"])
            self.huber_beta = float(config["huber_beta"])
            self.ema_rate = float(config["target_ema_update_rate"])
            self.backup_horizon = int(config["backup_horizon_primitive_steps"])

        def train(self, mode: bool = True):
            super().train(mode)
            self.target_critic.eval()
            return self

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            checkpoint["v1_c3_data_generator_state"] = self.data_generator.get_state()
            checkpoint["v1_c3_goal_generator_state"] = self.goal_generator.get_state()
            checkpoint["v1_c3_validation_goal_generator_state"] = (
                self.validation_goal_generator.get_state()
            )
            checkpoint["v1_c3_validation_goal_epoch_state"] = (
                self._validation_goal_epoch_state.clone()
            )
            checkpoint["v1_c3_validation_history"] = copy.deepcopy(
                self.validation_history
            )

        def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            keys = (
                "v1_c3_data_generator_state",
                "v1_c3_goal_generator_state",
                "v1_c3_validation_goal_generator_state",
                "v1_c3_validation_goal_epoch_state",
            )
            if any(checkpoint.get(key) is None for key in keys):
                raise RuntimeError("V1-C3 resume checkpoint is missing RNG state.")
            self.data_generator.set_state(checkpoint[keys[0]].cpu())
            self.goal_generator.set_state(checkpoint[keys[1]].cpu())
            self.validation_goal_generator.set_state(checkpoint[keys[2]].cpu())
            self._validation_goal_epoch_state = checkpoint[keys[3]].cpu().clone()
            history = checkpoint.get("v1_c3_validation_history")
            if history is None or not isinstance(history, list):
                raise RuntimeError(
                    "V1-C3 resume checkpoint is missing validation history."
                )
            self.validation_history = copy.deepcopy(history)

        def on_validation_epoch_start(self) -> None:
            self.validation_goal_generator.set_state(
                self._validation_goal_epoch_state.clone()
            )
            self._validation_batches.clear()

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            state = batch.get("state")
            rows = batch.get("global_row")
            ends = batch.get("goal_future_end_row")
            if (
                not isinstance(state, torch.Tensor)
                or state.ndim != 2
                or state.shape[-1] != STATE_DIM
                or not bool(torch.isfinite(state).all())
            ):
                raise RuntimeError("V1-C3 state must be a finite [B,192] tensor.")
            batch_size = int(state.shape[0])
            integer_fields = {
                "global_row": rows,
                "goal_future_end_row": ends,
            }
            for name, value in integer_fields.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (batch_size,)
                    or value.is_floating_point()
                    or value.is_complex()
                ):
                    raise RuntimeError(f"V1-C3 {name} must be an integer [B] tensor.")
            context = sample_v1_c3_context(
                self.store,
                rows,
                ends,
                backup_horizon_primitive_steps=self.backup_horizon,
                generator=(
                    self.goal_generator
                    if stage == "train"
                    else self.validation_goal_generator
                ),
                device=state.device,
            )
            goal = context.goal.to(dtype=state.dtype)
            successor = context.successor.to(dtype=state.dtype)
            prediction = self.critic(state, goal)
            with torch.no_grad():
                bootstrap = self.target_critic(successor, goal)
                target_output = rp1_temporal_td_target_v1_c3(
                    context.delta_primitive,
                    context.n_eff_primitive,
                    bootstrap,
                    gamma=self.gamma,
                )
                target = target_output.target
            loss = expectile_huber_td_loss_v1_c3(
                prediction,
                target,
                tau=self.expectile,
                huber_beta=self.huber_beta,
            ).loss
            identity = self.critic(goal, goal)
            residual = prediction.detach().float() - target.detach().float()
            finite = (
                torch.isfinite(prediction)
                & torch.isfinite(target)
                & torch.isfinite(loss)
            )
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/td_prediction_mean": prediction.detach().float().mean(),
                f"{stage}/td_target_mean": target.detach().float().mean(),
                f"{stage}/td_residual_mean": residual.mean(),
                f"{stage}/td_residual_mae": residual.abs().mean(),
                f"{stage}/goal_identity_mean": identity.detach().float().mean(),
                f"{stage}/goal_identity_max": identity.detach().float().max(),
                f"{stage}/exact_target_fraction": (
                    context.delta_primitive.le(context.n_eff_primitive).float().mean()
                ),
                f"{stage}/delta_primitive_mean": (
                    context.delta_primitive.float().mean()
                ),
                f"{stage}/n_eff_primitive_mean": (
                    context.n_eff_primitive.float().mean()
                ),
                f"{stage}/finite_fraction": finite.float().mean(),
            }
            self.log_dict(
                metrics,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=stage == "validation",
                batch_size=batch_size,
                sync_dist=False,
            )
            if stage == "validation":
                self._validation_batches.append(
                    (
                        prediction.detach().float().cpu(),
                        context.delta_primitive.detach().float().cpu(),
                        target.detach().float().cpu(),
                        identity.detach().float().cpu(),
                    )
                )
            return loss

        def training_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "train")

        def validation_step(self, batch: dict[str, Any], batch_idx: int):
            del batch_idx
            return self._forward_loss(batch, "validation")

        def on_validation_epoch_end(self) -> None:
            if not self._validation_batches:
                return
            columns = [
                torch.cat([batch[index] for batch in self._validation_batches]).numpy()
                for index in range(4)
            ]
            bins = protocol["validation"]["delta_bins_primitive_steps"]
            summary = _validation_summary(*columns, bins=bins)
            summary["logical_epoch"] = int(self.current_epoch) + 1
            summary["global_step"] = int(self.global_step)
            self.validation_history.append(summary)
            scalar_metrics = {
                f"validation/{key}": next(self.critic.parameters()).new_tensor(value)
                for key, value in summary.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            if scalar_metrics:
                self.log_dict(scalar_metrics, on_epoch=True, sync_dist=False)
            if self.validation_metrics_path is not None:
                write_json(
                    self.validation_metrics_path,
                    {"epochs": self.validation_history},
                )
            self._validation_batches.clear()

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            del outputs, batch, batch_idx
            ema_update_target_v1_c3(
                self.target_critic,
                self.critic,
                rate=self.ema_rate,
            )

        def configure_optimizers(self):
            optimizer_config = protocol["optimizer"]
            parameters = list(self.critic.parameters())
            if not parameters or not all(
                parameter.requires_grad for parameter in parameters
            ):
                raise RuntimeError(
                    "Only trainable V1-C3 critic parameters may be optimized."
                )
            if any(
                parameter.requires_grad
                for parameter in self.target_critic.parameters()
            ):
                raise RuntimeError("V1-C3 EMA critic must remain frozen.")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(optimizer_config["critic_learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
            )
            initial = float(optimizer_config["critic_learning_rate"])
            final = float(optimizer_config["final_learning_rate"])
            minimum_scale = final / initial

            def learning_rate_scale(step: int) -> float:
                progress = min(max(step, 0), FORMAL_OPTIMIZER_STEPS) / float(
                    FORMAL_OPTIMIZER_STEPS
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return minimum_scale + (1.0 - minimum_scale) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=learning_rate_scale
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return V1C3TrainingModule()


def _critic_config(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        **copy.deepcopy(protocol["state_critic"]),
        "goal_sampling": copy.deepcopy(protocol["goal_sampling"]),
        "objective": copy.deepcopy(protocol["objective"]),
    }


def _deployment_payload(
    module: Any,
    *,
    protocol: dict[str, Any],
    parent_payload: Mapping[str, Any],
    parent_provenance: Mapping[str, Any],
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    _assert_parent_payload_immutable(parent_payload, parent_provenance)
    return {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "logical_epoch": int(epoch),
        "global_step": int(global_step),
        "world_model_state_dict": parent_payload["world_model_state_dict"],
        "world_model_config": parent_payload["world_model_config"],
        "predictor_state_dict": parent_payload["predictor_state_dict"],
        "target_predictor_state_dict": parent_payload[
            "target_predictor_state_dict"
        ],
        "predictor_config": copy.deepcopy(parent_payload["predictor_config"]),
        "critic_state_dict": module.critic.state_dict(),
        "target_critic_state_dict": module.target_critic.state_dict(),
        "critic_config": _critic_config(protocol),
        "pretrained_world_model_provenance": copy.deepcopy(
            parent_payload.get("pretrained_world_model_provenance", {})
        ),
        "source_v1_c_provenance": copy.deepcopy(protocol["source_v1_c"]),
        "source_v1_c_runtime_provenance": copy.deepcopy(dict(parent_provenance)),
        "parent_state_hashes": {
            key: parent_provenance[key]
            for key in (
                "world_model_state_sha256",
                "online_g_state_sha256",
                "target_g_state_sha256",
            )
        },
    }


def _deployment_checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return (
        run_dir
        / "checkpoints"
        / METHOD
        / VARIANT
        / f"epoch_{int(epoch):02d}.pt"
    )


def _build_export_callback(
    run_dir: Path,
    *,
    protocol: dict[str, Any],
    parent_payload: Mapping[str, Any],
    parent_provenance: Mapping[str, Any],
):
    import lightning as pl

    class V1C3ExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = int(trainer.current_epoch) + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            destination = _deployment_checkpoint_path(run_dir, epoch)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _deployment_payload(
                    pl_module,
                    protocol=protocol,
                    parent_payload=parent_payload,
                    parent_provenance=parent_provenance,
                    epoch=epoch,
                    global_step=int(trainer.global_step),
                ),
                destination,
            )

    return V1C3ExportCallback()


def _validate_resume_manifest(
    previous: Mapping[str, Any],
    *,
    protocol_sha256: str,
    seed: int,
    split_manifest: Mapping[str, Any],
    parent_provenance: Mapping[str, Any],
    store_info: Mapping[str, Any],
) -> None:
    checks = {
        "method": METHOD,
        "variant": VARIANT,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
    }
    for key, value in checks.items():
        if previous.get(key) != value:
            raise RuntimeError(f"V1-C3 resume manifest changed: {key}.")
    if (
        previous.get("dataset", {}).get("split", {}).get("train_indices_sha256")
        != split_manifest.get("train_indices_sha256")
    ):
        raise RuntimeError("V1-C3 resume split changed.")
    if (
        previous.get("source_v1_c", {}).get("parent_checkpoint_sha256")
        != parent_provenance["parent_checkpoint_sha256"]
    ):
        raise RuntimeError("V1-C3 resume parent V1-C checkpoint changed.")
    if (
        previous.get("frozen_latent_store", {}).get("manifest_sha256")
        != store_info.get("manifest_sha256")
    ):
        raise RuntimeError("V1-C3 resume frozen latent store changed.")


def train_actor_free_td_lewm_v1_c3(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    initial_v1_c_checkpoint_path: str | Path,
    frozen_latent_store_path: str | Path,
    split_indices_path: str | Path,
    resume: str = "auto",
    smoke: bool = False,
    max_steps: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Train only V1-C3's RP1 state critic on cached frozen V1-C latents."""

    protocol = load_actor_free_td_lewm_v1_c3_training_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the V1-C3 protocol.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be auto, never, or required.")
    if max_steps is not None and (not smoke or max_steps <= 0):
        raise ValueError("Positive max_steps is smoke-only.")
    if skip_validation and not smoke:
        raise ValueError("skip_validation is smoke-only.")

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
        raise ValueError("V1-C3 requires the audited Cube Lance dataset.")
    sequence = protocol["sequence"]
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
    if len(source_dataset.lengths) != int(protocol["dataset"]["expected_episodes"]):
        raise ValueError("V1-C3 dataset episode count changed.")
    if int(np.asarray(source_dataset.lengths).sum()) != int(
        protocol["dataset"]["expected_transitions"]
    ):
        raise ValueError("V1-C3 dataset transition count changed.")
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
        first_current_index=int(sequence["history_frames"]),
    )
    validation_set = FrozenActorFreeTDV1TransitionDataset(
        clip_dataset,
        validation_indices,
        first_current_index=int(sequence["history_frames"]),
    )
    overlap = int(
        np.intersect1d(
            train_set.global_rows,
            validation_set.global_rows,
            assume_unique=True,
        ).size
    )

    parent_payload, parent_provenance = _load_parent_v1_c(
        initial_v1_c_checkpoint_path,
        protocol,
    )
    if (
        parent_provenance["world_model_state_sha256"]
        != _state_dict_sha256(parent_payload["world_model_state_dict"])
    ):
        raise RuntimeError("V1-C3 failed to retain the exact parent LeWM state.")

    loader = protocol["loader"]
    batch_size = int(loader["batch_size"])
    steps_per_epoch = int(protocol["training"]["optimizer_steps_per_epoch"])
    workers = 0 if smoke else int(loader["workers"])
    data_generator = torch.Generator().manual_seed(seed)
    train_sampler = torch.utils.data.RandomSampler(
        train_set,
        replacement=True,
        num_samples=steps_per_epoch * batch_size,
        generator=data_generator,
    )
    train_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(loader["pin_memory"]),
    }
    if workers:
        train_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(loader["prefetch_factor"]),
        )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
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
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **validation_kwargs,
    )
    if len(train_loader) != steps_per_epoch:
        raise RuntimeError("V1-C3 sampler must yield exactly 1,000 batches per epoch.")

    goal_seed_offset = int(protocol["goal_sampling"]["goal_sampling_seed_offset"])
    goal_generator = torch.Generator().manual_seed(seed + goal_seed_offset)
    validation_goal_generator = torch.Generator().manual_seed(
        seed + goal_seed_offset + 1
    )
    validation_metrics_path = run_dir / "validation_offline_metrics.json"
    module = _build_v1_c3_training_module(
        protocol,
        store=store,
        data_generator=data_generator,
        goal_generator=goal_generator,
        validation_goal_generator=validation_goal_generator,
        validation_metrics_path=validation_metrics_path,
    )

    checkpoint_dir = run_dir / "checkpoints" / "lightning"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=1,
        save_last=True,
        save_top_k=-1,
    )
    callbacks = [
        checkpoint_callback,
        _build_export_callback(
            run_dir,
            protocol=protocol,
            parent_payload=parent_payload,
            parent_provenance=parent_provenance,
        ),
    ]
    protocol_hash = _canonical_sha256(protocol)
    manifest_path = run_dir / "training_manifest.json"
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume == "required" and not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)
    resume_checkpoint: str | None = None
    if resume != "never" and last_checkpoint.is_file():
        if not manifest_path.is_file():
            raise RuntimeError("Cannot verify V1-C3 resume without its manifest.")
        previous = json.loads(manifest_path.read_text())
        _validate_resume_manifest(
            previous,
            protocol_sha256=protocol_hash,
            seed=seed,
            split_manifest=split_manifest,
            parent_provenance=parent_provenance,
            store_info=store_info,
        )
        resume_checkpoint = str(last_checkpoint)

    manifest = {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).expanduser().resolve()),
        "protocol_sha256": protocol_hash,
        "seed": int(seed),
        "source_v1_c": parent_provenance,
        "frozen_latent_store": store_info,
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
            "cross_split_transition_overlap": overlap,
            "split": split_manifest,
        },
        "model": {
            "critic_parameters": sum(
                parameter.numel() for parameter in module.critic.parameters()
            ),
            "target_critic_parameters": sum(
                parameter.numel() for parameter in module.target_critic.parameters()
            ),
            "trainable_parent_parameters": 0,
            "trainable_modules": ["state_critic"],
        },
        "training": {
            "formal_optimizer_steps": FORMAL_OPTIMIZER_STEPS,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "epochs": int(protocol["training"]["epochs"]),
            "batch_size": batch_size,
            "data_source": "frozen_latent_store",
            "same_episode_goals_only": True,
            "primitive_step_time_units": True,
            "resumed_from": resume_checkpoint,
        },
        "runtime": {
            "stable_worldmodel": version,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tdwm_git_revision": _git_revision(),
            "compatibility_adapter": compatibility,
            **(
                {"cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
                if torch.cuda.is_available()
                else {}
            ),
        },
    }
    write_json(manifest_path, manifest)

    if smoke:
        effective_steps = int(max_steps or 2)
        max_epochs = 1
        limit_train_batches: int | float = min(effective_steps, len(train_loader))
        trainer_max_steps = effective_steps
    else:
        max_epochs = int(protocol["training"]["epochs"])
        limit_train_batches = steps_per_epoch
        trainer_max_steps = FORMAL_OPTIMIZER_STEPS
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
            max_epochs=max_epochs,
            max_steps=trainer_max_steps,
            gradient_clip_val=float(protocol["training"]["gradient_clip_norm"]),
            limit_train_batches=limit_train_batches,
            limit_val_batches=0.0 if skip_validation else 1.0,
            num_sanity_val_steps=0,
            logger=build_metrics_logger(run_dir, protocol["logging"]),
            callbacks=callbacks,
            log_every_n_steps=1 if smoke else 50,
        )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=resume_checkpoint,
    )
    completed_epoch = 1 if smoke else int(protocol["training"]["epochs"])
    deployment_checkpoint = _deployment_checkpoint_path(run_dir, completed_epoch)
    if not deployment_checkpoint.is_file():
        raise RuntimeError(
            f"V1-C3 did not produce deployment checkpoint {deployment_checkpoint}."
        )
    _assert_parent_payload_immutable(parent_payload, parent_provenance)
    result = {
        "method": METHOD,
        "variant": VARIANT,
        "run_dir": str(run_dir),
        "seed": int(seed),
        "logical_epoch": completed_epoch,
        "global_step": int(trainer.global_step),
        "last_checkpoint": str(last_checkpoint),
        "deployment_checkpoint": str(deployment_checkpoint),
        "deployment_checkpoint_sha256": _file_sha256(deployment_checkpoint),
        "source_v1_c_checkpoint_sha256": parent_provenance[
            "parent_checkpoint_sha256"
        ],
        "parent_state_hashes_verified": True,
        "protocol_sha256": protocol_hash,
        "frozen_latent_store_manifest_sha256": store_info["manifest_sha256"],
        "validation_metrics": str(validation_metrics_path),
    }
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "BLOCK_PRIMITIVE_STEPS",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_OPTIMIZER_STEPS",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "STATE_DIM",
    "V1C3SampledContext",
    "VARIANT",
    "_build_v1_c3_training_module",
    "_deployment_payload",
    "_validation_summary",
    "load_actor_free_td_lewm_v1_c3_training_protocol",
    "sample_v1_c3_context",
    "train_actor_free_td_lewm_v1_c3",
    "validate_actor_free_td_lewm_v1_c3_training_protocol",
]
