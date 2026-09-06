"""Formal frozen-LeWM training runtime for the V1-C4 state-only successor.

C4 is a separate ablation from V1-C.  It loads the same completed LeWM
checkpoint and the same immutable latent store, freezes every LeWM parameter,
and optimizes only one online state/task successor.  A frozen EMA copy supplies
the shared TD target for aligned real-state and frozen-F-predicted-state
branches.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    sample_mixed_tasks_v1,
)
from tdwm.methods.actor_free_td_lewm_v1_c4 import (
    ActorFreeTDJEPAPredictorV1C4,
    build_two_branch_td_loss_v1_c4,
    ema_update_target_v1_c4,
    predict_frozen_lewm_aligned_state_v1_c4,
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
from tdwm.training.frozen_actor_free_td_v1 import (
    _canonical_sha256,
    _cuda_runtime_provenance,
    _record_peak_cuda_memory,
    _reset_peak_cuda_memory,
    _state_dict_sha256,
)
from tdwm.training.frozen_actor_free_td_v1_data import (
    FrozenActorFreeTDV1C4TransitionDataset,
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

METHOD = "actor_free_td_lewm_v1_c4"
METHOD_FAMILY = "actor_free_td_lewm_v1"
VARIANT = "c4"
IMPLEMENTATION_VERSION = "v1"
OBJECTIVE_VERSION = 0
DEPLOYMENT_CHECKPOINT_VERSION = 1
FORMAL_EPOCHS = 10
FORMAL_STEPS_PER_EPOCH = 12_796


def _require_exact(
    mapping: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"{label}.{key} must be {value!r}.")


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


def load_actor_free_td_lewm_v1_c4_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict):
        raise ValueError("V1-C4 training protocol must contain a mapping.")
    validate_actor_free_td_lewm_v1_c4_training_protocol(protocol)
    return protocol


def validate_actor_free_td_lewm_v1_c4_training_protocol(
    protocol: dict[str, Any],
) -> None:
    """Fail closed on the complete C4 method and comparison boundary."""

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
            "initialization": "frozen_pretrained_lewm_new_state_only_g",
            "seeds": [3072],
        },
        label="protocol",
    )
    _require_exact(
        protocol.get("runtime", {}),
        {
            "stable_worldmodel_version": "0.1.1",
            "import": "import stable_worldmodel as swm",
        },
        label="runtime",
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
        label="pretrained_world_model",
    )
    _require_exact(
        protocol.get("sequence", {}),
        {
            "frame_skip": 5,
            "history_frames": 3,
            "prediction_frames": 1,
            "num_steps": 19,
        },
        label="sequence",
    )
    _require_exact(
        protocol.get("context", {}),
        {
            "g_state_frames": 1,
            "lewm_rollout_history_frames": 3,
        },
        label="context",
    )
    if int(protocol.get("model", {}).get("embed_dim", 0)) != V1_STATE_DIM:
        raise ValueError("V1-C4 requires the frozen 192D LeWM latent space.")

    g = protocol.get("g", {})
    _require_exact(
        g,
        {
            "architecture": "td_jepa_state_only_forward_map_v1_c4",
            "state_dim": 192,
            "task_dim": 192,
            "output_dim": 192,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "embedding_layers": 2,
            "num_parallel": 1,
            "action_input": "none",
            "action_effect": "only_via_f_predicted_state",
            "goal_conditioning": "task_input",
            "successor_semantics": "includes_current_input_state",
            "actor": "none",
            "reward": "none",
        },
        label="g",
    )
    forbidden_action_keys = {"raw_action_dim", "action_dim", "action_embedding_dim"}
    present = forbidden_action_keys.intersection(g)
    if present:
        raise ValueError(f"V1-C4 g must not declare action inputs: {sorted(present)}.")
    gamma = _finite_number(g.get("gamma"), label="g.gamma")
    decay = _finite_number(g.get("target_ema_decay"), label="g.target_ema_decay")
    if not 0.0 <= gamma < 1.0 or not 0.0 <= decay < 1.0:
        raise ValueError("V1-C4 gamma and target EMA decay must lie in [0,1).")

    _require_exact(
        protocol.get("task_sampling", {}),
        {
            "sampling": "per_transition_bernoulli",
            "goal_probability": 0.5,
            "random_source": "isotropic_gaussian_sphere",
            "goal_source": "uniform_reachable_future_frozen_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "transition_minibatch",
            "goal_sampling_seed_offset": 170003,
            "task_sampling_seed_offset": 270007,
        },
        label="task_sampling",
    )
    _require_exact(
        protocol.get("time_alignment", {}),
        {
            "replay_anchor": "z_i_minus_1",
            "real_online_input": "real_z_i",
            "predicted_online_input": "stop_gradient_f_of_z_i_minus_1_a_i_minus_1",
            "shared_target_current_feature": "real_z_i",
            "shared_target_bootstrap_input": "real_z_i_plus_1",
            "terminal_semantics": "d_i_true_when_real_z_i_is_terminal",
            "terminal_target": "y_i_equals_z_i",
            "f_history_states": 3,
            "f_output_gradient": "stop_gradient",
        },
        label="time_alignment",
    )
    objective = protocol.get("joint_objective", {})
    _require_exact(
        objective,
        {
            "objective": "equal_real_predicted_state_only_goal_projected_td",
            "vector_td_population": "all_transitions_both_branches",
            "vector_reduction": "mean_of_squared_l2_norm",
            "goal_subset": "goal_derived_tasks_only",
            "goal_projection_weight": 1.0,
            "goal_projection_target": "detached_shared_td_target_projection",
            "branch_combination": "one_half_real_plus_predicted",
            "target_gradient": "stop_gradient",
            "trainable_modules": ["online_g_c4"],
            "frozen_modules": [
                "lewm_observation_encoder",
                "lewm_action_encoder",
                "lewm_world_model_predictor_f",
                "target_g_c4",
            ],
            "lewm_prediction_loss": "none",
            "sigreg_loss": "none",
        },
        label="joint_objective",
    )

    _require_exact(
        protocol.get("loader", {}),
        {
            "batch_size": 256,
            "sampling_unit": "transition",
            "transition_population": (
                "unique_legal_td_rows_from_exact_v1_c_clip_split"
            ),
            "train_sampling": "random_with_replacement",
            "validation_sampling": "sequential_without_replacement",
            "frozen_latent_mmap": True,
            "train_drop_last": True,
            "validation_drop_last": False,
        },
        label="loader",
    )
    loader = protocol["loader"]
    for key in ("workers", "validation_workers"):
        if int(loader.get(key, -1)) < 0:
            raise ValueError(f"loader.{key} cannot be negative.")
    if int(loader.get("prefetch_factor", 0)) <= 0:
        raise ValueError("loader.prefetch_factor must be positive.")

    _require_exact(
        protocol.get("training", {}),
        {
            "epochs": FORMAL_EPOCHS,
            "scheduler_epochs": FORMAL_EPOCHS,
            "optimizer_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
            "precision": "bf16-mixed",
            "model_compile": False,
            "gradient_clip_norm": 1.0,
            "checkpoint_every_epochs": 1,
            "resume": True,
        },
        label="training",
    )
    if FORMAL_EPOCHS * FORMAL_STEPS_PER_EPOCH != FORMAL_OPTIMIZER_UPDATES:
        raise RuntimeError("V1-C4 formal update count drifted from V1-C.")
    optimizer = protocol.get("optimizer", {})
    _require_exact(
        optimizer,
        {"type": "AdamW", "world_model_learning_rate": 0.0},
        label="optimizer",
    )
    if _finite_number(
        optimizer.get("g_learning_rate"), label="optimizer.g_learning_rate"
    ) <= 0.0:
        raise ValueError("optimizer.g_learning_rate must be positive.")
    if _finite_number(
        optimizer.get("weight_decay"), label="optimizer.weight_decay"
    ) < 0.0:
        raise ValueError("optimizer.weight_decay cannot be negative.")
    _require_exact(
        protocol.get("scheduler", {}),
        {"type": "linear_warmup_cosine_annealing", "interval": "optimizer_step"},
        label="scheduler",
    )
    warmup = _finite_number(
        protocol["scheduler"].get("warmup_fraction"),
        label="scheduler.warmup_fraction",
    )
    if not 0.0 <= warmup < 1.0:
        raise ValueError("scheduler.warmup_fraction must lie in [0,1).")


def _build_v1_c4_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    data_generator: torch.Generator,
    goal_generator: torch.Generator,
    task_generator: torch.Generator,
    validation_goal_generator: torch.Generator,
    validation_task_generator: torch.Generator,
    latent_store: Any | None,
):
    import lightning as pl

    class ActorFreeTDLeWMV1C4TrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model.requires_grad_(False).eval()
            config = protocol["g"]
            self.online_g = ActorFreeTDJEPAPredictorV1C4(
                hidden_dim=int(config["hidden_dim"]),
                hidden_layers=int(config["hidden_layers"]),
                embedding_layers=int(config["embedding_layers"]),
            )
            self.target_g = self.online_g.make_target()
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
            self.latent_store = latent_store
            self.gamma = float(config["gamma"])
            self.target_ema_decay = float(config["target_ema_decay"])

        def train(self, mode: bool = True):
            super().train(mode)
            self.model.eval()
            self.target_g.eval()
            return self

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            checkpoint["v1_c4_data_generator_state"] = self.data_generator.get_state()
            checkpoint["v1_c4_goal_generator_state"] = self.goal_generator.get_state()
            checkpoint["v1_c4_task_generator_state"] = self.task_generator.get_state()
            checkpoint["v1_c4_validation_goal_generator_state"] = (
                self.validation_goal_generator.get_state()
            )
            checkpoint["v1_c4_validation_task_generator_state"] = (
                self.validation_task_generator.get_state()
            )
            checkpoint["v1_c4_validation_goal_epoch_state"] = (
                self._validation_goal_epoch_state.clone()
            )
            checkpoint["v1_c4_validation_task_epoch_state"] = (
                self._validation_task_epoch_state.clone()
            )

        def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            keys = (
                "v1_c4_data_generator_state",
                "v1_c4_goal_generator_state",
                "v1_c4_task_generator_state",
                "v1_c4_validation_goal_generator_state",
                "v1_c4_validation_task_generator_state",
                "v1_c4_validation_goal_epoch_state",
                "v1_c4_validation_task_epoch_state",
            )
            if any(checkpoint.get(key) is None for key in keys):
                raise RuntimeError("V1-C4 resume checkpoint is missing RNG state.")
            generators = (
                self.data_generator,
                self.goal_generator,
                self.task_generator,
                self.validation_goal_generator,
                self.validation_task_generator,
            )
            for generator, key in zip(generators, keys[:5], strict=True):
                generator.set_state(checkpoint[key].cpu())
            self._validation_goal_epoch_state = checkpoint[keys[5]].cpu().clone()
            self._validation_task_epoch_state = checkpoint[keys[6]].cpu().clone()

        def on_validation_epoch_start(self) -> None:
            self.validation_goal_generator.set_state(
                self._validation_goal_epoch_state.clone()
            )
            self.validation_task_generator.set_state(
                self._validation_task_epoch_state.clone()
            )

        @staticmethod
        def _finite_batch_vector(
            batch: Mapping[str, Any], name: str, final_dim: int
        ) -> torch.Tensor:
            value = batch.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[-1] != final_dim
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise RuntimeError(
                    f"V1-C4 {name} must be a finite float [B,{final_dim}] tensor."
                )
            return value

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
            real_state = self._finite_batch_vector(
                batch, "c4_real_state", V1_STATE_DIM
            )
            next_state = self._finite_batch_vector(
                batch, "c4_bootstrap_next_state", V1_STATE_DIM
            )
            predecessor_action = self._finite_batch_vector(
                batch, "action", V1_RAW_ACTION_DIM
            )
            batch_size = int(real_state.shape[0])
            if next_state.shape[0] != batch_size or predecessor_action.shape[0] != batch_size:
                raise RuntimeError("V1-C4 transition fields have different batch sizes.")
            state_history = batch.get("c4_f_state_history")
            previous_actions = batch.get("c4_f_previous_actions")
            if (
                not isinstance(state_history, torch.Tensor)
                or state_history.shape != (batch_size, 3, V1_STATE_DIM)
                or not state_history.is_floating_point()
                or not bool(torch.isfinite(state_history).all())
            ):
                raise RuntimeError("V1-C4 F state history must be [B,3,192].")
            if (
                not isinstance(previous_actions, torch.Tensor)
                or previous_actions.shape != (batch_size, 2, V1_RAW_ACTION_DIM)
                or not previous_actions.is_floating_point()
                or not bool(torch.isfinite(previous_actions).all())
            ):
                raise RuntimeError("V1-C4 F action history must be [B,2,25].")
            terminal = batch.get("c4_terminal")
            if (
                not isinstance(terminal, torch.Tensor)
                or terminal.shape != (batch_size,)
                or terminal.dtype != torch.bool
            ):
                raise RuntimeError("V1-C4 c4_terminal must be a boolean [B] tensor.")
            base_terminal = batch.get("terminal")
            if (
                not isinstance(base_terminal, torch.Tensor)
                or base_terminal.dtype != torch.bool
                or base_terminal.shape != terminal.shape
                or not torch.equal(base_terminal, terminal)
            ):
                raise RuntimeError(
                    "V1-C4 terminal mapping must equal the base next-state terminal."
                )
            rows = batch.get("global_row")
            ends = batch.get("goal_future_end_row")
            for name, value in {"global_row": rows, "goal_future_end_row": ends}.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (batch_size,)
                    or value.is_floating_point()
                    or value.is_complex()
                ):
                    raise RuntimeError(f"V1-C4 {name} must be an integer [B] tensor.")

            if self.latent_store is None:
                matched_goals = batch.get("_tdwm_matched_goal")
                if (
                    not isinstance(matched_goals, torch.Tensor)
                    or matched_goals.shape != (batch_size, V1_STATE_DIM)
                ):
                    raise RuntimeError("V1-C4 requires real matched goal latents.")
                matched_goals = matched_goals.to(real_state)
            else:
                matched_goals = sample_reachable_future_latents_v1(
                    self.latent_store,
                    rows,
                    ends,
                    generator=(
                        self.goal_generator
                        if stage == "train"
                        else self.validation_goal_generator
                    ),
                    device=real_state.device,
                ).latents.to(dtype=real_state.dtype)
            mixed = sample_mixed_tasks_v1(
                matched_goals,
                goal_probability=float(
                    protocol["task_sampling"]["goal_probability"]
                ),
                generator=(
                    self.task_generator
                    if stage == "train"
                    else self.validation_task_generator
                ),
            )
            task = mixed.task.to(real_state)
            goal_mask = mixed.goal_mask.to(device=real_state.device)
            predicted_state = predict_frozen_lewm_aligned_state_v1_c4(
                self.model,
                state_history.to(real_state),
                previous_actions.to(real_state),
                predecessor_action,
            )
            if predicted_state.requires_grad or predicted_state.grad_fn is not None:
                raise RuntimeError("V1-C4 frozen F output must be fully detached.")
            output = build_two_branch_td_loss_v1_c4(
                self.online_g,
                self.target_g,
                real_state,
                predicted_state,
                next_state,
                task,
                goal_mask,
                gamma=self.gamma,
                terminal=terminal,
                goal_projection_weight=float(
                    protocol["joint_objective"]["goal_projection_weight"]
                ),
            )
            loss = output.total_loss
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/real_vector_loss": output.real.vector_loss.detach(),
                f"{stage}/real_goal_loss": output.real.goal_loss.detach(),
                f"{stage}/predicted_vector_loss": (
                    output.predicted.vector_loss.detach()
                ),
                f"{stage}/predicted_goal_loss": output.predicted.goal_loss.detach(),
                f"{stage}/c4_total_loss": loss.detach(),
                f"{stage}/goal_task_fraction": goal_mask.float().mean(),
                f"{stage}/random_task_fraction": (~goal_mask).float().mean(),
                f"{stage}/terminal_fraction": terminal.float().mean(),
                f"{stage}/td_pairs": loss.new_tensor(float(batch_size)),
                f"{stage}/real_prediction_mean": output.real.prediction.detach().mean(),
                f"{stage}/predicted_prediction_mean": (
                    output.predicted.prediction.detach().mean()
                ),
                f"{stage}/td_target_mean": output.target.detach().mean(),
                f"{stage}/f_prediction_alignment_mse": (
                    (predicted_state.float() - real_state.detach().float())
                    .square()
                    .mean()
                ),
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
            ema_update_target_v1_c4(
                self.target_g,
                self.online_g,
                decay=self.target_ema_decay,
            )

        def configure_optimizers(self):
            if any(parameter.requires_grad for parameter in self.model.parameters()):
                raise RuntimeError("Every V1-C4 LeWM parameter must remain frozen.")
            if any(parameter.requires_grad for parameter in self.target_g.parameters()):
                raise RuntimeError("The V1-C4 EMA target must remain frozen.")
            parameters = list(self.online_g.parameters())
            if not parameters or not all(parameter.requires_grad for parameter in parameters):
                raise RuntimeError("Only trainable online C4 G parameters may be optimized.")
            optimizer_cfg = protocol["optimizer"]
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(optimizer_cfg["g_learning_rate"]),
                weight_decay=float(optimizer_cfg["weight_decay"]),
            )
            expected_ids = {id(parameter) for parameter in parameters}
            observed_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if observed_ids != expected_ids:
                raise RuntimeError("V1-C4 optimizer must contain exactly online G.")
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

    return ActorFreeTDLeWMV1C4TrainingModule()


def _g_config(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        **copy.deepcopy(protocol["g"]),
        "task_sampling": copy.deepcopy(protocol["task_sampling"]),
        "joint_objective": copy.deepcopy(protocol["joint_objective"]),
        "time_alignment": copy.deepcopy(protocol["time_alignment"]),
        "pretrained_world_model": copy.deepcopy(
            protocol["pretrained_world_model"]
        ),
    }


def _deployment_payload(
    module: Any,
    *,
    protocol: dict[str, Any],
    model_config: dict[str, Any],
    initialization_info: dict[str, Any],
    frozen_world_model_sha256: str,
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    observed_world_hash = _state_dict_sha256(module.model.state_dict())
    if observed_world_hash != frozen_world_model_sha256:
        raise RuntimeError("Frozen V1-C4 LeWM parameters changed during training.")
    return {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "world_model_state_dict": module.model.state_dict(),
        "world_model_config": model_config,
        "pretrained_world_model_provenance": copy.deepcopy(initialization_info),
        "online_g_state_dict": module.online_g.state_dict(),
        "target_g_state_dict": module.target_g.state_dict(),
        "g_config": _g_config(protocol),
        "frozen_world_model_state_sha256": frozen_world_model_sha256,
    }


def _deployment_checkpoint_path(run_dir: Path, *, epoch: int) -> Path:
    return run_dir / "checkpoints" / METHOD / VARIANT / f"epoch_{int(epoch):02d}.pt"


def _build_export_callback(
    run_dir: Path,
    *,
    protocol: dict[str, Any],
    model_config: dict[str, Any],
    initialization_info: dict[str, Any],
    frozen_world_model_sha256: str,
):
    import lightning as pl

    class V1C4ExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = int(trainer.current_epoch) + 1
            if epoch % int(protocol["training"]["checkpoint_every_epochs"]):
                return
            destination = _deployment_checkpoint_path(run_dir, epoch=epoch)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _deployment_payload(
                    pl_module,
                    protocol=protocol,
                    model_config=model_config,
                    initialization_info=initialization_info,
                    frozen_world_model_sha256=frozen_world_model_sha256,
                    epoch=epoch,
                    global_step=int(trainer.global_step),
                ),
                destination,
            )

    return V1C4ExportCallback()


def _validate_resume_manifest(
    previous: Mapping[str, Any],
    *,
    protocol_sha256: str,
    seed: int,
    split_manifest: Mapping[str, Any],
    source_checkpoint_sha256: str,
    store_info: Mapping[str, Any],
) -> None:
    checks = {
        "method": METHOD,
        "variant": VARIANT,
        "protocol_sha256": protocol_sha256,
        "seed": int(seed),
    }
    for key, expected in checks.items():
        if previous.get(key) != expected:
            raise RuntimeError(f"V1-C4 resume manifest changed: {key}.")
    if (
        previous.get("dataset", {}).get("split", {}).get("train_indices_sha256")
        != split_manifest.get("train_indices_sha256")
        or previous.get("dataset", {}).get("split", {}).get(
            "validation_indices_sha256"
        )
        != split_manifest.get("validation_indices_sha256")
    ):
        raise RuntimeError("V1-C4 resume split changed.")
    if (
        previous.get("frozen_latent_store", {}).get("manifest_sha256")
        != store_info.get("manifest_sha256")
    ):
        raise RuntimeError("V1-C4 resume frozen latent store changed.")
    if (
        previous.get("model", {})
        .get("initialization", {})
        .get("source_checkpoint_sha256")
        != source_checkpoint_sha256
    ):
        raise RuntimeError("V1-C4 resume LeWM checkpoint changed.")


def train_actor_free_td_lewm_v1_c4(
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
    frozen_latent_store_path: str | Path | None = None,
    split_indices_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train C4 from the same immutable LeWM inputs as formal V1-C."""

    protocol = load_actor_free_td_lewm_v1_c4_training_protocol(protocol_path)
    if seed != 3072 or seed not in protocol["seeds"]:
        raise ValueError("V1-C4 formal training uses seed 3072 only.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be auto, never, or required.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if not smoke and (max_steps is not None or skip_validation):
        raise ValueError("max_steps and skip_validation are smoke-only.")
    if initial_world_model_checkpoint_path is None:
        raise ValueError("V1-C4 requires --initial-world-model-checkpoint.")
    if frozen_latent_store_path is None or split_indices_path is None:
        raise ValueError("V1-C4 requires frozen latent store and split indices.")

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
        raise ValueError("V1-C4 frozen training requires the audited Lance dataset.")
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
    dataset_cfg = protocol["dataset"]
    if len(source_dataset.lengths) != int(dataset_cfg["expected_episodes"]):
        raise ValueError("Dataset episode count differs from V1-C4 protocol.")
    if int(np.asarray(source_dataset.lengths).sum()) != int(
        dataset_cfg["expected_transitions"]
    ):
        raise ValueError("Dataset transition count differs from V1-C4 protocol.")

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
    train_set = FrozenActorFreeTDV1C4TransitionDataset(
        clip_dataset,
        train_indices,
        first_current_index=int(sequence["history_frames"]),
    )
    validation_set = FrozenActorFreeTDV1C4TransitionDataset(
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

    loader = protocol["loader"]
    batch_size = int(loader["batch_size"])
    steps_per_epoch = int(protocol["training"]["optimizer_steps_per_epoch"])
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
        num_samples=steps_per_epoch * batch_size,
        generator=data_generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
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
        batch_size=batch_size,
        shuffle=False,
        drop_last=bool(loader["validation_drop_last"]),
        **validation_kwargs,
    )
    if len(train_loader) != steps_per_epoch:
        raise RuntimeError("V1-C4 sampler must yield exactly 12796 steps per epoch.")

    source_name, source_file, source_cache = _resolve_local_pretrained_lewm_export(
        initial_world_model_checkpoint_path
    )
    source_hash = _file_sha256(source_file)
    if source_hash != protocol["pretrained_world_model"]["checkpoint_sha256"]:
        raise ValueError("V1-C4 pretrained checkpoint SHA differs from protocol.")
    if source_hash != store_info["pretrained_checkpoint_sha256"]:
        raise ValueError("V1-C4 latent store was encoded by another LeWM checkpoint.")
    source_training = _verify_completed_pretrained_lewm_run(
        source_run_name=source_name,
        source_cache=source_cache,
        expected_seed=int(protocol["pretrained_world_model"]["source_seed"]),
        expected_epoch=int(protocol["pretrained_world_model"]["source_epoch"]),
    )
    world_model = swm.wm.load_pretrained(source_name, cache_dir=str(source_cache))
    world_model.requires_grad_(False).eval()
    frozen_world_model_sha256 = _state_dict_sha256(world_model.state_dict())
    initialization_info = {
        "strategy": "frozen_pretrained_lewm_new_state_only_g",
        "source_method": "lewm",
        "source_seed": int(protocol["pretrained_world_model"]["source_seed"]),
        "source_epoch": int(protocol["pretrained_world_model"]["source_epoch"]),
        "source_run_name": source_name,
        "source_checkpoint_path": str(source_file),
        "source_checkpoint_sha256": source_hash,
        "world_model_state_sha256": frozen_world_model_sha256,
        "frozen": True,
        **source_training,
    }
    model_config = build_model_config(protocol, CUBE_ACTION_DIM)
    parameter_count = sum(parameter.numel() for parameter in world_model.parameters())
    expected_parameters = protocol["model"].get("parameters")
    if expected_parameters and parameter_count != int(expected_parameters):
        raise ValueError("Loaded LeWM parameter count differs from C4 protocol.")

    train_limit = resolve_train_batch_limit(
        smoke=smoke,
        max_steps=max_steps,
        train_loader_length=len(train_loader),
    )
    if not smoke and max_steps is None:
        train_limit = steps_per_epoch
    schedule = resolve_actor_free_training_schedule(
        protocol,
        smoke=smoke,
        resume=resume,
        max_steps=max_steps,
        train_limit=train_limit,
    )
    goal_offset = int(protocol["task_sampling"]["goal_sampling_seed_offset"])
    task_offset = int(protocol["task_sampling"]["task_sampling_seed_offset"])
    module = _build_v1_c4_training_module(
        world_model,
        protocol,
        schedule.total_scheduler_steps,
        data_generator=data_generator,
        goal_generator=torch.Generator().manual_seed(seed + goal_offset),
        task_generator=torch.Generator().manual_seed(seed + task_offset),
        validation_goal_generator=torch.Generator().manual_seed(
            seed + goal_offset + 1
        ),
        validation_task_generator=torch.Generator().manual_seed(
            seed + task_offset + 1
        ),
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
            model_config=model_config,
            initialization_info=initialization_info,
            frozen_world_model_sha256=frozen_world_model_sha256,
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
            raise RuntimeError("Cannot verify V1-C4 resume without its manifest.")
        previous = json.loads(manifest_path.read_text())
        _validate_resume_manifest(
            previous,
            protocol_sha256=protocol_hash,
            seed=seed,
            split_manifest=split_manifest,
            source_checkpoint_sha256=source_hash,
            store_info=store_info,
        )
        resume_checkpoint = str(last_checkpoint)

    cuda_device, cuda_runtime = _cuda_runtime_provenance()
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
            "config": model_config,
            "initialization": initialization_info,
            "lewm_parameters": parameter_count,
            "trainable_lewm_parameters": 0,
            "online_g_parameters": sum(
                parameter.numel() for parameter in module.online_g.parameters()
            ),
            "target_g_parameters": sum(
                parameter.numel() for parameter in module.target_g.parameters()
            ),
            "trainable_modules": ["online_g_c4"],
            "optimizer_scope": "exact_online_g_parameters_only",
        },
        "training": {
            "formal_optimizer_steps": FORMAL_OPTIMIZER_UPDATES,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "configured_optimizer_steps": schedule.total_scheduler_steps,
            "epochs": int(protocol["training"]["epochs"]),
            "available_batches_per_epoch": len(train_loader),
            "validation_batches": len(validation_loader),
            "validation_skipped": smoke or skip_validation,
            "resumed_from": resume_checkpoint,
            "data_source": "frozen_latent_store",
            "world_model_visual_encode_during_training": False,
            "frozen_action_encoder_forward_during_training": True,
            "frozen_f_predictor_forward_during_training": True,
            "f_output_stop_gradient": True,
            "lewm_prediction_loss": False,
            "sigreg_loss": False,
            "loss_metrics": [
                "real_vector_loss",
                "real_goal_loss",
                "predicted_vector_loss",
                "predicted_goal_loss",
                "c4_total_loss",
            ],
        },
        "runtime": {
            "stable_worldmodel": version,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tdwm_git_revision": _git_revision(),
            "compatibility_adapter": compatibility,
            **cuda_runtime,
        },
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
        ckpt_path=resume_checkpoint,
    )
    deployment_checkpoint = _deployment_checkpoint_path(
        run_dir, epoch=schedule.max_epochs
    )
    if not deployment_checkpoint.is_file():
        raise RuntimeError(
            f"Completed V1-C4 run did not produce {deployment_checkpoint}."
        )
    if _state_dict_sha256(module.model.state_dict()) != frozen_world_model_sha256:
        raise RuntimeError("Frozen V1-C4 LeWM changed before completion.")
    result = {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "run_dir": str(run_dir),
        "seed": int(seed),
        "last_checkpoint": str(last_checkpoint),
        "deployment_checkpoint": str(deployment_checkpoint),
        "deployment_checkpoint_sha256": _file_sha256(deployment_checkpoint),
        "final_epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "protocol_sha256": protocol_hash,
        "pretrained_world_model_sha256": source_hash,
        "frozen_latent_store_manifest_sha256": store_info["manifest_sha256"],
        "frozen_world_model_verified": True,
    }
    _record_peak_cuda_memory(result, cuda_device)
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_EPOCHS",
    "FORMAL_STEPS_PER_EPOCH",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "_build_v1_c4_training_module",
    "_deployment_payload",
    "load_actor_free_td_lewm_v1_c4_training_protocol",
    "train_actor_free_td_lewm_v1_c4",
    "validate_actor_free_td_lewm_v1_c4_training_protocol",
]
