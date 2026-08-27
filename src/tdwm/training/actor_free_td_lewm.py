"""End-to-end training for the three actor-free TD-LeWM variants."""

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
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.methods.actor_free_td_lewm import (
    SUPPORTED_VARIANTS,
    ActorFreeSuccessorHead,
    actor_free_td_objective,
    ema_update,
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
from tdwm.training.lance_batch import (
    EpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
)
from tdwm.training.lewm import _git_revision
from tdwm.training.rf_successor_lewm import (
    DECODED_FRAME_STORE_ENV,
    _prepare_decoded_frame_store,
)

METHOD = "actor_free_td_lewm"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1
FORMAL_OPTIMIZER_UPDATES = 127_960


def load_actor_free_td_training_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_actor_free_td_training_protocol(protocol)
    return protocol


def validate_actor_free_td_training_protocol(protocol: dict[str, Any]) -> None:
    """Reject protocol drift that would invalidate a three-way comparison."""

    if protocol.get("schema_version") != 1 or protocol.get("method") != METHOD:
        raise ValueError("Actor-Free TD-LeWM requires its schema 1 method.")
    if (
        protocol.get("environment") != "cube"
        or protocol.get("stage") != "full_training"
        or protocol.get("initialization") != "random_from_scratch"
    ):
        raise ValueError("Actor-Free TD-LeWM is locked to full Cube training.")
    variant = protocol.get("variant")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported Actor-Free TD-LeWM variant: {variant!r}.")
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

    expected_weights = {
        "serial_decoupled": (0.0, 1.0, True),
        "serial_coupled": (0.0, 1.0, False),
        "hybrid": (1.0, 1.0, False),
    }[variant]
    objective = protocol.get("joint_objective", {})
    expected_objective = {
        "local_prediction": "original_lewm_one_step_mse",
        "regularization": "original_lewm_sigreg",
        "td_target": "ema_next_latent_plus_ema_successor_dataset_next_action",
        "target_encoder": "ema_world_model",
        "bootstrap_action": "dataset_next_action",
        "terminal_mask": "next_action_nan_invalid",
        "goal_conditioning": "none",
        "actor": "none",
        "reward": "none",
        "local_prediction_weight": 1.0,
        "real_td_weight": expected_weights[0],
        "predicted_td_weight": expected_weights[1],
        "predicted_context_detach": expected_weights[2],
    }
    for key, expected in expected_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"joint_objective.{key} must be {expected!r}.")

    successor = protocol.get("successor", {})
    locked_successor = {
        "objective_version": OBJECTIVE_VERSION,
        "architecture": "actor_free_successor_head",
        "feature_basis": "augmented_latent_squared_distance",
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "goal_conditioning": "none",
        "actor": "none",
        "reward": "none",
    }
    for key, expected in locked_successor.items():
        if successor.get(key) != expected:
            raise ValueError(f"successor.{key} must be {expected!r}.")
    if int(successor.get("hidden_dim", 0)) <= 0:
        raise ValueError("successor.hidden_dim must be positive.")
    if not 0.0 <= float(successor.get("gamma", -1.0)) < 1.0:
        raise ValueError("successor.gamma must lie in [0, 1).")
    for key in ("target_world_ema_decay", "target_successor_ema_decay"):
        if not 0.0 <= float(successor.get(key, -1.0)) < 1.0:
            raise ValueError(f"successor.{key} must lie in [0, 1).")
    if not 0.0 <= float(successor.get("loss_warmup_fraction", -1.0)) < 1.0:
        raise ValueError("successor.loss_warmup_fraction must lie in [0, 1).")
    if (
        min(
            float(successor.get("planning_weight", -1.0)),
            float(successor.get("terminal_weight", -1.0)),
        )
        < 0.0
    ):
        raise ValueError("Successor planning weights cannot be negative.")
    if successor.get("clamp_successor_cost") is not True:
        raise ValueError("The first comparison clamps the successor planning cost.")

    loss = protocol.get("loss", {})
    sigreg = loss.get("sigreg", {})
    if loss.get("prediction") != "mse" or float(sigreg.get("weight", -1.0)) != 0.09:
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
    if (
        min(
            float(optimizer.get("world_model_learning_rate", 0.0)),
            float(optimizer.get("successor_learning_rate", 0.0)),
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


@dataclass(frozen=True)
class ActorFreeTrainingSchedule:
    total_scheduler_steps: int
    max_epochs: int


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
    online_model: Any, target_model: Any, encoder_input: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    online = online_model.encode(dict(encoder_input))
    with torch.no_grad():
        target = target_model.encode(dict(encoder_input))
    return online["emb"], online["act_emb"], target["emb"]


def _build_training_module(
    world_model: Any,
    protocol: dict[str, Any],
    total_steps: int,
    *,
    action_block_dim: int,
    device_image_preprocessing: bool,
):
    import lightning as pl
    import stable_worldmodel as swm

    class ActorFreeTDLeWMTrainingModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = world_model
            self.target_model = copy.deepcopy(world_model).requires_grad_(False)
            self.target_model.eval()
            successor_cfg = protocol["successor"]
            self.successor = ActorFreeSuccessorHead(
                embed_dim=int(protocol["model"]["embed_dim"]),
                action_dim=action_block_dim,
                history_size=int(protocol["sequence"]["history_frames"]),
                hidden_dim=int(successor_cfg["hidden_dim"]),
            )
            self.target_successor = self.successor.make_target()
            self.target_successor.eval()
            self.variant = str(protocol["variant"])
            self.history_size = int(protocol["sequence"]["history_frames"])
            self.gamma = float(successor_cfg["gamma"])
            self.target_world_ema_decay = float(successor_cfg["target_world_ema_decay"])
            self.target_successor_ema_decay = float(
                successor_cfg["target_successor_ema_decay"]
            )
            self.auxiliary_warmup_steps = int(
                float(successor_cfg["loss_warmup_fraction"]) * total_steps
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
            self.sigreg = swm.wm.SIGReg(
                knots=int(sigreg["knots"]), num_proj=int(sigreg["num_projections"]),
            )

        def train(self, mode: bool = True):
            super().train(mode)
            self.target_model.eval()
            self.target_successor.eval()
            return self

        def _auxiliary_scale(self) -> float:
            if self.auxiliary_warmup_steps <= 0:
                return 1.0
            return min(
                1.0, float(self.global_step + 1) / float(self.auxiliary_warmup_steps),
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

        def _forward_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
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
            )
            real_td_loss = (
                td_output.real_td_loss
                if td_output.real_td_loss is not None
                else prediction_loss.new_zeros(())
            )
            objective = protocol["joint_objective"]
            weighted_td = (
                float(objective["predicted_td_weight"]) * td_output.predicted_td_loss
                + float(objective["real_td_weight"]) * real_td_loss
            )
            auxiliary_scale = self._auxiliary_scale()
            loss = (
                prediction_loss
                + float(protocol["loss"]["sigreg"]["weight"]) * sigreg_loss
                + auxiliary_scale * weighted_td
            )
            metrics = {
                f"{stage}/loss": loss.detach(),
                f"{stage}/prediction_loss": prediction_loss.detach(),
                f"{stage}/sigreg_loss": sigreg_loss.detach(),
                f"{stage}/td_loss": weighted_td.detach(),
                f"{stage}/predicted_td_loss": td_output.predicted_td_loss.detach(),
                f"{stage}/real_td_loss": real_td_loss.detach(),
                f"{stage}/td_prediction_mean": td_output.prediction_mean.detach(),
                f"{stage}/td_target_mean": td_output.target_mean.detach(),
                f"{stage}/terminal_fraction": td_output.terminal_fraction.detach(),
                f"{stage}/td_pairs": loss.new_tensor(float(td_output.pair_count)),
                f"{stage}/td_weight_scale": loss.new_tensor(auxiliary_scale),
            }
            if episode_ids is not None:
                metrics[f"{stage}/unique_episodes_per_batch"] = loss.new_tensor(
                    float(torch.unique(episode_ids).numel())
                )
            if cache_bytes is not None:
                metrics[f"{stage}/compressed_cache_gib"] = loss.new_tensor(
                    float(cache_bytes) / 1024 ** 3
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
                self.target_model, self.model, decay=self.target_world_ema_decay,
            )
            ema_update(
                self.target_successor,
                self.successor,
                decay=self.target_successor_ema_decay,
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
                        "params": list(self.successor.parameters()),
                        "lr": float(optimizer_cfg["successor_learning_rate"]),
                    },
                ],
                weight_decay=float(optimizer_cfg["weight_decay"]),
            )
            warmup_steps = max(
                1, int(float(protocol["scheduler"]["warmup_fraction"]) * total_steps),
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


def _successor_config(
    protocol: dict[str, Any],
    *,
    action_block_dim: int,
    base_export_run_name: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    successor = protocol["successor"]
    return {
        "method": METHOD,
        "variant": protocol["variant"],
        "objective_version": OBJECTIVE_VERSION,
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
        "predicted_context_detach": protocol["joint_objective"][
            "predicted_context_detach"
        ],
        "target_world_ema_decay": float(successor["target_world_ema_decay"]),
        "target_successor_ema_decay": float(successor["target_successor_ema_decay"]),
        "planning_weight": float(successor["planning_weight"]),
        "terminal_weight": float(successor["terminal_weight"]),
        "clamp_successor_cost": bool(successor["clamp_successor_cost"]),
        "base_export_run_name": base_export_run_name,
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }


def _deployment_payload(
    module: Any,
    *,
    protocol: dict[str, Any],
    model_config: dict[str, Any],
    action_block_dim: int,
    epoch: int,
    global_step: int,
    base_export_run_name: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build one self-contained and version-locked deployment checkpoint."""

    return {
        "method": METHOD,
        "variant": protocol["variant"],
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "world_model_state_dict": module.model.state_dict(),
        "target_world_model_state_dict": module.target_model.state_dict(),
        "successor_state_dict": module.successor.state_dict(),
        "target_successor_state_dict": module.target_successor.state_dict(),
        "world_model_config": model_config,
        "successor_config": _successor_config(
            protocol,
            action_block_dim=action_block_dim,
            base_export_run_name=base_export_run_name,
            base_checkpoint_sha256=base_checkpoint_sha256,
        ),
    }


def _build_export_callback(
    run_dir: Path,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
    action_block_dim: int,
):
    import lightning as pl
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    export_config = OmegaConf.create(model_config)

    class ActorFreeTDExportCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            epoch = int(trainer.current_epoch) + 1
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
            deployment_dir = run_dir / "checkpoints" / METHOD / str(protocol["variant"])
            deployment_dir.mkdir(parents=True, exist_ok=True)
            payload = _deployment_payload(
                pl_module,
                protocol=protocol,
                model_config=model_config,
                action_block_dim=action_block_dim,
                epoch=epoch,
                global_step=int(trainer.global_step),
                base_export_run_name=base_run_name,
                base_checkpoint_sha256=_file_sha256(base_weights[0]),
            )
            torch.save(payload, deployment_dir / f"epoch_{epoch:02d}.pt")

    return ActorFreeTDExportCallback()


def _build_generator_callback(generator: torch.Generator, *, variant: str):
    import lightning as pl

    class DataLoaderGeneratorCallback(pl.Callback):
        @property
        def state_key(self) -> str:
            return f"tdwm_{METHOD}_{variant}_dataloader_generator"

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


def train_actor_free_td_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    smoke: bool = False,
    resume: str = "auto",
    max_steps: int | None = None,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Train one goal-free Actor-Free TD-LeWM variant from raw Cube data."""

    protocol = load_actor_free_td_training_protocol(protocol_path)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is not in the locked seeds {protocol['seeds']}.")
    if resume not in {"auto", "never", "required"}:
        raise ValueError("resume must be one of: auto, never, required.")
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
        num_steps=int(sequence["num_steps"]),
        frameskip=int(sequence["frame_skip"]),
        keys_to_load=list(dataset_cfg["keys_to_load"]),
        keys_to_cache=list(dataset_cfg["keys_to_cache"]),
        keys_to_merge=dict(dataset_cfg["keys_to_merge"]),
    )
    if len(dataset.lengths) != int(dataset_cfg["expected_episodes"]):
        raise ValueError("Dataset episode count differs from the protocol.")
    if int(np.asarray(dataset.lengths).sum()) != int(
        dataset_cfg["expected_transitions"]
    ):
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
    decoded_frame_store_metadata = None
    if dataset_source["format"] == "lance":
        (
            decoded_frame_store,
            decoded_frame_store_metadata,
        ) = _prepare_decoded_frame_store(protocol, dataset_source, dataset)
        dataset = StrideAwareLanceDataset(
            dataset, decoded_frame_store=decoded_frame_store,
        )
    elif os.environ.get(DECODED_FRAME_STORE_ENV) is not None:
        raise ValueError(
            f"{DECODED_FRAME_STORE_ENV} is only supported for Lance datasets."
        )

    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(
        dataset,
        [
            float(protocol["split"]["train_fraction"]),
            float(protocol["split"]["validation_fraction"]),
        ],
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

    action_dim = int(dataset.get_dim("action"))
    action_block_dim = int(sequence["frame_skip"]) * action_dim
    model_config = build_model_config(protocol, action_dim)
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
        smoke=smoke, max_steps=max_steps, train_loader_length=available_epoch_steps,
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
        action_block_dim=action_block_dim,
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
        _build_export_callback(run_dir, model_config, protocol, action_block_dim),
        _build_generator_callback(generator, variant=str(protocol["variant"])),
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
            previous_protocol.get("method") == METHOD
            and previous_protocol.get("variant") == protocol["variant"]
            and previous_protocol.get("successor", {}).get("objective_version")
            == OBJECTIVE_VERSION
            and previous.get("deployment_checkpoint_version")
            == DEPLOYMENT_CHECKPOINT_VERSION
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
    write_json(
        run_dir / "training_manifest.json",
        {
            "method": METHOD,
            "variant": protocol["variant"],
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "protocol": protocol,
            "protocol_path": str(Path(protocol_path).resolve()),
            "seed": seed,
            "dataset": dataset_manifest,
            "model": {
                "config": model_config,
                "lewm_parameters": parameter_count,
                "successor_parameters": sum(
                    parameter.numel() for parameter in module.successor.parameters()
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
        },
    )

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
        "method": METHOD,
        "variant": protocol["variant"],
        "run_dir": str(run_dir),
        "seed": seed,
        "last_checkpoint": str(last_checkpoint),
        "final_epoch": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
    }
    if torch.cuda.is_available():
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    write_json(run_dir / "training_result.json", result)
    return result


__all__ = [
    "ActorFreeTDInputs",
    "ActorFreeTrainingSchedule",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_OPTIMIZER_UPDATES",
    "METHOD",
    "OBJECTIVE_VERSION",
    "build_actor_free_td_inputs",
    "load_actor_free_td_training_protocol",
    "resolve_actor_free_training_schedule",
    "train_actor_free_td_lewm",
    "validate_actor_free_td_training_protocol",
]
