"""Explicit three-stage EffAction training with atomic, exact-step recovery.

This engine owns proposed-method heads and optimizers only. Frozen visual
latents and the pretrained LeWM action encoder remain external; their source
identities belong in the caller-supplied provenance. No episode sampling or
evaluation protocol is defined here.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import validate_frozen_lewm_action_encoder_v1
from tdwm.methods.eff_action import (
    EffActionSuccessor,
    EffActionValue,
    build_eff_action_loss,
    ema_update_eff_action,
)
from tdwm.methods.eff_action_plan import (
    EffActionPlanner,
    build_eff_action_plan_loss,
    iterate_eff_action_plan,
    project_eff_action,
)
from tdwm.training.eff_action_data import EFF_ACTION_LOSS_BATCH_KEYS

STAGES = ("gv", "stage1", "stage2")


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain finite JSON-compatible values."
        ) from error


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}.")


def _integer(value: Any, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _number(value: Any, name: str, *, minimum: float, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(value) or value < minimum or (positive and value == minimum):
        relation = ">" if positive else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}.")


def _optimizer_config(config: dict[str, Any], name: str) -> None:
    _keys(config, {"lr", "weight_decay", "betas", "eps"}, name)
    _number(config["lr"], f"{name}.lr", minimum=0, positive=True)
    _number(config["weight_decay"], f"{name}.weight_decay", minimum=0)
    _number(config["eps"], f"{name}.eps", minimum=0, positive=True)
    betas = config["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError(f"{name}.betas must contain two values.")
    for beta in betas:
        _number(beta, f"{name}.betas", minimum=0)
        if beta >= 1:
            raise ValueError(f"{name}.betas must be less than 1.")


def validate_eff_action_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require all scientific choices; missing/null pending choices fail closed."""

    config = _json_copy(config, "config")
    _keys(config, {"seed", "precision", "g", "v", "planner", *STAGES}, "config")
    _integer(config["seed"], "seed", minimum=0)
    if config["seed"] >= 2**32:
        raise ValueError("seed must be below 2**32 for NumPy compatibility.")
    if config["precision"] not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16.")
    _keys(config["g"], {"hidden_dim", "hidden_layers", "embedding_layers"}, "g")
    for key in config["g"]:
        _integer(
            config["g"][key], f"g.{key}", minimum=0 if key == "hidden_layers" else 1
        )
    if config["g"]["hidden_dim"] % 2 or config["g"]["embedding_layers"] < 2:
        raise ValueError("G requires even hidden_dim and embedding_layers >= 2.")
    _keys(config["v"], {"hidden_dim", "hidden_layers", "output_activation"}, "v")
    for key in ("hidden_dim", "hidden_layers"):
        _integer(config["v"][key], f"v.{key}", minimum=1)
    if config["v"]["output_activation"] not in {"softplus", "relu", "square"}:
        raise ValueError("Unsupported V output activation.")
    planner = config["planner"]
    _keys(
        planner,
        {
            "raw_action_dim",
            "hidden_dim",
            "hidden_layers",
            "iterations",
            "epsilon",
            "lower_bound",
            "upper_bound",
            "initialization",
            "lambda_traj",
            "lambda_eff",
        },
        "planner",
    )
    for key in ("raw_action_dim", "hidden_dim", "hidden_layers", "iterations"):
        _integer(planner[key], f"planner.{key}", minimum=1)
    if planner["raw_action_dim"] != 25:
        raise ValueError("The current action encoder requires a 25D block.")
    _number(planner["epsilon"], "planner.epsilon", minimum=0, positive=True)
    for key in ("lambda_traj", "lambda_eff"):
        _number(planner[key], f"planner.{key}", minimum=0)
    if planner["lambda_traj"] == planner["lambda_eff"] == 0:
        raise ValueError("Stage 2 cannot disable both losses.")
    for key in ("lower_bound", "upper_bound"):
        bound = np.asarray(planner[key], dtype=np.float64)
        if bound.shape not in ((), (25,)) or not np.isfinite(bound).all():
            raise ValueError(f"planner.{key} must be a finite scalar or 25D vector.")
    if np.any(np.asarray(planner["lower_bound"]) > np.asarray(planner["upper_bound"])):
        raise ValueError("Planner action bounds are reversed.")
    init = planner["initialization"]
    if not isinstance(init, dict) or init.get("distribution") not in {
        "uniform",
        "normal",
        "zeros",
    }:
        raise ValueError(
            "initialization.distribution must be uniform, normal, or zeros."
        )
    expected = (
        {"distribution", "mean", "std"}
        if init["distribution"] == "normal"
        else {"distribution"}
    )
    _keys(init, expected, "planner.initialization")
    if init["distribution"] == "normal":
        if not isinstance(init["mean"], (float, int)) or not math.isfinite(
            init["mean"]
        ):
            raise ValueError("Normal initialization mean must be finite.")
        _number(init["std"], "initialization.std", minimum=0, positive=True)
    for stage in STAGES:
        item = config[stage]
        expected = {"steps", "warmup_steps", "min_lr_ratio", "gradient_clip_norm"}
        expected |= (
            {"g_optimizer", "v_optimizer", "ema_decay"}
            if stage == "gv"
            else {"optimizer"}
        )
        if stage == "stage2":
            expected.add("optimizer_transition")
        _keys(item, expected, stage)
        _integer(item["steps"], f"{stage}.steps", minimum=1)
        _integer(item["warmup_steps"], f"{stage}.warmup_steps", minimum=0)
        if item["warmup_steps"] > item["steps"]:
            raise ValueError(f"{stage}.warmup_steps exceeds its training budget.")
        _number(item["min_lr_ratio"], f"{stage}.min_lr_ratio", minimum=0)
        if item["min_lr_ratio"] > 1:
            raise ValueError(f"{stage}.min_lr_ratio must not exceed 1.")
        if item["gradient_clip_norm"] is not None:
            _number(
                item["gradient_clip_norm"],
                f"{stage}.gradient_clip_norm",
                minimum=0,
                positive=True,
            )
        for key in ("g_optimizer", "v_optimizer") if stage == "gv" else ("optimizer",):
            _optimizer_config(item[key], f"{stage}.{key}")
    _number(config["gv"]["ema_decay"], "gv.ema_decay", minimum=0)
    if config["gv"]["ema_decay"] >= 1:
        raise ValueError("gv.ema_decay must be less than 1.")
    if config["stage2"]["optimizer_transition"] not in {"reset", "carry"}:
        raise ValueError("stage2.optimizer_transition must be reset or carry.")
    return config


def eff_action_learning_rate_factor(
    completed_steps: int, *, total_steps: int, warmup_steps: int, min_lr_ratio: float
) -> float:
    """Learning-rate multiplier for the next update (zero-based step).

    Warmup reaches 1 on its last update. Cosine starts at 1 on the first
    post-warmup update and reaches min_lr_ratio on the final budgeted update
    when there are at least two post-warmup updates.
    """

    if not 0 <= completed_steps < total_steps or not 0 <= warmup_steps <= total_steps:
        raise ValueError("Schedule step or warmup lies outside the configured budget.")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must lie in [0, 1].")
    if completed_steps < warmup_steps:
        return (completed_steps + 1) / warmup_steps
    progress = (completed_steps - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _adamw(module: nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    # Explicit scalar implementation avoids a device-dependent foreach/fused
    # choice during exact recovery and keeps optimizer behavior auditable.
    return torch.optim.AdamW(
        module.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
        foreach=False,
        fused=False,
    )


class EffActionTrainer:
    """Train G/V, then an independently initialized P in two ordered stages.

    The caller samples through ``sample_rng`` and passes that batch to
    ``train_step``. Save the completed stage-1 checkpoint before calling
    ``begin_planner_stage(2)`` when retaining the BC-only deployment ablation.
    """

    def __init__(
        self,
        action_encoder: nn.Module,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
        device: str | torch.device,
    ) -> None:
        self.config = validate_eff_action_training_config(config)
        self.provenance = _json_copy(provenance, "provenance")
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise ValueError("Nonempty external encoder/data provenance is required.")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("This runtime supports audited CPU or CUDA execution.")
        self.action_encoder = (
            action_encoder.to(self.device).requires_grad_(False).eval()
        )
        validate_frozen_lewm_action_encoder_v1(self.action_encoder)
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.sample_rng = np.random.default_rng(seed)
        self.g = EffActionSuccessor(**self.config["g"]).to(self.device)
        self.v = EffActionValue(**self.config["v"]).to(self.device)
        self.target_g, self.target_v = self.g.make_target(), self.v.make_target()
        self.g_optimizer = _adamw(self.g, self.config["gv"]["g_optimizer"])
        self.v_optimizer = _adamw(self.v, self.config["gv"]["v_optimizer"])
        self.planner: EffActionPlanner | None = None
        self.planner_optimizer: torch.optim.AdamW | None = None
        self.stage = "gv"
        self.counters = {stage: 0 for stage in STAGES}

    @property
    def is_stage_complete(self) -> bool:
        return self.counters[self.stage] == self.config[self.stage]["steps"]

    @property
    def global_step(self) -> int:
        return sum(self.counters.values())

    def _create_planner(self) -> None:
        config = self.config["planner"]
        self.planner = EffActionPlanner(
            **{
                key: config[key]
                for key in ("raw_action_dim", "hidden_dim", "hidden_layers")
            }
        ).to(self.device)

    def _apply_training_modes(self) -> None:
        for module in (self.g, self.v):
            module.requires_grad_(self.stage == "gv").train(self.stage == "gv")
            if self.stage != "gv":
                module.zero_grad(set_to_none=True)
        self.action_encoder.requires_grad_(False).eval()
        self.target_g.requires_grad_(False).eval()
        self.target_v.requires_grad_(False).eval()
        if self.planner is not None:
            self.planner.requires_grad_(True).train()

    def begin_planner_stage(self, stage: int) -> None:
        if isinstance(stage, bool) or not isinstance(stage, int) or stage not in (1, 2):
            raise ValueError("Planner stage must be 1 or 2.")
        preceding = "gv" if stage == 1 else "stage1"
        if self.stage != preceding or not self.is_stage_complete:
            raise RuntimeError(
                f"Complete {preceding} before beginning planner stage {stage}."
            )
        if stage == 1:
            self._create_planner()
        old_optimizer = self.planner_optimizer
        self.stage = f"stage{stage}"
        config = self.config[self.stage]["optimizer"]
        self.planner_optimizer = _adamw(self.planner, config)
        if stage == 2 and self.config["stage2"]["optimizer_transition"] == "carry":
            self.planner_optimizer.load_state_dict(old_optimizer.state_dict())
            for group in self.planner_optimizer.param_groups:
                group.update(
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    betas=tuple(config["betas"]),
                    eps=config["eps"],
                )
        self._apply_training_modes()

    def initialize_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Initialize without examining demonstration action values."""

        config = self.config["planner"]
        shape = state.shape[:-1] + (config["raw_action_dim"],)
        lower = torch.as_tensor(
            config["lower_bound"], device=state.device, dtype=state.dtype
        )
        upper = torch.as_tensor(
            config["upper_bound"], device=state.device, dtype=state.dtype
        )
        init = config["initialization"]
        if init["distribution"] == "uniform":
            action = lower + torch.rand(
                shape, device=state.device, dtype=state.dtype
            ) * (upper - lower)
        elif init["distribution"] == "normal":
            action = (
                torch.randn(shape, device=state.device, dtype=state.dtype) * init["std"]
                + init["mean"]
            )
        else:
            action = torch.zeros(shape, device=state.device, dtype=state.dtype)
        return project_eff_action(action, lower_bound=lower, upper_bound=upper)

    def _set_learning_rates(self) -> dict[str, float]:
        config = self.config[self.stage]
        factor = eff_action_learning_rate_factor(
            self.counters[self.stage],
            total_steps=config["steps"],
            warmup_steps=config["warmup_steps"],
            min_lr_ratio=config["min_lr_ratio"],
        )
        entries = (
            (
                ("g", self.g_optimizer, config["g_optimizer"]),
                ("v", self.v_optimizer, config["v_optimizer"]),
            )
            if self.stage == "gv"
            else (("planner", self.planner_optimizer, config["optimizer"]),)
        )
        result = {}
        for name, optimizer, options in entries:
            learning_rate = options["lr"] * factor
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            result[f"lr_{name}"] = learning_rate
        return result

    def _gradient_norm(self, module: nn.Module) -> float:
        bound = self.config[self.stage]["gradient_clip_norm"]
        norm = nn.utils.clip_grad_norm_(
            module.parameters(),
            math.inf if bound is None else bound,
            error_if_nonfinite=True,
            foreach=False,
        )
        return float(norm.detach())

    def _batch_metrics(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        valid = batch["valid_mask"]
        count = int(valid.sum())
        direct = int((valid & batch["direct_branch"]).sum())
        terminal = int((valid & batch["goal_terminal"]).sum())
        return {
            "stage": self.stage,
            "step": self.counters[self.stage],
            "global_step": self.global_step,
            "valid_count": count,
            "direct_count": direct,
            "direct_frac": direct / count,
            "terminal_frac": terminal / count,
        }

    def _gv_diagnostics(self, output: Any) -> dict[str, float]:
        valid = output.valid_mask
        return {
            "v_prediction_mean": float(
                output.v_prediction[valid].detach().float().mean()
            ),
            "v_target_mean": float(output.v_target[valid].float().mean()),
            "g_target_norm_mean": float(
                torch.linalg.vector_norm(output.g_target[valid].float(), dim=-1).mean()
            ),
        }

    def _planner_diagnostics(self, plan: Any) -> dict[str, float]:
        action = plan.action.detach()
        config = self.config["planner"]
        lower = torch.as_tensor(
            config["lower_bound"], device=action.device, dtype=action.dtype
        )
        upper = torch.as_tensor(
            config["upper_bound"], device=action.device, dtype=action.dtype
        )
        return {
            "initial_cost": float(plan.costs[0].detach().float().mean()),
            "final_cost": float(plan.final_cost.detach().float().mean()),
            "action_delta_norm": float(
                torch.linalg.vector_norm(
                    action.float() - plan.reference_action.float(), dim=-1
                ).mean()
            ),
            "bound_fraction": float(
                ((action <= lower) | (action >= upper)).float().mean()
            ),
            "action_min": float(action.min()),
            "action_max": float(action.max()),
        }

    def train_step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        if self.is_stage_complete:
            raise RuntimeError(f"The {self.stage} training budget is already complete.")
        self._apply_training_modes()
        missing = set(EFF_ACTION_LOSS_BATCH_KEYS) - batch.keys()
        if missing:
            raise ValueError(f"Training batch is missing {sorted(missing)}.")
        batch = {key: batch[key].to(self.device) for key in EFF_ACTION_LOSS_BATCH_KEYS}
        for key in ("valid_mask", "direct_branch", "goal_terminal"):
            if (
                batch[key].dtype != torch.bool
                or batch[key].shape != batch["state"].shape[:-1]
            ):
                raise ValueError(
                    f"{key} must be boolean and match the state leading shape."
                )
        metrics = self._set_learning_rates()
        enabled = self.config["precision"] == "bfloat16"
        if self.stage == "gv":
            self.g_optimizer.zero_grad(set_to_none=True)
            self.v_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=enabled
            ):
                output = build_eff_action_loss(
                    self.g,
                    self.v,
                    self.target_g,
                    self.target_v,
                    self.action_encoder,
                    **batch,
                )
            output.loss.backward()
            metrics["gradient_norm_g"] = self._gradient_norm(self.g)
            metrics["gradient_norm_v"] = self._gradient_norm(self.v)
            self.g_optimizer.step()
            self.v_optimizer.step()
            rate = 1.0 - self.config["gv"]["ema_decay"]
            ema_update_eff_action(self.target_g, self.g, rate=rate)
            ema_update_eff_action(self.target_v, self.v, rate=rate)
            for key in (
                "loss",
                "g_loss",
                "v_loss",
                "g_direct_loss",
                "g_td_loss",
                "v_direct_loss",
                "v_td_loss",
            ):
                metrics[key] = float(getattr(output, key).detach())
            metrics.update(self._gv_diagnostics(output))
        else:
            self.planner_optimizer.zero_grad(set_to_none=True)
            valid = batch["valid_mask"]
            if valid.dtype != torch.bool or not bool(valid.any()):
                raise ValueError(
                    "Planner training requires at least one valid example."
                )
            if valid.shape != batch["state"].shape[:-1]:
                raise ValueError("valid_mask must match the state leading shape.")
            state, goal, task, action = (
                batch[key][valid] for key in ("state", "goal", "task", "raw_action")
            )
            config = self.config["planner"]
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=enabled
            ):
                plan = iterate_eff_action_plan(
                    self.planner,
                    self.g,
                    self.v,
                    self.action_encoder,
                    state=state,
                    goal=goal,
                    task=task,
                    initial_action=self.initialize_actions(state),
                    iterations=config["iterations"],
                    epsilon=config["epsilon"],
                    lower_bound=config["lower_bound"],
                    upper_bound=config["upper_bound"],
                    track_grad=True,
                )
                output = build_eff_action_plan_loss(
                    plan,
                    dataset_action=action,
                    stage=int(self.stage[-1]),
                    lambda_traj=config["lambda_traj"],
                    lambda_eff=config["lambda_eff"],
                    valid_mask=torch.ones(
                        state.shape[:-1], device=self.device, dtype=torch.bool
                    ),
                )
            output.loss.backward()
            metrics["gradient_norm_planner"] = self._gradient_norm(self.planner)
            self.planner_optimizer.step()
            metrics.update(
                loss=float(output.loss.detach()),
                trajectory_loss=float(output.trajectory_loss.detach()),
                efficiency_loss=float(output.efficiency_loss.detach()),
            )
            metrics.update(self._planner_diagnostics(plan))
        self.counters[self.stage] += 1
        metrics.update(self._batch_metrics(batch))
        return metrics

    def evaluate_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Read-only validation with fixed reference initialization and RNG restore.

        P uses the configured training seed for validation reference actions,
        so a fixed validation batch has fixed initialization at every audit.
        Every head mode, RNG stream, counter, optimizer and gradient remains
        unchanged. The caller separately owns its validation sample generator.
        """

        missing = set(EFF_ACTION_LOSS_BATCH_KEYS) - batch.keys()
        if missing:
            raise ValueError(f"Validation batch is missing {sorted(missing)}.")
        batch = {key: batch[key].to(self.device) for key in EFF_ACTION_LOSS_BATCH_KEYS}
        rng = self._rng_state()
        modules = [self.g, self.v, self.target_g, self.target_v, self.action_encoder]
        if self.planner is not None:
            modules.append(self.planner)
        modes = {part: part.training for module in modules for part in module.modules()}
        try:
            for module in modules:
                module.eval()
            enabled = self.config["precision"] == "bfloat16"
            if self.stage == "gv":
                with (
                    torch.no_grad(),
                    torch.autocast(
                        self.device.type, dtype=torch.bfloat16, enabled=enabled
                    ),
                ):
                    output = build_eff_action_loss(
                        self.g,
                        self.v,
                        self.target_g,
                        self.target_v,
                        self.action_encoder,
                        **batch,
                    )
                metrics = {
                    key: float(getattr(output, key))
                    for key in (
                        "loss",
                        "g_loss",
                        "v_loss",
                        "g_direct_loss",
                        "g_td_loss",
                        "v_direct_loss",
                        "v_td_loss",
                    )
                }
                metrics.update(self._gv_diagnostics(output))
            else:
                valid = batch["valid_mask"]
                if (
                    valid.dtype != torch.bool
                    or valid.shape != batch["state"].shape[:-1]
                    or not bool(valid.any())
                ):
                    raise ValueError(
                        "Planner validation requires a correctly shaped nonempty valid mask."
                    )
                state, goal, task, action = (
                    batch[key][valid] for key in ("state", "goal", "task", "raw_action")
                )
                config = self.config["planner"]
                torch.manual_seed(self.config["seed"])
                with torch.autocast(
                    self.device.type, dtype=torch.bfloat16, enabled=enabled
                ):
                    plan = iterate_eff_action_plan(
                        self.planner,
                        self.g,
                        self.v,
                        self.action_encoder,
                        state=state,
                        goal=goal,
                        task=task,
                        initial_action=self.initialize_actions(state),
                        iterations=config["iterations"],
                        epsilon=config["epsilon"],
                        lower_bound=config["lower_bound"],
                        upper_bound=config["upper_bound"],
                        track_grad=False,
                    )
                    output = build_eff_action_plan_loss(
                        plan,
                        dataset_action=action,
                        stage=int(self.stage[-1]),
                        lambda_traj=config["lambda_traj"],
                        lambda_eff=config["lambda_eff"],
                        valid_mask=torch.ones(
                            state.shape[:-1], device=self.device, dtype=torch.bool
                        ),
                    )
                metrics = {
                    key: float(getattr(output, key))
                    for key in ("loss", "trajectory_loss", "efficiency_loss")
                }
                metrics.update(self._planner_diagnostics(plan))
            metrics.update(self._batch_metrics(batch))
            return metrics
        finally:
            for module, training in modes.items():
                module.train(training)
            self._restore_rng(rng)

    def _rng_state(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        return {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            "numpy_global": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "numpy_sampling": copy.deepcopy(self.sample_rng.bit_generator.state),
            "python": random.getstate(),
        }

    def _restore_rng(self, state: dict[str, Any]) -> None:
        torch.random.set_rng_state(state["torch_cpu"].cpu())
        if self.device.type == "cuda":
            if state["torch_cuda"] is None:
                raise ValueError("CUDA recovery requires the saved CUDA RNG state.")
            torch.cuda.set_rng_state(state["torch_cuda"].cpu(), self.device)
        numpy_state = state["numpy_global"]
        np.random.set_state(
            (
                numpy_state[0],
                np.asarray(numpy_state[1], dtype=np.uint32),
                *numpy_state[2:],
            )
        )
        self.sample_rng.bit_generator.state = state["numpy_sampling"]
        random.setstate(state["python"])

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a training snapshot without copying the frozen E/e weights."""

        return {
            "format": "eff_action_training",
            "format_version": 1,
            "device_type": self.device.type,
            "config": copy.deepcopy(self.config),
            "provenance": copy.deepcopy(self.provenance),
            "stage": self.stage,
            "counters": self.counters.copy(),
            "planner_initialized": self.planner is not None,
            "planner_trained": self.counters["stage1"] + self.counters["stage2"] > 0,
            "models": {
                "g": self.g.state_dict(),
                "v": self.v.state_dict(),
                "target_g": self.target_g.state_dict(),
                "target_v": self.target_v.state_dict(),
                "planner": self.planner.state_dict()
                if self.planner is not None
                else None,
            },
            "optimizers": {
                "g": self.g_optimizer.state_dict(),
                "v": self.v_optimizer.state_dict(),
                "planner": self.planner_optimizer.state_dict()
                if self.planner_optimizer is not None
                else None,
            },
            "rng": self._rng_state(),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Flush a sibling temporary file and atomically replace the destination."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                torch.save(self.checkpoint_state(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        action_encoder: nn.Module,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
        device: str | torch.device,
    ) -> EffActionTrainer:
        """Strict resume; restore RNG only after model/optimizer construction."""

        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state.get("format") != "eff_action_training"
            or state.get("format_version") != 1
        ):
            raise ValueError("Unsupported EffAction checkpoint format.")
        config = validate_eff_action_training_config(config)
        if state["config"] != config:
            raise ValueError(
                "Checkpoint config does not match the requested training config."
            )
        if state["provenance"] != _json_copy(provenance, "provenance"):
            raise ValueError(
                "Checkpoint provenance does not match the external data/encoders."
            )
        if state["device_type"] != torch.device(device).type:
            raise ValueError("Exact training recovery requires the same device type.")
        stage, counters = state["stage"], state["counters"]
        if stage not in STAGES or set(counters) != set(STAGES):
            raise ValueError("Invalid checkpoint stage/counters.")
        for key, count in counters.items():
            _integer(count, f"counters.{key}", minimum=0)
            if count > config[key]["steps"]:
                raise ValueError(
                    "Checkpoint counter exceeds its configured stage budget."
                )
        stage_index = STAGES.index(stage)
        for index, key in enumerate(STAGES):
            if index < stage_index and counters[key] != config[key]["steps"]:
                raise ValueError("Checkpoint skipped an incomplete preceding stage.")
            if index > stage_index and counters[key] != 0:
                raise ValueError("Checkpoint has updates from a future stage.")
        initialized = stage != "gv"
        trained = counters["stage1"] + counters["stage2"] > 0
        if (
            state["planner_initialized"] != initialized
            or state["planner_trained"] != trained
        ):
            raise ValueError("Checkpoint planner training status is inconsistent.")
        if (state["models"]["planner"] is not None) != initialized or (
            state["optimizers"]["planner"] is not None
        ) != initialized:
            raise ValueError(
                "Checkpoint planner weights/optimizer are inconsistent with its stage."
            )
        trainer = cls(action_encoder, config, provenance, device)
        for key in ("g", "v", "target_g", "target_v"):
            getattr(trainer, key).load_state_dict(state["models"][key], strict=True)
        trainer.g_optimizer.load_state_dict(state["optimizers"]["g"])
        trainer.v_optimizer.load_state_dict(state["optimizers"]["v"])
        trainer.stage, trainer.counters = stage, counters.copy()
        if initialized:
            trainer._create_planner()
            trainer.planner.load_state_dict(state["models"]["planner"], strict=True)
            trainer.planner_optimizer = _adamw(
                trainer.planner, config[stage]["optimizer"]
            )
            trainer.planner_optimizer.load_state_dict(state["optimizers"]["planner"])
        trainer._apply_training_modes()
        trainer._restore_rng(state["rng"])
        return trainer


__all__ = [
    "EffActionTrainer",
    "eff_action_learning_rate_factor",
    "validate_eff_action_training_config",
]
