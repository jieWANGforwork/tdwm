"""Planning adapter for goal-free Actor-Free TD-LeWM checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm import (
    SUPPORTED_VARIANTS,
    ActorFreeSuccessorHead,
)
from tdwm.methods.successor_geometry import latent_goal_cost, successor_goal_cost

METHOD = "actor_free_td_lewm"
OBJECTIVE_VERSION = 1
GOAL_OBJECTIVE_VERSION = 2
IMAGINARY_OBJECTIVE_VERSION = 3
DEPLOYMENT_CHECKPOINT_VERSION = 1
GOAL_VARIANT = "goal_hybrid"
IMAGINARY_VARIANT = "imaginary_hybrid"


def _objective_version_for_variant(variant: str) -> int:
    if variant == GOAL_VARIANT:
        return GOAL_OBJECTIVE_VERSION
    if variant == IMAGINARY_VARIANT:
        return IMAGINARY_OBJECTIVE_VERSION
    return OBJECTIVE_VERSION


class ActorFreeTDLeWM(nn.Module):
    """Splice explicit LeWM costs with a goal-free TD successor tail.

    For an ``H``-action candidate, LeWM explicitly scores states reached by the
    first ``H-1`` actions.  The successor is queried from the history ending
    after action ``H-1``, with action ``H`` as its explicit current-action
    input; consequently no transition is counted by both terms.
    """

    def __init__(
        self,
        world_model: nn.Module,
        successor: ActorFreeSuccessorHead,
        *,
        gamma: float,
        clamp_tail_cost: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        self.world_model = world_model
        self.successor = successor
        self.gamma = float(gamma)
        self.clamp_tail_cost = bool(clamp_tail_cost)

    @property
    def history_size(self) -> int:
        return self.successor.history_size

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        return self.world_model.encode(info)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.world_model.predict(emb, act_emb)

    def rollout(
        self,
        info: dict[str, Any],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, Any]:
        return self.world_model.rollout(
            info,
            action_sequence,
            history_size=history_size or self.history_size,
        )

    def criterion(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.get_cost(info_dict, action_candidates)

    def get_cost(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        if "goal" not in info_dict and "goal_emb" not in info_dict:
            raise AssertionError("goal not in info_dict")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, dim)."
            )
        if action_candidates.shape[-1] != self.successor.action_dim:
            raise ValueError("Candidate action blocks have the wrong dimension.")
        batch, samples, horizon = action_candidates.shape[:3]
        if horizon <= 0:
            raise ValueError("The planning horizon must be positive.")

        observed_frames = self._observed_frames(info_dict)
        rollout_info = self.world_model.rollout(
            info_dict,
            action_candidates,
            history_size=self.history_size,
        )
        predicted = rollout_info.get("predicted_emb")
        if predicted is None or predicted.ndim != 4:
            raise ValueError(
                "LeWM rollout must return predicted_emb with shape "
                "(batch, samples, time, latent_dim)."
            )
        if predicted.shape[:2] != (batch, samples):
            raise ValueError("LeWM rollout does not match the candidate batch.")
        if predicted.shape[-1] != self.successor.embed_dim:
            raise ValueError("LeWM and successor latent dimensions differ.")
        if predicted.shape[-2] < observed_frames:
            raise ValueError("LeWM rollout contains fewer than the observed frames.")
        future = predicted[..., observed_frames:, :]
        if future.shape[-2] != horizon:
            raise ValueError(
                "LeWM rollout future length does not match the CEM horizon: "
                f"{future.shape[-2]} != {horizon}."
            )

        goal = self._goal_for_samples(info_dict, batch=batch, samples=samples)
        explicit_future = future[..., :-1, :]
        if explicit_future.shape[-2]:
            stage_cost = latent_goal_cost(explicit_future, goal.unsqueeze(-2))
            discounts = torch.pow(
                stage_cost.new_tensor(self.gamma),
                torch.arange(stage_cost.shape[-1], device=stage_cost.device),
            )
            explicit_cost = (1.0 - self.gamma) * (stage_cost * discounts).sum(dim=-1)
        else:
            explicit_cost = future.new_zeros(batch, samples)

        # Exclude the state reached by the final action from the head input.  It
        # is the immediate feature predicted by the final-action successor.
        before_final = predicted[..., : observed_frames + horizon - 1, :]
        tail_history = self._pad_latent_history(before_final)
        previous_actions = self._tail_previous_actions(
            info_dict,
            action_candidates,
            batch=batch,
            samples=samples,
        )
        final_action = action_candidates[..., -1, :]
        successor = self.successor(
            tail_history,
            previous_actions,
            final_action,
        )
        tail_cost = successor_goal_cost(successor, goal)
        if self.clamp_tail_cost:
            tail_cost = tail_cost.clamp_min(0.0)
        return explicit_cost + (self.gamma ** (horizon - 1)) * tail_cost

    @staticmethod
    def _observed_frames(info: dict[str, Any]) -> int:
        pixels = info.get("pixels")
        if torch.is_tensor(pixels) and pixels.ndim >= 3:
            return int(pixels.shape[2])
        embeddings = info.get("emb")
        if torch.is_tensor(embeddings) and embeddings.ndim in (3, 4):
            return int(embeddings.shape[-2])
        raise ValueError("pixels or cached emb is required to infer observed history.")

    def _pad_latent_history(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] != self.successor.embed_dim:
            raise ValueError("Unexpected latent dimension.")
        available = int(latents.shape[-2])
        if available <= 0:
            raise ValueError("At least one latent is required for successor history.")
        if available >= self.history_size:
            return latents[..., -self.history_size :, :]
        padding = latents[..., :1, :].expand(
            *latents.shape[:-2], self.history_size - available, latents.shape[-1]
        )
        return torch.cat((padding, latents), dim=-2)

    def _tail_previous_actions(
        self,
        info: dict[str, Any],
        action_candidates: torch.Tensor,
        *,
        batch: int,
        samples: int,
    ) -> torch.Tensor:
        required = self.history_size - 1
        if required == 0:
            return action_candidates.new_empty(
                batch, samples, 0, self.successor.action_dim
            )

        candidate_prefix = action_candidates[..., :-1, :]
        history = info.get("action_history")
        if history is None:
            historical = action_candidates.new_empty(
                batch, samples, 0, self.successor.action_dim
            )
        else:
            historical = history.to(
                device=action_candidates.device, dtype=action_candidates.dtype
            )
            if historical.ndim == 3:
                if historical.shape[0] != batch:
                    raise ValueError("action_history has the wrong batch size.")
                historical = historical.unsqueeze(1).expand(-1, samples, -1, -1)
            elif historical.ndim == 4:
                if historical.shape[0] != batch:
                    raise ValueError("action_history has the wrong batch size.")
                if historical.shape[1] == 1:
                    historical = historical.expand(-1, samples, -1, -1)
                elif historical.shape[1] != samples:
                    raise ValueError("action_history has the wrong sample count.")
            else:
                raise ValueError(
                    "action_history must have shape (batch, time, action_dim) or "
                    "(batch, samples, time, action_dim)."
                )
            if historical.shape[-1] != self.successor.action_dim:
                raise ValueError("action_history has the wrong action dimension.")

        available = torch.cat((historical, candidate_prefix), dim=-2)
        if available.shape[-2] >= required:
            return available[..., -required:, :]
        padding = action_candidates.new_zeros(
            batch,
            samples,
            required - available.shape[-2],
            self.successor.action_dim,
        )
        return torch.cat((padding, available), dim=-2)

    def _goal_for_samples(
        self, info: dict[str, Any], *, batch: int, samples: int
    ) -> torch.Tensor:
        goal = self._get_or_encode_goal(info)
        while goal.ndim > 2:
            goal = goal[..., -1, :]
        if goal.ndim != 2 or goal.shape[0] != batch:
            raise ValueError("Expected one goal embedding per environment.")
        if goal.shape[-1] != self.successor.embed_dim:
            raise ValueError("Goal and successor latent dimensions differ.")
        return goal.unsqueeze(1).expand(batch, samples, -1)

    def _get_or_encode_goal(self, info: dict[str, Any]) -> torch.Tensor:
        if "goal_emb" in info:
            return info["goal_emb"]
        goal_info = {
            key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)
        }
        goal_info["pixels"] = goal_info["goal"]
        for key in list(goal_info):
            if key.startswith("goal_"):
                goal_info[key[len("goal_") :]] = goal_info.pop(key)
        goal_info.pop("action", None)
        goal_info.pop("action_history", None)
        encoded = self.world_model.encode(goal_info)["emb"]
        info["goal_emb"] = encoded
        return encoded


def load_actor_free_td_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, ActorFreeSuccessorHead, dict[str, Any], dict[str, Any]]:
    """Restore the jointly exported LeWM and actor-free successor."""

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("method") != METHOD:
        raise ValueError("The checkpoint is not Actor-Free TD-LeWM.")
    payload_variant = payload.get("variant")
    if payload_variant not in SUPPORTED_VARIANTS:
        raise ValueError("Checkpoint contains an unsupported TD variant.")
    expected_objective_version = _objective_version_for_variant(payload_variant)
    if payload.get("objective_version") != expected_objective_version:
        raise ValueError("Unsupported Actor-Free TD-LeWM objective version.")
    if payload.get("deployment_checkpoint_version") != DEPLOYMENT_CHECKPOINT_VERSION:
        raise ValueError("Unsupported Actor-Free TD-LeWM deployment checkpoint.")
    required = {
        "successor_state_dict",
        "successor_config",
        "world_model_state_dict",
        "world_model_config",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Actor-Free TD-LeWM checkpoint is missing {sorted(missing)}.")

    config = dict(payload["successor_config"])
    required_config = {
        "embed_dim",
        "action_dim",
        "history_size",
        "hidden_dim",
        "gamma",
        "variant",
    }
    missing_config = required_config - config.keys()
    if missing_config:
        raise ValueError(f"successor_config is missing {sorted(missing_config)}.")
    if config["variant"] not in SUPPORTED_VARIANTS:
        raise ValueError("Checkpoint contains an unsupported TD variant.")
    if config["variant"] != payload_variant:
        raise ValueError("Checkpoint and successor_config variants differ.")
    metadata_checks = {
        "method": METHOD,
        "objective_version": expected_objective_version,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
    }
    for key, expected in metadata_checks.items():
        if config.get(key) != expected:
            raise ValueError(f"successor_config.{key} must be {expected!r}.")
    if not 0.0 <= float(config["gamma"]) < 1.0:
        raise ValueError("Checkpoint gamma must lie in [0, 1).")
    semantic_checks = {
        "architecture": "actor_free_successor_head",
        "feature_basis": "augmented_latent_squared_distance",
        "goal_conditioning": "none",
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "actor": "none",
        "reward": "none",
        "predicted_context_detach": payload_variant == "serial_decoupled",
    }
    for key, expected in semantic_checks.items():
        if config.get(key) != expected:
            raise ValueError(f"successor_config.{key} must be {expected!r}.")
    if payload_variant == GOAL_VARIANT:
        goal_semantics = {
            "goal_readout_training": True,
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "goal_offset_weighting": "uniform_per_transition",
            "goal_terminal_condition": "dataset_terminal_or_next_state_is_goal",
            "goal_readout_branches": ["real_context", "predicted_context"],
            "goal_readout_precision": "float32",
            "goal_cost": "normalized_discounted_latent_mse",
            "goal_enters_successor_head": False,
            "predicted_goal_td_weight": 1.0,
            "real_goal_td_weight": 1.0,
        }
        for key, expected in goal_semantics.items():
            if config.get(key) != expected:
                raise ValueError(f"successor_config.{key} must be {expected!r}.")
    if payload_variant == IMAGINARY_VARIANT:
        imaginary_semantics = {
            "immediate_feature_source": "real_ema_next_latent",
            "bootstrap_state_source": ("ema_lewm_predicted_next_from_real_ema_history"),
            "imaginary_horizon": 1,
            "imaginary_predictor_gradient": "target_ema_stop_gradient",
        }
        for key, expected in imaginary_semantics.items():
            if config.get(key) != expected:
                raise ValueError(f"successor_config.{key} must be {expected!r}.")

    successor = ActorFreeSuccessorHead(
        embed_dim=int(config["embed_dim"]),
        action_dim=int(config["action_dim"]),
        history_size=int(config["history_size"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    successor.load_state_dict(payload["successor_state_dict"], strict=True)

    import hydra
    from omegaconf import OmegaConf

    world_model = hydra.utils.instantiate(
        OmegaConf.create(payload["world_model_config"])
    )
    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    world_model.eval().requires_grad_(False)
    successor.eval().requires_grad_(False)
    return world_model, successor, config, payload


def make_actor_free_td_policy(
    *,
    world_model: nn.Module,
    successor: ActorFreeSuccessorHead,
    planning: dict[str, Any],
    gamma: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    clamp_tail_cost: bool = True,
):
    """Build the unchanged Stable World Model CEM policy around this adapter."""

    import stable_worldmodel as swm

    wrapped = ActorFreeTDLeWM(
        world_model,
        successor,
        gamma=gamma,
        clamp_tail_cost=clamp_tail_cost,
    ).to(device)
    wrapped.eval().requires_grad_(False)
    solver = swm.solver.CEMSolver(
        model=wrapped,
        batch_size=planning["solver_batch_size"],
        num_samples=planning["candidates"],
        var_scale=planning["initial_variance"],
        n_steps=planning["iterations"],
        topk=planning["elites"],
        device=device,
        seed=planning["planning_seed"],
    )
    config = swm.PlanConfig(
        horizon=planning["horizon"],
        receding_horizon=planning["receding_horizon"],
        history_len=planning.get(
            "plan_config_history_len", planning.get("history_len", 1)
        ),
        action_block=planning["action_block"],
        warm_start=planning["warm_start"],
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


__all__ = [
    "ActorFreeTDLeWM",
    "load_actor_free_td_checkpoint",
    "make_actor_free_td_policy",
]
