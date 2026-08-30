"""Checkpoint and CEM planning support for Actor-Free TD-LeWM V0.

V0 is deliberately separate from the V-1 history successor adapter.  Its
single symmetric TD-JEPA predictor consumes one frozen LeWM state, one
normalized raw five-action block, and one task vector.  The predictor output is
scored directly with the task; there is no actor, LeWM action encoder, ensemble
mean, or minimum reduction in this adapter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v0 import (
    V0_ACTION_DIM,
    V0_OUTPUT_DIM,
    V0_STATE_DIM,
    V0_TASK_DIM,
    ActorFreeTDJEPAPredictorV0,
    project_tasks_to_sphere_v0,
)

METHOD_FAMILY = "actor_free_td_lewm_v0"
IMPLEMENTATION_VERSION = "v0"
OBJECTIVE_VERSION = 0
DEPLOYMENT_CHECKPOINT_VERSION = 1
ACTION_BLOCK_STEPS = 5
LEWM_HISTORY_SIZE = 3
SCORE_MODES = frozenset({"f_only", "g_only", "f_plus_g"})


@dataclass(frozen=True)
class FrozenActorFreeTDV0MethodSpec:
    """Checkpoint identity and objective-specific validation for one V0 method."""

    method: str
    variant: str
    display_name: str
    objective_keys: tuple[str, ...]
    validate_method_config: Callable[[Mapping[str, Any]], None]


def require_exact_values(
    values: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"{label}.{key} must be {expected_value!r}.")


def require_positive_float(
    values: Mapping[str, Any], key: str, *, label: str
) -> float:
    try:
        value = float(values.get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.{key} must be finite and positive.") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label}.{key} must be finite and positive.")
    return value


def _positive_integer(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"predictor_config.{key} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"predictor_config.{key} must be a positive integer."
        ) from error
    if integer <= 0 or integer != value:
        raise ValueError(f"predictor_config.{key} must be a positive integer.")
    return integer


def validate_frozen_actor_free_td_v0_payload(
    payload: Mapping[str, Any],
    *,
    spec: FrozenActorFreeTDV0MethodSpec,
) -> dict[str, Any]:
    """Validate the deployment contract without instantiating either model."""

    require_exact_values(
        payload,
        {
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        },
        label="checkpoint",
    )
    required = {
        "predictor_state_dict",
        "predictor_config",
        "world_model_state_dict",
        "world_model_config",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{spec.display_name} checkpoint is missing {sorted(missing)}.")

    config_value = payload["predictor_config"]
    if not isinstance(config_value, Mapping):
        raise ValueError("checkpoint.predictor_config must be a mapping.")
    config = dict(config_value)
    required_config = {
        "method",
        "method_family",
        "variant",
        "implementation_version",
        "objective_version",
        "deployment_checkpoint_version",
        "state_dim",
        "action_dim",
        "task_dim",
        "output_dim",
        "hidden_dim",
        "hidden_layers",
        "embedding_layers",
        "gamma",
        "num_parallel",
        "task_sampling",
        "joint_objective",
        "pretrained_world_model",
    }
    missing_config = required_config - config.keys()
    if missing_config:
        raise ValueError(f"predictor_config is missing {sorted(missing_config)}.")

    require_exact_values(
        config,
        {
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "architecture": "td_jepa_forward_map_v0",
            "state_dim": V0_STATE_DIM,
            "action_dim": V0_ACTION_DIM,
            "task_dim": V0_TASK_DIM,
            "output_dim": V0_OUTPUT_DIM,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "embedding_layers": 2,
            "num_parallel": 1,
            "action_processing": "normalized_raw_lewm_action_block",
            "shared_lewm_action_encoder": False,
            "state_parameterization": "symmetric_shared_frozen_lewm_latent",
            "goal_conditioning": "task_input",
            "bootstrap_action": "dataset_next_action",
            "actor": "none",
            "reward": "none",
        },
        label="predictor_config",
    )
    for key in ("state_dim", "action_dim", "task_dim", "output_dim", "hidden_dim"):
        _positive_integer(config, key)
    for key in ("hidden_layers", "embedding_layers"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"predictor_config.{key} must be non-negative.")
    if config["embedding_layers"] < 2:
        raise ValueError("predictor_config.embedding_layers must be at least two.")
    if config.get("num_parallel") != 1:
        raise ValueError("predictor_config.num_parallel must be exactly 1 in V0.")
    try:
        gamma = float(config["gamma"])
    except (TypeError, ValueError) as error:
        raise ValueError("predictor_config.gamma must lie in [0, 1).") from error
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("predictor_config.gamma must lie in [0, 1).")
    if not isinstance(payload["world_model_config"], Mapping):
        raise ValueError("checkpoint.world_model_config must be a mapping.")
    for key in ("task_sampling", "joint_objective", "pretrained_world_model"):
        if not isinstance(config[key], Mapping):
            raise ValueError(f"predictor_config.{key} must be a mapping.")
    require_exact_values(
        config["task_sampling"],
        {
            "goal_probability": 0.5,
            "sampling": "per_transition_bernoulli",
            "random_source": "isotropic_gaussian_sphere",
            "goal_source": "uniform_reachable_future_frozen_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "transition_minibatch",
        },
        label="predictor_config.task_sampling",
    )
    require_exact_values(
        config["joint_objective"],
        {
            "base_td_population": "all_transitions",
            "random_task_weight": 1.0,
            "goal_subset": "goal_derived_tasks_only",
            "final_weight_normalization": "mean_one_over_all_transitions",
            "weight_gradient": "stop_gradient",
            "candidate_td_targets": "none",
        },
        label="predictor_config.joint_objective",
    )
    require_exact_values(
        config["pretrained_world_model"],
        {"frozen": True},
        label="predictor_config.pretrained_world_model",
    )
    spec.validate_method_config(config)
    return config


def load_frozen_actor_free_td_v0_checkpoint(
    checkpoint_path: str | Path,
    *,
    spec: FrozenActorFreeTDV0MethodSpec,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, ActorFreeTDJEPAPredictorV0, dict[str, Any], dict[str, Any]]:
    """Restore one V0 predictor and the exact frozen LeWM it was trained on."""

    payload_value = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload_value, Mapping):
        raise ValueError("Deployment checkpoint must contain a mapping payload.")
    payload = dict(payload_value)
    config = validate_frozen_actor_free_td_v0_payload(payload, spec=spec)

    predictor = ActorFreeTDJEPAPredictorV0(
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        embedding_layers=int(config["embedding_layers"]),
    )
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)

    import hydra
    from omegaconf import OmegaConf

    world_model = hydra.utils.instantiate(
        OmegaConf.create(payload["world_model_config"])
    )
    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    world_model.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    return world_model, predictor, config, payload


class ActorFreeTDLeWMV0(nn.Module):
    """Plan with frozen LeWM F, the V0 successor G, or their clean splice."""

    supported_score_modes = SCORE_MODES
    default_score_mode = "f_plus_g"

    def __init__(
        self,
        world_model: nn.Module,
        predictor: ActorFreeTDJEPAPredictorV0,
        *,
        gamma: float,
        score_mode: str | None = None,
        lewm_history_size: int = LEWM_HISTORY_SIZE,
    ) -> None:
        super().__init__()
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        self.world_model = world_model
        self.predictor = predictor
        self.gamma = float(gamma)
        if lewm_history_size != LEWM_HISTORY_SIZE:
            raise ValueError(
                f"V0 must use the frozen LeWM baseline history size "
                f"{LEWM_HISTORY_SIZE}."
            )
        self.lewm_history_size = int(lewm_history_size)
        self.score_mode = score_mode or self.default_score_mode
        if self.score_mode not in self.supported_score_modes:
            raise ValueError(
                f"Unsupported {type(self).__name__} score mode "
                f"{self.score_mode!r}; expected one of "
                f"{sorted(self.supported_score_modes)}."
            )
        for attribute, expected in (
            ("state_dim", V0_STATE_DIM),
            ("action_dim", V0_ACTION_DIM),
            ("task_dim", V0_TASK_DIM),
            ("output_dim", V0_OUTPUT_DIM),
        ):
            if getattr(predictor, attribute, None) != expected:
                raise ValueError(
                    f"V0 predictor {attribute} must be {expected}, found "
                    f"{getattr(predictor, attribute, None)!r}."
                )

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
        if history_size is None:
            history_size = self.lewm_history_size
        return self.world_model.rollout(
            info, action_sequence, history_size=history_size
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
                "action_candidates must have shape (batch, samples, horizon, 25)."
            )
        if action_candidates.shape[-1] != V0_ACTION_DIM:
            raise ValueError("V0 expects normalized raw 25D action blocks.")
        if not action_candidates.is_floating_point():
            raise TypeError("action_candidates must have a floating-point dtype.")
        if not bool(torch.isfinite(action_candidates).all()):
            raise ValueError("action_candidates must contain only finite values.")
        batch, samples, horizon = action_candidates.shape[:3]
        if horizon <= 0:
            raise ValueError("The planning horizon must be positive.")
        if self.score_mode == "g_only" and horizon != 1:
            raise ValueError("V0 g_only requires CEM planning horizon=1.")

        goal = self._goal_for_samples(
            info_dict,
            batch=batch,
            samples=samples,
            reference=action_candidates,
        )
        task = project_tasks_to_sphere_v0(goal)

        if self.score_mode == "g_only":
            # Strict direct G planning: encoding is allowed, but the frozen
            # world model rollout/predictor path must not be used.
            current = self._current_state_for_samples(
                info_dict,
                batch=batch,
                samples=samples,
                reference=action_candidates,
            )
            return -self._goal_score(current, action_candidates[..., 0, :], task)

        if self.score_mode == "f_only":
            future = self._rollout_future(
                info_dict,
                action_candidates,
                batch=batch,
                samples=samples,
                horizon=horizon,
            )
            return self._explicit_terminal_cost(future, goal)

        if horizon > 1:
            # The hybrid boundary is exact: F receives only the first H-1
            # action blocks, and G alone receives the final action block.
            explicit_future = self._rollout_future(
                info_dict,
                action_candidates[..., :-1, :],
                batch=batch,
                samples=samples,
                horizon=horizon - 1,
            )
            explicit_cost = self._explicit_terminal_cost(explicit_future, goal)
            before_final = explicit_future[..., -1, :]
        else:
            # H=1 has no F prefix, so do not invoke the frozen LeWM rollout.
            explicit_cost = action_candidates.new_zeros(batch, samples)
            before_final = self._current_state_for_samples(
                info_dict,
                batch=batch,
                samples=samples,
                reference=action_candidates,
            )
        final_score = self._goal_score(
            before_final,
            action_candidates[..., -1, :],
            task,
        )
        # G is a value, so CEM receives its negative as a cost.  V0 deliberately
        # does not clamp this term.
        return explicit_cost - (self.gamma ** (horizon - 1)) * final_score

    def _goal_score(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.predictor(state, action, task)
        if prediction.shape != state.shape:
            raise ValueError(
                "V0 predictor must return one 192D output per state with no "
                "parallel-head axis."
            )
        return (prediction * task).sum(dim=-1)

    @staticmethod
    def _explicit_terminal_cost(
        future: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Return Stable World Model 0.1.1 LeWM's terminal summed MSE."""

        if future.shape[-2] <= 0:
            raise ValueError("LeWM explicit cost requires at least one future state.")
        terminal = future[..., -1, :]
        if terminal.shape != goal.shape:
            raise ValueError("LeWM terminal and raw goal embeddings must align.")
        return (terminal - goal).square().sum(dim=-1)

    def _rollout_future(
        self,
        info: dict[str, Any],
        actions: torch.Tensor,
        *,
        batch: int,
        samples: int,
        horizon: int,
    ) -> torch.Tensor:
        observed_frames = self._observed_frames(info)
        rollout_info = self.world_model.rollout(
            info,
            actions,
            history_size=self.lewm_history_size,
        )
        predicted = rollout_info.get("predicted_emb")
        if not torch.is_tensor(predicted) or predicted.ndim != 4:
            raise ValueError(
                "LeWM rollout must return predicted_emb with shape "
                "(batch, samples, time, 192)."
            )
        if predicted.shape[:2] != (batch, samples):
            raise ValueError("LeWM rollout does not match the candidate batch.")
        if predicted.shape[-1] != V0_STATE_DIM:
            raise ValueError("LeWM rollout must use the frozen 192D latent space.")
        if predicted.shape[-2] < observed_frames:
            raise ValueError("LeWM rollout contains fewer than the observed frames.")
        future = predicted[..., observed_frames:, :]
        if future.shape[-2] != horizon:
            raise ValueError(
                "LeWM rollout future length does not match the CEM horizon: "
                f"{future.shape[-2]} != {horizon}."
            )
        return future

    @staticmethod
    def _observed_frames(info: Mapping[str, Any]) -> int:
        pixels = info.get("pixels")
        if torch.is_tensor(pixels) and pixels.ndim >= 3:
            return int(pixels.shape[2])
        embedding = info.get("emb")
        if torch.is_tensor(embedding):
            if embedding.ndim == 4:
                return int(embedding.shape[-2])
            if embedding.ndim == 3:
                return int(embedding.shape[-2])
        raise ValueError("pixels or cached emb is required to infer observed history.")

    def _current_state_for_samples(
        self,
        info: dict[str, Any],
        *,
        batch: int,
        samples: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        embedding = info.get("emb")
        if embedding is None:
            pixels = info.get("pixels")
            if not torch.is_tensor(pixels):
                raise ValueError("pixels or cached emb is required for V0 G scoring.")
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
                and key not in {"goal", "goal_emb", "predicted_emb"}
            }
            encoded = self.world_model.encode(initial)
            embedding = encoded.get("emb")
            if not torch.is_tensor(embedding):
                raise ValueError("LeWM encode must return a tensor under 'emb'.")
            if embedding.ndim == 3:
                info["emb"] = embedding.unsqueeze(1).expand(
                    batch, samples, -1, -1
                )
            elif embedding.ndim == 2:
                info["emb"] = embedding.unsqueeze(1).expand(batch, samples, -1)

        state = embedding.to(device=reference.device, dtype=reference.dtype)
        if state.ndim == 4:
            if state.shape[:2] != (batch, samples):
                raise ValueError("Cached emb has the wrong batch or sample axes.")
            state = state[..., -1, :]
        elif state.ndim == 3:
            if state.shape[0] != batch:
                raise ValueError("Cached emb has the wrong batch axis.")
            state = state[..., -1, :].unsqueeze(1).expand(batch, samples, -1)
        elif state.ndim == 2:
            if state.shape[0] != batch:
                raise ValueError("Cached emb has the wrong batch axis.")
            state = state.unsqueeze(1).expand(batch, samples, -1)
        else:
            raise ValueError("Cached emb must have shape (B,D), (B,T,D), or (B,S,T,D).")
        if state.shape != (batch, samples, V0_STATE_DIM):
            raise ValueError("Current frozen LeWM state must be 192-dimensional.")
        return state

    def _goal_for_samples(
        self,
        info: dict[str, Any],
        *,
        batch: int,
        samples: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        goal = info.get("goal_emb")
        if goal is None:
            goal_info = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
                and key not in {"emb", "goal_emb", "predicted_emb"}
            }
            if "goal" not in goal_info:
                raise ValueError("goal or goal_emb is required for V0 planning.")
            goal_info["pixels"] = goal_info["goal"]
            for key in list(goal_info):
                if key.startswith("goal_"):
                    goal_info[key[len("goal_") :]] = goal_info.pop(key)
            goal_info.pop("action", None)
            encoded = self.world_model.encode(goal_info)
            goal = encoded.get("emb")
            if not torch.is_tensor(goal):
                raise ValueError("LeWM goal encode must return a tensor under 'emb'.")
            info["goal_emb"] = goal

        if not torch.is_tensor(goal):
            raise TypeError("goal_emb must be a torch.Tensor.")
        goal = goal.to(device=reference.device, dtype=reference.dtype)
        if goal.ndim < 2 or goal.shape[0] != batch:
            raise ValueError("goal_emb has the wrong batch axis or rank.")
        if goal.shape[-1] != V0_TASK_DIM:
            raise ValueError("Goal latent must be 192-dimensional.")

        # CEM may insert a sample axis, but the goal remains an environment
        # property.  Select one copy, then collapse every remaining temporal
        # axis to its last item before broadcasting the single [B,192] goal.
        if goal.ndim >= 4:
            if goal.shape[1] != samples:
                raise ValueError("goal_emb has the wrong CEM sample axis.")
            goal = goal[:, 0]
        elif goal.ndim == 3 and goal.shape[1] == samples:
            goal = goal[:, 0]
        while goal.ndim > 2:
            goal = goal.select(dim=-2, index=goal.shape[-2] - 1)
        if goal.shape != (batch, V0_TASK_DIM):
            raise ValueError(
                "goal_emb must collapse to one 192D goal per environment."
            )
        return goal.unsqueeze(1).expand(batch, samples, V0_TASK_DIM)


def make_frozen_actor_free_td_v0_policy(
    *,
    world_model: nn.Module,
    predictor: ActorFreeTDJEPAPredictorV0,
    planning: dict[str, Any],
    gamma: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    score_mode: str | None = None,
):
    """Build Stable World Model 0.1.1's public CEM policy around V0."""

    resolved_mode = score_mode or ActorFreeTDLeWMV0.default_score_mode
    if resolved_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported V0 score mode {resolved_mode!r}.")
    if int(planning["action_block"]) != ACTION_BLOCK_STEPS:
        raise ValueError("V0 requires planning.action_block=5 (25 normalized values).")
    if resolved_mode == "g_only" and int(planning["horizon"]) != 1:
        raise ValueError("V0 g_only requires planning.horizon=1.")

    import stable_worldmodel as swm

    wrapped = ActorFreeTDLeWMV0(
        world_model,
        predictor,
        gamma=gamma,
        score_mode=resolved_mode,
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
    plan_config = swm.PlanConfig(
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
        config=plan_config,
        process=process,
        transform=transform,
    )


__all__ = [
    "ACTION_BLOCK_STEPS",
    "ActorFreeTDLeWMV0",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FrozenActorFreeTDV0MethodSpec",
    "IMPLEMENTATION_VERSION",
    "LEWM_HISTORY_SIZE",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "SCORE_MODES",
    "load_frozen_actor_free_td_v0_checkpoint",
    "make_frozen_actor_free_td_v0_policy",
    "require_exact_values",
    "require_positive_float",
    "validate_frozen_actor_free_td_v0_payload",
]
