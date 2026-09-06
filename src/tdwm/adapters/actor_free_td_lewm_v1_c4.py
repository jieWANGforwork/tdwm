"""Deployment adapter for post-action-ghost Actor-Free TD-LeWM V1-C4.

C4 keeps the pretrained LeWM world model frozen and removes action from the
successor predictor interface.  Every G read consumes a stopped post-action
ghost state produced by F; neither raw actions nor action embeddings can enter
G directly.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.adapters.frozen_actor_free_td_v1_common import (
    ACTION_BLOCK_STEPS,
    LEWM_HISTORY_SIZE,
    _normalize_cem_candidate_scores,
)
from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    V1_TASK_DIM,
    project_tasks_to_sphere_v1,
    tdjepa_goal_score_v1,
    validate_frozen_lewm_action_encoder_v1,
)
from tdwm.methods.actor_free_td_lewm_v1_c4 import ActorFreeTDJEPAPredictorV1C4

METHOD = "actor_free_td_lewm_v1_c4"
METHOD_FAMILY = "actor_free_td_lewm_v1"
VARIANT = "c4"
IMPLEMENTATION_VERSION = "v1"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1
C4_ACTION_EFFECT = "only_via_f_post_action_states"
C4_TIME_ALIGNMENT = {
    "replay_anchor": "z_i",
    "online_input": "stop_gradient_f_of_real_z_i_a_i",
    "target_current_feature": "real_z_i_plus_1",
    "target_bootstrap_input": (
        "stop_gradient_ema_g_of_f_real_z_i_plus_1_a_i_plus_1"
    ),
    "terminal_semantics": "d_i_true_when_transition_after_a_i_terminates",
    "terminal_target": "y_i_equals_real_z_i_plus_1",
    "f_history_states": 3,
    "f_output_role": "online_and_ema_bootstrap_post_action_states",
    "f_output_gradient": "stop_gradient",
}
C4_JOINT_OBJECTIVE = {
    "objective": "single_post_action_ghost_goal_projected_td",
    "vector_td_population": "all_transitions_single_online_branch",
    "vector_reduction": "mean_of_squared_l2_norm",
    "goal_subset": "goal_derived_tasks_only",
    "goal_projection_weight": 1.0,
    "goal_projection_target": (
        "detached_real_immediate_ghost_bootstrap_projection"
    ),
    "branch_combination": "single_online_branch",
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
}

F_ONLY_SCORE_MODE = "f_only"
C4_ONLY_SCORE_MODE = "g_only"
F_PLUS_C4_SCORE_MODE = "f_plus_g"
FIRST_Q_SCORE_MODE = "f_plus_g_first"
FIRST_Q2_SCORE_MODE = "f_plus_g_first_q2"
MEAN_Q_SCORE_MODE = "g_only_f_rollout_mean"
FIRST_ACTION_SCORE_MODES = frozenset({FIRST_Q_SCORE_MODE, FIRST_Q2_SCORE_MODE})
SCORE_MODES = frozenset(
    {
        F_ONLY_SCORE_MODE,
        C4_ONLY_SCORE_MODE,
        F_PLUS_C4_SCORE_MODE,
        FIRST_Q_SCORE_MODE,
        FIRST_Q2_SCORE_MODE,
        MEAN_Q_SCORE_MODE,
    }
)
FORMAL_HORIZON_BY_SCORE_MODE = {
    F_ONLY_SCORE_MODE: 5,
    C4_ONLY_SCORE_MODE: 1,
    F_PLUS_C4_SCORE_MODE: 5,
    FIRST_Q_SCORE_MODE: 5,
    FIRST_Q2_SCORE_MODE: 5,
    MEAN_Q_SCORE_MODE: 5,
}


def _require_exact_values(
    values: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"{label}.{key} must be {expected_value!r}.")


def _positive_integer(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"g_config.{key} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"g_config.{key} must be a positive integer.") from error
    if integer <= 0 or integer != value:
        raise ValueError(f"g_config.{key} must be a positive integer.")
    return integer


def _validate_state_only_forward(module: nn.Module) -> None:
    signature = inspect.signature(type(module).forward)
    parameters = tuple(signature.parameters)
    if parameters != ("self", "state", "task"):
        raise ValueError("C4 G.forward must have exactly the interface (state, task).")
    for forbidden in ("raw_action_dim", "action_dim", "action_embedding_dim"):
        if hasattr(module, forbidden):
            raise ValueError(f"C4 G must not declare {forbidden}.")


def validate_actor_free_td_lewm_v1_c4_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a C4 checkpoint without accepting the action-conditioned C form."""

    _require_exact_values(
        payload,
        {
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "variant": VARIANT,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
        },
        label="checkpoint",
    )
    required = {
        "epoch",
        "global_step",
        "world_model_state_dict",
        "world_model_config",
        "online_g_state_dict",
        "target_g_state_dict",
        "g_config",
        "pretrained_world_model_provenance",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"V1-C4 checkpoint is missing {sorted(missing)}.")
    for forbidden in (
        "actor_state_dict",
        "action_encoder_state_dict",
        "predictor_state_dict",
        "target_predictor_state_dict",
        "successor_state_dict",
    ):
        if forbidden in payload:
            raise ValueError(f"V1-C4 checkpoint must not contain {forbidden}.")
    if not isinstance(payload["world_model_config"], Mapping):
        raise ValueError("checkpoint.world_model_config must be a mapping.")
    if not isinstance(payload["g_config"], Mapping):
        raise ValueError("checkpoint.g_config must be a mapping.")
    config = dict(payload["g_config"])
    required_config = {
        "method",
        "method_family",
        "variant",
        "implementation_version",
        "objective_version",
        "deployment_checkpoint_version",
        "architecture",
        "state_dim",
        "task_dim",
        "output_dim",
        "hidden_dim",
        "hidden_layers",
        "embedding_layers",
        "num_parallel",
        "action_input",
        "action_effect",
        "goal_conditioning",
        "successor_semantics",
        "gamma",
        "target_ema_decay",
        "task_sampling",
        "joint_objective",
        "time_alignment",
        "pretrained_world_model",
    }
    missing_config = required_config - config.keys()
    if missing_config:
        raise ValueError(f"checkpoint.g_config is missing {sorted(missing_config)}.")
    for forbidden in ("raw_action_dim", "action_dim", "action_embedding_dim"):
        if forbidden in config:
            raise ValueError(f"C4 g_config must not contain {forbidden}.")
    _require_exact_values(
        config,
        {
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "variant": VARIANT,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "architecture": "td_jepa_state_only_forward_map_v1_c4",
            "state_dim": V1_STATE_DIM,
            "task_dim": V1_TASK_DIM,
            "output_dim": V1_STATE_DIM,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "embedding_layers": 2,
            "num_parallel": 1,
            "action_input": "none",
            "action_effect": C4_ACTION_EFFECT,
            "goal_conditioning": "task_input",
            "successor_semantics": "includes_current_input_state",
            "actor": "none",
            "reward": "none",
        },
        label="g_config",
    )
    for key in ("state_dim", "task_dim", "output_dim", "hidden_dim"):
        _positive_integer(config, key)
    for key in ("hidden_layers", "embedding_layers"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"g_config.{key} must be a non-negative integer.")
    try:
        gamma = float(config["gamma"])
        decay = float(config["target_ema_decay"])
    except (TypeError, ValueError) as error:
        raise ValueError("C4 gamma and target_ema_decay must lie in [0, 1).") from error
    if not 0.0 <= gamma < 1.0 or not 0.0 <= decay < 1.0:
        raise ValueError("C4 gamma and target_ema_decay must lie in [0, 1).")
    for key in (
        "task_sampling",
        "joint_objective",
        "time_alignment",
        "pretrained_world_model",
    ):
        if not isinstance(config[key], Mapping):
            raise ValueError(f"g_config.{key} must be a mapping.")
    if dict(config["time_alignment"]) != C4_TIME_ALIGNMENT:
        raise ValueError(
            "g_config.time_alignment must encode the post-action ghost TD path."
        )
    if dict(config["joint_objective"]) != C4_JOINT_OBJECTIVE:
        raise ValueError(
            "g_config.joint_objective must encode the single post-action ghost "
            "objective."
        )
    _require_exact_values(
        config["pretrained_world_model"],
        {"frozen": True},
        label="g_config.pretrained_world_model",
    )
    return config


def load_actor_free_td_lewm_v1_c4_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, ActorFreeTDJEPAPredictorV1C4, dict[str, Any], dict[str, Any]]:
    """Restore online C4 G and the exact frozen LeWM used for training."""

    payload_value = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload_value, Mapping):
        raise ValueError("Deployment checkpoint must contain a mapping payload.")
    payload = dict(payload_value)
    config = validate_actor_free_td_lewm_v1_c4_payload(payload)

    online_g = ActorFreeTDJEPAPredictorV1C4(
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        embedding_layers=int(config["embedding_layers"]),
    )
    _validate_state_only_forward(online_g)
    online_g.load_state_dict(payload["online_g_state_dict"], strict=True)
    target_g = online_g.make_target()
    _validate_state_only_forward(target_g)
    target_g.load_state_dict(payload["target_g_state_dict"], strict=True)

    import hydra
    from omegaconf import OmegaConf

    world_model = hydra.utils.instantiate(OmegaConf.create(payload["world_model_config"]))
    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    world_model.eval().requires_grad_(False)
    action_encoder = getattr(world_model, "action_encoder", None)
    if not isinstance(action_encoder, nn.Module):
        raise ValueError("V1-C4 checkpoint world model is missing action_encoder.")
    validate_frozen_lewm_action_encoder_v1(action_encoder)
    online_g.eval().requires_grad_(False)
    return world_model, online_g, config, payload


def _resolve_first_q_weight(score_mode: str, value: float | None) -> float | None:
    if score_mode not in FIRST_ACTION_SCORE_MODES:
        if value is not None:
            raise ValueError("g_first_weight is only valid for First-Q modes.")
        return None
    if value is None or isinstance(value, bool):
        raise ValueError("First-Q modes require a finite non-negative weight.")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("First-Q modes require a finite non-negative weight.")
    return 0.0 if weight == 0.0 else weight


class ActorFreeTDLeWMV1C4(nn.Module):
    """CEM cost adapter whose state-only G can see actions only through F."""

    supported_score_modes = SCORE_MODES
    default_score_mode = F_PLUS_C4_SCORE_MODE

    def __init__(
        self,
        world_model: nn.Module,
        predictor: ActorFreeTDJEPAPredictorV1C4,
        *,
        gamma: float,
        score_mode: str | None = None,
        g_first_weight: float | None = None,
        lewm_history_size: int = LEWM_HISTORY_SIZE,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(gamma) < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        if lewm_history_size != LEWM_HISTORY_SIZE:
            raise ValueError(f"C4 requires LeWM history size {LEWM_HISTORY_SIZE}.")
        action_encoder = getattr(world_model, "action_encoder", None)
        if not isinstance(action_encoder, nn.Module):
            raise ValueError("C4 requires frozen LeWM F with an action encoder.")
        validate_frozen_lewm_action_encoder_v1(action_encoder)
        _validate_state_only_forward(predictor)
        for attribute, expected in (
            ("state_dim", V1_STATE_DIM),
            ("task_dim", V1_TASK_DIM),
            ("output_dim", V1_STATE_DIM),
        ):
            if getattr(predictor, attribute, None) != expected:
                raise ValueError(f"C4 G.{attribute} must be {expected}.")

        self.world_model = world_model
        self.predictor = predictor
        self.gamma = float(gamma)
        self.lewm_history_size = int(lewm_history_size)
        self.score_mode = score_mode or self.default_score_mode
        if self.score_mode not in SCORE_MODES:
            raise ValueError(f"Unsupported C4 score mode {self.score_mode!r}.")
        self.g_first_weight = _resolve_first_q_weight(
            self.score_mode, g_first_weight
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
        return self.world_model.rollout(
            info,
            action_sequence,
            history_size=(self.lewm_history_size if history_size is None else history_size),
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
        if action_candidates.ndim != 4 or action_candidates.shape[-1] != V1_RAW_ACTION_DIM:
            raise ValueError(
                "action_candidates must have shape (batch, samples, horizon, 25)."
            )
        if not action_candidates.is_floating_point():
            raise TypeError("action_candidates must have a floating-point dtype.")
        if not bool(torch.isfinite(action_candidates).all()):
            raise ValueError("action_candidates must contain only finite values.")
        batch, samples, horizon = action_candidates.shape[:3]
        expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[self.score_mode]
        if horizon != expected_horizon:
            raise ValueError(
                f"C4 {self.score_mode} requires CEM planning horizon={expected_horizon}."
            )

        goal = self._goal_for_samples(
            info_dict,
            batch=batch,
            samples=samples,
            reference=action_candidates,
        )
        ghost_states = self._rollout_next_ghost_states(
            info_dict,
            action_candidates,
            batch=batch,
            samples=samples,
            horizon=horizon,
        )
        if self.score_mode == F_ONLY_SCORE_MODE:
            return self._explicit_terminal_cost(ghost_states, goal)

        task = project_tasks_to_sphere_v1(goal)
        if self.score_mode == C4_ONLY_SCORE_MODE:
            return -self._goal_score(ghost_states[..., -1, :], task)
        if self.score_mode == MEAN_Q_SCORE_MODE:
            step_tasks = task.unsqueeze(-2).expand_as(ghost_states)
            return -self._goal_score(ghost_states, step_tasks).mean(dim=-1)
        if self.score_mode in FIRST_ACTION_SCORE_MODES:
            weight = self.g_first_weight
            if weight is None:
                raise RuntimeError("First-Q weight was not initialized.")
            explicit_cost = self._explicit_terminal_cost(ghost_states, goal)
            if weight == 0.0:
                if self.score_mode == FIRST_Q2_SCORE_MODE:
                    return _normalize_cem_candidate_scores(explicit_cost)
                return explicit_cost
            first_score = self._goal_score(ghost_states[..., 0, :], task)
            if self.score_mode == FIRST_Q2_SCORE_MODE:
                explicit_cost = _normalize_cem_candidate_scores(explicit_cost)
                first_score = _normalize_cem_candidate_scores(first_score)
            return explicit_cost - weight * first_score

        if self.score_mode != F_PLUS_C4_SCORE_MODE:
            raise RuntimeError(f"Unhandled C4 score mode {self.score_mode!r}.")
        prefix_cost = self._explicit_terminal_cost(ghost_states[..., :-1, :], goal)
        final_score = self._goal_score(ghost_states[..., -1, :], task)
        return prefix_cost - (self.gamma ** (horizon - 1)) * final_score

    def _goal_score(self, state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        prediction = self.predictor(state, task)
        if prediction.shape != state.shape:
            raise ValueError("C4 G must return one 192D vector per input state.")
        return tdjepa_goal_score_v1(prediction, task)

    @staticmethod
    def _explicit_terminal_cost(
        future: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        if future.shape[-2] <= 0:
            raise ValueError("LeWM explicit cost requires at least one future state.")
        terminal = future[..., -1, :]
        if terminal.shape != goal.shape:
            raise ValueError("LeWM terminal and goal embeddings must align.")
        return (terminal - goal).square().sum(dim=-1)

    def _rollout_next_ghost_states(
        self,
        info: dict[str, Any],
        actions: torch.Tensor,
        *,
        batch: int,
        samples: int,
        horizon: int,
    ) -> torch.Tensor:
        """Return stopped F states after each candidate action block.

        The returned axis is ``(zhat_1, ..., zhat_H)``: ``zhat_1`` is the
        frozen-F successor of the current real state under ``A_1`` and every
        later ``zhat_k`` is the successor of the preceding planned ghost state
        under ``A_k``.  These tensors, never the candidate action or its
        embedding, are the only action-dependent inputs that C4 G can receive.
        """

        observed_frames = self._observed_frames(info)
        rollout_info = self.world_model.rollout(
            info, actions, history_size=self.lewm_history_size
        )
        predicted = rollout_info.get("predicted_emb")
        if not torch.is_tensor(predicted) or predicted.ndim != 4:
            raise ValueError(
                "LeWM rollout must return predicted_emb with shape "
                "(batch, samples, time, 192)."
            )
        if predicted.shape[:2] != (batch, samples) or predicted.shape[-1] != V1_STATE_DIM:
            raise ValueError("LeWM rollout has incompatible batch/sample/feature axes.")
        if predicted.shape[-2] < observed_frames:
            raise ValueError("LeWM rollout contains fewer than the observed frames.")
        ghost_states = predicted[..., observed_frames:, :]
        if ghost_states.shape[-2] != horizon:
            raise ValueError(
                "LeWM rollout future length differs from the CEM horizon: "
                f"{ghost_states.shape[-2]} != {horizon}."
            )
        # Formal evaluation already runs under torch.inference_mode(), but the
        # explicit detach makes the C4 action->F->ghost-state boundary true for
        # every direct adapter caller as well.
        return ghost_states.detach()

    @staticmethod
    def _observed_frames(info: Mapping[str, Any]) -> int:
        pixels = info.get("pixels")
        if torch.is_tensor(pixels) and pixels.ndim >= 3:
            return int(pixels.shape[2])
        embedding = info.get("emb")
        if torch.is_tensor(embedding) and embedding.ndim in {3, 4}:
            return int(embedding.shape[-2])
        raise ValueError("pixels or cached emb is required to infer observed history.")

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
                raise ValueError("goal or goal_emb is required for C4 planning.")
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
        if goal.ndim < 2 or goal.shape[0] != batch or goal.shape[-1] != V1_TASK_DIM:
            raise ValueError("goal_emb has incompatible batch/feature axes.")
        if goal.ndim >= 4:
            if goal.shape[1] != samples:
                raise ValueError("goal_emb has the wrong CEM sample axis.")
            goal = goal[:, 0]
        elif goal.ndim == 3 and goal.shape[1] == samples:
            goal = goal[:, 0]
        while goal.ndim > 2:
            goal = goal.select(dim=-2, index=goal.shape[-2] - 1)
        if goal.shape != (batch, V1_TASK_DIM):
            raise ValueError("goal_emb must collapse to one 192D goal per environment.")
        return goal.unsqueeze(1).expand(batch, samples, V1_TASK_DIM)


def make_actor_free_td_lewm_v1_c4_policy(
    *,
    world_model: nn.Module,
    predictor: ActorFreeTDJEPAPredictorV1C4,
    planning: dict[str, Any],
    gamma: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    score_mode: str | None = None,
    g_first_weight: float | None = None,
):
    """Build the public Stable World Model CEM policy for C4."""

    resolved_mode = score_mode or ActorFreeTDLeWMV1C4.default_score_mode
    if resolved_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported C4 score mode {resolved_mode!r}.")
    if int(planning["action_block"]) != ACTION_BLOCK_STEPS:
        raise ValueError("C4 requires planning.action_block=5.")
    expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[resolved_mode]
    if int(planning["horizon"]) != expected_horizon:
        raise ValueError(
            f"C4 {resolved_mode} requires planning.horizon={expected_horizon}."
        )
    resolved_weight = _resolve_first_q_weight(resolved_mode, g_first_weight)

    import stable_worldmodel as swm

    wrapped = ActorFreeTDLeWMV1C4(
        world_model,
        predictor,
        gamma=gamma,
        score_mode=resolved_mode,
        g_first_weight=resolved_weight,
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
        history_len=planning.get("plan_config_history_len", planning.get("history_len", 1)),
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
    "ActorFreeTDLeWMV1C4",
    "C4_ACTION_EFFECT",
    "C4_JOINT_OBJECTIVE",
    "C4_ONLY_SCORE_MODE",
    "C4_TIME_ALIGNMENT",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FIRST_ACTION_SCORE_MODES",
    "FIRST_Q2_SCORE_MODE",
    "FIRST_Q_SCORE_MODE",
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "F_ONLY_SCORE_MODE",
    "F_PLUS_C4_SCORE_MODE",
    "IMPLEMENTATION_VERSION",
    "MEAN_Q_SCORE_MODE",
    "METHOD",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "SCORE_MODES",
    "VARIANT",
    "load_actor_free_td_lewm_v1_c4_checkpoint",
    "make_actor_free_td_lewm_v1_c4_policy",
    "validate_actor_free_td_lewm_v1_c4_payload",
]
