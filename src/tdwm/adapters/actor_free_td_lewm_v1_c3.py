"""Deployment adapter for the V1-C3 RP1-style goal-conditioned State-V critic.

V1-C3 is an actor-free continuation of the frozen V1-C checkpoint.  The
checkpoint retains V1-C's frozen LeWM and online/EMA G pair.  The original
State-V ablation deliberately uses neither G nor latent distance.  The paired
``state_v_plus_first_q2`` ablation keeps the same full frozen-F rollout and
adds only V1-C's online first-action Q after independently normalizing the
State-V and Q signals over each CEM candidate set.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v1_c import METHOD_SPEC as PARENT_METHOD_SPEC
from tdwm.adapters.frozen_actor_free_td_v1_common import (
    ACTION_BLOCK_STEPS,
    LEWM_HISTORY_SIZE,
    _normalize_cem_candidate_scores,
    validate_frozen_actor_free_td_v1_payload,
)
from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    ActorFreeTDJEPAPredictorV1,
    encode_frozen_action_blocks_v1,
    project_tasks_to_sphere_v1,
    tdjepa_goal_score_v1,
    validate_frozen_lewm_action_encoder_v1,
)
from tdwm.methods.actor_free_td_lewm_v1_c3 import RP1StateValueV1C3

METHOD = "actor_free_td_lewm_v1_c3"
METHOD_FAMILY = "actor_free_td_lewm_v1"
VARIANT = "c3"
IMPLEMENTATION_VERSION = "v1"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1
PARENT_METHOD = "actor_free_td_lewm_v1_c"
PARENT_VARIANT = "c"
PARENT_OBJECTIVE_VERSION = 0
PARENT_CHECKPOINT_SHA256 = (
    "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3"
)
STATE_V_SCORE_MODE = "state_v_terminal"
STATE_V_FIRST_Q2_SCORE_MODE = "state_v_plus_first_q2"
STATE_V_SCORE_MODES = frozenset({STATE_V_SCORE_MODE, STATE_V_FIRST_Q2_SCORE_MODE})
STATE_V_FIRST_Q2_WEIGHT = 0.25
STATE_V_CONSTANT_SANITY_OFFSET = 25.0


@dataclass(frozen=True)
class ActorFreeTDLeWMV1C3Checkpoint:
    """Strictly restored deployment components from one V1-C3 checkpoint."""

    world_model: nn.Module
    predictor: ActorFreeTDJEPAPredictorV1
    target_predictor: ActorFreeTDJEPAPredictorV1
    critic: RP1StateValueV1C3
    target_critic: RP1StateValueV1C3
    predictor_config: dict[str, Any]
    critic_config: dict[str, Any]
    payload: dict[str, Any]


def _require_exact_values(
    values: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"{label}.{key} must be {expected_value!r}.")


def _validate_source_v1_c_provenance(provenance: Mapping[str, Any]) -> None:
    _require_exact_values(
        provenance,
        {
            "method": PARENT_METHOD,
            "method_family": METHOD_FAMILY,
            "variant": PARENT_VARIANT,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": PARENT_OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "source_seed": 3072,
            "source_epoch": 10,
            "source_global_step": 127_960,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parameter_state": "strict_all_model_parameters_frozen",
            "optimizer_state": "not_loaded",
            "scheduler_state": "not_loaded",
            "epoch_and_global_step": "reset",
        },
        label="source_v1_c_provenance",
    )


def validate_actor_free_td_lewm_v1_c3_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate C3 identity, its frozen V1-C parent, and the State-V pair."""

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
        "logical_epoch",
        "global_step",
        "world_model_state_dict",
        "world_model_config",
        "predictor_state_dict",
        "target_predictor_state_dict",
        "predictor_config",
        "critic_state_dict",
        "target_critic_state_dict",
        "critic_config",
        "pretrained_world_model_provenance",
        "source_v1_c_provenance",
        "source_v1_c_runtime_provenance",
        "parent_state_hashes",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"V1-C3 checkpoint is missing {sorted(missing)}.")
    for forbidden in ("actor_state_dict", "successor_state_dict"):
        if forbidden in payload:
            raise ValueError(f"V1-C3 must not contain {forbidden}.")

    predictor_value = payload["predictor_config"]
    critic_value = payload["critic_config"]
    source_value = payload["source_v1_c_provenance"]
    if not isinstance(predictor_value, Mapping):
        raise ValueError("checkpoint.predictor_config must be a mapping.")
    if not isinstance(critic_value, Mapping):
        raise ValueError("checkpoint.critic_config must be a mapping.")
    if not isinstance(source_value, Mapping):
        raise ValueError("checkpoint.source_v1_c_provenance must be a mapping.")
    predictor_config = dict(predictor_value)
    critic_config = dict(critic_value)
    _validate_source_v1_c_provenance(source_value)

    parent_hashes = payload["parent_state_hashes"]
    runtime_provenance = payload["source_v1_c_runtime_provenance"]
    if not isinstance(parent_hashes, Mapping):
        raise ValueError("checkpoint.parent_state_hashes must be a mapping.")
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("checkpoint.source_v1_c_runtime_provenance must be a mapping.")
    state_dicts = {
        "world_model_state_sha256": payload["world_model_state_dict"],
        "online_g_state_sha256": payload["predictor_state_dict"],
        "target_g_state_sha256": payload["target_predictor_state_dict"],
    }
    for key, state_dict in state_dicts.items():
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"checkpoint state for {key} must be a mapping.")
        actual = _state_dict_sha256(state_dict)
        if parent_hashes.get(key) != actual:
            raise ValueError(f"checkpoint.parent_state_hashes.{key} is invalid.")
        if runtime_provenance.get(key) != actual:
            raise ValueError(
                f"checkpoint.source_v1_c_runtime_provenance.{key} is invalid."
            )
    _require_exact_values(
        runtime_provenance,
        {
            "strategy": "strict_frozen_v1_c_epoch_10_parent",
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_method": PARENT_METHOD,
            "parent_epoch": 10,
            "parent_global_step": 127_960,
        },
        label="source_v1_c_runtime_provenance",
    )
    if runtime_provenance.get("predictor_config_sha256") != _canonical_sha256(
        predictor_config
    ):
        raise ValueError(
            "source_v1_c_runtime_provenance.predictor_config_sha256 is invalid."
        )

    # Validate the retained parent with the existing audited V1-C contract.
    # Only the C3 envelope identity differs; the embedded predictor remains the
    # exact original V1-C model rather than a renamed C3 predictor.
    parent_payload = dict(payload)
    parent_payload.update(
        {
            "method": PARENT_METHOD,
            "variant": PARENT_VARIANT,
            "objective_version": PARENT_OBJECTIVE_VERSION,
        }
    )
    validate_frozen_actor_free_td_v1_payload(
        parent_payload,
        spec=PARENT_METHOD_SPEC,
    )

    _require_exact_values(
        critic_config,
        {
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "variant": VARIANT,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "architecture": "rp1_mrn_quasimetric",
            "input": "current_and_goal_frozen_lewm_latents",
            "state_dim": V1_STATE_DIM,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "depth": 2,
            "output": "nonnegative_primitive_temporal_cost_to_go",
            "action_input": "none",
            "actor": "none",
            "goal_identity_value": "exact_zero_by_architecture",
            "block_primitive_steps": 5,
            "backup_horizon_primitive_steps": 50,
            "gamma_per_primitive_step": 0.98,
            "expectile": 0.03,
            "huber_beta": 1.0,
            "huber_beta_source": "local_prelock_paper_does_not_report",
            "target_ema_update_rate": 0.005,
            "target_ema_source": ("nearest_documented_rp1_co_trained_critic_value"),
        },
        label="critic_config",
    )
    goal_sampling = critic_config.get("goal_sampling")
    objective = critic_config.get("objective")
    if not isinstance(goal_sampling, Mapping):
        raise ValueError("critic_config.goal_sampling must be a mapping.")
    if not isinstance(objective, Mapping):
        raise ValueError("critic_config.objective must be a mapping.")
    _require_exact_values(
        goal_sampling,
        {
            "source": "same_episode_reachable_future_frozen_latent",
            "distribution": "uniform_temporal_offset_over_available_horizon",
            "cross_episode_probability": 0.0,
            "cross_episode_reason": (
                "omitted_because_paper_does_not_specify_unreachable_label"
            ),
            "goal_sampling_seed_offset": 470_003,
        },
        label="critic_config.goal_sampling",
    )
    _require_exact_values(
        objective,
        {
            "name": "rp1_state_value_n_step_expectile_huber_td",
            "exact_goal_branch": "delta_if_goal_inside_n_step_backup",
            "bootstrap_branch": ("discounted_primitive_step_cost_plus_ema_state_value"),
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
        label="critic_config.objective",
    )
    return predictor_config, critic_config


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash parent tensor names, metadata, and bytes exactly as training does."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a tensor.")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _build_critic(config: Mapping[str, Any]) -> RP1StateValueV1C3:
    return RP1StateValueV1C3(
        state_dim=int(config["state_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        depth=int(config["depth"]),
    )


def load_actor_free_td_lewm_v1_c3_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ActorFreeTDLeWMV1C3Checkpoint:
    """Restore all parent and critic modules, failing closed on provenance."""

    payload_value = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload_value, Mapping):
        raise ValueError("Deployment checkpoint must contain a mapping payload.")
    payload = dict(payload_value)
    predictor_config, critic_config = validate_actor_free_td_lewm_v1_c3_payload(payload)

    predictor = ActorFreeTDJEPAPredictorV1(
        hidden_dim=int(predictor_config["hidden_dim"]),
        hidden_layers=int(predictor_config["hidden_layers"]),
        embedding_layers=int(predictor_config["embedding_layers"]),
    )
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    target_predictor = predictor.make_target()
    target_predictor.load_state_dict(
        payload["target_predictor_state_dict"], strict=True
    )

    critic = _build_critic(critic_config)
    critic.load_state_dict(payload["critic_state_dict"], strict=True)
    target_critic = _build_critic(critic_config)
    target_critic.load_state_dict(payload["target_critic_state_dict"], strict=True)

    import hydra
    from omegaconf import OmegaConf

    world_model = hydra.utils.instantiate(
        OmegaConf.create(payload["world_model_config"])
    )
    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    world_model.eval().requires_grad_(False)
    action_encoder = getattr(world_model, "action_encoder", None)
    if not isinstance(action_encoder, nn.Module):
        raise ValueError("V1-C3 checkpoint world model is missing action_encoder.")
    validate_frozen_lewm_action_encoder_v1(action_encoder)

    for module in (predictor, target_predictor, critic, target_critic):
        module.eval().requires_grad_(False)
    return ActorFreeTDLeWMV1C3Checkpoint(
        world_model=world_model,
        predictor=predictor,
        target_predictor=target_predictor,
        critic=critic,
        target_critic=target_critic,
        predictor_config=predictor_config,
        critic_config=critic_config,
        payload=payload,
    )


def assert_constant_shift_preserves_selection(
    costs: torch.Tensor,
    *,
    offset: float = STATE_V_CONSTANT_SANITY_OFFSET,
) -> None:
    """Assert the documented ``25 + V`` sanity has the same chosen plan."""

    if costs.ndim != 2:
        raise ValueError("CEM costs must have shape (batch, samples).")
    if not costs.is_floating_point() or not bool(torch.isfinite(costs).all()):
        raise ValueError("CEM costs must be finite floating-point values.")
    # Float64 makes the sanity check about the mathematical additive constant,
    # rather than introducing avoidable float32 rounding around near-ties.
    reference = costs.double()
    shifted = reference + reference.new_tensor(offset)
    if not torch.equal(
        torch.argsort(reference, dim=1, stable=True),
        torch.argsort(shifted, dim=1, stable=True),
    ):
        raise RuntimeError("Adding the constant block cost changed CEM ranking.")
    if not torch.equal(reference.argmin(dim=1), shifted.argmin(dim=1)):
        raise RuntimeError("Adding the constant block cost changed CEM selection.")


class ActorFreeTDLeWMV1C3(nn.Module):
    """Full frozen-F rollout scored by State-V, optionally plus first-Q2."""

    supported_score_modes = STATE_V_SCORE_MODES
    default_score_mode = STATE_V_SCORE_MODE

    def __init__(
        self,
        world_model: nn.Module,
        target_critic: RP1StateValueV1C3,
        predictor: ActorFreeTDJEPAPredictorV1 | None = None,
        *,
        score_mode: str = STATE_V_SCORE_MODE,
        g_first_weight: float | None = None,
        lewm_history_size: int = LEWM_HISTORY_SIZE,
        rollout_horizon: int = 5,
        run_constant_shift_sanity: bool = True,
    ) -> None:
        super().__init__()
        if lewm_history_size != LEWM_HISTORY_SIZE:
            raise ValueError("V1-C3 requires the frozen LeWM history size 3.")
        if rollout_horizon != 5:
            raise ValueError("V1-C3 requires exactly five imagined action blocks.")
        self.world_model = world_model
        self.target_critic = target_critic
        if score_mode not in self.supported_score_modes:
            raise ValueError(
                f"Unsupported V1-C3 score mode {score_mode!r}; expected one of "
                f"{sorted(self.supported_score_modes)}."
            )
        self.score_mode = score_mode
        if score_mode == STATE_V_FIRST_Q2_SCORE_MODE:
            if predictor is None:
                raise ValueError("state_v_plus_first_q2 requires the retained online G.")
            if g_first_weight != STATE_V_FIRST_Q2_WEIGHT:
                raise ValueError(
                    "state_v_plus_first_q2 requires the pre-registered weight 0.25."
                )
        elif g_first_weight is not None:
            raise ValueError("g_first_weight is only valid for state_v_plus_first_q2.")
        self.predictor = predictor
        self.g_first_weight = g_first_weight
        self.lewm_history_size = int(lewm_history_size)
        self.rollout_horizon = int(rollout_horizon)
        self.run_constant_shift_sanity = bool(run_constant_shift_sanity)
        self.constant_shift_sanity_checked = False

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
            history_size=history_size or self.lewm_history_size,
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
        if action_candidates.shape[-1] != V1_RAW_ACTION_DIM:
            raise ValueError("V1-C3 expects normalized raw 25D action blocks.")
        if not action_candidates.is_floating_point():
            raise TypeError("action_candidates must have a floating-point dtype.")
        if not bool(torch.isfinite(action_candidates).all()):
            raise ValueError("action_candidates must contain only finite values.")
        batch, samples, horizon = action_candidates.shape[:3]
        if horizon != self.rollout_horizon:
            raise ValueError(
                "V1-C3 State-V scoring requires a full five-block rollout."
            )

        current = None
        if self.score_mode == STATE_V_FIRST_Q2_SCORE_MODE:
            current = self._current_state_for_samples(
                dict(info_dict),
                batch=batch,
                samples=samples,
                reference=action_candidates,
            )
        goal = self._goal_for_samples(
            info_dict,
            batch=batch,
            samples=samples,
            reference=action_candidates,
        )
        terminal = self._terminal_future(
            info_dict,
            action_candidates,
            batch=batch,
            samples=samples,
        )
        state_v = self.target_critic(terminal, goal)
        if state_v.ndim == 3 and state_v.shape[-1] == 1:
            state_v = state_v.squeeze(-1)
        if state_v.shape != (batch, samples):
            raise ValueError("EMA State-V must return one scalar per CEM candidate.")
        if not bool(torch.isfinite(state_v).all()):
            raise ValueError("EMA State-V returned NaN or Inf costs.")
        if bool((state_v < 0).any()):
            raise ValueError("EMA State-V must be nonnegative.")
        if self.run_constant_shift_sanity and not self.constant_shift_sanity_checked:
            assert_constant_shift_preserves_selection(state_v)
            self.constant_shift_sanity_checked = True
        if self.score_mode == STATE_V_SCORE_MODE:
            return state_v

        if current is None or self.predictor is None or self.g_first_weight is None:
            raise RuntimeError("V1-C3 first-Q2 components were not initialized.")
        task = project_tasks_to_sphere_v1(goal)
        q_first = self._goal_score(
            current,
            action_candidates[..., 0, :],
            task,
        )
        return _normalize_cem_candidate_scores(state_v) - self.g_first_weight * (
            _normalize_cem_candidate_scores(q_first)
        )

    def _goal_score(
        self,
        state: torch.Tensor,
        raw_action: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        if self.predictor is None:
            raise RuntimeError("The retained online G is unavailable.")
        action_embedding = encode_frozen_action_blocks_v1(
            self.world_model.action_encoder,
            raw_action,
            reference=state,
        )
        prediction = self.predictor(state, action_embedding, task)
        if prediction.shape != state.shape:
            raise ValueError("V1-C3 retained online G must return one 192D vector.")
        return tdjepa_goal_score_v1(prediction, task)

    def _terminal_future(
        self,
        info: dict[str, Any],
        actions: torch.Tensor,
        *,
        batch: int,
        samples: int,
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
            raise ValueError("LeWM rollout does not match the CEM candidate batch.")
        if predicted.shape[-1] != V1_STATE_DIM:
            raise ValueError("LeWM rollout must use the frozen 192D latent space.")
        future = predicted[..., observed_frames:, :]
        if future.shape[-2] != self.rollout_horizon:
            raise ValueError("LeWM rollout future must contain exactly five blocks.")
        return future[..., -1, :]

    def _current_state_for_samples(
        self,
        info: dict[str, Any],
        *,
        batch: int,
        samples: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return the real current encoder state used by V1-C First-Q2."""

        embedding = info.get("emb")
        if embedding is None:
            pixels = info.get("pixels")
            if not torch.is_tensor(pixels):
                raise ValueError("pixels or cached emb is required for first-Q2.")
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value)
                and key not in {"action", "goal", "goal_emb", "predicted_emb"}
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
            raise ValueError(
                "Cached emb must have shape (B,D), (B,T,D), or (B,S,T,D)."
            )
        if state.shape != (batch, samples, V1_STATE_DIM):
            raise ValueError("Current frozen LeWM state must be 192-dimensional.")
        return state

    @staticmethod
    def _observed_frames(info: Mapping[str, Any]) -> int:
        pixels = info.get("pixels")
        if torch.is_tensor(pixels) and pixels.ndim >= 3:
            return int(pixels.shape[2])
        embedding = info.get("emb")
        if torch.is_tensor(embedding):
            if embedding.ndim in (3, 4):
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
                raise ValueError("goal or goal_emb is required for V1-C3 planning.")
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
        if goal.shape[-1] != V1_STATE_DIM:
            raise ValueError("Goal latent must be 192-dimensional.")
        if goal.ndim >= 4:
            if goal.shape[1] != samples:
                raise ValueError("goal_emb has the wrong CEM sample axis.")
            goal = goal[:, 0]
        elif goal.ndim == 3 and goal.shape[1] == samples:
            goal = goal[:, 0]
        while goal.ndim > 2:
            goal = goal.select(dim=-2, index=goal.shape[-2] - 1)
        if goal.shape != (batch, V1_STATE_DIM):
            raise ValueError("goal_emb must collapse to one 192D goal per environment.")
        return goal.unsqueeze(1).expand(batch, samples, V1_STATE_DIM)


def make_actor_free_td_lewm_v1_c3_policy(
    *,
    world_model: nn.Module,
    target_critic: RP1StateValueV1C3,
    predictor: ActorFreeTDJEPAPredictorV1 | None = None,
    planning: dict[str, Any],
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    reduced_evaluation: bool = False,
    score_mode: str = STATE_V_SCORE_MODE,
    g_first_weight: float | None = None,
):
    """Build Stable World Model 0.1.1's public actor-free CEM policy."""

    invariant = {
        "horizon": 5,
        "initial_variance": 1.0,
        "action_block": ACTION_BLOCK_STEPS,
        "frame_skip": ACTION_BLOCK_STEPS,
        "planning_seed": 42,
        "solver_batch_size": 1,
        "receding_horizon": 1,
        "warm_start": True,
        "initial_distribution": "cem_gaussian_no_actor",
    }
    for key, expected_value in invariant.items():
        if planning.get(key) != expected_value:
            raise ValueError(
                f"V1-C3 formal policy requires planning.{key}={expected_value!r}."
            )
    formal_search = {"candidates": 300, "iterations": 30, "elites": 30}
    reduced_searches = (
        {"candidates": 8, "iterations": 1, "elites": 2},
        {"candidates": 128, "iterations": 10, "elites": 16},
    )
    actual_search = {key: planning.get(key) for key in formal_search}
    if reduced_evaluation:
        if actual_search not in reduced_searches:
            raise ValueError("Unknown reduced V1-C3 CEM search protocol.")
    elif actual_search != formal_search:
        raise ValueError("V1-C3 formal policy requires CEM 300x30 with topk=30.")

    import stable_worldmodel as swm

    wrapped = ActorFreeTDLeWMV1C3(
        world_model,
        target_critic,
        predictor,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
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
    "ActorFreeTDLeWMV1C3",
    "ActorFreeTDLeWMV1C3Checkpoint",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "PARENT_CHECKPOINT_SHA256",
    "PARENT_METHOD",
    "STATE_V_CONSTANT_SANITY_OFFSET",
    "STATE_V_FIRST_Q2_SCORE_MODE",
    "STATE_V_FIRST_Q2_WEIGHT",
    "STATE_V_SCORE_MODE",
    "STATE_V_SCORE_MODES",
    "VARIANT",
    "assert_constant_shift_preserves_selection",
    "load_actor_free_td_lewm_v1_c3_checkpoint",
    "make_actor_free_td_lewm_v1_c3_policy",
    "validate_actor_free_td_lewm_v1_c3_payload",
]
