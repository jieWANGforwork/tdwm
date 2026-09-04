"""Controlled Cube O50 evaluation for Actor-Free TD-LeWM V0 methods.

V0 is evaluated independently from the V-1 history-conditioned successor.
Every C--G3 checkpoint deploys the same single symmetric goal-conditioned
predictor.  Neighbor retrieval and action-prefix construction are training-only
signals and are intentionally absent from this runtime.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters.frozen_actor_free_td_common import (
    FORMAL_DEPLOYMENT_EPOCH,
    FORMAL_DEPLOYMENT_GLOBAL_STEP,
    is_lower_sha256,
)
from tdwm.adapters.frozen_actor_free_td_v0_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    METHOD_FAMILY,
    OBJECTIVE_VERSION,
    SCORE_MODES,
    FrozenActorFreeTDV0MethodSpec,
    require_exact_values,
)
from tdwm.adapters.runtime import prepare_cloud_runtime
from tdwm.evaluation.frozen_actor_free_td_common import (
    _resolve_frozen_dataset_source,
    _resolve_joint_checkpoint,
    _validate_dataset_protocol,
)
from tdwm.evaluation.lewm_checkpoint import (
    REQUIRED_PLANNING_KEYS,
    _git_revision,
    _jsonable,
    _sha256,
    _write_json,
    sample_start_goal_pairs,
)
from tdwm.evaluation.mc_gt_lewm import _load_action_processor
from tdwm.methods.actor_free_td_lewm_v0 import (
    V0_ACTION_DIM,
    V0_OUTPUT_DIM,
    V0_STATE_DIM,
    V0_TASK_DIM,
)

FORMAL_O50_PLANNING = {
    "horizon": 5,
    "candidates": 300,
    "iterations": 30,
    "elites": 30,
    "initial_variance": 1.0,
    "action_block": 5,
    "frame_skip": 5,
    "receding_horizon": 1,
    "episode_budget": 100,
    "planning_seed": 42,
    "solver_batch_size": 1,
    "warm_start": True,
    "initial_distribution": "cem_gaussian_no_actor",
}
FORMAL_HORIZON_BY_SCORE_MODE = {
    "f_only": 5,
    "g_only": 1,
    "f_plus_g": 5,
    "f_plus_g_first": 5,
    "g_only_f_rollout_mean": 5,
}
FIRST_ACTION_SCORE_MODE = "f_plus_g_first"
FIRST_ACTION_SCORE_DEFINITION = {
    "formula": "f_cost - g_first_weight * q_first",
    "f_cost": "terminal_summed_mse(z_hat5, z_goal)",
    "f_rollout": "full_five_action_blocks_A1_through_A5",
    "q_first": "dot(G(z0, normalized_raw_A1, w_goal), w_goal)",
    "q_first_state": "current_frozen_lewm_encoder_state_z0",
    "q_first_action": "first_candidate_raw_action_block_A1",
    "q_first_action_processing": "normalized_raw_25d_action_block",
    "q_first_task": "sqrt_dim_l2_normalized_goal_vector",
    "q_first_discount": "none",
    "cem_execution": "execute_A1_from_minimum_total_cost_plan",
}
ROLLOUT_MEAN_SCORE_MODE = "g_only_f_rollout_mean"
ROLLOUT_MEAN_G_SCORE = "mean_goal_projection_over_all_rollout_blocks"
ROLLOUT_MEAN_SCORE_DEFINITION = {
    "formula": "cost = -mean(q1, q2, q3, q4, q5)",
    "score": (
        "negative_mean_goal_projection_over_f_rollout_aligned_predecessor_action_pairs"
    ),
    "f_transition_used": True,
    "f_goal_distance_used": False,
    "g_score": ROLLOUT_MEAN_G_SCORE,
    "g_aggregation": "mean_over_5_blocks",
    "rollout_horizon": 5,
    "state_source_for_q1": "current_frozen_lewm_encoder_state",
    "state_source_for_q2_to_q5": "frozen_lewm_rollout_predicted_states",
    "state_sequence": "z0_and_first_h_minus_one_full_f_rollout_states",
    "action_sequence": "all_h_normalized_raw_candidate_action_blocks",
    "action_alignment": "qk_uses_same_candidate_action_block_Ak",
    "action_processing": "normalized_raw_25d_action_block",
    "task": "sqrt_dim_l2_normalized_goal_vector_broadcast_over_5_blocks",
    "gamma": "unused",
    "terminal_f_cost": "unused",
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}
ROLLOUT_MEAN_INFERENCE_FIELDS = {
    "f_transition_used": True,
    "f_goal_distance_used": False,
    "g_score": ROLLOUT_MEAN_G_SCORE,
    "g_aggregation": "mean_over_5_blocks",
    "rollout_horizon": 5,
    "state_source_for_q1": "current_frozen_lewm_encoder_state",
    "state_source_for_q2_to_q5": "frozen_lewm_rollout_predicted_states",
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}
ROLLOUT_MEAN_ONLY_INFERENCE_KEYS = frozenset(ROLLOUT_MEAN_INFERENCE_FIELDS) - {
    "g_score",
    "replanning",
}
LEGACY_F_SCORE = "lewm_rollout_goal_distance"
LEGACY_F_SCORE_REDUCER = "final_predicted_latent_summed_mse"
LEGACY_G_SCORE = "negative_goal_projection_of_v0_predictor"

CheckpointLoader = Callable[..., tuple[Any, Any, dict[str, Any], dict[str, Any]]]
PolicyFactory = Callable[..., Any]


def validate_v0_score_mode(score_mode: str) -> str:
    """Return a supported V0 planner score mode or fail closed."""

    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"score_mode {score_mode!r} is incompatible with V0; expected one "
            f"of {sorted(SCORE_MODES)}."
        )
    return score_mode


def _resolve_g_first_weight(
    protocol: Mapping[str, Any],
    *,
    score_mode: str,
    g_first_weight: float | None,
) -> float | None:
    inference = protocol.get("inference_objective", {})
    configured_weight = (
        inference.get("g_first_weight") if isinstance(inference, Mapping) else None
    )
    if score_mode != FIRST_ACTION_SCORE_MODE:
        if g_first_weight is not None:
            raise ValueError(
                "g_first_weight is only valid for score_mode='f_plus_g_first'."
            )
        return None
    raw_weight = g_first_weight if g_first_weight is not None else configured_weight
    if raw_weight is None or isinstance(raw_weight, bool):
        raise ValueError(
            "score_mode='f_plus_g_first' requires an explicit g_first_weight."
        )
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as error:
        raise ValueError("g_first_weight must be finite and non-negative.") from error
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("g_first_weight must be finite and non-negative.")
    return 0.0 if weight == 0.0 else weight


def _g_first_weight_slug(weight: float) -> str:
    value = format(weight, ".15g").lower()
    return value.replace("+", "").replace("-", "m").replace(".", "p")


def _configure_first_action_score(
    protocol: dict[str, Any],
    *,
    score_mode: str,
    g_first_weight: float | None,
) -> float | None:
    inference = protocol.setdefault("inference_objective", {})
    weight = _resolve_g_first_weight(
        protocol,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    if score_mode == FIRST_ACTION_SCORE_MODE:
        inference["g_first_weight"] = weight
        inference["score_definition"] = deepcopy(FIRST_ACTION_SCORE_DEFINITION)
    else:
        inference.pop("g_first_weight", None)
        inference.pop("score_definition", None)
    return weight


def _configure_rollout_mean_score(
    protocol: dict[str, Any],
    *,
    score_mode: str,
) -> None:
    inference = protocol.setdefault("inference_objective", {})
    if score_mode == ROLLOUT_MEAN_SCORE_MODE:
        inference.update(deepcopy(ROLLOUT_MEAN_INFERENCE_FIELDS))
        inference["f_score"] = "none"
        inference["f_score_reducer"] = "none"
        inference["score_definition"] = deepcopy(ROLLOUT_MEAN_SCORE_DEFINITION)
        return
    for key in ROLLOUT_MEAN_ONLY_INFERENCE_KEYS:
        inference.pop(key, None)
    if inference.get("f_score") == "none":
        inference["f_score"] = LEGACY_F_SCORE
    if inference.get("f_score_reducer") == "none":
        inference["f_score_reducer"] = LEGACY_F_SCORE_REDUCER
    if inference.get("g_score") == ROLLOUT_MEAN_G_SCORE:
        inference["g_score"] = LEGACY_G_SCORE


def _first_action_output_metadata(
    protocol: Mapping[str, Any],
    planning: Mapping[str, Any],
) -> dict[str, Any]:
    """Return fields emitted only by the first-action critic evaluation."""

    inference = protocol.get("inference_objective", {})
    if inference.get("score_mode") != FIRST_ACTION_SCORE_MODE:
        return {}
    return {
        "g_first_weight": inference["g_first_weight"],
        "planning": {"horizon": planning["horizon"]},
        "score_definition": deepcopy(inference["score_definition"]),
    }


def _rollout_mean_output_metadata(
    protocol: Mapping[str, Any],
    planning: Mapping[str, Any],
) -> dict[str, Any]:
    inference = protocol.get("inference_objective", {})
    if inference.get("score_mode") != ROLLOUT_MEAN_SCORE_MODE:
        return {}
    return {
        "g_aggregation": inference["g_aggregation"],
        "state_source_for_q1": inference["state_source_for_q1"],
        "state_source_for_q2_to_q5": inference["state_source_for_q2_to_q5"],
        "f_goal_distance_used": False,
        "f_transition_used": True,
        "planning_horizon": planning["horizon"],
        "rollout_horizon": planning["horizon"],
        "executed_action_block": "first_block_only",
        "replanning": "every_action_block",
        "score_definition": deepcopy(inference["score_definition"]),
    }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_protocol_mapping(path: Path, *, seen: frozenset[Path]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        raise ValueError("V0 evaluation protocol inheritance contains a cycle.")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("V0 evaluation protocol must contain a mapping.")
    current = dict(value)
    parent = current.get("extends")
    if parent is None:
        return current
    if not isinstance(parent, str) or not parent:
        raise ValueError("protocol.extends must be a non-empty relative path.")
    base = _load_protocol_mapping(
        (resolved.parent / parent).resolve(),
        seen=seen | {resolved},
    )
    return _deep_merge(base, current)


def actor_free_td_v0_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> str:
    """Return a collision-free output name for one V0 evaluation mode."""

    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    method = protocol.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Evaluation protocol method must be a non-empty string.")
    selected_mode = validate_v0_score_mode(
        score_mode
        or str(protocol.get("inference_objective", {}).get("score_mode", "f_plus_g"))
    )
    weight = _resolve_g_first_weight(
        protocol,
        score_mode=selected_mode,
        g_first_weight=g_first_weight,
    )
    run_mode = "smoke" if smoke else "pilot" if pilot else "formal"
    if selected_mode == FIRST_ACTION_SCORE_MODE:
        assert weight is not None
        return (
            f"{method}_cube_o50_{selected_mode}_alpha_"
            f"{_g_first_weight_slug(weight)}_{run_mode}"
        )
    return f"{method}_cube_o50_{selected_mode}_{run_mode}"


def _validate_pretrained_protocol(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    pretrained = protocol.get("pretrained_world_model")
    if not isinstance(pretrained, Mapping):
        raise ValueError("V0 evaluation requires pretrained_world_model metadata.")
    require_exact_values(
        pretrained,
        {
            "source_method": "lewm",
            "source_seed": 3072,
            "source_epoch": 10,
            "frozen": True,
        },
        label="pretrained_world_model",
    )
    if not is_lower_sha256(pretrained.get("checkpoint_sha256")):
        raise ValueError(
            "pretrained_world_model.checkpoint_sha256 must be lowercase SHA-256."
        )
    return pretrained


def _validate_predictor_protocol(predictor: Mapping[str, Any]) -> None:
    require_exact_values(
        predictor,
        {
            "objective_version": OBJECTIVE_VERSION,
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
            "bootstrap_action": "dataset_next_action",
            "goal_conditioning": "task_input",
            "actor": "none",
            "reward": "none",
        },
        label="predictor",
    )
    try:
        gamma = float(predictor.get("gamma"))
        ema_decay = float(predictor.get("target_ema_decay"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "predictor.gamma and predictor.target_ema_decay must lie in [0, 1)."
        ) from error
    if not 0.0 <= gamma < 1.0 or not 0.0 <= ema_decay < 1.0:
        raise ValueError(
            "predictor.gamma and predictor.target_ema_decay must lie in [0, 1)."
        )


def _validate_planning_protocol(
    planning: Mapping[str, Any], *, score_mode: str
) -> None:
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    if planning.get("solver") != "CEM":
        raise ValueError("V0 formal evaluation requires planning.solver='CEM'.")
    expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[score_mode]
    if planning.get("horizon") != expected_horizon:
        raise ValueError(
            f"V0 score_mode={score_mode!r} requires planning.horizon="
            f"{expected_horizon}."
        )
    for key, expected in FORMAL_O50_PLANNING.items():
        if key == "horizon":
            continue
        if planning.get(key) != expected:
            raise ValueError(
                f"The formal Cube O50 protocol requires planning.{key}={expected!r}."
            )
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed planning horizon.")
    if planning["action_block"] != planning["frame_skip"]:
        raise ValueError("Planning action blocks must match training frame skip.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The explicit plan exceeds the episode budget.")


def validate_frozen_actor_free_td_v0_evaluation_protocol(
    protocol: Mapping[str, Any],
    *,
    spec: FrozenActorFreeTDV0MethodSpec,
) -> None:
    """Validate one formal V0 C--G3 Cube O50 protocol."""

    require_exact_values(
        protocol,
        {
            "schema_version": 1,
            "implementation_version": IMPLEMENTATION_VERSION,
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "environment": "cube",
            "stage": "planner_evaluation",
        },
        label="protocol",
    )
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("V0 evaluation requires stable-worldmodel 0.1.1.")
    _validate_pretrained_protocol(protocol)
    _validate_dataset_protocol(protocol)
    if protocol.get("model", {}).get("embed_dim") != V0_STATE_DIM:
        raise ValueError("V0 evaluation requires model.embed_dim=192.")
    context = protocol.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("protocol.context must be a mapping.")
    require_exact_values(
        context,
        {
            "g_state_frames": 1,
            "lewm_rollout_history_frames": 3,
            "plan_config_history_len": 1,
        },
        label="context",
    )

    predictor = protocol.get("predictor")
    if not isinstance(predictor, Mapping):
        raise ValueError("protocol.predictor must be a mapping.")
    _validate_predictor_protocol(predictor)

    task_sampling = protocol.get("task_sampling")
    if not isinstance(task_sampling, Mapping):
        raise ValueError("protocol.task_sampling must be a mapping.")
    require_exact_values(
        task_sampling,
        {
            "sampling": "per_transition_bernoulli",
            "goal_probability": 0.5,
            "random_source": "isotropic_gaussian_sphere",
            "goal_source": "uniform_reachable_future_frozen_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "transition_minibatch",
        },
        label="task_sampling",
    )

    objective = protocol.get("joint_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("protocol.joint_objective must be a mapping.")
    missing_objective = set(spec.objective_keys) - objective.keys()
    if missing_objective:
        raise ValueError(f"joint_objective is missing {sorted(missing_objective)}.")
    # Adapter specs validate the deployment-shaped mapping, where the
    # method-owned objective remains nested under ``joint_objective``.
    spec.validate_method_config({"joint_objective": objective})

    inference = protocol.get("inference_objective")
    if not isinstance(inference, Mapping):
        raise ValueError("protocol.inference_objective must be a mapping.")
    score_mode = validate_v0_score_mode(str(inference.get("score_mode", "")))
    require_exact_values(
        inference,
        {
            "f_score": (
                "none" if score_mode == ROLLOUT_MEAN_SCORE_MODE else LEGACY_F_SCORE
            ),
            "f_score_reducer": (
                "none"
                if score_mode == ROLLOUT_MEAN_SCORE_MODE
                else LEGACY_F_SCORE_REDUCER
            ),
            "g_score": (
                ROLLOUT_MEAN_G_SCORE
                if score_mode == ROLLOUT_MEAN_SCORE_MODE
                else LEGACY_G_SCORE
            ),
            "f_plus_g_split": "first_h_minus_one_blocks_with_f_last_block_with_g",
            "f_plus_g_combination": (
                "prefix_final_f_cost_minus_gamma_power_tail_g_score"
            ),
            "g_only_horizon": 1,
            "goal_enters_predictor": True,
            "learned_actor": False,
            "training_only_auxiliary_used_at_evaluation": False,
        },
        label="inference_objective",
    )
    if score_mode == FIRST_ACTION_SCORE_MODE:
        _resolve_g_first_weight(
            protocol,
            score_mode=score_mode,
            g_first_weight=None,
        )
        if inference.get("score_definition") != FIRST_ACTION_SCORE_DEFINITION:
            raise ValueError(
                "inference_objective.score_definition must exactly describe "
                "the V0 first-action score."
            )
        for key in ROLLOUT_MEAN_ONLY_INFERENCE_KEYS:
            if key in inference:
                raise ValueError(
                    f"inference_objective.{key} requires "
                    "score_mode='g_only_f_rollout_mean'."
                )
    elif score_mode == ROLLOUT_MEAN_SCORE_MODE:
        if "g_first_weight" in inference:
            raise ValueError("g_first_weight requires score_mode='f_plus_g_first'.")
        require_exact_values(
            inference,
            {
                **ROLLOUT_MEAN_INFERENCE_FIELDS,
                "score_definition": ROLLOUT_MEAN_SCORE_DEFINITION,
            },
            label="inference_objective",
        )
    elif "g_first_weight" in inference or "score_definition" in inference:
        raise ValueError(
            "Special inference fields require f_plus_g_first or g_only_f_rollout_mean."
        )
    else:
        for key in ROLLOUT_MEAN_ONLY_INFERENCE_KEYS:
            if key in inference:
                raise ValueError(
                    f"inference_objective.{key} requires "
                    "score_mode='g_only_f_rollout_mean'."
                )
    training_only = inference.get("training_only_auxiliary", [])
    if not isinstance(training_only, list) or not all(
        isinstance(item, str) and item for item in training_only
    ):
        raise ValueError(
            "inference_objective.training_only_auxiliary must be a list of names."
        )
    for forbidden in (
        "neighbor_retrieval",
        "neighbor_action_scoring",
        "prefix_construction",
        "prefix_scoring",
    ):
        if inference.get(forbidden) not in (None, False, "none"):
            raise ValueError(
                f"inference_objective.{forbidden} is training-only and cannot "
                "be enabled during V0 evaluation."
            )

    planning = protocol.get("planning")
    if not isinstance(planning, Mapping):
        raise ValueError("protocol.planning must be a mapping.")
    _validate_planning_protocol(planning, score_mode=score_mode)
    if planning.get("history_len") != context["plan_config_history_len"]:
        raise ValueError(
            "planning.history_len must match context.plan_config_history_len."
        )

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes") != 50 or evaluation.get("goal_offset") != 50:
        raise ValueError("Actor-Free TD-LeWM V0 evaluation is locked to Cube O50/50.")


def load_frozen_actor_free_td_v0_evaluation_protocol(
    path: str | Path,
    *,
    spec: FrozenActorFreeTDV0MethodSpec,
) -> dict[str, Any]:
    protocol = _load_protocol_mapping(Path(path), seen=frozenset())
    validate_frozen_actor_free_td_v0_evaluation_protocol(protocol, spec=spec)
    return protocol


def configure_frozen_actor_free_td_v0_evaluation_mode(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> dict[str, Any]:
    """Apply a run/score override while preserving the V0 horizon contract."""

    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    configured = deepcopy(dict(protocol))
    selected_mode = validate_v0_score_mode(
        score_mode
        or str(configured.get("inference_objective", {}).get("score_mode", "f_plus_g"))
    )
    configured.setdefault("inference_objective", {})["score_mode"] = selected_mode
    _configure_first_action_score(
        configured,
        score_mode=selected_mode,
        g_first_weight=g_first_weight,
    )
    _configure_rollout_mean_score(configured, score_mode=selected_mode)
    configured.setdefault("planning", {})["horizon"] = FORMAL_HORIZON_BY_SCORE_MODE[
        selected_mode
    ]
    if smoke:
        configured["id"] = f"{configured['id']}_smoke"
        configured["evaluation"]["episodes"] = 1
        configured["planning"].update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
    elif pilot:
        configured["id"] = f"{configured['id']}_pilot"
        configured["evaluation"]["episodes"] = 10
        configured["planning"].update(
            {
                "candidates": 128,
                "iterations": 10,
                "elites": 16,
                "episode_budget": 100,
            }
        )
    return configured


def validate_frozen_actor_free_td_v0_checkpoint_protocol(
    *,
    payload: Mapping[str, Any],
    predictor_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    spec: FrozenActorFreeTDV0MethodSpec,
    require_formal_completion: bool = True,
) -> None:
    """Bind a single-predictor V0 checkpoint to its evaluation protocol."""

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
    expected_predictor = protocol["predictor"]
    for key, expected in expected_predictor.items():
        actual = predictor_config.get(key)
        if key in {"gamma", "target_ema_decay"}:
            matches = actual is not None and np.isclose(float(actual), float(expected))
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Predictor checkpoint {key} differs from the evaluation protocol."
            )
    if predictor_config.get("num_parallel") != 1:
        raise ValueError("V0 deployment requires exactly one predictor.")
    if predictor_config.get("joint_objective") != protocol["joint_objective"]:
        raise ValueError(
            "Predictor checkpoint joint_objective differs from the evaluation protocol."
        )
    if (
        "task_sampling" in protocol
        and predictor_config.get("task_sampling") != protocol["task_sampling"]
    ):
        raise ValueError(
            "Predictor checkpoint task_sampling differs from the evaluation protocol."
        )
    pretrained = _validate_pretrained_protocol(protocol)
    checkpoint_pretrained = predictor_config.get("pretrained_world_model")
    if not isinstance(checkpoint_pretrained, Mapping):
        raise ValueError("Predictor checkpoint is missing pretrained_world_model.")
    if (
        checkpoint_pretrained.get("checkpoint_sha256")
        != pretrained["checkpoint_sha256"]
    ):
        raise ValueError(
            "Pretrained source checkpoint SHA differs from the evaluation protocol."
        )
    provenance = payload.get("pretrained_world_model_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V0 checkpoint is missing pretrained_world_model_provenance.")
    if provenance.get("source_checkpoint_sha256") != pretrained["checkpoint_sha256"]:
        raise ValueError(
            "Pretrained source checkpoint SHA differs from checkpoint provenance."
        )
    if require_formal_completion:
        require_exact_values(
            payload,
            {
                "epoch": FORMAL_DEPLOYMENT_EPOCH,
                "global_step": FORMAL_DEPLOYMENT_GLOBAL_STEP,
            },
            label="checkpoint",
        )


def evaluate_frozen_actor_free_td_v0(
    *,
    spec: FrozenActorFreeTDV0MethodSpec,
    checkpoint_loader: CheckpointLoader,
    policy_factory: PolicyFactory,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
    pilot: bool = False,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> dict[str, Any]:
    """Run the audited Stable World Model Cube evaluation for one V0 method."""

    formal_protocol = load_frozen_actor_free_td_v0_evaluation_protocol(
        protocol_path,
        spec=spec,
    )
    protocol = configure_frozen_actor_free_td_v0_evaluation_mode(
        formal_protocol,
        smoke=smoke,
        pilot=pilot,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_source = _resolve_frozen_dataset_source(dataset_path, protocol["dataset"])
    compatibility = prepare_cloud_runtime()

    import stable_worldmodel as swm
    import torch
    from torchvision.transforms import v2 as transforms

    package_version = importlib.metadata.version("stable-worldmodel")
    expected_version = protocol["runtime"]["stable_worldmodel_version"]
    if package_version != expected_version:
        raise RuntimeError(
            f"Expected stable-worldmodel {expected_version}, found {package_version}."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_file = _resolve_joint_checkpoint(checkpoint_path)
    world_model, predictor, predictor_config, payload = checkpoint_loader(
        checkpoint_file,
        map_location=device,
    )
    validate_frozen_actor_free_td_v0_checkpoint_protocol(
        payload=payload,
        predictor_config=predictor_config,
        protocol=formal_protocol,
        spec=spec,
        require_formal_completion=not (smoke or pilot),
    )

    dataset_cfg = protocol["dataset"]
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=dataset_source["format"],
        keys_to_load=list(dataset_cfg["keys_to_load"]),
    )
    actual_episodes = len(dataset.lengths)
    actual_transitions = int(np.asarray(dataset.lengths).sum())
    if actual_episodes != dataset_cfg["expected_episodes"]:
        raise ValueError("Dataset episode count differs from protocol.")
    if actual_transitions != dataset_cfg["expected_transitions"]:
        raise ValueError("Dataset transition count differs from protocol.")
    expected_action_dim = int(dataset.get_dim("action")) * int(
        protocol["planning"]["action_block"]
    )
    if int(predictor_config["action_dim"]) != expected_action_dim:
        raise ValueError("The V0 predictor action-block dimension is incompatible.")

    evaluation = protocol["evaluation"]
    planning = protocol["planning"]
    episode_indices, start_steps, valid_ranks = sample_start_goal_pairs(
        np.asarray(dataset.lengths),
        goal_offset=evaluation["goal_offset"],
        episodes=evaluation["episodes"],
        seed=planning["planning_seed"],
    )
    selection = {
        "episode_indices": episode_indices,
        "start_steps": start_steps,
        "goal_steps": start_steps + evaluation["goal_offset"],
        "valid_row_ranks": valid_ranks,
    }
    _write_json(output_dir / "episode_selection.json", selection)
    action_processor, action_stats = _load_action_processor(
        dataset,
        output_dir / "action_normalization.json",
    )

    world_model = world_model.to(device).eval()
    world_model.requires_grad_(False)
    predictor = predictor.to(device).eval()
    predictor.requires_grad_(False)
    world_parameter_count = sum(
        parameter.numel() for parameter in world_model.parameters()
    )
    predictor_parameter_count = sum(
        parameter.numel() for parameter in predictor.parameters()
    )
    image = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image["mean"], std=image["std"]),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    policy_kwargs = {
        "world_model": world_model,
        "predictor": predictor,
        "planning": planning,
        "gamma": float(protocol["predictor"]["gamma"]),
        "process": {"action": action_processor},
        "transform": {"pixels": image_transform, "goal": image_transform},
        "device": device,
        "score_mode": protocol["inference_objective"]["score_mode"],
    }
    if protocol["inference_objective"]["score_mode"] == FIRST_ACTION_SCORE_MODE:
        policy_kwargs["g_first_weight"] = protocol["inference_objective"][
            "g_first_weight"
        ]
    policy = policy_factory(
        **policy_kwargs,
    )

    runtime = {
        "stable_worldmodel": package_version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tdwm_git_revision": _git_revision(),
        "device": device,
        "stablewm_home": os.environ.get("STABLEWM_HOME"),
        "compatibility_adapter": compatibility,
    }
    if torch.cuda.is_available():
        runtime["cuda_device"] = torch.cuda.get_device_name(0)
    manifest = {
        "score_mode": protocol["inference_objective"]["score_mode"],
        "protocol": protocol,
        "formal_protocol": formal_protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "dataset": {
            **dataset_source,
            "episodes": actual_episodes,
            "transitions": actual_transitions,
        },
        "checkpoint": {
            "path": str(checkpoint_file),
            "sha256": _sha256(checkpoint_file),
            "method": payload["method"],
            "method_family": payload["method_family"],
            "variant": payload["variant"],
            "implementation_version": payload["implementation_version"],
            "objective_version": payload["objective_version"],
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "formal_completion_required": not (smoke or pilot),
            "predictor_config": predictor_config,
            "pretrained_world_model_provenance": deepcopy(
                payload["pretrained_world_model_provenance"]
            ),
        },
        "selection": selection,
        "normalization": {"action": action_stats},
        "runtime": runtime,
    }
    manifest.update(_first_action_output_metadata(protocol, planning))
    manifest.update(_rollout_mean_output_metadata(protocol, planning))
    _write_json(output_dir / "protocol_manifest.json", manifest)

    world_cfg = protocol["world"]
    world = swm.World(
        world_cfg["env_name"],
        num_envs=evaluation["episodes"],
        image_shape=(world_cfg["image_size"], world_cfg["image_size"]),
        max_episode_steps=planning["episode_budget"],
        env_type=world_cfg["env_type"],
        ob_type=world_cfg["ob_type"],
        multiview=world_cfg["multiview"],
        width=world_cfg["image_size"],
        height=world_cfg["image_size"],
        visualize_info=world_cfg["visualize_info"],
        terminate_at_goal=world_cfg["terminate_at_goal"],
    )
    world.set_policy(policy)
    callables = [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]
    started = time.time()
    try:
        with torch.inference_mode():
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=episode_indices.tolist(),
                start_steps=start_steps.tolist(),
                goal_offset=evaluation["goal_offset"],
                eval_budget=planning["episode_budget"],
                callables=callables,
                video=output_dir / "videos" if video else None,
            )
    finally:
        world.close()
    result = {
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "world_model_parameter_count": world_parameter_count,
        "predictor_parameter_count": predictor_parameter_count,
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "score_mode": protocol["inference_objective"]["score_mode"],
        "planning_horizon": planning["horizon"],
        "smoke": smoke,
        "pilot": pilot,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    result.update(_first_action_output_metadata(protocol, planning))
    result.update(_rollout_mean_output_metadata(protocol, planning))
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "FIRST_ACTION_SCORE_DEFINITION",
    "FIRST_ACTION_SCORE_MODE",
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "FORMAL_O50_PLANNING",
    "ROLLOUT_MEAN_G_SCORE",
    "ROLLOUT_MEAN_INFERENCE_FIELDS",
    "ROLLOUT_MEAN_ONLY_INFERENCE_KEYS",
    "ROLLOUT_MEAN_SCORE_DEFINITION",
    "ROLLOUT_MEAN_SCORE_MODE",
    "actor_free_td_v0_output_directory_name",
    "configure_frozen_actor_free_td_v0_evaluation_mode",
    "evaluate_frozen_actor_free_td_v0",
    "load_frozen_actor_free_td_v0_evaluation_protocol",
    "validate_frozen_actor_free_td_v0_checkpoint_protocol",
    "validate_frozen_actor_free_td_v0_evaluation_protocol",
    "validate_v0_score_mode",
]
