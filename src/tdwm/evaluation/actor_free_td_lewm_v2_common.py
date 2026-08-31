"""Controlled Cube O50 evaluation for Actor-Free TD-LeWM V2 methods."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters.actor_free_td_lewm_v2_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    OBJECTIVE_VERSION,
    SOURCE_ARTIFACTS,
    ActorFreeTDV2MethodSpec,
    require_exact_values,
    validate_source_v1_v2,
)
from tdwm.adapters.frozen_actor_free_td_common import (
    FORMAL_DEPLOYMENT_EPOCH,
    FORMAL_DEPLOYMENT_GLOBAL_STEP,
    is_lower_sha256,
)
from tdwm.adapters.frozen_actor_free_td_v1_common import SCORE_MODES
from tdwm.evaluation.frozen_actor_free_td_common import _validate_dataset_protocol
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    FORMAL_HORIZON_BY_SCORE_MODE,
    FORMAL_O50_PLANNING,
    evaluate_actor_free_td_predictor_runtime,
    validate_v1_raw_action_compatibility,
)
from tdwm.evaluation.lewm_checkpoint import REQUIRED_PLANNING_KEYS
from tdwm.methods.actor_free_td_lewm_v2 import (
    V2_ACTION_DIM,
    V2_ACTION_EMBEDDING_DIM,
    V2_OUTPUT_DIM,
    V2_RAW_ACTION_DIM,
    V2_STATE_DIM,
    V2_TASK_DIM,
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
        raise ValueError("V2 evaluation protocol inheritance contains a cycle.")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("V2 evaluation protocol must contain a mapping.")
    current = dict(value)
    parent = current.get("extends")
    if parent is None:
        return current
    if not isinstance(parent, str) or not parent:
        raise ValueError("protocol.extends must be a non-empty relative path.")
    base = _load_protocol_mapping(
        (resolved.parent / parent).resolve(), seen=seen | {resolved}
    )
    return _deep_merge(base, current)


def validate_v2_score_mode(score_mode: str) -> str:
    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"score_mode {score_mode!r} is incompatible with V2; expected one "
            f"of {sorted(SCORE_MODES)}."
        )
    return score_mode


def actor_free_td_v2_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> str:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    method = protocol.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Evaluation protocol method must be a non-empty string.")
    selected = validate_v2_score_mode(
        score_mode
        or str(protocol.get("inference_objective", {}).get("score_mode", "f_plus_g"))
    )
    run_mode = "smoke" if smoke else "pilot" if pilot else "formal"
    return f"{method}_cube_o50_{selected}_{run_mode}"


def _validate_predictor_protocol(predictor: Mapping[str, Any]) -> None:
    require_exact_values(
        predictor,
        {
            "objective_version": OBJECTIVE_VERSION,
            "architecture": "td_jepa_forward_map_v1",
            "state_dim": V2_STATE_DIM,
            "raw_action_dim": V2_RAW_ACTION_DIM,
            "action_dim": V2_ACTION_DIM,
            "action_embedding_dim": V2_ACTION_EMBEDDING_DIM,
            "task_dim": V2_TASK_DIM,
            "output_dim": V2_OUTPUT_DIM,
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
    for key in ("gamma", "target_ema_decay", "target_world_ema_decay"):
        try:
            value = float(predictor.get(key))
        except (TypeError, ValueError) as error:
            raise ValueError(f"predictor.{key} must lie in [0, 1).") from error
        if not 0.0 <= value < 1.0:
            raise ValueError(f"predictor.{key} must lie in [0, 1).")


def _validate_planning_protocol(
    planning: Mapping[str, Any], *, score_mode: str
) -> None:
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    if planning.get("solver") != "CEM":
        raise ValueError("V2 formal evaluation requires planning.solver='CEM'.")
    expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[score_mode]
    if planning.get("horizon") != expected_horizon:
        raise ValueError(
            f"V2 score_mode={score_mode!r} requires planning.horizon="
            f"{expected_horizon}."
        )
    for key, expected in FORMAL_O50_PLANNING.items():
        if key != "horizon" and planning.get(key) != expected:
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


def validate_actor_free_td_v2_evaluation_protocol(
    protocol: Mapping[str, Any],
    *,
    spec: ActorFreeTDV2MethodSpec,
) -> None:
    """Fail closed on V1 artifacts or drift from the V2 deployment contract."""

    require_exact_values(
        protocol,
        {
            "schema_version": 1,
            "method": spec.method,
            "method_family": spec.method_family,
            "variant": spec.variant,
            "implementation_version": spec.implementation_version,
            "environment": "cube",
            "stage": spec.evaluation_stage,
            "initialization": spec.initialization,
        },
        label="protocol",
    )
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("V2 evaluation requires stable-worldmodel 0.1.1.")
    pretrained = protocol.get("pretrained_world_model")
    if not isinstance(pretrained, Mapping):
        raise ValueError("V2 evaluation requires pretrained_world_model metadata.")
    require_exact_values(
        pretrained,
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
    if not is_lower_sha256(pretrained.get("checkpoint_sha256")):
        raise ValueError("pretrained_world_model.checkpoint_sha256 must be SHA-256.")
    source_v1 = protocol.get("source_v1")
    if not isinstance(source_v1, Mapping):
        raise ValueError("V2 evaluation requires source_v1 metadata.")
    validate_source_v1_v2(source_v1, spec=spec)
    source_artifacts = protocol.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise ValueError("V2 evaluation requires source_artifacts metadata.")
    require_exact_values(
        source_artifacts,
        SOURCE_ARTIFACTS,
        label="source_artifacts",
    )
    _validate_dataset_protocol(protocol)

    if protocol.get("model", {}).get("embed_dim") != V2_STATE_DIM:
        raise ValueError("V2 evaluation requires model.embed_dim=192.")
    context = protocol.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("protocol.context must be a mapping.")
    require_exact_values(
        context,
        {
            "g_state_frames": 1,
            "lewm_rollout_history_frames": 3,
            "plan_config_history_len": 1,
            "real_branch": "online_encoded_state",
            "predicted_branch": "online_lewm_one_step_prediction",
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
            "goal_source": "uniform_reachable_future_ema_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "flattened_transition_minibatch",
        },
        label="task_sampling",
    )
    objective = protocol.get("joint_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("protocol.joint_objective must be a mapping.")
    missing = set(spec.objective_keys) - objective.keys()
    if missing:
        raise ValueError(f"joint_objective is missing {sorted(missing)}.")
    local_prediction_contract: dict[str, Any] = {
        "local_prediction": spec.local_prediction,
    }
    if spec.local_prediction_target is not None:
        local_prediction_contract.update(
            {
                "local_prediction_target": spec.local_prediction_target,
                "local_prediction_target_gradient": (
                    spec.local_prediction_target_gradient
                ),
            }
        )
    require_exact_values(
        objective,
        {
            **local_prediction_contract,
            "local_prediction_weight": 1.0,
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
        },
        label="joint_objective",
    )
    spec.validate_method_config({"joint_objective": objective})

    inference = protocol.get("inference_objective")
    if not isinstance(inference, Mapping):
        raise ValueError("protocol.inference_objective must be a mapping.")
    require_exact_values(
        inference,
        {
            "f_score": "lewm_rollout_goal_distance",
            "f_score_reducer": "final_predicted_latent_summed_mse",
            "g_score": spec.inference_g_score,
            "f_plus_g_split": "first_h_minus_one_blocks_with_f_last_block_with_g",
            "f_plus_g_combination": (
                "prefix_final_f_cost_minus_gamma_power_tail_g_score"
            ),
            "g_only_horizon": 1,
            "goal_enters_predictor": True,
            "learned_actor": False,
            "deployed_world_model": spec.deployed_world_model,
            "deployed_predictor": spec.deployed_predictor,
            "target_modules_used_at_evaluation": False,
            "deployed_modules_frozen": True,
            "training_only_auxiliary_used_at_evaluation": False,
        },
        label="inference_objective",
    )
    score_mode = validate_v2_score_mode(str(inference.get("score_mode", "")))
    training_only = inference.get("training_only_auxiliary")
    if not isinstance(training_only, list) or not all(
        isinstance(item, str) and item for item in training_only
    ):
        raise ValueError("training_only_auxiliary must be a list of names.")
    for forbidden in (
        "neighbor_retrieval",
        "neighbor_action_scoring",
        "prefix_construction",
        "prefix_scoring",
        "predicted_td_branch",
        "sigreg",
    ):
        if inference.get(forbidden) not in (None, False, "none"):
            raise ValueError(f"inference_objective.{forbidden} is training-only.")

    planning = protocol.get("planning")
    if not isinstance(planning, Mapping):
        raise ValueError("protocol.planning must be a mapping.")
    _validate_planning_protocol(planning, score_mode=score_mode)
    if planning.get("history_len") != context["plan_config_history_len"]:
        raise ValueError("planning.history_len must match the context contract.")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes") != 50 or evaluation.get("goal_offset") != 50:
        raise ValueError("Actor-Free TD-LeWM V2 evaluation is locked to Cube O50/50.")


def load_actor_free_td_v2_evaluation_protocol(
    path: str | Path,
    *,
    spec: ActorFreeTDV2MethodSpec,
) -> dict[str, Any]:
    protocol = _load_protocol_mapping(Path(path), seen=frozenset())
    validate_actor_free_td_v2_evaluation_protocol(protocol, spec=spec)
    return protocol


def configure_actor_free_td_v2_evaluation_mode(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> dict[str, Any]:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    configured = copy.deepcopy(dict(protocol))
    selected = validate_v2_score_mode(
        score_mode
        or str(configured.get("inference_objective", {}).get("score_mode", "f_plus_g"))
    )
    configured["inference_objective"]["score_mode"] = selected
    configured["planning"]["horizon"] = FORMAL_HORIZON_BY_SCORE_MODE[selected]
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
            {"candidates": 128, "iterations": 10, "elites": 16}
        )
    return configured


def validate_actor_free_td_v2_checkpoint_protocol(
    *,
    payload: Mapping[str, Any],
    predictor_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    spec: ActorFreeTDV2MethodSpec,
    require_formal_completion: bool = True,
) -> None:
    for values, label in (
        (payload, "checkpoint"),
        (predictor_config, "predictor_config"),
    ):
        require_exact_values(
            values,
            {
                "method": spec.method,
                "method_family": spec.method_family,
                "variant": spec.variant,
                "implementation_version": spec.implementation_version,
                "objective_version": OBJECTIVE_VERSION,
                "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            },
            label=label,
        )
    for key, expected in protocol["predictor"].items():
        actual = predictor_config.get(key)
        if key in {"gamma", "target_ema_decay", "target_world_ema_decay"}:
            matches = actual is not None and np.isclose(float(actual), float(expected))
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"V2 predictor checkpoint {key} differs from the evaluation protocol."
            )
    for key in (
        "task_sampling",
        "joint_objective",
        "source_v1",
        "source_artifacts",
    ):
        if predictor_config.get(key) != protocol[key]:
            raise ValueError(f"V2 predictor checkpoint {key} differs from protocol.")
    provenance = payload.get("source_v1_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V2 checkpoint is missing source_v1_provenance.")
    source = protocol["source_v1"]
    require_exact_values(
        provenance,
        {
            "checkpoint_sha256": source["checkpoint_sha256"],
            "source_epoch": source["source_epoch"],
            "source_global_step": source["source_global_step"],
            "optimizer_state_loaded": False,
            "target_world_initialization": "copy_of_v1_online_world_model",
        },
        label="source_v1_provenance",
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


def validate_v2_raw_action_compatibility(**kwargs) -> None:
    validate_v1_raw_action_compatibility(**kwargs)


def evaluate_actor_free_td_v2(
    *,
    spec: ActorFreeTDV2MethodSpec,
    checkpoint_loader,
    policy_factory,
    **kwargs,
) -> dict[str, Any]:
    return evaluate_actor_free_td_predictor_runtime(
        spec=spec,
        checkpoint_loader=checkpoint_loader,
        policy_factory=policy_factory,
        protocol_loader=load_actor_free_td_v2_evaluation_protocol,
        protocol_configurer=configure_actor_free_td_v2_evaluation_mode,
        checkpoint_validator=validate_actor_free_td_v2_checkpoint_protocol,
        raw_action_validator=validate_v2_raw_action_compatibility,
        checkpoint_provenance_keys=("source_v1_provenance",),
        **kwargs,
    )


__all__ = [
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "FORMAL_O50_PLANNING",
    "actor_free_td_v2_output_directory_name",
    "configure_actor_free_td_v2_evaluation_mode",
    "evaluate_actor_free_td_v2",
    "load_actor_free_td_v2_evaluation_protocol",
    "validate_actor_free_td_v2_checkpoint_protocol",
    "validate_actor_free_td_v2_evaluation_protocol",
    "validate_v2_raw_action_compatibility",
    "validate_v2_score_mode",
]
