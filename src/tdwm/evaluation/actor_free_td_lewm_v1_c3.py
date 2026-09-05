"""Controlled Cube O50 evaluation for V1-C3 terminal EMA State-V planning."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from tdwm.adapters.actor_free_td_lewm_v1_c3 import (
    DEFAULT_STATE_V_FIRST_Q_WEIGHT,
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    METHOD,
    METHOD_FAMILY,
    OBJECTIVE_VERSION,
    PARENT_CHECKPOINT_SHA256,
    PARENT_METHOD,
    STATE_V_CONSTANT_SANITY_OFFSET,
    STATE_V_FIRST_ACTION_SCORE_MODES,
    STATE_V_FIRST_Q2_SCORE_MODE,
    STATE_V_FIRST_Q_SCORE_MODE,
    STATE_V_SCORE_MODE,
    STATE_V_SCORE_MODES,
    VARIANT,
    load_actor_free_td_lewm_v1_c3_checkpoint,
    make_actor_free_td_lewm_v1_c3_policy,
    normalize_state_v_first_q_weight,
    validate_actor_free_td_lewm_v1_c3_payload,
)
from tdwm.adapters.frozen_actor_free_td_common import is_lower_sha256
from tdwm.adapters.frozen_actor_free_td_v1_common import FIRST_Q2_STD_EPSILON
from tdwm.adapters.runtime import prepare_cloud_runtime
from tdwm.evaluation.frozen_actor_free_td_common import (
    _resolve_frozen_dataset_source,
    _resolve_joint_checkpoint,
    _validate_dataset_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import _load_protocol_mapping
from tdwm.evaluation.lewm_checkpoint import (
    REQUIRED_PLANNING_KEYS,
    _git_revision,
    _jsonable,
    _sha256,
    _write_json,
    sample_start_goal_pairs,
)
from tdwm.evaluation.mc_gt_lewm import _load_action_processor
from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
)

FORMAL_SELECTION_SHA256 = (
    "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
)
FORMAL_O50_PLANNING = {
    "solver": "CEM",
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
    "history_len": 1,
    "warm_start": True,
    "initial_distribution": "cem_gaussian_no_actor",
}
STATE_V_SCORE_DEFINITION = {
    "formula": "target_state_v(F_frozen_rollout_5(z0,A1:A5),z_goal)",
    "optimization": "cem_minimize",
    "f_rollout": "full_five_action_blocks_A1_through_A5",
    "state_input": "terminal_imagined_frozen_lewm_latent",
    "goal_input": "frozen_lewm_encoded_goal_latent",
    "critic": "ema_target_state_value",
    "online_critic_used": False,
    "parent_online_g_used": False,
    "parent_target_g_used": False,
    "terminal_latent_l2_used": False,
    "constant_shift_sanity": "25_plus_value_has_identical_ranking_and_action",
    "constant_shift": STATE_V_CONSTANT_SANITY_OFFSET,
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}
STATE_V_FIRST_Q_SCORE_DEFINITION = {
    "formula": (
        "target_state_v(F_frozen_rollout_5(z0,A1:A5),z_goal) "
        "- g_first_weight * q_first"
    ),
    "optimization": "cem_minimize",
    "f_rollout": "full_five_action_blocks_A1_through_A5",
    "state_input": "terminal_imagined_frozen_lewm_latent",
    "goal_input": "frozen_lewm_encoded_goal_latent",
    "critic": "ema_target_state_value",
    "online_critic_used": False,
    "parent_online_g_used": True,
    "parent_target_g_used": False,
    "terminal_latent_l2_used": False,
    "q_first": "dot(G_online(z0, frozen_E_A(A1), w_goal), w_goal)",
    "q_first_state": "current_frozen_lewm_encoder_state_z0",
    "q_first_action": "first_candidate_raw_action_block_A1",
    "q_first_action_processing": "frozen_shared_lewm_action_encoder_to_192d",
    "q_first_task": "sqrt_dim_l2_normalized_goal_vector",
    "q_first_discount": "none",
    "normalization": "none_raw_scores",
    "raw_state_v_constant_shift_sanity": (
        "25_plus_value_has_identical_combined_ranking"
    ),
    "raw_state_v_constant_shift": STATE_V_CONSTANT_SANITY_OFFSET,
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}
STATE_V_FIRST_Q2_SCORE_DEFINITION = {
    "formula": (
        "zscore_samples(target_state_v(F_frozen_rollout_5(z0,A1:A5),z_goal)) "
        "- g_first_weight * zscore_samples(q_first)"
    ),
    "optimization": "cem_minimize",
    "f_rollout": "full_five_action_blocks_A1_through_A5",
    "state_input": "terminal_imagined_frozen_lewm_latent",
    "goal_input": "frozen_lewm_encoded_goal_latent",
    "critic": "ema_target_state_value",
    "online_critic_used": False,
    "parent_online_g_used": True,
    "parent_target_g_used": False,
    "terminal_latent_l2_used": False,
    "q_first": "dot(G_online(z0, frozen_E_A(A1), w_goal), w_goal)",
    "q_first_state": "current_frozen_lewm_encoder_state_z0",
    "q_first_action": "first_candidate_raw_action_block_A1",
    "q_first_action_processing": "frozen_shared_lewm_action_encoder_to_192d",
    "q_first_task": "sqrt_dim_l2_normalized_goal_vector",
    "q_first_discount": "none",
    "normalization": "population_z_score",
    "normalization_axis": "cem_candidate_sample_axis_dim_1_per_environment",
    "normalization_scope": "independent_per_get_cost_call",
    "normalization_epsilon": FIRST_Q2_STD_EPSILON,
    "degenerate_signal": "zeros_when_population_std_lte_epsilon",
    "raw_state_v_constant_shift_sanity": (
        "25_plus_value_has_identical_raw_state_v_ranking"
    ),
    "raw_state_v_constant_shift": STATE_V_CONSTANT_SANITY_OFFSET,
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}


def _state_v_score_definition(score_mode: str) -> dict[str, Any]:
    if score_mode == STATE_V_SCORE_MODE:
        return STATE_V_SCORE_DEFINITION
    if score_mode == STATE_V_FIRST_Q_SCORE_MODE:
        return STATE_V_FIRST_Q_SCORE_DEFINITION
    if score_mode == STATE_V_FIRST_Q2_SCORE_MODE:
        return STATE_V_FIRST_Q2_SCORE_DEFINITION
    raise ValueError(f"Unsupported V1-C3 score mode {score_mode!r}.")


def _resolve_state_v_first_q_weight(
    protocol: Mapping[str, Any],
    *,
    score_mode: str,
    g_first_weight: float | None,
) -> float | None:
    if score_mode not in STATE_V_FIRST_ACTION_SCORE_MODES:
        if g_first_weight is not None:
            raise ValueError(
                "g_first_weight is only valid for a V1-C3 first-action score."
            )
        return None
    inference = protocol.get("inference_objective", {})
    configured = (
        inference.get("g_first_weight") if isinstance(inference, Mapping) else None
    )
    raw_weight = g_first_weight
    if raw_weight is None:
        raw_weight = (
            configured if configured is not None else DEFAULT_STATE_V_FIRST_Q_WEIGHT
        )
    return normalize_state_v_first_q_weight(raw_weight)


def _state_v_first_q_weight_slug(weight: float) -> str:
    value = format(normalize_state_v_first_q_weight(weight), ".15g").lower()
    return value.replace("+", "").replace("-", "m").replace(".", "p")


def _require_exact_values(
    values: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"{label}.{key} must be {expected_value!r}.")


def _validate_source_v1_c(source: Mapping[str, Any]) -> None:
    _require_exact_values(
        source,
        {
            "method": PARENT_METHOD,
            "method_family": METHOD_FAMILY,
            "variant": "c",
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": 0,
            "deployment_checkpoint_version": 1,
            "source_seed": 3072,
            "source_epoch": 10,
            "source_global_step": 127_960,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parameter_state": "strict_all_model_parameters_frozen",
            "optimizer_state": "not_loaded",
            "scheduler_state": "not_loaded",
            "epoch_and_global_step": "reset",
        },
        label="source_v1_c",
    )


def _validate_state_critic(critic: Mapping[str, Any]) -> None:
    _require_exact_values(
        critic,
        {
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
        label="state_critic",
    )


def validate_actor_free_td_lewm_v1_c3_evaluation_protocol(
    protocol: Mapping[str, Any],
) -> None:
    """Fail closed unless the complete formal C3/O50 contract is present."""

    _require_exact_values(
        protocol,
        {
            "schema_version": 1,
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "variant": VARIANT,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "environment": "cube",
            "stage": "planner_evaluation",
        },
        label="protocol",
    )
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("V1-C3 evaluation requires stable-worldmodel 0.1.1.")
    _validate_dataset_protocol(protocol)

    pretrained = protocol.get("pretrained_world_model")
    if not isinstance(pretrained, Mapping):
        raise ValueError("protocol.pretrained_world_model must be a mapping.")
    _require_exact_values(
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
        raise ValueError("pretrained_world_model checkpoint SHA must be lowercase.")

    source = protocol.get("source_v1_c")
    if not isinstance(source, Mapping):
        raise ValueError("protocol.source_v1_c must be a mapping.")
    _validate_source_v1_c(source)
    if protocol.get("model", {}).get("embed_dim") != V1_STATE_DIM:
        raise ValueError("V1-C3 evaluation requires model.embed_dim=192.")
    context = protocol.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("protocol.context must be a mapping.")
    _require_exact_values(
        context,
        {
            "g_state_frames": 1,
            "lewm_rollout_history_frames": 3,
            "plan_config_history_len": 1,
        },
        label="context",
    )
    critic = protocol.get("state_critic")
    if not isinstance(critic, Mapping):
        raise ValueError("protocol.state_critic must be a mapping.")
    _validate_state_critic(critic)
    goal_sampling = protocol.get("goal_sampling")
    if not isinstance(goal_sampling, Mapping):
        raise ValueError("protocol.goal_sampling must be a mapping.")
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
        label="goal_sampling",
    )
    objective = protocol.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("protocol.objective must be a mapping.")
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
        label="objective",
    )

    planning = protocol.get("planning")
    if not isinstance(planning, Mapping):
        raise ValueError("protocol.planning must be a mapping.")
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    _require_exact_values(planning, FORMAL_O50_PLANNING, label="planning")
    if planning.get("history_len") != context["plan_config_history_len"]:
        raise ValueError("planning.history_len must match the context lock.")

    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("protocol.evaluation must be a mapping.")
    _require_exact_values(
        evaluation,
        {
            "episodes": 50,
            "goal_offset": 50,
            "start_goal_source": "same_dataset_episode",
            "selection_sha256": FORMAL_SELECTION_SHA256,
        },
        label="evaluation",
    )
    inference = protocol.get("inference_objective")
    if not isinstance(inference, Mapping):
        raise ValueError("protocol.inference_objective must be a mapping.")
    score_mode = inference.get("score_mode")
    if score_mode not in STATE_V_SCORE_MODES:
        raise ValueError(
            f"inference_objective.score_mode must be one of "
            f"{sorted(STATE_V_SCORE_MODES)}."
        )
    uses_parent_g = score_mode in STATE_V_FIRST_ACTION_SCORE_MODES
    inference_expected = {
        "score_mode": score_mode,
        "score_definition": _state_v_score_definition(score_mode),
        "learned_actor": False,
        "parent_g_used": uses_parent_g,
        "terminal_goal_distance_used": False,
        "critic": "ema_target",
        "replanning": "every_action_block",
    }
    if uses_parent_g:
        weight = normalize_state_v_first_q_weight(inference.get("g_first_weight"))
        inference_expected.update(
            {
                "parent_g": "online_predictor",
                "g_first_weight": weight,
            }
        )
    elif "parent_g" in inference or "g_first_weight" in inference:
        raise ValueError(
            "State-V-only inference must not contain parent_g or g_first_weight."
        )
    _require_exact_values(
        inference,
        inference_expected,
        label="inference_objective",
    )
    checkpoint = protocol.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("protocol.checkpoint must be a mapping.")
    _require_exact_values(
        checkpoint,
        {
            "source": "actor_free_td_lewm_v1_c3_deployment_export",
            "contains_world_model": True,
            "contains_parent_predictor_pair": True,
            "contains_state_critic_pair": True,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "formal_epoch": 12,
            "formal_logical_epoch": 12,
            "formal_global_step": 12_000,
        },
        label="checkpoint",
    )


def load_actor_free_td_lewm_v1_c3_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    protocol = _load_protocol_mapping(Path(path), seen=frozenset())
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol(protocol)
    return protocol


def configure_actor_free_td_lewm_v1_c3_evaluation_mode(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> dict[str, Any]:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    configured = deepcopy(dict(protocol))
    selected_score_mode = score_mode or str(
        configured["inference_objective"]["score_mode"]
    )
    if selected_score_mode not in STATE_V_SCORE_MODES:
        raise ValueError(
            f"Unsupported V1-C3 score mode {selected_score_mode!r}; expected one "
            f"of {sorted(STATE_V_SCORE_MODES)}."
        )
    weight = _resolve_state_v_first_q_weight(
        configured,
        score_mode=selected_score_mode,
        g_first_weight=g_first_weight,
    )
    uses_parent_g = selected_score_mode in STATE_V_FIRST_ACTION_SCORE_MODES
    inference = configured["inference_objective"]
    inference.update(
        {
            "score_mode": selected_score_mode,
            "score_definition": deepcopy(_state_v_score_definition(selected_score_mode)),
            "learned_actor": False,
            "parent_g_used": uses_parent_g,
            "terminal_goal_distance_used": False,
            "critic": "ema_target",
            "replanning": "every_action_block",
        }
    )
    if uses_parent_g:
        assert weight is not None
        inference["parent_g"] = "online_predictor"
        inference["g_first_weight"] = weight
        if selected_score_mode == STATE_V_FIRST_Q2_SCORE_MODE:
            configured["provenance"]["note"] = (
                "C3+First-Q2 planning combines candidate-normalized EMA State-V "
                "at the terminal latent of a full frozen-F five-block rollout "
                "with the candidate-normalized retained online-G readout at real "
                "z0 and A1; target G and terminal latent L2 remain excluded."
            )
        else:
            configured["provenance"]["note"] = (
                "C3+First-Q planning combines raw EMA State-V at the terminal "
                "latent of a full frozen-F five-block rollout with the raw retained "
                "online-G readout at real z0 and A1; target G, candidate score "
                "normalization, and terminal latent L2 remain excluded."
            )
    else:
        inference.pop("parent_g", None)
        inference.pop("g_first_weight", None)
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


def actor_free_td_lewm_v1_c3_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> str:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    run_mode = "smoke" if smoke else "pilot" if pilot else "formal"
    selected_score_mode = score_mode or str(
        protocol["inference_objective"]["score_mode"]
    )
    if selected_score_mode not in STATE_V_SCORE_MODES:
        raise ValueError(f"Unsupported V1-C3 score mode {selected_score_mode!r}.")
    weight = _resolve_state_v_first_q_weight(
        protocol,
        score_mode=selected_score_mode,
        g_first_weight=g_first_weight,
    )
    suffix = ""
    if weight is not None:
        suffix = f"_alpha_{_state_v_first_q_weight_slug(weight)}"
    return (
        f"{protocol['method']}_cube_o50_{selected_score_mode}{suffix}_{run_mode}"
    )


def validate_actor_free_td_lewm_v1_c3_checkpoint_protocol(
    *,
    payload: Mapping[str, Any],
    predictor_config: Mapping[str, Any],
    critic_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    require_formal_completion: bool,
    expected_checkpoint_epoch: int | None = None,
) -> None:
    validate_actor_free_td_lewm_v1_c3_payload(payload)
    if payload["source_v1_c_provenance"] != protocol["source_v1_c"]:
        raise ValueError("Checkpoint source_v1_c provenance differs from protocol.")
    if (
        payload["pretrained_world_model_provenance"].get("source_checkpoint_sha256")
        != protocol["pretrained_world_model"]["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint frozen LeWM provenance differs from protocol.")

    parent_predictor = protocol.get("predictor", {})
    for key, expected in parent_predictor.items():
        actual = predictor_config.get(key)
        if key in {"gamma", "target_ema_decay", "loss_warmup_fraction"}:
            matches = actual is not None and np.isclose(float(actual), float(expected))
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Parent predictor checkpoint {key} differs from protocol."
            )
    for key in ("task_sampling", "joint_objective"):
        if predictor_config.get(key) != protocol.get(key):
            raise ValueError(
                f"Parent predictor checkpoint {key} differs from protocol."
            )
    if predictor_config.get("pretrained_world_model") != protocol.get(
        "pretrained_world_model"
    ):
        raise ValueError(
            "Parent predictor pretrained_world_model differs from protocol."
        )

    protocol_critic = protocol["state_critic"]
    for key, expected in protocol_critic.items():
        if critic_config.get(key) != expected:
            raise ValueError(f"State critic checkpoint {key} differs from protocol.")
    for key in ("goal_sampling", "objective"):
        if critic_config.get(key) != protocol.get(key):
            raise ValueError(f"State critic checkpoint {key} differs from protocol.")
    checkpoint = protocol["checkpoint"]
    if require_formal_completion:
        expected = {
            "epoch": checkpoint["formal_epoch"],
            "logical_epoch": checkpoint["formal_logical_epoch"],
            "global_step": checkpoint["formal_global_step"],
        }
        _require_exact_values(payload, expected, label="checkpoint")
    if expected_checkpoint_epoch is not None:
        if payload.get("epoch") != expected_checkpoint_epoch:
            raise ValueError("Checkpoint epoch differs from --checkpoint-epoch.")
        if payload.get("logical_epoch") != expected_checkpoint_epoch:
            raise ValueError("Checkpoint logical_epoch differs from requested epoch.")


def _validate_action_compatibility(
    *, primitive_action_dim: int, action_block: int
) -> None:
    if int(primitive_action_dim) * int(action_block) != V1_RAW_ACTION_DIM:
        raise ValueError("V1-C3 requires five primitive 5D actions per block.")


def evaluate_actor_free_td_lewm_v1_c3(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
    pilot: bool = False,
    checkpoint_epoch: int | None = None,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> dict[str, Any]:
    """Run the audited public Stable World Model CEM/O50 evaluation."""

    formal_protocol = load_actor_free_td_lewm_v1_c3_evaluation_protocol(protocol_path)
    protocol = configure_actor_free_td_lewm_v1_c3_evaluation_mode(
        formal_protocol,
        smoke=smoke,
        pilot=pilot,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    if checkpoint_epoch is not None and (smoke or pilot):
        raise ValueError("--checkpoint-epoch is only valid for full O50 evaluation.")
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
    restored = load_actor_free_td_lewm_v1_c3_checkpoint(
        checkpoint_file, map_location=device
    )
    require_formal_completion = not (smoke or pilot) and checkpoint_epoch is None
    validate_actor_free_td_lewm_v1_c3_checkpoint_protocol(
        payload=restored.payload,
        predictor_config=restored.predictor_config,
        critic_config=restored.critic_config,
        protocol=formal_protocol,
        require_formal_completion=require_formal_completion,
        expected_checkpoint_epoch=checkpoint_epoch,
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
    _validate_action_compatibility(
        primitive_action_dim=int(dataset.get_dim("action")),
        action_block=int(protocol["planning"]["action_block"]),
    )

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
    selection_path = output_dir / "episode_selection.json"
    _write_json(selection_path, selection)
    selection_sha256 = _sha256(selection_path)
    if not (smoke or pilot) and selection_sha256 != FORMAL_SELECTION_SHA256:
        raise ValueError(
            "Generated episode selection differs from the locked seed-42 O50 set."
        )
    action_processor, action_stats = _load_action_processor(
        dataset, output_dir / "action_normalization.json"
    )

    selected_score_mode = protocol["inference_objective"]["score_mode"]
    world_model = restored.world_model.to(device).eval().requires_grad_(False)
    target_critic = restored.target_critic.to(device).eval().requires_grad_(False)
    predictor = None
    if selected_score_mode in STATE_V_FIRST_ACTION_SCORE_MODES:
        predictor = restored.predictor.to(device).eval().requires_grad_(False)
    image = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image["mean"], std=image["std"]),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    policy = make_actor_free_td_lewm_v1_c3_policy(
        world_model=world_model,
        target_critic=target_critic,
        predictor=predictor,
        planning=planning,
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device=device,
        reduced_evaluation=bool(smoke or pilot),
        score_mode=protocol["inference_objective"]["score_mode"],
        g_first_weight=protocol["inference_objective"].get("g_first_weight"),
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
    checkpoint_manifest = {
        "path": str(checkpoint_file),
        "sha256": _sha256(checkpoint_file),
        "method": restored.payload["method"],
        "method_family": restored.payload["method_family"],
        "variant": restored.payload["variant"],
        "implementation_version": restored.payload["implementation_version"],
        "objective_version": restored.payload["objective_version"],
        "epoch": restored.payload["epoch"],
        "logical_epoch": restored.payload["logical_epoch"],
        "global_step": restored.payload["global_step"],
        "formal_completion_required": require_formal_completion,
        "predictor_config": restored.predictor_config,
        "critic_config": restored.critic_config,
        "pretrained_world_model_provenance": deepcopy(
            restored.payload["pretrained_world_model_provenance"]
        ),
        "source_v1_c_provenance": deepcopy(restored.payload["source_v1_c_provenance"]),
    }
    if checkpoint_epoch is not None:
        checkpoint_manifest["requested_checkpoint_epoch"] = checkpoint_epoch
        checkpoint_manifest["checkpoint_role"] = "intermediate_epoch_o50"
    selected_score_definition = protocol["inference_objective"]["score_definition"]
    manifest = {
        "score_mode": selected_score_mode,
        "score_definition": deepcopy(selected_score_definition),
        "protocol": protocol,
        "formal_protocol": formal_protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "dataset": {
            **dataset_source,
            "episodes": actual_episodes,
            "transitions": actual_transitions,
        },
        "checkpoint": checkpoint_manifest,
        "selection": selection,
        "selection_sha256": selection_sha256,
        "normalization": {"action": action_stats},
        "runtime": runtime,
    }
    if selected_score_mode in STATE_V_FIRST_ACTION_SCORE_MODES:
        manifest["g_first_weight"] = protocol["inference_objective"][
            "g_first_weight"
        ]
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
        "world_model_parameter_count": sum(
            parameter.numel() for parameter in world_model.parameters()
        ),
        "critic_parameter_count": sum(
            parameter.numel() for parameter in restored.critic.parameters()
        ),
        "target_critic_parameter_count": sum(
            parameter.numel() for parameter in target_critic.parameters()
        ),
        "parent_predictor_parameter_count": sum(
            parameter.numel() for parameter in restored.predictor.parameters()
        ),
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "score_mode": selected_score_mode,
        "score_definition": deepcopy(selected_score_definition),
        "planning_horizon": planning["horizon"],
        "selection_sha256": selection_sha256,
        "smoke": smoke,
        "pilot": pilot,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    if selected_score_mode in STATE_V_FIRST_ACTION_SCORE_MODES:
        result["g_first_weight"] = protocol["inference_objective"][
            "g_first_weight"
        ]
        result["raw_state_v_constant_shift_sanity"] = (
            "checked_on_first_CEM_cost_call"
        )
    else:
        result["constant_shift_sanity"] = "checked_on_first_CEM_cost_call"
    if checkpoint_epoch is not None:
        result["checkpoint_epoch"] = restored.payload["epoch"]
        result["checkpoint_role"] = "intermediate_epoch_o50"
        result["formal_completion_required"] = False
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "FORMAL_O50_PLANNING",
    "FORMAL_SELECTION_SHA256",
    "STATE_V_FIRST_Q_SCORE_DEFINITION",
    "STATE_V_FIRST_Q2_SCORE_DEFINITION",
    "STATE_V_SCORE_DEFINITION",
    "actor_free_td_lewm_v1_c3_output_directory_name",
    "configure_actor_free_td_lewm_v1_c3_evaluation_mode",
    "evaluate_actor_free_td_lewm_v1_c3",
    "load_actor_free_td_lewm_v1_c3_evaluation_protocol",
    "validate_actor_free_td_lewm_v1_c3_checkpoint_protocol",
    "validate_actor_free_td_lewm_v1_c3_evaluation_protocol",
]
