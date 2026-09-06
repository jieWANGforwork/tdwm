"""Controlled Cube O25/O50/O100 evaluation for state-only V1-C4."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    C4_ONLY_SCORE_MODE,
    DEPLOYMENT_CHECKPOINT_VERSION,
    F_ONLY_SCORE_MODE,
    F_PLUS_C4_SCORE_MODE,
    FIRST_ACTION_SCORE_MODES,
    FIRST_Q2_SCORE_MODE,
    FORMAL_HORIZON_BY_SCORE_MODE,
    IMPLEMENTATION_VERSION,
    MEAN_Q_SCORE_MODE,
    METHOD,
    METHOD_FAMILY,
    OBJECTIVE_VERSION,
    SCORE_MODES,
    VARIANT,
    load_actor_free_td_lewm_v1_c4_checkpoint,
    make_actor_free_td_lewm_v1_c4_policy,
)
from tdwm.adapters.frozen_actor_free_td_common import (
    FORMAL_DEPLOYMENT_EPOCH,
    FORMAL_DEPLOYMENT_GLOBAL_STEP,
    is_lower_sha256,
)
from tdwm.adapters.frozen_actor_free_td_v1_common import (
    FIRST_Q2_STD_EPSILON,
    FrozenActorFreeTDV1MethodSpec,
)
from tdwm.evaluation.frozen_actor_free_td_common import _validate_dataset_protocol
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    FORMAL_EVALUATION_BY_PROTOCOL,
    FORMAL_O25_PLANNING,
    FORMAL_O50_PLANNING,
    FORMAL_O100_PLANNING,
    _execution_metadata,
    _load_protocol_mapping,
    evaluate_actor_free_td_predictor_runtime,
    v1_evaluation_protocol_label,
)
from tdwm.evaluation.lewm_checkpoint import REQUIRED_PLANNING_KEYS, _write_json
from tdwm.methods.actor_free_td_lewm_v1 import V1_RAW_ACTION_DIM, V1_STATE_DIM

FORMAL_SELECTION_SHA256_BY_PROTOCOL = {
    "o25": "56546fe8725ce0e4670f308c5b325bd64ff2a792373add8c20ddbcab02da6b37",
    "o50": "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7",
    "o100": "8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c",
}


def _validate_method_config(config: Mapping[str, Any]) -> None:
    objective = config.get("joint_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("g_config.joint_objective must be a mapping.")
    if float(objective.get("goal_projection_weight", -1.0)) != 1.0:
        raise ValueError("C4 goal_projection_weight must be exactly 1.0.")


METHOD_SPEC = FrozenActorFreeTDV1MethodSpec(
    method=METHOD,
    variant=VARIANT,
    display_name="Actor-Free TD-LeWM V1 C4",
    objective_keys=("goal_projection_weight",),
    validate_method_config=_validate_method_config,
)


def _first_q_weight(score_mode: str, value: float | None) -> float | None:
    if score_mode not in FIRST_ACTION_SCORE_MODES:
        if value is not None:
            raise ValueError("g_first_weight is only valid for First-Q modes.")
        return None
    if value is None or isinstance(value, bool):
        raise ValueError("First-Q modes require an explicit non-negative weight.")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("First-Q weight must be finite and non-negative.")
    return 0.0 if weight == 0.0 else weight


def _score_definition(score_mode: str) -> dict[str, Any]:
    common = {
        "optimization": "cem_minimize",
        "action_enters_g": False,
        "action_path": "candidate_action_to_frozen_f_to_predicted_state_to_c4",
        "task": "sqrt_dim_l2_normalized_goal_vector",
        "online_g_used": score_mode != F_ONLY_SCORE_MODE,
        "target_g_used": False,
    }
    if score_mode == F_ONLY_SCORE_MODE:
        return {
            **common,
            "formula": "terminal_summed_mse(zhat5_f,z_goal)",
            "f_rollout": "full_five_action_blocks_A1_through_A5",
            "g_state": "unused",
        }
    if score_mode == C4_ONLY_SCORE_MODE:
        return {
            **common,
            "formula": "-dot(G_C4(zhat1_f,m),m)",
            "f_rollout": "one_action_block_A1",
            "g_state": "frozen_f_predicted_state_after_A1_zhat1",
            "terminal_f_cost": "unused",
            "gamma": "unused",
        }
    if score_mode == F_PLUS_C4_SCORE_MODE:
        return {
            **common,
            "formula": (
                "terminal_summed_mse(zhat4_f,z_goal) "
                "- gamma_power_4 * dot(G_C4(zhat5_f,m),m)"
            ),
            "f_rollout": "full_five_action_blocks_A1_through_A5",
            "f_prefix_state": "frozen_f_predicted_state_after_A4_zhat4",
            "g_state": "frozen_f_predicted_state_after_A5_zhat5",
            "final_action_path": "A5_to_frozen_f_to_zhat5_to_state_only_g",
        }
    if score_mode in FIRST_ACTION_SCORE_MODES:
        definition = {
            **common,
            "formula": (
                "terminal_summed_mse(zhat5_f,z_goal) "
                "- g_first_weight * dot(G_C4(zhat1_f,m),m)"
            ),
            "f_rollout": "full_five_action_blocks_A1_through_A5",
            "q_first": "dot(G_C4(zhat1_f,m),m)",
            "q_first_state": "frozen_f_predicted_state_after_A1_zhat1",
            "q_first_action_path": "A1_to_frozen_f_to_zhat1_to_state_only_g",
            "q_first_discount": "none",
        }
        if score_mode == FIRST_Q2_SCORE_MODE:
            definition.update(
                {
                    "formula": (
                        "zscore_samples(terminal_summed_mse(zhat5_f,z_goal)) "
                        "- g_first_weight * zscore_samples(q_C4(zhat1_f,m))"
                    ),
                    "normalization": "population_z_score",
                    "normalization_axis": (
                        "cem_candidate_sample_axis_dim_1_per_environment"
                    ),
                    "normalization_scope": "independent_per_get_cost_call",
                    "normalization_epsilon": FIRST_Q2_STD_EPSILON,
                    "degenerate_signal": (
                        "zeros_when_population_std_lte_epsilon"
                    ),
                }
            )
        else:
            definition["normalization"] = "none_raw_scores"
        return definition
    if score_mode == MEAN_Q_SCORE_MODE:
        return {
            **common,
            "formula": "-(1/5)*sum_k_1_to_5(dot(G_C4(zhatk_f,m),m))",
            "f_rollout": "full_five_action_blocks_A1_through_A5",
            "g_state_sequence": "frozen_f_predicted_successor_states_zhat1_to_zhat5",
            "g_aggregation": "mean_over_5_blocks",
            "terminal_f_cost": "unused",
            "gamma": "unused",
        }
    raise ValueError(f"Unsupported C4 score mode {score_mode!r}.")


def _configured_inference(
    score_mode: str,
    *,
    planning: Mapping[str, Any],
    g_first_weight: float | None,
) -> dict[str, Any]:
    execution = _execution_metadata(planning)
    definition = _score_definition(score_mode)
    definition["executed_action_block"] = execution["executed_action_block"]
    definition["replanning"] = execution["replanning"]
    inference: dict[str, Any] = {
        "score_mode": score_mode,
        "score_definition": definition,
        "f_score": (
            "none" if score_mode in {C4_ONLY_SCORE_MODE, MEAN_Q_SCORE_MODE} else "lewm_rollout_goal_distance"
        ),
        "f_score_reducer": (
            "none"
            if score_mode in {C4_ONLY_SCORE_MODE, MEAN_Q_SCORE_MODE}
            else "final_predicted_latent_summed_mse"
        ),
        "g_score": (
            "unused"
            if score_mode == F_ONLY_SCORE_MODE
            else "negative_goal_projection_of_state_only_c4"
        ),
        "action_enters_g": False,
        "action_effect": "only_via_f_predicted_state",
        "goal_enters_g": True,
        "learned_actor": False,
        "training_only_auxiliary": ["dual_branch_goal_projected_td"],
        "training_only_auxiliary_used_at_evaluation": False,
        "replanning": execution["replanning"],
    }
    if score_mode in FIRST_ACTION_SCORE_MODES:
        inference["g_first_weight"] = g_first_weight
    if score_mode == MEAN_Q_SCORE_MODE:
        inference.update(
            {
                "f_transition_used": True,
                "f_goal_distance_used": False,
                "g_aggregation": "mean_over_5_blocks",
                "rollout_horizon": 5,
                "state_source_for_q1": "frozen_lewm_rollout_predicted_state_zhat1",
                "state_source_for_q2_to_q5": (
                    "frozen_lewm_rollout_predicted_states_zhat2_to_zhat5"
                ),
                "executed_action_block": execution["executed_action_block"],
            }
        )
    return inference


def _validate_g_protocol(g: Mapping[str, Any]) -> None:
    expected = {
        "architecture": "td_jepa_state_only_forward_map_v1_c4",
        "state_dim": V1_STATE_DIM,
        "task_dim": V1_STATE_DIM,
        "output_dim": V1_STATE_DIM,
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
    }
    for key, value in expected.items():
        if g.get(key) != value:
            raise ValueError(f"protocol.g.{key} must be {value!r}.")
    for forbidden in ("raw_action_dim", "action_dim", "action_embedding_dim"):
        if forbidden in g:
            raise ValueError(f"protocol.g must not contain {forbidden}.")
    for key in ("gamma", "target_ema_decay"):
        try:
            value = float(g[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"protocol.g.{key} must lie in [0, 1).") from error
        if not 0.0 <= value < 1.0:
            raise ValueError(f"protocol.g.{key} must lie in [0, 1).")


def _validate_planning(
    planning: Mapping[str, Any], *, score_mode: str, protocol_label: str
) -> None:
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"C4 planning is missing {sorted(missing)}.")
    expected = {
        "o25": FORMAL_O25_PLANNING,
        "o50": FORMAL_O50_PLANNING,
        "o100": FORMAL_O100_PLANNING,
    }[protocol_label]
    if protocol_label == "o25" and score_mode == C4_ONLY_SCORE_MODE:
        expected = {
            **expected,
            "receding_horizon": 1,
            "executed_environment_steps_before_replanning": 5,
        }
    for key, value in expected.items():
        if key == "horizon":
            continue
        if planning.get(key) != value:
            raise ValueError(f"C4 planning.{key} must be {value!r}.")
    expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[score_mode]
    if planning.get("horizon") != expected_horizon:
        raise ValueError(
            f"C4 {score_mode} requires planning.horizon={expected_horizon}."
        )


def validate_actor_free_td_lewm_v1_c4_evaluation_protocol(
    protocol: Mapping[str, Any],
) -> None:
    expected_identity = {
        "schema_version": 1,
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "environment": "cube",
        "stage": "planner_evaluation",
    }
    for key, value in expected_identity.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol.{key} must be {value!r}.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("C4 evaluation requires stable-worldmodel 0.1.1.")
    pretrained = protocol.get("pretrained_world_model")
    if not isinstance(pretrained, Mapping):
        raise ValueError("C4 protocol requires pretrained_world_model.")
    for key, value in {
        "source_method": "lewm",
        "source_seed": 3072,
        "source_epoch": 10,
        "frozen": True,
    }.items():
        if pretrained.get(key) != value:
            raise ValueError(f"pretrained_world_model.{key} must be {value!r}.")
    if not is_lower_sha256(pretrained.get("checkpoint_sha256")):
        raise ValueError("C4 pretrained checkpoint SHA-256 is invalid.")
    _validate_dataset_protocol(protocol)
    if protocol.get("model", {}).get("embed_dim") != V1_STATE_DIM:
        raise ValueError("C4 evaluation requires model.embed_dim=192.")
    context = protocol.get("context")
    if not isinstance(context, Mapping) or dict(context) != {
        "g_state_frames": 1,
        "lewm_rollout_history_frames": 3,
        "plan_config_history_len": 1,
    }:
        raise ValueError("C4 context must preserve the V1 LeWM planning context.")
    g = protocol.get("g")
    if not isinstance(g, Mapping):
        raise ValueError("C4 protocol requires a g mapping.")
    _validate_g_protocol(g)
    task_sampling = protocol.get("task_sampling")
    time_alignment = protocol.get("time_alignment")
    objective = protocol.get("joint_objective")
    if (
        not isinstance(task_sampling, Mapping)
        or not isinstance(time_alignment, Mapping)
        or not isinstance(objective, Mapping)
    ):
        raise ValueError(
            "C4 protocol requires task_sampling, time_alignment and joint_objective."
        )
    if float(objective.get("goal_projection_weight", -1.0)) != 1.0:
        raise ValueError("C4 goal_projection_weight must be 1.0.")

    inference = protocol.get("inference_objective")
    planning = protocol.get("planning")
    if not isinstance(inference, Mapping) or not isinstance(planning, Mapping):
        raise ValueError("C4 protocol requires inference_objective and planning.")
    score_mode = str(inference.get("score_mode", ""))
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported C4 score mode {score_mode!r}.")
    weight = _first_q_weight(
        score_mode,
        inference.get("g_first_weight") if score_mode in FIRST_ACTION_SCORE_MODES else None,
    )
    protocol_label = v1_evaluation_protocol_label(protocol)
    _validate_planning(planning, score_mode=score_mode, protocol_label=protocol_label)
    expected_inference = _configured_inference(
        score_mode, planning=planning, g_first_weight=weight
    )
    if dict(inference) != expected_inference:
        raise ValueError("C4 inference_objective differs from its locked score path.")
    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("C4 evaluation mapping is missing.")
    expected_evaluation = {
        **FORMAL_EVALUATION_BY_PROTOCOL[protocol_label],
        "start_goal_source": "same_dataset_episode",
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise ValueError(f"C4 evaluation.{key} must be {value!r}.")
    if planning.get("history_len") != context["plan_config_history_len"]:
        raise ValueError("C4 planning.history_len must be 1.")


def load_actor_free_td_lewm_v1_c4_evaluation_protocol(
    path: str | Path,
    *,
    spec: FrozenActorFreeTDV1MethodSpec = METHOD_SPEC,
) -> dict[str, Any]:
    if spec is not METHOD_SPEC:
        raise ValueError("C4 protocol loading requires the C4 method spec.")
    protocol = _load_protocol_mapping(Path(path), seen=frozenset())
    validate_actor_free_td_lewm_v1_c4_evaluation_protocol(protocol)
    return protocol


def configure_actor_free_td_lewm_v1_c4_evaluation_mode(
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
    protocol_label = v1_evaluation_protocol_label(configured)
    selected = score_mode or str(
        configured.get("inference_objective", {}).get(
            "score_mode", F_PLUS_C4_SCORE_MODE
        )
    )
    if selected not in SCORE_MODES:
        raise ValueError(f"Unsupported C4 score mode {selected!r}.")
    weight = _first_q_weight(selected, g_first_weight)
    planning = configured.setdefault("planning", {})
    planning["horizon"] = FORMAL_HORIZON_BY_SCORE_MODE[selected]
    planning["receding_horizon"] = (
        1 if protocol_label in {"o50", "o100"} or selected == C4_ONLY_SCORE_MODE else 5
    )
    if protocol_label == "o25":
        planning["executed_environment_steps_before_replanning"] = (
            planning["receding_horizon"] * 5
        )
    configured["inference_objective"] = _configured_inference(
        selected,
        planning=planning,
        g_first_weight=weight,
    )
    if smoke:
        configured["id"] = f"{configured['id']}_smoke"
        configured["evaluation"]["episodes"] = 1
        planning.update(
            {"candidates": 8, "iterations": 1, "elites": 2, "episode_budget": 25}
        )
    elif pilot:
        configured["id"] = f"{configured['id']}_pilot"
        configured["evaluation"]["episodes"] = 10
        planning.update(
            {
                "candidates": 128,
                "iterations": 10,
                "elites": 16,
                "episode_budget": 100,
            }
        )
    return configured


def actor_free_td_lewm_v1_c4_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
    g_first_weight: float | None = None,
) -> str:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    selected = score_mode or str(
        protocol.get("inference_objective", {}).get(
            "score_mode", F_PLUS_C4_SCORE_MODE
        )
    )
    if selected not in SCORE_MODES:
        raise ValueError(f"Unsupported C4 score mode {selected!r}.")
    weight = _first_q_weight(selected, g_first_weight)
    protocol_label = v1_evaluation_protocol_label(protocol)
    run_kind = "smoke" if smoke else "pilot" if pilot else "formal"
    if weight is None:
        return f"{METHOD}_cube_{protocol_label}_{selected}_{run_kind}"
    slug = format(weight, ".15g").replace(".", "p")
    return f"{METHOD}_cube_{protocol_label}_{selected}_alpha_{slug}_{run_kind}"


def validate_actor_free_td_lewm_v1_c4_checkpoint_protocol(
    *,
    payload: Mapping[str, Any],
    predictor_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    spec: FrozenActorFreeTDV1MethodSpec,
    require_formal_completion: bool = True,
) -> None:
    if spec is not METHOD_SPEC:
        raise ValueError("C4 checkpoint validation requires the C4 method spec.")
    for key, value in {
        "method": METHOD,
        "method_family": METHOD_FAMILY,
        "variant": VARIANT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
    }.items():
        if payload.get(key) != value or predictor_config.get(key) != value:
            raise ValueError(f"C4 checkpoint identity {key} is invalid.")
    for key, value in protocol["g"].items():
        actual = predictor_config.get(key)
        matches = (
            math.isclose(float(actual), float(value))
            if key in {"gamma", "target_ema_decay"} and actual is not None
            else actual == value
        )
        if not matches:
            raise ValueError(f"C4 checkpoint g_config.{key} differs from protocol.")
    for key in ("task_sampling", "time_alignment", "joint_objective"):
        if predictor_config.get(key) != protocol[key]:
            raise ValueError(f"C4 checkpoint {key} differs from protocol.")
    pretrained = protocol["pretrained_world_model"]
    checkpoint_pretrained = predictor_config.get("pretrained_world_model")
    provenance = payload.get("pretrained_world_model_provenance")
    if not isinstance(checkpoint_pretrained, Mapping) or not isinstance(
        provenance, Mapping
    ):
        raise ValueError("C4 checkpoint is missing pretrained LeWM provenance.")
    expected_sha = pretrained["checkpoint_sha256"]
    if (
        checkpoint_pretrained.get("checkpoint_sha256") != expected_sha
        or provenance.get("source_checkpoint_sha256") != expected_sha
    ):
        raise ValueError("C4 checkpoint uses a different pretrained LeWM.")
    if require_formal_completion:
        if (
            payload.get("epoch") != FORMAL_DEPLOYMENT_EPOCH
            or payload.get("global_step") != FORMAL_DEPLOYMENT_GLOBAL_STEP
        ):
            raise ValueError("C4 formal evaluation requires final E10 checkpoint.")


def validate_actor_free_td_lewm_v1_c4_raw_action_compatibility(
    *,
    primitive_action_dim: int,
    action_block: int,
    predictor_config: Mapping[str, Any],
) -> None:
    if int(primitive_action_dim) * int(action_block) != V1_RAW_ACTION_DIM:
        raise ValueError("C4 frozen F requires normalized 25D action blocks.")
    if predictor_config.get("action_input") != "none":
        raise ValueError("C4 G action_input must be none.")
    for forbidden in ("raw_action_dim", "action_dim", "action_embedding_dim"):
        if forbidden in predictor_config:
            raise ValueError(f"C4 G must not declare {forbidden}.")


def evaluate_actor_free_td_lewm_v1_c4(**kwargs) -> dict[str, Any]:
    """Run C4 and retain C4-native g_config naming in the audit manifest."""

    result = evaluate_actor_free_td_predictor_runtime(
        spec=METHOD_SPEC,
        checkpoint_loader=load_actor_free_td_lewm_v1_c4_checkpoint,
        policy_factory=make_actor_free_td_lewm_v1_c4_policy,
        protocol_loader=load_actor_free_td_lewm_v1_c4_evaluation_protocol,
        protocol_configurer=configure_actor_free_td_lewm_v1_c4_evaluation_mode,
        checkpoint_validator=validate_actor_free_td_lewm_v1_c4_checkpoint_protocol,
        raw_action_validator=(
            validate_actor_free_td_lewm_v1_c4_raw_action_compatibility
        ),
        checkpoint_provenance_keys=("pretrained_world_model_provenance",),
        **kwargs,
    )
    result_path = Path(kwargs["output_dir"]).expanduser().resolve() / "results.json"
    manifest_path = (
        Path(kwargs["output_dir"]).expanduser().resolve() / "protocol_manifest.json"
    )
    with result_path.open(encoding="utf-8") as stream:
        stored_result = json.load(stream)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    checkpoint = manifest["checkpoint"]
    checkpoint["g_config"] = checkpoint.pop("predictor_config")
    for values in (stored_result, manifest, result):
        values["state_only_g"] = True
        values["action_enters_g"] = False
        values["action_effect"] = "only_via_f_predicted_state"
    _write_json(result_path, stored_result)
    _write_json(manifest_path, manifest)
    return result


__all__ = [
    "FORMAL_O25_PLANNING",
    "FORMAL_O50_PLANNING",
    "FORMAL_O100_PLANNING",
    "FORMAL_SELECTION_SHA256_BY_PROTOCOL",
    "METHOD_SPEC",
    "actor_free_td_lewm_v1_c4_output_directory_name",
    "configure_actor_free_td_lewm_v1_c4_evaluation_mode",
    "evaluate_actor_free_td_lewm_v1_c4",
    "load_actor_free_td_lewm_v1_c4_evaluation_protocol",
    "validate_actor_free_td_lewm_v1_c4_checkpoint_protocol",
    "validate_actor_free_td_lewm_v1_c4_evaluation_protocol",
    "validate_actor_free_td_lewm_v1_c4_raw_action_compatibility",
]
