"""Explicit, isolated evaluation identity for the two G-weighted CEM tests."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from tdwm.adapters.g_weighted_cem import GWeightedCEMConfig


def configure_g_weighted_cem(
    protocol: Mapping[str, Any], config: GWeightedCEMConfig
) -> dict[str, Any]:
    """Keep the checkpoint, pairs, budget and feedback interval unchanged."""
    if protocol.get("inference_objective", {}).get("score_mode") != "f_only":
        raise ValueError("G-weighted evaluation must start from the F-only protocol.")
    planning = protocol["planning"]
    if planning["horizon"] != 5 or planning["action_block"] != 5:
        raise ValueError("G-weighted evaluation requires five five-action blocks.")
    if planning["elites"] < 2:
        raise ValueError("G weighting requires at least two elites.")
    configured = deepcopy(dict(protocol))
    state_only = configured.get("variant") == "c4"
    source = "f_post_action_state" if state_only else "f_pre_action_state_and_action"
    definition = {
        "name": config.evaluation_mode,
        "formula": "squared_l2(F_full_rollout_terminal_state, encoded_goal)",
        "elite_selection": "lowest_full_F_terminal_cost",
        "online_g_used": True,
        "g_role": "elite_distribution_update_only",
        "g_state_source": source,
        "action_enters_g": not state_only,
        "g_projection": "G_dot_sphere_projected_goal",
        "g_population": "selected_elites_only",
        "g_calls_per_iteration": int(planning["elites"] * planning["horizon"]),
        "path_reducer": "mean_of_five_Q" if config.mode == "path" else "none",
        "weight_formula": (
            "softmax_over_elites(mean_over_time(Q)/temperature)"
            if config.mode == "path"
            else "softmax_over_elites(Q_at_each_position/temperature)"
        ),
        "score_normalization": "none",
        "temperature": config.temperature,
        "mean_update": "sum_over_elites(w_at_position * action_at_position)",
        "variance_denominator": "1-sum_over_elites(w_squared)",
        "variance_degenerate_epsilon": "torch_finfo_weight_dtype_eps",
        "uniform_weight_behavior": "exact_upstream_mean_and_sample_std",
        "solver": "installed_stable_worldmodel_CEMSolver_with_public_callback",
        "training_changed": False,
        "actor": "none",
    }
    configured["g_weighted_cem"] = asdict(config)
    configured["inference_objective"] = {
        "score_mode": config.evaluation_mode,
        "score_definition": definition,
        "elite_selection_score_mode": "f_only",
        "source_f_only_objective": configured["inference_objective"],
    }
    return configured


def g_weighted_cem_metadata(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if "g_weighted_cem" not in protocol:
        return {}
    config = GWeightedCEMConfig(**protocol["g_weighted_cem"])
    inference = protocol["inference_objective"]
    if inference["score_mode"] != config.evaluation_mode:
        raise ValueError("G-weighted result identity contradicts the configured mode.")
    return {
        "score_mode": config.evaluation_mode,
        "g_weighted_cem": asdict(config),
        "elite_selection_score_mode": "f_only",
        "score_definition": deepcopy(inference["score_definition"]),
    }
