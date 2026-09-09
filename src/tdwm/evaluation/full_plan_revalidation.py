"""Opt-in 25-primitive-step execution, independent of the critic readout.

Historical protocols remain the checkpoint-validation source. This transform
is applied only after their original score-mode validation, and never changes
the model, candidate cost, task selection, search budget, or episode budget.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

EXECUTION_PROTOCOL = "full_plan_25_steps_real_feedback_v1"
FULL_PLAN_SCORE_MODES = frozenset(
    {
        "f_only",
        "f_plus_g",
        "f_plus_g_first",
        "f_plus_g_first_q2",
        "g_only_f_rollout_mean",
    }
)
FULL_PLAN_EXECUTION = {
    "executed_action_block": "all_five_blocks",
    "executed_action_blocks_before_replanning": 5,
    "executed_environment_steps_before_replanning": 25,
    "receding_horizon": 5,
    "replanning": "every_five_action_blocks",
    "cem_execution": "execute_A1_through_A5_from_minimum_total_cost_plan",
}


def configure_full_plan_revalidation(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a separate execution protocol without silently redefining G-only."""

    configured = deepcopy(dict(protocol))
    if "execution_revalidation" in configured:
        raise ValueError(
            "Full-plan revalidation must start from a historical protocol."
        )
    planning = configured["planning"]
    inference = configured["inference_objective"]
    mode = inference["score_mode"]
    if mode not in FULL_PLAN_SCORE_MODES:
        raise ValueError(
            "Full-plan revalidation requires a predeclared five-block score mode; "
            "legacy g_only plans one block and needs a separate protocol decision."
        )
    for key, expected in {"horizon": 5, "action_block": 5, "frame_skip": 5}.items():
        if type(planning.get(key)) is not int or planning[key] != expected:
            raise ValueError(
                f"Full-plan revalidation requires planning.{key}={expected}."
            )
    old_interval = planning.get("receding_horizon")
    if type(old_interval) is not int or old_interval not in {1, 5}:
        raise ValueError("Source receding_horizon must be one or five blocks.")
    if planning["episode_budget"] < 25:
        raise ValueError("Episode budget cannot be shorter than a complete plan.")
    source_hash = hashlib.sha256(
        json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    planning["receding_horizon"] = 5
    planning["executed_environment_steps_before_replanning"] = 25
    inference["replanning"] = FULL_PLAN_EXECUTION["replanning"]
    definition = inference.get("score_definition")
    if mode in {"f_plus_g_first", "f_plus_g_first_q2"}:
        definition["cem_execution"] = FULL_PLAN_EXECUTION["cem_execution"]
    elif mode == "g_only_f_rollout_mean":
        for key in ("executed_action_block", "replanning"):
            inference[key] = FULL_PLAN_EXECUTION[key]
            definition[key] = FULL_PLAN_EXECUTION[key]
    configured["execution_revalidation"] = {
        "id": EXECUTION_PROTOCOL,
        "source_configured_protocol_sha256": source_hash,
        "source_receding_horizon": old_interval,
        "source_executed_environment_steps_before_replanning": old_interval * 5,
        "feedback_source": "new_environment_pixels_encoded_when_action_buffer_is_empty",
        "predicted_state_used_as_replanning_observation": False,
        "score_formula_unchanged": True,
        "checkpoint_unchanged": True,
    }
    return configured


def full_plan_revalidation_metadata(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Keep manifest/result metadata consistent, including V2 postprocessing."""

    marker = protocol.get("execution_revalidation")
    if marker is None:
        return {}
    if not isinstance(marker, Mapping) or marker.get("id") != EXECUTION_PROTOCOL:
        raise ValueError("Unknown execution revalidation protocol.")
    planning = protocol["planning"]
    for key, expected in {
        "horizon": 5,
        "action_block": 5,
        "frame_skip": 5,
        "receding_horizon": 5,
        "executed_environment_steps_before_replanning": 25,
    }.items():
        if type(planning.get(key)) is not int or planning[key] != expected:
            raise ValueError(f"Execution manifest contradicts planning.{key}.")
    return {
        "execution_protocol": EXECUTION_PROTOCOL,
        "execution_revalidation": deepcopy(dict(marker)),
        **FULL_PLAN_EXECUTION,
    }


def require_new_revalidation_output(output_dir: str | Path) -> None:
    """Never overwrite a historical result or a partially completed rerun."""

    path = Path(output_dir).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Revalidation requires a new, empty output directory: {path}"
        )
