"""Controlled Cube evaluation for reward-free Successor-LeWM."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters import (
    load_rf_successor_checkpoint,
    make_rf_successor_policy,
    prepare_cloud_runtime,
)
from tdwm.evaluation.lewm_checkpoint import (
    REQUIRED_PLANNING_KEYS,
    _git_revision,
    _jsonable,
    _resolve_dataset_source,
    _resolve_local_export_checkpoint,
    _sha256,
    _write_json,
    sample_start_goal_pairs,
)
from tdwm.evaluation.mc_gt_lewm import _load_action_processor

METHOD = "rf_successor_lewm"
S_ONLY_METHOD = "rf_successor_sequence_wm"
BALANCED_SEQUENCE_METHOD = "rf_balanced_successor_sequence_wm"
EMA_BALANCED_SEQUENCE_METHOD = "rf_ema_balanced_successor_sequence_wm"
DIRECT_MOMENT_METHOD = "rf_direct_moment_sequence_wm"
E2E_MOMENT_METHOD = "rf_e2e_moment_sequence_wm"
MANIFOLD_PREFIX_METHOD = "rf_manifold_prefix_successor_wm"
EMA_MANIFOLD_PREFIX_METHOD = "rf_ema_manifold_prefix_successor_wm"
FROZEN_MANIFOLD_PREFIX_METHOD = "rf_frozen_manifold_prefix_successor_wm"
FROZEN_RESIDUAL_PREFIX_METHOD = "rf_frozen_residual_prefix_wm"
ANCHORED_E2E_MANIFOLD_PREFIX_METHOD = "rf_anchored_e2e_manifold_prefix_wm"
FROZEN_PRETRAINED_METHODS = frozenset(
    (FROZEN_MANIFOLD_PREFIX_METHOD, FROZEN_RESIDUAL_PREFIX_METHOD)
)
PRETRAINED_METHODS = frozenset(
    (*FROZEN_PRETRAINED_METHODS, ANCHORED_E2E_MANIFOLD_PREFIX_METHOD)
)
LEWM_BLEND_METHODS = frozenset(
    (FROZEN_MANIFOLD_PREFIX_METHOD, ANCHORED_E2E_MANIFOLD_PREFIX_METHOD)
)
MANIFOLD_PREFIX_METHODS = frozenset(
    (
        MANIFOLD_PREFIX_METHOD,
        EMA_MANIFOLD_PREFIX_METHOD,
        FROZEN_MANIFOLD_PREFIX_METHOD,
        FROZEN_RESIDUAL_PREFIX_METHOD,
        ANCHORED_E2E_MANIFOLD_PREFIX_METHOD,
    )
)
SEQUENCE_METHODS = frozenset(
    (
        S_ONLY_METHOD,
        BALANCED_SEQUENCE_METHOD,
        EMA_BALANCED_SEQUENCE_METHOD,
        DIRECT_MOMENT_METHOD,
        E2E_MOMENT_METHOD,
        MANIFOLD_PREFIX_METHOD,
        EMA_MANIFOLD_PREFIX_METHOD,
        FROZEN_MANIFOLD_PREFIX_METHOD,
        FROZEN_RESIDUAL_PREFIX_METHOD,
        ANCHORED_E2E_MANIFOLD_PREFIX_METHOD,
    )
)
SUPPORTED_METHODS = frozenset((METHOD, *SEQUENCE_METHODS))


def load_rf_successor_evaluation_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_rf_successor_evaluation_protocol(protocol)
    return protocol


def validate_rf_successor_evaluation_protocol(protocol: dict[str, Any]) -> None:
    method = protocol.get("method")
    if protocol.get("schema_version") != 1 or method not in SUPPORTED_METHODS:
        raise ValueError("Reward-free successor evaluation requires schema 1.")
    if protocol.get("environment") != "cube" or protocol.get("stage") != "planner_evaluation":
        raise ValueError("RF-Successor-LeWM evaluation is locked to Cube planning.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Evaluation requires stable-worldmodel 0.1.1.")

    successor = protocol.get("successor", {})
    expected = {
        "objective_version": {
            METHOD: 12,
            S_ONLY_METHOD: 2,
            BALANCED_SEQUENCE_METHOD: 3,
            EMA_BALANCED_SEQUENCE_METHOD: 4,
            DIRECT_MOMENT_METHOD: 5,
            E2E_MOMENT_METHOD: 6,
            MANIFOLD_PREFIX_METHOD: 7,
            EMA_MANIFOLD_PREFIX_METHOD: 8,
            FROZEN_MANIFOLD_PREFIX_METHOD: 9,
            FROZEN_RESIDUAL_PREFIX_METHOD: 10,
            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD: 11,
        }[method],
        "architecture": {
            METHOD: "masked_history_causal_gru_action_prefix",
            S_ONLY_METHOD: "causal_gru_successor_increments",
            BALANCED_SEQUENCE_METHOD: "causal_gru_successor_increments",
            EMA_BALANCED_SEQUENCE_METHOD: "causal_gru_successor_increments",
            DIRECT_MOMENT_METHOD: "causal_gru_successor_increments",
            E2E_MOMENT_METHOD: "causal_gru_successor_increments",
            MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
            EMA_MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
            FROZEN_MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
            FROZEN_RESIDUAL_PREFIX_METHOD: "causal_transformer_lewm_residual",
            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD: "causal_transformer_manifold_successor",
        }[method],
        "feature_basis": "augmented_latent_squared_distance",
        "horizon_normalization": "discounted_prefix_mean",
        "target": {
            METHOD: "direct_monte_carlo",
            S_ONLY_METHOD: "online_direct_monte_carlo",
            BALANCED_SEQUENCE_METHOD: "online_direct_monte_carlo",
            EMA_BALANCED_SEQUENCE_METHOD: "ema_direct_monte_carlo",
            DIRECT_MOMENT_METHOD: "online_stop_gradient_direct_moments",
            E2E_MOMENT_METHOD: "online_end_to_end_direct_moments",
            MANIFOLD_PREFIX_METHOD: "online_end_to_end_latents",
            EMA_MANIFOLD_PREFIX_METHOD: "ema_stop_gradient_latents",
            FROZEN_MANIFOLD_PREFIX_METHOD: "frozen_pretrained_latents",
            FROZEN_RESIDUAL_PREFIX_METHOD: "frozen_pretrained_residual_latents",
            ANCHORED_E2E_MANIFOLD_PREFIX_METHOD: "frozen_teacher_latents",
        }[method],
        "action_conditioning": "causal_prefix",
        "goal_conditioning": "none",
        "continuation_policy": "none",
        "td_bootstrap": False,
    }
    if method == METHOD:
        expected.update(
            {
                "history_padding": "left_zero",
                "history_masking": "explicit_validity",
                "history_supervision": "all_prefix_lengths",
            }
        )
    if method in SEQUENCE_METHODS:
        if method == FROZEN_RESIDUAL_PREFIX_METHOD:
            expected["latent_recovery"] = "base_plus_residual_manifold_latents"
        else:
            expected["latent_recovery"] = (
                "direct_manifold_latents"
                if method in MANIFOLD_PREFIX_METHODS
                else "exact_adjacent_successor_difference"
            )
    if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        expected["goal_encoder"] = "frozen_pretrained_teacher"
    if method in {
        BALANCED_SEQUENCE_METHOD,
        EMA_BALANCED_SEQUENCE_METHOD,
        DIRECT_MOMENT_METHOD,
        E2E_MOMENT_METHOD,
    }:
        expected["feature_group_reduction"] = "group_sum"
    for key, value in expected.items():
        if successor.get(key) != value:
            raise ValueError(f"successor.{key} must be {value!r}.")
    if int(successor.get("history_size", 0)) <= 0:
        raise ValueError("successor.history_size must be positive.")
    if int(successor.get("max_horizon", 0)) <= 0:
        raise ValueError("successor.max_horizon must be positive.")
    if method in MANIFOLD_PREFIX_METHODS:
        architecture_dimensions = (
            "prefix_depth",
            "prefix_heads",
            "prefix_mlp_dim",
            "predictor_depth",
            "predictor_mlp_dim",
            "fusion_dim",
        )
        if min(int(successor.get(key, 0)) for key in architecture_dimensions) <= 0:
            raise ValueError("Manifold-prefix architecture dimensions must be positive.")
        if int(protocol["model"]["embed_dim"]) % int(successor["prefix_heads"]):
            raise ValueError("model.embed_dim must be divisible by prefix_heads.")
        if not 0.0 <= float(successor.get("dropout", -1.0)) < 1.0:
            raise ValueError("successor.dropout must lie in [0, 1).")
        source_hash = successor.get("pretrained_world_model_sha256")
        if method in PRETRAINED_METHODS:
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError("The pretrained LeWM source hash is invalid.")
        elif source_hash is not None:
            raise ValueError("Only pretrained methods accept a source hash.")
        if method == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD and float(
            successor.get("geometry_anchor_weight", 0.0)
        ) <= 0.0:
            raise ValueError("The anchored checkpoint requires a positive anchor weight.")
    if not 0.0 <= float(successor.get("gamma", -1.0)) <= 1.0:
        raise ValueError("successor.gamma must lie in [0, 1].")
    if min(
        float(successor.get("planning_weight", -1.0)),
        float(successor.get("terminal_weight", -1.0)),
    ) < 0.0:
        raise ValueError("Planning cost weights cannot be negative.")
    if method in SEQUENCE_METHODS and (
        float(successor.get("planning_weight", -1.0)) != 1.0
        or float(successor.get("terminal_weight", -1.0)) != 0.0
    ):
        raise ValueError("The S-only planner must use only the successor score.")
    planning_query = successor.get("planning_query", "discounted_successor")
    if planning_query not in {
        "discounted_successor",
        "discounted_terminal_blend",
        "manifold_projected_successor",
        "terminal_moment",
        "lewm_direct_terminal_blend",
        "lewm_direct_terminal_cost_mix",
        "lewm_residual_terminal",
    }:
        raise ValueError("Unsupported successor planning query.")
    if planning_query != "discounted_successor" and method not in SEQUENCE_METHODS:
        raise ValueError("The selected planning query requires a sequence checkpoint.")
    query_weight = successor.get("terminal_query_weight")
    if planning_query == "discounted_terminal_blend":
        if query_weight is None or not 0.0 < float(query_weight) < 1.0:
            raise ValueError(
                "The blended query requires terminal_query_weight in (0, 1)."
            )
    elif query_weight is not None:
        raise ValueError(
            "terminal_query_weight is only valid for the blended planning query."
        )
    blend_weights = successor.get("lewm_blend_weights")
    if planning_query in {
        "lewm_direct_terminal_blend",
        "lewm_direct_terminal_cost_mix",
    }:
        if method not in LEWM_BLEND_METHODS:
            raise ValueError("The LeWM mixture requires a compatible direct model.")
        if (
            not isinstance(blend_weights, list)
            or len(blend_weights) != int(successor["max_horizon"])
            or any(not 0.0 <= float(weight) <= 1.0 for weight in blend_weights)
        ):
            raise ValueError("The LeWM mixture requires one valid weight per horizon.")
    elif blend_weights is not None:
        raise ValueError("lewm_blend_weights requires the matching planning query.")
    if (method == FROZEN_RESIDUAL_PREFIX_METHOD) != (
        planning_query == "lewm_residual_terminal"
    ):
        raise ValueError(
            "The residual method and residual planning query must be used together."
        )

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed planning horizon.")
    if planning["horizon"] > successor["max_horizon"]:
        raise ValueError("Planning exceeds the trained successor horizon.")
    if planning["action_block"] != planning.get("frame_skip"):
        raise ValueError("Planning action blocks must match training frame skip.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The explicit plan exceeds the episode budget.")
    if planning.get("initial_distribution") != "cem_gaussian_no_actor":
        raise ValueError("RF-Successor-LeWM must not use a learned actor warm start.")
    evaluation = protocol.get("evaluation", {})
    if min(int(evaluation.get("episodes", 0)), int(evaluation.get("goal_offset", 0))) <= 0:
        raise ValueError("Evaluation episodes and goal offset must be positive.")


def _resolve_successor_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        files = sorted(requested.glob("*.pt"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(
        "A reward-free successor deployment checkpoint must be a .pt file or a "
        "directory containing exactly one .pt file."
    )


def _validate_successor_config(
    config: dict[str, Any], protocol: dict[str, Any]
) -> None:
    successor = protocol["successor"]
    expected = {
        "objective_version": successor["objective_version"],
        "architecture": successor["architecture"],
        "embed_dim": protocol["model"]["embed_dim"],
        "history_size": successor["history_size"],
        "max_horizon": successor["max_horizon"],
        "feature_basis": successor["feature_basis"],
        "horizon_normalization": successor["horizon_normalization"],
        "target": successor["target"],
        "action_conditioning": successor["action_conditioning"],
        "goal_conditioning": successor["goal_conditioning"],
        "continuation_policy": successor["continuation_policy"],
        "td_bootstrap": successor["td_bootstrap"],
    }
    architecture_keys = (
        (
            "prefix_depth",
            "prefix_heads",
            "prefix_mlp_dim",
            "predictor_depth",
            "predictor_mlp_dim",
            "fusion_dim",
            "dropout",
        )
        if protocol["method"] in MANIFOLD_PREFIX_METHODS
        else ("hidden_dim",)
    )
    for key in architecture_keys:
        expected[key] = successor[key]
    for key in ("latent_recovery", "feature_group_reduction"):
        if key in successor:
            expected[key] = successor[key]
    for key in (
        "history_padding",
        "history_masking",
        "history_supervision",
    ):
        if key in successor:
            expected[key] = successor[key]
    if protocol["method"] in PRETRAINED_METHODS:
        expected["pretrained_world_model_sha256"] = successor[
            "pretrained_world_model_sha256"
        ]
    if protocol["method"] == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        expected["goal_encoder"] = successor["goal_encoder"]
        expected["geometry_anchor_weight"] = successor[
            "geometry_anchor_weight"
        ]
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Successor checkpoint {key} differs from protocol.")
    for key in ("gamma", "planning_weight", "terminal_weight"):
        if not np.isclose(float(config[key]), float(successor[key])):
            raise ValueError(f"Successor checkpoint {key} differs from protocol.")
    if "target_world_ema_decay" in successor and not np.isclose(
        float(config.get("target_world_ema_decay", -1.0)),
        float(successor["target_world_ema_decay"]),
    ):
        raise ValueError("Successor checkpoint EMA decay differs from protocol.")


def _validate_checkpoint_pair(
    *,
    base_name: str,
    base_file: Path,
    successor_config: dict[str, Any],
) -> None:
    if successor_config.get("base_export_run_name") != base_name:
        raise ValueError("The successor and LeWM exports came from different epochs.")
    expected_hash = successor_config.get("base_checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("The successor checkpoint is missing its paired LeWM hash.")
    if _sha256(base_file) != expected_hash:
        raise ValueError("The successor checkpoint does not match the LeWM weights.")


def configure_rf_successor_evaluation_mode(
    protocol: dict[str, Any],
    *,
    smoke: bool,
    pilot: bool,
) -> dict[str, Any]:
    """Apply one fixed screening budget without weakening the formal protocol."""

    if smoke and pilot:
        raise ValueError("Smoke and pilot evaluation modes are mutually exclusive.")
    configured = deepcopy(protocol)
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
    validate_rf_successor_evaluation_protocol(configured)
    return configured


def evaluate_rf_successor_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    successor_checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
    pilot: bool = False,
) -> dict[str, Any]:
    protocol = configure_rf_successor_evaluation_mode(
        load_rf_successor_evaluation_protocol(protocol_path),
        smoke=smoke,
        pilot=pilot,
    )

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_source = _resolve_dataset_source(dataset_path, protocol["dataset"])
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
    base_name, base_file, base_cache = _resolve_local_export_checkpoint(
        base_checkpoint_path
    )
    successor_file = _resolve_successor_checkpoint(successor_checkpoint_path)
    head, head_config, payload = load_rf_successor_checkpoint(
        successor_file, map_location=device
    )
    if payload.get("method") != protocol["method"]:
        raise ValueError("The deployment checkpoint method differs from protocol.")
    if protocol["method"] in FROZEN_PRETRAINED_METHODS:
        initialization = payload.get("initialization", {})
        if (
            initialization.get("strategy") != "frozen_pretrained_lewm"
            or initialization.get("frozen") is not True
            or initialization.get("source_checkpoint_sha256")
            != protocol["successor"]["pretrained_world_model_sha256"]
        ):
            raise ValueError("The frozen LeWM initialization differs from protocol.")
    elif protocol["method"] == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        initialization = payload.get("initialization", {})
        if (
            initialization.get("strategy") != "anchored_pretrained_lewm"
            or initialization.get("student_frozen") is not False
            or initialization.get("teacher_frozen") is not True
            or initialization.get("source_checkpoint_sha256")
            != protocol["successor"]["pretrained_world_model_sha256"]
            or "target_world_model_state_dict" not in payload
        ):
            raise ValueError(
                "The anchored student/teacher initialization differs from protocol."
            )
    _validate_successor_config(head_config, protocol)
    _validate_checkpoint_pair(
        base_name=base_name,
        base_file=base_file,
        successor_config=head_config,
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
    if int(head_config["action_dim"]) != expected_action_dim:
        raise ValueError("The successor action-block dimension is incompatible.")

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
        dataset, output_dir / "action_normalization.json"
    )

    model = swm.wm.load_pretrained(base_name, cache_dir=str(base_cache)).to(device)
    model.load_state_dict(payload["world_model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    goal_world_model = None
    if protocol["method"] == ANCHORED_E2E_MANIFOLD_PREFIX_METHOD:
        goal_world_model = deepcopy(model)
        goal_world_model.load_state_dict(payload["target_world_model_state_dict"])
        goal_world_model.eval()
        goal_world_model.requires_grad_(False)
    head = head.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    successor_parameter_count = sum(
        parameter.numel() for parameter in head.parameters()
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
    policy = make_rf_successor_policy(
        world_model=model,
        goal_world_model=goal_world_model,
        successor=head,
        planning=planning,
        successor_config=protocol["successor"],
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device=device,
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
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "dataset": {
            **dataset_source,
            "episodes": actual_episodes,
            "transitions": actual_transitions,
        },
        "checkpoints": {
            "base_name": base_name,
            "base_path": str(base_file),
            "base_sha256": _sha256(base_file),
            "successor_path": str(successor_file),
            "successor_sha256": _sha256(successor_file),
            "successor_config": head_config,
        },
        "selection": selection,
        "normalization": {"action": action_stats},
        "runtime": runtime,
    }
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
        "world_model_parameter_count": parameter_count,
        "successor_parameter_count": successor_parameter_count,
        "method": protocol["method"],
        "smoke": smoke,
        "pilot": pilot,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "ANCHORED_E2E_MANIFOLD_PREFIX_METHOD",
    "BALANCED_SEQUENCE_METHOD",
    "DIRECT_MOMENT_METHOD",
    "E2E_MOMENT_METHOD",
    "EMA_BALANCED_SEQUENCE_METHOD",
    "EMA_MANIFOLD_PREFIX_METHOD",
    "FROZEN_MANIFOLD_PREFIX_METHOD",
    "FROZEN_PRETRAINED_METHODS",
    "FROZEN_RESIDUAL_PREFIX_METHOD",
    "LEWM_BLEND_METHODS",
    "MANIFOLD_PREFIX_METHOD",
    "MANIFOLD_PREFIX_METHODS",
    "PRETRAINED_METHODS",
    "SEQUENCE_METHODS",
    "S_ONLY_METHOD",
    "configure_rf_successor_evaluation_mode",
    "evaluate_rf_successor_lewm",
    "load_rf_successor_evaluation_protocol",
    "validate_rf_successor_evaluation_protocol",
]
