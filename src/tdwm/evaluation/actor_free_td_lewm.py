"""Controlled Cube evaluation for the Actor-Free TD-LeWM ablations."""

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

from tdwm.adapters import prepare_cloud_runtime
from tdwm.adapters.actor_free_td_lewm import (
    load_actor_free_td_checkpoint,
    make_actor_free_td_policy,
)
from tdwm.evaluation.lewm_checkpoint import (
    REQUIRED_PLANNING_KEYS,
    _git_revision,
    _jsonable,
    _resolve_dataset_source,
    _sha256,
    _write_json,
    sample_start_goal_pairs,
)
from tdwm.evaluation.mc_gt_lewm import _load_action_processor

METHOD = "actor_free_td_lewm"
OBJECTIVE_VERSION = 1
SUPPORTED_VARIANTS = frozenset(
    {"serial_decoupled", "serial_coupled", "hybrid"}
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
CHECKPOINT_SEMANTICS = {
    "architecture": "actor_free_successor_head",
    "goal_conditioning": "none",
    "action_conditioning": "dataset_current_action",
    "bootstrap_action": "dataset_next_action",
    "terminal_source": "next_action_nan_invalid",
    "actor": "none",
    "reward": "none",
}


def load_actor_free_td_evaluation_protocol(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_actor_free_td_evaluation_protocol(protocol)
    return protocol


def validate_actor_free_td_evaluation_protocol(
    protocol: dict[str, Any],
) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("Actor-Free TD-LeWM evaluation requires schema 1.")
    if protocol.get("method") != METHOD:
        raise ValueError("This evaluator only accepts Actor-Free TD-LeWM.")
    variant = protocol.get("variant")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported Actor-Free TD-LeWM variant {variant!r}."
        )
    if (
        protocol.get("environment") != "cube"
        or protocol.get("stage") != "planner_evaluation"
    ):
        raise ValueError("Actor-Free TD-LeWM evaluation is locked to Cube O50.")
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Evaluation requires stable-worldmodel 0.1.1.")

    successor = protocol.get("successor", {})
    if int(successor.get("objective_version", -1)) != OBJECTIVE_VERSION:
        raise ValueError("Actor-Free TD-LeWM requires objective_version 1.")
    for key, expected in CHECKPOINT_SEMANTICS.items():
        if successor.get(key) != expected:
            raise ValueError(f"successor.{key} must be {expected!r}.")
    if min(
        int(successor.get("history_size", 0)),
        int(successor.get("hidden_dim", 0)),
    ) <= 0:
        raise ValueError("Successor history and hidden dimensions must be positive.")
    if successor.get("feature_basis") != "augmented_latent_squared_distance":
        raise ValueError("The successor feature basis differs from planning.")
    if not 0.0 <= float(successor.get("gamma", -1.0)) < 1.0:
        raise ValueError("successor.gamma must lie in [0, 1).")
    if successor.get("td_bootstrap") is not True:
        raise ValueError("Actor-Free TD-LeWM must use TD bootstrapping.")
    if successor.get("actor") != "none":
        raise ValueError("Actor-Free TD-LeWM must not contain a learned actor.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning keys: {sorted(missing)}")
    for key, expected in FORMAL_O50_PLANNING.items():
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

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes") != 50 or evaluation.get("goal_offset") != 50:
        raise ValueError("The formal Actor-Free TD-LeWM protocol is Cube O50/50 episodes.")
    objective = protocol.get("inference_objective", {})
    if objective.get("goal_usage") != "planning_linear_readout_only":
        raise ValueError("The goal may only be used by the planning-time readout.")
    if objective.get("goal_enters_successor_head") is not False:
        raise ValueError("The goal-free successor head cannot accept the goal.")
    if objective.get("learned_actor") is not False:
        raise ValueError("Actor-Free TD-LeWM cannot use a learned actor.")


def _resolve_joint_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        files = sorted(requested.glob("*.pt"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(
        "An Actor-Free TD-LeWM deployment checkpoint must be a .pt file or a "
        "directory containing exactly one .pt file."
    )


def _validate_checkpoint(
    *,
    payload: dict[str, Any],
    successor_config: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    expected_variant = protocol["variant"]
    checks = {
        "method": METHOD,
        "variant": expected_variant,
        "objective_version": OBJECTIVE_VERSION,
        "deployment_checkpoint_version": 1,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key} differs from the evaluation protocol."
            )
    required_payload = {
        "world_model_state_dict",
        "successor_state_dict",
        "world_model_config",
        "successor_config",
    }
    missing = required_payload - payload.keys()
    if missing:
        raise ValueError(
            f"The joint deployment checkpoint is missing {sorted(missing)}."
        )

    successor = protocol["successor"]
    expected_config: dict[str, Any] = {
        "embed_dim": protocol["model"]["embed_dim"],
        "history_size": successor["history_size"],
        "hidden_dim": successor["hidden_dim"],
        "variant": expected_variant,
        "gamma": successor["gamma"],
        "feature_basis": successor["feature_basis"],
        **CHECKPOINT_SEMANTICS,
    }
    for key, expected in expected_config.items():
        actual = successor_config.get(key)
        if key == "gamma":
            matches = actual is not None and np.isclose(
                float(actual), float(expected)
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Successor checkpoint {key} differs from the protocol."
            )


def configure_actor_free_td_evaluation_mode(
    protocol: dict[str, Any],
    *,
    smoke: bool,
    pilot: bool,
) -> dict[str, Any]:
    """Apply bounded execution budgets after validating the formal protocol."""

    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
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
    return configured


def evaluate_actor_free_td_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    video: bool = False,
    smoke: bool = False,
    pilot: bool = False,
) -> dict[str, Any]:
    formal_protocol = load_actor_free_td_evaluation_protocol(protocol_path)
    protocol = configure_actor_free_td_evaluation_mode(
        formal_protocol, smoke=smoke, pilot=pilot
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
    checkpoint_file = _resolve_joint_checkpoint(checkpoint_path)
    world_model, successor, successor_config, payload = (
        load_actor_free_td_checkpoint(checkpoint_file, map_location=device)
    )
    _validate_checkpoint(
        payload=payload,
        successor_config=successor_config,
        protocol=formal_protocol,
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
    if int(successor_config["action_dim"]) != expected_action_dim:
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

    world_model = world_model.to(device).eval()
    world_model.requires_grad_(False)
    successor = successor.to(device).eval()
    successor.requires_grad_(False)
    world_parameter_count = sum(
        parameter.numel() for parameter in world_model.parameters()
    )
    successor_parameter_count = sum(
        parameter.numel() for parameter in successor.parameters()
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
    policy = make_actor_free_td_policy(
        world_model=world_model,
        successor=successor,
        planning=planning,
        gamma=float(protocol["successor"]["gamma"]),
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device=device,
        clamp_tail_cost=bool(
            protocol["successor"].get("clamp_successor_cost", True)
        ),
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
            "variant": payload["variant"],
            "objective_version": payload["objective_version"],
            "successor_config": successor_config,
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
        "world_model_parameter_count": world_parameter_count,
        "successor_parameter_count": successor_parameter_count,
        "method": METHOD,
        "variant": protocol["variant"],
        "smoke": smoke,
        "pilot": pilot,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "FORMAL_O50_PLANNING",
    "METHOD",
    "OBJECTIVE_VERSION",
    "SUPPORTED_VARIANTS",
    "configure_actor_free_td_evaluation_mode",
    "evaluate_actor_free_td_lewm",
    "load_actor_free_td_evaluation_protocol",
    "validate_actor_free_td_evaluation_protocol",
]
