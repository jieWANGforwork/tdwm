"""Shared O50 runtime for independent frozen Actor-Free TD methods."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters.actor_free_td_lewm import SUCCESSOR_SCORE_MODES
from tdwm.adapters.frozen_actor_free_td_common import (
    FORMAL_DEPLOYMENT_EPOCH,
    FORMAL_DEPLOYMENT_GLOBAL_STEP,
    METHOD_FAMILY,
    SOURCE_EPOCH,
    SOURCE_METHOD,
    SOURCE_SEED,
    FrozenActorFreeTDMethodSpec,
    is_lower_sha256,
    require_exact_values,
)
from tdwm.adapters.runtime import prepare_cloud_runtime
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

CheckpointLoader = Callable[..., tuple[Any, Any, dict[str, Any], dict[str, Any]]]
PolicyFactory = Callable[..., Any]


def validate_score_mode(score_mode: str) -> str:
    if score_mode not in SUCCESSOR_SCORE_MODES:
        raise ValueError(
            f"score_mode {score_mode!r} is incompatible with a successor head; "
            f"expected one of {sorted(SUCCESSOR_SCORE_MODES)}."
        )
    return score_mode


def frozen_actor_free_td_output_directory_name(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> str:
    """Return a collision-free default directory for one evaluation mode."""

    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    method = protocol.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("Evaluation protocol method must be a non-empty string.")
    selected_score_mode = validate_score_mode(
        score_mode
        or str(protocol.get("inference_objective", {}).get("score_mode", "f_plus_g"))
    )
    run_mode = "smoke" if smoke else "pilot" if pilot else "formal"
    return f"{method}_cube_o50_{selected_score_mode}_{run_mode}"


def _validate_dataset_protocol(protocol: Mapping[str, Any]) -> None:
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("protocol.dataset must be a mapping.")
    source = dataset.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("protocol.dataset.source must bind the audited HDF5 source.")
    if not isinstance(source.get("file"), str) or not source["file"]:
        raise ValueError("dataset.source.file must be a non-empty string.")
    source_size = source.get("size_bytes")
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
    ):
        raise ValueError("dataset.source.size_bytes must be a positive integer.")
    if not is_lower_sha256(source.get("sha256")):
        raise ValueError("dataset.source.sha256 must be lowercase SHA-256.")
    if source_size not in dataset.get("accepted_size_bytes", []):
        raise ValueError("dataset.source.size_bytes must be an accepted dataset size.")

    lance = dataset.get("lance")
    if not isinstance(lance, Mapping):
        raise ValueError("protocol.dataset.lance must be a mapping.")
    if lance.get("manifest_suffix") != ".manifest.json":
        raise ValueError("dataset.lance.manifest_suffix must be '.manifest.json'.")
    if not is_lower_sha256(lance.get("manifest_sha256")):
        raise ValueError("dataset.lance.manifest_sha256 must be lowercase SHA-256.")


def _resolve_frozen_dataset_source(
    dataset_path: Path,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve and hash-bind the exact HDF5 or Lance evaluation input."""

    resolved = _resolve_dataset_source(dataset_path, dataset_config)
    source_config = dataset_config["source"]
    expected_source_sha = source_config["sha256"]
    expected_source_size = int(source_config["size_bytes"])
    if resolved["format"] == "hdf5":
        actual_source_sha = _sha256(dataset_path)
        if actual_source_sha != expected_source_sha:
            raise ValueError("Evaluation HDF5 SHA-256 differs from the protocol.")
        if dataset_path.stat().st_size != expected_source_size:
            raise ValueError("Evaluation HDF5 size differs from the audited source.")
        return {
            **resolved,
            "sha256": actual_source_sha,
            "source_sha256": actual_source_sha,
            "source_size_bytes": expected_source_size,
            "conversion_manifest_sha256": None,
        }

    manifest_path = Path(str(resolved["conversion_manifest_path"]))
    actual_manifest_sha = _sha256(manifest_path)
    expected_manifest_sha = dataset_config["lance"]["manifest_sha256"]
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError("Evaluation Lance manifest SHA-256 differs from the protocol.")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation Lance manifest must contain a JSON object.")
    manifest_source = manifest.get("source")
    if not isinstance(manifest_source, dict):
        raise ValueError("Evaluation Lance manifest is missing source metadata.")
    if manifest_source.get("sha256") != expected_source_sha:
        raise ValueError("Evaluation Lance source SHA-256 differs from the protocol.")
    if manifest_source.get("size_bytes") != expected_source_size:
        raise ValueError("Evaluation Lance source size differs from the protocol.")
    return {
        **resolved,
        "conversion_manifest_sha256": actual_manifest_sha,
        "source_sha256": manifest_source["sha256"],
        "source_size_bytes": manifest_source["size_bytes"],
    }


def _validate_pretrained_protocol(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    pretrained = protocol.get("pretrained_world_model")
    if not isinstance(pretrained, dict):
        raise ValueError(
            "Frozen successor evaluation requires pretrained_world_model metadata."
        )
    require_exact_values(
        pretrained,
        {
            "source_method": SOURCE_METHOD,
            "source_seed": SOURCE_SEED,
            "source_epoch": SOURCE_EPOCH,
            "frozen": True,
        },
        label="pretrained_world_model",
    )
    if not is_lower_sha256(pretrained.get("checkpoint_sha256")):
        raise ValueError(
            "pretrained_world_model.checkpoint_sha256 must be lowercase SHA-256."
        )
    return pretrained


def validate_frozen_actor_free_td_evaluation_protocol(
    protocol: Mapping[str, Any],
    *,
    spec: FrozenActorFreeTDMethodSpec,
) -> None:
    """Validate a protocol for exactly one independently named method."""

    require_exact_values(
        protocol,
        {
            "schema_version": 1,
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "environment": "cube",
            "stage": "planner_evaluation",
        },
        label="protocol",
    )
    if protocol.get("runtime", {}).get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("Evaluation requires stable-worldmodel 0.1.1.")
    _validate_pretrained_protocol(protocol)
    _validate_dataset_protocol(protocol)

    successor = protocol.get("successor")
    if not isinstance(successor, Mapping):
        raise ValueError("protocol.successor must be a mapping.")
    require_exact_values(
        successor,
        {
            "objective_version": spec.objective_version,
            "architecture": "actor_free_successor_head",
            "feature_basis": "augmented_latent_squared_distance",
            "action_conditioning": "dataset_current_action",
            "bootstrap_action": "dataset_next_action",
            "terminal_source": "next_action_nan_invalid",
            "goal_conditioning": "none",
            "actor": "none",
            "reward": "none",
            "td_bootstrap": True,
            "pretrained_world_model_frozen": True,
            "training_branches": ["real_context"],
        },
        label="successor",
    )
    if (
        min(
            int(successor.get("history_size", 0)),
            int(successor.get("hidden_dim", 0)),
        )
        <= 0
    ):
        raise ValueError("successor history and hidden dimensions must be positive.")
    if not 0.0 <= float(successor.get("gamma", -1.0)) < 1.0:
        raise ValueError("successor.gamma must lie in [0, 1).")
    if successor.get("target_world_ema_decay") not in (None, 0, 0.0):
        raise ValueError("successor.target_world_ema_decay must be 0 for frozen LeWM.")
    if successor.get("clamp_successor_cost") is not True:
        raise ValueError("successor.clamp_successor_cost must be true.")

    objective = protocol.get("joint_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("protocol.joint_objective must be a mapping.")
    missing_objective = set(spec.objective_keys) - objective.keys()
    if missing_objective:
        raise ValueError(f"joint_objective is missing {sorted(missing_objective)}.")
    spec.validate_method_config(objective)

    planning = protocol.get("planning")
    if not isinstance(planning, Mapping):
        raise ValueError("protocol.planning must be a mapping.")
    missing_planning = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing_planning:
        raise ValueError(f"Missing planning keys: {sorted(missing_planning)}")
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
        raise ValueError("Frozen Actor-Free TD evaluation is locked to Cube O50/50.")
    inference = protocol.get("inference_objective", {})
    require_exact_values(
        inference,
        {
            "goal_usage": "training_objective_and_planning_linear_readout",
            "goal_enters_successor_head": False,
            "learned_actor": False,
        },
        label="inference_objective",
    )
    validate_score_mode(str(inference.get("score_mode", "f_plus_g")))


def load_frozen_actor_free_td_evaluation_protocol(
    path: str | Path,
    *,
    spec: FrozenActorFreeTDMethodSpec,
) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict):
        raise ValueError("Evaluation protocol must contain a mapping.")
    validate_frozen_actor_free_td_evaluation_protocol(protocol, spec=spec)
    protocol.setdefault("inference_objective", {})["score_mode"] = validate_score_mode(
        str(protocol.get("inference_objective", {}).get("score_mode", "f_plus_g"))
    )
    return protocol


def configure_frozen_actor_free_td_evaluation_mode(
    protocol: Mapping[str, Any],
    *,
    smoke: bool,
    pilot: bool,
    score_mode: str | None = None,
) -> dict[str, Any]:
    if smoke and pilot:
        raise ValueError("Smoke and pilot modes are mutually exclusive.")
    configured = deepcopy(dict(protocol))
    configured.setdefault("inference_objective", {})["score_mode"] = (
        validate_score_mode(
            score_mode
            or str(
                configured.get("inference_objective", {}).get("score_mode", "f_plus_g")
            )
        )
    )
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


def _resolve_joint_checkpoint(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        files = sorted(requested.glob("*.pt"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(
        "A frozen Actor-Free TD deployment checkpoint must be a .pt file or a "
        "directory containing exactly one .pt file."
    )


def validate_frozen_actor_free_td_checkpoint_protocol(
    *,
    payload: Mapping[str, Any],
    successor_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    spec: FrozenActorFreeTDMethodSpec,
    require_formal_completion: bool = True,
) -> None:
    """Bind a validated method checkpoint to its exact evaluation protocol."""

    pretrained = _validate_pretrained_protocol(protocol)
    successor = protocol["successor"]
    expected = {
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "objective_version": spec.objective_version,
        "deployment_checkpoint_version": spec.deployment_checkpoint_version,
        "embed_dim": protocol["model"]["embed_dim"],
        "history_size": successor["history_size"],
        "hidden_dim": successor["hidden_dim"],
        "gamma": successor["gamma"],
        "feature_basis": successor["feature_basis"],
        "pretrained_world_model_frozen": True,
        "pretrained_world_model_source_method": pretrained["source_method"],
        "pretrained_world_model_source_seed": pretrained["source_seed"],
        "pretrained_world_model_source_epoch": pretrained["source_epoch"],
        "pretrained_world_model_sha256": pretrained["checkpoint_sha256"],
        "base_checkpoint_sha256": pretrained["checkpoint_sha256"],
        "training_branches": ["real_context"],
        "predicted_context_detach": True,
    }
    objective = protocol["joint_objective"]
    expected.update({key: objective[key] for key in spec.objective_keys})
    for key, expected_value in expected.items():
        actual = successor_config.get(key)
        if key == "gamma":
            matches = actual is not None and np.isclose(
                float(actual), float(expected_value)
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"Successor checkpoint {key} differs from the evaluation protocol."
            )
    require_exact_values(
        payload,
        {
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "objective_version": spec.objective_version,
            "deployment_checkpoint_version": spec.deployment_checkpoint_version,
        },
        label="checkpoint",
    )
    provenance = payload.get("pretrained_world_model_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "Frozen checkpoint is missing pretrained_world_model_provenance."
        )
    if provenance.get("source_checkpoint_sha256") != pretrained["checkpoint_sha256"]:
        raise ValueError(
            "Pretrained source checkpoint SHA differs from the evaluation protocol."
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


def evaluate_frozen_actor_free_td(
    *,
    spec: FrozenActorFreeTDMethodSpec,
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
) -> dict[str, Any]:
    formal_protocol = load_frozen_actor_free_td_evaluation_protocol(
        protocol_path,
        spec=spec,
    )
    protocol = configure_frozen_actor_free_td_evaluation_mode(
        formal_protocol,
        smoke=smoke,
        pilot=pilot,
        score_mode=score_mode,
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
    world_model, successor, successor_config, payload = checkpoint_loader(
        checkpoint_file,
        map_location=device,
    )
    validate_frozen_actor_free_td_checkpoint_protocol(
        payload=payload,
        successor_config=successor_config,
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
    if int(successor_config["action_dim"]) != expected_action_dim:
        raise ValueError("The TD head action-block dimension is incompatible.")

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
    policy = policy_factory(
        world_model=world_model,
        successor=successor,
        planning=planning,
        gamma=float(protocol["successor"]["gamma"]),
        process={"action": action_processor},
        transform={"pixels": image_transform, "goal": image_transform},
        device=device,
        clamp_tail_cost=bool(protocol["successor"]["clamp_successor_cost"]),
        score_mode=protocol["inference_objective"]["score_mode"],
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
            "objective_version": payload["objective_version"],
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "formal_completion_required": not (smoke or pilot),
            "successor_config": successor_config,
            "pretrained_world_model_provenance": deepcopy(
                payload["pretrained_world_model_provenance"]
            ),
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
        "method": spec.method,
        "method_family": METHOD_FAMILY,
        "variant": spec.variant,
        "score_mode": protocol["inference_objective"]["score_mode"],
        "smoke": smoke,
        "pilot": pilot,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)


__all__ = [
    "FORMAL_O50_PLANNING",
    "frozen_actor_free_td_output_directory_name",
    "configure_frozen_actor_free_td_evaluation_mode",
    "evaluate_frozen_actor_free_td",
    "load_frozen_actor_free_td_evaluation_protocol",
    "validate_frozen_actor_free_td_checkpoint_protocol",
    "validate_frozen_actor_free_td_evaluation_protocol",
    "validate_score_mode",
]
