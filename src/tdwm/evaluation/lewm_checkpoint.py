from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdwm.adapters import prepare_cloud_runtime
from tdwm.evaluation.full_plan_revalidation import (
    configure_full_plan_revalidation,
    full_plan_revalidation_metadata,
    require_new_revalidation_output,
)

REQUIRED_PLANNING_KEYS = {
    "horizon",
    "candidates",
    "iterations",
    "elites",
    "action_block",
    "receding_horizon",
    "episode_budget",
    "planning_seed",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("The experiment protocol must use schema_version 1.")
    if protocol.get("method") != "lewm":
        raise ValueError("This evaluator only accepts the original LeWM method.")
    if protocol.get("environment") != "cube":
        raise ValueError("This evaluator only accepts the OGBench-Cube environment.")

    planning = protocol.get("planning", {})
    missing = REQUIRED_PLANNING_KEYS - planning.keys()
    if missing:
        raise ValueError(f"Missing planning protocol keys: {sorted(missing)}")
    if planning["elites"] > planning["candidates"]:
        raise ValueError("CEM elites cannot exceed candidates.")
    if planning["receding_horizon"] > planning["horizon"]:
        raise ValueError("Receding horizon cannot exceed the CEM horizon.")
    if planning["horizon"] * planning["action_block"] > planning["episode_budget"]:
        raise ValueError("The planned action sequence exceeds the episode budget.")

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("episodes", 0) <= 0:
        raise ValueError("Evaluation episodes must be positive.")
    if evaluation.get("goal_offset", 0) <= 0:
        raise ValueError("Goal offset must be positive.")


def sample_start_goal_pairs(
    episode_lengths: np.ndarray,
    *,
    goal_offset: int,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the stable-worldmodel 0.1.1 valid-row sampler.

    The upstream evaluator samples ranks in the flattened set of valid dataset
    rows, excludes its final rank, sorts the chosen rows, and then resolves each
    row back to an episode and step.
    """

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    valid_per_episode = np.maximum(lengths - goal_offset, 0)
    cumulative = np.cumsum(valid_per_episode)
    total = int(cumulative[-1]) if cumulative.size else 0
    if total <= 1:
        raise ValueError("Dataset has no valid start/goal pairs.")
    if episodes > total - 1:
        raise ValueError(
            f"Requested {episodes} evaluations but only {total - 1} are sampleable."
        )

    rng = np.random.default_rng(seed)
    ranks = np.sort(rng.choice(total - 1, size=episodes, replace=False))
    episode_indices = np.searchsorted(cumulative, ranks, side="right")
    previous = np.where(episode_indices == 0, 0, cumulative[episode_indices - 1])
    start_steps = ranks - previous
    return episode_indices, start_steps, ranks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _resolve_checkpoint_file(cache_root: Path, checkpoint_name: str) -> Path:
    local = cache_root / "checkpoints" / checkpoint_name
    if local.suffix == ".pt":
        return local
    if local.is_dir():
        weights = sorted(local.glob("*.pt"))
        if len(weights) == 1:
            return weights[0]
    mirror = cache_root / "checkpoints" / (
        "models--" + checkpoint_name.replace("/", "--")
    )
    weights = sorted(mirror.glob("*.pt")) if mirror.is_dir() else []
    if len(weights) == 1:
        return weights[0]
    raise FileNotFoundError(
        f"Cannot find exactly one checkpoint for {checkpoint_name!r} under {cache_root}."
    )


def _resolve_local_export_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[str, Path, Path]:
    """Resolve a Stable World Model export written by the LeWM trainer."""

    requested = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir = requested if requested.is_dir() else requested.parent
    weights = sorted(checkpoint_dir.glob("*.pt"))
    if len(weights) != 1:
        raise FileNotFoundError(
            "A local LeWM export must be a directory containing exactly one .pt file."
        )
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "A local LeWM export must use Stable World Model's "
            "<cache_dir>/checkpoints/<run_name> layout."
        )
    return checkpoint_dir.name, weights[0], checkpoint_dir.parent.parent


def _resolve_dataset_source(
    dataset_path: Path, dataset_config: dict[str, Any]
) -> dict[str, Any]:
    if dataset_path.is_file():
        expected_sizes = dataset_config.get(
            "accepted_size_bytes", [dataset_config.get("expected_size_bytes")]
        )
        actual_size = dataset_path.stat().st_size
        if actual_size not in expected_sizes:
            raise ValueError(
                f"Dataset size mismatch: expected one of {expected_sizes}, "
                f"found {actual_size}."
            )
        return {
            "path": str(dataset_path),
            "format": "hdf5",
            "size_bytes": actual_size,
            "conversion_manifest_path": None,
        }

    lance = dataset_config.get("lance")
    if (
        not dataset_path.is_dir()
        or dataset_path.suffix.lower() != ".lance"
        or lance is None
    ):
        raise FileNotFoundError(f"Cube dataset not found: {dataset_path}")

    manifest_path = Path(f"{dataset_path}{lance['manifest_suffix']}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Audited Lance conversion manifest not found: {manifest_path}"
        )
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    destination = manifest.get("destination", {})
    conversion = manifest.get("conversion", {})
    if destination.get("format") != "lance":
        raise ValueError("The Cube Lance manifest does not describe a Lance table.")
    if destination.get("image_codec") != lance["image_codec"]:
        raise ValueError("The Cube Lance manifest image codec differs from protocol.")
    if destination.get("jpeg_quality") != lance["jpeg_quality"]:
        raise ValueError("The Cube Lance manifest JPEG quality differs from protocol.")
    if conversion.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("The Cube Lance manifest was not created by stable-worldmodel 0.1.1.")
    return {
        "path": str(dataset_path),
        "format": "lance",
        "size_bytes": destination.get("size_bytes"),
        "conversion_manifest_path": str(manifest_path),
    }


def evaluate_official_lewm(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    checkpoint_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    video: bool = False,
    smoke: bool = False,
    full_plan_revalidation: bool = False,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    formal_protocol = deepcopy(protocol) if full_plan_revalidation else None
    if smoke:
        protocol["id"] = f"{protocol['id']}_smoke"
        protocol["evaluation"]["episodes"] = 1
        protocol["planning"].update(
            {
                "candidates": 8,
                "iterations": 1,
                "elites": 2,
                "episode_budget": 25,
            }
        )
        protocol["smoke"] = True
        validate_protocol(protocol)
    if full_plan_revalidation:
        protocol = configure_full_plan_revalidation(protocol)
        require_new_revalidation_output(output_dir)
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_source = _resolve_dataset_source(dataset_path, protocol["dataset"])

    compatibility = prepare_cloud_runtime()

    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2 as transforms

    package_version = importlib.metadata.version("stable-worldmodel")
    if package_version != protocol["runtime"]["stable_worldmodel_version"]:
        raise RuntimeError(
            f"Expected stable-worldmodel {protocol['runtime']['stable_worldmodel_version']}, "
            f"found {package_version}."
        )

    cache_root = Path(
        os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable_worldmodel"))
    ).expanduser()
    if checkpoint_path is not None:
        if checkpoint_name is not None:
            raise ValueError("Pass either checkpoint_name or checkpoint_path, not both.")
        checkpoint_name, checkpoint_file, checkpoint_cache_dir = (
            _resolve_local_export_checkpoint(checkpoint_path)
        )
        checkpoint_source = "local_export"
    else:
        checkpoint_name = checkpoint_name or protocol["checkpoint"]["name"]
        checkpoint_file = _resolve_checkpoint_file(cache_root, checkpoint_name)
        checkpoint_cache_dir = cache_root
        checkpoint_source = "stable_worldmodel_cache"
    checkpoint_sha256 = _sha256(checkpoint_file)
    expected_checkpoint_hash = protocol["checkpoint"].get("sha256")
    if expected_checkpoint_hash and checkpoint_sha256 != expected_checkpoint_hash:
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_hash}, found {checkpoint_sha256}."
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
        raise ValueError(
            f"Expected {dataset_cfg['expected_episodes']} episodes, found {actual_episodes}."
        )
    if actual_transitions != dataset_cfg["expected_transitions"]:
        raise ValueError(
            f"Expected {dataset_cfg['expected_transitions']} transitions, "
            f"found {actual_transitions}."
        )

    evaluation_cfg = protocol["evaluation"]
    planning_cfg = protocol["planning"]
    episode_indices, start_steps, valid_ranks = sample_start_goal_pairs(
        np.asarray(dataset.lengths),
        goal_offset=evaluation_cfg["goal_offset"],
        episodes=evaluation_cfg["episodes"],
        seed=planning_cfg["planning_seed"],
    )
    goal_steps = start_steps + evaluation_cfg["goal_offset"]

    selection = {
        "episode_indices": episode_indices,
        "start_steps": start_steps,
        "goal_steps": goal_steps,
        "valid_row_ranks": valid_ranks,
    }
    _write_json(output_dir / "episode_selection.json", selection)

    runtime_manifest = {
        "protocol": protocol,
        "protocol_path": str(Path(protocol_path).resolve()),
        "dataset": {
            **dataset_source,
            "episodes": actual_episodes,
            "transitions": actual_transitions,
        },
        "checkpoint": {
            "name": checkpoint_name,
            "path": str(checkpoint_file),
            "cache_dir": str(checkpoint_cache_dir),
            "source": checkpoint_source,
            "sha256": checkpoint_sha256,
        },
        "selection": selection,
        "runtime": {
            "stable_worldmodel": package_version,
            "stable_pretraining": importlib.metadata.version("stable-pretraining"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tdwm_git_revision": _git_revision(),
            "cuda_device": torch.cuda.get_device_name(0),
            "compatibility_adapter": compatibility,
        },
    }
    if full_plan_revalidation:
        runtime_manifest["formal_protocol"] = formal_protocol
        runtime_manifest.update(full_plan_revalidation_metadata(protocol))
    _write_json(output_dir / "protocol_manifest.json", runtime_manifest)

    action_stats_path = output_dir / "action_normalization.json"
    if action_stats_path.is_file():
        print(f"Reusing action statistics from {action_stats_path}...", flush=True)
        with action_stats_path.open() as stream:
            action_stats = json.load(stream)
        action_processor = StandardScaler()
        action_processor.mean_ = np.asarray(action_stats["mean"], dtype=np.float64)
        action_processor.scale_ = np.asarray(action_stats["scale"], dtype=np.float64)
        action_processor.var_ = np.asarray(action_stats["variance"], dtype=np.float64)
        action_processor.n_features_in_ = len(action_processor.mean_)
        action_processor.n_samples_seen_ = int(action_stats["samples"])
    else:
        print("Loading action statistics from the full evaluation dataset...", flush=True)
        action_processor = StandardScaler().fit(dataset.get_col_data("action"))
        action_stats = {
            "mean": action_processor.mean_,
            "scale": action_processor.scale_,
            "variance": action_processor.var_,
            "samples": int(action_processor.n_samples_seen_),
        }
        _write_json(action_stats_path, action_stats)

    model = swm.wm.load_pretrained(
        checkpoint_name, cache_dir=str(checkpoint_cache_dir)
    ).to("cuda").eval()
    model.requires_grad_(False)
    expected_parameters = protocol["checkpoint"].get("parameters")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameters and parameter_count != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} model parameters, found {parameter_count}."
        )

    image_stats = protocol["image_preprocessing"]
    image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=image_stats["mean"], std=image_stats["std"]
            ),
            transforms.Resize(size=protocol["world"]["image_size"]),
        ]
    )
    process = {"action": action_processor}
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=planning_cfg["solver_batch_size"],
        num_samples=planning_cfg["candidates"],
        var_scale=planning_cfg["initial_variance"],
        n_steps=planning_cfg["iterations"],
        topk=planning_cfg["elites"],
        device="cuda",
        seed=planning_cfg["planning_seed"],
    )
    plan_config = swm.PlanConfig(
        horizon=planning_cfg["horizon"],
        receding_horizon=planning_cfg["receding_horizon"],
        history_len=protocol["context"]["plan_config_history_len"],
        action_block=planning_cfg["action_block"],
        warm_start=planning_cfg["warm_start"],
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform={"pixels": image_transform, "goal": image_transform},
    )

    world_cfg = protocol["world"]
    world = swm.World(
        world_cfg["env_name"],
        num_envs=evaluation_cfg["episodes"],
        image_shape=(world_cfg["image_size"], world_cfg["image_size"]),
        max_episode_steps=planning_cfg["episode_budget"],
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
            "args": {
                "qpos": {"value": "qpos"},
                "qvel": {"value": "qvel"},
            },
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

    video_path = output_dir / "videos" if video else None
    started = time.time()
    try:
        with torch.inference_mode():
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=episode_indices.tolist(),
                start_steps=start_steps.tolist(),
                goal_offset=evaluation_cfg["goal_offset"],
                eval_budget=planning_cfg["episode_budget"],
                callables=callables,
                video=video_path,
            )
    finally:
        world.close()
    elapsed = time.time() - started

    result = {
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "parameter_count": parameter_count,
        "smoke": smoke,
        "protocol_manifest": str(output_dir / "protocol_manifest.json"),
    }
    if full_plan_revalidation:
        result.update(
            {
                "score_mode": "f_only",
                "planning_horizon": planning_cfg["horizon"],
                "goal_offset": evaluation_cfg["goal_offset"],
                "episode_budget": planning_cfg["episode_budget"],
                **full_plan_revalidation_metadata(protocol),
            }
        )
    _write_json(output_dir / "results.json", result)
    return _jsonable(result)
