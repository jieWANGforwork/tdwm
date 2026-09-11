"""Explicit historical-pair or RP1-heldout evaluation for EffAction methods.

Pair selection and action execution are recorded separately: selecting the
historical C--G3 pairs does not turn this H1/RH1 method into its H5 baseline.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tdwm.adapters.eff_action import (
    canonical_eff_action_method,
    make_eff_action_policy,
    validate_eff_action_planning,
)
from tdwm.evaluation.lewm_checkpoint import (
    _git_revision,
    _jsonable,
    _resolve_dataset_source,
    _sha256,
    sample_start_goal_pairs,
)

HISTORICAL_SELECTION_SHA256 = {
    25: "56546fe8725ce0e4670f308c5b325bd64ff2a792373add8c20ddbcab02da6b37",
    50: "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7",
    100: "8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c",
}
CUBE_MANIFEST_SHA256 = (
    "9de531030c6bca21a7b3215d7abea3aaf277e68a1e4cec03c8c6e22ad0d20dcd"
)
CUBE_SOURCE_SHA256 = "3cf6477768f1a2979acefa3aeb6c27c45422b8b6fbce8527419943d3e679a245"
CUBE_SOURCE_BYTES = 74104077358
EVALUATION_KEYS = (
    "pixels",
    "action",
    "qpos",
    "qvel",
    "privileged_block_0_pos",
    "privileged_block_0_quat",
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _write_json(path: Path, value: Any) -> str:
    data = _json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def select_eff_action_episodes(
    episode_lengths: np.ndarray, selection: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select a declared set; heldout sampling is an explicit local convention.

    RP1's public 8000/2000 split is honored. Its unpublished evaluation RNG
    algorithm is not claimed: heldout rows use uniform valid-row sampling,
    without replacement, including the final eligible rank.
    """

    lengths = np.asarray(episode_lengths)
    if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or np.any(lengths <= 0):
        raise ValueError("episode_lengths must be a positive integer vector.")
    protocol = selection.get("protocol")
    if protocol not in {"historical_cg3", "rp1_heldout"}:
        raise ValueError("selection.protocol must be historical_cg3 or rp1_heldout.")
    offset, count, seed = (
        selection.get(k) for k in ("goal_offset", "episodes", "seed")
    )
    if type(offset) is not int or offset not in HISTORICAL_SELECTION_SHA256:
        raise ValueError(
            "goal_offset must be an explicit 25, 50 or 100 primitive steps."
        )
    if type(count) is not int or count <= 0:
        raise ValueError("selection.episodes must be a positive integer.")
    if type(seed) is not int:
        raise ValueError("selection.seed must be an integer.")
    if protocol == "historical_cg3":
        if lengths.size != 10000 or not np.all(lengths == 201):
            raise ValueError(
                "The locked historical Cube set requires 10000 x 201 rows."
            )
        if seed != 42 or count != 50:
            raise ValueError(
                "Historical C--G3 selection is locked to 50 pairs and seed42."
            )
        episodes, starts, ranks = sample_start_goal_pairs(
            lengths, goal_offset=offset, episodes=count, seed=seed
        )
        population_start, population_stop = 0, lengths.size
        algorithm = "historical_valid_row_sampler_excludes_final_rank"
    else:
        if lengths.size != 10000:
            raise ValueError(
                "RP1 heldout selection requires the complete 10000-episode dataset."
            )
        if seed not in (42, 43, 44):
            raise ValueError("RP1 heldout evaluation seeds must be 42, 43 or 44.")
        population_start, population_stop = 8000, 10000
        valid = np.maximum(lengths[population_start:population_stop] - offset, 0)
        cumulative = np.cumsum(valid, dtype=np.int64)
        total = int(cumulative[-1])
        if count > total or total == 0:
            raise ValueError("Not enough heldout valid start-goal rows.")
        ranks = np.sort(np.random.default_rng(seed).choice(total, count, replace=False))
        relative_episodes = np.searchsorted(cumulative, ranks, side="right")
        previous = np.where(
            relative_episodes == 0, 0, cumulative[relative_episodes - 1]
        )
        starts = ranks - previous
        episodes = relative_episodes + population_start
        algorithm = (
            "local_uniform_heldout_valid_rows_without_replacement_including_final_rank"
        )
    pairs = {
        "episode_indices": episodes.astype(np.int64),
        "start_steps": starts.astype(np.int64),
        "goal_steps": (starts + offset).astype(np.int64),
        "valid_row_ranks": ranks.astype(np.int64),
    }
    digest = hashlib.sha256(_json_bytes(pairs)).hexdigest()
    if protocol == "historical_cg3" and digest != HISTORICAL_SELECTION_SHA256[offset]:
        raise RuntimeError("Generated historical pair set failed its exact hash lock.")
    return pairs, {
        "protocol": protocol,
        "goal_offset": offset,
        "episodes": count,
        "seed": seed,
        "episode_range_half_open": [int(population_start), int(population_stop)],
        "sampler": algorithm,
        "selection_sha256": digest,
        "rp1_sampler_bitwise_reproduction_claimed": False,
        "historical_all_dataset_pair_set": protocol == "historical_cg3",
        "heldout_from_episode_range_0_8000": protocol == "rp1_heldout",
    }


def build_eff_action_action_processor(stats: Mapping[str, Any]):
    """Restore training normalization without silently fitting evaluation data."""

    from sklearn.preprocessing import StandardScaler

    if not isinstance(stats, Mapping):
        raise ValueError("Training action_normalization is required.")
    mean = np.asarray(stats.get("mean"), dtype=np.float64)
    scale = np.asarray(stats.get("scale"), dtype=np.float64)
    if mean.shape != (5,) or scale.shape != (5,):
        raise ValueError("Action normalization mean/scale must have five coordinates.")
    if not (np.isfinite(mean).all() and np.isfinite(scale).all()) or np.any(scale <= 0):
        raise ValueError(
            "Action mean/scale must be finite with strictly positive scale."
        )
    variance = np.asarray(stats.get("variance", scale**2), dtype=np.float64)
    if (
        variance.shape != (5,)
        or not np.isfinite(variance).all()
        or np.any(variance < 0)
    ):
        raise ValueError("Action variance must contain five finite nonnegative values.")
    processor = StandardScaler()
    processor.mean_, processor.scale_, processor.var_ = mean, scale, variance
    processor.n_features_in_ = 5
    if "samples" in stats:
        if type(stats["samples"]) is not int or stats["samples"] <= 0:
            raise ValueError("Normalization samples must be a positive integer.")
        processor.n_samples_seen_ = stats["samples"]
    normalized = {
        **dict(stats),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "variance": variance.tolist(),
    }
    return processor, normalized


def _configure_renderer(renderer: Any) -> dict[str, str]:
    if renderer not in {"egl", "osmesa"}:
        raise ValueError("runtime.renderer must explicitly be egl or osmesa.")
    for variable in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        existing = os.environ.get(variable)
        if existing and existing != renderer:
            raise ValueError(
                f"{variable}={existing!r} disagrees with renderer={renderer!r}."
            )
        os.environ[variable] = renderer
    return {key: os.environ[key] for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}


def _validate_dataset_source(
    path: Path, data_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the existing audited Lance conversion without rehashing its HDF5."""

    if (
        data_config.get("format", "lance") != "lance"
        or not path.is_dir()
        or path.suffix != ".lance"
    ):
        raise ValueError(
            "EffAction evaluation requires the audited Cube Lance directory."
        )
    if data_config.get("manifest_sha256") != CUBE_MANIFEST_SHA256:
        raise ValueError(
            "dataset.manifest_sha256 must declare the audited Cube conversion."
        )
    if data_config.get("source_sha256") != CUBE_SOURCE_SHA256:
        raise ValueError(
            "dataset.source_sha256 must declare the audited Cube HDF5 source."
        )
    source = _resolve_dataset_source(
        path,
        {
            "lance": {
                "manifest_suffix": ".manifest.json",
                "image_codec": "jpeg",
                "jpeg_quality": 100,
            }
        },
    )
    manifest_path = Path(source["conversion_manifest_path"])
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != data_config["manifest_sha256"]:
        raise ValueError(
            "Evaluation Lance manifest SHA256 differs from the locked dataset."
        )
    manifest = json.loads(manifest_path.read_bytes())
    origin = manifest.get("source", {})
    if origin.get("sha256") != data_config["source_sha256"]:
        raise ValueError(
            "Evaluation Lance source SHA256 differs from the locked dataset."
        )
    if origin.get("size_bytes") != CUBE_SOURCE_BYTES:
        raise ValueError("Evaluation Lance source size differs from the audited HDF5.")
    verification = manifest.get("verification", {})
    if (
        verification.get("episodes") != 10000
        or verification.get("transitions") != 2010000
    ):
        raise ValueError(
            "The Cube conversion audit must contain 10000 episodes and 2010000 rows."
        )
    return {
        **source,
        "conversion_manifest_sha256": manifest_sha,
        "source_sha256": origin["sha256"],
        "source_size_bytes": origin["size_bytes"],
        "identity_validation": "locked_conversion_manifest_and_loaded_episode_boundaries",
        "source_bytes_rehashed_this_run": False,
    }


def _validate_metrics(
    metrics: Mapping[str, Any], count: int
) -> tuple[np.ndarray, float]:
    # SWM 0.1.1's dataset branch returns bool; its ordinary episode branch
    # returns float64 0/1. In either case success_rate is a percentage.
    outcomes = np.asarray(metrics.get("episode_successes"))
    if (
        outcomes.shape != (count,)
        or outcomes.dtype.kind not in "biuf"
        or not (np.isfinite(outcomes).all() and np.isin(outcomes, (0, 1)).all())
    ):
        raise ValueError(
            "SWM must return one finite binary outcome per selected episode."
        )
    success_rate = float(metrics.get("success_rate", float("nan")))
    if not math.isfinite(success_rate) or not math.isclose(
        success_rate, 100.0 * float(outcomes.mean())
    ):
        raise ValueError(
            "SWM success_rate must equal the percentage of binary outcomes."
        )
    return outcomes, success_rate


def evaluate_eff_action(
    *,
    world_model: nn.Module,
    successor: nn.Module,
    value: nn.Module,
    config: Mapping[str, Any],
    dataset_path: str | Path,
    output_dir: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    planner: nn.Module | None = None,
    action_normalization: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    video: bool = False,
) -> dict[str, Any]:
    """Evaluate preloaded heads with public SWM, writing immutable run evidence.

    Checkpoint loading and content validation belong to the training runtime.
    This entry binds the supplied checkpoint identity to its output and fails
    if an existing or partial output would be overwritten.
    """

    method = canonical_eff_action_method(str(config.get("method", "")))
    planning = dict(config.get("planning", {}))
    validate_eff_action_planning(planning, method)
    budget = planning.get("episode_budget")
    if type(budget) is not int or budget <= 0 or budget % 5:
        raise ValueError(
            "episode_budget must be an explicit positive multiple of five primitive steps."
        )
    status = config.get("protocol_status", "provisional")
    if status not in {"provisional", "user_locked"}:
        raise ValueError("protocol_status must be provisional or user_locked.")
    runtime = dict(config.get("runtime", {}))
    if runtime.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("EffAction requires stable-worldmodel==0.1.1.")
    if runtime.get("precision") not in {"fp32", "32-true"}:
        raise ValueError("EffAction evaluation precision must explicitly be fp32.")
    if importlib.metadata.version("stable-worldmodel") != "0.1.1":
        raise RuntimeError("Installed stable-worldmodel is not version0.1.1.")
    renderer = _configure_renderer(runtime.get("renderer"))
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("checkpoint_metadata is required.")
    checkpoint = dict(checkpoint_metadata)
    digest = checkpoint.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Checkpoint metadata must bind an exact lowercase SHA256.")
    if canonical_eff_action_method(str(checkpoint.get("method", ""))) != method:
        raise ValueError("Checkpoint method differs from the evaluation method.")
    stats_input = (
        action_normalization
        if action_normalization is not None
        else config.get("action_normalization")
    )
    processor, stats = build_eff_action_action_processor(stats_input)
    path = Path(dataset_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(
            "EffAction evaluation requires a new empty output directory."
        )
    data_config = dict(config.get("dataset", {}))
    if not path.exists():
        raise FileNotFoundError(path)
    source = _validate_dataset_source(path, data_config)

    import stable_worldmodel as swm
    from torchvision.transforms import v2 as transforms

    dataset = swm.data.load_dataset(
        str(path), format="lance", keys_to_load=list(EVALUATION_KEYS)
    )
    if int(dataset.get_dim("action")) != 5:
        raise ValueError("Cube evaluation requires five primitive action coordinates.")
    lengths = np.asarray(dataset.lengths)
    if (
        lengths.shape != (10000,)
        or not np.all(lengths == 201)
        or int(lengths.sum()) != 2010000
    ):
        raise ValueError(
            "Loaded Cube data must contain 10000 x 201 rows, totaling 2010000."
        )
    source.update(episodes=int(lengths.size), rows=int(lengths.sum()))
    pairs, selection_metadata = select_eff_action_episodes(
        lengths, config.get("selection", {})
    )
    image = dict(config.get("image_preprocessing", {}))
    if (
        image.get("size") != 224
        or image.get("mean") != [0.485, 0.456, 0.406]
        or image.get("std") != [0.229, 0.224, 0.225]
    ):
        raise ValueError(
            "The frozen LeWM encoder requires its exact224px ImageNet preprocessing."
        )
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image["mean"], std=image["std"]),
            transforms.Resize(size=image["size"]),
        ]
    )
    world_config = dict(config.get("world", {}))
    for key, expected in {
        "env_name": "swm/OGBCube-v0",
        "env_type": "single",
        "ob_type": "states",
        "image_size": 224,
        "multiview": False,
        "visualize_info": False,
        "terminate_at_goal": True,
    }.items():
        if world_config.get(key) != expected:
            raise ValueError(f"world.{key} must be {expected!r}.")
    policy = make_eff_action_policy(
        world_model=world_model,
        successor=successor,
        value=value,
        planner=planner,
        method=method,
        planning=planning,
        epsilon=config["epsilon"],
        process={"action": processor},
        transform={"pixels": transform, "goal": transform},
        device=device,
    )
    destination.mkdir(parents=True, exist_ok=True)
    selection_sha = _write_json(destination / "episode_selection.json", pairs)
    stats_sha = _write_json(destination / "action_normalization.json", stats)
    execution = {
        "horizon_blocks": 1,
        "receding_horizon_blocks": 1,
        "primitive_actions_per_block": 5,
        "primitive_action_dim": 5,
        "planned_action_coordinates": 25,
        "real_feedback_interval_primitive_steps": 5,
        "episode_budget_primitive_steps": budget,
        "initial_real_history_frames": 1,
        "f_rollout_used": False,
        "solver": "CEM" if method == "EffAction" else "learned_action_updater",
        "candidate_bound_policy": "unbounded_normalized"
        if method == "EffAction"
        else "projected_normalized_bounds",
        "selected_action": "final_elite_mean"
        if method == "EffAction"
        else "final_residual_update",
        "environment_reset_seed_source": "SWM_dataset_seed_column_or_None",
    }
    manifest = {
        "schema_version": 1,
        "method": method,
        "protocol_status": status,
        "formal_protocol_claimed": status == "user_locked",
        "config": _jsonable(dict(config)),
        "checkpoint": checkpoint,
        "dataset": source,
        "selection": selection_metadata,
        "selection_sha256": selection_sha,
        "action_normalization_sha256": stats_sha,
        "execution": execution,
        "score_definition": "J(a)=-norm(z_goal-z_state)/(V(G(z_state,E_A(a),m),m)+epsilon)",
        "task_definition": "sqrt(192)*z_goal/norm(z_goal)",
        "historical_selection_does_not_imply_historical_execution_protocol": True,
        "runtime": {
            **runtime,
            **renderer,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(),
        },
    }
    _write_json(destination / "protocol_manifest.json", manifest)
    world = swm.World(
        world_config["env_name"],
        num_envs=selection_metadata["episodes"],
        image_shape=(224, 224),
        max_episode_steps=budget,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
    )
    started = time.perf_counter()
    try:
        world.set_policy(policy)
        with torch.inference_mode():
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=pairs["episode_indices"].tolist(),
                start_steps=pairs["start_steps"].tolist(),
                goal_offset=selection_metadata["goal_offset"],
                eval_budget=budget,
                video=destination / "videos" if video else None,
                callables=[
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
                ],
            )
    finally:
        world.close()
    count = selection_metadata["episodes"]
    outcomes, success_rate = _validate_metrics(metrics, count)
    result = {
        "method": method,
        "protocol_status": status,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "execution": execution,
        "selection_protocol": selection_metadata["protocol"],
        "selection_sha256": selection_sha,
        "goal_offset": selection_metadata["goal_offset"],
        "selection_seed": selection_metadata["seed"],
        "planning_seed": planning["planning_seed"],
        "renderer": runtime["renderer"],
        "checkpoint_sha256": digest,
        "protocol_manifest": str(destination / "protocol_manifest.json"),
        "success_count": int(outcomes.sum()),
        "success_rate_percent": success_rate,
        "swm_episode_outcomes_dtype": str(outcomes.dtype),
        "per_episode": [
            {
                "selection_index": i,
                "episode_index": int(pairs["episode_indices"][i]),
                "start_step": int(pairs["start_steps"][i]),
                "goal_step": int(pairs["goal_steps"][i]),
                "success": bool(outcomes[i]),
            }
            for i in range(count)
        ],
        "parameter_counts": {
            name: sum(p.numel() for p in module.parameters())
            for name, module in (
                ("world_model", world_model),
                ("successor", successor),
                ("value", value),
                ("planner", planner),
            )
            if module is not None
        },
    }
    _write_json(destination / "results.json", result)
    return _jsonable(result)


__all__ = [
    "HISTORICAL_SELECTION_SHA256",
    "build_eff_action_action_processor",
    "evaluate_eff_action",
    "select_eff_action_episodes",
]
