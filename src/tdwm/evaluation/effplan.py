"""Paired formal O25/O50/O100 evaluation of F-only, Eff and EffPlan."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from tdwm.adapters.effplan import EffCEMCost, EffPlanSolver, EffPlanTrackingCost
from tdwm.adapters.runtime import prepare_cloud_runtime
from tdwm.evaluation.frozen_actor_free_td_common import _resolve_frozen_dataset_source
from tdwm.evaluation.lewm_checkpoint import _git_revision, _jsonable
from tdwm.evaluation.mc_gt_lewm import _load_action_processor
from tdwm.training.eff_data import EpisodePartition, held_out_episode_pairs
from tdwm.training.eff_protocol import (
    baseline_reference,
    load_eff_protocol,
    sha256_file,
)
from tdwm.training.eff_run import EffRunSettings, run_directory_lock, write_json_atomic
from tdwm.training.eff_runtime import canonical_sha256, load_eff_model
from tdwm.training.effplan_runtime import load_effplan_planner
from tdwm.training.frozen_actor_free_td import _resolve_local_pretrained_lewm_export


def prepare_eff_selections(
    *,
    config_path: str | Path,
    terminal_metadata: str | Path,
    output_dir: str | Path,
) -> dict:
    """Seal pairs before choosing a model; never inherit all-episode C pairs."""
    config = load_eff_protocol(config_path)
    metadata = json.loads((Path(terminal_metadata) / "manifest.json").read_text())
    if metadata["dataset_source_sha256"] != config["source"]["dataset_source_sha256"]:
        raise ValueError("Selection metadata and model protocol use different data.")
    evaluation = config["evaluation"]
    if evaluation["planning_seed"] != 42 or evaluation["episodes_per_protocol"] != 50:
        raise ValueError("The declared formal draw is seed 42 with 50 episodes.")
    output = Path(output_dir).expanduser().resolve()
    paths = {}
    for offset in (25, 50, 100):
        pairs = held_out_episode_pairs(
            np.asarray(metadata["episode_lengths"]),
            episode_ids=EpisodePartition.rp1_cube().evaluation,
            goal_offset=offset,
            count=50,
            seed=42,
        )
        selection = {
            "format": "tdwm-eff-selection-v1",
            "goal_offset": offset,
            "planning_seed": 42,
            "episodes": 50,
            "dataset_source_sha256": config["source"]["dataset_source_sha256"],
            "held_out_episode_range": [8000, 10000],
            "pairs": pairs,
        }
        path = output / f"o{offset}_selection.json"
        if path.exists():
            if json.loads(path.read_text()) != selection:
                raise FileExistsError(
                    "Existing paired draw differs; do not replace it."
                )
        else:
            write_json_atomic(path, selection)
        paths[f"O{offset}"] = {"path": str(path), "sha256": sha256_file(path)}
    return paths


def validate_selection(selection: dict, *, source_sha256: str) -> None:
    if selection.get("format") != "tdwm-eff-selection-v1":
        raise ValueError("Unrecognized Eff paired selection artifact.")
    if selection.get("goal_offset") not in (25, 50, 100):
        raise ValueError("Only the predeclared O25/O50/O100 tasks are supported.")
    if selection.get("dataset_source_sha256") != source_sha256:
        raise ValueError("Evaluation pairs refer to different data.")
    if selection.get("planning_seed") != 42 or selection.get("episodes") != 50:
        raise ValueError("Formal evaluation needs the fixed 50-pair seed-42 draw.")
    pairs = selection["pairs"]
    if any(
        len(pairs[key]) != 50
        for key in ("episode_indices", "start_steps", "goal_steps", "valid_row_ranks")
    ):
        raise ValueError("Formal selection is incomplete.")
    ids = np.asarray(pairs["episode_indices"])
    starts = np.asarray(pairs["start_steps"])
    goals = np.asarray(pairs["goal_steps"])
    if (
        np.any((ids < 8000) | (ids >= 10000))
        or np.any(starts < 0)
        or np.any(goals > 200)
    ):
        raise ValueError(
            "Evaluation draw leaks training episodes or crosses an episode boundary."
        )
    if not np.all(goals - starts == selection["goal_offset"]):
        raise ValueError("Start-goal offsets do not match the declared task.")
    if len(set(zip(ids.tolist(), starts.tolist(), strict=True))) != 50:
        raise ValueError("Formal pairs must be distinct.")
    expected = held_out_episode_pairs(
        np.full(10000, 201),
        episode_ids=EpisodePartition.rp1_cube().evaluation,
        goal_offset=selection["goal_offset"],
        count=50,
        seed=42,
    )
    if pairs != expected:
        raise ValueError(
            "Pair selection does not match the predeclared sampling algorithm."
        )


def _read_eff_for_evaluation(config, checkpoint, manifest_path, device):
    if checkpoint is None or manifest_path is None:
        raise ValueError(
            "Eff evaluation requires its checkpoint and training manifest."
        )
    metadata = json.loads(Path(manifest_path).read_text())
    settings = EffRunSettings(**config["eff_training"]["settings"])
    if (
        metadata.get("status") != "complete"
        or metadata.get("completed_updates") != settings.total_updates
    ):
        raise ValueError("Eff training has not completed the formal update budget.")
    identity = metadata["identity"]
    if (
        identity["settings_sha256"] != canonical_sha256(metadata["settings"])
        or metadata["settings"] != config["eff_training"]["settings"]
    ):
        raise ValueError(
            "Eff training settings differ from the evaluation configuration."
        )
    for key in (
        "lewm_checkpoint_sha256",
        "dataset_source_sha256",
        "column_normalization_sha256",
        "frozen_store_manifest_sha256",
    ):
        if identity["source"][key] != config["source"][key]:
            raise ValueError(f"Eff source mismatch: {key}.")
    if identity["training_episodes"] != list(range(8000)) or identity[
        "validation_episodes"
    ] != list(range(8000, 10000)):
        raise ValueError(
            "Eff checkpoint was not trained on the declared episode split."
        )
    model, payload = load_eff_model(
        checkpoint,
        expected_identity=identity,
        expected_global_step=settings.total_updates,
        device=device,
    )
    return model, payload


def evaluate_effplan(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    lewm_checkpoint: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    method: str,
    device: str,
    eff_checkpoint: str | Path | None = None,
    eff_manifest: str | Path | None = None,
    planner_checkpoint: str | Path | None = None,
    planner_manifest: str | Path | None = None,
    video: bool = False,
) -> dict:
    """Full public SWM world.evaluate call; no reduced/smoke score substituted."""
    config = load_eff_protocol(config_path, stage="evaluation")
    if method not in {"F-only", "Eff", "EffPlan"}:
        raise ValueError("Unknown predeclared method.")
    ev = config["evaluation"]
    if (
        ev["horizon"],
        ev["action_block"],
        ev["candidates"],
        ev["iterations"],
        ev["elites"],
    ) != (5, 5, 300, 30, 30):
        raise ValueError("Formal CEM must retain the 5-block, 300x30, top-30 protocol.")
    if os.environ.get("MUJOCO_GL") != ev["render_backend"]:
        raise ValueError(
            "Set the locked MUJOCO_GL backend before starting this process."
        )
    if (
        ev["render_backend"] == "egl"
        and not os.environ.get("MUJOCO_EGL_DEVICE_ID", "").isdigit()
    ):
        raise ValueError("Formal EGL rendering requires an explicitly pinned device.")
    selection = json.loads(Path(selection_path).read_text())
    validate_selection(
        selection, source_sha256=config["source"]["dataset_source_sha256"]
    )
    offset = selection["goal_offset"]
    receding = ev["receding_horizons"][str(offset)]
    if type(receding) is not int or receding not in (1, 5):
        raise ValueError("Replanning interval must be explicitly one or five chunks.")
    if ev["episode_budget_multiplier"] != 2:
        raise ValueError("Formal episode budget must be twice the goal offset.")
    reference = baseline_reference(config, config_path)
    compatibility = prepare_cloud_runtime()
    import stable_worldmodel as swm
    import torch
    from torchvision.transforms import v2 as transforms

    if importlib.metadata.version("stable-worldmodel") != "0.1.1":
        raise RuntimeError("Wrong stable-worldmodel version.")
    name, checkpoint_file, cache = _resolve_local_pretrained_lewm_export(
        lewm_checkpoint
    )
    if sha256_file(checkpoint_file) != config["source"]["lewm_checkpoint_sha256"]:
        raise ValueError(
            "Evaluation LeWM differs from the critic's frozen world model."
        )
    world_model = (
        swm.wm.load_pretrained(name, cache_dir=str(cache))
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    eff = planner = None
    checkpoint_records = {
        "LeWM": {"path": str(checkpoint_file), "sha256": sha256_file(checkpoint_file)}
    }
    if method != "F-only":
        load_eff_protocol(config_path, stage="eff_training")
        eff, payload = _read_eff_for_evaluation(
            config, eff_checkpoint, eff_manifest, device
        )
        checkpoint_records["Eff"] = {
            "path": str(eff_checkpoint),
            "sha256": sha256_file(eff_checkpoint),
            "global_step": payload["global_step"],
        }
    if method == "EffPlan":
        load_eff_protocol(config_path, stage="planner_refinement")
        if planner_manifest is None or planner_checkpoint is None:
            raise ValueError(
                "EffPlan requires its completed second-stage checkpoint and manifest."
            )
        pmeta = json.loads(Path(planner_manifest).read_text())
        run = config["planner_refinement"]["run"]
        expected_steps = run["epochs"] * run["updates_per_epoch"]
        if (
            pmeta.get("status") != "complete"
            or pmeta.get("completed_updates") != expected_steps
        ):
            raise ValueError("EffPlan refinement has not completed its full budget.")
        if (
            pmeta["identity"]["source"]["eff_checkpoint_sha256"]
            != checkpoint_records["Eff"]["sha256"]
        ):
            raise ValueError(
                "EffPlan was trained with a different frozen Eff checkpoint."
            )
        planner, ppayload = load_effplan_planner(
            planner_checkpoint,
            expected_identity=pmeta["identity"],
            expected_global_step=expected_steps,
            expected_phase="refinement",
            device=device,
        )
        if ppayload["settings"]["target_readout"] != ev["target_readout"]:
            raise ValueError(
                "EffPlan training and deployment choose different G/V readouts."
            )
        checkpoint_records["EffPlan"] = {
            "path": str(planner_checkpoint),
            "sha256": sha256_file(planner_checkpoint),
            "global_step": expected_steps,
        }
    provenance = _resolve_frozen_dataset_source(
        Path(dataset_path).resolve(), reference["dataset"]
    )
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=provenance["format"],
        keys_to_load=reference["dataset"]["keys_to_load"],
    )
    if len(dataset.lengths) != 10000 or not np.all(np.asarray(dataset.lengths) == 201):
        raise ValueError("Evaluation dataset differs from the locked episode layout.")
    output = Path(output_dir).expanduser().resolve()
    with run_directory_lock(output):
        if (output / "protocol_manifest.json").exists():
            raise FileExistsError(
                "Evaluation output is already occupied; do not overwrite a run."
            )
        processor, action_stats = _load_action_processor(
            dataset, output / "action_normalization.json"
        )
        image = reference["image_preprocessing"]
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=image["mean"], std=image["std"]),
                transforms.Resize(size=reference["world"]["image_size"]),
            ]
        )
        if method == "EffPlan":
            model = EffPlanTrackingCost(world_model, eff, target=ev["target_readout"])
            allocation = tuple(ev["effplan_search_iterations"])
            if sum(allocation) != 30:
                raise ValueError(
                    "EffPlan must share the predeclared 30 CEM rounds across searches."
                )
            solver = EffPlanSolver(
                model=model,
                planner=planner,
                search_iterations=allocation,
                candidates=300,
                elites=30,
                batch_size=ev["cem_batch_size"],
                seed=42,
                device=device,
                epsilon=ev["epsilon"],
                dynamics_coefficient=ev["effplan_dynamics_coefficient"],
            )
            score = "state_path_tracking"
            extra_rerolls = len(allocation) - 1
        else:
            if method == "Eff" and ev["eff_score"] != "terminal_eff_cost":
                raise ValueError(
                    "The current Eff adapter implements terminal G/V scoring only."
                )
            model = (
                world_model
                if method == "F-only"
                else EffCEMCost(world_model, eff, target=ev["target_readout"])
            )
            solver = swm.solver.CEMSolver(
                model=model,
                batch_size=ev["cem_batch_size"],
                num_samples=300,
                topk=30,
                n_steps=30,
                var_scale=1.0,
                device=device,
                seed=42,
            )
            score = (
                "terminal_latent_squared_distance"
                if method == "F-only"
                else "terminal_eff_cost"
            )
            extra_rerolls = 0
        plan = swm.PlanConfig(
            horizon=5,
            receding_horizon=receding,
            history_len=1,
            action_block=5,
            warm_start=True,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan,
            process={"action": processor},
            transform={"pixels": transform, "goal": transform},
        )
        manifest = {
            "format": "tdwm-eff-formal-evaluation-v1",
            "status": "running",
            "method": method,
            "score_mode": score,
            "protocol": f"O{offset}",
            "config": config,
            "checkpoints": checkpoint_records,
            "dataset": provenance,
            "selection": selection,
            "selection_sha256": sha256_file(selection_path),
            "paired_protocol": {
                "goal_offset": offset,
                "episodes": 50,
                "planning_seed": 42,
                "receding_horizon": receding,
                "horizon": 5,
                "action_block": 5,
                "episode_budget": 2 * offset,
                "cem_candidates": 300,
                "cem_iterations": 30,
                "cem_elites": 30,
                "cem_batch_size": ev["cem_batch_size"],
                "warm_start": True,
                "render_backend": ev["render_backend"],
                "lewm_checkpoint_sha256": config["source"]["lewm_checkpoint_sha256"],
            },
            "compute": {
                "candidate_plans_per_decision": 9000,
                "extra_returned_action_rerolls": extra_rerolls,
            },
            "normalization": _jsonable(action_stats),
            "world": reference["world"],
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "stable_worldmodel": "0.1.1",
                "git_revision": _git_revision(),
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                "compatibility": compatibility,
            },
        }
        manifest_path = output / "protocol_manifest.json"
        write_json_atomic(manifest_path, manifest)
        wc = reference["world"]
        world = None
        started = time.monotonic()
        try:
            world = swm.World(
                wc["env_name"],
                num_envs=50,
                image_shape=(wc["image_size"], wc["image_size"]),
                max_episode_steps=2 * offset,
                env_type=wc["env_type"],
                ob_type=wc["ob_type"],
                multiview=wc["multiview"],
                width=wc["image_size"],
                height=wc["image_size"],
                visualize_info=wc["visualize_info"],
                terminate_at_goal=wc["terminate_at_goal"],
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
            pairs = selection["pairs"]
            with torch.inference_mode():
                metrics = world.evaluate(
                    dataset=dataset,
                    episodes_idx=pairs["episode_indices"],
                    start_steps=pairs["start_steps"],
                    goal_offset=offset,
                    eval_budget=2 * offset,
                    callables=callables,
                    video=output / "videos" if video else None,
                )
            successes = np.asarray(metrics["episode_successes"], dtype=bool)
            if successes.shape != (50,):
                raise ValueError(
                    "Formal evaluation did not return all 50 episode outcomes."
                )
            rate = float(successes.mean() * 100)
            if not np.isclose(rate, metrics["success_rate"]):
                raise ValueError(
                    "Aggregate success rate disagrees with episode outcomes."
                )
            episodes = [
                dict(
                    index=i,
                    episode=pairs["episode_indices"][i],
                    start=pairs["start_steps"][i],
                    goal=pairs["goal_steps"][i],
                    success=bool(successes[i]),
                )
                for i in range(50)
            ]
            result = {
                "method": method,
                "protocol": f"O{offset}",
                "score_mode": score,
                "successes": int(successes.sum()),
                "episodes": 50,
                "success_rate": rate,
                "episode_results": episodes,
                "selection_sha256": manifest["selection_sha256"],
                "paired_protocol": manifest["paired_protocol"],
                "metrics": _jsonable(metrics),
                "elapsed_seconds": time.monotonic() - started,
                "formal": True,
            }
            write_json_atomic(output / "result.json", result)
            write_json_atomic(output / "episode_results.json", {"episodes": episodes})
            manifest["status"] = "complete"
            return result
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if world is not None:
                world.close()
            manifest["elapsed_seconds"] = time.monotonic() - started
            write_json_atomic(manifest_path, manifest)


def paired_eff_comparison(baseline: dict, method: dict) -> dict:
    """New/Lost refer to exactly the same pairs, never to old all-episode C tables."""
    if (
        baseline.get("method") != "F-only"
        or not baseline.get("formal")
        or not method.get("formal")
    ):
        raise ValueError(
            "Comparison requires completed formal results and F-only as baseline."
        )
    if (
        baseline["selection_sha256"] != method["selection_sha256"]
        or baseline["paired_protocol"] != method["paired_protocol"]
    ):
        raise ValueError(
            "Unpaired selection or planning/render protocol; comparison refused."
        )
    a, b = baseline["episode_results"], method["episode_results"]
    if len(a) != 50 or len(b) != 50:
        raise ValueError("Paired comparison requires all 50 episodes.")
    for first, second in zip(a, b, strict=True):
        if any(
            first[key] != second[key] for key in ("index", "episode", "start", "goal")
        ):
            raise ValueError("Episode identities/order differ.")
    new = [
        x["index"]
        for x, y in zip(a, b, strict=True)
        if not x["success"] and y["success"]
    ]
    lost = [
        x["index"]
        for x, y in zip(a, b, strict=True)
        if x["success"] and not y["success"]
    ]
    return {
        "method": method["method"],
        "protocol": method["protocol"],
        "f_successes": baseline["successes"],
        "method_successes": method["successes"],
        "new": len(new),
        "lost": len(lost),
        "new_indices": new,
        "lost_indices": lost,
        "net_percentage_points": 2 * (len(new) - len(lost)),
        "f_preserved_union_oracle_successes": baseline["successes"] + len(new),
        "union_is_deployable_policy": False,
    }
