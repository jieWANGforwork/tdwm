"""Paired formal O25/O50/O100 evaluation of F-only, Eff and EffPlan."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from tdwm.adapters.effplan import (
    EFF_CUMULATIVE_SCORE_MODES,
    EFF_SCORE_MODES,
    EffCEMCost,
    EffPlanSolver,
    EffPlanTrackingCost,
)
from tdwm.adapters.runtime import prepare_cloud_runtime
from tdwm.evaluation.frozen_actor_free_td_common import _resolve_frozen_dataset_source
from tdwm.evaluation.lewm_checkpoint import _git_revision, _jsonable
from tdwm.evaluation.mc_gt_lewm import _load_action_processor
from tdwm.methods.effplan_safety import PlannerSafety
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


def validated_planner_safety(checkpoint_settings: dict, config: dict):
    safety = checkpoint_settings.get("safety")
    if safety != config["evaluation"].get("effplan_safety") or safety != config[
        "planner_refinement"
    ]["settings"].get("safety"):
        raise ValueError("EffPlan checkpoint/training/evaluation safety differs.")
    return None if safety is None else PlannerSafety(**safety)


def prepare_eff_selections(
    *, config_path: str | Path, terminal_metadata: str | Path, output_dir: str | Path,
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
        expected_v_parameterization=settings.v_parameterization,
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
    eff_score: str | None = None,
    cumulative_weight: float | None = None,
    adaptive_one_shot: bool = False,
    adaptive_rolling: bool = False,
) -> dict:
    """Full public SWM world.evaluate call; no reduced/smoke score substituted."""
    config = load_eff_protocol(config_path, stage="evaluation")
    if method not in {"F-only", "Eff", "EffPlan"}:
        raise ValueError("Unknown predeclared method.")
    if adaptive_one_shot and adaptive_rolling:
        raise ValueError("Choose either one-shot or rolling adaptive evaluation, not both.")
    if (adaptive_one_shot or adaptive_rolling) and method != "EffPlan":
        raise ValueError("Adaptive one-shot is an independent EffPlan evaluation mode.")
    ev = config["evaluation"]
    # A predeclared sweep over Eff scoring modes, not post-hoc tuning: every
    # override is recorded verbatim in the manifest next to the locked config.
    defaults = {
        "eff_score": None,
        "eff_cumulative_weight": 1.0,
    }
    overrides: dict[str, object] = {}
    for name, value in (
        ("eff_score", eff_score),
        ("eff_cumulative_weight", cumulative_weight),
    ):
        if value is None:
            continue
        if method != "Eff":
            raise ValueError(f"{name} only applies to the Eff method.")
        overrides[name] = {"from": ev.get(name, defaults[name]), "to": value}
        ev[name] = value
    if eff_score in EFF_CUMULATIVE_SCORE_MODES and ev["eff_cumulative_weight"] == 0:
        raise ValueError(
            "A cumulative Eff score with zero weight is just the terminal "
            "score; run terminal_eff_cost so the manifest stays unambiguous."
        )
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
    if type(receding) is not int or receding != 5:
        raise ValueError(
            "Eff/EffPlan must execute all five blocks (25 primitive steps) "
            "before replanning, independently of the goal offset."
        )
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
        planner_safety = validated_planner_safety(ppayload["settings"], config)
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
        execution_limits = None
        if adaptive_one_shot or adaptive_rolling:
            from tdwm.adapters.effplan_adaptive import (
                AdaptiveEffPlanSolver, AdaptiveTrackingCost, PlanExecutionLimits,
            )
            if planner_safety is None:
                raise ValueError("Adaptive mode requires the stable P safeguards.")
            execution_limits = PlanExecutionLimits(2 * offset)
            model = AdaptiveTrackingCost(world_model, eff, target=ev["target_readout"])
            solver = AdaptiveEffPlanSolver(
                model=model, planner=planner, limits=execution_limits,
                safety=planner_safety, budget=2*offset, device=device,
                search_iterations=tuple(ev["effplan_search_iterations"]),
                epsilon=ev["epsilon"], minimum_relative_gain=1e-6,
                dynamics_coefficient=ev["effplan_dynamics_coefficient"],
            )
            score = "adaptive_work_gain_one_shot_v1"
            extra_rerolls = len(ev["effplan_search_iterations"])-1
            overrides["adaptive_one_shot"] = dict(
                minimum_relative_gain=1e-6, subdivision="breadth_first",
                max_action_blocks=2*offset//5, root_forced_split=False,
                action_blocks="intermediate_nodes_plus_one",
                exhaustion="truncate_failure_without_replanning",
                planning_calls_per_episode=1, reuse_checkpoint_without_training=True,
                note="Variable-horizon open-loop protocol, not matched-compute H5/RH5.",
            )
            if adaptive_rolling:
                score = "adaptive_work_gain_rolling_v1"
                settings = overrides.pop("adaptive_one_shot")
                settings.update(
                    exhaustion="replan_from_real_observation_until_total_budget",
                    planning_calls_per_episode="adaptive",
                    max_action_blocks="remaining_primitive_budget_div_5",
                    note="Adaptive execution windows; same total environment budget, not matched compute.",
                )
                overrides["adaptive_rolling"] = settings
        elif method == "EffPlan":
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
                safety=planner_safety,
            )
            score = "state_path_tracking"
            extra_rerolls = len(allocation) - 1
        else:
            if method == "Eff" and ev["eff_score"] not in EFF_SCORE_MODES:
                raise ValueError(
                    f"Unsupported Eff score {ev['eff_score']!r}; expected one of "
                    f"{sorted(EFF_SCORE_MODES)}."
                )
            model = (
                world_model
                if method == "F-only"
                else EffCEMCost(
                    world_model,
                    eff,
                    target=ev["target_readout"],
                    score_mode=ev["eff_score"],
                    cumulative_weight=ev.get("eff_cumulative_weight", 1.0),
                )
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
                else ev["eff_score"]
            )
            extra_rerolls = 0
        plan = swm.PlanConfig(
            horizon=2*offset//5 if adaptive_one_shot else 5,
            receding_horizon=2*offset//5 if adaptive_one_shot else receding,
            history_len=1,
            action_block=5,
            warm_start=not adaptive_one_shot,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan,
            process={"action": processor},
            transform={"pixels": transform, "goal": transform},
        )
        if adaptive_rolling:
            from tdwm.adapters.effplan_adaptive_rolling import AdaptiveRollingPolicy
            policy = AdaptiveRollingPolicy(
                model=model, planner=planner, safety=planner_safety, budget=2*offset,
                device=device, search_iterations=tuple(ev["effplan_search_iterations"]),
                epsilon=ev["epsilon"], dynamics_coefficient=ev["effplan_dynamics_coefficient"],
                process={"action": processor}, transform={"pixels": transform, "goal": transform},
            )
        manifest = {
            "format": "tdwm-eff-formal-evaluation-v1",
            "status": "running",
            "method": method,
            "score_mode": score,
            "protocol": f"O{offset}",
            "config": config,
            "protocol_overrides": overrides,
            "eff_cumulative": {
                "weight": ev.get("eff_cumulative_weight", 1.0),
            },
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
        if adaptive_one_shot:
            manifest["paired_protocol"].update(
                horizon="adaptive", receding_horizon="no_replanning",
                max_action_blocks=2*offset//5, warm_start=False,
                execution="one_shot_until_success_or_plan_exhaustion",
            )
            manifest["compute"]["variable_horizon"] = True
        if adaptive_rolling:
            manifest["paired_protocol"].update(
                horizon="adaptive_remaining_budget", receding_horizon="all_adaptive_blocks",
                max_action_blocks=2*offset//5, warm_start=False,
                execution="replan_from_real_observation_until_success_or_total_budget",
            )
            manifest["compute"]["variable_horizon"] = True
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
                **({"pre_wrappers": [execution_limits.wrap]} if adaptive_one_shot else {}),
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
            if adaptive_one_shot:
                if solver.solve_calls != 1 or len(solver.records) != 50:
                    raise RuntimeError("One-shot evaluation did not plan exactly once per pair.")
                for i, wrapper in enumerate(execution_limits.wrappers):
                    record = solver.records[i]
                    assert wrapper.steps <= wrapper.limit <= 2*offset
                    assert wrapper.limit == 5*(record["intermediate_nodes"]+1)
                    record["executed_primitive_steps"] = wrapper.steps
                    record["success"] = bool(successes[i])
                    episodes[i].update(
                        action_blocks=record["action_blocks"],
                        intermediate_nodes=record["intermediate_nodes"],
                        planned_primitive_steps=wrapper.limit,
                        executed_primitive_steps=wrapper.steps,
                        planning_calls=1,
                    )
                write_json_atomic(output / "adaptive_planning.json", {"episodes": solver.records})
                torch.save(solver.artifacts, output / "adaptive_plans.pt")
                manifest["compute"]["per_episode"] = solver.records
            if adaptive_rolling:
                if len(policy.records) != 50:
                    raise RuntimeError("Adaptive rolling evaluation must cover every fixed pair.")
                for i, rounds in enumerate(policy.records):
                    executed = int(policy.executed_steps[i])
                    if not rounds or executed > 2*offset:
                        raise RuntimeError("Missing decisions or exceeded cumulative budget.")
                    if not successes[i] and executed != 2*offset:
                        raise RuntimeError("Unsuccessful episode stopped before its total budget.")
                    position = 0
                    for r in rounds:
                        assert r["start_primitive_step"] == position
                        assert r["remaining_budget_before"] == 2*offset-position
                        assert r["planned_primitive_steps"] <= r["remaining_budget_before"]
                        assert r["planned_primitive_steps"] == 5*(r["intermediate_nodes"]+1)
                        position += r["executed_primitive_steps"]
                    assert position == executed
                    episodes[i].update(
                        executed_primitive_steps=executed, planning_calls=len(rounds),
                        action_blocks_per_decision=[r["action_blocks"] for r in rounds],
                    )
                write_json_atomic(output / "adaptive_planning.json", {"episodes": policy.records})
                torch.save(policy.artifacts, output / "adaptive_plans.pt")
                manifest["compute"]["per_episode"] = policy.records
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
