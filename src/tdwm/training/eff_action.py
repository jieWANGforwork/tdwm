"""Artifact binding and staged experiment runner for EffAction/Plan."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tdwm.methods.actor_free_td_lewm_v1 import validate_frozen_lewm_action_encoder_v1
from tdwm.training.eff_action_data import EffActionEpisodeData, EffActionSamplingConfig
from tdwm.training.frozen_actor_free_td import _resolve_local_pretrained_lewm_export
from tdwm.training.frozen_latent_store import FrozenLatentStore, file_sha256
from tdwm.training.gt_lewm_support import write_json


def load_eff_action_protocol(path: str | Path, *, smoke: bool) -> dict[str, Any]:
    with Path(path).open() as stream:
        protocol = yaml.safe_load(stream)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1 or protocol.get("method_family") != "eff_action":
        raise ValueError("Expected the version-one eff_action experiment protocol.")
    if not smoke and protocol.get("protocol_status") != "user_locked":
        raise ValueError("Formal training requires the unresolved method choices to be fixed in the protocol; this configuration is provisional.")
    if protocol["runtime"]["stable_worldmodel_version"] != "0.1.1":
        raise ValueError("This project requires stable-worldmodel 0.1.1.")
    for name in ("gv", "planner"):
        cfg = EffActionSamplingConfig(**protocol["sampling"][name])
        if (cfg.episode_start, cfg.episode_stop) != (0, 8000):
            raise ValueError("RP1 training is restricted to whole episodes 0..7999.")
    validation = protocol["validation"]
    if validation["episode_start"] != 8000 or validation["episode_stop"] != 10000:
        raise ValueError("Validation must use the declared held-out episode population.")
    if validation["checkpoint_selection"] != "fixed_budget_no_test_selection":
        raise ValueError("The runner reports fixed-budget checkpoints without selecting on test success.")
    if protocol["training"]["planner"]["raw_action_dim"] != 25:
        raise ValueError("The current method scores one 25D five-step action block.")
    if protocol["sampling"]["gv"]["efficiency_epsilon"] != protocol["training"]["planner"]["epsilon"]:
        raise ValueError("Efficiency epsilon must be consistent between sampling and action scoring.")
    return protocol


def load_frozen_eff_action_backbone(pretrained: str | Path, *, expected_sha256: str, device: str | torch.device):
    """Use the installed SWM public loader for the immutable LeWM export."""
    import stable_worldmodel as swm

    if importlib.metadata.version("stable-worldmodel") != "0.1.1":
        raise RuntimeError("The installed stable-worldmodel version differs from 0.1.1.")
    name, weights, cache = _resolve_local_pretrained_lewm_export(pretrained)
    actual = file_sha256(weights)
    if actual != expected_sha256:
        raise ValueError("Frozen LeWM checkpoint SHA256 differs from the experiment.")
    model = swm.wm.load_pretrained(name, cache_dir=str(cache))
    model.to(device).requires_grad_(False).eval()
    validate_frozen_lewm_action_encoder_v1(model.action_encoder)
    return model, {"checkpoint_sha256": actual, "checkpoint_path": str(weights), "source_name": name}


def load_eff_action_store(path: str | Path, *, protocol: dict[str, Any]) -> FrozenLatentStore:
    binding = protocol["frozen_data"]
    store = FrozenLatentStore(
        path,
        expected_checkpoint_sha256=protocol["pretrained_world_model"]["checkpoint_sha256"],
        expected_dataset_source_sha256=binding["dataset_source_sha256"],
        expected_column_normalization_sha256=binding["column_normalization_sha256"],
        expected_frame_skip=5, expected_history_frames=3, expected_embed_dim=192, expected_action_dim=5,
    )
    if store.manifest_sha256 != binding["manifest_sha256"]:
        raise ValueError("Frozen latent store manifest differs from the audited artifact.")
    if store.total_rows != binding["total_rows"] or int(store.episode_ids[-1]) + 1 != binding["episodes"]:
        raise ValueError("Frozen data population differs from the experiment.")
    metadata = store.manifest.get("source_metadata", {})
    image = metadata.get("image_preprocessing", {})
    for key in ("size", "mean", "std"):
        if image.get(key) != protocol["image_preprocessing"][key]:
            raise ValueError("Cache and deployment image preprocessing differ.")
    if metadata.get("stable_worldmodel_version") != "0.1.1":
        raise ValueError("The latent store was created with another framework version.")
    return store


def load_eff_action_normalization(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Action normalization must match the frozen training cache exactly.")
    content = json.loads(path.read_text())
    stats = content["action"] if "action" in content else content
    mean, scale = np.asarray(stats["mean"]), np.asarray(stats["scale"])
    if mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid five-dimensional action normalization statistics.")
    return stats


def _code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unavailable"


def load_eff_action_deployment(checkpoint: str | Path, *, pretrained: str | Path,
                               method: str, device: str = "cuda",
                               allow_intermediate: bool = False):
    """Load deployable heads; training recovery remains a separate contract."""
    from tdwm.methods.eff_action import EffActionSuccessor, EffActionValue
    from tdwm.methods.eff_action_plan import EffActionPlanner
    from tdwm.training.eff_action_runtime import validate_eff_action_training_config

    if method not in {"EffAction", "EffActionPlan"}:
        raise ValueError("Deployment method must be EffAction or EffActionPlan.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "eff_action_training" or payload.get("format_version") != 1:
        raise ValueError("Unsupported EffAction checkpoint format.")
    config = validate_eff_action_training_config(payload["config"])
    counters = payload["counters"]
    stages = ("gv", "stage1", "stage2")
    stage = payload["stage"]
    if stage not in stages or set(counters) != set(stages):
        raise ValueError("Checkpoint stage/counters are invalid.")
    for index, key in enumerate(stages):
        count = counters[key]
        if type(count) is not int or not 0 <= count <= config[key]["steps"]:
            raise ValueError("Invalid checkpoint update count.")
        if index < stages.index(stage) and count != config[key]["steps"]:
            raise ValueError("Checkpoint skipped an unfinished stage.")
        if index > stages.index(stage) and count != 0:
            raise ValueError("Checkpoint contains future-stage updates.")
    needed_stage = "gv" if method == "EffAction" else stage
    if method == "EffActionPlan" and (stage == "gv" or not payload.get("planner_trained") or counters[stage] == 0):
        raise ValueError("EffActionPlan requires a trained planner checkpoint.")
    complete = counters[needed_stage] == config[needed_stage]["steps"]
    if counters["gv"] <= 0 or (not complete and not allow_intermediate):
        raise ValueError("An incomplete checkpoint requires explicit intermediate evaluation.")
    provenance = payload["provenance"]
    world, source = load_frozen_eff_action_backbone(pretrained, expected_sha256=provenance["pretrained_checkpoint_sha256"], device=device)
    successor = EffActionSuccessor(**config["g"])
    value = EffActionValue(**config["v"])
    successor.load_state_dict(payload["models"]["g"], strict=True)
    value.load_state_dict(payload["models"]["v"], strict=True)
    planner = None
    if method == "EffActionPlan":
        planner = EffActionPlanner(**{key: config["planner"][key] for key in ("raw_action_dim", "hidden_dim", "hidden_layers")})
        planner.load_state_dict(payload["models"]["planner"], strict=True)
    for module in (successor, value, planner):
        if module is not None:
            module.to(device).requires_grad_(False).eval()
    metadata = {
        "path": str(Path(checkpoint).resolve()), "sha256": file_sha256(checkpoint),
        "method": method, "stage": stage, "counters": counters,
        "training_config": config, "training_provenance": provenance,
        "stage_budget_complete": complete, "intermediate_evaluation_allowed": allow_intermediate,
        "planner_objective": "reconstruction_plus_efficiency" if stage == "stage2" else "reconstruction_only" if stage == "stage1" else None,
        "pretrained_world_model": source,
        "frozen_latent_store": {"manifest_sha256": provenance["frozen_latent_manifest_sha256"],
                                "extraction_precision": "bfloat16", "storage_dtype": "float32"},
    }
    return world, successor, value, planner, metadata


def _validation_data(store: FrozenLatentStore, protocol: dict[str, Any], stage: str) -> EffActionEpisodeData:
    values = dict(protocol["sampling"]["gv" if stage == "gv" else "planner"])
    values.update({key: protocol["validation"][key] for key in ("episode_start", "episode_stop")})
    return EffActionEpisodeData.from_store(store, config=EffActionSamplingConfig(**values))


def restore_eff_action_metrics(path: Path, counters: dict[str, int]) -> Path | None:
    """Keep an archived copy and rewind the active curve to the checkpoint."""
    if not path.exists():
        return None
    lines = path.read_text().splitlines(keepends=True)
    retained = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break  # A process can die during its last write; the archive retains it.
            raise
        if record["stage"] not in counters:
            raise ValueError("Metrics contain an unknown training stage.")
        if record["step"] <= counters[record["stage"]]:
            retained.append(line if line.endswith("\n") else line + "\n")
    backup = path.with_name(f"metrics_before_resume_{time.time_ns()}.jsonl")
    shutil.copyfile(path, backup)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(retained))
    temporary.replace(path)
    return backup


def _record_validation(trainer, data, *, run_dir, stage, step, batch_size, batches, seed, device):
    validation_rng = np.random.default_rng(seed)
    measured, pairs = [], []
    for _ in range(batches):
        batch = data.sample(batch_size, rng=validation_rng, device=device)
        measured.append(trainer.evaluate_batch(batch))
        pairs.extend(zip(batch["anchor_row"].cpu().tolist(), batch["goal_row"].cpu().tolist()))
    pair_path = run_dir / f"{stage}_validation_pairs.json"
    write_json(pair_path, {"anchor_goal_global_rows": pairs, "seed": seed})
    values = {key: float(np.mean([item[key] for item in measured])) for key in measured[0]
              if isinstance(measured[0][key], (int, float)) and key not in {"step", "global_step"}}
    write_json(run_dir / f"{stage}_validation_{step:06d}.json", {
        "stage": stage, "step": step, "metrics": values, "seed": seed, "batches": batches,
        "pair_manifest_sha256": file_sha256(pair_path), "checkpoint_selection": "fixed_budget_no_test_selection",
    })


def train_eff_action(*, config_path: str | Path, latent_store: str | Path, pretrained: str | Path,
                     output_dir: str | Path, action_normalization: str | Path | None = None,
                     device: str = "cuda", smoke: bool = False, resume: str | Path | None = None,
                     stop_after: str = "stage2") -> dict[str, Any]:
    """Run G/V, reconstruction P, then efficiency P with fixed stage budgets.

    All paths are injected by the caller. Results and checkpoints are written
    only to this run's output directory. This function never allocates a cloud
    instance, changes GPU visibility, kills another process, or shuts down.
    """
    from tdwm.training.eff_action_runtime import EffActionTrainer

    protocol = load_eff_action_protocol(config_path, smoke=smoke)
    if stop_after not in {"gv", "stage1", "stage2"}:
        raise ValueError("stop_after must be gv, stage1, or stage2.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; no training was started.")
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume and any(run_dir.iterdir()):
        raise FileExistsError("A new run needs an empty output directory; use --resume for an existing run.")
    train_config = copy.deepcopy(protocol["training"])
    batch_size = int(protocol["loader"]["batch_size"])
    if smoke:
        for stage in ("gv", "stage1", "stage2"):
            train_config[stage]["steps"] = 3 if stage == "gv" else 2
            train_config[stage]["warmup_steps"] = 1
        batch_size = min(batch_size, 4)
    store = load_eff_action_store(latent_store, protocol=protocol)
    normalization_path = action_normalization or store.manifest["source_metadata"]["column_normalization_path"]
    normalization = load_eff_action_normalization(normalization_path, expected_sha256=store.manifest["column_normalization_sha256"])
    model, source = load_frozen_eff_action_backbone(pretrained, expected_sha256=protocol["pretrained_world_model"]["checkpoint_sha256"], device=device)
    provenance = {
        "protocol_sha256": file_sha256(config_path), "run_mode": "smoke" if smoke else "formal",
        "frozen_latent_manifest_sha256": store.manifest_sha256,
        "pretrained_checkpoint_sha256": source["checkpoint_sha256"],
        "column_normalization_sha256": store.manifest["column_normalization_sha256"],
        "dataset_source_sha256": store.manifest["dataset_source_sha256"],
        "sampling": protocol["sampling"], "stable_worldmodel_version": "0.1.1",
        "torch_version": str(torch.__version__),
        "implementation_files": {
            name: file_sha256(Path(__file__).parents[1] / name)
            for name in ("training/eff_action.py", "training/eff_action_runtime.py", "training/eff_action_data.py", "methods/eff_action.py", "methods/eff_action_plan.py", "methods/actor_free_td_lewm_v1.py", "methods/actor_free_td_lewm_v2.py")
        },
    }
    if resume:
        trainer = EffActionTrainer.load_checkpoint(resume, action_encoder=model.action_encoder, config=train_config, provenance=provenance, device=device)
    else:
        trainer = EffActionTrainer(model.action_encoder, config=train_config, provenance=provenance, device=device)
    if ("gv", "stage1", "stage2").index(stop_after) < ("gv", "stage1", "stage2").index(trainer.stage):
        raise ValueError("Requested stop_after is earlier than the resumed checkpoint stage.")
    if resume:
        restore_eff_action_metrics(run_dir / "metrics.jsonl", trainer.counters)
    manifest = {
        "protocol": protocol, "effective_training": train_config, "effective_batch_size": batch_size,
        "provenance": provenance, "pretrained": source,
        "normalization": normalization,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": device,
                    "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
                    "git_revision": _code_revision(), "pid": os.getpid()},
    }
    write_json(run_dir / "training_manifest.json", manifest)
    write_json(run_dir / "action_normalization.json", normalization)
    normalization_copy = run_dir / "column_normalization.json"
    if Path(normalization_path).resolve() != normalization_copy:
        shutil.copyfile(normalization_path, normalization_copy)
    started = time.monotonic()
    save_every = int(protocol["logging"]["checkpoint_every_steps"])
    log_every = int(protocol["logging"]["metrics_every_steps"])
    if save_every <= 0 or log_every <= 0 or batch_size <= 0:
        raise ValueError("Batch and output intervals must be positive.")
    previous_stage = None
    pending_metrics: list[dict[str, Any]] = []
    with (run_dir / "metrics.jsonl").open("a") as log:
        while True:
            stage = trainer.stage
            if stage != previous_stage:
                train_data = EffActionEpisodeData.from_store(store, config=EffActionSamplingConfig(**protocol["sampling"]["gv" if stage == "gv" else "planner"]))
                validation_data = _validation_data(store, protocol, stage)
                write_json(run_dir / f"{stage}_sampling.json", {"training": train_data.metadata(), "validation": validation_data.metadata()})
                previous_stage = stage
            total_steps = train_config[stage]["steps"]
            current_step = trainer.counters[stage]
            # Recover a crash between the checkpoint write and validation output.
            if current_step and (current_step % save_every == 0 or current_step == total_steps) and not (run_dir / f"{stage}_validation_{current_step:06d}.json").exists():
                _record_validation(trainer, validation_data, run_dir=run_dir, stage=stage,
                                   step=current_step, batch_size=batch_size,
                                   batches=1 if smoke else int(protocol["validation"]["batches"]),
                                   seed=protocol["validation"]["seed"], device=device)
            if trainer.counters[stage] >= total_steps:
                trainer.save_checkpoint(run_dir / f"{stage}_complete.pt")
                if stage == stop_after or stage == "stage2":
                    break
                trainer.begin_planner_stage(1 if stage == "gv" else 2)
                trainer.save_checkpoint(run_dir / "latest.pt")
                continue
            batch = train_data.sample(batch_size, rng=trainer.sample_rng, device=device)
            metrics = trainer.train_step(batch)
            for key in ("efficiency", "net_distance", "path_length", "goal_chunks"):
                values = batch[key][batch["valid_mask"]].float()
                metrics[f"{key}_mean"] = float(values.mean())
                metrics[f"{key}_min"] = float(values.min())
                metrics[f"{key}_max"] = float(values.max())
            pending_metrics.append(metrics)
            step = trainer.counters[stage]
            if step % log_every == 0 or step % save_every == 0 or step == total_steps or smoke:
                means = {key: float(np.mean([item[key] for item in pending_metrics])) for key in metrics if isinstance(metrics[key], (int, float)) and key not in {"step", "global_step"}}
                record = {"stage": stage, "step": step, "window_start_step": step - len(pending_metrics) + 1,
                          "window_updates": len(pending_metrics), "metrics": means,
                          "elapsed_seconds_this_invocation": time.monotonic() - started}
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                pending_metrics.clear()
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % save_every == 0 or step == total_steps:
                trainer.save_checkpoint(run_dir / "latest.pt")
                trainer.save_checkpoint(run_dir / f"{stage}_step_{step:06d}.pt")
                _record_validation(trainer, validation_data, run_dir=run_dir, stage=stage,
                                   step=step, batch_size=batch_size,
                                   batches=1 if smoke else int(protocol["validation"]["batches"]),
                                   seed=protocol["validation"]["seed"], device=device)
    trainer.save_checkpoint(run_dir / "latest.pt")
    result = {
        "method_family": "eff_action", "stage_completed": trainer.stage,
        "counters": trainer.counters, "smoke": smoke,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "checkpoint": str(run_dir / f"{trainer.stage}_complete.pt"),
        "checkpoint_sha256": file_sha256(run_dir / f"{trainer.stage}_complete.pt"),
        "provenance": provenance,
    }
    write_json(run_dir / "training_result.json", result)
    return result
