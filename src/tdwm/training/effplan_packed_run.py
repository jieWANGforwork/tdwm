"""Independent packed-compute variable-length P run. Fixed-H5 files/entrypoints stay unchanged.

This runner retains the existing two-stage checkpoint/schedule protocol, but
binds an IID V-style full-path sampler and a variable-horizon trainer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tdwm.adapters.effplan_adaptive import AdaptiveTrackingCost
from tdwm.methods.effplan import StatePlanner
from tdwm.training.effplan_variable_data import sample_variable_planner_paths
from tdwm.training.effplan_packed_runtime import PackedEffPlanTrainer
from tdwm.training.eff_protocol import load_eff_protocol, load_eff_replays
from tdwm.training.eff_run import (
    EffRunSettings,
    eff_settings_payload,
    run_directory_lock,
    write_json_atomic,
)
from tdwm.training.eff_runtime import canonical_sha256, load_eff_model
from tdwm.training.effplan_runtime import (
    EffPlanTrainSettings,
    planner_settings_payload,
)

MANIFEST_FORMAT = "tdwm-effplan-run-v1"
SAMPLING = {
    "mode": "v_same_episode_full_path",
    "length_unit": "states",
    "minimum_states": 2,
    "maximum_states": "episode_length_at_stride",
    "macro_actions": "L-1",
    "interior_states": "L-2",
    "one_edge_path": "no_P_output_no_gradient",
    "aggregation": "IID_paths_bucketed_by_length_equal_path_weight",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def planner_learning_rate(
    settings: EffPlanTrainSettings, total_updates: int, update_index: int
) -> float:
    """Linear warmup then cosine; the planner has no separate schedule knob."""
    if not 0 <= update_index < total_updates:
        raise ValueError("Learning-rate index lies outside the run.")
    warmup_fraction = 0.01
    warmup = math.ceil(total_updates * warmup_fraction)
    if update_index < warmup:
        return settings.learning_rate * (update_index + 1) / warmup
    remaining = total_updates - warmup
    fraction = (update_index - warmup) / max(remaining - 1, 1)
    return settings.learning_rate * 0.5 * (1 + math.cos(math.pi * fraction))


def _planner_settings(stage_config: dict) -> EffPlanTrainSettings:
    settings = dict(stage_config["settings"])
    iterations = settings.get("search_iterations") or ()
    settings["search_iterations"] = tuple(int(x) for x in iterations)
    return EffPlanTrainSettings(**settings)


def load_frozen_world_model(lewm_checkpoint: str | Path, device: str, cache_dir: Path):
    """Load the public pretrained LeWM export used as the frozen F."""
    from tdwm.training.frozen_actor_free_td import _resolve_local_pretrained_lewm_export

    name, checkpoint_file, cache = _resolve_local_pretrained_lewm_export(
        lewm_checkpoint
    )
    import stable_worldmodel as swm

    world = (
        swm.wm.load_pretrained(name, cache_dir=str(cache or cache_dir))
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    return world, checkpoint_file


def run_packed_effplan_training(
    *,
    config_path: str | Path,
    phase: str,
    latent_store: str | Path,
    terminal_metadata: str | Path,
    eff_checkpoint: str | Path,
    eff_manifest: str | Path,
    lewm_checkpoint: str | Path | None,
    output_dir: str | Path,
    device: str,
    resume: str | Path | None = None,
    init_from: str | Path | None = None,
    stop_after_updates: int | None = None,
    branch_from: str | Path | None = None,
) -> dict:
    """Train one planner phase. ``init_from`` carries phase-one weights over.

    Phase two continues from the phase-one planner weights; its optimizer is
    rebuilt, so the handoff is weights-only and is recorded in the manifest.
    """
    stage = f"planner_{phase}"
    config = load_eff_protocol(config_path, stage=stage)
    if importlib.metadata.version("stable-worldmodel") != "0.1.1":
        raise RuntimeError("Wrong stable-worldmodel version.")
    if config.get("planner_sampling") != SAMPLING:
        raise ValueError("Explicit variable-P sampling configuration required.")
    if config["eff_training"]["settings"]["cross_episode_probability"] != 0:
        raise ValueError("Variable P must pair with the same-episode V protocol.")
    stage_config = config[stage]
    run = stage_config["run"]
    settings = _planner_settings(stage_config)
    if settings.phase != phase:
        raise ValueError("Configuration phase does not match the requested phase.")
    total_updates = int(run["epochs"]) * int(run["updates_per_epoch"])
    if total_updates < 1:
        raise ValueError("Planner run must declare a positive update budget.")
    if stop_after_updates is not None and (
        isinstance(stop_after_updates, bool)
        or not isinstance(stop_after_updates, int)
        or not 1 <= stop_after_updates <= total_updates
    ):
        raise ValueError(
            "stop_after_updates must be an absolute update within the full budget."
        )
    stop_at = total_updates if stop_after_updates is None else stop_after_updates
    if sum(x is not None for x in (resume, init_from, branch_from)) > 1:
        raise ValueError("Choose resume, initialization or schedule branch, not multiple.")

    eff_payload = json.loads(Path(eff_manifest).read_text())
    if eff_payload.get("status") != "complete":
        raise ValueError("Eff training must complete before planner training.")
    eff_settings = EffRunSettings(**config["eff_training"]["settings"])
    if eff_payload.get("completed_updates") != eff_settings.total_updates:
        raise ValueError("Eff checkpoint does not match the configured budget.")
    if eff_payload["identity"]["settings_sha256"] != canonical_sha256(
        eff_settings_payload(eff_settings)
    ):
        raise ValueError("Eff training settings differ from this configuration.")

    train_replay, validation_replay, source_identity = load_eff_replays(
        config=config,
        latent_store=latent_store,
        terminal_metadata=terminal_metadata,
    )
    if source_identity != eff_payload["identity"]["source"]:
        raise ValueError("P latent data/terminal metadata differs from its frozen V source.")
    eff, _ = load_eff_model(
        eff_checkpoint,
        expected_identity=eff_payload["identity"],
        expected_global_step=eff_settings.total_updates,
        device=device,
        expected_v_parameterization=eff_settings.v_parameterization,
    )
    identity = {
        "source": dict(
            source_identity,
            eff_checkpoint_sha256=sha256_file(eff_checkpoint),
            lewm_checkpoint_sha256=config["source"]["lewm_checkpoint_sha256"],
        ),
        "settings_sha256": canonical_sha256(planner_settings_payload(settings)),
        "phase": phase,
        "planner_hidden_dim": int(run["planner_hidden_dim"]),
        "training_episodes": train_replay.episodes.tolist(),
        "validation_episodes": validation_replay.episodes.tolist(),
        "state_stride": train_replay.stride,
        "planner_sampling": SAMPLING,
    }

    tracking_model = None
    if phase == "refinement":
        if lewm_checkpoint is None:
            raise ValueError("Refinement requires the frozen LeWM checkpoint.")
        if sha256_file(lewm_checkpoint) != config["source"]["lewm_checkpoint_sha256"]:
            raise ValueError("Frozen F checkpoint hash differs.")
        world, _ = load_frozen_world_model(lewm_checkpoint, device, Path(output_dir))
        tracking_model = AdaptiveTrackingCost(
            world, eff, target=settings.target_readout
        )

    output = Path(output_dir).expanduser().resolve()
    with run_directory_lock(output):
        manifest_path = output / "training_manifest.json"
        existing = None
        if branch_from is not None and manifest_path.exists():
            raise FileExistsError("A schedule branch needs a new output directory.")
        if manifest_path.exists() and resume is None and init_from is None:
            raise FileExistsError(
                "EffPlan output already exists; use explicit resume or init-from."
            )
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing["identity"] != identity:
                raise ValueError("Output directory belongs to a different EffPlan run.")
        # Seed construction in BOTH phases, independently of P safety. Resume
        # subsequently restores RNG/weights; init_from still replaces weights.
        torch.manual_seed(settings.seed)
        planner = StatePlanner(hidden_dim=int(run["planner_hidden_dim"]))
        trainer = PackedEffPlanTrainer(
            planner=planner,
            eff=eff,
            settings=settings,
            identity=identity,
            device=device,
            tracking_model=tracking_model,
        )
        if init_from is not None:
            weights = torch.load(init_from, map_location="cpu", weights_only=False)
            if (phase != "refinement"
                or weights.get("identity", {}).get("phase") != "generation"
                or weights["identity"].get("planner_sampling") != SAMPLING
                or weights["identity"]["source"] != identity["source"]):
                raise ValueError("Variable P refinement needs its paired generation checkpoint.")
            if weights.get("planner_hidden_dim") != planner.hidden_dim:
                raise ValueError("Phase handoff planner width differs.")
            planner.load_state_dict(weights["planner"], strict=True)
        if resume is not None:
            trainer.resume(resume)
        if branch_from is not None:
            trainer.resume(branch_from, schedule_branch=True)
        if trainer.global_step > stop_at:
            raise ValueError("Requested stop precedes the restored checkpoint.")
        if (
            existing is not None
            and existing["completed_updates"] != trainer.global_step
        ):
            raise ValueError(
                "Resume must use this directory's last checkpoint; "
                "use a new directory for older checkpoints."
            )
        manifest = {
            "format": MANIFEST_FORMAT,
            "method": "EffPlan",
            "phase": phase,
            "status": "running",
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "settings": planner_settings_payload(settings),
            "run": run,
            "planner_sampling": SAMPLING,
            "total_updates": total_updates,
            "completed_updates": trainer.global_step,
            "init_from": None if init_from is None else str(Path(init_from).resolve()),
            "resume_source": None if resume is None else str(Path(resume).resolve()),
            "schedule_transition": getattr(trainer, "schedule_transition", None),
            "stop_after_updates_this_invocation": stop_after_updates,
            "runtime": {
                "execution_backend": "packed_cross_length_v1",
                "checkpoint_compatibility": "preserve_optimizer_step_rng_and_training_identity",
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True,
                ).strip(),
                "config_sha256": sha256_file(config_path),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "stable_worldmodel": importlib.metadata.version("stable-worldmodel"),
                "device": str(device),
            },
            "checkpoints": {} if existing is None else existing.get("checkpoints", {}),
        }
        write_json_atomic(manifest_path, manifest)
        started = time.monotonic()

        def save_checkpoint(epoch: int, *, completed_epoch: bool) -> None:
            last = output / "last.pt"
            checksum = trainer.save(last, epoch=epoch)
            manifest["last_recoverable_checkpoint"] = str(last)
            manifest["last_checkpoint_sha256"] = checksum
            manifest["completed_updates"] = trainer.global_step
            if completed_epoch:
                snapshot = output / f"epoch_{epoch:04d}.pt"
                if snapshot.exists():
                    raise FileExistsError(
                        f"Refusing to replace existing {snapshot.name}."
                    )
                snapshot.write_bytes(last.read_bytes())
                manifest["checkpoints"][str(epoch)] = {
                    "path": str(snapshot),
                    "sha256": checksum,
                    "global_step": trainer.global_step,
                }
            write_json_atomic(manifest_path, manifest)

        def draw(replay, rng):
            return sample_variable_planner_paths(
                replay,
                batch_size=int(run["batch_size"]),
                rng=rng,
                backup_primitive_steps=config["eff_training"]["settings"]["backup_primitive_steps"],
                epsilon=settings.epsilon,
            )

        updates_per_epoch = int(run["updates_per_epoch"])
        if branch_from is not None:
            save_checkpoint(trainer.global_step // updates_per_epoch, completed_epoch=False)
        with (output / f"metrics-{trainer.global_step:08d}.jsonl").open("a") as log:
            manifest["metrics_path"] = str(log.name)
            write_json_atomic(manifest_path, manifest)
            try:
                while trainer.global_step < stop_at:
                    if (output / "STOP_REQUESTED").exists():
                        save_checkpoint(trainer.global_step // updates_per_epoch, completed_epoch=False)
                        manifest["status"] = "paused"
                        break
                    # Legacy runs used constant LR despite defining this schedule.
                    # The isolated safety-v1 run uses the declared full-budget
                    # warmup/cosine schedule, independent of a diagnostic pause.
                    if settings.safety is not None:
                        lr = planner_learning_rate(
                            settings, total_updates, trainer.global_step
                        )
                        for group in trainer.optimizer.param_groups:
                            group["lr"] = lr
                    metrics = trainer.step(draw(train_replay, trainer.rng),)
                    metrics["event"] = "training"
                    metrics["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    epoch, inside_epoch = divmod(trainer.global_step, updates_per_epoch)
                    if inside_epoch == 0:
                        validation_rng = np.random.default_rng(run["validation_seed"])
                        values = [
                            trainer.validate(draw(validation_replay, validation_rng))
                            for _ in range(int(run["validation_batches"]))
                        ]
                        validation = {
                            key: float(np.mean([v[key] for v in values]))
                            for key in values[0]
                        }
                        log.write(
                            json.dumps(
                                dict(
                                    validation,
                                    event="validation",
                                    epoch=epoch,
                                    global_step=trainer.global_step,
                                ),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        log.flush()
                        save_checkpoint(epoch, completed_epoch=True)
                    elif trainer.global_step % int(
                        run["checkpoint_every_updates"]
                    ) == 0 or (
                        settings.safety is not None
                        and trainer.global_step in (1, 5, 10, 25, 50, 100)
                    ):
                        save_checkpoint(epoch, completed_epoch=False)
                else:
                    if trainer.global_step < total_updates:
                        save_checkpoint(
                            trainer.global_step // updates_per_epoch,
                            completed_epoch=False,
                        )
                        manifest["status"] = "paused"
                    else:
                        manifest["status"] = "complete"
                        write_json_atomic(
                            output / "planner_manifest.json",
                            {
                                "format": MANIFEST_FORMAT,
                                "method": "EffPlan",
                                "phase": phase,
                                "status": "complete",
                                "completed_updates": trainer.global_step,
                                "identity": identity,
                                "identity_sha256": canonical_sha256(identity),
                                "settings": planner_settings_payload(settings),
                                "checkpoint": manifest["last_recoverable_checkpoint"],
                            },
                        )
            except BaseException as exc:
                manifest["status"] = "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                manifest["accepted_updates_in_memory"] = trainer.global_step
                manifest["elapsed_seconds_this_invocation"] = time.monotonic() - started
                write_json_atomic(manifest_path, manifest)
    return manifest
