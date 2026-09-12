"""Full Eff training loop over an audited, episode-disjoint frozen store.

All training choices are supplied explicitly. This is not a smoke-only loop:
the same loop executes every requested optimizer update and validation epoch,
and leaves an atomic checkpoint/manifest for recovery if interrupted.
"""

from __future__ import annotations

import dataclasses
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tdwm.methods.eff import EffModel
from tdwm.training.eff_data import EffEpisodeReplay
from tdwm.training.eff_runtime import EffTrainer, canonical_sha256


def write_json_atomic(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".eff-json-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def run_directory_lock(output_dir: Path):
    """A process lock protects this experiment only; no global/server lock."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".eff.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another process owns this Eff output directory."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class EffRunSettings:
    seed: int
    epochs: int
    updates_per_epoch: int
    batch_size: int
    g_hidden_dim: int
    v_hidden_dim: int
    learning_rate: float
    weight_decay: float
    gamma_g: float
    beta: float
    critic_coefficient: float
    ema_rate: float
    gradient_clip: float
    include_goal_boundary: bool
    backup_primitive_steps: int
    cross_episode_probability: float
    epsilon: float
    warmup_fraction: float
    validation_batches: int
    validation_seed: int
    checkpoint_every_updates: int

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "updates_per_epoch",
            "batch_size",
            "g_hidden_dim",
            "v_hidden_dim",
            "backup_primitive_steps",
            "validation_batches",
            "checkpoint_every_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        if not 0 <= self.cross_episode_probability <= 1:
            raise ValueError("cross_episode_probability must be in [0, 1].")
        if not isinstance(self.include_goal_boundary, bool):
            raise TypeError("include_goal_boundary must be explicit Boolean.")

    @property
    def total_updates(self) -> int:
        return self.epochs * self.updates_per_epoch

    def trainer_arguments(self) -> dict:
        names = (
            "seed",
            "learning_rate",
            "weight_decay",
            "gamma_g",
            "beta",
            "critic_coefficient",
            "ema_rate",
            "gradient_clip",
            "include_goal_boundary",
        )
        return {name: getattr(self, name) for name in names}

    def sample_arguments(self) -> dict:
        return dict(
            batch_size=self.batch_size,
            backup_primitive_steps=self.backup_primitive_steps,
            cross_episode_probability=self.cross_episode_probability,
            epsilon=self.epsilon,
        )


def scheduled_learning_rate(settings: EffRunSettings, update_index: int) -> float:
    """Linear warmup then cosine, indexed by accepted optimizer updates."""
    if not 0 <= update_index < settings.total_updates:
        raise ValueError("Learning-rate index lies outside the run.")
    warmup = math.ceil(settings.total_updates * settings.warmup_fraction)
    if update_index < warmup:
        return settings.learning_rate * (update_index + 1) / warmup
    remaining = settings.total_updates - warmup
    fraction = (update_index - warmup) / max(remaining - 1, 1)
    return settings.learning_rate * 0.5 * (1 + math.cos(math.pi * fraction))


def run_eff_training(
    *,
    replay: EffEpisodeReplay,
    validation_replay: EffEpisodeReplay,
    settings: EffRunSettings,
    source_identity: dict,
    output_dir: str | Path,
    device: str,
    resume: str | Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Run the full configured budget; stop requests are checked between steps.

    Evaluation episodes are never used for optimizer updates. Validation uses
    its own fixed RNG so it cannot change subsequent training samples.
    Unexpected failures preserve the last already committed checkpoint; no
    possibly half-updated optimizer state is advertised as recoverable.
    """
    if set(replay.episodes).intersection(validation_replay.episodes):
        raise ValueError("Training/validation episode leakage.")
    if replay.stride != validation_replay.stride:
        raise ValueError("Training/validation time units differ.")
    if settings.backup_primitive_steps % replay.stride:
        raise ValueError("Backup length must align with the frozen state stride.")
    output = Path(output_dir).expanduser().resolve()
    identity = {
        "source": source_identity,
        "settings_sha256": canonical_sha256(dataclasses.asdict(settings)),
        "training_episodes": replay.episodes.tolist(),
        "validation_episodes": validation_replay.episodes.tolist(),
        "state_stride": replay.stride,
    }
    with run_directory_lock(output):
        manifest_path = output / "training_manifest.json"
        existing = None
        if manifest_path.exists() and resume is None:
            raise FileExistsError("Eff output already exists; use an explicit resume.")
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing["identity"] != identity:
                raise ValueError("Output directory belongs to a different Eff run.")
        torch.manual_seed(settings.seed)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(settings.seed)
        trainer = EffTrainer(
            EffModel(
                g_hidden_dim=settings.g_hidden_dim, v_hidden_dim=settings.v_hidden_dim
            ),
            identity=identity,
            device=device,
            **settings.trainer_arguments(),
        )
        if resume is not None:
            trainer.resume(resume)
        if (
            existing is not None
            and existing["completed_updates"] != trainer.global_step
        ):
            raise ValueError(
                "Resume must use this directory's last checkpoint; use a new directory for older checkpoints."
            )
        if trainer.global_step > settings.total_updates:
            raise ValueError("Resume checkpoint exceeds the configured update budget.")
        manifest = {
            "format": "tdwm-eff-run-v1",
            "method": "Eff",
            "status": "running",
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "settings": dataclasses.asdict(settings),
            "total_updates": settings.total_updates,
            "completed_updates": trainer.global_step,
            "resume_source": None if resume is None else str(Path(resume).resolve()),
            "runtime": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "stable_worldmodel": importlib.metadata.version("stable-worldmodel"),
                "device": str(device),
            },
            "checkpoints": {} if existing is None else existing.get("checkpoints", {}),
        }
        if resume is not None:
            manifest["last_recoverable_checkpoint"] = str(Path(resume).resolve())
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
                os.link(last, snapshot)
                manifest["checkpoints"][str(epoch)] = {
                    "path": str(snapshot),
                    "sha256": checksum,
                    "global_step": trainer.global_step,
                }
            write_json_atomic(manifest_path, manifest)

        # Each invocation gets a separate log; prior runs are never truncated.
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output,
            prefix=f"metrics-from-{trainer.global_step:08d}-",
            suffix=".jsonl",
            delete=False,
        ) as log:
            manifest["metrics_path"] = log.name
            write_json_atomic(manifest_path, manifest)
            try:
                while trainer.global_step < settings.total_updates:
                    if should_stop is not None and should_stop():
                        save_checkpoint(
                            trainer.global_step // settings.updates_per_epoch,
                            completed_epoch=False,
                        )
                        manifest["status"] = "stopped"
                        break
                    batch = replay.sample(
                        rng=trainer.rng, **settings.sample_arguments()
                    )
                    metrics = trainer.step(
                        batch,
                        learning_rate=scheduled_learning_rate(
                            settings, trainer.global_step
                        ),
                    )
                    metrics["event"] = "training"
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    epoch, inside_epoch = divmod(
                        trainer.global_step, settings.updates_per_epoch
                    )
                    if inside_epoch == 0:
                        validation_rng = np.random.default_rng(settings.validation_seed)
                        values = [
                            trainer.validate(
                                validation_replay.sample(
                                    rng=validation_rng, **settings.sample_arguments()
                                )
                            )
                            for _ in range(settings.validation_batches)
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
                    elif trainer.global_step % settings.checkpoint_every_updates == 0:
                        save_checkpoint(epoch, completed_epoch=False)
                else:
                    manifest["status"] = "complete"
            except BaseException as exc:
                manifest["status"] = "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                manifest["accepted_updates_in_memory"] = trainer.global_step
                manifest["elapsed_seconds_this_invocation"] = time.monotonic() - started
                write_json_atomic(manifest_path, manifest)
    return manifest
