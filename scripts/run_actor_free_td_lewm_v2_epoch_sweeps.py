#!/usr/bin/env python3
"""Roll intermediate V2 checkpoints through the complete Cube O50 sweep.

This scheduler deliberately excludes epoch 10, which remains owned by the
audited 18-cell formal launcher.  It watches epochs 3--9, verifies that each
checkpoint stopped changing, and runs six variants by three score modes on up
to two evaluation processes per selected GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

VARIANTS = ("g1", "g2", "g3", "d", "f", "c")
SCORE_MODES = ("f_only", "g_only", "f_plus_g")
EPOCHS = tuple(range(3, 10))
SEED = 3072
STEPS_PER_EPOCH = 12_796
REQUIRED_OUTPUT_FILES = (
    "results.json",
    "protocol_manifest.json",
    "episode_selection.json",
    "action_normalization.json",
)
EXIT_MARKER_NAMES = (
    "exit_code.txt",
    "exit_code",
    "process_exit_code.txt",
    ".exit_code",
)
PID_FILE_NAMES = (
    "wrapper_pid.txt",
    "wrapper.pid",
    "pid.txt",
    "pid",
    "child_pid.txt",
)
LOG_FILE_NAMES = ("stdout_stderr.log", "eval.log", "output.log")
DEFAULT_LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class SweepPaths:
    repository: Path
    dataset: Path
    formal_root: Path
    bundle_root: Path
    sweep_root: Path
    launcher_root: Path


@dataclass(frozen=True)
class Cell:
    cell_id: str
    epoch: int
    variant: str
    score_mode: str
    checkpoint: str
    output_dir: str
    job_dir: str
    config_path: str
    argv: tuple[str, ...]


@dataclass
class ActiveCell:
    cell: Cell
    pid: int
    gpu: int
    process: subprocess.Popen[Any] | None
    log_handle: Any | None
    adopted: bool


@dataclass
class CheckpointObservation:
    signature: tuple[int, int]
    matching_polls: int


def checkpoint_path(formal_root: Path, variant: str, epoch: int) -> Path:
    return (
        formal_root
        / variant
        / f"seed_{SEED}"
        / "checkpoints"
        / f"actor_free_td_lewm_v2_{variant}"
        / variant
        / f"epoch_{epoch:02d}.pt"
    )


def build_cells(*, paths: SweepPaths, python: str) -> list[Cell]:
    cells: list[Cell] = []
    for epoch in EPOCHS:
        for variant in VARIANTS:
            checkpoint = checkpoint_path(paths.formal_root, variant, epoch)
            config = (
                f"configs/experiment/actor_free_td_lewm_v2_{variant}"
                "_cube_checkpoint_o50.yaml"
            )
            for mode in SCORE_MODES:
                cell_id = f"e{epoch:02d}_{variant}_{mode}"
                output = paths.sweep_root / f"epoch_{epoch:02d}" / variant / mode
                argv = (
                    python,
                    f"scripts/evaluate_actor_free_td_lewm_v2_{variant}.py",
                    "--config",
                    config,
                    "--dataset",
                    str(paths.dataset),
                    "--checkpoint-path",
                    str(checkpoint),
                    "--checkpoint-epoch",
                    str(epoch),
                    "--score-mode",
                    mode,
                    "--output-dir",
                    str(output),
                )
                cells.append(
                    Cell(
                        cell_id=cell_id,
                        epoch=epoch,
                        variant=variant,
                        score_mode=mode,
                        checkpoint=str(checkpoint),
                        output_dir=str(output),
                        job_dir=str(paths.launcher_root / "jobs" / cell_id),
                        config_path=config,
                        argv=argv,
                    )
                )
    if len(cells) != 126:
        raise AssertionError(f"Sweep must contain exactly 126 cells, got {len(cells)}")
    if len({cell.output_dir for cell in cells}) != len(cells):
        raise AssertionError("Sweep output directories are not unique.")
    return cells


def load_checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Strict checkpoint verification requires torch.") from error
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return value


def audit_stable_checkpoint(
    path: Path,
    *,
    epoch: int,
    strict_metadata: bool,
    metadata_loader: Callable[[Path], Mapping[str, Any]] = load_checkpoint_metadata,
) -> dict[str, Any]:
    before = path.stat()
    digest = file_sha256(path)
    metadata: dict[str, Any] | None = None
    if strict_metadata:
        payload = metadata_loader(path)
        expected = {"epoch": epoch, "global_step": STEPS_PER_EPOCH * epoch}
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"{path} {key}={payload.get(key)!r}, expected {expected_value}."
                )
        metadata = expected
    after = path.stat()
    before_signature = (before.st_size, before.st_mtime_ns)
    after_signature = (after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"Checkpoint changed while it was being audited: {path}")
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
        "metadata": metadata,
    }


def observe_checkpoint(
    path: Path, previous: CheckpointObservation | None,
) -> CheckpointObservation | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file() or stat.st_size <= 0:
        return None
    signature = (stat.st_size, stat.st_mtime_ns)
    if previous is not None and previous.signature == signature:
        return CheckpointObservation(signature, previous.matching_polls + 1)
    return CheckpointObservation(signature, 1)


def audit_complete_output(cell: Cell) -> dict[str, Any]:
    output = Path(cell.output_dir)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{cell.cell_id} is missing output files: {missing}")
    results = read_json(output / "results.json")
    expected_results = {
        "method": f"actor_free_td_lewm_v2_{cell.variant}",
        "method_family": "actor_free_td_lewm_v2",
        "variant": cell.variant,
        "score_mode": cell.score_mode,
        "smoke": False,
        "pilot": False,
    }
    for key, expected in expected_results.items():
        if results.get(key) != expected:
            raise ValueError(
                f"{cell.cell_id} results.{key}={results.get(key)!r}, expected {expected!r}."
            )
    successes = results.get("metrics", {}).get("episode_successes")
    if not isinstance(successes, list) or len(successes) != 50:
        raise ValueError(
            f"{cell.cell_id} does not contain exactly 50 episode outcomes."
        )
    manifest = read_json(output / "protocol_manifest.json")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{cell.cell_id} has no checkpoint manifest mapping.")
    expected_checkpoint = {
        "method": f"actor_free_td_lewm_v2_{cell.variant}",
        "method_family": "actor_free_td_lewm_v2",
        "variant": cell.variant,
        "epoch": cell.epoch,
        "global_step": STEPS_PER_EPOCH * cell.epoch,
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"{cell.cell_id} checkpoint.{key}={checkpoint.get(key)!r}, "
                f"expected {expected!r}."
            )
    actual_checkpoint = Path(cell.checkpoint)
    recorded_path = checkpoint.get("path")
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).resolve() != actual_checkpoint.resolve()
    ):
        raise ValueError(f"{cell.cell_id} manifest binds a different checkpoint path.")
    recorded_sha = checkpoint.get("sha256")
    actual_sha = file_sha256(actual_checkpoint)
    if recorded_sha != actual_sha:
        raise ValueError(
            f"{cell.cell_id} manifest checkpoint SHA-256 is stale or wrong."
        )
    return {
        name: {
            "path": str(output / name),
            "size_bytes": (output / name).stat().st_size,
            "sha256": file_sha256(output / name),
        }
        for name in REQUIRED_OUTPUT_FILES
    }


def find_named_file(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def read_exit_code(job_dir: Path) -> tuple[int | None, Path | None]:
    marker = find_named_file(job_dir, EXIT_MARKER_NAMES)
    if marker is None:
        return None, None
    try:
        return int(marker.read_text(encoding="utf-8").strip()), marker
    except ValueError as error:
        raise ValueError(f"Invalid exit marker: {marker}") from error


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            if proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_record(job_dir: Path) -> dict[str, Any] | None:
    for name in ("start_evidence.json", "start.json", "job_start.json"):
        candidate = job_dir / name
        if candidate.is_file():
            return read_json(candidate)
    return None


def discover_pid(job_dir: Path) -> int | None:
    record = _start_record(job_dir)
    if record is not None:
        for key in ("wrapper_pid", "pid", "child_pid"):
            value = record.get(key)
            if isinstance(value, int) and value > 0:
                return value
    for name in PID_FILE_NAMES:
        candidate = job_dir / name
        if candidate.is_file():
            try:
                value = int(candidate.read_text(encoding="utf-8").strip())
            except ValueError:
                continue
            if value > 0:
                return value
    return None


def _proc_values(pid: int, name: str, separator: bytes) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/{name}").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(separator) if part]


def discover_gpu(job_dir: Path, pid: int) -> int | None:
    record = _start_record(job_dir)
    if record is not None:
        gpu = record.get("gpu")
        if isinstance(gpu, int):
            return gpu
        if isinstance(gpu, Mapping) and isinstance(gpu.get("index"), int):
            return int(gpu["index"])
    for value in _proc_values(pid, "environ", b"\0"):
        if value.startswith("CUDA_VISIBLE_DEVICES="):
            raw = value.partition("=")[2]
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def adoption_matches_cell(cell: Cell, job_dir: Path, pid: int) -> bool:
    expected = list(cell.argv)

    def evaluator_tail(argv: Sequence[str]) -> list[str] | None:
        for index, item in enumerate(argv):
            if item == expected[1]:
                return list(argv[index:])
        return None

    record = _start_record(job_dir)
    if record is not None:
        argv = record.get("argv")
        if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
            tail = evaluator_tail(argv)
            if tail is not None and tail == expected[1:]:
                return True
        recorded_hash = record.get("argv_sha256")
        if isinstance(recorded_hash, str):
            if recorded_hash == canonical_json_sha256(expected):
                return True
    command = _proc_values(pid, "cmdline", b"\0")
    if any(
        command[index : index + len(expected)] == expected
        for index in range(max(0, len(command) - len(expected) + 1))
    ):
        return True
    # A manually started bash wrapper usually stores the evaluator command in
    # one cmdline element.  Bind all cell-unique arguments before adopting it.
    flattened = " ".join(command)
    unique_tokens = (
        expected[1],
        "--checkpoint-path",
        cell.checkpoint,
        "--checkpoint-epoch",
        str(cell.epoch),
        "--score-mode",
        cell.score_mode,
        "--output-dir",
        cell.output_dir,
    )
    return bool(flattened) and all(token in flattened for token in unique_tokens)


def audit_log(job_dir: Path) -> dict[str, Any] | None:
    log = find_named_file(job_dir, LOG_FILE_NAMES)
    if log is None:
        return None
    return {
        "path": str(log),
        "size_bytes": log.stat().st_size,
        "sha256": file_sha256(log),
    }


def classify_existing_cells(
    cells: Sequence[Cell],
    gpu_indices: Sequence[int],
    adoption_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[Cell], dict[str, ActiveCell]]:
    states: dict[str, dict[str, Any]] = {}
    pending: list[Cell] = []
    active: dict[str, ActiveCell] = {}
    allowed_gpus = set(gpu_indices)
    for cell in cells:
        base = {
            **asdict(cell),
            "argv": list(cell.argv),
            "argv_sha256": canonical_json_sha256(list(cell.argv)),
        }
        output_error_text: str | None = None
        try:
            audit = audit_complete_output(cell)
        except Exception as output_error:
            audit = None
            output_error_text = str(output_error)
        else:
            states[cell.cell_id] = {**base, "state": "REUSED", "output_audit": audit}
            continue
        job_dir = Path(cell.job_dir)
        if job_dir.exists() and not job_dir.is_dir():
            states[cell.cell_id] = {
                **base,
                "state": "FAILED",
                "error": f"Existing job path is not a directory: {job_dir}",
            }
            continue
        if job_dir.is_dir():
            try:
                exit_code, marker = read_exit_code(job_dir)
            except Exception as error:
                states[cell.cell_id] = {**base, "state": "FAILED", "error": str(error)}
                continue
            override = (adoption_overrides or {}).get(cell.cell_id)
            pid = override[0] if override is not None else discover_pid(job_dir)
            if marker is None and pid is not None and process_alive(pid):
                gpu = (
                    override[1] if override is not None else discover_gpu(job_dir, pid)
                )
                if gpu not in allowed_gpus:
                    states[cell.cell_id] = {
                        **base,
                        "state": "FAILED",
                        "error": f"Running cell GPU {gpu!r} is not one of {sorted(allowed_gpus)}.",
                    }
                    continue
                if not adoption_matches_cell(cell, job_dir, pid):
                    states[cell.cell_id] = {
                        **base,
                        "state": "FAILED",
                        "error": "Running process cannot be bound to the exact expected argv.",
                    }
                    continue
                states[cell.cell_id] = {
                    **base,
                    "state": "ADOPTED_RUNNING",
                    "pid": pid,
                    "gpu": gpu,
                    "adopted": True,
                }
                active[cell.cell_id] = ActiveCell(cell, pid, gpu, None, None, True)
                continue
            reason = (
                f"Existing job has exit marker {marker} with code {exit_code} but no complete output."
                if marker is not None
                else "Existing job directory is not a live, attributable job and output is incomplete."
            )
            states[cell.cell_id] = {**base, "state": "FAILED", "error": reason}
            continue
        if Path(cell.output_dir).exists():
            states[cell.cell_id] = {
                **base,
                "state": "FAILED",
                "error": (
                    "Existing output is incomplete and will not be overwritten: "
                    f"{output_error_text}"
                ),
            }
            continue
        states[cell.cell_id] = {**base, "state": "WAITING_CHECKPOINT"}
        pending.append(cell)
    return states, pending, active


def _child_main(arguments: Sequence[str]) -> int:
    if len(arguments) < 4 or arguments[2] != "--":
        raise SystemExit(
            "internal usage: _child EXIT_PATH CHILD_PID_PATH -- COMMAND ..."
        )
    exit_path = Path(arguments[0])
    child_pid_path = Path(arguments[1])
    try:
        process = subprocess.Popen(list(arguments[3:]))
        atomic_write_text(child_pid_path, f"{process.pid}\n")
        return_code = process.wait()
    except BaseException:
        atomic_write_text(exit_path, "125\n")
        raise
    atomic_write_text(exit_path, f"{return_code}\n")
    return return_code


def _state_document(
    *,
    paths: SweepPaths,
    states: Mapping[str, Mapping[str, Any]],
    checkpoint_states: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for state in states.values():
        label = str(state["state"])
        counts[label] = counts.get(label, 0) + 1
    return {
        "schema_version": 1,
        "source": "actor_free_td_lewm_v2_epoch_sweep_scheduler",
        "updated_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "scheduler_pid": os.getpid(),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "cell_count": len(states),
        "state_counts": counts,
        "checkpoints": dict(checkpoint_states),
        "render_environment": dict(environment),
        "cells": dict(states),
    }


def run_scheduler(
    *,
    paths: SweepPaths,
    cells: Sequence[Cell],
    gpu_indices: Sequence[int],
    max_evals_per_gpu: int,
    poll_seconds: float,
    stable_polls: int,
    strict_checkpoint_metadata: bool,
    render_environment: Mapping[str, str],
    run_dir: Path,
    plan: Mapping[str, Any] | None = None,
    adoption_overrides: Mapping[str, tuple[int, int]] | None = None,
    max_polls: int | None = None,
    metadata_loader: Callable[[Path], Mapping[str, Any]] = load_checkpoint_metadata,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    if stable_polls < 2:
        raise ValueError("stable_polls must be at least two.")
    run_dir.mkdir(parents=True, exist_ok=False)
    if plan is not None:
        atomic_write_json(run_dir / "plan.json", plan)
    state_path = run_dir / "state.json"
    states, pending, active = classify_existing_cells(
        cells, gpu_indices, adoption_overrides
    )
    observations: dict[str, CheckpointObservation] = {}
    checkpoint_states: dict[str, dict[str, Any]] = {}
    ready: dict[str, dict[str, Any]] = {}
    polls = 0
    child_environment = os.environ.copy()
    child_environment.update(render_environment)
    child_environment.update(
        {
            "PYTHONPATH": str(paths.repository / "src"),
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    while pending or active:
        polls += 1
        for cell_id, running in list(active.items()):
            job_dir = Path(running.cell.job_dir)
            exit_code, marker = read_exit_code(job_dir)
            polled = running.process.poll() if running.process is not None else None
            alive = process_alive(running.pid) if running.adopted else polled is None
            if (
                running.adopted
                and alive
                and not adoption_matches_cell(running.cell, job_dir, running.pid)
            ):
                alive = False
            if marker is None and alive:
                continue
            if marker is None and not alive:
                error = "Process ended without an atomic exit marker."
            elif exit_code != 0:
                error = f"Child exit marker reports {exit_code}."
            elif running.process is not None and polled not in (None, exit_code):
                error = f"Wrapper return code {polled!r} differs from exit marker {exit_code}."
            else:
                try:
                    output_audit = audit_complete_output(running.cell)
                except Exception as verification_error:
                    error = f"Output verification failed: {verification_error}"
                else:
                    error = None
            if running.log_handle is not None:
                running.log_handle.close()
            final = {
                "schema_version": 1,
                "cell_id": cell_id,
                "ended_at_utc": utc_now(),
                "pid": running.pid,
                "gpu": running.gpu,
                "adopted": running.adopted,
                "exit_code_marker": exit_code,
                "exit_marker": str(marker) if marker is not None else None,
                "log": audit_log(job_dir),
                "error": error,
                "output_audit": output_audit if error is None else None,
            }
            if not running.adopted:
                atomic_write_json(job_dir / "final_evidence.json", final)
            states[cell_id].update(
                {
                    "state": "SUCCEEDED" if error is None else "FAILED",
                    "ended_at_utc": final["ended_at_utc"],
                    "exit_code_marker": exit_code,
                    "log": final["log"],
                    "output_audit": final["output_audit"],
                    "error": error,
                }
            )
            del active[cell_id]

        checkpoint_groups: dict[str, list[Cell]] = {}
        for cell in pending:
            checkpoint_groups.setdefault(cell.checkpoint, []).append(cell)
        failed_checkpoints: set[str] = set()
        for raw_path, group in checkpoint_groups.items():
            if raw_path in ready:
                continue
            path = Path(raw_path)
            observation = observe_checkpoint(path, observations.get(raw_path))
            if observation is None:
                observations.pop(raw_path, None)
                checkpoint_states[raw_path] = {"state": "MISSING"}
                continue
            observations[raw_path] = observation
            checkpoint_states[raw_path] = {
                "state": "STABILIZING",
                "size_bytes": observation.signature[0],
                "mtime_ns": observation.signature[1],
                "matching_polls": observation.matching_polls,
                "required_matching_polls": stable_polls,
            }
            if observation.matching_polls < stable_polls:
                continue
            try:
                audit = audit_stable_checkpoint(
                    path,
                    epoch=group[0].epoch,
                    strict_metadata=strict_checkpoint_metadata,
                    metadata_loader=metadata_loader,
                )
            except Exception as error:
                checkpoint_states[raw_path] = {"state": "INVALID", "error": str(error)}
                failed_checkpoints.add(raw_path)
            else:
                ready[raw_path] = audit
                checkpoint_states[raw_path] = {"state": "READY", **audit}
        if failed_checkpoints:
            survivors: list[Cell] = []
            for cell in pending:
                if cell.checkpoint in failed_checkpoints:
                    states[cell.cell_id].update(
                        {
                            "state": "FAILED",
                            "error": (
                                "Checkpoint failed closed verification: "
                                + checkpoint_states[cell.checkpoint]["error"]
                            ),
                        }
                    )
                else:
                    survivors.append(cell)
            pending = survivors

        occupancy = {gpu: 0 for gpu in gpu_indices}
        for running in active.values():
            occupancy[running.gpu] += 1
        dispatchable = [cell for cell in pending if cell.checkpoint in ready]
        for cell in dispatchable:
            candidates = [
                gpu for gpu in gpu_indices if occupancy[gpu] < max_evals_per_gpu
            ]
            if not candidates:
                break
            gpu = min(candidates, key=lambda index: (occupancy[index], index))
            checkpoint_stat = Path(cell.checkpoint).stat()
            checkpoint_audit = ready[cell.checkpoint]
            if (
                checkpoint_stat.st_size != checkpoint_audit["size_bytes"]
                or checkpoint_stat.st_mtime_ns != checkpoint_audit["mtime_ns"]
                or file_sha256(Path(cell.checkpoint)) != checkpoint_audit["sha256"]
            ):
                ready.pop(cell.checkpoint, None)
                observations.pop(cell.checkpoint, None)
                checkpoint_states[cell.checkpoint] = {
                    "state": "CHANGED_BEFORE_DISPATCH"
                }
                continue
            output = Path(cell.output_dir)
            if output.exists():
                try:
                    output_audit = audit_complete_output(cell)
                except Exception as output_error:
                    states[cell.cell_id].update(
                        {
                            "state": "FAILED",
                            "error": (
                                "Output appeared before dispatch and is incomplete; "
                                f"refusing overwrite: {output_error}"
                            ),
                        }
                    )
                else:
                    states[cell.cell_id].update(
                        {"state": "REUSED", "output_audit": output_audit}
                    )
                pending.remove(cell)
                continue
            job_dir = Path(cell.job_dir)
            try:
                job_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                states[cell.cell_id].update(
                    {
                        "state": "FAILED",
                        "error": "Job directory appeared before dispatch; refusing overwrite.",
                    }
                )
                pending.remove(cell)
                continue
            log_path = job_dir / "stdout_stderr.log"
            exit_path = job_dir / "exit_code.txt"
            child_pid_path = job_dir / "child_pid.txt"
            wrapper_pid_path = job_dir / "wrapper_pid.txt"
            log_handle = log_path.open("xb")
            environment = child_environment.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            wrapper = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_child",
                str(exit_path),
                str(child_pid_path),
                "--",
                *cell.argv,
            ]
            started_at = utc_now()
            try:
                process = popen(
                    wrapper,
                    cwd=paths.repository,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as launch_error:
                log_handle.close()
                atomic_write_json(
                    job_dir / "final_evidence.json",
                    {
                        "schema_version": 1,
                        "cell_id": cell.cell_id,
                        "ended_at_utc": utc_now(),
                        "error": f"Launch failed: {launch_error}",
                    },
                )
                states[cell.cell_id].update(
                    {"state": "FAILED", "error": f"Launch failed: {launch_error}"}
                )
                pending.remove(cell)
                continue
            atomic_write_text(wrapper_pid_path, f"{process.pid}\n")
            start = {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "started_at_utc": started_at,
                "wrapper_pid": process.pid,
                "gpu": gpu,
                "argv": list(cell.argv),
                "argv_sha256": canonical_json_sha256(list(cell.argv)),
                "wrapper_argv": wrapper,
                "cwd": str(paths.repository),
                "checkpoint": checkpoint_audit,
                "environment": {
                    key: environment[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "MUJOCO_GL",
                        "PYOPENGL_PLATFORM",
                        "LD_PRELOAD",
                    )
                    if key in environment
                },
                "log_path": str(log_path),
                "exit_code_path": str(exit_path),
            }
            atomic_write_json(job_dir / "start_evidence.json", start)
            active[cell.cell_id] = ActiveCell(
                cell, process.pid, gpu, process, log_handle, False
            )
            occupancy[gpu] += 1
            states[cell.cell_id].update(
                {
                    "state": "RUNNING",
                    "started_at_utc": started_at,
                    "pid": process.pid,
                    "gpu": gpu,
                    "start_evidence": str(job_dir / "start_evidence.json"),
                }
            )
            pending.remove(cell)

        atomic_write_json(
            state_path,
            _state_document(
                paths=paths,
                states=states,
                checkpoint_states=checkpoint_states,
                environment=render_environment,
            ),
        )
        if max_polls is not None and polls >= max_polls and (pending or active):
            return 2
        if pending or active:
            sleeper(poll_seconds)
    return 1 if any(state["state"] == "FAILED" for state in states.values()) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--formal-root")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-evals-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument(
        "--checkpoint-metadata",
        choices=("strict", "none"),
        default="strict",
        help="Strictly require epoch/global_step metadata, or only audit file stability/hash.",
    )
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument("--pyopengl-platform", default="osmesa")
    parser.add_argument("--ld-preload", default=DEFAULT_LD_PRELOAD)
    parser.add_argument("--launch-id")
    parser.add_argument(
        "--adopt-running",
        action="append",
        default=[],
        metavar="CELL=PID:GPU",
        help=(
            "Explicitly bind a pre-existing live wrapper when its job directory "
            "does not contain PID/GPU evidence. May be repeated."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("--gpus must contain distinct GPU indices.")
    if args.max_evals_per_gpu < 1:
        raise SystemExit("--max-evals-per-gpu must be positive.")
    if args.stable_polls < 2:
        raise SystemExit("--stable-polls must be at least two.")
    adoption_overrides: dict[str, tuple[int, int]] = {}
    for value in args.adopt_running:
        try:
            cell_id, raw_process = value.split("=", 1)
            raw_pid, raw_gpu = raw_process.split(":", 1)
            binding = (int(raw_pid), int(raw_gpu))
        except (ValueError, TypeError) as error:
            raise SystemExit(
                f"Invalid --adopt-running {value!r}; expected CELL=PID:GPU."
            ) from error
        if cell_id in adoption_overrides:
            raise SystemExit(f"Duplicate --adopt-running cell: {cell_id}")
        adoption_overrides[cell_id] = binding
    repository = Path(__file__).resolve().parents[1]
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    paths = SweepPaths(
        repository=repository,
        dataset=Path(args.dataset).expanduser().resolve(),
        formal_root=Path(args.formal_root).expanduser().resolve()
        if args.formal_root
        else bundle_root / "formal",
        bundle_root=bundle_root,
        sweep_root=bundle_root / "evaluation_sweeps",
        launcher_root=bundle_root / "evaluation_sweep_launcher",
    )
    python = Path(args.python).expanduser()
    if not python.is_absolute():
        python = Path(os.path.abspath(python))
    cells = build_cells(paths=paths, python=str(python))
    unknown_adoptions = set(adoption_overrides) - {cell.cell_id for cell in cells}
    if unknown_adoptions:
        raise SystemExit(f"Unknown --adopt-running cells: {sorted(unknown_adoptions)}")
    render_environment = {
        "MUJOCO_GL": args.mujoco_gl,
        "PYOPENGL_PLATFORM": args.pyopengl_platform,
        "LD_PRELOAD": args.ld_preload,
    }
    plan = {
        "schema_version": 1,
        "cell_count": len(cells),
        "epoch_count": len(EPOCHS),
        "excluded_epoch": 10,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "gpus": args.gpus,
        "max_evals_per_gpu": args.max_evals_per_gpu,
        "stable_polls": args.stable_polls,
        "checkpoint_metadata": args.checkpoint_metadata,
        "render_environment": render_environment,
        "adoption_overrides": {
            cell_id: {"pid": value[0], "gpu": value[1]}
            for cell_id, value in adoption_overrides.items()
        },
        "cells": [{**asdict(cell), "argv": list(cell.argv)} for cell in cells],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not paths.dataset.exists():
        raise FileNotFoundError(paths.dataset)
    paths.launcher_root.mkdir(parents=True, exist_ok=True)
    launch_id = args.launch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths.launcher_root / "runs" / launch_id
    code = run_scheduler(
        paths=paths,
        cells=cells,
        gpu_indices=args.gpus,
        max_evals_per_gpu=args.max_evals_per_gpu,
        poll_seconds=args.poll_seconds,
        stable_polls=args.stable_polls,
        strict_checkpoint_metadata=args.checkpoint_metadata == "strict",
        render_environment=render_environment,
        run_dir=run_dir,
        plan=plan,
        adoption_overrides=adoption_overrides,
    )
    return code


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(main())
