#!/usr/bin/env python3
"""Fail-closed five-GPU launcher for Actor-Free TD-LeWM V2.

The launcher owns orchestration only.  Training and evaluation remain in the
six audited method entry points.  Every child command is persisted as an argv
array, every GPU is identified by UUID, and no queued work is started after a
child or its output verification fails.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

VARIANTS = ("g1", "g2", "g3", "d", "f", "c")
SCORE_MODES = ("f_plus_g", "f_only", "g_only")
SEED = 3072
FORMAL_EPOCH = 10
FORMAL_GLOBAL_STEP = 127_960
DATASET_MANIFEST_SHA256 = (
    "9de531030c6bca21a7b3215d7abea3aaf277e68a1e4cec03c8c6e22ad0d20dcd"
)
SPLIT_SHA256 = "4594afb3603b4258431ff9076c82acbe3ddcaccb277940b825a99017ce83d830"
NEIGHBOR_MANIFEST_SHA256 = (
    "3b2d785790d86c4c45bc10f1cf706f9fc186a02071fb4f8b586eca75a2af76f2"
)
V1_SHA256 = {
    "c": "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3",
    "d": "3115fffeb83ba6ae7e0c272913fe7a1ba16d42953b2185f6a3f7b168899d819a",
    "f": "b4de1b511075d763194ad1e332d127cbe390553738162f3a402ef8847bb74fd0",
    "g1": "c224d18fcd8390247f115239c4b2db013479a062438cca92003674c739f3e24b",
    "g2": "1c290f91772b42fdf6824d92832c6fff4e2d8ca3ea08089ff1a41016ea1c2ebe",
    "g3": "b279a85b1dd0816bd5fb9724da490810d470755880639297aa13699c86c2d8fb",
}


def _utc_now() -> str:
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


def _run_text(command: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def audited_git_state(repository: Path) -> dict[str, Any]:
    revision = _run_text(("git", "rev-parse", "HEAD"), cwd=repository)
    if len(revision) != 40:
        raise RuntimeError("git rev-parse did not return a full revision.")
    dirty = _run_text(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=repository
    )
    if dirty:
        raise RuntimeError("V2 launcher requires a clean Git worktree.")
    return {"revision": revision, "short_revision": revision[:7], "clean": True}


def audit_python_runtime(python: Path) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    program = """
import importlib.metadata
import json
import platform
import sys
print(json.dumps({
    "executable": sys.executable,
    "python": platform.python_version(),
    "stable_worldmodel": importlib.metadata.version("stable-worldmodel"),
    "torch": importlib.metadata.version("torch"),
    "lightning": importlib.metadata.version("lightning"),
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", program], check=True, capture_output=True, text=True
    )
    runtime = json.loads(result.stdout)
    if runtime.get("stable_worldmodel") != "0.1.1":
        raise RuntimeError("V2 requires stable-worldmodel==0.1.1.")
    return runtime


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    total_memory_mib: int
    free_memory_mib: int
    compute_pids: tuple[int, ...] = ()


def query_gpus() -> list[GPUInfo]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    pids_by_uuid: dict[str, list[int]] = {}
    if compute.returncode == 0:
        for row in csv.reader(compute.stdout.splitlines()):
            if len(row) != 2:
                continue
            uuid, raw_pid = (item.strip() for item in row)
            try:
                pids_by_uuid.setdefault(uuid, []).append(int(raw_pid))
            except ValueError:
                continue
    gpus: list[GPUInfo] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {row!r}")
        index, uuid, name, total, free = (item.strip() for item in row)
        gpus.append(
            GPUInfo(
                index=int(index),
                uuid=uuid,
                name=name,
                total_memory_mib=int(total),
                free_memory_mib=int(free),
                compute_pids=tuple(sorted(pids_by_uuid.get(uuid, []))),
            )
        )
    return gpus


@dataclass(frozen=True)
class Paths:
    repository: Path
    artifact_root: Path
    dataset: Path
    dataset_manifest: Path
    split_indices: Path
    neighbor_index: Path
    v1_root: Path
    bundle_root: Path
    smoke_root: Path
    formal_root: Path
    evaluation_root: Path


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    variant: str
    score_mode: str | None
    argv: tuple[str, ...]
    config_path: str
    output_base: str
    run_dir: str
    expected_checkpoint: str | None
    expected_epoch: int | None
    expected_global_step: int | None


def _v1_checkpoint(paths: Paths, variant: str) -> Path:
    return (
        paths.v1_root
        / variant
        / f"seed_{SEED}"
        / "checkpoints"
        / f"actor_free_td_lewm_v1_{variant}"
        / variant
        / "epoch_10.pt"
    )


def _v2_checkpoint(root: Path, variant: str, epoch: int = FORMAL_EPOCH) -> Path:
    return (
        root
        / variant
        / f"seed_{SEED}"
        / "checkpoints"
        / f"actor_free_td_lewm_v2_{variant}"
        / variant
        / f"epoch_{epoch:02d}.pt"
    )


def build_jobs(
    *,
    stage: str,
    paths: Paths,
    python: str,
    formal_resume: str = "never",
) -> list[Job]:
    if stage not in {"smoke1", "smoke2", "formal", "eval"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if formal_resume not in {"never", "required"}:
        raise ValueError(
            "Formal resume must be 'never' or 'required'; auto is forbidden."
        )
    jobs: list[Job] = []
    if stage == "eval":
        for mode in SCORE_MODES:
            for variant in VARIANTS:
                config = (
                    f"configs/experiment/actor_free_td_lewm_v2_{variant}"
                    "_cube_checkpoint_o50.yaml"
                )
                output = paths.evaluation_root / variant / mode
                checkpoint = _v2_checkpoint(paths.formal_root, variant)
                argv = (
                    python,
                    f"scripts/evaluate_actor_free_td_lewm_v2_{variant}.py",
                    "--config",
                    config,
                    "--dataset",
                    str(paths.dataset),
                    "--checkpoint-path",
                    str(checkpoint),
                    "--score-mode",
                    mode,
                    "--output-dir",
                    str(output),
                )
                jobs.append(
                    Job(
                        job_id=f"{variant}__{mode}",
                        stage=stage,
                        variant=variant,
                        score_mode=mode,
                        argv=argv,
                        config_path=config,
                        output_base=str(output),
                        run_dir=str(output),
                        expected_checkpoint=str(checkpoint),
                        expected_epoch=FORMAL_EPOCH,
                        expected_global_step=FORMAL_GLOBAL_STEP,
                    )
                )
        return jobs

    smoke = stage in {"smoke1", "smoke2"}
    output_root = paths.smoke_root if smoke else paths.formal_root
    resume = "required" if stage == "smoke2" else formal_resume
    expected_epoch = 2 if stage == "smoke2" else 1 if stage == "smoke1" else 10
    # resolve_train_batch_limit(smoke=True) deliberately runs two batches per
    # epoch, independently of the smoke-only --max-steps CLI argument.  The
    # second required-resume call therefore reaches four cumulative updates.
    expected_step = 4 if stage == "smoke2" else 2 if stage == "smoke1" else 127_960
    for variant in VARIANTS:
        config = f"configs/experiment/actor_free_td_lewm_v2_{variant}_cube_train.yaml"
        output = output_root / variant
        run_dir = output / (f"seed_{SEED}_smoke" if smoke else f"seed_{SEED}")
        checkpoint = (
            run_dir
            / "checkpoints"
            / f"actor_free_td_lewm_v2_{variant}"
            / variant
            / f"epoch_{expected_epoch:02d}.pt"
        )
        argv_list = [
            python,
            f"scripts/train_actor_free_td_lewm_v2_{variant}.py",
            "--config",
            config,
            "--dataset",
            str(paths.dataset),
            "--output-dir",
            str(output),
            "--seed",
            str(SEED),
            "--initial-v1-checkpoint",
            str(_v1_checkpoint(paths, variant)),
            "--split-indices",
            str(paths.split_indices),
        ]
        if variant == "g1":
            argv_list.extend(("--neighbor-index", str(paths.neighbor_index)))
        if smoke:
            argv_list.extend(("--smoke", "--max-steps", "1", "--skip-validation"))
        argv_list.extend(("--resume", resume))
        jobs.append(
            Job(
                job_id=variant,
                stage=stage,
                variant=variant,
                score_mode=None,
                argv=tuple(argv_list),
                config_path=config,
                output_base=str(output),
                run_dir=str(run_dir),
                expected_checkpoint=str(checkpoint),
                expected_epoch=expected_epoch,
                expected_global_step=expected_step,
            )
        )
    return jobs


def _audit_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
    }


def _protocol_file_closure(
    path: Path, seen: frozenset[Path] = frozenset()
) -> set[Path]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Protocol inheritance cycle at {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Protocol is not a mapping: {resolved}")
    parent = value.get("extends")
    if parent is None:
        return {resolved}
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"Invalid protocol extends value: {resolved}")
    return {resolved} | _protocol_file_closure(
        resolved.parent / parent, seen | {resolved}
    )


def audit_inputs(paths: Paths, jobs: Sequence[Job]) -> dict[str, Any]:
    if not paths.dataset.exists():
        raise FileNotFoundError(paths.dataset)
    audit: dict[str, Any] = {
        "dataset": {
            "path": str(paths.dataset),
            "manifest": _audit_file(paths.dataset_manifest, DATASET_MANIFEST_SHA256),
        },
        "configs": {},
    }
    config_paths: set[Path] = set()
    for job in jobs:
        config_paths.update(_protocol_file_closure(paths.repository / job.config_path))
    for config_path in sorted(config_paths):
        relative = str(config_path.relative_to(paths.repository))
        audit["configs"][relative] = _audit_file(config_path)
    if jobs and jobs[0].stage != "eval":
        audit["split_indices"] = _audit_file(paths.split_indices, SPLIT_SHA256)
        audit["neighbor_index"] = {
            "path": str(paths.neighbor_index),
            "manifest": _audit_file(
                paths.neighbor_index / "manifest.json", NEIGHBOR_MANIFEST_SHA256
            ),
        }
        audit["initial_v1_checkpoints"] = {
            variant: _audit_file(_v1_checkpoint(paths, variant), V1_SHA256[variant])
            for variant in VARIANTS
        }
    else:
        audit["formal_checkpoints"] = {
            variant: _audit_file(_v2_checkpoint(paths.formal_root, variant))
            for variant in VARIANTS
        }
    return audit


def validate_stage_outputs(stage: str, jobs: Sequence[Job], formal_resume: str) -> None:
    """Reject accidental overwrites and incomplete resume sources before dispatch."""

    resume_required = stage == "smoke2" or (
        stage == "formal" and formal_resume == "required"
    )
    for job in jobs:
        if stage == "eval":
            output = Path(job.output_base)
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise RuntimeError(f"Evaluation output is not empty: {output}")
            continue
        run_dir = Path(job.run_dir)
        last_checkpoint = run_dir / "checkpoints" / "lightning" / "last.ckpt"
        manifest = run_dir / "training_manifest.json"
        if resume_required:
            if not last_checkpoint.is_file() or not manifest.is_file():
                raise RuntimeError(
                    f"Required resume evidence is incomplete for {job.job_id}: "
                    f"{last_checkpoint}, {manifest}"
                )
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"Refusing --resume never over non-empty run directory: {run_dir}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted local training output without importing torch at CLI import."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Checkpoint evidence verification requires torch.") from error
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected checkpoint mapping: {path}")
    return dict(value)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for {label}.")
    return dict(value)


def _formal_output_evidence(
    *,
    job: Job,
    git: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    method = f"actor_free_td_lewm_v2_{job.variant}"
    revision = git.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("Formal evidence requires a full lowercase Git revision.")
    if git.get("clean") is not True:
        raise ValueError("Formal evidence requires the locked clean Git revision.")

    deployment_path = Path(str(job.expected_checkpoint))
    last_path = Path(job.run_dir) / "checkpoints" / "lightning" / "last.ckpt"
    manifest_path = Path(job.run_dir) / "training_manifest.json"
    manifest = _read_json(manifest_path)
    protocol = _require_mapping(manifest.get("protocol"), label="training protocol")
    protocol_sha256 = canonical_json_sha256(protocol)
    expected_source_v1_sha256 = input_audit["initial_v1_checkpoints"][
        job.variant
    ]["sha256"]
    if expected_source_v1_sha256 != V1_SHA256[job.variant]:
        raise ValueError("Audited V1 checkpoint differs from the locked V1 input.")
    expected_neighbor_sha256 = (
        input_audit["neighbor_index"]["manifest"]["sha256"]
        if job.variant == "g1"
        else None
    )
    if job.variant == "g1" and expected_neighbor_sha256 != NEIGHBOR_MANIFEST_SHA256:
        raise ValueError("Audited G1 neighbor index differs from the locked input.")

    expected_manifest = {
        "method": method,
        "method_family": "actor_free_td_lewm_v2",
        "variant": job.variant,
        "implementation_version": "v2",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "protocol_sha256": protocol_sha256,
    }
    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Formal training_manifest.{key} differs from {expected_value!r}."
            )
    manifest_source = _require_mapping(
        manifest.get("source_v1"), label="training_manifest.source_v1"
    )
    if manifest_source.get("checkpoint_sha256") != expected_source_v1_sha256:
        raise ValueError("Formal training manifest binds another V1 checkpoint.")
    manifest_runtime = _require_mapping(
        manifest.get("runtime"), label="training_manifest.runtime"
    )
    if manifest_runtime.get("tdwm_git_revision") != revision:
        raise ValueError("Formal training manifest binds another Git revision.")
    manifest_neighbor = manifest.get("neighbor_index")
    if expected_neighbor_sha256 is None:
        if manifest_neighbor is not None:
            raise ValueError("Only formal G1 may bind a neighbor index.")
    else:
        manifest_neighbor = _require_mapping(
            manifest_neighbor, label="training_manifest.neighbor_index"
        )
        if manifest_neighbor.get("manifest_sha256") != expected_neighbor_sha256:
            raise ValueError("Formal training manifest binds another neighbor index.")

    deployment = _torch_load_checkpoint(deployment_path)
    expected_deployment = {
        "method": method,
        "variant": job.variant,
        "epoch": FORMAL_EPOCH,
        "global_step": FORMAL_GLOBAL_STEP,
    }
    for key, expected_value in expected_deployment.items():
        if deployment.get(key) != expected_value:
            raise ValueError(
                f"Formal deployment checkpoint {key} differs from {expected_value!r}."
            )
    deployment_source = _require_mapping(
        deployment.get("source_v1_provenance"),
        label="deployment source_v1_provenance",
    )
    if deployment_source.get("checkpoint_sha256") != expected_source_v1_sha256:
        raise ValueError("Formal deployment checkpoint binds another V1 checkpoint.")

    last = _torch_load_checkpoint(last_path)
    resume_identity = _require_mapping(
        last.get("v2_resume_identity"), label="last.ckpt v2_resume_identity"
    )
    expected_resume_identity = {
        "schema_version": 1,
        "method": method,
        "method_family": "actor_free_td_lewm_v2",
        "variant": job.variant,
        "implementation_version": "v2",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "protocol_sha256": protocol_sha256,
        "source_v1_sha256": expected_source_v1_sha256,
        "v2_start_revision": revision,
        "neighbor_index_manifest_sha256": expected_neighbor_sha256,
    }
    if resume_identity != expected_resume_identity:
        differing = sorted(
            key
            for key in set(resume_identity) | set(expected_resume_identity)
            if resume_identity.get(key) != expected_resume_identity.get(key)
        )
        raise ValueError(
            "Formal last.ckpt resume identity differs from locked run inputs: "
            f"{differing}"
        )

    deployment_audit = _audit_file(deployment_path)
    last_audit = _audit_file(last_path)
    return {
        "deployment_checkpoint": {
            "path": deployment_audit["path"],
            "size_bytes": deployment_audit["size_bytes"],
            "sha256": deployment_audit["sha256"],
            "epoch": FORMAL_EPOCH,
            "global_step": FORMAL_GLOBAL_STEP,
            "resume_identity": resume_identity,
            "resume_identity_source": f"{last_path}['v2_resume_identity']",
        },
        "lightning_last": {
            "path": last_audit["path"],
            "size_bytes": last_audit["size_bytes"],
            "sha256": last_audit["sha256"],
            "resume_identity": resume_identity,
        },
    }


def verify_job_output(job: Job) -> dict[str, Any]:
    output = Path(job.run_dir)
    if job.stage == "eval":
        results = output / "results.json"
        if not results.is_file():
            raise FileNotFoundError(results)
        value = _read_json(results)
        expected = {
            "method": f"actor_free_td_lewm_v2_{job.variant}",
            "method_family": "actor_free_td_lewm_v2",
            "variant": job.variant,
            "implementation_version": "v2",
            "score_mode": job.score_mode,
            "smoke": False,
            "pilot": False,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(
                    f"{job.job_id} results.{key} differs from {expected_value!r}."
                )
        outcomes = value.get("metrics", {}).get("episode_successes")
        if not isinstance(outcomes, list) or len(outcomes) != 50:
            raise ValueError(f"{job.job_id} did not complete 50 formal episodes.")
        required = {
            "results": results,
            "protocol_manifest": output / "protocol_manifest.json",
            "episode_selection": output / "episode_selection.json",
            "action_normalization": output / "action_normalization.json",
        }
        return {name: _audit_file(path) for name, path in required.items()}
    result_path = output / "training_result.json"
    checkpoint = Path(str(job.expected_checkpoint))
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = _read_json(result_path)
    if int(result.get("final_epoch", -1)) != job.expected_epoch:
        raise ValueError(f"{job.job_id} has the wrong final_epoch.")
    if int(result.get("global_step", -1)) != job.expected_global_step:
        raise ValueError(f"{job.job_id} has the wrong global_step.")
    if Path(str(result.get("deployment_checkpoint"))).resolve() != checkpoint.resolve():
        raise ValueError(f"{job.job_id} reports another deployment checkpoint.")
    return {
        "training_result": _audit_file(result_path),
        "deployment_checkpoint": _audit_file(checkpoint),
        "last_checkpoint": _audit_file(
            output / "checkpoints" / "lightning" / "last.ckpt"
        ),
        "peak_cuda_memory_bytes": result.get("peak_cuda_memory_bytes"),
    }


def _formal_execution_evidence(
    *,
    job: Job,
    gpu: GPUInfo,
    pid: int,
    started_at: str,
    ended_at: str,
    return_code: int,
    log_path: Path,
    paths: Paths,
    git: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    free_before: int,
    free_after: int,
) -> dict[str, Any]:
    neighbor: dict[str, Any] | None = None
    if job.variant == "g1":
        neighbor = {
            "path": str(paths.neighbor_index),
            "manifest_sha256": input_audit["neighbor_index"]["manifest"]["sha256"],
        }
    outputs = _formal_output_evidence(job=job, git=git, input_audit=input_audit)
    return {
        "schema_version": 1,
        "source": "v2_formal_training_launcher",
        "method": f"actor_free_td_lewm_v2_{job.variant}",
        "variant": job.variant,
        "hostname": socket.gethostname(),
        "gpu": {"index": gpu.index, "uuid": gpu.uuid, "name": gpu.name},
        "process": {
            "pid": pid,
            "argv": list(job.argv),
            "argv_sha256": canonical_json_sha256(list(job.argv)),
            "cwd": str(paths.repository),
            "git_revision": git["revision"],
            "git_clean": True,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "return_code": return_code,
        },
        "log": {
            "path": str(log_path),
            "size_bytes": log_path.stat().st_size,
            "sha256": file_sha256(log_path),
        },
        "inputs": {
            "dataset": {
                "path": str(paths.dataset),
                "manifest_path": str(paths.dataset_manifest),
                "manifest_sha256": input_audit["dataset"]["manifest"]["sha256"],
            },
            "initial_v1_checkpoint": {
                "path": str(_v1_checkpoint(paths, job.variant)),
                "sha256": input_audit["initial_v1_checkpoints"][job.variant]["sha256"],
            },
            "split_indices": {
                "path": str(paths.split_indices),
                "sha256": input_audit["split_indices"]["sha256"],
            },
            "neighbor_index": neighbor,
        },
        "outputs": outputs,
        "disk": {"free_bytes_before": free_before, "free_bytes_after": free_after},
    }


@dataclass
class RunningJob:
    job: Job
    gpu: GPUInfo
    process: subprocess.Popen[Any]
    log_handle: Any
    log_path: Path
    exit_code_path: Path
    child_pid_path: Path
    start_evidence_path: Path
    final_evidence_path: Path
    started_at: str
    free_bytes_before: int


def _child_main(arguments: Sequence[str]) -> int:
    if len(arguments) < 4 or arguments[2] != "--":
        raise SystemExit(
            "internal child usage: _child EXIT_PATH PID_PATH -- COMMAND ..."
        )
    exit_path = Path(arguments[0])
    pid_path = Path(arguments[1])
    try:
        process = subprocess.Popen(list(arguments[3:]))
        atomic_write_text(pid_path, f"{process.pid}\n")
        return_code = process.wait()
    except BaseException:
        return_code = 125
        atomic_write_text(exit_path, f"{return_code}\n")
        raise
    atomic_write_text(exit_path, f"{return_code}\n")
    return return_code


def _launcher_state(
    *,
    stage: str,
    git: Mapping[str, Any],
    paths: Paths,
    input_audit: Mapping[str, Any],
    jobs: Sequence[Job],
    job_states: Mapping[str, Mapping[str, Any]],
    fail_closed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "v2_parallel_launcher",
        "stage": stage,
        "updated_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "launcher_pid": os.getpid(),
        "git": dict(git),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "input_audit": input_audit,
        "queue_order": [job.job_id for job in jobs],
        "jobs": dict(job_states),
        "fail_closed": fail_closed,
    }


def _job_state(job: Job) -> dict[str, Any]:
    return {
        **asdict(job),
        "argv": list(job.argv),
        "argv_sha256": canonical_json_sha256(list(job.argv)),
        "state": "QUEUED",
    }


def run_launcher(
    *,
    stage: str,
    paths: Paths,
    jobs: Sequence[Job],
    git: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    gpu_indices: Sequence[int],
    minimum_gpu_free_mib: int,
    minimum_dispatch_disk_bytes: int,
    poll_seconds: float,
    launcher_dir: Path,
    gpu_provider: Callable[[], list[GPUInfo]] = query_gpus,
    expected_gpu_uuids: Mapping[int, str] | None = None,
) -> int:
    launcher_dir.mkdir(parents=True, exist_ok=False)
    state_path = launcher_dir / "state.json"
    job_states = {job.job_id: _job_state(job) for job in jobs}
    pending = list(jobs)
    running: dict[str, RunningJob] = {}
    fail_closed = False
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(paths.repository / "src"),
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MUJOCO_GL": "egl",
        }
    )

    while pending or running:
        for job_id, active in list(running.items()):
            polled = active.process.poll()
            if polled is None:
                continue
            active.log_handle.close()
            ended_at = _utc_now()
            free_after = shutil.disk_usage(paths.bundle_root.parent).free
            exit_code = None
            if active.exit_code_path.is_file():
                try:
                    exit_code = int(active.exit_code_path.read_text().strip())
                except ValueError:
                    exit_code = None
            child_pid = active.process.pid
            if active.child_pid_path.is_file():
                try:
                    child_pid = int(active.child_pid_path.read_text().strip())
                except ValueError:
                    child_pid = active.process.pid
            error: str | None = None
            verification: dict[str, Any] | None = None
            if exit_code != polled:
                error = f"exit marker {exit_code!r} differs from process code {polled}"
            elif polled != 0:
                error = f"child exited with {polled}"
            else:
                try:
                    verification = verify_job_output(active.job)
                except Exception as verification_error:
                    error = f"output verification failed: {verification_error}"
            if error is None and stage == "formal":
                try:
                    evidence = _formal_execution_evidence(
                        job=active.job,
                        gpu=active.gpu,
                        pid=child_pid,
                        started_at=active.started_at,
                        ended_at=ended_at,
                        return_code=polled,
                        log_path=active.log_path,
                        paths=paths,
                        git=git,
                        input_audit=input_audit,
                        free_before=active.free_bytes_before,
                        free_after=free_after,
                    )
                    atomic_write_json(
                        Path(active.job.run_dir) / "execution_evidence.json", evidence
                    )
                except Exception as evidence_error:
                    error = f"formal execution evidence failed: {evidence_error}"
            final = {
                "schema_version": 1,
                "job_id": job_id,
                "stage": stage,
                "ended_at_utc": ended_at,
                "return_code": polled,
                "exit_code_marker": exit_code,
                "verification": verification,
                "error": error,
                "log": _audit_file(active.log_path),
                "disk_free_bytes_after": free_after,
            }
            atomic_write_json(active.final_evidence_path, final)
            state = job_states[job_id]
            state.update(
                {
                    "state": "SUCCEEDED" if error is None else "FAILED",
                    "ended_at_utc": ended_at,
                    "return_code": polled,
                    "exit_code_marker": exit_code,
                    "pid": child_pid,
                    "final_evidence": str(active.final_evidence_path),
                    "error": error,
                }
            )
            if error is not None:
                fail_closed = True
            del running[job_id]

        if not fail_closed and pending:
            free_bytes = shutil.disk_usage(paths.bundle_root.parent).free
            if free_bytes < minimum_dispatch_disk_bytes:
                fail_closed = True
                job_states[pending[0].job_id]["error"] = (
                    "dispatch disk threshold is no longer satisfied"
                )
            else:
                inventory = {gpu.index: gpu for gpu in gpu_provider()}
                if expected_gpu_uuids is not None:
                    changed = {
                        index: {
                            "expected": uuid,
                            "actual": inventory.get(index).uuid
                            if index in inventory
                            else None,
                        }
                        for index, uuid in expected_gpu_uuids.items()
                        if index not in inventory or inventory[index].uuid != uuid
                    }
                    if changed:
                        fail_closed = True
                        job_states[pending[0].job_id]["error"] = (
                            f"GPU UUID mapping changed: {changed}"
                        )
                        inventory = {}
                occupied = {active.gpu.index for active in running.values()}
                available = [
                    inventory[index]
                    for index in gpu_indices
                    if index in inventory
                    and index not in occupied
                    and inventory[index].free_memory_mib >= minimum_gpu_free_mib
                    and not inventory[index].compute_pids
                ]
                for gpu in available:
                    if not pending:
                        break
                    job = pending.pop(0)
                    job_dir = launcher_dir / "jobs" / job.job_id
                    job_dir.mkdir(parents=True, exist_ok=False)
                    log_path = job_dir / "stdout_stderr.log"
                    exit_path = job_dir / "exit_code.txt"
                    child_pid_path = job_dir / "child_pid.txt"
                    start_path = job_dir / "start_evidence.json"
                    final_path = job_dir / "final_evidence.json"
                    log_handle = log_path.open("wb")
                    child_environment = environment.copy()
                    child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
                    wrapper = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_child",
                        str(exit_path),
                        str(child_pid_path),
                        "--",
                        *job.argv,
                    ]
                    started_at = _utc_now()
                    try:
                        process = subprocess.Popen(
                            wrapper,
                            cwd=paths.repository,
                            env=child_environment,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    except Exception as launch_error:
                        log_handle.close()
                        fail_closed = True
                        job_states[job.job_id].update(
                            {
                                "state": "FAILED",
                                "started_at_utc": started_at,
                                "error": f"child launch failed: {launch_error}",
                            }
                        )
                        atomic_write_json(
                            final_path,
                            {
                                "schema_version": 1,
                                "job_id": job.job_id,
                                "stage": stage,
                                "ended_at_utc": _utc_now(),
                                "return_code": None,
                                "error": f"child launch failed: {launch_error}",
                            },
                        )
                        break
                    start = {
                        "schema_version": 1,
                        "job_id": job.job_id,
                        "stage": stage,
                        "pid": process.pid,
                        "pid_kind": "wrapper",
                        "child_pid_path": str(child_pid_path),
                        "started_at_utc": started_at,
                        "argv": list(job.argv),
                        "argv_sha256": canonical_json_sha256(list(job.argv)),
                        "cwd": str(paths.repository),
                        "git_revision": git["revision"],
                        "git_clean": True,
                        "gpu": asdict(gpu),
                        "log_path": str(log_path),
                        "exit_code_path": str(exit_path),
                        "disk_free_bytes_before": free_bytes,
                    }
                    atomic_write_json(start_path, start)
                    running[job.job_id] = RunningJob(
                        job=job,
                        gpu=gpu,
                        process=process,
                        log_handle=log_handle,
                        log_path=log_path,
                        exit_code_path=exit_path,
                        child_pid_path=child_pid_path,
                        start_evidence_path=start_path,
                        final_evidence_path=final_path,
                        started_at=started_at,
                        free_bytes_before=free_bytes,
                    )
                    job_states[job.job_id].update(
                        {
                            "state": "RUNNING",
                            "pid": process.pid,
                            "gpu": asdict(gpu),
                            "started_at_utc": started_at,
                            "start_evidence": str(start_path),
                        }
                    )

        if fail_closed:
            for job in pending:
                job_states[job.job_id]["state"] = "BLOCKED"
                job_states[job.job_id].setdefault(
                    "error", "fail-closed after an earlier job failure"
                )
        atomic_write_json(
            state_path,
            _launcher_state(
                stage=stage,
                git=git,
                paths=paths,
                input_audit=input_audit,
                jobs=jobs,
                job_states=job_states,
                fail_closed=fail_closed,
            ),
        )
        if fail_closed and not running:
            break
        if pending or running:
            time.sleep(poll_seconds)
    return 1 if fail_closed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("smoke1", "smoke2", "formal", "eval")
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split-indices")
    parser.add_argument("--neighbor-index")
    parser.add_argument("--v1-root")
    parser.add_argument("--bundle-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs=5, type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=28 * 1024)
    parser.add_argument("--minimum-initial-disk-gib", type=int)
    parser.add_argument("--minimum-dispatch-disk-gib", type=int, default=15)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--formal-resume", choices=("never", "required"), default="never"
    )
    parser.add_argument("--launch-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _paths(args: argparse.Namespace, repository: Path, short_revision: str) -> Paths:
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    dataset = (
        Path(
            args.dataset
            or artifact_root / "data" / "lewm-cube" / "cube_single_expert_jpeg100.lance"
        )
        .expanduser()
        .resolve()
    )
    split = (
        Path(
            args.split_indices
            or artifact_root
            / "outputs"
            / "actor_free_td_lewm_cdfg1_eac18ce_20260829"
            / "shared"
            / "split_indices_seed3072_numsteps19.npz"
        )
        .expanduser()
        .resolve()
    )
    neighbor = (
        Path(
            args.neighbor_index
            or artifact_root
            / "outputs"
            / "actor_free_td_lewm_cdfg1_eac18ce_20260829"
            / "shared"
            / "g1_neighbor_index_k8"
        )
        .expanduser()
        .resolve()
    )
    v1_root = (
        Path(
            args.v1_root
            or artifact_root / "outputs" / "actor_free_td_lewm_v1_cg3_3c4e62e_20260830"
        )
        .expanduser()
        .resolve()
    )
    bundle = (
        Path(
            args.bundle_root
            or artifact_root / "outputs" / f"actor_free_td_lewm_v2_cg3_{short_revision}"
        )
        .expanduser()
        .resolve()
    )
    return Paths(
        repository=repository,
        artifact_root=artifact_root,
        dataset=dataset,
        dataset_manifest=Path(f"{dataset}.manifest.json"),
        split_indices=split,
        neighbor_index=neighbor,
        v1_root=v1_root,
        bundle_root=bundle,
        smoke_root=bundle / "smoke",
        formal_root=bundle / "formal",
        evaluation_root=bundle / "evaluations",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(args.gpus)) != 5:
        raise SystemExit("--gpus must name five distinct GPU indices.")
    if args.stage != "formal" and args.formal_resume != "never":
        raise SystemExit("--formal-resume is only valid for --stage formal.")
    repository = Path(__file__).resolve().parents[1]
    git = audited_git_state(repository)
    paths = _paths(args, repository, git["short_revision"])
    python = Path(args.python).expanduser().resolve()
    jobs = build_jobs(
        stage=args.stage,
        paths=paths,
        python=str(python),
        formal_resume=args.formal_resume,
    )
    input_audit = audit_inputs(paths, jobs)
    input_audit["runtime"] = audit_python_runtime(python)
    validate_stage_outputs(args.stage, jobs, args.formal_resume)
    disk_parent = paths.bundle_root.parent
    if not disk_parent.is_dir():
        raise FileNotFoundError(disk_parent)
    disk_free = shutil.disk_usage(disk_parent).free
    default_initial_gib = (
        45 if args.stage == "formal" and args.formal_resume == "never" else 15
    )
    initial_required = (
        args.minimum_initial_disk_gib
        if args.minimum_initial_disk_gib is not None
        else default_initial_gib
    ) * 1024**3
    if disk_free < initial_required:
        raise RuntimeError(
            f"Initial free disk {disk_free} is below {initial_required} bytes."
        )
    inventory = {gpu.index: gpu for gpu in query_gpus()}
    missing = set(args.gpus) - inventory.keys()
    if missing:
        raise RuntimeError(f"Requested GPUs do not exist: {sorted(missing)}")
    plan = {
        "schema_version": 1,
        "dry_run": bool(args.dry_run),
        "stage": args.stage,
        "git": git,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "jobs": [{**asdict(job), "argv": list(job.argv)} for job in jobs],
        "input_audit": input_audit,
        "gpu_inventory": [asdict(inventory[index]) for index in args.gpus],
        "disk_free_bytes": disk_free,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    launch_id = args.launch_id or (
        f"{args.stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    )
    launcher_dir = paths.bundle_root / "launcher" / launch_id
    paths.bundle_root.mkdir(parents=True, exist_ok=True)
    if launcher_dir.exists():
        raise FileExistsError(f"Launcher id already exists: {launcher_dir}")
    lock_path = paths.bundle_root / ".launcher.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another V2 launcher owns this bundle.") from error
        atomic_write_json(launcher_dir / "plan.json", plan)
        return run_launcher(
            stage=args.stage,
            paths=paths,
            jobs=jobs,
            git=git,
            input_audit=input_audit,
            gpu_indices=args.gpus,
            minimum_gpu_free_mib=args.minimum_gpu_free_mib,
            minimum_dispatch_disk_bytes=args.minimum_dispatch_disk_gib * 1024**3,
            poll_seconds=args.poll_seconds,
            launcher_dir=launcher_dir / "run",
            expected_gpu_uuids={index: inventory[index].uuid for index in args.gpus},
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(main())
