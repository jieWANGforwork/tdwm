#!/usr/bin/env python3
"""Fail-closed acceptance for the six formal Actor-Free TD-JEPA V1 trainings.

The command validates the completed training artifacts in place and atomically
publishes ``training_acceptance.json``.  A missing launcher exit code is the
only condition that can produce ``PASS_WITH_WARNINGS``; malformed or incomplete
training artifacts always produce ``FAIL``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
TRAINING_SEED = 3072
TRAINING_EPOCHS = 10
TRAINING_STEPS = 127_960
OPTIMIZER_STEPS_PER_EPOCH = 12_796
TRAINING_COMMIT = "3c4e62ef2ab72387536433f27ef11bce75477e7e"
PRETRAINED_WORLD_MODEL_SHA256 = (
    "198c468cadb63655066c968726cef69e36fe5682fcaec55620dd610a8b75e257"
)
# Canonical hash of world_model.action_encoder.state_dict(): the
# ``action_encoder.`` world-model prefix is removed before state_dict_sha256.
EXPECTED_ACTION_ENCODER_SHA256 = (
    "2657b55140013b4b071cd8cdea63f1eac5c65c498d55331c7499744ef31a9cd3"
)
TRAINING_PROTOCOL_SHA256 = {
    "c": "14e9b00346bab0e7b5e527544968a072987fb0bbfb752d3a9f6050c2f4f32b6b",
    "d": "cfd7af7852d8021bb59f30920577506f773ba9321482d715f1f00370e6781752",
    "f": "ba4f68cfec17d0a893b94b7d013f118536ee3e32f5aae5a46b6a90301b284750",
    "g1": "84b1c5bb9a3098eae7967e72f2b00e4f2c651fe12049532ac8a181153f2a293e",
    "g2": "409d62cfee01e5325e8220e3cfa26b8c16f5659afa314f71212f1d5bf8f844c4",
    "g3": "56dc578a7a0148d50f23bfd53bde311913b9598f3c241212f392d4417134fea1",
}


class AcceptanceAudit:
    """Accumulate all failures while preserving one complete evidence record."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def require(self, condition: bool, message: str) -> bool:
        if not condition:
            self.fail(message)
            return False
        return True


def parse_pid_map(value: str) -> dict[str, int]:
    """Parse ``c=PID,...,g3=PID`` and reject ambiguous formal mappings."""

    if not value.strip():
        return {}
    result: dict[str, int] = {}
    for item in value.split(","):
        try:
            variant, raw_pid = item.split("=", 1)
            variant = variant.strip()
            pid = int(raw_pid)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                "--pids must use c=PID,d=PID,f=PID,g1=PID,g2=PID,g3=PID"
            ) from error
        if variant in result or variant not in VARIANTS or pid <= 0:
            raise argparse.ArgumentTypeError(
                "--pids must contain each formal variant once with a positive PID"
            )
        result[variant] = pid
    if set(result) != set(VARIANTS):
        raise argparse.ArgumentTypeError("--pids must contain exactly c,d,f,g1,g2,g3")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(
    path: Path,
    *,
    audit: AcceptanceAudit,
    context: str,
) -> dict[str, Any]:
    if not audit.require(path.is_file(), f"{context}: missing {path}"):
        return {}
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        audit.fail(f"{context}: cannot parse JSON: {error}")
        return {}
    if not isinstance(value, dict):
        audit.fail(f"{context}: top level is not an object")
        return {}
    return value


def _load_checkpoint(
    path: Path,
    *,
    audit: AcceptanceAudit,
    context: str,
) -> dict[str, Any]:
    if not audit.require(path.is_file(), f"{context}: missing {path}"):
        return {}
    if not audit.require(path.stat().st_size > 0, f"{context}: empty checkpoint"):
        return {}
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    except Exception as error:
        audit.fail(f"{context}: torch.load failed: {error}")
        return {}
    if not isinstance(value, dict):
        audit.fail(f"{context}: checkpoint top level is not a mapping")
        return {}
    return value


def _mapping(
    value: Any,
    *,
    audit: AcceptanceAudit,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        audit.fail(f"{context}: expected a mapping")
        return {}
    return value


def _finite_tensors(
    value: Any,
    *,
    audit: AcceptanceAudit,
    context: str,
) -> dict[str, int]:
    stats = {"tensor_count": 0, "floating_tensor_count": 0, "tensor_numel": 0}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            stats["tensor_count"] += 1
            stats["tensor_numel"] += int(item.numel())
            if item.is_floating_point() or item.is_complex():
                stats["floating_tensor_count"] += 1
                tensor = item.detach()
                if tensor.layout != torch.strided:
                    tensor = tensor.to_dense()
                flattened = tensor.reshape(-1)
                for start in range(0, flattened.numel(), 1_000_000):
                    chunk = flattened[start : start + 1_000_000]
                    if not bool(torch.isfinite(chunk).all()):
                        audit.fail(f"{path}: contains NaN or infinity")
                        break
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, context)
    return stats


def _prefixed_state(
    state: Mapping[str, Any],
    prefix: str,
    *,
    audit: AcceptanceAudit,
    context: str,
) -> dict[str, torch.Tensor]:
    """Return submodule-local state keys, which defines the canonical hash."""

    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not audit.require(bool(selected), f"{context}: no {prefix!r} state"):
        return {}
    if not audit.require(
        all(isinstance(value, torch.Tensor) for value in selected.values()),
        f"{context}: {prefix!r} state contains a non-tensor value",
    ):
        return {}
    return dict(selected)


def _canonical_action_encoder_hash(
    world_state: Mapping[str, Any],
    hash_function: Callable[[dict[str, torch.Tensor]], str],
    *,
    audit: AcceptanceAudit,
    context: str,
) -> str | None:
    """Hash exactly the submodule-local action-encoder state dictionary."""

    action_state = _prefixed_state(
        world_state,
        "action_encoder.",
        audit=audit,
        context=context,
    )
    return hash_function(action_state) if action_state else None


def _exact_fields(
    mapping: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    audit: AcceptanceAudit,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        if mapping.get(key) != expected_value:
            audit.fail(
                f"{context}.{key}: expected {expected_value!r}, "
                f"got {mapping.get(key)!r}"
            )


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _metric_coverage(rows: Sequence[Mapping[str, float]], name: str) -> set[int]:
    return {
        int(row["epoch"])
        for row in rows
        if name in row and "epoch" in row and row["epoch"].is_integer()
    }


def _audit_metrics(
    run_dir: Path,
    *,
    variant: str,
    audit: AcceptanceAudit,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    paths = sorted(run_dir.glob("metrics/version_*/metrics.csv"))
    if not audit.require(bool(paths), f"{variant}.metrics: no metrics CSV"):
        return [], [], {"row_count": 0, "max_step": None}

    rows: list[dict[str, float]] = []
    files: list[dict[str, Any]] = []
    steps: list[int] = []
    for path in paths:
        files.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None:
                    audit.fail(f"{variant}.metrics: {path} has no CSV header")
                    continue
                for line_number, raw in enumerate(reader, start=2):
                    parsed: dict[str, float] = {}
                    for key, text in raw.items():
                        if key is None or text is None or not text.strip():
                            continue
                        try:
                            number = float(text)
                        except ValueError:
                            audit.fail(
                                f"{variant}.metrics: {path}:{line_number} "
                                f"{key} is not numeric"
                            )
                            continue
                        if not math.isfinite(number):
                            audit.fail(
                                f"{variant}.metrics: {path}:{line_number} "
                                f"{key} is non-finite"
                            )
                        parsed[key] = number
                    if "step" in parsed:
                        if parsed["step"].is_integer():
                            steps.append(int(parsed["step"]))
                        else:
                            audit.fail(f"{variant}.metrics: non-integral step")
                    if parsed:
                        rows.append(parsed)
        except Exception as error:
            audit.fail(f"{variant}.metrics: cannot read {path}: {error}")

    audit.require(
        bool(steps) and max(steps) == TRAINING_STEPS - 1,
        f"{variant}.metrics: final zero-based step must equal 127959",
    )
    common = [
        "loss",
        "base_td_loss",
        "method_td_loss",
        "goal_task_fraction",
        "random_task_fraction",
        "terminal_fraction",
        "td_pairs",
        "td_prediction_mean",
        "td_target_mean",
        "action_embedding_mean",
    ]
    diagnostics = {
        "c": ["goal_projection_loss", "goal_score_residual_mean"],
        "d": ["target_goal_score_mean", "weight_mean", "weight_std"],
        "f": ["same_future_advantage_mean", "weight_mean", "weight_std"],
        "g1": [
            "neighbor_objective_available",
            "neighbor_advantage_mean",
            "weight_mean",
        ],
        "g2": ["prefix_mean_advantage_mean", "weight_mean", "weight_std"],
        "g3": ["prefix_marginal_advantage_mean", "weight_mean", "weight_std"],
    }[variant]
    validation_diagnostics = (
        diagnostics if variant != "g1" else ["neighbor_objective_available"]
    )
    required = [f"train/{name}_epoch" for name in common + diagnostics]
    required.extend(f"validation/{name}" for name in common + validation_diagnostics)
    expected_epochs = set(range(TRAINING_EPOCHS))
    coverage: dict[str, list[int]] = {}
    for name in required + ["train/loss_step"]:
        observed = _metric_coverage(rows, name)
        coverage[name] = sorted(observed)
        missing = expected_epochs - observed
        audit.require(
            not missing,
            f"{variant}.metrics: {name} missing epochs {sorted(missing)}",
        )

    train_relations: set[int] = set()
    validation_relations: set[int] = set()
    aggregates: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if "epoch" not in row or not row["epoch"].is_integer():
            continue
        epoch = int(row["epoch"])
        for name in (
            "train/loss_epoch",
            "train/base_td_loss_epoch",
            "validation/loss",
        ):
            if name in row:
                aggregates.setdefault((epoch, name), []).append(row[name])

        if all(
            name in row for name in ("train/loss_epoch", "train/method_td_loss_epoch")
        ):
            train_relations.add(epoch)
            audit.require(
                math.isclose(
                    row["train/loss_epoch"],
                    row["train/method_td_loss_epoch"],
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ),
                f"{variant}.metrics: epoch {epoch} train loss != method loss",
            )
        if all(name in row for name in ("validation/loss", "validation/base_td_loss")):
            validation_relations.add(epoch)
            audit.require(
                math.isclose(
                    row["validation/loss"],
                    row["validation/base_td_loss"],
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ),
                f"{variant}.metrics: epoch {epoch} validation loss != base TD",
            )

        for stage, suffix in (("train", "_epoch"), ("validation", "")):
            goal = row.get(f"{stage}/goal_task_fraction{suffix}")
            random = row.get(f"{stage}/random_task_fraction{suffix}")
            terminal = row.get(f"{stage}/terminal_fraction{suffix}")
            if goal is not None and random is not None:
                audit.require(
                    0.0 <= goal <= 1.0
                    and 0.0 <= random <= 1.0
                    and math.isclose(goal + random, 1.0, abs_tol=1e-5),
                    f"{variant}.metrics: epoch {epoch} invalid {stage} task mix",
                )
            if terminal is not None:
                audit.require(
                    0.0 <= terminal <= 1.0,
                    f"{variant}.metrics: epoch {epoch} invalid terminal fraction",
                )
        for name, number in row.items():
            if "loss" in name:
                audit.require(
                    number >= -1e-8,
                    f"{variant}.metrics: epoch {epoch} negative {name}",
                )

    audit.require(
        expected_epochs <= train_relations,
        f"{variant}.metrics: incomplete train loss semantics",
    )
    audit.require(
        expected_epochs <= validation_relations,
        f"{variant}.metrics: incomplete validation loss semantics",
    )
    for (epoch, name), values in aggregates.items():
        if values:
            audit.require(
                all(
                    math.isclose(
                        value,
                        values[-1],
                        rel_tol=1e-8,
                        abs_tol=1e-10,
                    )
                    for value in values
                ),
                f"{variant}.metrics: conflicting {name} at epoch {epoch}",
            )

    if variant == "g1":
        for row in rows:
            train_available = row.get("train/neighbor_objective_available_epoch")
            validation_available = row.get("validation/neighbor_objective_available")
            if train_available is not None:
                audit.require(
                    math.isclose(train_available, 1.0, abs_tol=1e-8),
                    "g1.metrics: train neighbor objective availability is not 1",
                )
            if validation_available is not None:
                audit.require(
                    math.isclose(validation_available, 0.0, abs_tol=1e-8),
                    "g1.metrics: validation neighbor objective availability is not 0",
                )

    return (
        files,
        list(range(1, TRAINING_EPOCHS + 1)),
        {
            "row_count": len(rows),
            "max_step": max(steps) if steps else None,
            "coverage_zero_based": coverage,
        },
    )


def _read_persisted_exit_code(
    run_dir: Path,
    *,
    audit: AcceptanceAudit,
    variant: str,
) -> tuple[int | None, str | None]:
    for name in (
        "exit_code",
        "exit_code.txt",
        "process_exit_code.txt",
        ".exit_code",
    ):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            return int(path.read_text().strip()), str(path)
        except Exception as error:
            audit.fail(f"{variant}.process: invalid exit marker {path}: {error}")
            return None, str(path)
    return None, None


def _audit_process(
    run_dir: Path,
    *,
    variant: str,
    pid: int | None,
    audit: AcceptanceAudit,
) -> dict[str, Any]:
    code, source = _read_persisted_exit_code(run_dir, audit=audit, variant=variant)
    if source is not None:
        if code not in (0, None):
            audit.fail(f"{variant}.process: persisted exit code is {code}")
        return {
            "pid": pid,
            "state": "exited",
            "exit_code": code,
            "exit_code_evidence": source,
        }
    if pid is None:
        audit.warn(f"{variant}.process: launcher PID/exit code was not persisted")
        return {
            "pid": None,
            "state": "completed",
            "exit_code": None,
            "exit_code_evidence": "unavailable_not_persisted",
            "observed_state": "pid_not_recorded",
        }

    proc = Path("/proc") / str(pid)
    if not proc.exists():
        audit.warn(f"{variant}.process: PID {pid} was reaped without an exit marker")
        return {
            "pid": pid,
            "state": "completed",
            "exit_code": None,
            "exit_code_evidence": "unavailable_process_reaped_without_marker",
            "observed_state": "gone",
        }
    try:
        stat_text = (proc / "stat").read_text()
        fields = stat_text[stat_text.rfind(")") + 2 :].split()
        state = fields[0]
        command = (
            (proc / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except Exception as error:
        audit.warn(f"{variant}.process: PID {pid} inspection unavailable: {error}")
        return {
            "pid": pid,
            "state": "completed",
            "exit_code": None,
            "exit_code_evidence": "unavailable_proc_inspection",
            "observed_state": "unreadable",
        }

    if state == "Z":
        raw_status = int(fields[49]) if len(fields) > 49 else None
        exit_code = None
        if raw_status is not None:
            try:
                exit_code = os.waitstatus_to_exitcode(raw_status)
            except ValueError:
                audit.warn(f"{variant}.process: cannot decode wait status {raw_status}")
        else:
            audit.warn(f"{variant}.process: zombie PID {pid} has no exit status")
        if exit_code not in (0, None):
            audit.fail(f"{variant}.process: PID {pid} exited with {exit_code}")
        return {
            "pid": pid,
            "state": "exited",
            "exit_code": exit_code,
            "exit_code_evidence": "/proc/PID/stat field 52",
            "observed_state": "zombie",
        }

    expected_process = (
        str(run_dir) in command
        or f"actor_free_td_lewm_v1_{variant}" in command
        or f"v1_{variant}_cube_train" in command
    )
    if expected_process:
        audit.fail(f"{variant}.process: PID {pid} is still active ({state})")
        return {
            "pid": pid,
            "state": "running",
            "exit_code": None,
            "exit_code_evidence": "process_active",
            "command": command,
        }

    audit.warn(f"{variant}.process: PID {pid} was reused; exit code unavailable")
    return {
        "pid": pid,
        "state": "completed",
        "exit_code": None,
        "exit_code_evidence": "unavailable_pid_reused_without_marker",
        "observed_state": "pid_reused",
        "command": command,
    }


def _optimizer_audit(
    checkpoint: Mapping[str, Any],
    predictor_state: Mapping[str, torch.Tensor],
    *,
    variant: str,
    audit: AcceptanceAudit,
) -> dict[str, Any]:
    optimizers = checkpoint.get("optimizer_states")
    if not audit.require(
        isinstance(optimizers, list) and len(optimizers) == 1,
        f"{variant}.last: expected exactly one optimizer",
    ):
        return {}
    optimizer = _mapping(
        optimizers[0], audit=audit, context=f"{variant}.last.optimizer"
    )
    states = _mapping(
        optimizer.get("state"),
        audit=audit,
        context=f"{variant}.last.optimizer.state",
    )
    groups = optimizer.get("param_groups")
    if not audit.require(
        isinstance(groups, list) and len(groups) == 1,
        f"{variant}.last: optimizer must have one parameter group",
    ):
        groups = []
    parameter_ids = groups[0].get("params", []) if groups else []
    audit.require(
        len(parameter_ids) == len(predictor_state),
        f"{variant}.last: optimizer parameter count differs from predictor",
    )
    audit.require(
        set(parameter_ids) == set(states),
        f"{variant}.last: optimizer parameter IDs and states differ",
    )
    expected_shapes = Counter(tuple(value.shape) for value in predictor_state.values())
    observed_shapes: Counter[tuple[int, ...]] = Counter()
    for parameter_id, state in states.items():
        state_mapping = _mapping(
            state,
            audit=audit,
            context=f"{variant}.last.optimizer.state.{parameter_id}",
        )
        first = state_mapping.get("exp_avg")
        second = state_mapping.get("exp_avg_sq")
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            audit.fail(f"{variant}.last: AdamW state {parameter_id} lacks both moments")
            continue
        audit.require(
            first.shape == second.shape,
            f"{variant}.last: AdamW moment shapes differ for {parameter_id}",
        )
        observed_shapes[tuple(first.shape)] += 1
    audit.require(
        observed_shapes == expected_shapes,
        f"{variant}.last: optimizer state is not predictor-only by tensor shape",
    )

    schedulers = checkpoint.get("lr_schedulers")
    scheduler_last_epoch = None
    if audit.require(
        isinstance(schedulers, list) and len(schedulers) == 1,
        f"{variant}.last: expected exactly one scheduler",
    ):
        scheduler = _mapping(
            schedulers[0],
            audit=audit,
            context=f"{variant}.last.scheduler",
        )
        scheduler_last_epoch = scheduler.get("last_epoch")
        audit.require(
            scheduler_last_epoch == TRAINING_STEPS,
            f"{variant}.last: scheduler did not reach step 127960",
        )
    return {
        "optimizer_count": len(optimizers),
        "optimizer_parameter_tensors": len(parameter_ids),
        "scheduler_last_epoch": scheduler_last_epoch,
    }


def _git_revision(repo_root: Path, *, audit: AcceptanceAudit) -> str | None:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        audit.fail(f"repo: cannot resolve git revision: {error}")
        return None
    audit.require(
        revision == TRAINING_COMMIT,
        f"repo: HEAD {revision!r} is not formal training commit {TRAINING_COMMIT}",
    )
    return revision


def _directory_bytes(root: Path, *, excluded: Path | None) -> int:
    total = 0
    if not root.is_dir():
        return total
    excluded_resolved = excluded.resolve() if excluded is not None else None
    for path in root.rglob("*"):
        try:
            if path.is_file() and (
                excluded_resolved is None or path.resolve() != excluded_resolved
            ):
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def audit_training(
    *,
    repo_root: Path,
    output_root: Path,
    pids: Mapping[str, int] | None = None,
    minimum_free_bytes: int = 5 * 2**30,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Validate six completed formal runs and return archive-ready evidence."""

    repo_root = repo_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_json = output_json.expanduser().resolve() if output_json else None
    pids = dict(pids or {})
    audit = AcceptanceAudit()
    audit.require(repo_root.is_dir(), f"repo: missing directory {repo_root}")
    audit.require(
        (repo_root / "src/tdwm/training/frozen_actor_free_td_v1.py").is_file(),
        f"repo: {repo_root} is not the V1 worktree",
    )
    _git_revision(repo_root, audit=audit)
    audit.require(output_root.is_dir(), f"output: missing training root {output_root}")

    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from tdwm.training.clean_aligned_lewm import state_dict_sha256
        from tdwm.training.frozen_actor_free_td_v1 import (
            V1_SPECS,
            _canonical_sha256,
            validate_actor_free_td_lewm_v1_training_protocol,
        )
    except Exception as error:
        audit.fail(f"repo: cannot import V1 acceptance dependencies: {error}")
        state_dict_sha256 = None
        V1_SPECS = {}
        _canonical_sha256 = None
        validate_actor_free_td_lewm_v1_training_protocol = None

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "repo_root": str(repo_root),
        "training_commit": TRAINING_COMMIT,
        "seed": TRAINING_SEED,
        "expected_epoch": TRAINING_EPOCHS,
        "expected_global_step": TRAINING_STEPS,
        "expected_action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
        "variants": {},
    }
    world_hashes: dict[str, str] = {}
    latent_hashes: dict[str, str] = {}
    split_hashes: dict[str, tuple[Any, Any]] = {}

    for variant in VARIANTS:
        print(
            f"[V1 training acceptance] checking {variant}",
            file=sys.stderr,
            flush=True,
        )
        method = f"actor_free_td_lewm_v1_{variant}"
        run_dir = output_root / variant / f"seed_{TRAINING_SEED}"
        result_path = run_dir / "training_result.json"
        manifest_path = run_dir / "training_manifest.json"
        checkpoint_path = (
            run_dir
            / "checkpoints"
            / method
            / variant
            / f"epoch_{TRAINING_EPOCHS:02d}.pt"
        )
        last_path = run_dir / "checkpoints" / "lightning" / "last.ckpt"
        result = _read_json(
            result_path, audit=audit, context=f"{variant}.training_result"
        )
        manifest = _read_json(
            manifest_path, audit=audit, context=f"{variant}.training_manifest"
        )
        _exact_fields(
            result,
            {
                "method": method,
                "method_family": "actor_free_td_lewm_v1",
                "variant": variant,
                "implementation_version": "v1",
                "seed": TRAINING_SEED,
                "final_epoch": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
                "pretrained_world_model_sha256": PRETRAINED_WORLD_MODEL_SHA256,
            },
            audit=audit,
            context=f"{variant}.training_result",
        )
        audit.require(
            _same_path(result.get("run_dir"), run_dir),
            f"{variant}.training_result: run_dir differs from formal run",
        )
        audit.require(
            _same_path(result.get("deployment_checkpoint"), checkpoint_path),
            f"{variant}.training_result: wrong deployment checkpoint",
        )
        audit.require(
            _same_path(result.get("last_checkpoint"), last_path),
            f"{variant}.training_result: wrong Lightning checkpoint",
        )
        _exact_fields(
            manifest,
            {
                "method": method,
                "method_family": "actor_free_td_lewm_v1",
                "variant": variant,
                "implementation_version": "v1",
                "objective_version": 0,
                "deployment_checkpoint_version": 1,
                "seed": TRAINING_SEED,
            },
            audit=audit,
            context=f"{variant}.training_manifest",
        )

        protocol = _mapping(
            manifest.get("protocol"),
            audit=audit,
            context=f"{variant}.training_manifest.protocol",
        )
        if validate_actor_free_td_lewm_v1_training_protocol is not None:
            try:
                validate_actor_free_td_lewm_v1_training_protocol(
                    protocol, spec=V1_SPECS[variant]
                )
            except Exception as error:
                audit.fail(f"{variant}.protocol violates formal locks: {error}")
        protocol_hash = (
            _canonical_sha256(protocol) if _canonical_sha256 is not None else None
        )
        audit.require(
            protocol_hash == TRAINING_PROTOCOL_SHA256[variant],
            f"{variant}.protocol differs from the complete locked YAML",
        )
        audit.require(
            manifest.get("protocol_sha256")
            == protocol_hash
            == result.get("protocol_sha256"),
            f"{variant}: protocol hashes disagree",
        )

        runtime = _mapping(
            manifest.get("runtime"),
            audit=audit,
            context=f"{variant}.training_manifest.runtime",
        )
        training = _mapping(
            manifest.get("training"),
            audit=audit,
            context=f"{variant}.training_manifest.training",
        )
        model = _mapping(
            manifest.get("model"),
            audit=audit,
            context=f"{variant}.training_manifest.model",
        )
        _exact_fields(
            runtime,
            {"stable_worldmodel": "0.1.1", "tdwm_git_revision": TRAINING_COMMIT},
            audit=audit,
            context=f"{variant}.runtime",
        )
        _exact_fields(
            training,
            {
                "formal_optimizer_steps": TRAINING_STEPS,
                "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
                "configured_optimizer_steps": TRAINING_STEPS,
                "validation_skipped": False,
                "data_source": "frozen_latent_store",
                "sampling_unit": "transition",
                "train_sampling": "random_with_replacement",
                "world_model_visual_encode_during_training": False,
                "shared_action_encoder_forward_during_training": True,
            },
            audit=audit,
            context=f"{variant}.training",
        )
        audit.require(
            isinstance(training.get("validation_batches"), int)
            and training["validation_batches"] > 0,
            f"{variant}.training: validation_batches must be positive",
        )
        _exact_fields(
            model,
            {
                "lewm_parameters": 18_034_628,
                "trainable_lewm_parameters": 0,
                "predictor_parameters": 379_072,
            },
            audit=audit,
            context=f"{variant}.model",
        )
        initialization = _mapping(
            model.get("initialization"),
            audit=audit,
            context=f"{variant}.model.initialization",
        )
        _exact_fields(
            initialization,
            {
                "strategy": "frozen_pretrained_lewm",
                "source_method": "lewm",
                "source_seed": TRAINING_SEED,
                "source_epoch": TRAINING_EPOCHS,
                "source_checkpoint_sha256": PRETRAINED_WORLD_MODEL_SHA256,
                "frozen": True,
            },
            audit=audit,
            context=f"{variant}.model.initialization",
        )

        cuda_device = runtime.get("cuda_device")
        peak_memory = result.get("peak_cuda_memory_bytes")
        audit.require(
            (cuda_device is None and peak_memory is None)
            or (
                isinstance(cuda_device, str)
                and bool(cuda_device)
                and isinstance(peak_memory, int)
                and peak_memory > 0
            ),
            f"{variant}: CUDA provenance is partial or invalid",
        )

        latent = _mapping(
            manifest.get("frozen_latent_store"),
            audit=audit,
            context=f"{variant}.frozen_latent_store",
        )
        latent_hash = latent.get("manifest_sha256")
        if isinstance(latent_hash, str):
            latent_hashes[variant] = latent_hash
        audit.require(
            result.get("frozen_latent_store_manifest_sha256") == latent_hash,
            f"{variant}: frozen latent-store hashes disagree",
        )
        dataset = _mapping(
            manifest.get("dataset"),
            audit=audit,
            context=f"{variant}.dataset",
        )
        split = _mapping(
            dataset.get("split"),
            audit=audit,
            context=f"{variant}.dataset.split",
        )
        split_hashes[variant] = (
            split.get("train_indices_sha256"),
            split.get("validation_indices_sha256"),
        )
        neighbor = manifest.get("neighbor_index")
        if variant == "g1":
            neighbor_mapping = _mapping(
                neighbor,
                audit=audit,
                context="g1.neighbor_index",
            )
            audit.require(
                result.get("neighbor_index_manifest_sha256")
                == neighbor_mapping.get("manifest_sha256"),
                "g1: neighbor-index hashes disagree",
            )
        else:
            audit.require(
                neighbor is None
                and result.get("neighbor_index_manifest_sha256") is None,
                f"{variant}: only G1 may bind a neighbor index",
            )

        metric_files, metric_epochs, metric_audit = _audit_metrics(
            run_dir, variant=variant, audit=audit
        )
        checkpoint = _load_checkpoint(
            checkpoint_path, audit=audit, context=f"{variant}.epoch10"
        )
        last = _load_checkpoint(last_path, audit=audit, context=f"{variant}.last")
        _exact_fields(
            checkpoint,
            {
                "method": method,
                "method_family": "actor_free_td_lewm_v1",
                "variant": variant,
                "implementation_version": "v1",
                "objective_version": 0,
                "deployment_checkpoint_version": 1,
                "epoch": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
            },
            audit=audit,
            context=f"{variant}.epoch10",
        )
        expected_checkpoint_keys = {
            "method",
            "method_family",
            "variant",
            "implementation_version",
            "objective_version",
            "deployment_checkpoint_version",
            "epoch",
            "global_step",
            "world_model_state_dict",
            "world_model_config",
            "pretrained_world_model_provenance",
            "predictor_state_dict",
            "target_predictor_state_dict",
            "predictor_config",
        }
        audit.require(
            set(checkpoint) == expected_checkpoint_keys,
            f"{variant}.epoch10: deployment checkpoint schema differs",
        )
        world_state = _mapping(
            checkpoint.get("world_model_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.world_model_state_dict",
        )
        predictor_state = _mapping(
            checkpoint.get("predictor_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.predictor_state_dict",
        )
        target_state = _mapping(
            checkpoint.get("target_predictor_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.target_predictor_state_dict",
        )
        checkpoint_finite = _finite_tensors(
            {
                "world": world_state,
                "predictor": predictor_state,
                "target_predictor": target_state,
            },
            audit=audit,
            context=f"{variant}.epoch10",
        )
        action_hash = (
            _canonical_action_encoder_hash(
                world_state,
                state_dict_sha256,
                audit=audit,
                context=f"{variant}.epoch10.world_model_state_dict",
            )
            if state_dict_sha256 is not None
            else None
        )
        audit.require(
            action_hash == EXPECTED_ACTION_ENCODER_SHA256,
            f"{variant}: action encoder hash {action_hash!r} is not canonical",
        )
        audit.require(
            set(predictor_state) == set(target_state),
            f"{variant}: online and EMA target predictor keys differ",
        )
        for key in set(predictor_state) & set(target_state):
            left, right = predictor_state[key], target_state[key]
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                audit.require(
                    left.shape == right.shape and left.dtype == right.dtype,
                    f"{variant}: online/target predictor differs at {key}",
                )
        predictor_parameters = sum(
            int(value.numel())
            for value in predictor_state.values()
            if isinstance(value, torch.Tensor)
        )
        audit.require(
            predictor_parameters == 379_072,
            f"{variant}: predictor parameter count is not 379072",
        )
        for key in set(predictor_state) | set(target_state):
            audit.require(
                all(
                    forbidden not in key.lower()
                    for forbidden in ("actor", "successor", "action_encoder")
                ),
                f"{variant}: predictor state contains forbidden key {key!r}",
            )
        pretrained_provenance = _mapping(
            checkpoint.get("pretrained_world_model_provenance"),
            audit=audit,
            context=f"{variant}.epoch10.pretrained_world_model_provenance",
        )
        audit.require(
            pretrained_provenance.get("source_checkpoint_sha256")
            == PRETRAINED_WORLD_MODEL_SHA256,
            f"{variant}: deployment checkpoint uses another pretrained LeWM",
        )
        if state_dict_sha256 is not None and world_state:
            world_hashes[variant] = state_dict_sha256(world_state)

        audit.require(
            last.get("global_step") == TRAINING_STEPS,
            f"{variant}.last: global_step is not 127960",
        )
        audit.require(
            last.get("epoch") == TRAINING_EPOCHS - 1,
            f"{variant}.last: Lightning epoch is not 9",
        )
        required_rng = {
            "v1_data_generator_state",
            "v1_goal_generator_state",
            "v1_task_generator_state",
            "v1_validation_goal_generator_state",
            "v1_validation_task_generator_state",
            "v1_validation_goal_epoch_state",
            "v1_validation_task_epoch_state",
        }
        audit.require(
            required_rng <= set(last),
            f"{variant}.last: resume RNG state is incomplete",
        )
        last_finite = _finite_tensors(
            {
                "state_dict": last.get("state_dict"),
                "optimizer_states": last.get("optimizer_states"),
                "lr_schedulers": last.get("lr_schedulers"),
            },
            audit=audit,
            context=f"{variant}.last",
        )
        last_state = _mapping(
            last.get("state_dict"),
            audit=audit,
            context=f"{variant}.last.state_dict",
        )
        last_world_state = _prefixed_state(
            last_state,
            "model.",
            audit=audit,
            context=f"{variant}.last.state_dict",
        )
        last_predictor_state = _prefixed_state(
            last_state,
            "predictor.",
            audit=audit,
            context=f"{variant}.last.state_dict",
        )
        last_target_state = _prefixed_state(
            last_state,
            "target_predictor.",
            audit=audit,
            context=f"{variant}.last.state_dict",
        )
        if state_dict_sha256 is not None:
            audit.require(
                state_dict_sha256(last_world_state) == state_dict_sha256(world_state),
                f"{variant}: Lightning/deployment world states differ",
            )
            audit.require(
                state_dict_sha256(last_predictor_state)
                == state_dict_sha256(predictor_state),
                f"{variant}: Lightning/deployment online predictor states differ",
            )
            audit.require(
                state_dict_sha256(last_target_state) == state_dict_sha256(target_state),
                f"{variant}: Lightning/deployment target predictor states differ",
            )
        optimizer_audit = _optimizer_audit(
            last, predictor_state, variant=variant, audit=audit
        )

        checkpoint_sha = (
            _file_sha256(checkpoint_path) if checkpoint_path.is_file() else None
        )
        item = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "checkpoint_bytes": (
                checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
            ),
            "checkpoint_finite_stats": checkpoint_finite,
            "action_encoder_sha256": action_hash,
            "world_model_state_sha256": world_hashes.get(variant),
            "training_result_path": str(result_path),
            "training_result_sha256": (
                _file_sha256(result_path) if result_path.is_file() else None
            ),
            "training_manifest_path": str(manifest_path),
            "training_manifest_sha256": (
                _file_sha256(manifest_path) if manifest_path.is_file() else None
            ),
            "metrics_files": metric_files,
            "metrics_epochs": metric_epochs,
            "metrics_audit": metric_audit,
            "last_checkpoint_path": str(last_path),
            "last_checkpoint_bytes": (
                last_path.stat().st_size if last_path.is_file() else None
            ),
            "last_checkpoint_finite_stats": last_finite,
            "optimizer": optimizer_audit,
            "process": _audit_process(
                run_dir,
                variant=variant,
                pid=pids.get(variant),
                audit=audit,
            ),
            "cuda_device": cuda_device,
            "peak_cuda_memory_bytes": peak_memory,
        }
        evidence["variants"][variant] = item
        del checkpoint, last
        gc.collect()

    audit.require(
        len(world_hashes) == len(VARIANTS) and len(set(world_hashes.values())) == 1,
        "cross-run: six frozen world-model states are not identical",
    )
    audit.require(
        len(latent_hashes) == len(VARIANTS) and len(set(latent_hashes.values())) == 1,
        "cross-run: six methods do not share one frozen latent store",
    )
    audit.require(
        len(split_hashes) == len(VARIANTS) and len(set(split_hashes.values())) == 1,
        "cross-run: six methods do not share one train/validation split",
    )

    try:
        disk = shutil.disk_usage(
            output_root if output_root.exists() else output_root.parent
        )
        evidence["disk"] = {
            "outputs_bytes": _directory_bytes(output_root, excluded=output_json),
            "free_bytes": disk.free,
            "min_free_bytes": int(minimum_free_bytes),
            "recommended_prelaunch_free_bytes": 20 * 2**30,
            "estimated_six_run_training_bytes_low": 11 * 2**30,
            "estimated_six_run_training_bytes_high": 12 * 2**30,
            "recommended_posttraining_planning_free_bytes": 5 * 2**30,
        }
        audit.require(
            disk.free >= minimum_free_bytes,
            f"disk: {disk.free / 2**30:.2f} GiB free is below "
            f"{minimum_free_bytes / 2**30:.2f} GiB",
        )
    except Exception as error:
        audit.fail(f"disk: cannot inspect output filesystem: {error}")

    evidence["warnings"] = audit.warnings
    evidence["errors"] = audit.errors
    evidence["status"] = (
        "FAIL" if audit.errors else "PASS_WITH_WARNINGS" if audit.warnings else "PASS"
    )
    return evidence


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one complete JSON object and publish it with one atomic replace."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def acceptance_exit_code(evidence: Mapping[str, Any]) -> int:
    """PASS and PASS_WITH_WARNINGS are successful command outcomes."""

    return 1 if evidence.get("status") == "FAIL" else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate six formal Actor-Free TD-JEPA V1 training runs and "
            "atomically write training_acceptance.json."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPOSITORY_ROOT),
        help=(
            "Exact V1 worktree. Defaults to the repository containing this "
            "script, not an older sibling checkout."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root containing c,d,f,g1,g2,g3/seed_3072 training directories.",
    )
    parser.add_argument(
        "--pids",
        type=parse_pid_map,
        default={},
        help="Optional c=PID,d=PID,f=PID,g1=PID,g2=PID,g3=PID mapping.",
    )
    parser.add_argument(
        "--output-json",
        help=(
            "Acceptance JSON path. Defaults to <output-root>/training_acceptance.json."
        ),
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=5.0,
        help="Minimum remaining filesystem space required for acceptance.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0.0:
        parser.error("--min-free-gib must be finite and non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else output_root / "training_acceptance.json"
    )
    try:
        evidence = audit_training(
            repo_root=Path(args.repo_root),
            output_root=output_root,
            pids=args.pids,
            minimum_free_bytes=int(args.min_free_gib * 2**30),
            output_json=output_json,
        )
    except Exception as error:
        evidence = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output_root),
            "repo_root": str(Path(args.repo_root).expanduser().resolve()),
            "training_commit": TRAINING_COMMIT,
            "seed": TRAINING_SEED,
            "status": "FAIL",
            "expected_epoch": TRAINING_EPOCHS,
            "expected_global_step": TRAINING_STEPS,
            "expected_action_encoder_sha256": EXPECTED_ACTION_ENCODER_SHA256,
            "variants": {},
            "warnings": [],
            "errors": [f"internal acceptance error: {error}"],
        }
    try:
        _atomic_write_json(output_json, evidence)
    except Exception as error:
        print(f"Could not publish {output_json}: {error}", file=sys.stderr)
        return 1
    print(
        f"V1 TRAINING ACCEPTANCE: {evidence['status']} -> {output_json}",
        file=sys.stderr,
    )
    for error in evidence.get("errors", []):
        print(f"ERROR: {error}", file=sys.stderr)
    for warning in evidence.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    return acceptance_exit_code(evidence)


if __name__ == "__main__":
    raise SystemExit(main())
