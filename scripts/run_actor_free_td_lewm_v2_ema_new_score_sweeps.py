#!/usr/bin/env python3
"""Incrementally evaluate the two new V2 EMA scores at epochs 3--10.

The scheduler owns only ``f_plus_g_first`` and
``g_only_f_rollout_mean``.  It deliberately uses output and launcher roots
that are disjoint from the historical V2-EMA-SG 6x3 O50 sweep.  Every
checkpoint is watched independently, so one method can begin evaluation as
soon as its next checkpoint is stable; no cross-method epoch barrier exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence


def _load_base_scheduler() -> ModuleType:
    module_name = "_tdwm_v2_ema_sg_epoch_sweeps_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    source = Path(__file__).with_name(
        "run_actor_free_td_lewm_v2_ema_sg_epoch_sweeps.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the V2 EMA sweep scheduler from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_scheduler()

Cell = _base.Cell
SweepPaths = _base.SweepPaths

VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
SCORE_MODES = ("f_plus_g_first", "g_only_f_rollout_mean")
EPOCHS = tuple(range(3, 11))
SEED = _base.SEED
STEPS_PER_EPOCH = _base.STEPS_PER_EPOCH
TRAINING_COMMIT = "18cd574d522515f20f4103509b1e660b2fc89ea6"
VERSION_KEY = "v2_ema"
VERSION_DISPLAY_NAME = "V2 EMA"
METHOD_FAMILY = "actor_free_td_lewm_v2_ema_sg"
IMPLEMENTATION_VERSION = "v2_ema_sg"
EXPECTED_SELECTION_SHA256 = (
    "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
)
FIRST_ACTION_SCORE_MODE = SCORE_MODES[0]
ROLLOUT_MEAN_SCORE_MODE = SCORE_MODES[1]

FIRST_ACTION_SCORE_DEFINITION = {
    "formula": "f_cost - g_first_weight * q_first",
    "f_cost": "terminal_summed_mse(z_hat5, z_goal)",
    "f_rollout": "full_five_action_blocks_A1_through_A5",
    "q_first": "dot(G(z0, online_E_A(A1), w_goal), w_goal)",
    "q_first_state": "current_online_lewm_encoder_state_z0",
    "q_first_action": "first_candidate_raw_action_block_A1",
    "q_first_action_processing": "online_shared_lewm_action_encoder_to_192d",
    "q_first_task": "sqrt_dim_l2_normalized_goal_vector",
    "q_first_discount": "none",
    "cem_execution": "execute_A1_from_minimum_total_cost_plan",
}
ROLLOUT_MEAN_SCORE_DEFINITION = {
    "formula": "cost = -mean(q1, q2, q3, q4, q5)",
    "score": (
        "negative_mean_goal_projection_over_f_rollout_aligned_predecessor_action_pairs"
    ),
    "f_transition_used": True,
    "f_goal_distance_used": False,
    "g_score": "mean_goal_projection_over_all_rollout_blocks",
    "g_aggregation": "mean_over_5_blocks",
    "rollout_horizon": 5,
    "state_source_for_q1": "current_online_encoder_state",
    "state_source_for_q2_to_q5": "online_lewm_rollout_predicted_states",
    "state_sequence": "z0_and_first_h_minus_one_full_f_rollout_states",
    "action_sequence": "all_h_candidate_blocks_online_shared_lewm_action_encoder",
    "action_alignment": "qk_uses_same_candidate_action_block_Ak",
    "action_processing": "online_shared_lewm_action_encoder_to_192d",
    "task": "sqrt_dim_l2_normalized_goal_vector_broadcast_over_5_blocks",
    "gamma": "unused",
    "terminal_f_cost": "unused",
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}

_BASE_AUDIT_COMPLETE_OUTPUT = _base.audit_complete_output
_BASE_AUDIT_STABLE_CHECKPOINT = _base.audit_stable_checkpoint
_BASE_STATE_DOCUMENT = _base._state_document
_EXPECTED_EVALUATION_COMMIT: str | None = None


def git_revision(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"Malformed evaluation git revision: {revision!r}.")
    return revision


def git_is_clean(repository: Path) -> bool:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def _format_weight(weight: float) -> str:
    return format(weight, ".15g")


def _weight_slug(weight: float) -> str:
    return (
        _format_weight(weight)
        .lower()
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )


def _validated_weight(value: float) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("g_first_weight must be finite and non-negative.")
    return weight


def training_manifest_path(formal_root: Path, variant: str) -> Path:
    return formal_root / variant / f"seed_{SEED}" / "training_manifest.json"


def checkpoint_path(formal_root: Path, variant: str, epoch: int) -> Path:
    return _base.checkpoint_path(formal_root, variant, epoch)


def audit_training_manifest(path: Path, *, variant: str) -> dict[str, Any]:
    """Fail closed unless a run is the exact existing V2 EMA training."""

    manifest = _base.read_json(path)
    expected = {
        "method": f"{METHOD_FAMILY}_{variant}",
        "method_family": METHOD_FAMILY,
        "variant": variant,
        "implementation_version": IMPLEMENTATION_VERSION,
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "stage": "coupled_hybrid_ema_target_finetuning",
        "initialization": "corresponding_v1_deployment_finetune",
        "seed": SEED,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path} {key}={manifest.get(key)!r}, expected {value!r}.")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{path} has no runtime mapping.")
    revision = runtime.get("tdwm_git_revision")
    if revision != TRAINING_COMMIT:
        raise ValueError(
            f"{path} runtime.tdwm_git_revision={revision!r}, "
            f"expected {TRAINING_COMMIT!r}."
        )
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"{path} has no training protocol mapping.")
    protocol_sha256 = manifest.get("protocol_sha256")
    if protocol_sha256 != _base.canonical_json_sha256(protocol):
        raise ValueError(f"{path} has a missing or stale protocol_sha256.")
    for key in (
        "method",
        "method_family",
        "variant",
        "implementation_version",
        "stage",
        "initialization",
    ):
        value = expected[key]
        if protocol.get(key) != value:
            raise ValueError(
                f"{path} protocol.{key}={protocol.get(key)!r}, expected {value!r}."
            )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _base.file_sha256(path),
        "identity": expected,
        "protocol_sha256": protocol_sha256,
        "training_commit": TRAINING_COMMIT,
    }


def audit_training_manifests(formal_root: Path) -> dict[str, dict[str, Any]]:
    return {
        variant: audit_training_manifest(
            training_manifest_path(formal_root, variant), variant=variant
        )
        for variant in VARIANTS
    }


def _checkpoint_variant(path: Path) -> str:
    resolved = path.expanduser().resolve()
    variant = resolved.parent.name
    expected_method_dir = f"{METHOD_FAMILY}_{variant}"
    if variant not in VARIANTS or resolved.parent.parent.name != expected_method_dir:
        raise ValueError(f"Checkpoint has an unexpected V2 EMA path: {resolved}")
    return variant


def _checkpoint_training_manifest(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for ancestor in resolved.parents:
        if ancestor.name == "checkpoints":
            return ancestor.parent / "training_manifest.json"
    raise ValueError(f"Checkpoint is outside a run checkpoints directory: {resolved}")


def audit_stable_checkpoint(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Bind each ready checkpoint to its own training manifest before dispatch."""

    checkpoint_evidence = _BASE_AUDIT_STABLE_CHECKPOINT(path, **kwargs)
    variant = _checkpoint_variant(path)
    checkpoint_evidence["training_manifest"] = audit_training_manifest(
        _checkpoint_training_manifest(path),
        variant=variant,
    )
    return checkpoint_evidence


def build_cells(
    *,
    paths: SweepPaths,
    python: str,
    g_first_weight: float = 1.0,
) -> list[Cell]:
    weight = _validated_weight(g_first_weight)
    weight_text = _format_weight(weight)
    weight_slug = _weight_slug(weight)
    cells: list[Cell] = []
    for epoch in EPOCHS:
        for variant in VARIANTS:
            checkpoint = checkpoint_path(paths.formal_root, variant, epoch)
            training_manifest = training_manifest_path(paths.formal_root, variant)
            evaluator = f"scripts/evaluate_{METHOD_FAMILY}_{variant}.py"
            for score_mode in SCORE_MODES:
                if score_mode == FIRST_ACTION_SCORE_MODE:
                    config = (
                        f"configs/experiment/{METHOD_FAMILY}_{variant}"
                        "_cube_checkpoint_o50.yaml"
                    )
                    suffix = f"{score_mode}_alpha_{weight_slug}"
                    mode_arguments = (
                        "--score-mode",
                        score_mode,
                        "--g-first-weight",
                        weight_text,
                    )
                    output = (
                        paths.sweep_root
                        / f"epoch_{epoch:02d}"
                        / variant
                        / score_mode
                        / f"alpha_{weight_slug}"
                    )
                else:
                    config = (
                        f"configs/experiment/{METHOD_FAMILY}_{variant}"
                        "_cube_checkpoint_o50_g_only_f_rollout_mean.yaml"
                    )
                    suffix = score_mode
                    mode_arguments = ("--score-mode", score_mode)
                    output = (
                        paths.sweep_root / f"epoch_{epoch:02d}" / variant / score_mode
                    )
                cell_id = f"v2_ema_e{epoch:02d}_{variant}_{suffix}"
                checkpoint_epoch_arguments: tuple[str, ...] = ()
                if epoch < 10:
                    checkpoint_epoch_arguments = (
                        "--checkpoint-epoch",
                        str(epoch),
                    )
                argv = (
                    python,
                    evaluator,
                    "--config",
                    config,
                    "--dataset",
                    str(paths.dataset),
                    "--checkpoint-path",
                    str(checkpoint),
                    *checkpoint_epoch_arguments,
                    *mode_arguments,
                    "--training-manifest",
                    str(training_manifest),
                    "--output-dir",
                    str(output),
                )
                cells.append(
                    Cell(
                        cell_id=cell_id,
                        epoch=epoch,
                        variant=variant,
                        score_mode=score_mode,
                        checkpoint=str(checkpoint),
                        output_dir=str(output),
                        job_dir=str(paths.launcher_root / "jobs" / cell_id),
                        config_path=config,
                        argv=argv,
                    )
                )
    if len(cells) != 96:
        raise AssertionError(f"New V2 EMA sweep must have 96 cells, got {len(cells)}.")
    for attribute in ("cell_id", "output_dir", "job_dir"):
        values = [getattr(cell, attribute) for cell in cells]
        if len(set(values)) != len(values):
            raise AssertionError(f"Sweep {attribute} values are not unique.")
    return cells


def _argv_value(cell: Cell, option: str) -> str | None:
    try:
        index = cell.argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(cell.argv):
        raise ValueError(f"{cell.cell_id} has no value after {option}.")
    return cell.argv[index + 1]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _require_values(
    mapping: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"{label}.{key}={mapping.get(key)!r}, expected {value!r}.")


def audit_complete_output(cell: Cell) -> dict[str, Any]:
    """Audit an O50 cell plus all new V2 EMA identity/score metadata."""

    output_audit = _BASE_AUDIT_COMPLETE_OUTPUT(cell)
    output = Path(cell.output_dir)
    selection_sha256 = _base.file_sha256(output / "episode_selection.json")
    if selection_sha256 != EXPECTED_SELECTION_SHA256:
        raise ValueError(
            f"{cell.cell_id} episode_selection.json SHA-256={selection_sha256!r}, "
            f"expected {EXPECTED_SELECTION_SHA256!r}."
        )
    results = _base.read_json(output / "results.json")
    manifest = _base.read_json(output / "protocol_manifest.json")
    _require_values(
        manifest,
        {"score_mode": cell.score_mode},
        label=f"{cell.cell_id} manifest",
    )
    checkpoint_metadata = _require_mapping(
        manifest.get("checkpoint"), label=f"{cell.cell_id} manifest.checkpoint"
    )
    _require_values(
        checkpoint_metadata,
        {"implementation_version": IMPLEMENTATION_VERSION},
        label=f"{cell.cell_id} manifest.checkpoint",
    )
    checkpoint_sha256 = _base.file_sha256(Path(cell.checkpoint))
    manifest_path_text = _argv_value(cell, "--training-manifest")
    if manifest_path_text is None:
        raise ValueError(f"{cell.cell_id} has no --training-manifest binding.")
    training_manifest = Path(manifest_path_text).resolve()
    training_evidence = audit_training_manifest(training_manifest, variant=cell.variant)
    training_manifest_sha256 = training_evidence["sha256"]
    identity = {
        "version_key": VERSION_KEY,
        "version_display_name": VERSION_DISPLAY_NAME,
        "training_commit": TRAINING_COMMIT,
        "method": f"{METHOD_FAMILY}_{cell.variant}",
        "epoch": cell.epoch,
        "checkpoint_epoch": cell.epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_path": str(training_manifest),
        "training_manifest_sha256": training_manifest_sha256,
    }
    for label, document in (("results", results), ("manifest", manifest)):
        _require_values(document, identity, label=f"{cell.cell_id} {label}")
    runtime = _require_mapping(
        manifest.get("runtime"), label=f"{cell.cell_id} manifest.runtime"
    )
    evaluation_commit = runtime.get("tdwm_git_revision")
    expected_evaluation_commit = _EXPECTED_EVALUATION_COMMIT or git_revision(
        Path(__file__).resolve().parents[1]
    )
    if evaluation_commit != expected_evaluation_commit:
        raise ValueError(
            f"{cell.cell_id} manifest.runtime.tdwm_git_revision="
            f"{evaluation_commit!r}, expected {expected_evaluation_commit!r}."
        )
    for label, document in (("results", results), ("manifest", manifest)):
        _require_values(
            document,
            {"evaluation_commit": expected_evaluation_commit},
            label=f"{cell.cell_id} {label}",
        )
    protocol = _require_mapping(
        manifest.get("protocol"), label=f"{cell.cell_id} manifest.protocol"
    )
    planning = _require_mapping(
        protocol.get("planning"), label=f"{cell.cell_id} protocol.planning"
    )
    inference = _require_mapping(
        protocol.get("inference_objective"),
        label=f"{cell.cell_id} protocol.inference_objective",
    )
    _require_values(
        planning,
        {"horizon": 5},
        label=f"{cell.cell_id} protocol.planning",
    )
    _require_values(
        inference,
        {"score_mode": cell.score_mode},
        label=f"{cell.cell_id} protocol.inference_objective",
    )
    if cell.score_mode == FIRST_ACTION_SCORE_MODE:
        weight_text = _argv_value(cell, "--g-first-weight")
        if weight_text is None:
            raise ValueError(f"{cell.cell_id} has no first-action alpha.")
        weight = float(weight_text)
        score_metadata = {
            "g_first_weight": weight,
            "planning": {"horizon": 5},
            "score_definition": FIRST_ACTION_SCORE_DEFINITION,
        }
        _require_values(
            inference,
            {
                "g_first_weight": weight,
                "score_definition": FIRST_ACTION_SCORE_DEFINITION,
            },
            label=f"{cell.cell_id} protocol.inference_objective",
        )
        for label, document in (("results", results), ("manifest", manifest)):
            _require_values(document, score_metadata, label=f"{cell.cell_id} {label}")
    elif cell.score_mode == ROLLOUT_MEAN_SCORE_MODE:
        if _argv_value(cell, "--g-first-weight") is not None:
            raise ValueError(f"{cell.cell_id} rollout-mean argv contains alpha.")
        if "g_first_weight" in inference:
            raise ValueError(f"{cell.cell_id} rollout-mean protocol contains alpha.")
        mean_metadata = {
            "g_aggregation": "mean_over_5_blocks",
            "state_source_for_q1": "current_online_encoder_state",
            "state_source_for_q2_to_q5": ("online_lewm_rollout_predicted_states"),
            "f_goal_distance_used": False,
            "f_transition_used": True,
            "planning_horizon": 5,
            "rollout_horizon": 5,
            "executed_action_block": "first_block_only",
            "replanning": "every_action_block",
            "score_definition": ROLLOUT_MEAN_SCORE_DEFINITION,
        }
        _require_values(
            inference,
            {
                "f_goal_distance_used": False,
                "f_transition_used": True,
                "g_aggregation": "mean_over_5_blocks",
                "rollout_horizon": 5,
                "score_definition": ROLLOUT_MEAN_SCORE_DEFINITION,
            },
            label=f"{cell.cell_id} protocol.inference_objective",
        )
        for label, document in (("results", results), ("manifest", manifest)):
            if "g_first_weight" in document:
                raise ValueError(f"{cell.cell_id} {label} contains a stale alpha.")
            _require_values(document, mean_metadata, label=f"{cell.cell_id} {label}")
    else:  # pragma: no cover - build_cells makes this unreachable.
        raise ValueError(f"Unsupported new-score mode: {cell.score_mode}")
    return output_audit


def adoption_matches_cell(cell: Cell, job_dir: Path, pid: int) -> bool:
    """Retain exact adoption while allowing epoch 10 to omit its CLI flag."""

    expected = list(cell.argv)
    record = _base._start_record(job_dir)
    if record is not None:
        argv = record.get("argv")
        if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
            for index, item in enumerate(argv):
                if item == expected[1] and argv[index:] == expected[1:]:
                    return True
        if record.get("argv_sha256") == _base.canonical_json_sha256(expected):
            return True
    command = _base._proc_values(pid, "cmdline", b"\0")
    if any(
        command[index : index + len(expected)] == expected
        for index in range(max(0, len(command) - len(expected) + 1))
    ):
        return True
    flattened = " ".join(command)
    unique_tokens = [
        expected[1],
        "--checkpoint-path",
        cell.checkpoint,
        "--score-mode",
        cell.score_mode,
        "--training-manifest",
        str(_argv_value(cell, "--training-manifest")),
        "--output-dir",
        cell.output_dir,
    ]
    checkpoint_epoch = _argv_value(cell, "--checkpoint-epoch")
    if checkpoint_epoch is not None:
        unique_tokens.extend(("--checkpoint-epoch", checkpoint_epoch))
    first_weight = _argv_value(cell, "--g-first-weight")
    if first_weight is not None:
        unique_tokens.extend(("--g-first-weight", first_weight))
    return bool(flattened) and all(token in flattened for token in unique_tokens)


def _state_document(**kwargs: Any) -> dict[str, Any]:
    document = _BASE_STATE_DOCUMENT(**kwargs)
    document.update(
        {
            "source": "actor_free_td_lewm_v2_ema_new_score_sweep_scheduler",
            "version_key": VERSION_KEY,
            "version_display_name": VERSION_DISPLAY_NAME,
            "training_commit": TRAINING_COMMIT,
            "expected_evaluation_commit": _EXPECTED_EVALUATION_COMMIT,
            "expected_selection_sha256": EXPECTED_SELECTION_SHA256,
            "score_modes": list(SCORE_MODES),
        }
    )
    return document


@contextmanager
def _new_score_scheduler_hooks() -> Iterator[None]:
    old_audit = _base.audit_complete_output
    old_checkpoint_audit = _base.audit_stable_checkpoint
    old_adoption = _base.adoption_matches_cell
    old_state_document = _base._state_document
    _base.audit_complete_output = audit_complete_output
    _base.audit_stable_checkpoint = audit_stable_checkpoint
    _base.adoption_matches_cell = adoption_matches_cell
    _base._state_document = _state_document
    try:
        yield
    finally:
        _base.audit_complete_output = old_audit
        _base.audit_stable_checkpoint = old_checkpoint_audit
        _base.adoption_matches_cell = old_adoption
        _base._state_document = old_state_document


def run_scheduler(**kwargs: Any) -> int:
    with _new_score_scheduler_hooks():
        return _base.run_scheduler(**kwargs)


def classify_existing_cells(
    cells: Sequence[Cell],
    gpu_indices: Sequence[int],
    adoption_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[Cell], dict[str, _base.ActiveCell]]:
    with _new_score_scheduler_hooks():
        return _base.classify_existing_cells(cells, gpu_indices, adoption_overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--formal-root")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--g-first-weight", type=float, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-evals-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument(
        "--checkpoint-metadata", choices=("strict", "none"), default="strict"
    )
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument("--pyopengl-platform", default="osmesa")
    parser.add_argument("--ld-preload", default=_base.DEFAULT_LD_PRELOAD)
    parser.add_argument("--launch-id")
    parser.add_argument(
        "--adopt-running", action="append", default=[], metavar="CELL=PID:GPU"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_adoption_overrides(values: Sequence[str]) -> dict[str, tuple[int, int]]:
    overrides: dict[str, tuple[int, int]] = {}
    for value in values:
        try:
            cell_id, raw_process = value.split("=", 1)
            raw_pid, raw_gpu = raw_process.split(":", 1)
            binding = (int(raw_pid), int(raw_gpu))
        except (ValueError, TypeError) as error:
            raise SystemExit(
                f"Invalid --adopt-running {value!r}; expected CELL=PID:GPU."
            ) from error
        if cell_id in overrides:
            raise SystemExit(f"Duplicate --adopt-running cell: {cell_id}")
        overrides[cell_id] = binding
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    global _EXPECTED_EVALUATION_COMMIT

    args = build_parser().parse_args(argv)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("--gpus must contain distinct GPU indices.")
    if args.max_evals_per_gpu < 1:
        raise SystemExit("--max-evals-per-gpu must be positive.")
    if args.stable_polls < 2:
        raise SystemExit("--stable-polls must be at least two.")
    try:
        weight = _validated_weight(args.g_first_weight)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    adoption_overrides = _parse_adoption_overrides(args.adopt_running)
    repository = Path(__file__).resolve().parents[1]
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    paths = SweepPaths(
        repository=repository,
        dataset=Path(args.dataset).expanduser().resolve(),
        formal_root=(
            Path(args.formal_root).expanduser().resolve()
            if args.formal_root
            else bundle_root / "formal"
        ),
        bundle_root=bundle_root,
        sweep_root=bundle_root / "new_score_evaluation_sweeps",
        launcher_root=bundle_root / "new_score_evaluation_sweep_launcher",
    )
    _EXPECTED_EVALUATION_COMMIT = git_revision(repository)
    evaluation_git_clean = git_is_clean(repository)
    python = Path(args.python).expanduser()
    if not python.is_absolute():
        python = Path(os.path.abspath(python))
    cells = build_cells(paths=paths, python=str(python), g_first_weight=weight)
    unknown_adoptions = set(adoption_overrides) - {cell.cell_id for cell in cells}
    if unknown_adoptions:
        raise SystemExit(f"Unknown --adopt-running cells: {sorted(unknown_adoptions)}")
    render_environment = {
        "MUJOCO_GL": args.mujoco_gl,
        "PYOPENGL_PLATFORM": args.pyopengl_platform,
        "LD_PRELOAD": args.ld_preload,
    }
    plan: dict[str, Any] = {
        "schema_version": 1,
        "source": "actor_free_td_lewm_v2_ema_new_score_sweep_scheduler",
        "version_key": VERSION_KEY,
        "version_display_name": VERSION_DISPLAY_NAME,
        "training_commit": TRAINING_COMMIT,
        "expected_evaluation_commit": _EXPECTED_EVALUATION_COMMIT,
        "evaluation_git_clean": evaluation_git_clean,
        "expected_selection_sha256": EXPECTED_SELECTION_SHA256,
        "cell_count": len(cells),
        "epochs": list(EPOCHS),
        "variants": list(VARIANTS),
        "score_modes": list(SCORE_MODES),
        "g_first_weight": weight,
        "epoch_10_owner": "this_new_score_scheduler",
        "historical_6x3_outputs_modified": False,
        "training_manifest_validation": "per_checkpoint_before_dispatch",
        "training_manifest_paths": {
            variant: str(training_manifest_path(paths.formal_root, variant))
            for variant in VARIANTS
        },
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
        plan["training_manifest_audit"] = "not_run_in_dry_run"
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not evaluation_git_clean:
        raise RuntimeError(
            "V2 EMA new-score evaluation requires a clean git checkout so "
            "evaluation_commit identifies the exact code being executed."
        )
    if not paths.dataset.exists():
        raise FileNotFoundError(paths.dataset)
    paths.launcher_root.mkdir(parents=True, exist_ok=True)
    launch_id = args.launch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths.launcher_root / "runs" / launch_id
    return run_scheduler(
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


if __name__ == "__main__":
    raise SystemExit(main())
