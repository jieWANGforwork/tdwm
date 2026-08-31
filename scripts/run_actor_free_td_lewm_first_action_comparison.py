#!/usr/bin/env python3
"""Run the isolated first-action and V2 rollout-mean inference comparison.

This launcher owns evaluation orchestration only.  It never trains a model,
selects an alpha, or writes into the historical V0/V1/V2 result bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

VERSIONS = ("v0", "v1", "v2")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
FIRST_ACTION_MODE = "f_plus_g_first"
FORMAL_SCORE_MODES = ("f_only", "f_plus_g", FIRST_ACTION_MODE)
ROLLOUT_MEAN_MODE = "g_only_f_rollout_mean"
V2_ONLY_FORMAL_SCORE_MODES = ("g_only", ROLLOUT_MEAN_MODE)
ALL_FORMAL_SCORE_MODES = FORMAL_SCORE_MODES + V2_ONLY_FORMAL_SCORE_MODES
DEVELOPMENT_ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
EXPECTED_HORIZON = 5
EXPECTED_EPISODES_BY_STAGE = {"smoke": 1, "development": 10, "formal": 50}
ROLLOUT_MEAN_METADATA = {
    "g_aggregation": "mean_over_5_blocks",
    "state_source_for_q1": "current_online_encoder_state",
    "state_source_for_q2_to_q5": "online_lewm_rollout_predicted_states",
    "f_goal_distance_used": False,
}
ROLLOUT_MEAN_INFERENCE_METADATA = {
    "f_transition_used": True,
    "f_goal_distance_used": False,
    "g_score": "mean_goal_projection_over_all_rollout_blocks",
    "rollout_horizon": 5,
    "executed_action_block": "first_block_only",
    "replanning": "every_action_block",
}
REQUIRED_OUTPUT_FILES = (
    "results.json",
    "protocol_manifest.json",
    "episode_selection.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_alpha(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("alpha must be finite and non-negative.")
    alpha = float(value)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative.")
    return 0.0 if alpha == 0.0 else alpha


def alpha_slug(value: float) -> str:
    alpha = normalize_alpha(value)
    encoded = format(alpha, ".15g").lower()
    return encoded.replace("+", "").replace("-", "m").replace(".", "p")


def _unique_alphas(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(normalize_alpha(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("alpha values must be unique after normalization.")
    return normalized


@dataclass(frozen=True)
class StagePlan:
    stage: str
    variants: tuple[str, ...]
    score_modes: tuple[str, ...]
    v2_only_score_modes: tuple[str, ...]
    alphas: tuple[float, ...]


def resolve_stage_plan(
    *,
    stage: str,
    variants: Sequence[str] | None,
    score_modes: Sequence[str] | None,
    alphas: Sequence[float] | None,
) -> StagePlan:
    if stage not in {"smoke", "development", "formal"}:
        raise ValueError(f"Unsupported stage {stage!r}.")
    selected_variants = tuple(variants or (("c",) if stage == "smoke" else VARIANTS))
    if not selected_variants or len(set(selected_variants)) != len(selected_variants):
        raise ValueError("variants must be a non-empty unique list.")
    unsupported_variants = set(selected_variants) - set(VARIANTS)
    if unsupported_variants:
        raise ValueError(f"Unsupported variants: {sorted(unsupported_variants)}")

    if stage == "formal":
        requested_modes = tuple(score_modes or ALL_FORMAL_SCORE_MODES)
        if not requested_modes or len(set(requested_modes)) != len(requested_modes):
            raise ValueError("formal score modes must be a non-empty unique list.")
        unsupported_modes = set(requested_modes) - set(ALL_FORMAL_SCORE_MODES)
        if unsupported_modes:
            raise ValueError(
                f"Unsupported formal score modes: {sorted(unsupported_modes)}"
            )
        selected_modes = tuple(
            mode for mode in requested_modes if mode in FORMAL_SCORE_MODES
        )
        selected_v2_only_modes = tuple(
            mode for mode in requested_modes if mode in V2_ONLY_FORMAL_SCORE_MODES
        )
        if alphas is None or len(alphas) != 1:
            raise ValueError(
                "formal comparison requires exactly one explicit --alpha; "
                "this launcher never selects alpha."
            )
        selected_alphas = _unique_alphas(alphas)
    elif stage == "development":
        if score_modes not in (None, (), [FIRST_ACTION_MODE], (FIRST_ACTION_MODE,)):
            raise ValueError("development runs only f_plus_g_first.")
        selected_modes = (FIRST_ACTION_MODE,)
        selected_v2_only_modes = ()
        selected_alphas = _unique_alphas(alphas or DEVELOPMENT_ALPHA_GRID)
    else:
        if score_modes not in (None, (), [FIRST_ACTION_MODE], (FIRST_ACTION_MODE,)):
            raise ValueError("smoke runs only f_plus_g_first.")
        selected_modes = (FIRST_ACTION_MODE,)
        selected_v2_only_modes = ()
        selected_alphas = _unique_alphas(alphas or (1.0,))
        if len(selected_alphas) != 1:
            raise ValueError("smoke requires exactly one alpha.")

    return StagePlan(
        stage=stage,
        variants=selected_variants,
        score_modes=selected_modes,
        v2_only_score_modes=selected_v2_only_modes,
        alphas=selected_alphas,
    )


def load_checkpoint_manifest(path: str | Path) -> dict[str, dict[str, Path]]:
    manifest_path = Path(path).expanduser().resolve()
    value = read_json(manifest_path)
    if set(value) != set(VERSIONS):
        raise ValueError("checkpoint manifest must contain exactly v0, v1, and v2.")
    resolved: dict[str, dict[str, Path]] = {}
    for version in VERSIONS:
        version_value = value[version]
        if not isinstance(version_value, Mapping) or set(version_value) != set(
            VARIANTS
        ):
            raise ValueError(
                f"checkpoint manifest {version} must contain exactly "
                f"{', '.join(VARIANTS)}."
            )
        resolved[version] = {}
        for variant in VARIANTS:
            raw_path = version_value[variant]
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"checkpoint {version}.{variant} must be a path.")
            checkpoint = Path(raw_path).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = manifest_path.parent / checkpoint
            checkpoint = checkpoint.resolve()
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise FileNotFoundError(checkpoint)
            resolved[version][variant] = checkpoint
    return resolved


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    version: str
    variant: str
    score_mode: str
    alpha: float | None
    checkpoint: str
    config_path: str
    output_dir: str
    log_path: str
    argv: tuple[str, ...]


def _job_output_dir(
    *,
    stage_root: Path,
    version: str,
    variant: str,
    score_mode: str,
    alpha: float | None,
) -> Path:
    output = stage_root / version / variant / score_mode
    if score_mode == FIRST_ACTION_MODE:
        if alpha is None:
            raise ValueError("First-action jobs require alpha.")
        output /= f"alpha_{alpha_slug(alpha)}"
    elif alpha is not None:
        raise ValueError("Legacy score modes must not receive alpha.")
    return output


def expected_horizon(score_mode: str) -> int:
    return 1 if score_mode == "g_only" else EXPECTED_HORIZON


def evaluation_config_path(
    repository: str | Path,
    *,
    version: str,
    variant: str,
    score_mode: str,
) -> Path:
    repository_path = Path(repository).expanduser().resolve()
    if score_mode == ROLLOUT_MEAN_MODE:
        if version != "v2":
            raise ValueError(f"{ROLLOUT_MEAN_MODE} is V2-only.")
        filename = (
            f"actor_free_td_lewm_v2_{variant}_cube_checkpoint_o50_"
            "g_only_f_rollout_mean.yaml"
        )
    else:
        filename = f"actor_free_td_lewm_{version}_{variant}_cube_checkpoint_o50.yaml"
    return repository_path / "configs" / "experiment" / filename


def build_jobs(
    *,
    repository: str | Path,
    output_root: str | Path,
    checkpoints: Mapping[str, Mapping[str, Path]],
    dataset: str | Path,
    python: str,
    plan: StagePlan,
) -> list[Job]:
    repository_path = Path(repository).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    stage_root = Path(output_root).expanduser().resolve() / plan.stage
    jobs: list[Job] = []
    for version in VERSIONS:
        for variant in plan.variants:
            checkpoint = Path(checkpoints[version][variant]).resolve()
            evaluator = (
                repository_path
                / "scripts"
                / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
            )
            version_score_modes = plan.score_modes
            if version == "v2":
                version_score_modes += plan.v2_only_score_modes
            for score_mode in version_score_modes:
                config = evaluation_config_path(
                    repository_path,
                    version=version,
                    variant=variant,
                    score_mode=score_mode,
                )
                mode_alphas: tuple[float | None, ...]
                if score_mode == FIRST_ACTION_MODE:
                    mode_alphas = plan.alphas
                else:
                    mode_alphas = (None,)
                for alpha in mode_alphas:
                    output = _job_output_dir(
                        stage_root=stage_root,
                        version=version,
                        variant=variant,
                        score_mode=score_mode,
                        alpha=alpha,
                    )
                    alpha_suffix = (
                        f"__alpha_{alpha_slug(alpha)}" if alpha is not None else ""
                    )
                    job_id = f"{version}__{variant}__{score_mode}{alpha_suffix}"
                    log = stage_root / "_launcher" / "jobs" / f"{job_id}.log"
                    argv = [
                        python,
                        str(evaluator),
                        "--config",
                        str(config),
                        "--dataset",
                        str(dataset_path),
                        "--checkpoint-path",
                        str(checkpoint),
                        "--score-mode",
                        score_mode,
                        "--output-dir",
                        str(output),
                    ]
                    if alpha is not None:
                        argv.extend(("--g-first-weight", format(alpha, ".17g")))
                    if plan.stage == "smoke":
                        argv.append("--smoke")
                    elif plan.stage == "development":
                        argv.append("--pilot")
                    jobs.append(
                        Job(
                            job_id=job_id,
                            stage=plan.stage,
                            version=version,
                            variant=variant,
                            score_mode=score_mode,
                            alpha=alpha,
                            checkpoint=str(checkpoint),
                            config_path=str(config),
                            output_dir=str(output),
                            log_path=str(log),
                            argv=tuple(argv),
                        )
                    )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("Comparison job identifiers are not unique.")
    if len({job.output_dir for job in jobs}) != len(jobs):
        raise AssertionError("Comparison output directories are not unique.")
    return jobs


def validate_job_inputs(
    jobs: Sequence[Job],
    *,
    dataset: Path,
    checkpoint_manifest: Path,
) -> None:
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not checkpoint_manifest.is_file():
        raise FileNotFoundError(checkpoint_manifest)
    for job in jobs:
        for path in (
            Path(job.checkpoint),
            Path(job.config_path),
            Path(job.argv[1]),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)


def ensure_fresh_stage(stage_root: Path, jobs: Sequence[Job]) -> None:
    if stage_root.exists():
        if not stage_root.is_dir():
            raise FileExistsError(f"Stage output is not a directory: {stage_root}")
        if any(stage_root.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty stage directory: {stage_root}"
            )
    for job in jobs:
        output = Path(job.output_dir)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(
                f"Refusing to overwrite non-empty job directory: {output}"
            )


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _selection_ranks(value: Mapping[str, Any], *, label: str) -> tuple[int, ...]:
    ranks = value.get("valid_row_ranks")
    if not isinstance(ranks, list) or not ranks:
        raise ValueError(f"{label}.valid_row_ranks must be a non-empty list.")
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in ranks
    ):
        raise ValueError(f"{label}.valid_row_ranks must contain non-negative integers.")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{label}.valid_row_ranks must not contain duplicates.")
    return tuple(ranks)


def load_selection(path: str | Path) -> tuple[int, ...]:
    value = read_json(Path(path).expanduser().resolve())
    if "valid_row_ranks" in value:
        return _selection_ranks(value, label="selection")
    selection = value.get("selection")
    if isinstance(selection, Mapping):
        return _selection_ranks(selection, label="selection")
    protocol = value.get("protocol")
    if isinstance(protocol, Mapping) and isinstance(protocol.get("selection"), Mapping):
        return _selection_ranks(protocol["selection"], label="protocol.selection")
    raise ValueError("Formal selection JSON has no valid_row_ranks selection.")


def validate_job_output(job: Job) -> dict[str, Any]:
    output = Path(job.output_dir)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{job.job_id} is missing {missing}.")
    results = read_json(output / "results.json")
    expected_identity = {
        "method": f"actor_free_td_lewm_{job.version}_{job.variant}",
        "method_family": f"actor_free_td_lewm_{job.version}",
        "variant": job.variant,
        "implementation_version": job.version,
        "score_mode": job.score_mode,
        "planning_horizon": expected_horizon(job.score_mode),
        "smoke": job.stage == "smoke",
        "pilot": job.stage == "development",
    }
    for key, expected in expected_identity.items():
        if results.get(key) != expected:
            raise ValueError(
                f"{job.job_id} results.{key}={results.get(key)!r}, "
                f"expected {expected!r}."
            )
    expected_episodes = EXPECTED_EPISODES_BY_STAGE[job.stage]
    metrics = results.get("metrics")
    if isinstance(metrics, Mapping) and "episode_successes" in metrics:
        successes = metrics["episode_successes"]
        if not isinstance(successes, list) or len(successes) != expected_episodes:
            raise ValueError(
                f"{job.job_id} results.metrics.episode_successes must contain "
                f"exactly {expected_episodes} outcomes."
            )
    manifest = read_json(output / "protocol_manifest.json")
    if manifest.get("score_mode") != job.score_mode:
        raise ValueError(f"{job.job_id} manifest score_mode is incorrect.")
    protocol = _required_mapping(manifest.get("protocol"), label="manifest.protocol")
    inference = _required_mapping(
        protocol.get("inference_objective"),
        label="manifest.protocol.inference_objective",
    )
    planning = _required_mapping(
        protocol.get("planning"), label="manifest.protocol.planning"
    )
    for key, expected in (
        ("method", expected_identity["method"]),
        ("method_family", expected_identity["method_family"]),
        ("variant", job.variant),
        ("implementation_version", job.version),
    ):
        if protocol.get(key) != expected:
            raise ValueError(f"{job.job_id} protocol.{key} is incorrect.")
    if inference.get("score_mode") != job.score_mode:
        raise ValueError(f"{job.job_id} protocol score_mode is incorrect.")
    mode_horizon = expected_horizon(job.score_mode)
    if planning.get("horizon") != mode_horizon:
        raise ValueError(
            f"{job.job_id} protocol planning.horizon must be {mode_horizon}."
        )

    if job.score_mode == FIRST_ACTION_MODE:
        if job.alpha is None:
            raise AssertionError("First-action job has no alpha.")
        for values, label in (
            (results, "results"),
            (manifest, "manifest"),
            (inference, "inference"),
        ):
            actual = values.get("g_first_weight")
            if isinstance(actual, bool) or actual is None or float(actual) != job.alpha:
                raise ValueError(f"{job.job_id} {label}.g_first_weight is incorrect.")
            definition = values.get("score_definition")
            if not isinstance(definition, Mapping) or not definition:
                raise ValueError(f"{job.job_id} {label}.score_definition is missing.")
    elif job.score_mode == ROLLOUT_MEAN_MODE:
        if job.version != "v2":
            raise ValueError(f"{job.job_id} assigns the V2-only mode to {job.version}.")
        for values, label in ((results, "results"), (manifest, "manifest")):
            for key, expected in ROLLOUT_MEAN_METADATA.items():
                if values.get(key) != expected:
                    raise ValueError(
                        f"{job.job_id} {label}.{key}={values.get(key)!r}, "
                        f"expected {expected!r}."
                    )
            definition = values.get("score_definition")
            if not isinstance(definition, Mapping) or not definition:
                raise ValueError(f"{job.job_id} {label}.score_definition is missing.")
            if "g_first_weight" in values:
                raise ValueError(
                    f"{job.job_id} rollout-mean {label} contains first-action alpha."
                )
        for key, expected in ROLLOUT_MEAN_INFERENCE_METADATA.items():
            if inference.get(key) != expected:
                raise ValueError(
                    f"{job.job_id} inference.{key}={inference.get(key)!r}, "
                    f"expected {expected!r}."
                )
        definition = inference.get("score_definition")
        if not isinstance(definition, Mapping) or not definition:
            raise ValueError(f"{job.job_id} inference.score_definition is missing.")
        if "g_first_weight" in inference:
            raise ValueError(
                f"{job.job_id} rollout-mean inference contains first-action alpha."
            )
    else:
        for values, label in (
            (results, "results"),
            (manifest, "manifest"),
            (inference, "inference"),
        ):
            if "g_first_weight" in values or "score_definition" in values:
                raise ValueError(
                    f"{job.job_id} legacy {label} contains first-action metadata."
                )
            if any(key in values for key in ROLLOUT_MEAN_METADATA):
                raise ValueError(
                    f"{job.job_id} legacy {label} contains rollout-mean metadata."
                )

    checkpoint = _required_mapping(
        manifest.get("checkpoint"), label="manifest.checkpoint"
    )
    recorded_checkpoint = checkpoint.get("path")
    if not isinstance(recorded_checkpoint, str) or (
        Path(recorded_checkpoint).resolve() != Path(job.checkpoint).resolve()
    ):
        raise ValueError(f"{job.job_id} manifest binds a different checkpoint.")
    selection = read_json(output / "episode_selection.json")
    ranks = _selection_ranks(selection, label=f"{job.job_id}.selection")
    if len(ranks) != expected_episodes:
        raise ValueError(
            f"{job.job_id} selection must contain exactly "
            f"{expected_episodes} valid row ranks."
        )
    return {
        "results_path": str(output / "results.json"),
        "manifest_path": str(output / "protocol_manifest.json"),
        "selection_path": str(output / "episode_selection.json"),
        "valid_row_ranks": list(ranks),
        "selection_sha256": canonical_json_sha256(list(ranks)),
    }


def require_identical_selections(
    evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[int, ...]:
    if not evidence:
        raise ValueError("No completed comparison jobs were provided.")
    reference_job = next(iter(evidence))
    reference = tuple(evidence[reference_job]["valid_row_ranks"])
    for job_id, values in evidence.items():
        actual = tuple(values["valid_row_ranks"])
        if actual != reference:
            raise ValueError(
                f"Selection mismatch: {job_id} differs from {reference_job}."
            )
    return reference


def verify_formal_disjointness(
    development_ranks: Sequence[int],
    formal_selection: str | Path | None,
) -> dict[str, Any]:
    if formal_selection is None:
        return {
            "formal_disjointness_verified": False,
            "formal_selection_path": None,
            "formal_selection_overlap": None,
            "note": (
                "No formal selection was supplied; this launcher does not claim "
                "that development pairs are disjoint from formal O50."
            ),
        }
    formal_path = Path(formal_selection).expanduser().resolve()
    formal_ranks = load_selection(formal_path)
    overlap = sorted(set(development_ranks) & set(formal_ranks))
    if overlap:
        raise ValueError(
            "Development pairs overlap the supplied formal selection at valid "
            f"row ranks {overlap}."
        )
    return {
        "formal_disjointness_verified": True,
        "formal_selection_path": str(formal_path),
        "formal_selection_overlap": [],
        "formal_selection_sha256": canonical_json_sha256(list(formal_ranks)),
    }


@dataclass
class ActiveJob:
    job: Job
    process: Any
    log_handle: Any
    gpu: str | None


def _launcher_payload(
    *,
    plan: StagePlan,
    repository: Path,
    dataset: Path,
    checkpoint_manifest: Path,
    output_root: Path,
    jobs: Sequence[Job],
    gpus: Sequence[str],
    max_concurrency: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "launcher": "actor_free_td_lewm_first_action_comparison",
        "inference_only": True,
        "training_performed": False,
        "alpha_selection_performed": False,
        "stage": plan.stage,
        "versions": list(VERSIONS),
        "variants": list(plan.variants),
        "shared_score_modes": list(plan.score_modes),
        "v2_only_score_modes": list(plan.v2_only_score_modes),
        "score_modes_by_version": {
            "v0": list(plan.score_modes),
            "v1": list(plan.score_modes),
            "v2": list(plan.score_modes + plan.v2_only_score_modes),
        },
        "alphas": list(plan.alphas),
        "repository": str(repository),
        "dataset": str(dataset),
        "checkpoint_manifest": str(checkpoint_manifest),
        "output_root": str(output_root),
        "gpus": list(gpus),
        "max_concurrency": max_concurrency,
        "formal_disjointness_verified": False,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "PLANNED",
        "jobs": {
            job.job_id: {**asdict(job), "argv": list(job.argv), "state": "PENDING"}
            for job in jobs
        },
    }


def run_jobs(
    *,
    jobs: Sequence[Job],
    plan: StagePlan,
    repository: str | Path,
    dataset: str | Path,
    checkpoint_manifest: str | Path,
    output_root: str | Path,
    gpus: Sequence[str],
    max_concurrency: int,
    formal_selection: str | Path | None,
    poll_seconds: float,
    popen: Callable[..., Any] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive.")
    if poll_seconds < 0.0:
        raise ValueError("poll_seconds must be non-negative.")
    if formal_selection is not None and plan.stage != "development":
        raise ValueError("--formal-selection is only valid for development.")
    repository_path = Path(repository).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    checkpoint_manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    stage_root = output_root_path / plan.stage
    validate_job_inputs(
        jobs,
        dataset=dataset_path,
        checkpoint_manifest=checkpoint_manifest_path,
    )
    ensure_fresh_stage(stage_root, jobs)
    stage_root.mkdir(parents=True, exist_ok=True)
    launcher_root = stage_root / "_launcher"
    launcher_root.mkdir(parents=True, exist_ok=False)
    manifest_path = launcher_root / "launcher_manifest.json"
    payload = _launcher_payload(
        plan=plan,
        repository=repository_path,
        dataset=dataset_path,
        checkpoint_manifest=checkpoint_manifest_path,
        output_root=output_root_path,
        jobs=jobs,
        gpus=gpus,
        max_concurrency=max_concurrency,
    )
    payload["status"] = "RUNNING"
    atomic_write_json(manifest_path, payload)

    pending = list(jobs)
    active: dict[str, ActiveJob] = {}
    evidence: dict[str, dict[str, Any]] = {}
    failed = False
    launched = 0

    try:
        while pending or active:
            while pending and len(active) < max_concurrency and not failed:
                job = pending.pop(0)
                output = Path(job.output_dir)
                output.mkdir(parents=True, exist_ok=False)
                log_path = Path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("x", encoding="utf-8")
                gpu = str(gpus[launched % len(gpus)]) if gpus else None
                environment = os.environ.copy()
                if gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = gpu
                try:
                    process = popen(
                        list(job.argv),
                        cwd=str(repository_path),
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                except Exception as error:
                    log_handle.close()
                    payload["jobs"][job.job_id].update(
                        {"state": "FAILED_TO_START", "error": str(error)}
                    )
                    failed = True
                    break
                active[job.job_id] = ActiveJob(
                    job=job,
                    process=process,
                    log_handle=log_handle,
                    gpu=gpu,
                )
                payload["jobs"][job.job_id].update(
                    {
                        "state": "RUNNING",
                        "pid": getattr(process, "pid", None),
                        "gpu": gpu,
                        "started_at": utc_now(),
                    }
                )
                launched += 1
                payload["updated_at"] = utc_now()
                atomic_write_json(manifest_path, payload)

            completed_any = False
            for job_id, running in list(active.items()):
                return_code = running.process.poll()
                if return_code is None:
                    continue
                completed_any = True
                running.log_handle.close()
                del active[job_id]
                state = payload["jobs"][job_id]
                state.update({"exit_code": int(return_code), "finished_at": utc_now()})
                if return_code != 0:
                    state["state"] = "FAILED"
                    failed = True
                    continue
                try:
                    job_evidence = validate_job_output(running.job)
                except Exception as error:
                    state.update({"state": "FAILED_VALIDATION", "error": str(error)})
                    failed = True
                else:
                    evidence[job_id] = job_evidence
                    state.update({"state": "SUCCEEDED", "evidence": job_evidence})
            payload["updated_at"] = utc_now()
            atomic_write_json(manifest_path, payload)
            if failed and not active:
                break
            if active and not completed_any:
                sleeper(poll_seconds)

        if pending:
            for job in pending:
                payload["jobs"][job.job_id]["state"] = "SKIPPED_AFTER_FAILURE"

        if not failed:
            try:
                ranks = require_identical_selections(evidence)
                payload["selection"] = {
                    "valid_row_ranks": list(ranks),
                    "sha256": canonical_json_sha256(list(ranks)),
                    "identical_across_all_jobs": True,
                }
                if plan.stage == "development":
                    payload.update(verify_formal_disjointness(ranks, formal_selection))
            except Exception as error:
                failed = True
                payload["selection_validation_error"] = str(error)
    except KeyboardInterrupt:
        failed = True
        payload["interrupted"] = True
        for running in active.values():
            terminate = getattr(running.process, "terminate", None)
            if callable(terminate):
                terminate()
            running.log_handle.close()

    payload["status"] = "FAILED" if failed else "SUCCEEDED"
    payload["finished_at"] = utc_now()
    payload["updated_at"] = utc_now()
    atomic_write_json(manifest_path, payload)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an inference-only V0/V1/V2 first-action critic comparison, "
            "including the V2 rollout-mean G-only ablation, without touching "
            "historical result bundles."
        )
    )
    parser.add_argument(
        "--stage", choices=("smoke", "development", "formal"), required=True
    )
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--repository", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--score-modes", nargs="+", choices=ALL_FORMAL_SCORE_MODES)
    parser.add_argument("--alpha", action="append", type=float, dest="alphas")
    parser.add_argument("--formal-selection")
    parser.add_argument("--gpus", nargs="*", default=())
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    repository = Path(args.repository).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    plan = resolve_stage_plan(
        stage=args.stage,
        variants=args.variants,
        score_modes=args.score_modes,
        alphas=args.alphas,
    )
    checkpoints = load_checkpoint_manifest(args.checkpoint_manifest)
    jobs = build_jobs(
        repository=repository,
        output_root=args.output_root,
        checkpoints=checkpoints,
        dataset=dataset,
        python=args.python,
        plan=plan,
    )
    return run_jobs(
        jobs=jobs,
        plan=plan,
        repository=repository,
        dataset=dataset,
        checkpoint_manifest=args.checkpoint_manifest,
        output_root=args.output_root,
        gpus=args.gpus,
        max_concurrency=args.max_concurrency,
        formal_selection=args.formal_selection,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
