#!/usr/bin/env python3
"""Run the isolated V1-C3/V1-C first-action alpha sensitivity sweep."""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)
EXPECTED_EPISODES = 50
EXPECTED_SELECTION_FILE_SHA256 = (
    "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
)
FAMILIES = (
    ("c3_raw", "c3", "state_v_plus_first_q"),
    ("c3_zscore", "c3", "state_v_plus_first_q2"),
    ("v1_c_first_q", "v1_c", "f_plus_g_first"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_alpha(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("alpha must be finite and non-negative.")
    alpha = float(value)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative.")
    return 0.0 if alpha == 0.0 else alpha


def _alpha_slug(value: float) -> str:
    encoded = format(_normalize_alpha(value), ".15g").lower()
    return encoded.replace("+", "").replace("-", "m").replace(".", "p")


def _unique_alphas(values: Sequence[float]) -> tuple[float, ...]:
    alphas = tuple(_normalize_alpha(value) for value in values)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alpha values must be a non-empty unique list.")
    return alphas


@dataclass(frozen=True)
class SweepJob:
    job_id: str
    family: str
    model: str
    score_mode: str
    alpha: float
    checkpoint: str
    checkpoint_sha256: str
    config_path: str
    output_dir: str
    log_path: str
    argv: tuple[str, ...]


def build_jobs(
    *,
    repository: str | Path,
    output_root: str | Path,
    dataset: str | Path,
    python: str | Path,
    c3_checkpoint: str | Path,
    v1_c_checkpoint: str | Path,
    alphas: Sequence[float],
) -> list[SweepJob]:
    """Build three matched full-O50 alpha grids without training anything."""

    repository = Path(repository).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    # Preserve a virtual-environment interpreter path. Resolving its symlink can
    # select the base interpreter and lose the environment's installed packages.
    python = Path(os.path.abspath(Path(python).expanduser()))
    checkpoints = {
        "c3": Path(c3_checkpoint).expanduser().resolve(),
        "v1_c": Path(v1_c_checkpoint).expanduser().resolve(),
    }
    for path in (repository, dataset, python, *checkpoints.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_hashes = {
        model: _file_sha256(checkpoint) for model, checkpoint in checkpoints.items()
    }
    selected_alphas = _unique_alphas(alphas)
    jobs: list[SweepJob] = []
    for family, model, score_mode in FAMILIES:
        if model == "c3":
            evaluator = repository / "scripts/evaluate_actor_free_td_lewm_v1_c3.py"
            config = (
                repository
                / "configs/experiment/actor_free_td_lewm_v1_c3_cube_checkpoint_o50.yaml"
            )
        else:
            evaluator = repository / "scripts/evaluate_actor_free_td_lewm_v1_c.py"
            config = (
                repository
                / "configs/experiment/actor_free_td_lewm_v1_c_cube_checkpoint_o50.yaml"
            )
        for path in (evaluator, config):
            if not path.is_file():
                raise FileNotFoundError(path)
        for alpha in selected_alphas:
            slug = _alpha_slug(alpha)
            job_id = f"{family}__alpha_{slug}"
            output_dir = output_root / family / f"alpha_{slug}"
            log_path = output_root / "_launcher/jobs" / f"{job_id}.log"
            argv = (
                str(python),
                str(evaluator),
                "--config",
                str(config),
                "--dataset",
                str(dataset),
                "--checkpoint-path",
                str(checkpoints[model]),
                "--score-mode",
                score_mode,
                "--g-first-weight",
                format(alpha, ".17g"),
                "--output-dir",
                str(output_dir),
            )
            jobs.append(
                SweepJob(
                    job_id=job_id,
                    family=family,
                    model=model,
                    score_mode=score_mode,
                    alpha=alpha,
                    checkpoint=str(checkpoints[model]),
                    checkpoint_sha256=checkpoint_hashes[model],
                    config_path=str(config),
                    output_dir=str(output_dir),
                    log_path=str(log_path),
                    argv=argv,
                )
            )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("Sweep job identifiers are not unique.")
    if len({job.output_dir for job in jobs}) != len(jobs):
        raise AssertionError("Sweep output directories are not unique.")
    return jobs


def _validate_score_definition(job: SweepJob, definition: Any) -> None:
    if not isinstance(definition, Mapping):
        raise ValueError(f"{job.job_id} has no score_definition mapping.")
    formula = definition.get("formula")
    if not isinstance(formula, str) or "g_first_weight" not in formula:
        raise ValueError(f"{job.job_id} score formula does not expose alpha.")
    if job.family == "c3_raw":
        if definition.get("normalization") != "none_raw_scores":
            raise ValueError(f"{job.job_id} is not the raw C3 score.")
    elif job.family == "c3_zscore":
        if definition.get("normalization") != "population_z_score":
            raise ValueError(f"{job.job_id} is not the z-scored C3 score.")
    elif "zscore_samples" in formula:
        raise ValueError(f"{job.job_id} unexpectedly uses candidate z-scoring.")


def validate_job_output(job: SweepJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    required = (
        output_dir / "results.json",
        output_dir / "protocol_manifest.json",
        output_dir / "episode_selection.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{job.job_id} is missing {missing}.")
    results = _read_json(required[0])
    manifest = _read_json(required[1])
    expected_results = {
        "score_mode": job.score_mode,
        "g_first_weight": job.alpha,
        "smoke": False,
        "pilot": False,
        "planning_horizon": 5,
    }
    for key, expected in expected_results.items():
        if results.get(key) != expected:
            raise ValueError(
                f"{job.job_id} results.{key}={results.get(key)!r}; "
                f"expected {expected!r}."
            )
    for key, expected in (
        ("score_mode", job.score_mode),
        ("g_first_weight", job.alpha),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"{job.job_id} manifest.{key}={manifest.get(key)!r}; "
                f"expected {expected!r}."
            )
    _validate_score_definition(job, results.get("score_definition"))
    _validate_score_definition(job, manifest.get("score_definition"))
    metrics = results.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{job.job_id} has no metrics mapping.")
    outcomes = metrics.get("episode_successes")
    if not isinstance(outcomes, list) or len(outcomes) != EXPECTED_EPISODES:
        raise ValueError(f"{job.job_id} must contain exactly 50 outcomes.")
    if any(not isinstance(outcome, bool) for outcome in outcomes):
        raise ValueError(f"{job.job_id} outcomes must be Boolean.")
    success_count = sum(outcomes)
    success_rate = float(metrics.get("success_rate"))
    if not math.isclose(success_rate, success_count * 100.0 / EXPECTED_EPISODES):
        raise ValueError(f"{job.job_id} success_rate disagrees with its outcomes.")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"{job.job_id} manifest has no configured protocol.")
    inference = protocol.get("inference_objective")
    if not isinstance(inference, Mapping):
        raise ValueError(f"{job.job_id} protocol has no inference objective.")
    if inference.get("score_mode") != job.score_mode:
        raise ValueError(f"{job.job_id} protocol score mode changed.")
    if inference.get("g_first_weight") != job.alpha:
        raise ValueError(f"{job.job_id} protocol alpha changed.")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{job.job_id} manifest has no checkpoint mapping.")
    if Path(str(checkpoint.get("path"))).resolve() != Path(job.checkpoint):
        raise ValueError(f"{job.job_id} used a different checkpoint path.")
    if checkpoint.get("sha256") != job.checkpoint_sha256:
        raise ValueError(f"{job.job_id} used a different checkpoint hash.")
    selection_sha256 = _file_sha256(required[2])
    if selection_sha256 != EXPECTED_SELECTION_FILE_SHA256:
        raise ValueError(f"{job.job_id} used a different O50 selection.")
    # C3 records this redundant digest in results.json; the older V1-C
    # evaluator does not. The authoritative selection file was hashed above.
    reported_selection_sha256 = results.get("selection_sha256")
    if (
        reported_selection_sha256 is not None
        and reported_selection_sha256 != EXPECTED_SELECTION_FILE_SHA256
    ):
        raise ValueError(f"{job.job_id} results selection hash changed.")
    return {
        "family": job.family,
        "model": job.model,
        "score_mode": job.score_mode,
        "alpha": job.alpha,
        "success_count": success_count,
        "success_rate": success_rate,
        "episode_successes": outcomes,
        "elapsed_seconds": results.get("elapsed_seconds"),
        "checkpoint": job.checkpoint,
        "checkpoint_sha256": job.checkpoint_sha256,
        "selection_file_sha256": selection_sha256,
        "results_path": str(required[0]),
        "manifest_path": str(required[1]),
    }


def _write_summary(output_root: Path, evidence: Mapping[str, Mapping[str, Any]]) -> None:
    families: dict[str, Any] = {}
    for family, _, score_mode in FAMILIES:
        cells = [dict(cell) for cell in evidence.values() if cell["family"] == family]
        cells.sort(key=lambda cell: (-cell["success_count"], cell["alpha"]))
        best_count = cells[0]["success_count"]
        families[family] = {
            "score_mode": score_mode,
            "best_success_count": best_count,
            "best_success_rate": best_count * 2.0,
            "best_alphas": [
                cell["alpha"] for cell in cells if cell["success_count"] == best_count
            ],
            "ranking": cells,
        }
    summary = {
        "schema_version": 1,
        "analysis_role": "exploratory_same_o50_alpha_sensitivity",
        "alpha_selection_performed": True,
        "selection_bias_warning": (
            "The same locked O50 set is used for alpha comparison; a winning alpha "
            "requires confirmation on a disjoint selection before an unbiased claim."
        ),
        "episodes_per_cell": EXPECTED_EPISODES,
        "selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "families": families,
        "created_at": _utc_now(),
    }
    _atomic_write_json(output_root / "alpha_sweep_summary.json", summary)
    csv_path = output_root / "alpha_sweep_results.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "family",
                "model",
                "score_mode",
                "alpha",
                "success_count",
                "success_rate",
                "elapsed_seconds",
                "checkpoint_sha256",
                "selection_file_sha256",
                "results_path",
                "manifest_path",
            ),
        )
        writer.writeheader()
        for job_id in sorted(evidence):
            writer.writerow(
                {
                    key: value
                    for key, value in evidence[job_id].items()
                    if key in writer.fieldnames
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, csv_path)


@dataclass
class ActiveJob:
    job: SweepJob
    process: Any
    log_handle: Any
    gpu: str | None


def run_jobs(
    *,
    jobs: Sequence[SweepJob],
    output_root: str | Path,
    gpus: Sequence[str],
    max_concurrency: int,
    poll_seconds: float,
    popen: Callable[..., Any] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive.")
    if poll_seconds < 0.0:
        raise ValueError("poll_seconds must be non-negative.")
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}.")
    output_root.mkdir(parents=True, exist_ok=True)
    launcher_root = output_root / "_launcher"
    launcher_root.mkdir()
    manifest_path = launcher_root / "launcher_manifest.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "launcher": "actor_free_td_lewm_v1_c3_first_q_alpha_sweep",
        "inference_only": True,
        "training_performed": False,
        "analysis_role": "exploratory_same_o50_alpha_sensitivity",
        "alphas": sorted({job.alpha for job in jobs}),
        "families": [family for family, _, _ in FAMILIES],
        "gpus": list(gpus),
        "max_concurrency": max_concurrency,
        "selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "status": "RUNNING",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "jobs": {
            job.job_id: {**asdict(job), "argv": list(job.argv), "state": "PENDING"}
            for job in jobs
        },
    }
    _atomic_write_json(manifest_path, payload)
    pending = list(jobs)
    active: dict[str, ActiveJob] = {}
    evidence: dict[str, dict[str, Any]] = {}
    failed = False
    launched = 0
    try:
        while pending or active:
            while pending and len(active) < max_concurrency and not failed:
                job = pending.pop(0)
                Path(job.output_dir).mkdir(parents=True, exist_ok=False)
                log_path = Path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("x", encoding="utf-8")
                gpu = str(gpus[launched % len(gpus)]) if gpus else None
                environment = os.environ.copy()
                if gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment.update(
                    {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    }
                )
                try:
                    process = popen(
                        list(job.argv),
                        cwd=str(Path(job.argv[1]).resolve().parents[1]),
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
                active[job.job_id] = ActiveJob(job, process, log_handle, gpu)
                payload["jobs"][job.job_id].update(
                    {
                        "state": "RUNNING",
                        "pid": getattr(process, "pid", None),
                        "gpu": gpu,
                        "started_at": _utc_now(),
                    }
                )
                launched += 1
                payload["updated_at"] = _utc_now()
                _atomic_write_json(manifest_path, payload)
            completed_any = False
            for job_id, running in list(active.items()):
                return_code = running.process.poll()
                if return_code is None:
                    continue
                completed_any = True
                running.log_handle.close()
                del active[job_id]
                state = payload["jobs"][job_id]
                state.update(
                    {"exit_code": int(return_code), "finished_at": _utc_now()}
                )
                if return_code != 0:
                    state["state"] = "FAILED"
                    failed = True
                    continue
                try:
                    cell = validate_job_output(running.job)
                except Exception as error:
                    state.update({"state": "FAILED_VALIDATION", "error": str(error)})
                    failed = True
                else:
                    evidence[job_id] = cell
                    state.update({"state": "SUCCEEDED", "evidence": cell})
            payload["updated_at"] = _utc_now()
            _atomic_write_json(manifest_path, payload)
            if failed and not active:
                break
            if active and not completed_any:
                sleeper(poll_seconds)
        if pending:
            for job in pending:
                payload["jobs"][job.job_id]["state"] = "SKIPPED_AFTER_FAILURE"
        if not failed:
            if len(evidence) != len(jobs):
                raise RuntimeError("Not every alpha-sweep job produced evidence.")
            _write_summary(output_root, evidence)
    except KeyboardInterrupt:
        failed = True
        payload["interrupted"] = True
        for running in active.values():
            terminate = getattr(running.process, "terminate", None)
            if callable(terminate):
                terminate()
            running.log_handle.close()
    except Exception as error:
        failed = True
        payload["launcher_error"] = str(error)
    payload["status"] = "FAILED" if failed else "SUCCEEDED"
    payload["finished_at"] = _utc_now()
    payload["updated_at"] = _utc_now()
    _atomic_write_json(manifest_path, payload)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched full-O50 alpha grids for C3 raw First-Q, C3 z-scored "
            "First-Q2, and the original V1-C First-Q."
        )
    )
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--c3-checkpoint", required=True)
    parser.add_argument("--v1-c-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--repository", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--alpha", action="append", type=float, dest="alphas")
    parser.add_argument("--gpus", nargs="*", default=())
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    alphas = args.alphas or DEFAULT_ALPHAS
    jobs = build_jobs(
        repository=args.repository,
        output_root=args.output_root,
        dataset=args.dataset,
        python=args.python,
        c3_checkpoint=args.c3_checkpoint,
        v1_c_checkpoint=args.v1_c_checkpoint,
        alphas=alphas,
    )
    return run_jobs(
        jobs=jobs,
        output_root=args.output_root,
        gpus=args.gpus,
        max_concurrency=args.max_concurrency,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
