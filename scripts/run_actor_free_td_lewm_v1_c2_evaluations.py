#!/usr/bin/env python3
"""Run the locked endpoint O50 matrix for V1-C2 and First-Q2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_actor_free_td_lewm_first_action_comparison import (  # noqa: E402
    FIRST_ACTION_MODES,
    Job,
    StagePlan,
    _job_output_dir,
    alpha_slug,
    atomic_write_json,
    evaluation_config_path,
    file_sha256,
    normalize_alpha,
    run_jobs,
)

C2_SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "f_plus_g_first_q2",
    "g_only_f_rollout_mean",
)
V1_C_REFERENCE_SCORE_MODES = ("f_plus_g_first_q2",)
PRESPECIFIED_ALPHA = 0.25


def _build_job(
    *,
    repository: Path,
    output_root: Path,
    dataset: Path,
    python: str,
    stage: str,
    variant: str,
    checkpoint: Path,
    score_mode: str,
    alpha: float,
) -> Job:
    mode_alpha = alpha if score_mode in FIRST_ACTION_MODES else None
    output = _job_output_dir(
        stage_root=output_root / stage,
        version="v1",
        variant=variant,
        score_mode=score_mode,
        alpha=mode_alpha,
    )
    evaluator = repository / "scripts" / f"evaluate_actor_free_td_lewm_v1_{variant}.py"
    config = evaluation_config_path(
        repository,
        version="v1",
        variant=variant,
        score_mode=score_mode,
    )
    alpha_suffix = f"__alpha_{alpha_slug(mode_alpha)}" if mode_alpha is not None else ""
    job_id = f"v1__{variant}__{score_mode}{alpha_suffix}"
    argv = [
        python,
        str(evaluator),
        "--config",
        str(config),
        "--dataset",
        str(dataset),
        "--checkpoint-path",
        str(checkpoint),
        "--score-mode",
        score_mode,
        "--output-dir",
        str(output),
    ]
    if mode_alpha is not None:
        argv.extend(("--g-first-weight", format(mode_alpha, ".17g")))
    if stage == "smoke":
        argv.append("--smoke")
    return Job(
        job_id=job_id,
        stage=stage,
        version="v1",
        variant=variant,
        score_mode=score_mode,
        alpha=mode_alpha,
        checkpoint=str(checkpoint),
        config_path=str(config),
        output_dir=str(output),
        log_path=str(output_root / stage / "_launcher" / "jobs" / f"{job_id}.log"),
        argv=tuple(argv),
    )


def build_c2_evaluation_jobs(
    *,
    repository: str | Path,
    output_root: str | Path,
    dataset: str | Path,
    python: str,
    c2_checkpoint: str | Path,
    v1_c_checkpoint: str | Path,
    stage: str,
    alpha: float = PRESPECIFIED_ALPHA,
) -> list[Job]:
    if stage not in {"smoke", "formal"}:
        raise ValueError("V1-C2 endpoint evaluation stage must be smoke or formal.")
    alpha = normalize_alpha(alpha)
    repository = Path(repository).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    checkpoints = {
        "c2": Path(c2_checkpoint).expanduser().resolve(),
        "c": Path(v1_c_checkpoint).expanduser().resolve(),
    }
    jobs = [
        _build_job(
            repository=repository,
            output_root=output_root,
            dataset=dataset,
            python=python,
            stage=stage,
            variant="c2",
            checkpoint=checkpoints["c2"],
            score_mode=score_mode,
            alpha=alpha,
        )
        for score_mode in C2_SCORE_MODES
    ]
    jobs.extend(
        _build_job(
            repository=repository,
            output_root=output_root,
            dataset=dataset,
            python=python,
            stage=stage,
            variant="c",
            checkpoint=checkpoints["c"],
            score_mode=score_mode,
            alpha=alpha,
        )
        for score_mode in V1_C_REFERENCE_SCORE_MODES
    )
    return jobs


def _write_checkpoint_manifest(
    output_root: Path,
    *,
    c2_checkpoint: Path,
    v1_c_checkpoint: Path,
) -> Path:
    manifest = {
        "schema_version": 1,
        "purpose": "v1_c2_endpoint_o50_and_v1_c_first_q2_reference",
        "v1": {
            "c": {
                "path": str(v1_c_checkpoint),
                "sha256": file_sha256(v1_c_checkpoint),
            },
            "c2": {
                "path": str(c2_checkpoint),
                "sha256": file_sha256(c2_checkpoint),
            },
        },
    }
    path = output_root / "checkpoint_manifest.json"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise RuntimeError("Existing C2 checkpoint manifest binds other files.")
    else:
        atomic_write_json(path, manifest)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run six V1-C2 endpoint score modes plus the V1-C First-Q2 "
            "reference on one identical Cube selection."
        )
    )
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--c2-checkpoint", required=True)
    parser.add_argument("--v1-c-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--repository", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--alpha", type=float, default=PRESPECIFIED_ALPHA)
    parser.add_argument("--gpus", nargs="*", default=())
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--expected-selection-file-sha256")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(args.repository).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    c2_checkpoint = Path(args.c2_checkpoint).expanduser().resolve()
    v1_c_checkpoint = Path(args.v1_c_checkpoint).expanduser().resolve()
    for path in (repository, dataset, c2_checkpoint, v1_c_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = _write_checkpoint_manifest(
        output_root,
        c2_checkpoint=c2_checkpoint,
        v1_c_checkpoint=v1_c_checkpoint,
    )
    alpha = normalize_alpha(args.alpha)
    jobs = build_c2_evaluation_jobs(
        repository=repository,
        output_root=output_root,
        dataset=dataset,
        python=args.python,
        c2_checkpoint=c2_checkpoint,
        v1_c_checkpoint=v1_c_checkpoint,
        stage=args.stage,
        alpha=alpha,
    )
    plan = StagePlan(
        stage=args.stage,
        versions=("v1",),
        variants=("c2", "c"),
        score_modes=C2_SCORE_MODES,
        v2_only_score_modes=(),
        alphas=(alpha,),
    )
    return run_jobs(
        jobs=jobs,
        plan=plan,
        repository=repository,
        dataset=dataset,
        checkpoint_manifest=checkpoint_manifest,
        output_root=output_root,
        gpus=args.gpus,
        max_concurrency=args.max_concurrency,
        formal_selection=None,
        expected_selection_file_sha256=args.expected_selection_file_sha256,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
