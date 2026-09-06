#!/usr/bin/env python3
"""Run the six locked V1-C E10 score modes under the formal Cube O25 protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tdwm.evaluation.actor_free_td_lewm_v1_c import (
    validate_actor_free_td_lewm_v1_c_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    v1_evaluation_protocol_label,
)

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
    file_sha256,
    read_json,
    run_jobs,
    validate_job_output,
)

SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "f_plus_g_first_q2",
    "g_only_f_rollout_mean",
)
PRESPECIFIED_ALPHA = 0.25
EXPECTED_CHECKPOINT_SHA256 = (
    "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3"
)
EXPECTED_SELECTION_FILE_SHA256 = (
    "56546fe8725ce0e4670f308c5b325bd64ff2a792373add8c20ddbcab02da6b37"
)
EXPECTED_SELECTION_RANKS_SHA256 = (
    "72af45d4bad65a25288c5d405072d18ab5c0b4f0b67ddc970ac3f344b3c22fd9"
)
EXPECTED_ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)


def build_o25_jobs(
    *,
    repository: str | Path,
    output_root: str | Path,
    dataset: str | Path,
    checkpoint: str | Path,
    python: str,
) -> list[Job]:
    repository = Path(repository).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    evaluator = repository / "scripts" / "evaluate_actor_free_td_lewm_v1_c.py"
    config = (
        repository
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c_cube_checkpoint_o25.yaml"
    )
    jobs: list[Job] = []
    for score_mode in SCORE_MODES:
        alpha = PRESPECIFIED_ALPHA if score_mode in FIRST_ACTION_MODES else None
        output = _job_output_dir(
            stage_root=output_root / "formal" / "o25",
            version="v1",
            variant="c",
            score_mode=score_mode,
            alpha=alpha,
        )
        suffix = f"__alpha_{alpha_slug(alpha)}" if alpha is not None else ""
        job_id = f"v1__c__o25__{score_mode}{suffix}"
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
        if alpha is not None:
            argv.extend(("--g-first-weight", format(alpha, ".17g")))
        jobs.append(
            Job(
                job_id=job_id,
                stage="formal",
                version="v1",
                variant="c",
                score_mode=score_mode,
                alpha=alpha,
                checkpoint=str(checkpoint),
                config_path=str(config),
                output_dir=str(output),
                log_path=str(
                    output_root / "formal" / "_launcher" / "jobs" / f"{job_id}.log"
                ),
                argv=tuple(argv),
            )
        )
    if len(jobs) != 6 or len({job.output_dir for job in jobs}) != 6:
        raise AssertionError("The V1-C O25 launcher must produce six isolated jobs.")
    return jobs


def validate_o25_job_set(jobs: Sequence[Job]) -> None:
    if len(jobs) != 6 or {job.score_mode for job in jobs} != set(SCORE_MODES):
        raise ValueError("The formal O25 launch must contain each of six modes once.")
    if len({job.score_mode for job in jobs}) != len(jobs):
        raise ValueError("The formal O25 launch contains a duplicate score mode.")
    if len({job.job_id for job in jobs}) != 6 or len(
        {job.output_dir for job in jobs}
    ) != 6:
        raise ValueError("O25 job identifiers and output directories must be unique.")
    shared_python: set[str] = set()
    shared_dataset: set[str] = set()
    shared_checkpoint: set[str] = set()
    shared_config: set[str] = set()
    shared_evaluator: set[str] = set()
    for job in jobs:
        expected_alpha = (
            PRESPECIFIED_ALPHA if job.score_mode in FIRST_ACTION_MODES else None
        )
        if (
            job.stage != "formal"
            or job.version != "v1"
            or job.variant != "c"
            or job.alpha != expected_alpha
            or not job.config_path.endswith(
                "actor_free_td_lewm_v1_c_cube_checkpoint_o25.yaml"
            )
        ):
            raise ValueError(f"{job.job_id} violates the locked V1-C O25 job set.")
        valued_flags = (
            "--config",
            "--dataset",
            "--checkpoint-path",
            "--score-mode",
            "--output-dir",
        )
        if any(job.argv.count(flag) != 1 for flag in valued_flags):
            raise ValueError(f"{job.job_id} has an ambiguous evaluator command.")
        if "--smoke" in job.argv or "--pilot" in job.argv:
            raise ValueError(f"{job.job_id} is not a formal-only command.")
        try:
            values = {
                flag: job.argv[job.argv.index(flag) + 1]
                for flag in valued_flags
            }
        except (ValueError, IndexError) as error:
            raise ValueError(f"{job.job_id} has an incomplete evaluator command.") from error
        expected_evaluator = (
            Path(job.config_path).resolve().parents[2]
            / "scripts"
            / "evaluate_actor_free_td_lewm_v1_c.py"
        )
        if (
            values["--config"] != job.config_path
            or values["--checkpoint-path"] != job.checkpoint
            or values["--score-mode"] != job.score_mode
            or values["--output-dir"] != job.output_dir
            or Path(job.argv[1]).resolve() != expected_evaluator
        ):
            raise ValueError(f"{job.job_id} evaluator arguments differ from its manifest.")
        if expected_alpha is None:
            if "--g-first-weight" in job.argv:
                raise ValueError(f"{job.job_id} unexpectedly sets alpha.")
        elif job.argv.count("--g-first-weight") != 1:
            raise ValueError(f"{job.job_id} must set the fixed alpha exactly once.")
        elif job.argv[job.argv.index("--g-first-weight") + 1] != format(
            PRESPECIFIED_ALPHA, ".17g"
        ):
            raise ValueError(f"{job.job_id} changed the fixed first-action alpha.")
        shared_python.add(job.argv[0])
        shared_dataset.add(values["--dataset"])
        shared_checkpoint.add(values["--checkpoint-path"])
        shared_config.add(values["--config"])
        shared_evaluator.add(job.argv[1])
    if any(
        len(values) != 1
        for values in (
            shared_python,
            shared_dataset,
            shared_checkpoint,
            shared_config,
            shared_evaluator,
        )
    ):
        raise ValueError("All six O25 jobs must share one exact evaluator and input set.")


def _require_locked_checkpoint(checkpoint: Path) -> str:
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)
    actual = file_sha256(checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"V1-C checkpoint SHA-256 {actual} does not match the locked E10 artifact."
        )
    return actual


def write_checkpoint_manifest(output_root: Path, *, checkpoint: Path) -> Path:
    checkpoint_sha256 = _require_locked_checkpoint(checkpoint)
    payload = {
        "schema_version": 1,
        "purpose": "v1_c_e10_six_mode_formal_o25",
        "protocol_label": "o25",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "expected_selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "expected_selection_ranks_sha256": EXPECTED_SELECTION_RANKS_SHA256,
        "expected_action_normalization_sha256": (
            EXPECTED_ACTION_NORMALIZATION_SHA256
        ),
    }
    path = output_root / "checkpoint_manifest.json"
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError("Existing O25 checkpoint manifest binds other inputs.")
    else:
        atomic_write_json(path, payload)
    return path


def _require_exact_execution_metadata(
    values: Mapping[str, Any], *, job: Job, receding_horizon: int
) -> None:
    expected = {
        "receding_horizon": receding_horizon,
        "executed_action_blocks_before_replanning": receding_horizon,
        "executed_environment_steps_before_replanning": receding_horizon * 5,
        "executed_action_block": (
            "first_block_only" if receding_horizon == 1 else "all_five_blocks"
        ),
        "replanning": (
            "every_action_block"
            if receding_horizon == 1
            else "every_five_action_blocks"
        ),
        "cem_execution": (
            "execute_A1_from_minimum_total_cost_plan"
            if receding_horizon == 1
            else "execute_A1_through_A5_from_minimum_total_cost_plan"
        ),
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(
                f"{job.job_id} {key}={values.get(key)!r}, expected {expected_value!r}."
            )


def validate_o25_job_output(
    job: Job,
    *,
    expected_selection_file_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_selection_file_sha256 not in (
        None,
        EXPECTED_SELECTION_FILE_SHA256,
    ):
        raise ValueError("The V1-C O25 selection lock cannot be overridden.")
    evidence = validate_job_output(
        job,
        expected_selection_file_sha256=EXPECTED_SELECTION_FILE_SHA256,
    )
    if evidence.get("valid_row_ranks_sha256") != EXPECTED_SELECTION_RANKS_SHA256:
        raise ValueError(f"{job.job_id} O25 selection ranks changed.")
    output = Path(job.output_dir)
    results = read_json(output / "results.json")
    successes = results.get("metrics", {}).get("episode_successes")
    if (
        not isinstance(successes, list)
        or len(successes) != 50
        or any(type(success) is not bool for success in successes)
    ):
        raise ValueError(
            f"{job.job_id} must contain exactly 50 boolean episode outcomes."
        )
    manifest = read_json(output / "protocol_manifest.json")
    protocol = manifest.get("protocol")
    formal_protocol = manifest.get("formal_protocol")
    if not isinstance(protocol, Mapping) or not isinstance(formal_protocol, Mapping):
        raise ValueError(f"{job.job_id} is missing its protocol audit mappings.")
    validate_actor_free_td_lewm_v1_c_evaluation_protocol(protocol)
    validate_actor_free_td_lewm_v1_c_evaluation_protocol(formal_protocol)
    if (
        v1_evaluation_protocol_label(protocol) != "o25"
        or v1_evaluation_protocol_label(formal_protocol) != "o25"
    ):
        raise ValueError(f"{job.job_id} did not run the formal O25 protocol.")
    if (
        manifest.get("protocol_label") != "o25"
        or results.get("protocol_label") != "o25"
        or manifest.get("evaluation_protocol") != "O25"
        or results.get("evaluation_protocol") != "O25"
        or manifest.get("goal_offset") != 25
        or results.get("goal_offset") != 25
        or manifest.get("episode_budget") != 50
        or results.get("episode_budget") != 50
    ):
        raise ValueError(f"{job.job_id} has an incorrect output protocol label.")
    planning = protocol["planning"]
    receding_horizon = 1 if job.score_mode == "g_only" else 5
    if planning.get("horizon") != (1 if job.score_mode == "g_only" else 5):
        raise ValueError(f"{job.job_id} has an incorrect O25 planning horizon.")
    _require_exact_execution_metadata(
        results,
        job=job,
        receding_horizon=receding_horizon,
    )
    _require_exact_execution_metadata(
        manifest,
        job=job,
        receding_horizon=receding_horizon,
    )
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError(f"{job.job_id} used a different V1-C checkpoint.")
    if file_sha256(Path(job.checkpoint)) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{job.job_id} checkpoint changed during evaluation.")
    selection = read_json(output / "episode_selection.json")
    if manifest.get("selection") != selection:
        raise ValueError(f"{job.job_id} manifest selection differs from its file.")
    action_path = output / "action_normalization.json"
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    action_sha256 = file_sha256(action_path)
    if action_sha256 != EXPECTED_ACTION_NORMALIZATION_SHA256:
        raise ValueError(f"{job.job_id} action normalization hash changed.")
    evidence.update(
        {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "action_normalization_path": str(action_path),
            "action_normalization_sha256": action_sha256,
            "protocol_label": "o25",
        }
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked six-mode V1-C E10 formal Cube O25 evaluation."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--repository", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="*", default=())
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(args.repository).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not repository.is_dir() or not dataset.exists():
        raise FileNotFoundError(repository if not repository.is_dir() else dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = write_checkpoint_manifest(
        output_root,
        checkpoint=checkpoint,
    )
    jobs = build_o25_jobs(
        repository=repository,
        output_root=output_root,
        dataset=dataset,
        checkpoint=checkpoint,
        python=args.python,
    )
    validate_o25_job_set(jobs)
    plan = StagePlan(
        stage="formal",
        versions=("v1",),
        variants=("c",),
        score_modes=SCORE_MODES,
        v2_only_score_modes=(),
        alphas=(PRESPECIFIED_ALPHA,),
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
        expected_selection_file_sha256=EXPECTED_SELECTION_FILE_SHA256,
        poll_seconds=args.poll_seconds,
        job_output_validator=validate_o25_job_output,
        launcher_metadata={
            "launcher": "actor_free_td_lewm_v1_c_o25_evaluations",
            "protocol_label": "o25",
            "evaluation_protocol": "O25",
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "expected_selection_ranks_sha256": EXPECTED_SELECTION_RANKS_SHA256,
            "expected_action_normalization_sha256": (
                EXPECTED_ACTION_NORMALIZATION_SHA256
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
