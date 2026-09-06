#!/usr/bin/env python3
"""Run the locked V1-C3 E12 State-V + First-Q2 alpha=.1 Cube O25 cell."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tdwm.adapters.actor_free_td_lewm_v1_c3 import (
    STATE_V_FIRST_Q2_SCORE_MODE,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    STATE_V_FIRST_Q2_SCORE_DEFINITION,
    configure_actor_free_td_lewm_v1_c3_evaluation_mode,
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    _execution_metadata,
    v1_evaluation_protocol_label,
)

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_actor_free_td_lewm_first_action_comparison import (  # noqa: E402
    Job,
    StagePlan,
    alpha_slug,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
    run_jobs,
)

PRESPECIFIED_ALPHA = 0.1
EXPECTED_CHECKPOINT_SHA256 = (
    "5e240053d7c33fc016ef2ff64f3a4a79706dbe10dfde347d5c5f3cd45043e5b2"
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
EXPECTED_EPISODES = 50


def build_o25_job(
    *,
    repository: str | Path,
    output_root: str | Path,
    dataset: str | Path,
    checkpoint: str | Path,
    python: str,
) -> Job:
    repository = Path(repository).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    evaluator = repository / "scripts" / "evaluate_actor_free_td_lewm_v1_c3.py"
    config = (
        repository
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c3_cube_checkpoint_o25.yaml"
    )
    output = (
        output_root
        / "formal"
        / "o25"
        / "v1"
        / "c3"
        / STATE_V_FIRST_Q2_SCORE_MODE
        / f"alpha_{alpha_slug(PRESPECIFIED_ALPHA)}"
    )
    job_id = (
        f"v1__c3__o25__{STATE_V_FIRST_Q2_SCORE_MODE}"
        f"__alpha_{alpha_slug(PRESPECIFIED_ALPHA)}"
    )
    argv = (
        python,
        str(evaluator),
        "--config",
        str(config),
        "--dataset",
        str(dataset),
        "--checkpoint-path",
        str(checkpoint),
        "--score-mode",
        STATE_V_FIRST_Q2_SCORE_MODE,
        "--g-first-weight",
        format(PRESPECIFIED_ALPHA, ".17g"),
        "--output-dir",
        str(output),
    )
    return Job(
        job_id=job_id,
        stage="formal",
        version="v1",
        variant="c3",
        score_mode=STATE_V_FIRST_Q2_SCORE_MODE,
        alpha=PRESPECIFIED_ALPHA,
        checkpoint=str(checkpoint),
        config_path=str(config),
        output_dir=str(output),
        log_path=str(
            output_root / "formal" / "_launcher" / "jobs" / f"{job_id}.log"
        ),
        argv=argv,
    )


def validate_o25_job(job: Job) -> None:
    expected = {
        "stage": "formal",
        "version": "v1",
        "variant": "c3",
        "score_mode": STATE_V_FIRST_Q2_SCORE_MODE,
        "alpha": PRESPECIFIED_ALPHA,
    }
    for key, value in expected.items():
        if getattr(job, key) != value:
            raise ValueError(f"V1-C3 O25 job {key} must be {value!r}.")
    if not job.config_path.endswith(
        "actor_free_td_lewm_v1_c3_cube_checkpoint_o25.yaml"
    ):
        raise ValueError("V1-C3 O25 must use the audited O25 configuration.")
    valued_flags = (
        "--config",
        "--dataset",
        "--checkpoint-path",
        "--score-mode",
        "--g-first-weight",
        "--output-dir",
    )
    if any(job.argv.count(flag) != 1 for flag in valued_flags):
        raise ValueError("V1-C3 O25 evaluator arguments are ambiguous.")
    if "--smoke" in job.argv or "--pilot" in job.argv:
        raise ValueError("V1-C3 O25 launcher only permits the formal run.")
    values = {flag: job.argv[job.argv.index(flag) + 1] for flag in valued_flags}
    expected_evaluator = (
        Path(job.config_path).resolve().parents[2]
        / "scripts"
        / "evaluate_actor_free_td_lewm_v1_c3.py"
    )
    if (
        values["--config"] != job.config_path
        or values["--checkpoint-path"] != job.checkpoint
        or values["--score-mode"] != STATE_V_FIRST_Q2_SCORE_MODE
        or values["--g-first-weight"] != format(PRESPECIFIED_ALPHA, ".17g")
        or values["--output-dir"] != job.output_dir
        or Path(job.argv[1]).resolve() != expected_evaluator
    ):
        raise ValueError("V1-C3 O25 evaluator arguments changed from the job record.")


def _require_locked_checkpoint(checkpoint: Path) -> str:
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)
    actual = file_sha256(checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"V1-C3 checkpoint SHA-256 {actual} does not match the locked E12 artifact."
        )
    return actual


def write_checkpoint_manifest(output_root: Path, *, checkpoint: Path) -> Path:
    checkpoint_sha256 = _require_locked_checkpoint(checkpoint)
    payload = {
        "schema_version": 1,
        "purpose": "v1_c3_e12_state_v_plus_first_q2_alpha_0p1_formal_o25",
        "protocol_label": "o25",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "score_mode": STATE_V_FIRST_Q2_SCORE_MODE,
        "g_first_weight": PRESPECIFIED_ALPHA,
        "expected_selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "expected_selection_ranks_sha256": EXPECTED_SELECTION_RANKS_SHA256,
        "expected_action_normalization_sha256": (
            EXPECTED_ACTION_NORMALIZATION_SHA256
        ),
    }
    path = output_root / "checkpoint_manifest.json"
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError("Existing O25 manifest binds another C3 evaluation.")
    else:
        atomic_write_json(path, payload)
    return path


def _require_exact_execution_metadata(values: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "receding_horizon": 5,
        "executed_action_blocks_before_replanning": 5,
        "executed_environment_steps_before_replanning": 25,
        "executed_action_block": "all_five_blocks",
        "replanning": "every_five_action_blocks",
        "cem_execution": "execute_A1_through_A5_from_minimum_total_cost_plan",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"{label}.{key} must be {value!r}.")


def _require_exact_score_definition(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    expected = {
        **STATE_V_FIRST_Q2_SCORE_DEFINITION,
        "executed_action_block": "all_five_blocks",
        "replanning": "every_five_action_blocks",
    }
    if dict(value) != expected:
        raise ValueError(f"{label} is not the locked candidate-z-scored C3 score.")


def validate_o25_job_output(
    job: Job,
    *,
    expected_selection_file_sha256: str | None = None,
) -> dict[str, Any]:
    validate_o25_job(job)
    if expected_selection_file_sha256 not in (
        None,
        EXPECTED_SELECTION_FILE_SHA256,
    ):
        raise ValueError("The V1-C3 O25 selection lock cannot be overridden.")
    output = Path(job.output_dir)
    required = {
        name: output / name
        for name in (
            "results.json",
            "protocol_manifest.json",
            "episode_selection.json",
            "action_normalization.json",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{job.job_id} is missing {missing}.")
    results = read_json(required["results.json"])
    manifest = read_json(required["protocol_manifest.json"])
    for values, label in ((results, "results"), (manifest, "manifest")):
        for key, expected in {
            "evaluation_protocol": "O25",
            "protocol_label": "o25",
            "goal_offset": 25,
            "episode_budget": 50,
            "score_mode": STATE_V_FIRST_Q2_SCORE_MODE,
            "g_first_weight": PRESPECIFIED_ALPHA,
        }.items():
            if values.get(key) != expected:
                raise ValueError(f"{label}.{key} must be {expected!r}.")
        _require_exact_execution_metadata(values, label=label)
        _require_exact_score_definition(
            values.get("score_definition"), label=f"{label}.score_definition"
        )
    for key, expected in {
        "method": "actor_free_td_lewm_v1_c3",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c3",
        "implementation_version": "v1",
        "planning_horizon": 5,
        "smoke": False,
        "pilot": False,
    }.items():
        if results.get(key) != expected:
            raise ValueError(f"results.{key} must be {expected!r}.")

    metrics = results.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("results.metrics must be a mapping.")
    outcomes = metrics.get("episode_successes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EXPECTED_EPISODES
        or any(type(outcome) is not bool for outcome in outcomes)
    ):
        raise ValueError("V1-C3 O25 must contain exactly 50 Boolean outcomes.")
    success_count = sum(outcomes)
    success_rate = float(metrics.get("success_rate"))
    if not math.isclose(success_rate, success_count * 2.0):
        raise ValueError("V1-C3 O25 success rate disagrees with its outcomes.")

    protocol = manifest.get("protocol")
    formal_protocol = manifest.get("formal_protocol")
    if not isinstance(protocol, Mapping) or not isinstance(formal_protocol, Mapping):
        raise ValueError("V1-C3 O25 is missing its protocol audit mappings.")
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol(protocol)
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol(formal_protocol)
    if (
        v1_evaluation_protocol_label(protocol) != "o25"
        or v1_evaluation_protocol_label(formal_protocol) != "o25"
    ):
        raise ValueError("V1-C3 evaluator did not use the formal O25 envelope.")
    expected_protocol = configure_actor_free_td_lewm_v1_c3_evaluation_mode(
        formal_protocol,
        smoke=False,
        pilot=False,
        score_mode=STATE_V_FIRST_Q2_SCORE_MODE,
        g_first_weight=PRESPECIFIED_ALPHA,
    )
    if protocol != expected_protocol:
        raise ValueError("Configured V1-C3 O25 protocol differs from the locked score.")
    _require_exact_execution_metadata(
        _execution_metadata(protocol["planning"]), label="protocol.execution"
    )

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("manifest.checkpoint must be a mapping.")
    if (
        Path(str(checkpoint.get("path"))).resolve() != Path(job.checkpoint).resolve()
        or checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("epoch") != 12
        or checkpoint.get("logical_epoch") != 12
        or checkpoint.get("global_step") != 12_000
        or checkpoint.get("formal_completion_required") is not True
    ):
        raise ValueError("V1-C3 O25 did not use the locked final E12 checkpoint.")
    if file_sha256(job.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V1-C3 checkpoint changed during evaluation.")

    selection_file_sha256 = file_sha256(required["episode_selection.json"])
    if selection_file_sha256 != EXPECTED_SELECTION_FILE_SHA256:
        raise ValueError("V1-C3 O25 episode selection changed.")
    selection = read_json(required["episode_selection.json"])
    ranks = selection.get("valid_row_ranks")
    if (
        not isinstance(ranks, list)
        or len(ranks) != EXPECTED_EPISODES
        or any(type(rank) is not int or rank < 0 for rank in ranks)
        or len(set(ranks)) != EXPECTED_EPISODES
    ):
        raise ValueError("V1-C3 O25 selection ranks are invalid.")
    ranks_sha256 = canonical_json_sha256(ranks)
    if ranks_sha256 != EXPECTED_SELECTION_RANKS_SHA256:
        raise ValueError("V1-C3 O25 selection rank digest changed.")
    if manifest.get("selection") != selection:
        raise ValueError("V1-C3 O25 manifest selection differs from its file.")
    if (
        manifest.get("selection_sha256") != EXPECTED_SELECTION_FILE_SHA256
        or results.get("selection_sha256") != EXPECTED_SELECTION_FILE_SHA256
    ):
        raise ValueError("V1-C3 O25 recorded selection digest is incorrect.")

    action_sha256 = file_sha256(required["action_normalization.json"])
    if action_sha256 != EXPECTED_ACTION_NORMALIZATION_SHA256:
        raise ValueError("V1-C3 O25 action normalization changed.")
    return {
        "success_count": success_count,
        "success_rate": success_rate,
        "episode_successes": outcomes,
        "elapsed_seconds": results.get("elapsed_seconds"),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selection_path": str(required["episode_selection.json"]),
        "selection_file_sha256": selection_file_sha256,
        "valid_row_ranks": ranks,
        "valid_row_ranks_sha256": ranks_sha256,
        "selection_sha256": ranks_sha256,
        "action_normalization_sha256": action_sha256,
        "results_path": str(required["results.json"]),
        "manifest_path": str(required["protocol_manifest.json"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked V1-C3 E12 candidate-z-scored State-V + First-Q2 "
            "alpha=.1 formal Cube O25 evaluation."
        )
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
    job = build_o25_job(
        repository=repository,
        output_root=output_root,
        dataset=dataset,
        checkpoint=checkpoint,
        python=args.python,
    )
    validate_o25_job(job)
    plan = StagePlan(
        stage="formal",
        versions=("v1",),
        variants=("c3",),
        score_modes=(STATE_V_FIRST_Q2_SCORE_MODE,),
        v2_only_score_modes=(),
        alphas=(PRESPECIFIED_ALPHA,),
    )
    return run_jobs(
        jobs=(job,),
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
            "launcher": "actor_free_td_lewm_v1_c3_first_q2_o25",
            "protocol_label": "o25",
            "evaluation_protocol": "O25",
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "expected_selection_ranks_sha256": EXPECTED_SELECTION_RANKS_SHA256,
            "expected_action_normalization_sha256": (
                EXPECTED_ACTION_NORMALIZATION_SHA256
            ),
            "historical_o50_success_count": 31,
            "historical_o50_success_rate": 62.0,
            "historical_o50_role": "exploratory_same_o50_alpha_selected_peak",
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
