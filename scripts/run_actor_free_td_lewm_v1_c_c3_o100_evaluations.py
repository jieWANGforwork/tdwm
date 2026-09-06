#!/usr/bin/env python3
"""Run the seven locked V1-C/C3 inference cells under Cube O100."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tdwm.adapters.actor_free_td_lewm_v1_c3 import (
    STATE_V_FIRST_Q2_SCORE_MODE,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c import (
    validate_actor_free_td_lewm_v1_c_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
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
    FIRST_ACTION_MODES,
    Job,
    StagePlan,
    alpha_slug,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
    run_jobs,
    validate_job_output,
)

C_SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "f_plus_g_first_q2",
    "g_only_f_rollout_mean",
)
C3_SCORE_MODES = (STATE_V_FIRST_Q2_SCORE_MODE,)
C_FIRST_ACTION_ALPHA = 0.25
C3_FIRST_ACTION_ALPHA = 0.1
EXPECTED_C_CHECKPOINT_SHA256 = (
    "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3"
)
EXPECTED_C3_CHECKPOINT_SHA256 = (
    "5e240053d7c33fc016ef2ff64f3a4a79706dbe10dfde347d5c5f3cd45043e5b2"
)
EXPECTED_SELECTION_FILE_SHA256 = (
    "8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c"
)
EXPECTED_SELECTION_RANKS_SHA256 = (
    "36994b1ab36656666ff91b379a59829c4b2af150b1f4ed23d409deb5cca9654e"
)
EXPECTED_ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)
EXPECTED_EPISODES = 50
EXPECTED_EPISODE_BUDGET = 200


def _alpha_for(variant: str, score_mode: str) -> float | None:
    if variant == "c":
        return C_FIRST_ACTION_ALPHA if score_mode in FIRST_ACTION_MODES else None
    if variant == "c3":
        return (
            C3_FIRST_ACTION_ALPHA if score_mode == STATE_V_FIRST_Q2_SCORE_MODE else None
        )
    raise ValueError(f"Unsupported variant {variant!r}.")


def _job_output_dir(
    *, output_root: Path, variant: str, score_mode: str, alpha: float | None
) -> Path:
    output = output_root / "formal" / "o100" / "v1" / variant / score_mode
    if alpha is not None:
        output /= f"alpha_{alpha_slug(alpha)}"
    return output


def build_o100_jobs(
    *,
    repository: str | Path,
    output_root: str | Path,
    dataset: str | Path,
    c_checkpoint: str | Path,
    c3_checkpoint: str | Path,
    python: str,
) -> list[Job]:
    repository = Path(repository).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    checkpoints = {
        "c": Path(c_checkpoint).expanduser().resolve(),
        "c3": Path(c3_checkpoint).expanduser().resolve(),
    }
    score_modes = {"c": C_SCORE_MODES, "c3": C3_SCORE_MODES}
    evaluators = {
        "c": repository / "scripts" / "evaluate_actor_free_td_lewm_v1_c.py",
        "c3": repository / "scripts" / "evaluate_actor_free_td_lewm_v1_c3.py",
    }
    configs = {
        "c": repository
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c_cube_checkpoint_o100.yaml",
        "c3": repository
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c3_cube_checkpoint_o100.yaml",
    }
    jobs: list[Job] = []
    for variant in ("c", "c3"):
        for score_mode in score_modes[variant]:
            alpha = _alpha_for(variant, score_mode)
            output = _job_output_dir(
                output_root=output_root,
                variant=variant,
                score_mode=score_mode,
                alpha=alpha,
            )
            suffix = f"__alpha_{alpha_slug(alpha)}" if alpha is not None else ""
            job_id = f"v1__{variant}__o100__{score_mode}{suffix}"
            argv = [
                python,
                str(evaluators[variant]),
                "--config",
                str(configs[variant]),
                "--dataset",
                str(dataset),
                "--checkpoint-path",
                str(checkpoints[variant]),
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
                    variant=variant,
                    score_mode=score_mode,
                    alpha=alpha,
                    checkpoint=str(checkpoints[variant]),
                    config_path=str(configs[variant]),
                    output_dir=str(output),
                    log_path=str(
                        output_root / "formal" / "_launcher" / "jobs" / f"{job_id}.log"
                    ),
                    argv=tuple(argv),
                )
            )
    if len(jobs) != 7 or len({job.output_dir for job in jobs}) != 7:
        raise AssertionError("The V1-C/C3 O100 launcher must produce seven jobs.")
    return jobs


def validate_o100_job_set(jobs: Sequence[Job]) -> None:
    expected_cells = {
        *(("c", score_mode) for score_mode in C_SCORE_MODES),
        *(("c3", score_mode) for score_mode in C3_SCORE_MODES),
    }
    if (
        len(jobs) != 7
        or {(job.variant, job.score_mode) for job in jobs} != expected_cells
    ):
        raise ValueError(
            "O100 must contain each of the seven locked cells exactly once."
        )
    if (
        len({job.job_id for job in jobs}) != 7
        or len({job.output_dir for job in jobs}) != 7
    ):
        raise ValueError("O100 job identifiers and output directories must be unique.")
    shared_python: set[str] = set()
    shared_dataset: set[str] = set()
    checkpoints_by_variant: dict[str, set[str]] = {"c": set(), "c3": set()}
    for job in jobs:
        expected_alpha = _alpha_for(job.variant, job.score_mode)
        expected_config_suffix = (
            f"actor_free_td_lewm_v1_{job.variant}_cube_checkpoint_o100.yaml"
        )
        expected_evaluator = (
            Path(job.config_path).resolve().parents[2]
            / "scripts"
            / f"evaluate_actor_free_td_lewm_v1_{job.variant}.py"
        )
        if (
            job.stage != "formal"
            or job.version != "v1"
            or job.alpha != expected_alpha
            or not job.config_path.endswith(expected_config_suffix)
        ):
            raise ValueError(f"{job.job_id} violates the locked O100 matrix.")
        flags = (
            "--config",
            "--dataset",
            "--checkpoint-path",
            "--score-mode",
            "--output-dir",
        )
        if any(job.argv.count(flag) != 1 for flag in flags):
            raise ValueError(f"{job.job_id} has ambiguous evaluator arguments.")
        if "--smoke" in job.argv or "--pilot" in job.argv:
            raise ValueError(f"{job.job_id} is not a formal-only command.")
        values = {flag: job.argv[job.argv.index(flag) + 1] for flag in flags}
        if (
            values["--config"] != job.config_path
            or values["--checkpoint-path"] != job.checkpoint
            or values["--score-mode"] != job.score_mode
            or values["--output-dir"] != job.output_dir
            or Path(job.argv[1]).resolve() != expected_evaluator
        ):
            raise ValueError(f"{job.job_id} command differs from its job record.")
        if expected_alpha is None:
            if "--g-first-weight" in job.argv:
                raise ValueError(f"{job.job_id} unexpectedly sets alpha.")
        elif job.argv.count("--g-first-weight") != 1 or job.argv[
            job.argv.index("--g-first-weight") + 1
        ] != format(expected_alpha, ".17g"):
            raise ValueError(f"{job.job_id} changed its locked alpha.")
        shared_python.add(job.argv[0])
        shared_dataset.add(values["--dataset"])
        checkpoints_by_variant[job.variant].add(job.checkpoint)
    if len(shared_python) != 1 or len(shared_dataset) != 1:
        raise ValueError("All seven jobs must share one Python runtime and dataset.")
    if any(len(paths) != 1 for paths in checkpoints_by_variant.values()):
        raise ValueError("Each method must use one exact checkpoint across all scores.")


def _require_locked_checkpoint(path: Path, expected_sha256: str, *, label: str) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} checkpoint SHA-256 {actual} is not the locked artifact."
        )
    return actual


def write_checkpoint_manifest(
    output_root: Path, *, c_checkpoint: Path, c3_checkpoint: Path
) -> Path:
    payload = {
        "schema_version": 1,
        "purpose": "v1_c_e10_six_scores_and_v1_c3_e12_first_q2_formal_o100",
        "protocol_label": "o100",
        "checkpoints": {
            "v1_c_e10": {
                "path": str(c_checkpoint),
                "sha256": _require_locked_checkpoint(
                    c_checkpoint,
                    EXPECTED_C_CHECKPOINT_SHA256,
                    label="V1-C E10",
                ),
            },
            "v1_c3_e12": {
                "path": str(c3_checkpoint),
                "sha256": _require_locked_checkpoint(
                    c3_checkpoint,
                    EXPECTED_C3_CHECKPOINT_SHA256,
                    label="V1-C3 E12",
                ),
            },
        },
        "expected_selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "expected_selection_ranks_sha256": EXPECTED_SELECTION_RANKS_SHA256,
        "expected_action_normalization_sha256": EXPECTED_ACTION_NORMALIZATION_SHA256,
    }
    path = output_root / "checkpoint_manifest.json"
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError("Existing O100 checkpoint manifest binds other inputs.")
    else:
        atomic_write_json(path, payload)
    return path


def _validate_common_output(
    job: Job, *, expected_checkpoint_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
            "evaluation_protocol": "O100",
            "protocol_label": "o100",
            "goal_offset": 100,
            "episode_budget": EXPECTED_EPISODE_BUDGET,
            "score_mode": job.score_mode,
        }.items():
            if values.get(key) != expected:
                raise ValueError(f"{job.job_id} {label}.{key} must be {expected!r}.")
    metrics = results.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{job.job_id} results.metrics must be a mapping.")
    outcomes = metrics.get("episode_successes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EXPECTED_EPISODES
        or any(type(outcome) is not bool for outcome in outcomes)
    ):
        raise ValueError(f"{job.job_id} must contain exactly 50 Boolean outcomes.")
    success_count = sum(outcomes)
    success_rate = float(metrics.get("success_rate"))
    if not math.isclose(success_rate, success_count * 2.0):
        raise ValueError(f"{job.job_id} success rate disagrees with its outcomes.")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or (
        Path(str(checkpoint.get("path"))).resolve() != Path(job.checkpoint).resolve()
        or checkpoint.get("sha256") != expected_checkpoint_sha256
    ):
        raise ValueError(f"{job.job_id} used a different checkpoint.")
    if file_sha256(job.checkpoint) != expected_checkpoint_sha256:
        raise ValueError(f"{job.job_id} checkpoint changed during evaluation.")
    selection_file_sha256 = file_sha256(required["episode_selection.json"])
    if selection_file_sha256 != EXPECTED_SELECTION_FILE_SHA256:
        raise ValueError(f"{job.job_id} episode selection changed.")
    selection = read_json(required["episode_selection.json"])
    ranks = selection.get("valid_row_ranks")
    if (
        not isinstance(ranks, list)
        or len(ranks) != EXPECTED_EPISODES
        or any(type(rank) is not int or rank < 0 for rank in ranks)
        or len(set(ranks)) != EXPECTED_EPISODES
    ):
        raise ValueError(f"{job.job_id} selection ranks are invalid.")
    ranks_sha256 = canonical_json_sha256(ranks)
    if ranks_sha256 != EXPECTED_SELECTION_RANKS_SHA256:
        raise ValueError(f"{job.job_id} selection rank digest changed.")
    if manifest.get("selection") != selection:
        raise ValueError(f"{job.job_id} manifest selection differs from its file.")
    if file_sha256(required["action_normalization.json"]) != (
        EXPECTED_ACTION_NORMALIZATION_SHA256
    ):
        raise ValueError(f"{job.job_id} action normalization changed.")
    evidence = {
        "success_count": success_count,
        "success_rate": success_rate,
        "episode_successes": outcomes,
        "elapsed_seconds": results.get("elapsed_seconds"),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "selection_path": str(required["episode_selection.json"]),
        "selection_file_sha256": selection_file_sha256,
        "valid_row_ranks": ranks,
        "valid_row_ranks_sha256": ranks_sha256,
        "selection_sha256": ranks_sha256,
        "action_normalization_sha256": EXPECTED_ACTION_NORMALIZATION_SHA256,
        "results_path": str(required["results.json"]),
        "manifest_path": str(required["protocol_manifest.json"]),
    }
    return evidence, results, manifest


def validate_o100_job_output(
    job: Job, *, expected_selection_file_sha256: str | None = None
) -> dict[str, Any]:
    if expected_selection_file_sha256 not in (None, EXPECTED_SELECTION_FILE_SHA256):
        raise ValueError("The V1-C/C3 O100 selection lock cannot be overridden.")
    expected_checkpoint_sha256 = (
        EXPECTED_C_CHECKPOINT_SHA256
        if job.variant == "c"
        else EXPECTED_C3_CHECKPOINT_SHA256
    )
    if job.variant == "c":
        evidence = validate_job_output(
            job,
            expected_selection_file_sha256=EXPECTED_SELECTION_FILE_SHA256,
        )
        _, results, manifest = _validate_common_output(
            job, expected_checkpoint_sha256=expected_checkpoint_sha256
        )
        protocol = manifest.get("protocol")
        formal_protocol = manifest.get("formal_protocol")
        if not isinstance(protocol, Mapping) or not isinstance(
            formal_protocol, Mapping
        ):
            raise ValueError(f"{job.job_id} is missing protocol audit mappings.")
        validate_actor_free_td_lewm_v1_c_evaluation_protocol(protocol)
        validate_actor_free_td_lewm_v1_c_evaluation_protocol(formal_protocol)
        if (
            v1_evaluation_protocol_label(protocol) != "o100"
            or v1_evaluation_protocol_label(formal_protocol) != "o100"
        ):
            raise ValueError(f"{job.job_id} did not use formal O100.")
        evidence.update(
            {
                "success_count": sum(results["metrics"]["episode_successes"]),
                "success_rate": float(results["metrics"]["success_rate"]),
                "checkpoint_sha256": expected_checkpoint_sha256,
                "action_normalization_sha256": EXPECTED_ACTION_NORMALIZATION_SHA256,
                "protocol_label": "o100",
            }
        )
        return evidence

    if job.variant != "c3":
        raise ValueError(f"Unsupported O100 job variant {job.variant!r}.")
    evidence, results, manifest = _validate_common_output(
        job, expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    expected_alpha = _alpha_for("c3", job.score_mode)
    for values, label in ((results, "results"), (manifest, "manifest")):
        if expected_alpha is None:
            if "g_first_weight" in values:
                raise ValueError(f"{job.job_id} {label} unexpectedly records alpha.")
        elif float(values.get("g_first_weight")) != expected_alpha:
            raise ValueError(f"{job.job_id} {label}.g_first_weight is incorrect.")
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
            raise ValueError(f"{job.job_id} results.{key} must be {expected!r}.")
    protocol = manifest.get("protocol")
    formal_protocol = manifest.get("formal_protocol")
    if not isinstance(protocol, Mapping) or not isinstance(formal_protocol, Mapping):
        raise ValueError(f"{job.job_id} is missing protocol audit mappings.")
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol(protocol)
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol(formal_protocol)
    if (
        v1_evaluation_protocol_label(protocol) != "o100"
        or v1_evaluation_protocol_label(formal_protocol) != "o100"
    ):
        raise ValueError(f"{job.job_id} did not use formal O100.")
    expected_protocol = configure_actor_free_td_lewm_v1_c3_evaluation_mode(
        formal_protocol,
        smoke=False,
        pilot=False,
        score_mode=job.score_mode,
        g_first_weight=expected_alpha,
    )
    if protocol != expected_protocol:
        raise ValueError(f"{job.job_id} configured protocol differs from the lock.")
    expected_execution = _execution_metadata(protocol["planning"])
    for values, label in ((results, "results"), (manifest, "manifest")):
        for key, expected in expected_execution.items():
            if values.get(key) != expected:
                raise ValueError(f"{job.job_id} {label}.{key} must be {expected!r}.")
        if values.get("score_definition") != protocol["inference_objective"].get(
            "score_definition"
        ):
            raise ValueError(f"{job.job_id} {label}.score_definition changed.")
    checkpoint = manifest["checkpoint"]
    if (
        checkpoint.get("epoch") != 12
        or checkpoint.get("logical_epoch") != 12
        or checkpoint.get("global_step") != 12_000
        or checkpoint.get("formal_completion_required") is not True
    ):
        raise ValueError(f"{job.job_id} did not use the final C3 E12 checkpoint.")
    evidence["protocol_label"] = "o100"
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked V1-C E10 six-score and V1-C3 E12 First-Q2 "
            "formal Cube O100 matrix."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--c-checkpoint", required=True)
    parser.add_argument("--c3-checkpoint", required=True)
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
    c_checkpoint = Path(args.c_checkpoint).expanduser().resolve()
    c3_checkpoint = Path(args.c3_checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not repository.is_dir() or not dataset.exists():
        raise FileNotFoundError(repository if not repository.is_dir() else dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = write_checkpoint_manifest(
        output_root,
        c_checkpoint=c_checkpoint,
        c3_checkpoint=c3_checkpoint,
    )
    jobs = build_o100_jobs(
        repository=repository,
        output_root=output_root,
        dataset=dataset,
        c_checkpoint=c_checkpoint,
        c3_checkpoint=c3_checkpoint,
        python=args.python,
    )
    validate_o100_job_set(jobs)
    plan = StagePlan(
        stage="formal",
        versions=("v1",),
        variants=("c", "c3"),
        score_modes=(*C_SCORE_MODES, *C3_SCORE_MODES),
        v2_only_score_modes=(),
        alphas=(C_FIRST_ACTION_ALPHA, C3_FIRST_ACTION_ALPHA),
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
        job_output_validator=validate_o100_job_output,
        launcher_metadata={
            "launcher": "actor_free_td_lewm_v1_c_c3_o100_evaluations",
            "protocol_label": "o100",
            "evaluation_protocol": "O100",
            "expected_c_checkpoint_sha256": EXPECTED_C_CHECKPOINT_SHA256,
            "expected_c3_checkpoint_sha256": EXPECTED_C3_CHECKPOINT_SHA256,
            "expected_selection_ranks_sha256": EXPECTED_SELECTION_RANKS_SHA256,
            "expected_action_normalization_sha256": (
                EXPECTED_ACTION_NORMALIZATION_SHA256
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
