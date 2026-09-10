#!/usr/bin/env python3
"""Complete the 30 missing V1 D/F/G1/G2/G3 G-weighted formal evaluations.

Reuse the six completed C cells. Never train or schedule an F-only repeat.
One shared GPU pool handles all goal offsets, longest-budget jobs first.
The launcher only schedules evaluations; it does not power off any server.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_actor_free_td_lewm_first_action_comparison import (  # noqa: E402
    Job,
    StagePlan,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
    run_jobs,
)
from run_actor_free_td_lewm_v2_parallel import V1_SHA256  # noqa: E402

MISSING_VARIANTS = ("d", "f", "g1", "g2", "g3")
PROTOCOLS = ("o100", "o50", "o25")
WEIGHT_MODES = ("path", "action")
SELECTION_SHA256 = {
    "o25": "56546fe8725ce0e4670f308c5b325bd64ff2a792373add8c20ddbcab02da6b37",
    "o50": "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7",
    "o100": "8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c",
}
ACTION_SHA256 = "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
DATASET_MANIFEST_SHA256 = (
    "9de531030c6bca21a7b3215d7abea3aaf277e68a1e4cec03c8c6e22ad0d20dcd"
)
RENDER_ENVIRONMENT = {
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "WANDB_MODE": "disabled",
}


def checkpoint_path(root: Path, variant: str) -> Path:
    return (
        root / variant / "seed_3072" / "checkpoints"
        / f"actor_free_td_lewm_v1_{variant}" / variant / "epoch_10.pt"
    )


def protocol_group(job: Job) -> str:
    matches = [label for label in PROTOCOLS if f"__{label}__" in job.job_id]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous goal-offset group: {job.job_id}")
    return matches[0]


def build_completion_jobs(*, repository, output_root, checkpoint_root, dataset, python):
    repository, output_root, checkpoint_root, dataset = map(
        Path, (repository, output_root, checkpoint_root, dataset)
    )
    jobs = []
    for label in PROTOCOLS:
        for variant in MISSING_VARIANTS:
            for mode in WEIGHT_MODES:
                score = f"g_{mode}_weighted_cem"
                job_id = f"v1__{variant}__{label}__{score}"
                config = repository / "configs/experiment" / (
                    f"actor_free_td_lewm_v1_{variant}_cube_checkpoint_{label}.yaml"
                )
                checkpoint = checkpoint_path(checkpoint_root, variant)
                output = output_root / "formal" / label / "v1" / variant / score
                argv = (
                    str(python),
                    str(repository / "scripts/evaluate_actor_free_td_lewm_g_weighted_cem.py"),
                    "--version", "v1", "--variant", variant,
                    "--config", str(config), "--weight-mode", mode,
                    "--temperature", "1.0", "--dataset", str(dataset),
                    "--checkpoint-path", str(checkpoint),
                    "--checkpoint-sha256", V1_SHA256[variant],
                    "--output-dir", str(output),
                )
                jobs.append(Job(
                    job_id=job_id, stage="formal", version="v1", variant=variant,
                    score_mode=score, alpha=None, checkpoint=str(checkpoint),
                    config_path=str(config), output_dir=str(output),
                    log_path=str(output_root / "formal/_launcher/jobs" / f"{job_id}.log"),
                    argv=argv,
                ))
    if len(jobs) != 30 or len({j.output_dir for j in jobs}) != 30:
        raise AssertionError("Completion must schedule exactly 30 distinct new cells.")
    return jobs


def validate_weighted_output(job, *, expected_selection_file_sha256=None):
    label = protocol_group(job)
    expected_selection = SELECTION_SHA256[label]
    if expected_selection_file_sha256 not in (None, expected_selection):
        raise ValueError("The expected selection belongs to a different goal offset.")
    output = Path(job.output_dir)
    result = read_json(output / "results.json")
    manifest = read_json(output / "protocol_manifest.json")
    selection = read_json(output / "episode_selection.json")
    identity = {
        "method": f"actor_free_td_lewm_v1_{job.variant}",
        "variant": job.variant, "implementation_version": "v1",
        "score_mode": job.score_mode, "protocol_label": label,
        "planning_horizon": 5, "smoke": False, "pilot": False,
    }
    for key, expected in identity.items():
        if result.get(key) != expected:
            raise ValueError(f"{job.job_id}: incorrect results.{key}")
    outcomes = result.get("metrics", {}).get("episode_successes")
    if not isinstance(outcomes, list) or len(outcomes) != 50:
        raise ValueError("Formal output must contain 50 episode outcomes.")
    if any(type(value) is not bool for value in outcomes):
        raise ValueError("Episode outcomes must be booleans.")
    rate = result["metrics"].get("success_rate")
    if not isinstance(rate, (int, float)) or not math.isclose(
        rate, 2 * sum(outcomes), abs_tol=1e-8
    ):
        raise ValueError("Success rate disagrees with the 50 episode outcomes.")
    mode = job.score_mode.removeprefix("g_").removesuffix("_weighted_cem")
    for value in (result, manifest):
        if value.get("g_weighted_cem") != {"mode": mode, "temperature": 1.0}:
            raise ValueError("The fixed G weighting/temperature changed.")
        definition = value.get("score_definition", {})
        for key, expected in {
            "g_role": "elite_distribution_update_only",
            "elite_selection": "lowest_full_F_terminal_cost",
            "score_normalization": "none",
            "g_population": "selected_elites_only",
        }.items():
            if definition.get(key) != expected:
                raise ValueError(f"G-weighting semantics changed: {key}")
    checkpoint = manifest["checkpoint"]
    if checkpoint.get("sha256") != V1_SHA256[job.variant]:
        raise ValueError("Wrong historical checkpoint digest.")
    if checkpoint.get("epoch") != 10 or checkpoint.get("global_step") != 127960:
        raise ValueError("The endpoint must remain epoch 10 / update 127960.")
    planning = manifest["protocol"]["planning"]
    for key, expected in {
        "horizon": 5, "candidates": 300, "iterations": 30, "elites": 30,
        "action_block": 5, "planning_seed": 42,
        "receding_horizon": 5 if label == "o25" else 1,
        "episode_budget": 2 * int(label[1:]),
    }.items():
        if planning.get(key) != expected:
            raise ValueError(f"Planning protocol changed: {key}")
    if file_sha256(output / "episode_selection.json") != expected_selection:
        raise ValueError("Start-goal pair selection changed.")
    if file_sha256(output / "action_normalization.json") != ACTION_SHA256:
        raise ValueError("Action normalization changed.")
    ranks = selection["valid_row_ranks"]
    return {
        "results_path": str(output / "results.json"),
        "manifest_path": str(output / "protocol_manifest.json"),
        "valid_row_ranks": ranks,
        "valid_row_ranks_sha256": canonical_json_sha256(ranks),
        "selection_file_sha256": expected_selection,
        "checkpoint_sha256": checkpoint["sha256"],
        "success_count": sum(outcomes), "episode_count": 50,
        "success_rate_percent": rate, "episode_successes": outcomes,
        "results_sha256": file_sha256(output / "results.json"),
    }


def validate_reused_c(root):
    root = Path(root)
    launch = read_json(root / "launch_manifest.json")
    if launch.get("environment", {}).get("MUJOCO_GL") != "osmesa":
        raise ValueError("Completed C cells must use the same OSMesa renderer.")
    evidence = {}
    for label in PROTOCOLS:
        for mode in WEIGHT_MODES:
            score = f"g_{mode}_weighted_cem"
            output = root / label / "v1_c" / score
            job = Job(
                job_id=f"v1__c__{label}__{score}", stage="formal", version="v1",
                variant="c", score_mode=score, alpha=None,
                checkpoint=launch["checkpoint_path"], config_path="",
                output_dir=str(output), log_path="", argv=(),
            )
            evidence[job.job_id] = validate_weighted_output(job)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reuse-c-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--max-jobs-per-gpu", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=12)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    repository = SCRIPT_DIRECTORY.parent
    if args.max_concurrency < 1 or args.max_jobs_per_gpu < 1:
        parser.error("Concurrency limits must be positive.")
    if args.max_concurrency > len(args.gpus) * args.max_jobs_per_gpu:
        parser.error("Concurrency exceeds the configured GPU slot capacity.")
    checkpoints = {}
    for variant in MISSING_VARIANTS:
        path = checkpoint_path(args.checkpoint_root, variant)
        if file_sha256(path) != V1_SHA256[variant]:
            raise ValueError(f"{variant}: historical checkpoint mismatch.")
        checkpoints[variant] = {"path": str(path), "sha256": V1_SHA256[variant]}
    dataset_manifest = Path(str(args.dataset) + ".manifest.json")
    if not args.dataset.is_dir() or file_sha256(dataset_manifest) != DATASET_MANIFEST_SHA256:
        raise ValueError("The historical Cube dataset is missing or changed.")
    reused = validate_reused_c(args.reuse_c_root)
    jobs = build_completion_jobs(
        repository=repository, output_root=args.output_root,
        checkpoint_root=args.checkpoint_root, dataset=args.dataset, python=sys.executable,
    )
    for job in jobs:
        if not Path(job.config_path).is_file():
            raise FileNotFoundError(job.config_path)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    plan = {
        "code_revision": revision, "total_expected_cells": 36,
        "reused_c_cells": reused, "new_cell_count": 30,
        "training_performed": False, "f_only_repeats_scheduled": False,
        "environment": {**RENDER_ENVIRONMENT, **{
            key: os.environ.get(key) for key in
            ("LD_PRELOAD", "LD_LIBRARY_PATH", "STABLEWM_HOME")
        }},
        "checkpoints": checkpoints, "jobs": [asdict(job) for job in jobs],
        "max_concurrency": args.max_concurrency,
        "max_jobs_per_gpu": args.max_jobs_per_gpu,
    }
    if args.preflight_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    # Do not allocate GPUs during CPU-mode preflight. Query only before execution.
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    ).split()
    if len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= set(devices):
        raise ValueError("Requested CUDA devices are not all available.")
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "completion_plan.json", plan)
    checkpoint_manifest = args.output_root / "checkpoint_manifest.json"
    atomic_write_json(checkpoint_manifest, checkpoints)
    os.environ.update(RENDER_ENVIRONMENT)
    os.environ["PYTHONPATH"] = str(repository / "src")
    return run_jobs(
        jobs=jobs,
        plan=StagePlan("formal", ("v1",), MISSING_VARIANTS,
                       tuple(f"g_{mode}_weighted_cem" for mode in WEIGHT_MODES), (), ()),
        repository=repository, dataset=args.dataset,
        checkpoint_manifest=checkpoint_manifest, output_root=args.output_root,
        gpus=args.gpus, max_concurrency=args.max_concurrency,
        max_jobs_per_gpu=args.max_jobs_per_gpu, formal_selection=None,
        poll_seconds=2.0, job_output_validator=validate_weighted_output,
        selection_group_key=protocol_group,
        launcher_metadata={
            "launcher": "v1_g_weighted_completion", "reused_c_cells": reused,
            "total_expected_cells": 36, "new_cell_count": 30,
            "f_only_repeats_scheduled": False, "runtime_plan": plan,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
