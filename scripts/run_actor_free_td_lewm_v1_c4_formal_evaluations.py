#!/usr/bin/env python3
"""Run the locked 18-cell V1-C4 Cube O25/O50/O100 evaluation matrix.

The launcher is inference-only.  It fixes the six score modes, both First-Q
weights, and the three historical start-goal selections before any result is
observed.  Each protocol is delegated to the existing parallel job runner so
selection equality is checked within (and never incorrectly across) O25,
O50, and O100.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    C4_ACTION_EFFECT,
    C4_JOINT_OBJECTIVE,
    FIRST_ACTION_SCORE_MODES,
    OBJECTIVE_VERSION,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c4 import (
    configure_actor_free_td_lewm_v1_c4_evaluation_mode,
    validate_actor_free_td_lewm_v1_c4_evaluation_protocol,
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

PROTOCOL_LABELS = ("o25", "o50", "o100")
SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "g_only_f_rollout_mean",
    "f_plus_g_first_q2",
)
PRESPECIFIED_ALPHA = 0.25
EXPECTED_EPISODES = 50
EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL = {
    "o25": "56546fe8725ce0e4670f308c5b325bd64ff2a792373add8c20ddbcab02da6b37",
    "o50": "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7",
    "o100": "8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c",
}
EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL = {
    "o25": "72af45d4bad65a25288c5d405072d18ab5c0b4f0b67ddc970ac3f344b3c22fd9",
    "o50": "88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9",
    "o100": "36994b1ab36656666ff91b379a59829c4b2af150b1f4ed23d409deb5cca9654e",
}
EXPECTED_ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)
EXPECTED_GOAL_OFFSET_BY_PROTOCOL = {"o25": 25, "o50": 50, "o100": 100}
EXPECTED_EPISODE_BUDGET_BY_PROTOCOL = {"o25": 50, "o50": 100, "o100": 200}
PROTOCOL_COST_WEIGHTS = {"o25": 2, "o50": 3, "o100": 5}
MAX_GLOBAL_JOBS_PER_GPU = 2
MAX_PROTOCOL_JOBS_PER_GPU = 1


def _alpha_for(score_mode: str) -> float | None:
    return PRESPECIFIED_ALPHA if score_mode in FIRST_ACTION_SCORE_MODES else None


def _protocol_from_job(job: Job) -> str:
    matches = [
        protocol
        for protocol in PROTOCOL_LABELS
        if f"__{protocol}__" in job.job_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{job.job_id} does not bind one C4 protocol label.")
    return matches[0]


def _output_directory(
    *,
    output_root: Path,
    protocol_label: str,
    score_mode: str,
    alpha: float | None,
) -> Path:
    output = output_root / "formal" / protocol_label / "v1" / "c4" / score_mode
    if alpha is not None:
        output /= f"alpha_{alpha_slug(alpha)}"
    return output


def build_c4_formal_evaluation_jobs(
    *,
    repository: str | Path,
    output_root: str | Path,
    dataset: str | Path,
    checkpoint: str | Path,
    python: str,
) -> list[Job]:
    """Build exactly three protocols by six predeclared C4 scores."""

    repository = Path(repository).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    evaluator = repository / "scripts" / "evaluate_actor_free_td_lewm_v1_c4.py"
    jobs: list[Job] = []
    for protocol_label in PROTOCOL_LABELS:
        config = (
            repository
            / "configs"
            / "experiment"
            / f"actor_free_td_lewm_v1_c4_cube_checkpoint_{protocol_label}.yaml"
        )
        for score_mode in SCORE_MODES:
            alpha = _alpha_for(score_mode)
            output = _output_directory(
                output_root=output_root,
                protocol_label=protocol_label,
                score_mode=score_mode,
                alpha=alpha,
            )
            alpha_suffix = (
                f"__alpha_{alpha_slug(alpha)}" if alpha is not None else ""
            )
            job_id = f"v1__c4__{protocol_label}__{score_mode}{alpha_suffix}"
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
                    variant="c4",
                    score_mode=score_mode,
                    alpha=alpha,
                    checkpoint=str(checkpoint),
                    config_path=str(config),
                    output_dir=str(output),
                    log_path=str(
                        output_root
                        / "formal"
                        / "_launcher"
                        / "jobs"
                        / f"{job_id}.log"
                    ),
                    argv=tuple(argv),
                )
            )
    if len(jobs) != 18:
        raise AssertionError("The C4 formal matrix must contain exactly 18 jobs.")
    return jobs


def _argument_values(job: Job) -> dict[str, str]:
    flags = (
        "--config",
        "--dataset",
        "--checkpoint-path",
        "--score-mode",
        "--output-dir",
    )
    if any(job.argv.count(flag) != 1 for flag in flags):
        raise ValueError(f"{job.job_id} has ambiguous evaluator arguments.")
    try:
        return {flag: job.argv[job.argv.index(flag) + 1] for flag in flags}
    except (IndexError, ValueError) as error:
        raise ValueError(f"{job.job_id} has incomplete evaluator arguments.") from error


def validate_c4_formal_job_set(jobs: Sequence[Job]) -> None:
    expected_cells = {
        (protocol_label, score_mode)
        for protocol_label in PROTOCOL_LABELS
        for score_mode in SCORE_MODES
    }
    actual_cells = {(_protocol_from_job(job), job.score_mode) for job in jobs}
    if len(jobs) != 18 or actual_cells != expected_cells:
        raise ValueError("C4 formal launch must contain each of the 18 cells once.")
    if len({job.job_id for job in jobs}) != 18 or len(
        {job.output_dir for job in jobs}
    ) != 18:
        raise ValueError("C4 job identifiers and output directories must be unique.")

    shared_python: set[str] = set()
    shared_dataset: set[str] = set()
    shared_checkpoint: set[str] = set()
    shared_evaluator: set[str] = set()
    for job in jobs:
        protocol_label = _protocol_from_job(job)
        expected_alpha = _alpha_for(job.score_mode)
        expected_config_suffix = (
            f"actor_free_td_lewm_v1_c4_cube_checkpoint_{protocol_label}.yaml"
        )
        if (
            job.stage != "formal"
            or job.version != "v1"
            or job.variant != "c4"
            or job.alpha != expected_alpha
            or not job.config_path.endswith(expected_config_suffix)
        ):
            raise ValueError(f"{job.job_id} violates the locked C4 matrix.")
        if "--smoke" in job.argv or "--pilot" in job.argv:
            raise ValueError(f"{job.job_id} is not a formal-only command.")
        values = _argument_values(job)
        expected_evaluator = (
            Path(job.config_path).resolve().parents[2]
            / "scripts"
            / "evaluate_actor_free_td_lewm_v1_c4.py"
        )
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
                raise ValueError(f"{job.job_id} unexpectedly sets First-Q alpha.")
        elif (
            job.argv.count("--g-first-weight") != 1
            or job.argv[job.argv.index("--g-first-weight") + 1]
            != format(PRESPECIFIED_ALPHA, ".17g")
        ):
            raise ValueError(f"{job.job_id} changed the locked First-Q alpha.")
        shared_python.add(job.argv[0])
        shared_dataset.add(values["--dataset"])
        shared_checkpoint.add(values["--checkpoint-path"])
        shared_evaluator.add(job.argv[1])
    if any(
        len(values) != 1
        for values in (
            shared_python,
            shared_dataset,
            shared_checkpoint,
            shared_evaluator,
        )
    ):
        raise ValueError(
            "All 18 C4 jobs must share one Python runtime, dataset, checkpoint, "
            "and evaluator."
        )


def write_checkpoint_manifest(output_root: Path, *, checkpoint: Path) -> Path:
    """Dynamically bind the newly trained C4 checkpoint before evaluation."""

    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    payload = {
        "schema_version": 1,
        "purpose": "v1_c4_objective1_six_scores_formal_o25_o50_o100",
        "method": "actor_free_td_lewm_v1_c4",
        "variant": "c4",
        "objective_version": OBJECTIVE_VERSION,
        "training_objective": C4_JOINT_OBJECTIVE["objective"],
        "action_effect": C4_ACTION_EFFECT,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "required_objective_version": OBJECTIVE_VERSION,
        },
        "protocols": {
            protocol_label: {
                "evaluation_protocol": protocol_label.upper(),
                "episodes": EXPECTED_EPISODES,
                "goal_offset": EXPECTED_GOAL_OFFSET_BY_PROTOCOL[protocol_label],
                "episode_budget": EXPECTED_EPISODE_BUDGET_BY_PROTOCOL[
                    protocol_label
                ],
                "selection_file_sha256": (
                    EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL[protocol_label]
                ),
                "selection_ranks_sha256": (
                    EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL[protocol_label]
                ),
                "action_normalization_sha256": (
                    EXPECTED_ACTION_NORMALIZATION_SHA256
                ),
            }
            for protocol_label in PROTOCOL_LABELS
        },
        "score_modes": list(SCORE_MODES),
        "first_q_alpha": PRESPECIFIED_ALPHA,
        "training_performed": False,
        "alpha_selection_performed": False,
    }
    path = output_root / "checkpoint_manifest.json"
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError("Existing C4 manifest binds other formal inputs.")
    else:
        atomic_write_json(path, payload)
    return path


def _find_checkpoint_manifest(job: Job) -> Mapping[str, Any]:
    output = Path(job.output_dir).resolve()
    candidates = [
        parent / "checkpoint_manifest.json" for parent in (output, *output.parents)
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{job.job_id} must resolve exactly one checkpoint_manifest.json."
        )
    return read_json(matches[0])


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def validate_c4_job_output(
    job: Job,
    *,
    expected_selection_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit one formal C4 result against its pre-run checkpoint/protocol lock."""

    protocol_label = _protocol_from_job(job)
    expected_selection_sha = EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL[
        protocol_label
    ]
    if expected_selection_file_sha256 not in (None, expected_selection_sha):
        raise ValueError(f"The C4 {protocol_label.upper()} selection cannot change.")
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

    lock = _find_checkpoint_manifest(job)
    lock_checkpoint = _required_mapping(
        lock.get("checkpoint"), label="checkpoint_manifest.checkpoint"
    )
    locked_checkpoint_path = Path(str(lock_checkpoint.get("path"))).resolve()
    locked_checkpoint_sha = lock_checkpoint.get("sha256")
    if (
        lock.get("objective_version") != OBJECTIVE_VERSION
        or lock.get("training_objective") != C4_JOINT_OBJECTIVE["objective"]
        or lock.get("action_effect") != C4_ACTION_EFFECT
        or lock_checkpoint.get("required_objective_version") != OBJECTIVE_VERSION
    ):
        raise ValueError(
            f"{job.job_id} checkpoint lock does not describe final C4 objective v1."
        )
    if (
        locked_checkpoint_path != Path(job.checkpoint).resolve()
        or not isinstance(locked_checkpoint_sha, str)
        or file_sha256(job.checkpoint) != locked_checkpoint_sha
    ):
        raise ValueError(f"{job.job_id} checkpoint differs from the pre-run lock.")

    results = read_json(required["results.json"])
    manifest = read_json(required["protocol_manifest.json"])
    expected_common = {
        "method": "actor_free_td_lewm_v1_c4",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c4",
        "implementation_version": "v1",
        "objective_version": OBJECTIVE_VERSION,
        "evaluation_protocol": protocol_label.upper(),
        "protocol_label": protocol_label,
        "goal_offset": EXPECTED_GOAL_OFFSET_BY_PROTOCOL[protocol_label],
        "episode_budget": EXPECTED_EPISODE_BUDGET_BY_PROTOCOL[protocol_label],
        "score_mode": job.score_mode,
        "planning_horizon": 1 if job.score_mode == "g_only" else 5,
        "smoke": False,
        "pilot": False,
        "state_only_g": True,
        "action_enters_g": False,
        "action_effect": C4_ACTION_EFFECT,
        "g_state_source": "stopped_f_post_action_ghost_state",
    }
    for key, expected in expected_common.items():
        if results.get(key) != expected:
            raise ValueError(
                f"{job.job_id} results.{key}={results.get(key)!r}, "
                f"expected {expected!r}."
            )
    for key in (
        "evaluation_protocol",
        "protocol_label",
        "goal_offset",
        "episode_budget",
        "score_mode",
        "state_only_g",
        "action_enters_g",
        "action_effect",
        "objective_version",
        "g_state_source",
    ):
        if manifest.get(key) != expected_common[key]:
            raise ValueError(f"{job.job_id} manifest.{key} is incorrect.")

    metrics = _required_mapping(results.get("metrics"), label="results.metrics")
    outcomes = metrics.get("episode_successes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != EXPECTED_EPISODES
        or any(type(outcome) is not bool for outcome in outcomes)
    ):
        raise ValueError(f"{job.job_id} must contain exactly 50 Boolean outcomes.")
    success_count = sum(outcomes)
    try:
        success_rate = float(metrics["success_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{job.job_id} success_rate must be numeric.") from error
    if not math.isclose(success_rate, success_count * 100.0 / EXPECTED_EPISODES):
        raise ValueError(f"{job.job_id} success_rate disagrees with its outcomes.")

    protocol = _required_mapping(manifest.get("protocol"), label="protocol")
    formal_protocol = _required_mapping(
        manifest.get("formal_protocol"), label="formal_protocol"
    )
    validate_actor_free_td_lewm_v1_c4_evaluation_protocol(protocol)
    validate_actor_free_td_lewm_v1_c4_evaluation_protocol(formal_protocol)
    if (
        v1_evaluation_protocol_label(protocol) != protocol_label
        or v1_evaluation_protocol_label(formal_protocol) != protocol_label
    ):
        raise ValueError(f"{job.job_id} protocol label differs from its job.")
    expected_protocol = configure_actor_free_td_lewm_v1_c4_evaluation_mode(
        formal_protocol,
        smoke=False,
        pilot=False,
        score_mode=job.score_mode,
        g_first_weight=_alpha_for(job.score_mode),
    )
    if protocol != expected_protocol:
        raise ValueError(f"{job.job_id} configured protocol differs from its lock.")

    expected_execution = _execution_metadata(protocol["planning"])
    expected_score_definition = protocol["inference_objective"]["score_definition"]
    for values, label in ((results, "results"), (manifest, "manifest")):
        for key, expected in expected_execution.items():
            if values.get(key) != expected:
                raise ValueError(f"{job.job_id} {label}.{key} is incorrect.")
        if values.get("score_definition") != expected_score_definition:
            raise ValueError(f"{job.job_id} {label}.score_definition changed.")
        expected_alpha = _alpha_for(job.score_mode)
        if expected_alpha is None:
            if "g_first_weight" in values:
                raise ValueError(f"{job.job_id} {label} unexpectedly records alpha.")
        elif float(values.get("g_first_weight")) != expected_alpha:
            raise ValueError(f"{job.job_id} {label}.g_first_weight changed.")

    checkpoint = _required_mapping(manifest.get("checkpoint"), label="checkpoint")
    if (
        Path(str(checkpoint.get("path"))).resolve() != locked_checkpoint_path
        or checkpoint.get("sha256") != locked_checkpoint_sha
        or checkpoint.get("epoch") != 10
        or checkpoint.get("global_step") != 127_960
        or checkpoint.get("objective_version") != OBJECTIVE_VERSION
        or checkpoint.get("formal_completion_required") is not True
        or "g_config" not in checkpoint
        or "predictor_config" in checkpoint
    ):
        raise ValueError(f"{job.job_id} did not use the locked final C4 checkpoint.")

    selection_file_sha = file_sha256(required["episode_selection.json"])
    if selection_file_sha != expected_selection_sha:
        raise ValueError(f"{job.job_id} episode selection file changed.")
    selection = read_json(required["episode_selection.json"])
    ranks = selection.get("valid_row_ranks")
    if (
        not isinstance(ranks, list)
        or len(ranks) != EXPECTED_EPISODES
        or any(type(rank) is not int or rank < 0 for rank in ranks)
        or len(set(ranks)) != EXPECTED_EPISODES
    ):
        raise ValueError(f"{job.job_id} selection ranks are invalid.")
    ranks_sha = canonical_json_sha256(ranks)
    if ranks_sha != EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL[protocol_label]:
        raise ValueError(f"{job.job_id} selection ranks changed.")
    if manifest.get("selection") != selection:
        raise ValueError(f"{job.job_id} manifest selection differs from its file.")
    action_sha = file_sha256(required["action_normalization.json"])
    if action_sha != EXPECTED_ACTION_NORMALIZATION_SHA256:
        raise ValueError(f"{job.job_id} action normalization changed.")

    return {
        "protocol_label": protocol_label,
        "score_mode": job.score_mode,
        "alpha": _alpha_for(job.score_mode),
        "success_count": success_count,
        "success_rate": success_rate,
        "episode_successes": outcomes,
        "checkpoint_sha256": locked_checkpoint_sha,
        "selection_path": str(required["episode_selection.json"]),
        "selection_file_sha256": selection_file_sha,
        "valid_row_ranks": ranks,
        "valid_row_ranks_sha256": ranks_sha,
        "selection_sha256": ranks_sha,
        "action_normalization_sha256": action_sha,
        "results_path": str(required["results.json"]),
        "manifest_path": str(required["protocol_manifest.json"]),
    }


def _protocol_groups(jobs: Sequence[Job]) -> dict[str, list[Job]]:
    groups = {protocol_label: [] for protocol_label in PROTOCOL_LABELS}
    for job in jobs:
        groups[_protocol_from_job(job)].append(job)
    if any(len(group) != len(SCORE_MODES) for group in groups.values()):
        raise ValueError("Every C4 protocol group must contain exactly six jobs.")
    return groups


def resolve_cost_weighted_gpu_allocation(
    *,
    gpus: Sequence[str],
    max_concurrency: int,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Allocate formal slots by measured protocol cost without oversubscription.

    O100 is the long pole, so the fixed 2:3:5 O25/O50/O100 cost weights bias
    slots toward it.  A GPU may serve at most two protocol runners globally,
    while each runner is limited to one active process per assigned GPU.
    """

    normalized_gpus = tuple(str(gpu) for gpu in gpus)
    if len(normalized_gpus) < 2:
        raise ValueError("C4 formal weighted dispatch requires at least two GPUs.")
    if len(set(normalized_gpus)) != len(normalized_gpus):
        raise ValueError("C4 formal GPU identifiers must be unique.")
    if max_concurrency < len(PROTOCOL_LABELS):
        raise ValueError(
            "max_concurrency must be at least three so every protocol runs."
        )
    safe_capacity = MAX_GLOBAL_JOBS_PER_GPU * len(normalized_gpus)
    if max_concurrency > safe_capacity:
        raise ValueError(
            "max_concurrency would exceed two jobs per GPU "
            f"({max_concurrency} > {safe_capacity})."
        )

    per_protocol_capacity = min(len(SCORE_MODES), len(normalized_gpus))
    candidates: list[tuple[tuple[int, int, int], dict[str, int]]] = []
    weight_total = sum(PROTOCOL_COST_WEIGHTS.values())
    for values in itertools.product(
        range(1, per_protocol_capacity + 1), repeat=len(PROTOCOL_LABELS)
    ):
        if sum(values) != max_concurrency:
            continue
        slots = dict(zip(PROTOCOL_LABELS, values, strict=True))
        proportional_error = sum(
            (
                slots[label] * weight_total
                - max_concurrency * PROTOCOL_COST_WEIGHTS[label]
            )
            ** 2
            for label in PROTOCOL_LABELS
        )
        # Stable ties keep capacity on the more expensive protocols.
        score = (proportional_error, -slots["o100"], -slots["o50"])
        candidates.append((score, slots))
    if not candidates:
        raise ValueError(
            "The requested concurrency cannot be assigned safely across all "
            "three protocols with one process per protocol/GPU pair."
        )
    slots = min(candidates, key=lambda item: item[0])[1]

    gpu_index = {gpu: index for index, gpu in enumerate(normalized_gpus)}
    gpu_load = {gpu: 0 for gpu in normalized_gpus}
    gpu_groups: dict[str, list[str]] = {}
    for label in sorted(
        PROTOCOL_LABELS,
        key=lambda item: (-PROTOCOL_COST_WEIGHTS[item], item),
    ):
        available = sorted(
            (gpu for gpu in normalized_gpus if gpu_load[gpu] < 2),
            key=lambda gpu: (gpu_load[gpu], gpu_index[gpu]),
        )
        assigned = available[: slots[label]]
        if len(assigned) != slots[label]:
            raise ValueError(f"No safe GPU allocation remains for {label}.")
        gpu_groups[label] = assigned
        for gpu in assigned:
            gpu_load[gpu] += 1

    if (
        sum(slots.values()) != max_concurrency
        or any(load > MAX_GLOBAL_JOBS_PER_GPU for load in gpu_load.values())
        or any(len(set(group)) != len(group) for group in gpu_groups.values())
    ):
        raise AssertionError("Internal C4 weighted GPU allocation is unsafe.")
    return slots, {label: gpu_groups[label] for label in PROTOCOL_LABELS}


def _run_protocol_group(
    *,
    protocol_label: str,
    jobs: Sequence[Job],
    repository: Path,
    dataset: Path,
    checkpoint_manifest: Path,
    output_root: Path,
    gpus: Sequence[str],
    max_concurrency: int,
    poll_seconds: float,
    allocation_metadata: Mapping[str, Any],
) -> int:
    plan = StagePlan(
        stage="formal",
        versions=("v1",),
        variants=("c4",),
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
        output_root=output_root / "_protocol_runners" / protocol_label,
        gpus=gpus,
        max_concurrency=max_concurrency,
        max_jobs_per_gpu=MAX_PROTOCOL_JOBS_PER_GPU,
        formal_selection=None,
        expected_selection_file_sha256=(
            EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL[protocol_label]
        ),
        poll_seconds=poll_seconds,
        job_output_validator=validate_c4_job_output,
        launcher_metadata={
            **allocation_metadata,
            "launcher": "actor_free_td_lewm_v1_c4_formal_evaluations",
            "matrix_protocol_label": protocol_label,
            "evaluation_protocol": protocol_label.upper(),
            "expected_selection_ranks_sha256": (
                EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL[protocol_label]
            ),
            "expected_action_normalization_sha256": (
                EXPECTED_ACTION_NORMALIZATION_SHA256
            ),
        },
    )


def run_c4_formal_evaluations(
    *,
    jobs: Sequence[Job],
    repository: Path,
    dataset: Path,
    checkpoint_manifest: Path,
    output_root: Path,
    gpus: Sequence[str],
    max_concurrency: int,
    poll_seconds: float,
) -> int:
    """Run protocol groups concurrently without mixing their rank selections."""

    groups = _protocol_groups(jobs)
    slots, gpu_groups = resolve_cost_weighted_gpu_allocation(
        gpus=gpus,
        max_concurrency=max_concurrency,
    )
    allocation_metadata = {
        "dispatch_policy": "fixed_cost_weighted_2_3_5",
        "protocol_cost_weights": dict(PROTOCOL_COST_WEIGHTS),
        "protocol_slots": dict(slots),
        "protocol_gpus": {label: list(values) for label, values in gpu_groups.items()},
        "global_max_jobs_per_gpu": MAX_GLOBAL_JOBS_PER_GPU,
        "protocol_max_jobs_per_gpu": MAX_PROTOCOL_JOBS_PER_GPU,
    }
    with ThreadPoolExecutor(max_workers=len(PROTOCOL_LABELS)) as executor:
        futures = {
            label: executor.submit(
                _run_protocol_group,
                protocol_label=label,
                jobs=groups[label],
                repository=repository,
                dataset=dataset,
                checkpoint_manifest=checkpoint_manifest,
                output_root=output_root,
                gpus=gpu_groups[label],
                max_concurrency=slots[label],
                poll_seconds=poll_seconds,
                allocation_metadata=allocation_metadata,
            )
            for label in PROTOCOL_LABELS
        }
        # Each child manifest records the complete static allocation through
        # launcher metadata assembled in _run_protocol_group.
        failed = any(future.result() != 0 for future in futures.values())
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked V1-C4 six-score matrix on formal Cube O25, O50, "
            "and O100."
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
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help=(
            "Global evaluation slots. With five GPUs use 10: O100/O50/O25 "
            "receive 5/3/2 slots, with at most two jobs per GPU."
        ),
    )
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
        output_root, checkpoint=checkpoint
    )
    jobs = build_c4_formal_evaluation_jobs(
        repository=repository,
        output_root=output_root,
        dataset=dataset,
        checkpoint=checkpoint,
        python=args.python,
    )
    validate_c4_formal_job_set(jobs)
    return run_c4_formal_evaluations(
        jobs=jobs,
        repository=repository,
        dataset=dataset,
        checkpoint_manifest=checkpoint_manifest,
        output_root=output_root,
        gpus=args.gpus,
        max_concurrency=args.max_concurrency,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
