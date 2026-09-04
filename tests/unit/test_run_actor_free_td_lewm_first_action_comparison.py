from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_first_action_comparison.py"
SPEC = importlib.util.spec_from_file_location("first_action_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARISON
SPEC.loader.exec_module(COMPARISON)


def _checkpoint_manifest(tmp_path: Path) -> Path:
    checkpoint_root = tmp_path / "checkpoints"
    value: dict[str, dict[str, str]] = {}
    for version in COMPARISON.VERSIONS:
        value[version] = {}
        for variant in COMPARISON.VARIANTS:
            checkpoint = checkpoint_root / version / f"{variant}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{version}-{variant}".encode())
            value[version][variant] = str(checkpoint.relative_to(tmp_path))
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps(value))
    return manifest


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "cube.lance"
    path.mkdir()
    return path


def _jobs(
    tmp_path: Path,
    *,
    stage: str,
    versions: tuple[str, ...] | None = None,
    variants: tuple[str, ...] | None = None,
    score_modes: tuple[str, ...] | None = None,
    alphas: tuple[float, ...] | None = None,
):
    checkpoint_manifest = _checkpoint_manifest(tmp_path)
    checkpoints = COMPARISON.load_checkpoint_manifest(checkpoint_manifest)
    plan = COMPARISON.resolve_stage_plan(
        stage=stage,
        versions=versions,
        variants=variants,
        score_modes=score_modes,
        alphas=alphas,
    )
    jobs = COMPARISON.build_jobs(
        repository=ROOT,
        output_root=tmp_path / "comparison",
        checkpoints=checkpoints,
        dataset=_dataset(tmp_path),
        python="/venv/bin/python",
        plan=plan,
    )
    return checkpoint_manifest, plan, jobs


def _write_job_output(job, *, ranks: list[int] | None = None) -> None:
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected_episodes = COMPARISON.EXPECTED_EPISODES_BY_STAGE[job.stage]
    selected_ranks = ranks if ranks is not None else list(range(expected_episodes))
    result = {
        "method": f"actor_free_td_lewm_{job.version}_{job.variant}",
        "method_family": f"actor_free_td_lewm_{job.version}",
        "variant": job.variant,
        "implementation_version": job.version,
        "score_mode": job.score_mode,
        "planning_horizon": COMPARISON.expected_horizon(job.score_mode),
        "smoke": job.stage == "smoke",
        "pilot": job.stage == "development",
        "metrics": {"episode_successes": [False] * expected_episodes},
    }
    inference = {"score_mode": job.score_mode}
    manifest = {
        "score_mode": job.score_mode,
        "protocol": {
            "method": result["method"],
            "method_family": result["method_family"],
            "variant": job.variant,
            "implementation_version": job.version,
            "inference_objective": inference,
            "planning": {"horizon": COMPARISON.expected_horizon(job.score_mode)},
        },
        "checkpoint": {"path": job.checkpoint},
    }
    if job.score_mode == COMPARISON.FIRST_ACTION_MODE:
        definition = {"formula": "f_cost - g_first_weight * q_first"}
        for values in (result, inference, manifest):
            values["g_first_weight"] = job.alpha
            values["score_definition"] = definition
    elif job.score_mode == COMPARISON.ROLLOUT_MEAN_MODE:
        definition = {
            "formula": "cost = -mean(q1, q2, q3, q4, q5)",
            "action_processing": (
                COMPARISON.ROLLOUT_MEAN_ACTION_PROCESSING_BY_VERSION[job.version]
            ),
        }
        metadata = COMPARISON.ROLLOUT_MEAN_METADATA_BY_VERSION[job.version]
        for values in (result, manifest):
            values.update(metadata)
            values["score_definition"] = definition
        inference.update(
            COMPARISON.ROLLOUT_MEAN_INFERENCE_METADATA_BY_VERSION[job.version]
        )
        inference["score_definition"] = definition
    (output / "results.json").write_text(json.dumps(result))
    (output / "protocol_manifest.json").write_text(json.dumps(manifest))
    (output / "episode_selection.json").write_text(
        json.dumps({"valid_row_ranks": selected_ranks})
    )


def test_stage_contracts_are_explicit_and_formal_never_selects_alpha() -> None:
    smoke = COMPARISON.resolve_stage_plan(
        stage="smoke", variants=None, score_modes=None, alphas=None
    )
    assert smoke.variants == ("c",)
    assert smoke.versions == COMPARISON.VERSIONS
    assert smoke.score_modes == ("f_plus_g_first",)
    assert smoke.v2_only_score_modes == ()
    assert smoke.alphas == (1.0,)

    development = COMPARISON.resolve_stage_plan(
        stage="development", variants=None, score_modes=None, alphas=None
    )
    assert development.variants == COMPARISON.VARIANTS
    assert development.score_modes == ("f_plus_g_first",)
    assert development.v2_only_score_modes == ()
    assert development.alphas == (0.0, 0.25, 0.5, 1.0, 2.0)

    with pytest.raises(ValueError, match="exactly one explicit"):
        COMPARISON.resolve_stage_plan(
            stage="formal", variants=None, score_modes=None, alphas=None
        )
    with pytest.raises(ValueError, match="exactly one explicit"):
        COMPARISON.resolve_stage_plan(
            stage="formal",
            variants=None,
            score_modes=None,
            alphas=(0.5, 1.0),
        )
    formal = COMPARISON.resolve_stage_plan(
        stage="formal", variants=None, score_modes=None, alphas=(0.5,)
    )
    assert formal.variants == COMPARISON.VARIANTS
    assert formal.score_modes == ("f_only", "f_plus_g", "f_plus_g_first")
    assert formal.v2_only_score_modes == ("g_only", "g_only_f_rollout_mean")
    assert formal.alphas == (0.5,)


def test_checkpoint_manifest_is_exact_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    manifest = _checkpoint_manifest(tmp_path)
    loaded = COMPARISON.load_checkpoint_manifest(manifest)
    assert set(loaded) == set(COMPARISON.VERSIONS)
    assert all(set(value) == set(COMPARISON.VARIANTS) for value in loaded.values())
    assert loaded["v2"]["g3"] == (tmp_path / "checkpoints/v2/g3.pt").resolve()

    malformed = json.loads(manifest.read_text())
    del malformed["v1"]["g2"]
    manifest.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="v1 must contain exactly"):
        COMPARISON.load_checkpoint_manifest(manifest)


def test_development_builds_three_by_six_by_five_isolated_pilot_cells(
    tmp_path: Path,
) -> None:
    _, plan, jobs = _jobs(tmp_path, stage="development")
    assert plan.alphas == COMPARISON.DEVELOPMENT_ALPHA_GRID
    assert len(jobs) == 3 * 6 * 5
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert len({job.output_dir for job in jobs}) == len(jobs)
    for job in jobs:
        assert job.score_mode == "f_plus_g_first"
        assert "--pilot" in job.argv and "--smoke" not in job.argv
        assert "--g-first-weight" in job.argv
        assert Path(job.output_dir) == (
            tmp_path
            / "comparison"
            / "development"
            / job.version
            / job.variant
            / job.score_mode
            / f"alpha_{COMPARISON.alpha_slug(job.alpha)}"
        )


def test_formal_adds_v2_only_g_modes_without_sending_them_to_v0_or_v1(
    tmp_path: Path,
) -> None:
    _, _, jobs = _jobs(tmp_path, stage="formal", alphas=(1.0,))
    assert len(jobs) == (2 * 6 * 3) + (1 * 6 * 5)
    assert {(job.version, job.variant, job.score_mode) for job in jobs} == {
        (version, variant, mode)
        for version in COMPARISON.VERSIONS
        for variant in COMPARISON.VARIANTS
        for mode in (
            COMPARISON.FORMAL_SCORE_MODES
            + (COMPARISON.V2_ONLY_FORMAL_SCORE_MODES if version == "v2" else ())
        )
    }
    for job in jobs:
        assert "--pilot" not in job.argv and "--smoke" not in job.argv
        if job.score_mode == "f_plus_g_first":
            assert job.alpha == 1.0
            assert "--g-first-weight" in job.argv
            assert Path(job.output_dir).name == "alpha_1"
        else:
            assert job.alpha is None
            assert "--g-first-weight" not in job.argv
            assert Path(job.output_dir).name == job.score_mode
        if job.score_mode == "g_only":
            assert job.version == "v2"
        if job.score_mode == "g_only_f_rollout_mean":
            assert job.version == "v2"
            assert Path(job.config_path).name == (
                f"actor_free_td_lewm_v2_{job.variant}_cube_checkpoint_o50_"
                "g_only_f_rollout_mean.yaml"
            )
            assert job.argv[job.argv.index("--score-mode") + 1] == job.score_mode


def test_explicit_v0_v1_rollout_mean_builds_exactly_twelve_jobs_and_manifest(
    tmp_path: Path,
) -> None:
    checkpoint_manifest, plan, jobs = _jobs(
        tmp_path,
        stage="formal",
        versions=("v0", "v1"),
        score_modes=(COMPARISON.ROLLOUT_MEAN_MODE,),
    )

    assert plan.versions == ("v0", "v1")
    assert plan.score_modes == (COMPARISON.ROLLOUT_MEAN_MODE,)
    assert plan.v2_only_score_modes == ()
    assert plan.alphas == ()
    assert len(jobs) == 12
    assert {(job.version, job.variant, job.score_mode) for job in jobs} == {
        (version, variant, COMPARISON.ROLLOUT_MEAN_MODE)
        for version in ("v0", "v1")
        for variant in COMPARISON.VARIANTS
    }
    assert all("--g-first-weight" not in job.argv for job in jobs)
    assert all("v2" not in Path(job.config_path).name for job in jobs)
    assert all(Path(job.config_path).is_file() for job in jobs)

    expected_selection_file_sha256 = (
        "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
    )
    payload = COMPARISON._launcher_payload(
        plan=plan,
        repository=COMPARISON.ROOT if hasattr(COMPARISON, "ROOT") else ROOT,
        dataset=tmp_path / "cube.lance",
        checkpoint_manifest=checkpoint_manifest,
        output_root=tmp_path / "comparison",
        jobs=jobs,
        gpus=(),
        max_concurrency=12,
        expected_selection_file_sha256=expected_selection_file_sha256,
    )
    assert payload["versions"] == ["v0", "v1"]
    assert set(payload["score_modes_by_version"]) == {"v0", "v1"}
    assert len(payload["jobs"]) == 12
    assert payload["expected_selection_file_sha256"] == expected_selection_file_sha256


def test_formal_selection_file_sha_is_distinct_from_rank_sha_and_fail_closed(
    tmp_path: Path,
) -> None:
    _, _, jobs = _jobs(
        tmp_path,
        stage="formal",
        versions=("v0",),
        variants=("c",),
        score_modes=(COMPARISON.ROLLOUT_MEAN_MODE,),
    )
    job = jobs[0]
    _write_job_output(job)
    selection_path = Path(job.output_dir) / "episode_selection.json"
    expected_file_sha = COMPARISON.file_sha256(selection_path)

    evidence = COMPARISON.validate_job_output(
        job,
        expected_selection_file_sha256=expected_file_sha,
    )

    assert evidence["selection_file_sha256"] == expected_file_sha
    assert evidence["valid_row_ranks_sha256"] == evidence["selection_sha256"]
    assert evidence["selection_file_sha256"] != evidence["valid_row_ranks_sha256"]
    with pytest.raises(ValueError, match="does not match expected"):
        COMPARISON.validate_job_output(
            job,
            expected_selection_file_sha256="0" * 64,
        )


def test_smoke_defaults_to_c_for_all_three_versions_and_supports_variants(
    tmp_path: Path,
) -> None:
    _, _, default_jobs = _jobs(tmp_path, stage="smoke")
    assert len(default_jobs) == 3
    assert {(job.version, job.variant) for job in default_jobs} == {
        (version, "c") for version in COMPARISON.VERSIONS
    }
    assert all("--smoke" in job.argv for job in default_jobs)

    other = tmp_path / "other"
    other.mkdir()
    _, _, selected_jobs = _jobs(
        other,
        stage="smoke",
        variants=("c", "g3"),
        alphas=(2.0,),
    )
    assert len(selected_jobs) == 6
    assert {job.variant for job in selected_jobs} == {"c", "g3"}


def test_nonempty_stage_or_job_output_is_never_overwritten(tmp_path: Path) -> None:
    _, _, jobs = _jobs(tmp_path, stage="smoke")
    stage_root = tmp_path / "comparison" / "smoke"
    stage_root.mkdir(parents=True)
    (stage_root / "prior.txt").write_text("owned")
    with pytest.raises(FileExistsError, match="non-empty stage"):
        COMPARISON.ensure_fresh_stage(stage_root, jobs)


def test_output_validation_locks_identity_horizon_alpha_and_selection(
    tmp_path: Path,
) -> None:
    _, _, jobs = _jobs(tmp_path, stage="smoke")
    for job in jobs:
        _write_job_output(job)
    evidence = {job.job_id: COMPARISON.validate_job_output(job) for job in jobs}
    assert COMPARISON.require_identical_selections(evidence) == (0,)

    result_path = Path(jobs[0].output_dir) / "results.json"
    result = json.loads(result_path.read_text())
    result["planning_horizon"] = 4
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="planning_horizon"):
        COMPARISON.validate_job_output(jobs[0])

    _write_job_output(jobs[0])
    result = json.loads(result_path.read_text())
    result["metrics"]["episode_successes"] = []
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="exactly 1 outcomes"):
        COMPARISON.validate_job_output(jobs[0])

    _write_job_output(jobs[0])
    manifest_path = Path(jobs[0].output_dir) / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["g_first_weight"] = 99.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="g_first_weight"):
        COMPARISON.validate_job_output(jobs[0])

    _write_job_output(jobs[0], ranks=[1, 2])
    with pytest.raises(ValueError, match="selection must contain exactly 1"):
        COMPARISON.validate_job_output(jobs[0])


def test_g_only_and_rollout_mean_outputs_have_mode_specific_contracts(
    tmp_path: Path,
) -> None:
    _, _, jobs = _jobs(
        tmp_path,
        stage="formal",
        variants=("c",),
        alphas=(1.0,),
    )
    g_only = next(job for job in jobs if job.score_mode == "g_only")
    rollout_mean = next(
        job for job in jobs if job.score_mode == "g_only_f_rollout_mean"
    )
    assert g_only.version == rollout_mean.version == "v2"

    _write_job_output(g_only)
    g_only_evidence = COMPARISON.validate_job_output(g_only)
    assert len(g_only_evidence["valid_row_ranks"]) == 50
    assert (
        json.loads((Path(g_only.output_dir) / "results.json").read_text())[
            "planning_horizon"
        ]
        == 1
    )

    _write_job_output(rollout_mean)
    mean_evidence = COMPARISON.validate_job_output(rollout_mean)
    assert len(mean_evidence["valid_row_ranks"]) == 50
    mean_result_path = Path(rollout_mean.output_dir) / "results.json"
    mean_result = json.loads(mean_result_path.read_text())
    assert mean_result["planning_horizon"] == 5
    for key, expected in COMPARISON.ROLLOUT_MEAN_METADATA_BY_VERSION["v2"].items():
        assert mean_result[key] == expected
    assert mean_result["score_definition"]

    mean_manifest_path = Path(rollout_mean.output_dir) / "protocol_manifest.json"
    mean_manifest = json.loads(mean_manifest_path.read_text())
    for key, expected in COMPARISON.ROLLOUT_MEAN_METADATA_BY_VERSION["v2"].items():
        assert mean_manifest[key] == expected
    mean_inference = mean_manifest["protocol"]["inference_objective"]
    for key, expected in COMPARISON.ROLLOUT_MEAN_INFERENCE_METADATA_BY_VERSION[
        "v2"
    ].items():
        assert mean_inference[key] == expected
    assert mean_inference["score_definition"]

    mean_result["state_source_for_q1"] = "wrong_state"
    mean_result_path.write_text(json.dumps(mean_result))
    with pytest.raises(ValueError, match="results.state_source_for_q1"):
        COMPARISON.validate_job_output(rollout_mean)

    _write_job_output(rollout_mean)
    mean_manifest = json.loads(mean_manifest_path.read_text())
    mean_manifest["protocol"]["inference_objective"]["g_score"] = "wrong_score"
    mean_manifest_path.write_text(json.dumps(mean_manifest))
    with pytest.raises(ValueError, match="inference.g_score"):
        COMPARISON.validate_job_output(rollout_mean)

    _write_job_output(rollout_mean)
    mean_manifest = json.loads(mean_manifest_path.read_text())
    del mean_manifest["score_definition"]
    mean_manifest_path.write_text(json.dumps(mean_manifest))
    with pytest.raises(ValueError, match="manifest.score_definition"):
        COMPARISON.validate_job_output(rollout_mean)


def test_selection_mismatch_and_optional_formal_disjointness_fail_closed(
    tmp_path: Path,
) -> None:
    evidence = {
        "v0": {"valid_row_ranks": [1, 2]},
        "v1": {"valid_row_ranks": [1, 3]},
    }
    with pytest.raises(ValueError, match="Selection mismatch"):
        COMPARISON.require_identical_selections(evidence)

    unverified = COMPARISON.verify_formal_disjointness([1, 2], None)
    assert unverified["formal_disjointness_verified"] is False
    assert "does not claim" in unverified["note"]

    formal = tmp_path / "formal_selection.json"
    formal.write_text(json.dumps({"valid_row_ranks": [10, 20]}))
    verified = COMPARISON.verify_formal_disjointness([1, 2], formal)
    assert verified["formal_disjointness_verified"] is True
    assert verified["formal_selection_overlap"] == []

    formal.write_text(json.dumps({"valid_row_ranks": [2, 20]}))
    with pytest.raises(ValueError, match="overlap"):
        COMPARISON.verify_formal_disjointness([1, 2], formal)


class _ImmediateProcess:
    next_pid = 70_000

    def __init__(self) -> None:
        self.pid = type(self).next_pid
        type(self).next_pid += 1

    @staticmethod
    def poll() -> int:
        return 0


def test_scheduler_runs_in_parallel_and_writes_logs_manifest_and_evidence(
    tmp_path: Path,
) -> None:
    checkpoint_manifest, plan, jobs = _jobs(
        tmp_path,
        stage="development",
        variants=("c",),
        alphas=(0.5,),
    )
    by_output = {job.output_dir: job for job in jobs}
    gpu_assignments: list[str | None] = []

    def fake_popen(argv: list[str], **kwargs: object) -> _ImmediateProcess:
        output = argv[argv.index("--output-dir") + 1]
        job = by_output[output]
        _write_job_output(job)
        environment = kwargs["env"]
        gpu_assignments.append(environment.get("CUDA_VISIBLE_DEVICES"))
        log = kwargs["stdout"]
        log.write(f"completed {job.job_id}\n")
        log.flush()
        return _ImmediateProcess()

    code = COMPARISON.run_jobs(
        jobs=jobs,
        plan=plan,
        repository=ROOT,
        dataset=tmp_path / "cube.lance",
        checkpoint_manifest=checkpoint_manifest,
        output_root=tmp_path / "comparison",
        gpus=("0", "1"),
        max_concurrency=2,
        formal_selection=None,
        poll_seconds=0.0,
        popen=fake_popen,
        sleeper=lambda _: None,
    )

    assert code == 0
    assert gpu_assignments == ["0", "1", "0"]
    manifest_path = tmp_path / "comparison/development/_launcher/launcher_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "SUCCEEDED"
    assert manifest["inference_only"] is True
    assert manifest["training_performed"] is False
    assert manifest["alpha_selection_performed"] is False
    assert manifest["formal_disjointness_verified"] is False
    assert "does not claim" in manifest["note"]
    assert manifest["selection"]["identical_across_all_jobs"] is True
    assert all(value["state"] == "SUCCEEDED" for value in manifest["jobs"].values())
    for job in jobs:
        assert Path(job.log_path).read_text() == f"completed {job.job_id}\n"
