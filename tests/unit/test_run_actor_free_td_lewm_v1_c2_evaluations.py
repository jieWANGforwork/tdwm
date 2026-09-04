from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v1_c2_evaluations.py"
SPEC = importlib.util.spec_from_file_location("v1_c2_evaluation_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_c2_endpoint_launcher_builds_six_c2_jobs_and_one_c_reference() -> None:
    jobs = MODULE.build_c2_evaluation_jobs(
        repository=ROOT,
        output_root=ROOT / "unused-output",
        dataset=ROOT / "unused-dataset.lance",
        python="python",
        c2_checkpoint=ROOT / "c2.pt",
        v1_c_checkpoint=ROOT / "c.pt",
        stage="formal",
        alpha=0.25,
    )

    assert len(jobs) == 7
    assert len({job.job_id for job in jobs}) == 7
    assert len({job.output_dir for job in jobs}) == 7
    c2_jobs = [job for job in jobs if job.variant == "c2"]
    c_jobs = [job for job in jobs if job.variant == "c"]
    assert {job.score_mode for job in c2_jobs} == set(MODULE.C2_SCORE_MODES)
    assert [job.score_mode for job in c_jobs] == ["f_plus_g_first_q2"]
    assert all(job.stage == "formal" and job.version == "v1" for job in jobs)
    assert all(
        job.alpha == 0.25 for job in jobs if job.score_mode in MODULE.FIRST_ACTION_MODES
    )
    assert all(
        job.alpha is None
        for job in jobs
        if job.score_mode not in MODULE.FIRST_ACTION_MODES
    )
    assert any(
        job.config_path.endswith(
            "actor_free_td_lewm_v1_c2_cube_checkpoint_o50_g_only_f_rollout_mean.yaml"
        )
        for job in c2_jobs
    )


def test_c2_endpoint_launcher_smoke_flag_is_forwarded() -> None:
    jobs = MODULE.build_c2_evaluation_jobs(
        repository=ROOT,
        output_root=ROOT / "unused-output",
        dataset=ROOT / "unused-dataset.lance",
        python="python",
        c2_checkpoint=ROOT / "c2.pt",
        v1_c_checkpoint=ROOT / "c.pt",
        stage="smoke",
    )

    assert all(job.argv[-1] == "--smoke" for job in jobs)


def test_c2_endpoint_launcher_rejects_uncontrolled_stage_or_alpha() -> None:
    kwargs = {
        "repository": ROOT,
        "output_root": ROOT / "unused-output",
        "dataset": ROOT / "unused-dataset.lance",
        "python": "python",
        "c2_checkpoint": ROOT / "c2.pt",
        "v1_c_checkpoint": ROOT / "c.pt",
    }
    with pytest.raises(ValueError, match="smoke or formal"):
        MODULE.build_c2_evaluation_jobs(stage="development", **kwargs)
    with pytest.raises(ValueError, match="finite and non-negative"):
        MODULE.build_c2_evaluation_jobs(stage="formal", alpha=-1.0, **kwargs)


def test_c2_endpoint_launcher_prespecifies_previous_first_q_alpha() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "--stage",
            "formal",
            "--dataset",
            "dataset",
            "--c2-checkpoint",
            "c2.pt",
            "--v1-c-checkpoint",
            "c.pt",
            "--output-root",
            "outputs",
        ]
    )

    assert args.alpha == MODULE.PRESPECIFIED_ALPHA == 0.25
