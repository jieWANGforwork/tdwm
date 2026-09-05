from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/run_actor_free_td_lewm_v1_c3_first_q_alpha_sweep.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("v1_c3_first_q_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path: Path):
    repository = tmp_path / "repository"
    (repository / "scripts").mkdir(parents=True)
    (repository / "configs/experiment").mkdir(parents=True)
    for name in (
        "evaluate_actor_free_td_lewm_v1_c3.py",
        "evaluate_actor_free_td_lewm_v1_c.py",
    ):
        (repository / "scripts" / name).write_text("# evaluator\n")
    for name in (
        "actor_free_td_lewm_v1_c3_cube_checkpoint_o50.yaml",
        "actor_free_td_lewm_v1_c_cube_checkpoint_o50.yaml",
    ):
        (repository / "configs/experiment" / name).write_text("id: test\n")
    dataset = tmp_path / "dataset.lance"
    dataset.mkdir()
    c3_checkpoint = tmp_path / "c3.pt"
    c3_checkpoint.write_bytes(b"c3")
    v1_c_checkpoint = tmp_path / "v1_c.pt"
    v1_c_checkpoint.write_bytes(b"v1-c")
    return {
        "repository": repository,
        "output_root": tmp_path / "outputs",
        "dataset": dataset,
        "python": Path(sys.executable),
        "c3_checkpoint": c3_checkpoint,
        "v1_c_checkpoint": v1_c_checkpoint,
    }


def test_build_jobs_creates_three_matched_alpha_grids(tmp_path: Path) -> None:
    module = _load_module()
    jobs = module.build_jobs(**_inputs(tmp_path), alphas=(0.1, 0.25, 1.0))

    assert len(jobs) == 9
    assert {job.family for job in jobs} == {
        "c3_raw",
        "c3_zscore",
        "v1_c_first_q",
    }
    assert {job.alpha for job in jobs} == {0.1, 0.25, 1.0}
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert len({job.output_dir for job in jobs}) == len(jobs)
    for job in jobs:
        assert "--g-first-weight" in job.argv
        assert "--score-mode" in job.argv
        assert job.argv[job.argv.index("--g-first-weight") + 1] == format(
            job.alpha, ".17g"
        )
        assert job.checkpoint_sha256 == module._file_sha256(Path(job.checkpoint))


def test_build_jobs_preserves_virtual_environment_python_symlink(
    tmp_path: Path,
) -> None:
    module = _load_module()
    inputs = _inputs(tmp_path)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(inputs["python"])
    inputs["python"] = venv_python

    jobs = module.build_jobs(**inputs, alphas=(0.25,))

    assert all(job.argv[0] == str(venv_python.absolute()) for job in jobs)


@pytest.mark.parametrize("alphas", [(), (0.1, 0.1), (-0.1,), (float("nan"),)])
def test_build_jobs_rejects_invalid_alpha_grid(
    tmp_path: Path, alphas: tuple[float, ...]
) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="alpha"):
        module.build_jobs(**_inputs(tmp_path), alphas=alphas)


def test_validate_job_output_binds_formula_checkpoint_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    job = module.build_jobs(**_inputs(tmp_path), alphas=(0.5,))[0]
    output = Path(job.output_dir)
    output.mkdir(parents=True)
    selection = {
        "episode_indices": list(range(50)),
        "start_steps": [0] * 50,
        "goal_steps": [50] * 50,
        "valid_row_ranks": list(range(50)),
    }
    selection_path = output / "episode_selection.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
    selection_sha = module._file_sha256(selection_path)
    monkeypatch.setattr(module, "EXPECTED_SELECTION_FILE_SHA256", selection_sha)
    definition = {
        "formula": "target_state_v - g_first_weight * q_first",
        "normalization": "none_raw_scores",
    }
    outcomes = [index % 2 == 0 for index in range(50)]
    results = {
        "score_mode": job.score_mode,
        "g_first_weight": job.alpha,
        "score_definition": definition,
        "smoke": False,
        "pilot": False,
        "planning_horizon": 5,
        "selection_sha256": selection_sha,
        "elapsed_seconds": 1.0,
        "metrics": {"episode_successes": outcomes, "success_rate": 50.0},
    }
    manifest = {
        "score_mode": job.score_mode,
        "g_first_weight": job.alpha,
        "score_definition": definition,
        "protocol": {
            "inference_objective": {
                "score_mode": job.score_mode,
                "g_first_weight": job.alpha,
            }
        },
        "checkpoint": {
            "path": job.checkpoint,
            "sha256": job.checkpoint_sha256,
        },
    }
    (output / "results.json").write_text(json.dumps(results))
    (output / "protocol_manifest.json").write_text(json.dumps(manifest))

    evidence = module.validate_job_output(job)

    assert evidence["success_count"] == 25
    assert evidence["success_rate"] == 50.0
    assert evidence["alpha"] == 0.5


def test_validate_v1_c_accepts_selection_file_without_redundant_result_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    job = module.build_jobs(**_inputs(tmp_path), alphas=(0.5,))[2]
    output = Path(job.output_dir)
    output.mkdir(parents=True)
    selection_path = output / "episode_selection.json"
    selection_path.write_text(json.dumps({"locked": list(range(50))}) + "\n")
    selection_sha = module._file_sha256(selection_path)
    monkeypatch.setattr(module, "EXPECTED_SELECTION_FILE_SHA256", selection_sha)
    outcomes = [index % 2 == 0 for index in range(50)]
    definition = {"formula": "f_cost - g_first_weight * q_first"}
    results = {
        "score_mode": job.score_mode,
        "g_first_weight": job.alpha,
        "score_definition": definition,
        "smoke": False,
        "pilot": False,
        "planning_horizon": 5,
        "elapsed_seconds": 1.0,
        "metrics": {"episode_successes": outcomes, "success_rate": 50.0},
    }
    manifest = {
        "score_mode": job.score_mode,
        "g_first_weight": job.alpha,
        "score_definition": definition,
        "protocol": {
            "inference_objective": {
                "score_mode": job.score_mode,
                "g_first_weight": job.alpha,
            }
        },
        "checkpoint": {
            "path": job.checkpoint,
            "sha256": job.checkpoint_sha256,
        },
    }
    (output / "results.json").write_text(json.dumps(results))
    (output / "protocol_manifest.json").write_text(json.dumps(manifest))

    evidence = module.validate_job_output(job)

    assert evidence["success_count"] == 25
    assert evidence["selection_file_sha256"] == selection_sha


def test_score_definitions_fail_closed_across_raw_and_zscore(tmp_path: Path) -> None:
    module = _load_module()
    jobs = module.build_jobs(**_inputs(tmp_path), alphas=(0.25,))
    raw = next(job for job in jobs if job.family == "c3_raw")
    zscore = next(job for job in jobs if job.family == "c3_zscore")
    original = next(job for job in jobs if job.family == "v1_c_first_q")

    module._validate_score_definition(
        raw,
        {"formula": "V - g_first_weight * Q", "normalization": "none_raw_scores"},
    )
    module._validate_score_definition(
        zscore,
        {
            "formula": "zscore(V) - g_first_weight * zscore(Q)",
            "normalization": "population_z_score",
        },
    )
    module._validate_score_definition(
        original,
        {"formula": "F - g_first_weight * Q"},
    )
    with pytest.raises(ValueError, match="not the raw"):
        module._validate_score_definition(
            raw,
            {
                "formula": "zscore(V) - g_first_weight * zscore(Q)",
                "normalization": "population_z_score",
            },
        )
