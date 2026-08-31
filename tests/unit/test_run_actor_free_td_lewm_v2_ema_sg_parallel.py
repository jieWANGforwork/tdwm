from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v2_ema_sg_parallel.py"
LEGACY_SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v2_parallel.py"
SPEC = importlib.util.spec_from_file_location("v2_ema_sg_parallel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)


def _paths(tmp_path: Path) -> object:
    bundle = tmp_path / "actor_free_td_lewm_v2_ema_sg_bundle"
    return LAUNCHER.Paths(
        repository=ROOT,
        artifact_root=tmp_path,
        dataset=tmp_path / "cube.lance",
        dataset_manifest=tmp_path / "cube.lance.manifest.json",
        split_indices=tmp_path / "split_indices.npz",
        neighbor_index=tmp_path / "neighbor_index",
        v1_root=tmp_path / "v1",
        bundle_root=bundle,
        smoke_root=bundle / "smoke",
        formal_root=bundle / "formal",
        evaluation_root=bundle / "evaluations",
    )


def _expected_source() -> str:
    return (
        LEGACY_SCRIPT.read_text()
        .replace("actor_free_td_lewm_v2", "actor_free_td_lewm_v2_ema_sg")
        .replace("Actor-Free TD-LeWM V2", "Actor-Free TD-LeWM V2-EMA-SG")
        .replace("V2 launcher", "V2-EMA-SG launcher")
        .replace("V2 requires", "V2-EMA-SG requires")
        .replace("Another V2 launcher", "Another V2-EMA-SG launcher")
        .replace(
            '"implementation_version": "v2"',
            '"implementation_version": "v2_ema_sg"',
        )
        .replace(
            '"source": "v2_formal_training_launcher"',
            '"source": "v2_ema_sg_formal_training_launcher"',
        )
        .replace(
            '"source": "v2_parallel_launcher"',
            '"source": "v2_ema_sg_parallel_launcher"',
        )
    )


def test_launcher_is_server_tested_v2_orchestration_with_only_ema_sg_identity() -> None:
    assert ast.dump(ast.parse(SCRIPT.read_text())) == ast.dump(
        ast.parse(_expected_source())
    )


def test_formal_jobs_start_from_matching_v1_and_never_resume_by_default(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(stage="formal", paths=paths, python="python")

    assert len(jobs) == 6
    assert {job.variant for job in jobs} == set(LAUNCHER.VARIANTS)
    for job in jobs:
        method = f"actor_free_td_lewm_v2_ema_sg_{job.variant}"
        assert job.argv[1] == f"scripts/train_{method}.py"
        assert job.config_path == (f"configs/experiment/{method}_cube_train.yaml")
        assert job.argv[-2:] == ("--resume", "never")
        assert "--initial-v1-checkpoint" in job.argv
        assert "--initial-v2-checkpoint" not in job.argv
        initial = Path(job.argv[job.argv.index("--initial-v1-checkpoint") + 1])
        assert initial == LAUNCHER._v1_checkpoint(paths, job.variant)
        assert method in str(job.expected_checkpoint)

    resumed = LAUNCHER.build_jobs(
        stage="formal",
        paths=paths,
        python="python",
        formal_resume="required",
    )
    assert all(job.argv[-2:] == ("--resume", "required") for job in resumed)


def test_eval_jobs_use_independent_ema_sg_configs_scripts_and_outputs(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(stage="eval", paths=paths, python="python")

    assert len(jobs) == 18
    assert len({job.output_base for job in jobs}) == 18
    for job in jobs:
        method = f"actor_free_td_lewm_v2_ema_sg_{job.variant}"
        assert job.argv[1] == f"scripts/evaluate_{method}.py"
        assert job.config_path == (
            f"configs/experiment/{method}_cube_checkpoint_o50.yaml"
        )
        assert Path(job.output_base) == (
            paths.evaluation_root / job.variant / str(job.score_mode)
        )
        assert method in str(job.expected_checkpoint)


def test_eval_verification_rejects_legacy_v2_identity(tmp_path: Path) -> None:
    job = LAUNCHER.build_jobs(stage="eval", paths=_paths(tmp_path), python="python")[0]
    output = Path(job.run_dir)
    output.mkdir(parents=True)
    result = {
        "method": f"actor_free_td_lewm_v2_{job.variant}",
        "method_family": "actor_free_td_lewm_v2",
        "variant": job.variant,
        "implementation_version": "v2",
        "score_mode": job.score_mode,
        "smoke": False,
        "pilot": False,
        "metrics": {"episode_successes": [False] * 50},
    }
    (output / "results.json").write_text(json.dumps(result))
    for name in (
        "protocol_manifest.json",
        "episode_selection.json",
        "action_normalization.json",
    ):
        (output / name).write_text("{}")

    with pytest.raises(ValueError, match="results.method"):
        LAUNCHER.verify_job_output(job)

    result.update(
        {
            "method": f"actor_free_td_lewm_v2_ema_sg_{job.variant}",
            "method_family": "actor_free_td_lewm_v2_ema_sg",
            "implementation_version": "v2_ema_sg",
        }
    )
    (output / "results.json").write_text(json.dumps(result))
    assert set(LAUNCHER.verify_job_output(job)) == {
        "results",
        "protocol_manifest",
        "episode_selection",
        "action_normalization",
    }


def test_default_bundle_is_independent_and_formal_resume_is_never(
    tmp_path: Path,
) -> None:
    args = LAUNCHER.build_parser().parse_args(
        ["--stage", "formal", "--artifact-root", str(tmp_path)]
    )
    paths = LAUNCHER._paths(args, ROOT, "abcdef0")

    assert args.formal_resume == "never"
    assert paths.bundle_root == (
        tmp_path / "outputs" / "actor_free_td_lewm_v2_ema_sg_cg3_abcdef0"
    )
