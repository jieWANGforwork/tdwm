from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tdwm.adapters.actor_free_td_lewm_v1_c3 import STATE_V_FIRST_Q2_SCORE_MODE

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v1_c_c3_o100_evaluations.py"
SPEC = importlib.util.spec_from_file_location("v1_c_c3_o100_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def _jobs(tmp_path: Path):
    return LAUNCHER.build_o100_jobs(
        repository=ROOT,
        output_root=tmp_path / "outputs",
        dataset=tmp_path / "cube.lance",
        c_checkpoint=tmp_path / "v1_c_epoch_10.pt",
        c3_checkpoint=tmp_path / "v1_c3_epoch_12.pt",
        python="python",
    )


def test_o100_launcher_builds_exact_seven_cell_matrix(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path)
    LAUNCHER.validate_o100_job_set(jobs)

    assert len(jobs) == 7
    assert {(job.variant, job.score_mode) for job in jobs} == {
        *(("c", mode) for mode in LAUNCHER.C_SCORE_MODES),
        ("c3", STATE_V_FIRST_Q2_SCORE_MODE),
    }
    assert all("/formal/o100/v1/" in job.output_dir for job in jobs)
    c3_job = next(job for job in jobs if job.variant == "c3")
    assert c3_job.alpha == 0.1
    assert c3_job.config_path.endswith("_v1_c3_cube_checkpoint_o100.yaml")
    assert c3_job.argv[c3_job.argv.index("--g-first-weight") + 1] == (
        "0.10000000000000001"
    )
    for job in jobs:
        if job.variant == "c":
            expected_alpha = (
                0.25 if job.score_mode in LAUNCHER.FIRST_ACTION_MODES else None
            )
            assert job.alpha == expected_alpha
            assert job.config_path.endswith("_v1_c_cube_checkpoint_o100.yaml")


@pytest.mark.parametrize(
    "drift",
    ("drop", "duplicate", "alpha", "config", "score_arg", "checkpoint"),
)
def test_o100_launcher_rejects_matrix_or_command_drift(
    tmp_path: Path, drift: str
) -> None:
    jobs = _jobs(tmp_path)
    if drift == "drop":
        changed = jobs[:-1]
    elif drift == "duplicate":
        changed = [jobs[0], *jobs[:-1]]
    elif drift == "alpha":
        changed = [replace(jobs[0], alpha=0.25), *jobs[1:]]
    elif drift == "config":
        changed = [
            replace(jobs[0], config_path=jobs[0].config_path.replace("o100", "o50")),
            *jobs[1:],
        ]
    elif drift == "score_arg":
        argv = list(jobs[0].argv)
        argv[argv.index("--score-mode") + 1] = "g_only"
        changed = [replace(jobs[0], argv=tuple(argv)), *jobs[1:]]
    else:
        changed = [replace(jobs[1], checkpoint=jobs[0].checkpoint), *jobs[1:]]

    with pytest.raises(ValueError):
        LAUNCHER.validate_o100_job_set(changed)


def test_o100_checkpoint_manifest_locks_both_models_and_protocol_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c_checkpoint = tmp_path / "v1_c_epoch_10.pt"
    c3_checkpoint = tmp_path / "v1_c3_epoch_12.pt"
    c_checkpoint.write_bytes(b"c")
    c3_checkpoint.write_bytes(b"c3")

    def fake_sha(path: str | Path) -> str:
        return (
            LAUNCHER.EXPECTED_C_CHECKPOINT_SHA256
            if Path(path) == c_checkpoint
            else LAUNCHER.EXPECTED_C3_CHECKPOINT_SHA256
        )

    monkeypatch.setattr(LAUNCHER, "file_sha256", fake_sha)
    manifest_path = LAUNCHER.write_checkpoint_manifest(
        tmp_path,
        c_checkpoint=c_checkpoint,
        c3_checkpoint=c3_checkpoint,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["protocol_label"] == "o100"
    assert manifest["checkpoints"]["v1_c_e10"]["sha256"] == (
        LAUNCHER.EXPECTED_C_CHECKPOINT_SHA256
    )
    assert manifest["checkpoints"]["v1_c3_e12"]["sha256"] == (
        LAUNCHER.EXPECTED_C3_CHECKPOINT_SHA256
    )
    assert manifest["expected_selection_file_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_FILE_SHA256
    )
    assert manifest["expected_selection_ranks_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256
    )
    assert manifest["expected_action_normalization_sha256"] == (
        LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
    )


def test_o100_main_dispatches_seven_jobs_through_shared_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "cube.lance"
    c_checkpoint = tmp_path / "v1_c_epoch_10.pt"
    c3_checkpoint = tmp_path / "v1_c3_epoch_12.pt"
    dataset.mkdir()
    c_checkpoint.write_bytes(b"c")
    c3_checkpoint.write_bytes(b"c3")

    def fake_sha(path: str | Path) -> str:
        return (
            LAUNCHER.EXPECTED_C_CHECKPOINT_SHA256
            if Path(path) == c_checkpoint
            else LAUNCHER.EXPECTED_C3_CHECKPOINT_SHA256
        )

    captured = {}

    def fake_run_jobs(**kwargs):
        captured.update(kwargs)
        return 17

    monkeypatch.setattr(LAUNCHER, "file_sha256", fake_sha)
    monkeypatch.setattr(LAUNCHER, "run_jobs", fake_run_jobs)
    result = LAUNCHER.main(
        [
            "--dataset",
            str(dataset),
            "--c-checkpoint",
            str(c_checkpoint),
            "--c3-checkpoint",
            str(c3_checkpoint),
            "--output-root",
            str(tmp_path / "outputs"),
            "--repository",
            str(ROOT),
            "--python",
            "python",
            "--gpus",
            "0",
            "1",
            "--max-concurrency",
            "4",
        ]
    )

    assert result == 17
    assert len(captured["jobs"]) == 7
    assert captured["plan"].variants == ("c", "c3")
    assert captured["gpus"] == ["0", "1"]
    assert captured["max_concurrency"] == 4
    assert captured["expected_selection_file_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_FILE_SHA256
    )
    assert captured["job_output_validator"] is LAUNCHER.validate_o100_job_output


def test_o100_validator_rejects_any_selection_override(tmp_path: Path) -> None:
    job = _jobs(tmp_path)[0]
    with pytest.raises(ValueError, match="cannot be overridden"):
        LAUNCHER.validate_o100_job_output(job, expected_selection_file_sha256="0" * 64)
