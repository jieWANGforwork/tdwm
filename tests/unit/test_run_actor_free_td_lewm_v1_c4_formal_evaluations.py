from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tdwm.evaluation.actor_free_td_lewm_v1_c4 import (
    configure_actor_free_td_lewm_v1_c4_evaluation_mode,
    load_actor_free_td_lewm_v1_c4_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import _execution_metadata

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v1_c4_formal_evaluations.py"
SPEC = importlib.util.spec_from_file_location("v1_c4_formal_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def _jobs(tmp_path: Path):
    return LAUNCHER.build_c4_formal_evaluation_jobs(
        repository=ROOT,
        output_root=tmp_path / "outputs",
        dataset=tmp_path / "cube.lance",
        checkpoint=tmp_path / "v1_c4_epoch_10.pt",
        python="python",
    )


def test_c4_launcher_builds_exact_locked_18_cell_matrix(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path)
    LAUNCHER.validate_c4_formal_job_set(jobs)

    assert len(jobs) == 18
    assert len({job.job_id for job in jobs}) == 18
    assert len({job.output_dir for job in jobs}) == 18
    assert {
        (LAUNCHER._protocol_from_job(job), job.score_mode) for job in jobs
    } == {
        (protocol, score_mode)
        for protocol in LAUNCHER.PROTOCOL_LABELS
        for score_mode in LAUNCHER.SCORE_MODES
    }
    for job in jobs:
        expected_alpha = (
            0.25
            if job.score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}
            else None
        )
        assert job.alpha == expected_alpha
        assert "--smoke" not in job.argv
        assert "--pilot" not in job.argv
        assert job.config_path.endswith(
            f"_checkpoint_{LAUNCHER._protocol_from_job(job)}.yaml"
        )


@pytest.mark.parametrize(
    "drift",
    (
        "drop",
        "duplicate",
        "output",
        "dataset",
        "checkpoint",
        "python",
        "config",
        "score_arg",
        "alpha",
        "smoke",
    ),
)
def test_c4_launcher_rejects_matrix_or_input_drift(
    tmp_path: Path, drift: str
) -> None:
    jobs = _jobs(tmp_path)
    if drift == "drop":
        changed = jobs[:-1]
    elif drift == "duplicate":
        changed = [jobs[0], *jobs[:-1]]
    elif drift == "output":
        changed = [replace(jobs[0], output_dir=jobs[1].output_dir), *jobs[1:]]
    elif drift in {"dataset", "checkpoint"}:
        flag = f"--{drift}" if drift == "dataset" else "--checkpoint-path"
        argv = list(jobs[0].argv)
        argv[argv.index(flag) + 1] = f"other-{drift}"
        changed = [replace(jobs[0], argv=tuple(argv)), *jobs[1:]]
    elif drift == "python":
        argv = ("other-python", *jobs[0].argv[1:])
        changed = [replace(jobs[0], argv=argv), *jobs[1:]]
    elif drift == "config":
        changed = [
            replace(jobs[0], config_path=jobs[0].config_path.replace("o25", "o50")),
            *jobs[1:],
        ]
    elif drift == "score_arg":
        argv = list(jobs[0].argv)
        argv[argv.index("--score-mode") + 1] = "g_only"
        changed = [replace(jobs[0], argv=tuple(argv)), *jobs[1:]]
    elif drift == "alpha":
        changed = [replace(jobs[0], alpha=0.25), *jobs[1:]]
    else:
        changed = [replace(jobs[0], argv=(*jobs[0].argv, "--smoke")), *jobs[1:]]

    with pytest.raises(ValueError):
        LAUNCHER.validate_c4_formal_job_set(changed)


def test_checkpoint_manifest_dynamically_hashes_and_locks_new_c4_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch_10.pt"
    checkpoint.write_bytes(b"new-c4-checkpoint")

    path = LAUNCHER.write_checkpoint_manifest(tmp_path, checkpoint=checkpoint)
    payload = json.loads(path.read_text())

    assert payload["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "sha256": LAUNCHER.file_sha256(checkpoint),
    }
    assert set(payload["protocols"]) == set(LAUNCHER.PROTOCOL_LABELS)
    assert payload["score_modes"] == list(LAUNCHER.SCORE_MODES)
    assert payload["first_q_alpha"] == 0.25
    assert payload["alpha_selection_performed"] is False
    assert payload["training_performed"] is False
    for protocol, evidence in payload["protocols"].items():
        assert evidence["selection_file_sha256"] == (
            LAUNCHER.EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL[protocol]
        )
        assert evidence["selection_ranks_sha256"] == (
            LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL[protocol]
        )
        assert evidence["action_normalization_sha256"] == (
            LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
        )

    checkpoint.write_bytes(b"changed-after-lock")
    with pytest.raises(RuntimeError, match="binds other"):
        LAUNCHER.write_checkpoint_manifest(tmp_path, checkpoint=checkpoint)


def test_c4_dispatch_reuses_shared_runner_per_protocol_and_total_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = _jobs(tmp_path)
    calls: list[dict] = []

    def fake_run_jobs(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(LAUNCHER, "run_jobs", fake_run_jobs)
    result = LAUNCHER.run_c4_formal_evaluations(
        jobs=jobs,
        repository=ROOT,
        dataset=tmp_path / "cube.lance",
        checkpoint_manifest=tmp_path / "checkpoint_manifest.json",
        output_root=tmp_path / "outputs",
        gpus=("0", "1", "2"),
        max_concurrency=6,
        poll_seconds=0.0,
    )

    assert result == 0
    assert len(calls) == 3
    assert sum(call["max_concurrency"] for call in calls) == 6
    assert {call["launcher_metadata"]["matrix_protocol_label"] for call in calls} == {
        "o25",
        "o50",
        "o100",
    }
    assert sorted(gpu for call in calls for gpu in call["gpus"]) == ["0", "1", "2"]
    assert all(len(call["jobs"]) == 6 for call in calls)
    assert all(call["job_output_validator"] is LAUNCHER.validate_c4_job_output for call in calls)


def _write_valid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, list[int]]:
    jobs = _jobs(tmp_path)
    job = next(
        item
        for item in jobs
        if LAUNCHER._protocol_from_job(item) == "o50"
        and item.score_mode == "f_plus_g_first_q2"
    )
    checkpoint = Path(job.checkpoint)
    checkpoint.write_bytes(b"c4")
    output_root = tmp_path / "outputs"
    LAUNCHER.write_checkpoint_manifest(output_root, checkpoint=checkpoint)
    output = Path(job.output_dir)
    output.mkdir(parents=True)

    formal = load_actor_free_td_lewm_v1_c4_evaluation_protocol(
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c4_cube_checkpoint_o50.yaml"
    )
    protocol = configure_actor_free_td_lewm_v1_c4_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=job.score_mode,
        g_first_weight=0.25,
    )
    ranks = list(range(50))
    selection = {"valid_row_ranks": ranks}
    (output / "episode_selection.json").write_text(json.dumps(selection))
    (output / "action_normalization.json").write_text("{}\n")

    outcomes = [index % 2 == 0 for index in range(50)]
    common = {
        "method": "actor_free_td_lewm_v1_c4",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c4",
        "implementation_version": "v1",
        "evaluation_protocol": "O50",
        "protocol_label": "o50",
        "goal_offset": 50,
        "episode_budget": 100,
        "score_mode": job.score_mode,
        "state_only_g": True,
        "action_enters_g": False,
        "action_effect": "only_via_f_predicted_state",
        "g_first_weight": 0.25,
        "score_definition": protocol["inference_objective"]["score_definition"],
        **_execution_metadata(protocol["planning"]),
    }
    results = {
        **common,
        "planning_horizon": 5,
        "smoke": False,
        "pilot": False,
        "metrics": {"episode_successes": outcomes, "success_rate": 50.0},
    }
    checkpoint_sha = LAUNCHER.file_sha256(checkpoint)
    manifest = {
        **common,
        "protocol": protocol,
        "formal_protocol": formal,
        "selection": selection,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "epoch": 10,
            "global_step": 127_960,
            "formal_completion_required": True,
            "g_config": {},
        },
    }
    (output / "results.json").write_text(json.dumps(results))
    (output / "protocol_manifest.json").write_text(json.dumps(manifest))

    real_file_sha256 = LAUNCHER.file_sha256

    def selected_hash(path: str | Path) -> str:
        path = Path(path)
        if path.name == "episode_selection.json":
            return LAUNCHER.EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL["o50"]
        if path.name == "action_normalization.json":
            return LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
        return real_file_sha256(path)

    monkeypatch.setattr(LAUNCHER, "file_sha256", selected_hash)
    monkeypatch.setitem(
        LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL,
        "o50",
        LAUNCHER.canonical_json_sha256(ranks),
    )
    return job, ranks


def test_c4_output_validator_locks_protocol_outcomes_and_all_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, ranks = _write_valid_output(tmp_path, monkeypatch)

    evidence = LAUNCHER.validate_c4_job_output(
        job,
        expected_selection_file_sha256=(
            LAUNCHER.EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL["o50"]
        ),
    )

    assert evidence["protocol_label"] == "o50"
    assert evidence["success_count"] == 25
    assert evidence["episode_successes"] == [index % 2 == 0 for index in ranks]
    assert evidence["checkpoint_sha256"] == LAUNCHER.file_sha256(job.checkpoint)
    assert evidence["valid_row_ranks_sha256"] == (
        LAUNCHER.canonical_json_sha256(ranks)
    )
    assert evidence["action_normalization_sha256"] == (
        LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
    )


def test_c4_output_validator_rejects_non_boolean_episode_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, _ = _write_valid_output(tmp_path, monkeypatch)
    results_path = Path(job.output_dir) / "results.json"
    results = json.loads(results_path.read_text())
    results["metrics"]["episode_successes"][0] = 1
    results_path.write_text(json.dumps(results))

    with pytest.raises(ValueError, match="50 Boolean"):
        LAUNCHER.validate_c4_job_output(job)


def test_c4_main_builds_all_jobs_and_forwards_gpu_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "cube.lance"
    dataset.mkdir()
    checkpoint = tmp_path / "epoch_10.pt"
    checkpoint.write_bytes(b"c4")
    captured = {}

    def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return 13

    monkeypatch.setattr(LAUNCHER, "run_c4_formal_evaluations", fake_dispatch)
    result = LAUNCHER.main(
        [
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
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
            "8",
        ]
    )

    assert result == 13
    assert len(captured["jobs"]) == 18
    assert captured["gpus"] == ["0", "1"]
    assert captured["max_concurrency"] == 8
    assert captured["checkpoint_manifest"].is_file()
