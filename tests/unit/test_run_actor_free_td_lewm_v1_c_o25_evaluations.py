from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tdwm.evaluation.actor_free_td_lewm_v1_c import (
    configure_actor_free_td_lewm_v1_c_evaluation_mode,
    load_actor_free_td_lewm_v1_c_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    _execution_metadata,
    _first_action_output_metadata,
    _rollout_mean_output_metadata,
)
from tdwm.evaluation.lewm_checkpoint import sample_start_goal_pairs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v1_c_o25_evaluations.py"
SPEC = importlib.util.spec_from_file_location("v1_c_o25_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)

ACTION_NORMALIZATION = {
    "mean": [
        0.0108846967805138,
        -0.0031414329928013945,
        0.0026465825646189004,
        0.0004239286647865428,
        0.15925256653030173,
    ],
    "samples": 2_000_000,
    "scale": [
        0.2894198325506555,
        0.39371697495490987,
        0.6431365217957304,
        0.3928016202283424,
        0.2503073640045173,
    ],
    "variance": [
        0.08376383947364949,
        0.15501305636764512,
        0.41362458566751004,
        0.15429311285401096,
        0.06265377647488993,
    ],
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _selection() -> dict[str, list[int]]:
    episodes, starts, ranks = sample_start_goal_pairs(
        np.full(10_000, 201),
        goal_offset=25,
        episodes=50,
        seed=42,
    )
    return {
        "episode_indices": episodes.tolist(),
        "start_steps": starts.tolist(),
        "goal_steps": (starts + 25).tolist(),
        "valid_row_ranks": ranks.tolist(),
    }


def _jobs(tmp_path: Path):
    return LAUNCHER.build_o25_jobs(
        repository=ROOT,
        output_root=tmp_path / "outputs",
        dataset=tmp_path / "cube.lance",
        checkpoint=tmp_path / "epoch_10.pt",
        python="python",
    )


def test_o25_launcher_builds_exactly_six_isolated_fixed_alpha_jobs(
    tmp_path: Path,
) -> None:
    jobs = _jobs(tmp_path)
    LAUNCHER.validate_o25_job_set(jobs)

    assert len(jobs) == 6
    assert {job.score_mode for job in jobs} == set(LAUNCHER.SCORE_MODES)
    assert all("/formal/o25/v1/c/" in job.output_dir for job in jobs)
    for job in jobs:
        expected_alpha = (
            0.25 if job.score_mode in LAUNCHER.FIRST_ACTION_MODES else None
        )
        assert job.alpha == expected_alpha
        assert job.config_path.endswith("_v1_c_cube_checkpoint_o25.yaml")


@pytest.mark.parametrize(
    "drift",
    ("drop", "duplicate", "score_arg", "o50_config", "trailing_dataset"),
)
def test_o25_job_set_rejects_any_incomplete_or_drifted_manifest(
    tmp_path: Path,
    drift: str,
) -> None:
    jobs = _jobs(tmp_path)
    if drift == "drop":
        changed = jobs[:-1]
    elif drift == "duplicate":
        changed = [*jobs[:-1], jobs[0]]
    elif drift == "score_arg":
        argv = list(jobs[0].argv)
        argv[argv.index("--score-mode") + 1] = "g_only"
        changed = [replace(jobs[0], argv=tuple(argv)), *jobs[1:]]
    elif drift == "o50_config":
        changed = [
            replace(
                jobs[0],
                config_path=jobs[0].config_path.replace("o25.yaml", "o50.yaml"),
            ),
            *jobs[1:],
        ]
    else:
        changed = [
            replace(jobs[0], argv=(*jobs[0].argv, "--dataset", "/tmp/evil")),
            *jobs[1:],
        ]

    with pytest.raises(ValueError):
        LAUNCHER.validate_o25_job_set(changed)


def test_o25_checkpoint_manifest_is_bound_to_all_three_fixed_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "epoch_10.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        LAUNCHER,
        "file_sha256",
        lambda _path: LAUNCHER.EXPECTED_CHECKPOINT_SHA256,
    )

    path = LAUNCHER.write_checkpoint_manifest(tmp_path, checkpoint=checkpoint)
    manifest = json.loads(path.read_text())

    assert manifest["checkpoint"]["sha256"] == LAUNCHER.EXPECTED_CHECKPOINT_SHA256
    assert manifest["expected_selection_file_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_FILE_SHA256
    )
    assert manifest["expected_selection_ranks_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256
    )
    assert manifest["expected_action_normalization_sha256"] == (
        LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
    )


@pytest.mark.parametrize("score_mode", LAUNCHER.SCORE_MODES)
def test_o25_validator_accepts_each_complete_locked_job(
    tmp_path: Path,
    score_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = next(job for job in _jobs(tmp_path) if job.score_mode == score_mode)
    checkpoint = Path(job.checkpoint)
    checkpoint.write_bytes(b"fixture")
    real_file_sha256 = LAUNCHER.file_sha256
    monkeypatch.setattr(
        LAUNCHER,
        "file_sha256",
        lambda path: (
            LAUNCHER.EXPECTED_CHECKPOINT_SHA256
            if Path(path) == checkpoint
            else real_file_sha256(path)
        ),
    )
    formal = load_actor_free_td_lewm_v1_c_evaluation_protocol(job.config_path)
    protocol = configure_actor_free_td_lewm_v1_c_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=score_mode,
        g_first_weight=job.alpha,
    )
    planning = protocol["planning"]
    special = {
        **_first_action_output_metadata(protocol, planning),
        **_rollout_mean_output_metadata(protocol, planning),
        **_execution_metadata(planning),
    }
    output = Path(job.output_dir)
    results = {
        "method": "actor_free_td_lewm_v1_c",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c",
        "implementation_version": "v1",
        "protocol_label": "o25",
        "evaluation_protocol": "O25",
        "goal_offset": 25,
        "episode_budget": 50,
        "score_mode": score_mode,
        "planning_horizon": planning["horizon"],
        "smoke": False,
        "pilot": False,
        "metrics": {
            "episode_successes": [False] * 50,
            "success_rate": 0.0,
        },
        "protocol_manifest": str(output / "protocol_manifest.json"),
        **special,
    }
    manifest = {
        "protocol_label": "o25",
        "evaluation_protocol": "O25",
        "goal_offset": 25,
        "episode_budget": 50,
        "score_mode": score_mode,
        "protocol": protocol,
        "formal_protocol": formal,
        "protocol_path": str(Path(job.config_path).resolve()),
        "checkpoint": {
            "path": job.checkpoint,
            "sha256": LAUNCHER.EXPECTED_CHECKPOINT_SHA256,
        },
        "selection": _selection(),
        **special,
    }
    _write_json(output / "results.json", results)
    _write_json(output / "protocol_manifest.json", manifest)
    _write_json(output / "episode_selection.json", _selection())
    _write_json(output / "action_normalization.json", ACTION_NORMALIZATION)

    evidence = LAUNCHER.validate_o25_job_output(job)

    assert evidence["selection_file_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_FILE_SHA256
    )
    assert evidence["valid_row_ranks_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256
    )
    assert evidence["action_normalization_sha256"] == (
        LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
    )

    results["metrics"]["episode_successes"][0] = 1
    _write_json(output / "results.json", results)
    with pytest.raises(ValueError, match="50 boolean"):
        LAUNCHER.validate_o25_job_output(job)
