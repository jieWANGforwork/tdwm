from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tdwm.adapters.actor_free_td_lewm_v1_c3 import STATE_V_FIRST_Q2_SCORE_MODE
from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    configure_actor_free_td_lewm_v1_c3_evaluation_mode,
    load_actor_free_td_lewm_v1_c3_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import _execution_metadata
from tdwm.evaluation.lewm_checkpoint import sample_start_goal_pairs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v1_c3_first_q2_o25.py"
SPEC = importlib.util.spec_from_file_location("v1_c3_first_q2_o25", SCRIPT)
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


def _job(tmp_path: Path):
    return LAUNCHER.build_o25_job(
        repository=ROOT,
        output_root=tmp_path / "outputs",
        dataset=tmp_path / "cube.lance",
        checkpoint=tmp_path / "epoch_12.pt",
        python="python",
    )


def test_c3_first_q2_o25_launcher_builds_one_exact_cell(tmp_path: Path) -> None:
    job = _job(tmp_path)
    LAUNCHER.validate_o25_job(job)

    assert job.variant == "c3"
    assert job.score_mode == STATE_V_FIRST_Q2_SCORE_MODE
    assert job.alpha == 0.1
    assert "/formal/o25/v1/c3/state_v_plus_first_q2/alpha_0p1" in (
        job.output_dir
    )
    assert job.config_path.endswith("_v1_c3_cube_checkpoint_o25.yaml")
    assert job.argv[job.argv.index("--g-first-weight") + 1] == (
        "0.10000000000000001"
    )


@pytest.mark.parametrize("drift", ("alpha", "mode", "config", "smoke"))
def test_c3_first_q2_o25_launcher_rejects_protocol_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    job = _job(tmp_path)
    if drift == "alpha":
        changed = replace(job, alpha=0.25)
    elif drift == "mode":
        changed = replace(job, score_mode="state_v_terminal")
    elif drift == "config":
        changed = replace(job, config_path=job.config_path.replace("o25", "o50"))
    else:
        changed = replace(job, argv=(*job.argv, "--smoke"))
    with pytest.raises(ValueError):
        LAUNCHER.validate_o25_job(changed)


def test_c3_first_q2_o25_checkpoint_manifest_locks_all_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "epoch_12.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        LAUNCHER,
        "file_sha256",
        lambda _path: LAUNCHER.EXPECTED_CHECKPOINT_SHA256,
    )

    path = LAUNCHER.write_checkpoint_manifest(tmp_path, checkpoint=checkpoint)
    manifest = json.loads(path.read_text())

    assert manifest["checkpoint"]["sha256"] == LAUNCHER.EXPECTED_CHECKPOINT_SHA256
    assert manifest["score_mode"] == STATE_V_FIRST_Q2_SCORE_MODE
    assert manifest["g_first_weight"] == 0.1
    assert manifest["expected_selection_file_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_FILE_SHA256
    )
    assert manifest["expected_selection_ranks_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256
    )
    assert manifest["expected_action_normalization_sha256"] == (
        LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
    )


def test_c3_first_q2_o25_validator_accepts_only_complete_locked_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(tmp_path)
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
    formal = load_actor_free_td_lewm_v1_c3_evaluation_protocol(job.config_path)
    protocol = configure_actor_free_td_lewm_v1_c3_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=STATE_V_FIRST_Q2_SCORE_MODE,
        g_first_weight=0.1,
    )
    execution = _execution_metadata(protocol["planning"])
    score_definition = protocol["inference_objective"]["score_definition"]
    selection = _selection()
    output = Path(job.output_dir)
    results = {
        "metrics": {
            "episode_successes": [False] * 50,
            "success_rate": 0.0,
        },
        "method": "actor_free_td_lewm_v1_c3",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c3",
        "implementation_version": "v1",
        "evaluation_protocol": "O25",
        "protocol_label": "o25",
        "goal_offset": 25,
        "episode_budget": 50,
        "score_mode": STATE_V_FIRST_Q2_SCORE_MODE,
        "score_definition": score_definition,
        "g_first_weight": 0.1,
        "planning_horizon": 5,
        "selection_sha256": LAUNCHER.EXPECTED_SELECTION_FILE_SHA256,
        "smoke": False,
        "pilot": False,
        **execution,
    }
    manifest = {
        "evaluation_protocol": "O25",
        "protocol_label": "o25",
        "goal_offset": 25,
        "episode_budget": 50,
        "score_mode": STATE_V_FIRST_Q2_SCORE_MODE,
        "score_definition": score_definition,
        "g_first_weight": 0.1,
        "protocol": protocol,
        "formal_protocol": formal,
        "checkpoint": {
            "path": job.checkpoint,
            "sha256": LAUNCHER.EXPECTED_CHECKPOINT_SHA256,
            "epoch": 12,
            "logical_epoch": 12,
            "global_step": 12_000,
            "formal_completion_required": True,
        },
        "selection": selection,
        "selection_sha256": LAUNCHER.EXPECTED_SELECTION_FILE_SHA256,
        **execution,
    }
    _write_json(output / "results.json", results)
    _write_json(output / "protocol_manifest.json", manifest)
    _write_json(output / "episode_selection.json", selection)
    _write_json(output / "action_normalization.json", ACTION_NORMALIZATION)

    evidence = LAUNCHER.validate_o25_job_output(job)

    assert evidence["success_count"] == 0
    assert evidence["checkpoint_sha256"] == LAUNCHER.EXPECTED_CHECKPOINT_SHA256
    assert evidence["selection_file_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_FILE_SHA256
    )
    assert evidence["valid_row_ranks_sha256"] == (
        LAUNCHER.EXPECTED_SELECTION_RANKS_SHA256
    )
    assert evidence["action_normalization_sha256"] == (
        LAUNCHER.EXPECTED_ACTION_NORMALIZATION_SHA256
    )

    results["score_definition"]["normalization"] = "none_raw_scores"
    _write_json(output / "results.json", results)
    with pytest.raises(ValueError, match="candidate-z-scored"):
        LAUNCHER.validate_o25_job_output(job)
