from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "g_weighted_completion",
    ROOT / "scripts/run_actor_free_td_lewm_v1_g_weighted_completion.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
COMPARISON = sys.modules["run_actor_free_td_lewm_first_action_comparison"]


def jobs(tmp_path):
    return MODULE.build_completion_jobs(
        repository=ROOT, output_root=tmp_path / "out",
        checkpoint_root=tmp_path / "ckpts", dataset=tmp_path / "cube.lance",
        python="/venv/bin/python",
    )


def test_completion_is_exactly_30_new_cells_and_never_repeats_c_or_f_only(tmp_path):
    cells = jobs(tmp_path)
    assert len(cells) == 30
    assert {j.variant for j in cells} == {"d", "f", "g1", "g2", "g3"}
    assert len({j.output_dir for j in cells}) == len({j.job_id for j in cells}) == 30
    assert [MODULE.protocol_group(j) for j in cells] == (
        ["o100"] * 10 + ["o50"] * 10 + ["o25"] * 10
    )
    for j in cells:
        assert Path(j.config_path).is_file()
        assert j.score_mode in {"g_path_weighted_cem", "g_action_weighted_cem"}
        assert j.argv[j.argv.index("--checkpoint-sha256") + 1] == MODULE.V1_SHA256[j.variant]
        assert j.argv[j.argv.index("--temperature") + 1] == "1.0"
        assert "--smoke" not in j.argv and "--pilot" not in j.argv


def write_output(job, monkeypatch):
    out = Path(job.output_dir)
    out.mkdir(parents=True)
    label = MODULE.protocol_group(job)
    mode = job.score_mode.removeprefix("g_").removesuffix("_weighted_cem")
    definition = {
        "g_role": "elite_distribution_update_only",
        "elite_selection": "lowest_full_F_terminal_cost",
        "score_normalization": "none", "g_population": "selected_elites_only",
    }
    metadata = {"g_weighted_cem": {"mode": mode, "temperature": 1.0},
                "score_definition": definition}
    result = {**metadata, "method": f"actor_free_td_lewm_v1_{job.variant}",
              "variant": job.variant, "implementation_version": "v1",
              "score_mode": job.score_mode, "protocol_label": label,
              "planning_horizon": 5, "smoke": False, "pilot": False,
              "metrics": {"episode_successes": [True] * 20 + [False] * 30,
                          "success_rate": 40.0}}
    manifest = {**metadata, "checkpoint": {"sha256": MODULE.V1_SHA256[job.variant],
                                          "epoch": 10, "global_step": 127960},
                "protocol": {"planning": {
                    "horizon": 5, "candidates": 300, "iterations": 30, "elites": 30,
                    "action_block": 5, "planning_seed": 42,
                    "receding_horizon": 5 if label == "o25" else 1,
                    "episode_budget": int(label[1:]) * 2}}}
    files = {"results.json": result, "protocol_manifest.json": manifest,
             "episode_selection.json": {"valid_row_ranks": list(range(50))},
             "action_normalization.json": {"fixture": True}}
    for name, value in files.items():
        (out / name).write_text(json.dumps(value))
    monkeypatch.setitem(MODULE.SELECTION_SHA256, label,
                        MODULE.file_sha256(out / "episode_selection.json"))
    monkeypatch.setattr(MODULE, "ACTION_SHA256",
                        MODULE.file_sha256(out / "action_normalization.json"))
    return out


def test_formal_output_is_bound_to_checkpoint_pairs_and_weighting(tmp_path, monkeypatch):
    job = jobs(tmp_path)[0]
    write_output(job, monkeypatch)
    result = MODULE.validate_weighted_output(job)
    assert result["success_count"] == 20 and result["episode_count"] == 50


@pytest.mark.parametrize("change", ("checkpoint", "pairs", "normalization", "rate",
                                   "outcomes", "temperature", "budget", "readout"))
def test_corrupt_or_different_outputs_are_not_marked_completed(tmp_path, monkeypatch, change):
    job = jobs(tmp_path)[0]
    out = write_output(job, monkeypatch)
    result_path, manifest_path = out / "results.json", out / "protocol_manifest.json"
    result, manifest = json.loads(result_path.read_text()), json.loads(manifest_path.read_text())
    if change == "checkpoint":
        manifest["checkpoint"]["sha256"] = "0" * 64
    elif change == "pairs":
        (out / "episode_selection.json").write_text('{"valid_row_ranks": [999]}')
    elif change == "normalization":
        (out / "action_normalization.json").write_text('{}')
    elif change == "rate":
        result["metrics"]["success_rate"] = 99
    elif change == "outcomes":
        result["metrics"]["episode_successes"] = [True]
    elif change == "temperature":
        result["g_weighted_cem"]["temperature"] = 2
    elif change == "budget":
        manifest["protocol"]["planning"]["episode_budget"] = 50
    elif change == "readout":
        result["score_definition"]["g_role"] = "elite_selection"
    result_path.write_text(json.dumps(result))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        MODULE.validate_weighted_output(job)


def test_different_offsets_share_one_pool_but_not_one_pair_comparison(tmp_path):
    cells = jobs(tmp_path)
    evidence = {j.job_id: {
        "valid_row_ranks": [int(MODULE.protocol_group(j)[1:])],
        "selection_file_sha256": MODULE.SELECTION_SHA256[MODULE.protocol_group(j)],
    } for j in cells}
    with pytest.raises(ValueError, match="Selection mismatch"):
        COMPARISON.summarize_job_selections(evidence, cells)
    summary = COMPARISON.summarize_job_selections(evidence, cells, MODULE.protocol_group)
    assert set(summary["selection_groups"]) == {"o25", "o50", "o100"}
    evidence[cells[1].job_id]["valid_row_ranks"] = [999]
    with pytest.raises(ValueError, match="Selection mismatch"):
        COMPARISON.summarize_job_selections(evidence, cells, MODULE.protocol_group)


def test_reused_c_requires_all_six_successfully_validated_cells(tmp_path, monkeypatch):
    root = tmp_path / "completed_c"
    root.mkdir()
    (root / "launch_manifest.json").write_text(json.dumps({
        "environment": {"MUJOCO_GL": "osmesa"}, "checkpoint_path": "/c_epoch10.pt"}))
    seen = []
    def validate(job):
        seen.append(job)
        return {"success_count": 20}
    monkeypatch.setattr(MODULE, "validate_weighted_output", validate)
    evidence = MODULE.validate_reused_c(root)
    assert len(evidence) == 6
    assert {j.variant for j in seen} == {"c"}
    assert {MODULE.protocol_group(j) for j in seen} == {"o25", "o50", "o100"}
    (root / "launch_manifest.json").write_text(json.dumps({
        "environment": {"MUJOCO_GL": "egl"}, "checkpoint_path": "/c_epoch10.pt"}))
    with pytest.raises(ValueError, match="OSMesa"):
        MODULE.validate_reused_c(root)
