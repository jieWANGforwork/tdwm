from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v2_ema_new_score_sweeps.py"
SPEC = importlib.util.spec_from_file_location("v2_ema_new_score_sweeps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SWEEPS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEPS
SPEC.loader.exec_module(SWEEPS)


def _paths(tmp_path: Path) -> object:
    bundle = tmp_path / "actor_free_td_lewm_v2_ema_bundle"
    return SWEEPS.SweepPaths(
        repository=ROOT,
        dataset=tmp_path / "cube.lance",
        formal_root=bundle / "formal",
        bundle_root=bundle,
        sweep_root=bundle / "new_score_evaluation_sweeps",
        launcher_root=bundle / "new_score_evaluation_sweep_launcher",
    )


def _argument(cell: object, option: str) -> str | None:
    argv = list(cell.argv)
    if option not in argv:
        return None
    return argv[argv.index(option) + 1]


def _write_training_manifest(paths: object, variant: str) -> Path:
    path = SWEEPS.training_manifest_path(paths.formal_root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": f"{SWEEPS.METHOD_FAMILY}_{variant}",
        "method_family": SWEEPS.METHOD_FAMILY,
        "variant": variant,
        "implementation_version": SWEEPS.IMPLEMENTATION_VERSION,
        "stage": "coupled_hybrid_ema_target_finetuning",
        "initialization": "corresponding_v1_deployment_finetune",
    }
    path.write_text(
        json.dumps(
            {
                "method": f"{SWEEPS.METHOD_FAMILY}_{variant}",
                "method_family": SWEEPS.METHOD_FAMILY,
                "variant": variant,
                "implementation_version": SWEEPS.IMPLEMENTATION_VERSION,
                "objective_version": 0,
                "deployment_checkpoint_version": 1,
                "stage": "coupled_hybrid_ema_target_finetuning",
                "initialization": "corresponding_v1_deployment_finetune",
                "seed": SWEEPS.SEED,
                "protocol": protocol,
                "protocol_sha256": SWEEPS._base.canonical_json_sha256(protocol),
                "runtime": {"tdwm_git_revision": SWEEPS.TRAINING_COMMIT},
            }
        )
    )
    return path


def _write_complete_output(cell: object, paths: object) -> None:
    training_manifest = _write_training_manifest(paths, cell.variant)
    checkpoint = Path(cell.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{cell.cell_id}".encode())
    checkpoint_sha = SWEEPS._base.file_sha256(checkpoint)
    training_sha = SWEEPS._base.file_sha256(training_manifest)
    method = f"{SWEEPS.METHOD_FAMILY}_{cell.variant}"
    evaluation_commit = SWEEPS.git_revision(ROOT)
    identity = {
        "version_key": SWEEPS.VERSION_KEY,
        "version_display_name": SWEEPS.VERSION_DISPLAY_NAME,
        "training_commit": SWEEPS.TRAINING_COMMIT,
        "method": method,
        "epoch": cell.epoch,
        "checkpoint_epoch": cell.epoch,
        "checkpoint_sha256": checkpoint_sha,
        "training_manifest_path": str(training_manifest.resolve()),
        "training_manifest_sha256": training_sha,
        "evaluation_commit": evaluation_commit,
    }
    inference: dict[str, object] = {"score_mode": cell.score_mode}
    score_metadata: dict[str, object]
    if cell.score_mode == SWEEPS.FIRST_ACTION_SCORE_MODE:
        weight = float(_argument(cell, "--g-first-weight"))
        inference.update(
            {
                "g_first_weight": weight,
                "score_definition": SWEEPS.FIRST_ACTION_SCORE_DEFINITION,
            }
        )
        score_metadata = {
            "g_first_weight": weight,
            "planning": {"horizon": 5},
            "score_definition": SWEEPS.FIRST_ACTION_SCORE_DEFINITION,
        }
    else:
        inference.update(
            {
                "f_goal_distance_used": False,
                "f_transition_used": True,
                "g_aggregation": "mean_over_5_blocks",
                "rollout_horizon": 5,
                "score_definition": SWEEPS.ROLLOUT_MEAN_SCORE_DEFINITION,
            }
        )
        score_metadata = {
            "g_aggregation": "mean_over_5_blocks",
            "state_source_for_q1": "current_online_encoder_state",
            "state_source_for_q2_to_q5": ("online_lewm_rollout_predicted_states"),
            "f_goal_distance_used": False,
            "f_transition_used": True,
            "planning_horizon": 5,
            "rollout_horizon": 5,
            "executed_action_block": "first_block_only",
            "replanning": "every_action_block",
            "score_definition": SWEEPS.ROLLOUT_MEAN_SCORE_DEFINITION,
        }
    output = Path(cell.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "method": method,
        "method_family": SWEEPS.METHOD_FAMILY,
        "variant": cell.variant,
        "score_mode": cell.score_mode,
        "smoke": False,
        "pilot": False,
        "metrics": {"episode_successes": [False] * 50},
        **identity,
        **score_metadata,
    }
    manifest = {
        "score_mode": cell.score_mode,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "method": method,
            "method_family": SWEEPS.METHOD_FAMILY,
            "variant": cell.variant,
            "implementation_version": SWEEPS.IMPLEMENTATION_VERSION,
            "epoch": cell.epoch,
            "global_step": SWEEPS.STEPS_PER_EPOCH * cell.epoch,
        },
        "protocol": {
            "planning": {"horizon": 5},
            "inference_objective": inference,
        },
        "runtime": {"tdwm_git_revision": evaluation_commit},
        **identity,
        **score_metadata,
    }
    (output / "results.json").write_text(json.dumps(results))
    (output / "protocol_manifest.json").write_text(json.dumps(manifest))
    for name in ("episode_selection.json", "action_normalization.json"):
        (output / name).write_text("{}")


def _accept_fixture_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SWEEPS,
        "EXPECTED_SELECTION_SHA256",
        hashlib.sha256(b"{}").hexdigest(),
    )


def test_plan_has_exact_96_cells_and_disjoint_score_paths(tmp_path: Path) -> None:
    assert SWEEPS.EXPECTED_SELECTION_SHA256 == (
        "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
    )
    paths = _paths(tmp_path)
    cells = SWEEPS.build_cells(
        paths=paths, python="/venv/bin/python", g_first_weight=0.25
    )

    assert len(cells) == 96
    assert {cell.epoch for cell in cells} == set(range(3, 11))
    assert {cell.variant for cell in cells} == set(SWEEPS.VARIANTS)
    assert {cell.score_mode for cell in cells} == set(SWEEPS.SCORE_MODES)
    assert len({cell.cell_id for cell in cells}) == 96
    assert len({cell.output_dir for cell in cells}) == 96
    assert len({cell.job_dir for cell in cells}) == 96
    for cell in cells:
        assert paths.sweep_root in Path(cell.output_dir).parents
        assert paths.launcher_root in Path(cell.job_dir).parents
        assert _argument(cell, "--training-manifest") == str(
            SWEEPS.training_manifest_path(paths.formal_root, cell.variant)
        )
        if cell.score_mode == SWEEPS.FIRST_ACTION_SCORE_MODE:
            assert cell.cell_id.startswith("v2_ema_")
            assert cell.cell_id.endswith("f_plus_g_first_alpha_0p25")
            assert Path(cell.output_dir).name == "alpha_0p25"
            assert _argument(cell, "--g-first-weight") == "0.25"
            assert cell.config_path.endswith("_cube_checkpoint_o50.yaml")
        else:
            assert cell.cell_id.startswith("v2_ema_")
            assert "alpha" not in cell.cell_id
            assert _argument(cell, "--g-first-weight") is None
            assert cell.config_path.endswith(
                "_cube_checkpoint_o50_g_only_f_rollout_mean.yaml"
            )


def test_epoch_10_is_owned_once_and_omits_intermediate_epoch_flag(
    tmp_path: Path,
) -> None:
    cells = SWEEPS.build_cells(paths=_paths(tmp_path), python="python")
    epoch_10 = [cell for cell in cells if cell.epoch == 10]
    assert len(epoch_10) == 12
    assert all("--checkpoint-epoch" not in cell.argv for cell in epoch_10)
    assert all("epoch_10.pt" in cell.checkpoint for cell in epoch_10)
    assert all(
        _argument(cell, "--checkpoint-epoch") == str(cell.epoch)
        for cell in cells
        if cell.epoch < 10
    )


def test_launcher_requires_explicit_first_action_weight() -> None:
    with pytest.raises(SystemExit):
        SWEEPS.build_parser().parse_args(
            ["--bundle-root", "/bundle", "--dataset", "/data/cube.lance"]
        )
    args = SWEEPS.build_parser().parse_args(
        [
            "--bundle-root",
            "/bundle",
            "--dataset",
            "/data/cube.lance",
            "--g-first-weight",
            "0.5",
        ]
    )
    assert args.g_first_weight == 0.5


def test_dry_run_plan_records_evaluation_and_selection_locks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        SWEEPS.main(
            [
                "--bundle-root",
                str(tmp_path / "bundle"),
                "--dataset",
                str(tmp_path / "cube.lance"),
                "--g-first-weight",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["cell_count"] == 96
    assert plan["expected_evaluation_commit"] == SWEEPS.git_revision(ROOT)
    assert plan["expected_selection_sha256"] == SWEEPS.EXPECTED_SELECTION_SHA256
    assert plan["evaluation_git_clean"] is SWEEPS.git_is_clean(ROOT)


def test_formal_launch_rejects_dirty_evaluation_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "cube.lance"
    dataset.write_text("fixture")
    monkeypatch.setattr(SWEEPS, "git_is_clean", lambda _repository: False)
    with pytest.raises(RuntimeError, match="clean git checkout"):
        SWEEPS.main(
            [
                "--bundle-root",
                str(tmp_path / "bundle"),
                "--dataset",
                str(dataset),
                "--g-first-weight",
                "1",
            ]
        )


def test_training_manifests_are_bound_to_exact_training_commit(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    for variant in SWEEPS.VARIANTS:
        _write_training_manifest(paths, variant)
    evidence = SWEEPS.audit_training_manifests(paths.formal_root)
    assert set(evidence) == set(SWEEPS.VARIANTS)
    assert all(
        item["training_commit"] == SWEEPS.TRAINING_COMMIT for item in evidence.values()
    )

    path = SWEEPS.training_manifest_path(paths.formal_root, "g2")
    manifest = json.loads(path.read_text())
    manifest["runtime"]["tdwm_git_revision"] = "0" * 40
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tdwm_git_revision"):
        SWEEPS.audit_training_manifests(paths.formal_root)


@pytest.mark.parametrize("score_mode", SWEEPS.SCORE_MODES)
def test_complete_outputs_are_reused_but_identity_mismatch_fails_closed(
    tmp_path: Path,
    score_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_fixture_selection(monkeypatch)
    paths = _paths(tmp_path)
    cell = next(
        item
        for item in SWEEPS.build_cells(paths=paths, python="python")
        if item.epoch == 3 and item.variant == "c" and item.score_mode == score_mode
    )
    _write_complete_output(cell, paths)
    assert set(SWEEPS.audit_complete_output(cell)) == set(
        SWEEPS._base.REQUIRED_OUTPUT_FILES
    )
    states, pending, active = SWEEPS.classify_existing_cells([cell], [0])
    assert states[cell.cell_id]["state"] == "REUSED"
    assert pending == []
    assert active == {}

    results_path = Path(cell.output_dir) / "results.json"
    results = json.loads(results_path.read_text())
    results["training_commit"] = "f" * 40
    results_path.write_text(json.dumps(results))
    with pytest.raises(ValueError, match="training_commit"):
        SWEEPS.audit_complete_output(cell)


def test_incomplete_existing_output_fails_closed_without_overwrite(
    tmp_path: Path,
) -> None:
    cell = SWEEPS.build_cells(paths=_paths(tmp_path), python="python")[0]
    output = Path(cell.output_dir)
    output.mkdir(parents=True)
    (output / "results.json").write_text("{}")
    states, pending, active = SWEEPS.classify_existing_cells([cell], [0])
    assert states[cell.cell_id]["state"] == "FAILED"
    assert "incomplete" in states[cell.cell_id]["error"].lower()
    assert pending == []
    assert active == {}


@pytest.mark.parametrize(
    ("filename", "field", "bad_value"),
    (
        ("results.json", "checkpoint_epoch", 4),
        ("protocol_manifest.json", "evaluation_commit", "f" * 40),
    ),
)
def test_output_reuse_rejects_checkpoint_epoch_or_evaluation_commit_mismatch(
    tmp_path: Path,
    filename: str,
    field: str,
    bad_value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_fixture_selection(monkeypatch)
    paths = _paths(tmp_path)
    cell = SWEEPS.build_cells(paths=paths, python="python")[0]
    _write_complete_output(cell, paths)
    path = Path(cell.output_dir) / filename
    document = json.loads(path.read_text())
    document[field] = bad_value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=field):
        SWEEPS.audit_complete_output(cell)


def test_output_reuse_rejects_wrong_formal_selection_hash(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    cell = SWEEPS.build_cells(paths=paths, python="python")[0]
    _write_complete_output(cell, paths)
    with pytest.raises(ValueError, match="episode_selection.json SHA-256"):
        SWEEPS.audit_complete_output(cell)


def test_scheduler_dispatches_ready_checkpoint_without_cross_method_barrier(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cells = SWEEPS.build_cells(paths=paths, python="python")
    selected = [
        cell for cell in cells if cell.epoch == 3 and cell.variant in {"c", "d"}
    ]
    assert len(selected) == 4
    ready_checkpoint = Path(
        next(cell.checkpoint for cell in selected if cell.variant == "c")
    )
    ready_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ready_checkpoint.write_bytes(b"stable-c-epoch-3")
    _write_training_manifest(paths, "c")
    assert not SWEEPS.training_manifest_path(paths.formal_root, "d").exists()

    class RunningProcess:
        next_pid = 20_000

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.pid = RunningProcess.next_pid
            RunningProcess.next_pid += 1

        def poll(self) -> None:
            return None

    code = SWEEPS.run_scheduler(
        paths=paths,
        cells=selected,
        gpu_indices=[0],
        max_evals_per_gpu=2,
        poll_seconds=0,
        stable_polls=2,
        strict_checkpoint_metadata=False,
        render_environment={},
        run_dir=paths.launcher_root / "runs" / "async-test",
        max_polls=2,
        popen=RunningProcess,
        sleeper=lambda _seconds: None,
    )
    assert code == 2
    state = json.loads(
        (paths.launcher_root / "runs" / "async-test" / "state.json").read_text()
    )
    assert state["source"] == "actor_free_td_lewm_v2_ema_new_score_sweep_scheduler"
    assert state["expected_selection_sha256"] == SWEEPS.EXPECTED_SELECTION_SHA256
    assert {cell_id: value["state"] for cell_id, value in state["cells"].items()} == {
        cell.cell_id: "RUNNING" if cell.variant == "c" else "WAITING_CHECKPOINT"
        for cell in selected
    }
    assert all(Path(cell.job_dir).is_dir() for cell in selected if cell.variant == "c")
    assert all(
        not Path(cell.job_dir).exists() for cell in selected if cell.variant == "d"
    )
