from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v2_epoch_sweeps.py"
SPEC = importlib.util.spec_from_file_location("v2_epoch_sweeps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SWEEPS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEPS
SPEC.loader.exec_module(SWEEPS)


def _paths(tmp_path: Path) -> object:
    bundle = tmp_path / "bundle"
    return SWEEPS.SweepPaths(
        repository=ROOT,
        dataset=tmp_path / "cube.lance",
        formal_root=bundle / "formal",
        bundle_root=bundle,
        sweep_root=bundle / "evaluation_sweeps",
        launcher_root=bundle / "evaluation_sweep_launcher",
    )


def _write_complete_output(cell: object) -> None:
    checkpoint = Path(cell.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(f"checkpoint-{cell.epoch}-{cell.variant}".encode())
    output = Path(cell.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(
            {
                "method": f"actor_free_td_lewm_v2_{cell.variant}",
                "method_family": "actor_free_td_lewm_v2",
                "variant": cell.variant,
                "score_mode": cell.score_mode,
                "smoke": False,
                "pilot": False,
                "metrics": {"episode_successes": [False] * 50},
            }
        )
    )
    (output / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": SWEEPS.file_sha256(checkpoint),
                    "method": f"actor_free_td_lewm_v2_{cell.variant}",
                    "method_family": "actor_free_td_lewm_v2",
                    "variant": cell.variant,
                    "epoch": cell.epoch,
                    "global_step": SWEEPS.STEPS_PER_EPOCH * cell.epoch,
                }
            }
        )
    )
    for name in ("episode_selection.json", "action_normalization.json"):
        (output / name).write_text("{}")


def test_plan_is_exactly_seven_by_six_by_three_and_excludes_epoch_ten(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cells = SWEEPS.build_cells(paths=paths, python="/venv/bin/python")

    assert len(cells) == 126
    assert {cell.epoch for cell in cells} == set(range(3, 10))
    assert len({cell.cell_id for cell in cells}) == 126
    assert len({cell.output_dir for cell in cells}) == 126
    assert {(cell.variant, cell.score_mode) for cell in cells} == {
        (variant, mode) for variant in SWEEPS.VARIANTS for mode in SWEEPS.SCORE_MODES
    }
    for cell in cells:
        assert Path(cell.output_dir) == (
            paths.bundle_root
            / "evaluation_sweeps"
            / f"epoch_{cell.epoch:02d}"
            / cell.variant
            / cell.score_mode
        )
        assert cell.argv[cell.argv.index("--checkpoint-epoch") + 1] == str(cell.epoch)
        assert "--smoke" not in cell.argv and "--pilot" not in cell.argv


def test_checkpoint_requires_two_unchanged_size_mtime_polls_and_strict_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch_03.pt"
    checkpoint.write_bytes(b"stable")

    first = SWEEPS.observe_checkpoint(checkpoint, None)
    second = SWEEPS.observe_checkpoint(checkpoint, first)
    assert first.matching_polls == 1
    assert second.matching_polls == 2
    audit = SWEEPS.audit_stable_checkpoint(
        checkpoint,
        epoch=3,
        strict_metadata=True,
        metadata_loader=lambda _: {"epoch": 3, "global_step": 38_388},
    )
    assert audit["sha256"] == SWEEPS.file_sha256(checkpoint)
    assert audit["metadata"] == {"epoch": 3, "global_step": 38_388}

    with pytest.raises(ValueError, match="global_step"):
        SWEEPS.audit_stable_checkpoint(
            checkpoint,
            epoch=3,
            strict_metadata=True,
            metadata_loader=lambda _: {"epoch": 3, "global_step": 1},
        )


def test_complete_output_is_reused_but_unowned_partial_output_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cells = SWEEPS.build_cells(paths=paths, python="python")[:2]
    _write_complete_output(cells[0])
    partial = Path(cells[1].output_dir)
    partial.mkdir(parents=True)
    (partial / "results.json").write_text("{}")

    states, pending, active = SWEEPS.classify_existing_cells(cells, [0, 1, 2, 3, 4])

    assert states[cells[0].cell_id]["state"] == "REUSED"
    assert states[cells[1].cell_id]["state"] == "FAILED"
    assert "will not be overwritten" in states[cells[1].cell_id]["error"]
    assert pending == [] and active == {}


def test_ten_manual_epoch_three_cells_are_adopted_without_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    epoch_three = [
        cell
        for cell in SWEEPS.build_cells(paths=paths, python="/venv/bin/python")
        if cell.epoch == 3
        and cell.variant in {"g1", "g2", "g3", "d", "f"}
        and cell.score_mode in {"f_only", "g_only"}
    ]
    assert len(epoch_three) == 10
    for index, cell in enumerate(epoch_three):
        job_dir = Path(cell.job_dir)
        job_dir.mkdir(parents=True)
        (job_dir / "wrapper_pid.txt").write_text(f"{265_000 + index}\n")
        SWEEPS.atomic_write_json(
            job_dir / "start_evidence.json",
            {
                "wrapper_pid": 265_000 + index,
                "gpu": index // 2,
                "argv": list(cell.argv),
                "argv_sha256": SWEEPS.canonical_json_sha256(list(cell.argv)),
            },
        )
        output = Path(cell.output_dir)
        output.mkdir(parents=True)
        (output / "partial.log").write_text("running")
    monkeypatch.setattr(SWEEPS, "process_alive", lambda pid: True)

    states, pending, active = SWEEPS.classify_existing_cells(
        epoch_three, [0, 1, 2, 3, 4]
    )

    assert pending == []
    assert len(active) == 10
    assert all(state["state"] == "ADOPTED_RUNNING" for state in states.values())
    assert {running.gpu for running in active.values()} == {0, 1, 2, 3, 4}


def test_dead_manual_job_without_exit_marker_is_failed_not_relaunched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = SWEEPS.build_cells(paths=_paths(tmp_path), python="python")[0]
    job_dir = Path(cell.job_dir)
    job_dir.mkdir(parents=True)
    (job_dir / "wrapper_pid.txt").write_text("123\n")
    monkeypatch.setattr(SWEEPS, "process_alive", lambda pid: False)

    states, pending, active = SWEEPS.classify_existing_cells([cell], [0])

    assert states[cell.cell_id]["state"] == "FAILED"
    assert pending == [] and active == {}
    assert not (job_dir / "start_evidence.json").exists()


class _ImmediateProcess:
    next_pid = 40_000

    def __init__(self, return_code: int) -> None:
        self.pid = type(self).next_pid
        type(self).next_pid += 1
        self.return_code = return_code

    def poll(self) -> int:
        return self.return_code


def test_per_cell_failure_does_not_block_independent_cells_and_capacity_is_two(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    all_cells = SWEEPS.build_cells(paths=paths, python="python")
    cells = all_cells[:3]
    checkpoint = Path(cells[0].checkpoint)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    launches: list[tuple[str, int]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _ImmediateProcess:
        environment = kwargs["env"]
        child_index = command.index("_child")
        assert command[child_index + 3] == "--"
        expected_argv = command[command.index("--") + 1 :]
        cell = next(
            candidate for candidate in cells if list(candidate.argv) == expected_argv
        )
        return_code = 9 if len(launches) == 0 else 0
        launches.append((cell.cell_id, int(environment["CUDA_VISIBLE_DEVICES"])))
        job_dir = Path(cell.job_dir)
        SWEEPS.atomic_write_text(job_dir / "exit_code.txt", f"{return_code}\n")
        SWEEPS.atomic_write_text(job_dir / "child_pid.txt", "555\n")
        if return_code == 0:
            _write_complete_output(cell)
        return _ImmediateProcess(return_code)

    code = SWEEPS.run_scheduler(
        paths=paths,
        cells=cells,
        gpu_indices=[0],
        max_evals_per_gpu=2,
        poll_seconds=0,
        stable_polls=2,
        strict_checkpoint_metadata=True,
        render_environment={
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LD_PRELOAD": SWEEPS.DEFAULT_LD_PRELOAD,
        },
        run_dir=tmp_path / "run",
        metadata_loader=lambda _: {"epoch": 3, "global_step": 38_388},
        popen=fake_popen,
        sleeper=lambda _: None,
    )

    assert code == 1
    assert [cell_id for cell_id, _ in launches] == [cell.cell_id for cell in cells]
    assert [gpu for _, gpu in launches] == [0, 0, 0]
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["cells"][cells[0].cell_id]["state"] == "FAILED"
    assert all(
        state["cells"][cell.cell_id]["state"] == "SUCCEEDED" for cell in cells[1:]
    )
    for cell in cells:
        start = json.loads((Path(cell.job_dir) / "start_evidence.json").read_text())
        assert start["environment"] == {
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_PRELOAD": SWEEPS.DEFAULT_LD_PRELOAD,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        }


def test_default_cli_uses_server_osmesa_environment() -> None:
    args = SWEEPS.build_parser().parse_args(
        ["--bundle-root", "/bundle", "--dataset", "/data/cube.lance"]
    )
    assert args.mujoco_gl == "osmesa"
    assert args.pyopengl_platform == "osmesa"
    assert args.ld_preload == "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    assert args.gpus == [0, 1, 2, 3, 4]
    assert args.max_evals_per_gpu == 2


def test_child_wrapper_writes_atomic_exit_marker(tmp_path: Path) -> None:
    exit_path = tmp_path / "exit_code.txt"
    pid_path = tmp_path / "child_pid.txt"
    code = SWEEPS._child_main(
        [
            str(exit_path),
            str(pid_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(4)",
        ]
    )
    assert code == 4
    assert exit_path.read_text() == "4\n"
    assert int(pid_path.read_text()) > 0
    assert not list(tmp_path.glob("*.tmp"))
