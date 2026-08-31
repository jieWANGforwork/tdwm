from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v2_ema_sg_epoch_sweeps.py"
LEGACY_SCRIPT = ROOT / "scripts" / "run_actor_free_td_lewm_v2_epoch_sweeps.py"
SPEC = importlib.util.spec_from_file_location("v2_ema_sg_epoch_sweeps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SWEEPS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEPS
SPEC.loader.exec_module(SWEEPS)


def _paths(tmp_path: Path) -> object:
    bundle = tmp_path / "actor_free_td_lewm_v2_ema_sg_bundle"
    return SWEEPS.SweepPaths(
        repository=ROOT,
        dataset=tmp_path / "cube.lance",
        formal_root=bundle / "formal",
        bundle_root=bundle,
        sweep_root=bundle / "evaluation_sweeps",
        launcher_root=bundle / "evaluation_sweep_launcher",
    )


def _write_complete_output(cell: object, *, legacy: bool = False) -> None:
    checkpoint = Path(cell.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{cell.epoch}-{cell.variant}".encode())
    output = Path(cell.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    family = "actor_free_td_lewm_v2" if legacy else "actor_free_td_lewm_v2_ema_sg"
    method = f"{family}_{cell.variant}"
    (output / "results.json").write_text(
        json.dumps(
            {
                "method": method,
                "method_family": family,
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
                    "method": method,
                    "method_family": family,
                    "variant": cell.variant,
                    "epoch": cell.epoch,
                    "global_step": SWEEPS.STEPS_PER_EPOCH * cell.epoch,
                }
            }
        )
    )
    for name in ("episode_selection.json", "action_normalization.json"):
        (output / name).write_text("{}")


def test_scheduler_preserves_server_tested_safety_with_only_ema_sg_identity() -> None:
    expected = (
        LEGACY_SCRIPT.read_text()
        .replace("actor_free_td_lewm_v2", "actor_free_td_lewm_v2_ema_sg")
        .replace("intermediate V2 checkpoints", "intermediate V2-EMA-SG checkpoints")
    )
    assert ast.dump(ast.parse(SCRIPT.read_text())) == ast.dump(ast.parse(expected))


def test_sweep_plan_uses_ema_sg_checkpoints_configs_and_evaluators(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cells = SWEEPS.build_cells(paths=paths, python="/venv/bin/python")

    assert len(cells) == 126
    assert {cell.epoch for cell in cells} == set(range(3, 10))
    assert len({cell.output_dir for cell in cells}) == 126
    for cell in cells:
        method = f"actor_free_td_lewm_v2_ema_sg_{cell.variant}"
        assert method in cell.checkpoint
        assert cell.config_path == (
            f"configs/experiment/{method}_cube_checkpoint_o50.yaml"
        )
        assert cell.argv[1] == f"scripts/evaluate_{method}.py"
        assert Path(cell.output_dir) == (
            paths.sweep_root
            / f"epoch_{cell.epoch:02d}"
            / cell.variant
            / cell.score_mode
        )


def test_output_audit_accepts_ema_sg_and_rejects_legacy_v2(tmp_path: Path) -> None:
    cell = SWEEPS.build_cells(paths=_paths(tmp_path), python="python")[0]
    _write_complete_output(cell)
    assert set(SWEEPS.audit_complete_output(cell)) == {
        "results.json",
        "protocol_manifest.json",
        "episode_selection.json",
        "action_normalization.json",
    }

    _write_complete_output(cell, legacy=True)
    with pytest.raises(ValueError, match="results.method"):
        SWEEPS.audit_complete_output(cell)


def test_default_server_rendering_and_capacity_fixes_are_retained() -> None:
    args = SWEEPS.build_parser().parse_args(
        ["--bundle-root", "/bundle", "--dataset", "/data/cube.lance"]
    )
    assert args.mujoco_gl == "osmesa"
    assert args.pyopengl_platform == "osmesa"
    assert args.ld_preload == SWEEPS.DEFAULT_LD_PRELOAD
    assert args.gpus == [0, 1, 2, 3, 4]
    assert args.max_evals_per_gpu == 2
    assert args.stable_polls == 2
