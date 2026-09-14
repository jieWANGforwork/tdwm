"""Exercise the formal evaluator's real SWM action queue with CPU fixtures.

The solver records scheduling, not physics or success rates. Existing public
CEM integration tests separately exercise Eff scores and EffPlan state search.
No dataset, checkpoint download, GPU, or installed-package edit is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stable_worldmodel as swm
import torch
import yaml
from gymnasium.spaces import Box
from torch import nn

from tdwm.evaluation import effplan as module
from tdwm.methods.eff import EffModel

CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment/effplan_cube.yaml"


@pytest.mark.parametrize("method,offset,offset_window", [
    (method, offset, False)
    for method in ["F-only", "Eff", "EffPlan"] for offset in [25, 50, 100]
] + [("EffPlan", 50, True), ("EffPlan", 100, True)])
@pytest.mark.parametrize("stop_after", [None, 7, "second_window"])
def test_formal_evaluator_executes_full_plan_before_replanning(
    monkeypatch, tmp_path, method, offset, offset_window, stop_after
):
    window = offset if offset_window else 25
    horizon = window // 5
    if stop_after == "second_window":
        stop_after = window + 7
    config = module.load_eff_protocol(CONFIG, stage="evaluation")
    reference = module.baseline_reference(config, CONFIG)
    checkpoint = tmp_path / "f.pt"
    checkpoint.write_bytes(b"f-fixture")
    config["source"]["lewm_checkpoint_sha256"] = module.sha256_file(checkpoint)
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text(yaml.safe_dump(config))
    eff_checkpoint = tmp_path / "eff.pt"
    eff_checkpoint.write_bytes(b"eff-fixture")
    planner_checkpoint = tmp_path / "planner.pt"
    planner_checkpoint.write_bytes(b"planner-fixture")
    pmeta = tmp_path / "planner_manifest.json"
    pmeta.write_text(json.dumps({
        "status": "complete",
        "completed_updates": 12800,
        "identity": {"source": {
            "eff_checkpoint_sha256": module.sha256_file(eff_checkpoint)
        }},
    }))
    pairs = module.held_out_episode_pairs(
        np.full(10000, 201),
        episode_ids=module.EpisodePartition.rp1_cube().evaluation,
        goal_offset=offset, count=50, seed=42,
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "format": "tdwm-eff-selection-v1", "goal_offset": offset,
        "planning_seed": 42, "episodes": 50,
        "dataset_source_sha256": config["source"]["dataset_source_sha256"],
        "pairs": pairs,
    }))

    world_model = nn.Linear(1, 1).requires_grad_(False)
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).requires_grad_(False).eval()
    planner = nn.Linear(1, 1).requires_grad_(False)
    monkeypatch.setenv("MUJOCO_GL", config["evaluation"]["render_backend"])
    monkeypatch.setattr(module, "baseline_reference", lambda *_: reference)
    monkeypatch.setattr(module, "prepare_cloud_runtime", lambda: {})
    monkeypatch.setattr(module, "_git_revision", lambda: "execution-fixture")
    monkeypatch.setattr(
        module, "_resolve_local_pretrained_lewm_export",
        lambda _: ("fixture", checkpoint, tmp_path),
    )
    monkeypatch.setattr(swm.wm, "load_pretrained", lambda *_, **__: world_model)
    monkeypatch.setattr(
        module, "_read_eff_for_evaluation",
        lambda *_, **__: (eff, {"global_step": 127960}),
    )
    monkeypatch.setattr(
        module, "load_effplan_planner",
        lambda *_, **__: (planner, {"settings": {"target_readout": True}}),
    )
    monkeypatch.setattr(
        module, "_resolve_frozen_dataset_source", lambda *_: {"format": "lance"}
    )
    monkeypatch.setattr(
        swm.data, "load_dataset",
        lambda *_, **__: SimpleNamespace(lengths=np.full(10000, 201)),
    )
    monkeypatch.setattr(
        module, "_load_action_processor",
        lambda *_: (SimpleNamespace(inverse_transform=lambda x: x), {}),
    )
    seen = {"calls": [], "constructors": []}

    class QueueProbeSolver:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            seen["constructors"].append(kwargs)

        def configure(self, *, action_space, n_envs, config):
            assert action_space.shape == (50, 5)
            self.n_envs = n_envs
            self.action_dim = 25
            self.horizon = config.horizon
            seen["plan"] = config

        def solve(self, info, init_action=None):
            tick = int(info["clock"][0])
            assert torch.all(info["clock"] == tick)
            # H == RH: no unexecuted tail is carried to the next decision.
            assert init_action is None
            seen["calls"].append(tick)
            actions = torch.arange(horizon*25, dtype=torch.float32).reshape(1, horizon, 25)
            return {"actions": actions.expand(len(info["clock"]), -1, -1) + tick * 1000}

        def __call__(self, *args, **kwargs):
            return self.solve(*args, **kwargs)

    monkeypatch.setattr(swm.solver, "CEMSolver", QueueProbeSolver)
    monkeypatch.setattr(module, "EffPlanSolver", QueueProbeSolver)

    class QueueWorld:
        def __init__(self, _name, **kwargs):
            self.num_envs = kwargs["num_envs"]
            assert self.num_envs == 50
            assert kwargs["max_episode_steps"] == 2 * offset
            self.action_space = Box(-1, 1, (50, 5), dtype=np.float32)
            self.single_action_space = Box(-1, 1, (5,), dtype=np.float32)

        def set_policy(self, policy):
            assert isinstance(policy, swm.policy.WorldModelPolicy)
            self.policy = policy
            policy.set_env(self)

        def evaluate(self, **kwargs):
            assert kwargs["goal_offset"] == offset
            assert kwargs["eval_budget"] == 2 * offset
            assert kwargs["episodes_idx"] == pairs["episode_indices"]
            assert kwargs["start_steps"] == pairs["start_steps"]
            seen["eval_budget"] = kwargs["eval_budget"]
            for tick in range(kwargs["eval_budget"]):
                dead = stop_after is not None and tick >= stop_after
                action = self.policy.get_action({
                    "clock": np.full(50, tick, dtype=np.int64),
                    "terminated": np.full(50, dead, dtype=bool),
                })
                if dead:
                    assert np.isnan(action).all()
                else:
                    # Every one of the 25 returned actions is consumed in order.
                    expected = (tick // window) * window * 1000 + (tick % window) * 5
                    np.testing.assert_array_equal(action[:, 0], np.full(50, expected))
            success = stop_after is not None
            return {
                "episode_successes": np.full(50, success),
                "success_rate": 100.0 if success else 0.0,
            }

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(swm, "World", QueueWorld)
    output = tmp_path / "evaluation"
    result = module.evaluate_effplan(
        config_path=config_path, dataset_path=tmp_path / "fixture.lance",
        lewm_checkpoint=checkpoint, selection_path=selection_path,
        output_dir=output, method=method, device="cpu",
        eff_checkpoint=eff_checkpoint, eff_manifest=tmp_path / "unused.json",
        planner_checkpoint=planner_checkpoint, planner_manifest=pmeta,
        offset_window=offset_window,
    )
    expected_calls = list(range(0, stop_after or 2*offset, window))
    assert seen["calls"] == expected_calls
    assert seen["plan"].horizon == seen["plan"].receding_horizon == horizon
    assert seen["plan"].action_block == 5
    assert seen["eval_budget"] == 2 * offset
    assert seen["closed"]
    paired = result["paired_protocol"]
    assert paired["receding_horizon"] == paired["horizon"] == horizon
    assert paired["action_block"] == 5
    assert paired["episode_budget"] == 2 * offset
    assert (paired["cem_candidates"], paired["cem_iterations"], paired["cem_elites"]) == (300, 30, 30)
    assert result["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    manifest = json.loads((output / "protocol_manifest.json").read_text())
    assert manifest["paired_protocol"] == paired
    assert len(result["episode_results"]) == 50
    constructor = seen["constructors"][0]
    if method == "EffPlan":
        if offset_window:
            from tdwm.adapters.effplan_adaptive import AdaptiveTrackingCost
            assert isinstance(constructor["model"], AdaptiveTrackingCost)
            assert manifest["protocol_overrides"]["offset_window"]["maximum_decisions"] == 2
            assert constructor["planning_horizon"] == horizon
        else:
            assert isinstance(constructor["model"], module.EffPlanTrackingCost)
        assert sum(constructor["search_iterations"]) == 30
    elif method == "Eff":
        assert isinstance(constructor["model"], module.EffCEMCost)
    else:
        assert constructor["model"] is world_model


@pytest.mark.parametrize("method", ["F-only", "Eff", "EffPlan"])
@pytest.mark.parametrize("offset", [50, 100])
def test_formal_entry_rejects_legacy_rh1_before_loading_artifacts(
    tmp_path, method, offset
):
    config = module.load_eff_protocol(CONFIG)
    config["evaluation"]["receding_horizons"][str(offset)] = 1
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="25 primitive"):
        module.evaluate_effplan(
            config_path=path, dataset_path=tmp_path / "missing.lance",
            lewm_checkpoint=tmp_path / "missing.pt",
            selection_path=tmp_path / "missing-selection.json",
            output_dir=output, method=method, device="cpu",
        )
    assert not output.exists()
