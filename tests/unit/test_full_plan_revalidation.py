from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.evaluation.full_plan_revalidation import (
    EXECUTION_PROTOCOL,
    configure_full_plan_revalidation,
    full_plan_revalidation_metadata,
    require_new_revalidation_output,
)

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ("v0", "v1", "v2", "v2_ema_sg")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
MODES = ("f_only", "f_plus_g", "f_plus_g_first", "g_only_f_rollout_mean")


def historical_protocol(version, variant, mode, offset=50):
    stem = f"actor_free_td_lewm_{version}_{variant}"
    module = importlib.import_module(f"tdwm.evaluation.{stem}")
    source = getattr(module, f"load_{stem}_evaluation_protocol")(
        ROOT / "configs" / "experiment" / f"{stem}_cube_checkpoint_o{offset}.yaml"
    )
    return getattr(module, f"configure_{stem}_evaluation_mode")(
        source,
        smoke=False,
        pilot=False,
        score_mode=mode,
        g_first_weight=0.25 if mode.startswith("f_plus_g_first") else None,
    )


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
def test_all_existing_five_block_modes_change_execution_only(version, variant, mode):
    original = historical_protocol(version, variant, mode)
    snapshot = deepcopy(original)
    changed = configure_full_plan_revalidation(original)
    assert original == snapshot
    assert changed["planning"]["receding_horizon"] == 5
    assert changed["planning"]["executed_environment_steps_before_replanning"] == 25
    for key, value in original.items():
        if key not in {"planning", "inference_objective"}:
            assert changed[key] == value
    for key, value in original["planning"].items():
        if key not in {
            "receding_horizon",
            "executed_environment_steps_before_replanning",
        }:
            assert changed["planning"][key] == value
    for key, value in original["inference_objective"].items():
        if key not in {"replanning", "executed_action_block", "score_definition"}:
            assert changed["inference_objective"][key] == value
    old_definition = original["inference_objective"].get("score_definition", {})
    new_definition = changed["inference_objective"].get("score_definition", {})
    for key, value in old_definition.items():
        if key not in {"cem_execution", "replanning", "executed_action_block"}:
            assert new_definition[key] == value
    metadata = full_plan_revalidation_metadata(changed)
    assert metadata["execution_protocol"] == EXECUTION_PROTOCOL
    assert metadata["executed_environment_steps_before_replanning"] == 25
    assert metadata["execution_revalidation"]["score_formula_unchanged"] is True
    assert full_plan_revalidation_metadata(original) == {}


@pytest.mark.parametrize("offset", (25, 50, 100))
@pytest.mark.parametrize("mode", (*MODES, "f_plus_g_first_q2"))
def test_v1_c_goal_offset_and_budget_are_not_changed(offset, mode):
    original = historical_protocol("v1", "c", mode, offset)
    changed = configure_full_plan_revalidation(original)
    assert changed["evaluation"] == original["evaluation"]
    assert changed["evaluation"]["goal_offset"] == offset
    assert changed["planning"]["episode_budget"] == offset * 2
    assert changed["planning"]["receding_horizon"] == 5


@pytest.mark.parametrize("version", VERSIONS)
def test_g_only_is_rejected_instead_of_being_silently_redefined(version):
    original = historical_protocol(version, "c", "g_only")
    with pytest.raises(ValueError, match="g_only"):
        configure_full_plan_revalidation(original)


@pytest.mark.parametrize(
    "key,value",
    (("horizon", 1), ("action_block", 1), ("frame_skip", 1), ("receding_horizon", 2)),
)
def test_unsupported_execution_contracts_fail_closed(key, value):
    original = historical_protocol("v1", "c", "f_only")
    original["planning"][key] = value
    with pytest.raises(ValueError):
        configure_full_plan_revalidation(original)


def test_metadata_cannot_claim_25_steps_for_a_five_step_policy():
    changed = configure_full_plan_revalidation(historical_protocol("v1", "c", "f_only"))
    changed["planning"]["receding_horizon"] = 1
    with pytest.raises(ValueError, match="contradicts"):
        full_plan_revalidation_metadata(changed)


def test_nonempty_output_is_preserved(tmp_path):
    old = tmp_path / "results.json"
    old.write_text('{"old_result":true}')
    with pytest.raises(FileExistsError, match="new, empty"):
        require_new_revalidation_output(tmp_path)
    assert old.read_text() == '{"old_result":true}'
    require_new_revalidation_output(tmp_path / "new_run")


@pytest.mark.parametrize("version", VERSIONS)
def test_rollout_mean_postprocessing_keeps_25_step_execution_metadata(version):
    changed = configure_full_plan_revalidation(
        historical_protocol(version, "c", "g_only_f_rollout_mean")
    )
    module_name = {
        "v0": "frozen_actor_free_td_v0_common",
        "v1": "frozen_actor_free_td_v1_common",
        "v2": "actor_free_td_lewm_v2_common",
        "v2_ema_sg": "actor_free_td_lewm_v2_common",
    }[version]
    module = importlib.import_module(f"tdwm.evaluation.{module_name}")
    metadata = module._rollout_mean_output_metadata(changed, changed["planning"])
    assert metadata["executed_action_block"] == "all_five_blocks"
    assert metadata["replanning"] == "every_five_action_blocks"
    assert (
        metadata["score_definition"]["formula"]
        == changed["inference_objective"]["score_definition"]["formula"]
    )


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "full_plan_revalidation_entrypoint",
        ROOT / "scripts" / "evaluate_actor_free_td_lewm_full_plan.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entrypoint_args(version="v1"):
    return [
        "--version",
        version,
        "--variant",
        "c",
        "--config",
        str(
            ROOT
            / "configs"
            / "experiment"
            / f"actor_free_td_lewm_{version}_c_cube_checkpoint_o50.yaml"
        ),
        "--score-mode",
        "f_plus_g_first",
        "--g-first-weight",
        "0.25",
    ]


@pytest.mark.parametrize("version", VERSIONS)
def test_cpu_dry_run_resolves_each_version_without_data_or_checkpoint(version, capsys):
    entrypoint = load_entrypoint()
    entrypoint.main([*entrypoint_args(version), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["protocol"]["planning"]["receding_horizon"] == 5
    assert result["protocol"]["inference_objective"]["g_first_weight"] == 0.25


def test_formal_entrypoint_does_not_start_without_a_gpu(tmp_path, monkeypatch):
    entrypoint = load_entrypoint()
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.write_bytes(b"not loaded without a GPU")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "new_output"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no evaluation was started"):
        entrypoint.main(
            [
                *entrypoint_args(),
                "--dataset",
                "not_loaded",
                "--checkpoint-path",
                str(checkpoint),
                "--checkpoint-sha256",
                digest,
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("version", VERSIONS)
def test_entrypoint_forwards_opt_in_to_existing_evaluator(
    version, tmp_path, monkeypatch, capsys
):
    entrypoint = load_entrypoint()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"unchanged checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    stem = f"actor_free_td_lewm_{version}_c"
    module = importlib.import_module(f"tdwm.evaluation.{stem}")
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"mock_evaluation": True}

    monkeypatch.setattr(module, f"evaluate_{stem}", evaluate)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    entrypoint.main(
        [
            *entrypoint_args(version),
            "--dataset",
            "unchanged_dataset",
            "--checkpoint-path",
            str(checkpoint),
            "--checkpoint-sha256",
            digest,
            "--output-dir",
            str(tmp_path / "new_output"),
        ]
    )
    assert len(calls) == 1
    assert calls[0]["full_plan_revalidation"] is True
    assert calls[0]["checkpoint_path"] == str(checkpoint)
    assert calls[0]["g_first_weight"] == 0.25
    assert json.loads(capsys.readouterr().out)["mock_evaluation"] is True


@pytest.mark.parametrize(
    "receding_horizon,expected_calls", ((1, [0, 5, 10, 15, 20, 25]), (5, [0, 25]))
)
def test_installed_world_model_policy_replans_from_latest_real_observation(
    receding_horizon, expected_calls
):
    import stable_worldmodel as swm
    from gymnasium.spaces import Box

    class RecordingCost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.observations = []

        def get_cost(self, info, candidates):
            self.observations.append(int(info["pixels"].flatten()[0]))
            # Imagined states from a cost evaluation must not become the next
            # real observation passed to the solver after execution.
            info["predicted_emb"] = torch.full((1, 4, 5, 192), 999.0)
            return candidates.square().sum(dim=(-1, -2))

    cost = RecordingCost()
    solver = swm.solver.CEMSolver(
        model=cost,
        batch_size=1,
        num_samples=4,
        n_steps=1,
        topk=2,
        device="cpu",
        seed=42,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=5, receding_horizon=receding_horizon, action_block=5
        ),
    )
    policy.set_env(
        SimpleNamespace(
            num_envs=1,
            single_action_space=Box(-1.0, 1.0, shape=(5,)),
            action_space=Box(-1.0, 1.0, shape=(1, 5)),
        )
    )
    for real_step in range(26):
        action = policy.get_action(
            {
                "pixels": np.full((1, 1, 1, 1, 3), real_step, dtype=np.uint8),
                "terminated": np.array([False]),
            }
        )
        assert action.shape == (1, 5)
    assert cost.observations == expected_calls
