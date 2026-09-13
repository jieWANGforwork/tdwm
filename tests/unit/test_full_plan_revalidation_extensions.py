from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.evaluation.full_plan_revalidation import (
    EXECUTION_PROTOCOL,
    FULL_PLAN_EXECUTION,
    configure_full_plan_revalidation,
    full_plan_revalidation_metadata,
)
from tdwm.evaluation.lewm_checkpoint import load_protocol

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiment"
FIVE_BLOCK_MODES = (
    "f_only",
    "f_plus_g",
    "f_plus_g_first",
    "f_plus_g_first_q2",
    "g_only_f_rollout_mean",
)
STATE_V_MODES = (
    "state_v_terminal",
    "state_v_plus_first_q",
    "state_v_plus_first_q2",
)
EXTENSION_CASES = [
    (variant, offset, mode)
    for variant, offsets, modes in (
        ("c2", (50,), FIVE_BLOCK_MODES),
        ("c3", (25, 50, 100), STATE_V_MODES),
        ("c4", (25, 50, 100), FIVE_BLOCK_MODES),
    )
    for offset in offsets
    for mode in modes
]


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "full_plan_extensions_entrypoint",
        ROOT / "scripts" / "evaluate_actor_free_td_lewm_full_plan.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extension_protocol(variant, offset, mode):
    stem = f"actor_free_td_lewm_v1_{variant}"
    module = importlib.import_module(f"tdwm.evaluation.{stem}")
    source = getattr(module, f"load_{stem}_evaluation_protocol")(
        CONFIGS / f"{stem}_cube_checkpoint_o{offset}.yaml"
    )
    return getattr(module, f"configure_{stem}_evaluation_mode")(
        source,
        smoke=False,
        pilot=False,
        score_mode=mode,
        g_first_weight=0.25 if "first" in mode else None,
    )


@pytest.mark.parametrize("variant,offset,mode", EXTENSION_CASES)
def test_extensions_only_change_execution_not_method(variant, offset, mode):
    original = extension_protocol(variant, offset, mode)
    snapshot = deepcopy(original)
    changed = configure_full_plan_revalidation(original)
    assert original == snapshot
    assert changed["evaluation"] == original["evaluation"]
    assert changed["planning"]["episode_budget"] == 2 * offset
    assert changed["planning"]["horizon"] == 5
    assert changed["planning"]["action_block"] == 5
    assert changed["planning"]["receding_horizon"] == 5
    for key in original:
        if key not in {"planning", "inference_objective"}:
            assert changed[key] == original[key]
    for key, value in original["planning"].items():
        if key not in {
            "receding_horizon",
            "executed_environment_steps_before_replanning",
        }:
            assert changed["planning"][key] == value
    for level in ("inference_objective", "score_definition"):
        before = original["inference_objective"]
        after = changed["inference_objective"]
        if level == "score_definition":
            before, after = before.get(level, {}), after.get(level, {})
        for key, value in before.items():
            if key in FULL_PLAN_EXECUTION:
                assert after[key] == FULL_PLAN_EXECUTION[key]
            elif key != "score_definition":
                assert after[key] == value
    assert (
        full_plan_revalidation_metadata(changed)["execution_protocol"]
        == EXECUTION_PROTOCOL
    )


@pytest.mark.parametrize("offset", (25, 50, 100))
def test_lewm_baseline_keeps_model_data_and_search_settings(offset):
    filename = (
        "lewm_cube_seed3072_o100_full_plan.yaml"
        if offset == 100
        else f"lewm_cube_seed3072_o{offset}.yaml"
    )
    original = load_protocol(CONFIGS / filename)
    snapshot = deepcopy(original)
    changed = configure_full_plan_revalidation(original)
    assert original == snapshot
    assert "inference_objective" not in changed
    assert changed["evaluation"]["goal_offset"] == offset
    assert changed["planning"]["episode_budget"] == 2 * offset
    assert changed["planning"]["receding_horizon"] == 5
    for key, value in original.items():
        if key != "planning":
            assert changed[key] == value
    assert (
        full_plan_revalidation_metadata(changed)[
            "executed_environment_steps_before_replanning"
        ]
        == 25
    )


def test_new_o100_baseline_only_extends_offset_budget_and_execution():
    original = load_protocol(CONFIGS / "lewm_cube_seed3072_o50.yaml")
    extended = load_protocol(CONFIGS / "lewm_cube_seed3072_o100_full_plan.yaml")
    for key, value in original.items():
        if key not in {"id", "evaluation", "planning", "provenance"}:
            assert extended[key] == value
    for key, value in original["planning"].items():
        if key not in {
            "receding_horizon",
            "executed_environment_steps_before_replanning",
            "episode_budget",
        }:
            assert extended["planning"][key] == value
    for key, value in original["evaluation"].items():
        if key != "goal_offset":
            assert extended["evaluation"][key] == value


@pytest.mark.parametrize(
    "variant,mode",
    (
        ("c2", "f_plus_g_first_q2"),
        ("c3", "state_v_terminal"),
        ("c3", "state_v_plus_first_q2"),
        ("c4", "f_plus_g"),
        (None, "f_only"),
    ),
)
def test_independent_entrypoint_dry_run_supports_extensions(variant, mode, capsys):
    version = "v1" if variant else "lewm"
    filename = (
        f"actor_free_td_lewm_v1_{variant}_cube_checkpoint_o50.yaml"
        if variant
        else "lewm_cube_seed3072_o50.yaml"
    )
    args = [
        "--version",
        version,
        "--config",
        str(CONFIGS / filename),
        "--score-mode",
        mode,
        "--dry-run",
    ]
    if variant:
        args += ["--variant", variant]
    if "first" in mode:
        args += ["--g-first-weight", "0.25"]
    load_entrypoint().main(args)
    result = json.loads(capsys.readouterr().out)
    assert result["protocol"]["planning"]["receding_horizon"] == 5
    assert result["protocol"]["planning"]["episode_budget"] == 100


@pytest.mark.parametrize(
    "variant,mode",
    (
        ("c2", "f_plus_g_first_q2"),
        ("c3", "state_v_terminal"),
        ("c4", "f_plus_g"),
        (None, "f_only"),
    ),
)
def test_entrypoint_forwards_opt_in_without_changing_old_evaluator_defaults(
    variant,
    mode,
    monkeypatch,
    tmp_path,
    capsys,
):
    import torch

    if variant:
        stem = f"actor_free_td_lewm_v1_{variant}"
        module = importlib.import_module(f"tdwm.evaluation.{stem}")
        target = f"evaluate_{stem}"
        filename = f"{stem}_cube_checkpoint_o50.yaml"
    else:
        module = importlib.import_module("tdwm.evaluation.lewm_checkpoint")
        target = "evaluate_official_lewm"
        filename = "lewm_cube_seed3072_o50.yaml"
    if variant in (None, "c3"):
        assert (
            inspect.signature(getattr(module, target))
            .parameters["full_plan_revalidation"]
            .default
            is False
        )
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"test_only": True}

    monkeypatch.setattr(module, target, evaluate)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    checkpoint = tmp_path / "historical_object.ckpt"
    checkpoint.write_bytes(b"unchanged historical checkpoint")
    args = [
        "--version",
        "v1" if variant else "lewm",
        "--score-mode",
        mode,
        "--config",
        str(CONFIGS / filename),
        "--dataset",
        "unchanged_dataset",
        "--checkpoint-path",
        str(checkpoint),
        "--checkpoint-sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "--output-dir",
        str(tmp_path / "new_output"),
    ]
    if variant:
        args += ["--variant", variant]
    if "first" in mode:
        args += ["--g-first-weight", "0.25"]
    load_entrypoint().main(args)
    assert len(calls) == 1
    assert calls[0]["full_plan_revalidation"] is True
    assert calls[0]["checkpoint_path"] == str(checkpoint)
    assert calls[0]["dataset_path"] == "unchanged_dataset"
    if variant:
        assert calls[0]["score_mode"] == mode
    else:
        assert "score_mode" not in calls[0]
        assert "g_first_weight" not in calls[0]
    assert json.loads(capsys.readouterr().out)["test_only"] is True


@pytest.mark.parametrize(
    "version,variant,mode",
    (
        ("v0", "c3", "state_v_terminal"),
        ("lewm", "c", "f_only"),
        ("lewm", None, "f_plus_g"),
    ),
)
def test_invalid_method_combinations_are_not_silently_changed(version, variant, mode):
    args = [
        "--version",
        version,
        "--config",
        "unused.yaml",
        "--score-mode",
        mode,
        "--dry-run",
    ]
    if variant:
        args += ["--variant", variant]
    with pytest.raises(SystemExit):
        load_entrypoint().parse_args(args)


@pytest.mark.parametrize("method", ("c3", "lewm"))
def test_new_runtime_option_refuses_historical_output_before_loading_data(
    method, tmp_path
):
    output = tmp_path / "historical"
    output.mkdir()
    result_path = output / "results.json"
    result_path.write_text('{"historical":true}')
    if method == "c3":
        module = importlib.import_module("tdwm.evaluation.actor_free_td_lewm_v1_c3")
        evaluate = module.evaluate_actor_free_td_lewm_v1_c3
        config = CONFIGS / "actor_free_td_lewm_v1_c3_cube_checkpoint_o50.yaml"
    else:
        module = importlib.import_module("tdwm.evaluation.lewm_checkpoint")
        evaluate = module.evaluate_official_lewm
        config = CONFIGS / "lewm_cube_seed3072_o50.yaml"
    with pytest.raises(FileExistsError, match="new, empty"):
        evaluate(
            protocol_path=config,
            dataset_path="not_loaded",
            checkpoint_path="not_loaded",
            output_dir=output,
            full_plan_revalidation=True,
        )
    assert result_path.read_text() == '{"historical":true}'
