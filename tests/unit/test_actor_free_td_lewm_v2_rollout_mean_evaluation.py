from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tdwm.evaluation import actor_free_td_lewm_v2_common as v2_common
from tdwm.evaluation import actor_free_td_lewm_v2_ema_sg_common as ema_sg_common
from tdwm.evaluation import frozen_actor_free_td_v0_common as v0_common
from tdwm.evaluation import frozen_actor_free_td_v1_common as v1_common
from tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_cli import (
    run_actor_free_td_lewm_v2_ema_sg_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "experiment"
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
MODE = "g_only_f_rollout_mean"
TOP_LEVEL_METADATA = {
    "g_aggregation": "mean_over_5_blocks",
    "state_source_for_q1": "current_online_encoder_state",
    "state_source_for_q2_to_q5": "online_lewm_rollout_predicted_states",
    "f_goal_distance_used": False,
}


def _evaluation_module(version: str, variant: str) -> ModuleType:
    return importlib.import_module(
        f"tdwm.evaluation.actor_free_td_lewm_{version}_{variant}"
    )


def _base_config(version: str, variant: str) -> Path:
    return (
        CONFIG_ROOT / f"actor_free_td_lewm_{version}_{variant}_cube_checkpoint_o50.yaml"
    )


def _overlay(variant: str) -> Path:
    return (
        CONFIG_ROOT / f"actor_free_td_lewm_v2_{variant}_cube_checkpoint_o50_{MODE}.yaml"
    )


def _load_v2(variant: str, path: Path) -> dict[str, Any]:
    module = _evaluation_module("v2", variant)
    loader = getattr(
        module,
        f"load_actor_free_td_lewm_v2_{variant}_evaluation_protocol",
    )
    return loader(path)


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_v2_rollout_mean_overlay_is_minimal_loadable_and_horizon_five(
    variant: str,
) -> None:
    path = _overlay(variant)
    raw = path.read_text()
    protocol = _load_v2(variant, path)
    inference = protocol["inference_objective"]

    assert raw.startswith(
        f"extends: actor_free_td_lewm_v2_{variant}_cube_checkpoint_o50.yaml\n"
    )
    assert protocol["id"].endswith(f"checkpoint_o50_{MODE}")
    assert protocol["planning"]["horizon"] == 5
    assert inference["score_mode"] == MODE
    for key, expected in v2_common.ROLLOUT_MEAN_INFERENCE_FIELDS.items():
        assert inference[key] == expected
    assert inference["score_definition"] == (v2_common.ROLLOUT_MEAN_SCORE_DEFINITION)
    assert inference["f_score"] == "none"
    assert inference["f_score_reducer"] == "none"
    assert "g_first_weight" not in inference


def test_v2_configurer_can_add_mean_mode_to_old_protocol_and_cleanly_leave_it() -> None:
    module = _evaluation_module("v2", "c")
    base = _load_v2("c", _base_config("v2", "c"))
    configured = v2_common.configure_actor_free_td_v2_evaluation_mode(
        base,
        smoke=False,
        pilot=False,
        score_mode=MODE,
    )

    assert configured["planning"]["horizon"] == 5
    assert configured["inference_objective"]["score_definition"] == (
        v2_common.ROLLOUT_MEAN_SCORE_DEFINITION
    )
    for key, expected in v2_common.ROLLOUT_MEAN_INFERENCE_FIELDS.items():
        assert configured["inference_objective"][key] == expected

    overlay = _load_v2("c", _overlay("c"))
    legacy = v2_common.configure_actor_free_td_v2_evaluation_mode(
        overlay,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g",
    )
    legacy_inference = legacy["inference_objective"]
    assert legacy_inference["replanning"] == "every_action_block"
    assert legacy_inference["f_score"] == v2_common.LEGACY_F_SCORE
    assert legacy_inference["f_score_reducer"] == (v2_common.LEGACY_F_SCORE_REDUCER)
    assert legacy_inference["g_score"] == v2_common.LEGACY_G_SCORE
    assert "score_definition" not in legacy_inference
    for key in v2_common.ROLLOUT_MEAN_ONLY_INFERENCE_KEYS:
        assert key not in legacy_inference
    v2_common.validate_actor_free_td_v2_evaluation_protocol(
        legacy,
        spec=module.METHOD_SPEC,
    )

    first = v2_common.configure_actor_free_td_v2_evaluation_mode(
        overlay,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g_first",
        g_first_weight=0.5,
    )
    assert first["inference_objective"]["score_definition"] == (
        v2_common.FIRST_ACTION_SCORE_DEFINITION
    )
    assert first["inference_objective"]["g_first_weight"] == 0.5
    for key in v2_common.ROLLOUT_MEAN_ONLY_INFERENCE_KEYS:
        assert key not in first["inference_objective"]
    v2_common.validate_actor_free_td_v2_evaluation_protocol(
        first,
        spec=module.METHOD_SPEC,
    )


def test_rollout_mean_directory_is_isolated_and_legacy_names_are_unchanged() -> None:
    protocol = _load_v2("g3", _base_config("v2", "g3"))

    assert (
        v2_common.actor_free_td_v2_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode=MODE,
        )
        == f"actor_free_td_lewm_v2_g3_cube_o50_{MODE}_formal"
    )
    assert (
        v2_common.actor_free_td_v2_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="f_plus_g",
        )
        == "actor_free_td_lewm_v2_g3_cube_o50_f_plus_g_formal"
    )
    assert (
        v2_common.actor_free_td_v2_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="f_plus_g_first",
            g_first_weight=0.25,
        )
        == "actor_free_td_lewm_v2_g3_cube_o50_f_plus_g_first_alpha_0p25_formal"
    )


def test_rollout_mean_formal_protocol_rejects_wrong_horizon_or_metadata() -> None:
    module = _evaluation_module("v2", "c")
    protocol = _load_v2("c", _overlay("c"))

    wrong_horizon = deepcopy(protocol)
    wrong_horizon["planning"]["horizon"] = 1
    with pytest.raises(ValueError, match=r"g_only_f_rollout_mean.*horizon=5"):
        v2_common.validate_actor_free_td_v2_evaluation_protocol(
            wrong_horizon,
            spec=module.METHOD_SPEC,
        )

    wrong_flag = deepcopy(protocol)
    wrong_flag["inference_objective"]["f_goal_distance_used"] = True
    with pytest.raises(ValueError, match="f_goal_distance_used"):
        v2_common.validate_actor_free_td_v2_evaluation_protocol(
            wrong_flag,
            spec=module.METHOD_SPEC,
        )

    wrong_definition = deepcopy(protocol)
    wrong_definition["inference_objective"]["score_definition"][
        "state_source_for_q1"
    ] = "wrong"
    with pytest.raises(ValueError, match="score_definition"):
        v2_common.validate_actor_free_td_v2_evaluation_protocol(
            wrong_definition,
            spec=module.METHOD_SPEC,
        )


def _load_cli(variant: str) -> ModuleType:
    path = ROOT / "scripts" / f"evaluate_actor_free_td_lewm_v2_{variant}.py"
    spec = importlib.util.spec_from_file_location(
        f"_test_rollout_mean_cli_{variant}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_six_v2_cli_entrypoints_forward_rollout_mean_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variant: str,
) -> None:
    module = _load_cli(variant)
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> dict[str, bool]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        module,
        f"evaluate_actor_free_td_lewm_v2_{variant}",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--config",
            str(_overlay(variant)),
            "--dataset",
            "dataset.lance",
            "--checkpoint-path",
            "checkpoint.pt",
            "--output-dir",
            "results",
            "--score-mode",
            MODE,
        ],
    )

    module.main()

    assert captured["score_mode"] == MODE
    assert captured["g_first_weight"] is None
    assert captured["output_dir"] == "results"
    assert '"ok": true' in capsys.readouterr().out


def test_v2_runtime_records_exact_rollout_mean_fields_in_manifest_and_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _evaluation_module("v2", "c")

    def fake_runtime(**kwargs: Any) -> dict[str, Any]:
        formal = kwargs["protocol_loader"](
            kwargs["protocol_path"],
            spec=kwargs["spec"],
        )
        protocol = kwargs["protocol_configurer"](
            formal,
            smoke=False,
            pilot=False,
            score_mode=kwargs.get("score_mode"),
            g_first_weight=kwargs.get("g_first_weight"),
        )
        manifest = {
            "score_mode": protocol["inference_objective"]["score_mode"],
            "protocol": protocol,
        }
        result = {
            "score_mode": protocol["inference_objective"]["score_mode"],
            "planning_horizon": protocol["planning"]["horizon"],
        }
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "protocol_manifest.json").write_text(json.dumps(manifest))
        (output / "results.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(
        v2_common,
        "evaluate_actor_free_td_predictor_runtime",
        fake_runtime,
    )
    result = v2_common.evaluate_actor_free_td_v2(
        spec=module.METHOD_SPEC,
        checkpoint_loader=object(),
        policy_factory=object(),
        protocol_path=_overlay("c"),
        dataset_path="unused.lance",
        output_dir=tmp_path,
        checkpoint_path="unused.pt",
        score_mode=MODE,
    )
    manifest = json.loads((tmp_path / "protocol_manifest.json").read_text())
    recorded_result = json.loads((tmp_path / "results.json").read_text())

    for payload in (result, manifest, recorded_result):
        for key, expected in TOP_LEVEL_METADATA.items():
            assert payload[key] == expected
        assert payload["planning_horizon"] == 5
    inference = manifest["protocol"]["inference_objective"]
    for key, expected in TOP_LEVEL_METADATA.items():
        assert inference[key] == expected
    assert inference["f_transition_used"] is True
    assert inference["g_score"] == v2_common.ROLLOUT_MEAN_G_SCORE
    assert inference["rollout_horizon"] == 5
    assert inference["executed_action_block"] == "first_block_only"
    assert inference["replanning"] == "every_action_block"


def test_v0_v1_reject_but_v2_ema_supports_rollout_mean_mode() -> None:
    v0_module = _evaluation_module("v0", "c")
    v0_protocol = getattr(
        v0_module,
        "load_actor_free_td_lewm_v0_c_evaluation_protocol",
    )(_base_config("v0", "c"))
    with pytest.raises(ValueError, match="incompatible with V0"):
        v0_common.configure_frozen_actor_free_td_v0_evaluation_mode(
            v0_protocol,
            smoke=False,
            pilot=False,
            score_mode=MODE,
        )

    v1_module = _evaluation_module("v1", "c")
    v1_protocol = getattr(
        v1_module,
        "load_actor_free_td_lewm_v1_c_evaluation_protocol",
    )(_base_config("v1", "c"))
    with pytest.raises(ValueError, match="incompatible with V1"):
        v1_common.configure_frozen_actor_free_td_v1_evaluation_mode(
            v1_protocol,
            smoke=False,
            pilot=False,
            score_mode=MODE,
        )

    ema_module = importlib.import_module(
        "tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_c"
    )
    ema_protocol = ema_module.load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        _base_config("v2_ema_sg", "c")
    )
    configured = ema_sg_common.configure_actor_free_td_v2_ema_sg_evaluation_mode(
        ema_protocol,
        smoke=False,
        pilot=False,
        score_mode=MODE,
    )
    assert ema_sg_common.validate_v2_score_mode(MODE) == MODE
    assert configured["planning"]["horizon"] == 5
    assert configured["inference_objective"]["f_goal_distance_used"] is False
    assert configured["inference_objective"]["g_aggregation"] == "mean_over_5_blocks"


def test_v2_ema_cli_offers_and_forwards_rollout_mean_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def evaluate(**kwargs: Any) -> dict[str, bool]:
        captured.update(kwargs)
        return {"ok": True}

    fake_module = type(
        "FakeModule",
        (),
        {
            "load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol": staticmethod(
                lambda _path: {}
            ),
            "evaluate_actor_free_td_lewm_v2_ema_sg_c": staticmethod(evaluate),
        },
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_actor_free_td_lewm_v2_ema_sg_c.py",
            "--config",
            "protocol.yaml",
            "--dataset",
            "dataset.lance",
            "--checkpoint-path",
            "checkpoint.pt",
            "--output-dir",
            "results",
            "--score-mode",
            MODE,
            "--training-manifest",
            "training_manifest.json",
        ],
    )

    run_actor_free_td_lewm_v2_ema_sg_evaluation("c")

    assert captured["score_mode"] == MODE
    assert captured["g_first_weight"] is None
    assert captured["training_manifest_path"] == "training_manifest.json"
    assert captured["output_dir"] == "results"
    assert '"ok": true' in capsys.readouterr().out
