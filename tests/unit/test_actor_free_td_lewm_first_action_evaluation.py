from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tdwm.evaluation import actor_free_td_lewm_v2_common as v2_common
from tdwm.evaluation import actor_free_td_lewm_v2_ema_sg_common as ema_sg_common
from tdwm.evaluation import frozen_actor_free_td_v0_common as v0_common
from tdwm.evaluation import frozen_actor_free_td_v1_common as v1_common

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "experiment"
VERSIONS = ("v0", "v1", "v2")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def _version_case(version: str) -> dict[str, Any]:
    evaluation = importlib.import_module(
        f"tdwm.evaluation.actor_free_td_lewm_{version}_c"
    )
    if version == "v0":
        common = v0_common
        configure = common.configure_frozen_actor_free_td_v0_evaluation_mode
        output_name = common.actor_free_td_v0_output_directory_name
        metadata = common._first_action_output_metadata
    elif version == "v1":
        common = v1_common
        configure = common.configure_frozen_actor_free_td_v1_evaluation_mode
        output_name = common.actor_free_td_v1_output_directory_name
        metadata = common._first_action_output_metadata
    else:
        common = v2_common
        configure = common.configure_actor_free_td_v2_evaluation_mode
        output_name = common.actor_free_td_v2_output_directory_name
        # The V2 runtime intentionally reuses V1's generic evaluator.
        metadata = v1_common._first_action_output_metadata
    loader = getattr(
        evaluation,
        f"load_actor_free_td_lewm_{version}_c_evaluation_protocol",
    )
    protocol = loader(
        CONFIG_ROOT / f"actor_free_td_lewm_{version}_c_cube_checkpoint_o50.yaml"
    )
    return {
        "common": common,
        "configure": configure,
        "output_name": output_name,
        "metadata": metadata,
        "protocol": protocol,
    }


@pytest.mark.parametrize("version", VERSIONS)
def test_first_action_mode_configures_horizon_alpha_and_definition(
    version: str,
) -> None:
    case = _version_case(version)
    configured = case["configure"](
        case["protocol"],
        smoke=False,
        pilot=False,
        score_mode="f_plus_g_first",
        g_first_weight=0.25,
    )

    inference = configured["inference_objective"]
    assert configured["planning"]["horizon"] == 5
    assert inference["score_mode"] == "f_plus_g_first"
    assert inference["g_first_weight"] == 0.25
    assert inference["score_definition"] == case["common"].FIRST_ACTION_SCORE_DEFINITION
    assert (
        inference["score_definition"]
        is not case["common"].FIRST_ACTION_SCORE_DEFINITION
    )


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("invalid_weight", [None, -0.1, math.nan, math.inf, True])
def test_first_action_mode_rejects_missing_or_invalid_alpha(
    version: str,
    invalid_weight: float | bool | None,
) -> None:
    case = _version_case(version)

    with pytest.raises(ValueError, match="g_first_weight"):
        case["configure"](
            case["protocol"],
            smoke=False,
            pilot=False,
            score_mode="f_plus_g_first",
            g_first_weight=invalid_weight,
        )


@pytest.mark.parametrize("version", VERSIONS)
def test_old_mode_metadata_and_output_name_remain_unchanged(version: str) -> None:
    case = _version_case(version)
    configured = case["configure"](
        case["protocol"],
        smoke=False,
        pilot=False,
        score_mode="f_plus_g",
    )

    assert configured["planning"]["horizon"] == 5
    assert "g_first_weight" not in configured["inference_objective"]
    assert "score_definition" not in configured["inference_objective"]
    assert case["metadata"](configured, configured["planning"]) == {}
    assert (
        case["output_name"](
            case["protocol"],
            smoke=False,
            pilot=False,
            score_mode="f_plus_g",
        )
        == f"actor_free_td_lewm_{version}_c_cube_o50_f_plus_g_formal"
    )

    with pytest.raises(ValueError, match="only valid"):
        case["configure"](
            case["protocol"],
            smoke=False,
            pilot=False,
            score_mode="f_plus_g",
            g_first_weight=1.0,
        )


@pytest.mark.parametrize("version", VERSIONS)
def test_first_action_output_names_include_a_unique_alpha_slug(version: str) -> None:
    case = _version_case(version)
    weights_and_slugs = (
        (0.0, "0"),
        (0.25, "0p25"),
        (0.5, "0p5"),
        (1.0, "1"),
        (2.0, "2"),
    )
    names = {
        case["output_name"](
            case["protocol"],
            smoke=False,
            pilot=False,
            score_mode="f_plus_g_first",
            g_first_weight=weight,
        )
        for weight, _ in weights_and_slugs
    }

    assert len(names) == len(weights_and_slugs)
    for weight, slug in weights_and_slugs:
        assert case["output_name"](
            case["protocol"],
            smoke=False,
            pilot=False,
            score_mode="f_plus_g_first",
            g_first_weight=weight,
        ) == (
            f"actor_free_td_lewm_{version}_c_cube_o50_"
            f"f_plus_g_first_alpha_{slug}_formal"
        )


@pytest.mark.parametrize("version", VERSIONS)
def test_first_action_manifest_and_result_metadata_is_structured_and_conditional(
    version: str,
) -> None:
    case = _version_case(version)
    configured = case["configure"](
        case["protocol"],
        smoke=False,
        pilot=False,
        score_mode="f_plus_g_first",
        g_first_weight=1.0,
    )

    metadata = case["metadata"](configured, configured["planning"])
    assert metadata == {
        "g_first_weight": 1.0,
        "planning": {"horizon": 5},
        "score_definition": case["common"].FIRST_ACTION_SCORE_DEFINITION,
    }
    metadata["score_definition"]["formula"] = "mutated test copy"
    assert configured["inference_objective"]["score_definition"]["formula"] == (
        "f_cost - g_first_weight * q_first"
    )


def test_v2_ema_supports_first_action_without_changing_legacy_horizons() -> None:
    module = importlib.import_module("tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_c")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v2_ema_sg_c_cube_checkpoint_o50.yaml"
    )
    assert set(ema_sg_common.FORMAL_HORIZON_BY_SCORE_MODE) == {
        "f_only",
        "g_only",
        "f_plus_g",
        "f_plus_g_first",
        "g_only_f_rollout_mean",
    }
    configured = ema_sg_common.configure_actor_free_td_v2_ema_sg_evaluation_mode(
        protocol,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g_first",
        g_first_weight=0.5,
    )
    assert configured["planning"]["horizon"] == 5
    assert configured["inference_objective"]["g_first_weight"] == 0.5
    assert configured["inference_objective"]["score_definition"] == (
        v2_common.FIRST_ACTION_SCORE_DEFINITION
    )
    assert (
        ema_sg_common.actor_free_td_v2_ema_sg_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="f_plus_g_first",
            g_first_weight=0.5,
        )
        == "actor_free_td_lewm_v2_ema_sg_c_cube_o50_"
        "f_plus_g_first_alpha_0p5_formal"
    )


def _load_cli(version: str, variant: str) -> ModuleType:
    path = ROOT / "scripts" / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
    module_name = f"_test_first_action_cli_{version}_{variant}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_all_18_cli_entrypoints_accept_and_forward_first_action_alpha(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
    variant: str,
) -> None:
    module = _load_cli(version, variant)
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> dict[str, bool]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        module,
        f"evaluate_actor_free_td_lewm_{version}_{variant}",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--config",
            "protocol.yaml",
            "--dataset",
            "dataset.npz",
            "--checkpoint-path",
            "checkpoint.pt",
            "--output-dir",
            "results",
            "--score-mode",
            "f_plus_g_first",
            "--g-first-weight",
            "0.5",
        ],
    )

    module.main()

    assert captured["score_mode"] == "f_plus_g_first"
    assert captured["g_first_weight"] == 0.5
    assert captured["output_dir"] == "results"
    assert '"ok": true' in capsys.readouterr().out


def test_v1_first_q2_protocol_records_exact_candidate_normalization() -> None:
    case = _version_case("v1")
    configured = case["configure"](
        case["protocol"],
        smoke=False,
        pilot=False,
        score_mode=v1_common.FIRST_Q2_SCORE_MODE,
        g_first_weight=0.25,
    )

    inference = configured["inference_objective"]
    assert configured["planning"]["horizon"] == 5
    assert inference["score_mode"] == "f_plus_g_first_q2"
    assert inference["g_first_weight"] == 0.25
    assert inference["score_definition"] == v1_common.FIRST_Q2_SCORE_DEFINITION
    assert inference["score_definition"]["normalization"] == "population_z_score"
    assert inference["score_definition"]["normalization_axis"] == (
        "cem_candidate_sample_axis_dim_1_per_environment"
    )
    assert inference["score_definition"]["normalization_scope"] == (
        "independent_per_get_cost_call"
    )
    assert inference["score_definition"]["normalization_epsilon"] == 1e-6
    v1_common.validate_frozen_actor_free_td_v1_evaluation_protocol(
        configured,
        spec=case["common"].FrozenActorFreeTDV1MethodSpec(
            method=configured["method"],
            variant=configured["variant"],
            display_name="test",
            objective_keys=tuple(configured["joint_objective"]),
            validate_method_config=lambda _: None,
        ),
    )


def test_first_q2_is_v1_only_and_not_inherited_by_v0_or_v2() -> None:
    assert v1_common.FIRST_Q2_SCORE_MODE in v1_common.SCORE_MODES
    assert v1_common.FIRST_Q2_SCORE_MODE not in v0_common.SCORE_MODES
    assert v1_common.FIRST_Q2_SCORE_MODE not in v2_common.V2_SCORE_MODES

    for configure, protocol in (
        (
            v0_common.configure_frozen_actor_free_td_v0_evaluation_mode,
            _version_case("v0")["protocol"],
        ),
        (
            v2_common.configure_actor_free_td_v2_evaluation_mode,
            _version_case("v2")["protocol"],
        ),
    ):
        with pytest.raises(ValueError, match="incompatible"):
            configure(
                protocol,
                smoke=False,
                pilot=False,
                score_mode=v1_common.FIRST_Q2_SCORE_MODE,
                g_first_weight=0.25,
            )


def test_v1_first_q2_output_name_and_metadata_are_isolated_from_first_q() -> None:
    case = _version_case("v1")
    configured = case["configure"](
        case["protocol"],
        smoke=False,
        pilot=False,
        score_mode=v1_common.FIRST_Q2_SCORE_MODE,
        g_first_weight=0.5,
    )

    assert case["output_name"](
        case["protocol"],
        smoke=False,
        pilot=False,
        score_mode=v1_common.FIRST_Q2_SCORE_MODE,
        g_first_weight=0.5,
    ) == ("actor_free_td_lewm_v1_c_cube_o50_f_plus_g_first_q2_alpha_0p5_formal")
    metadata = case["metadata"](configured, configured["planning"])
    assert metadata["g_first_weight"] == 0.5
    assert metadata["planning"] == {"horizon": 5}
    assert metadata["score_definition"] == v1_common.FIRST_Q2_SCORE_DEFINITION
    assert metadata["score_definition"] != v1_common.FIRST_ACTION_SCORE_DEFINITION


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_v1_cli_entrypoints_accept_and_forward_first_q2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variant: str,
) -> None:
    module = _load_cli("v1", variant)
    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> dict[str, bool]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        module,
        f"evaluate_actor_free_td_lewm_v1_{variant}",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--config",
            "protocol.yaml",
            "--dataset",
            "dataset.npz",
            "--checkpoint-path",
            "checkpoint.pt",
            "--output-dir",
            "results",
            "--score-mode",
            v1_common.FIRST_Q2_SCORE_MODE,
            "--g-first-weight",
            "0.25",
        ],
    )

    module.main()

    assert captured["score_mode"] == v1_common.FIRST_Q2_SCORE_MODE
    assert captured["g_first_weight"] == 0.25
    assert captured["output_dir"] == "results"
    assert '"ok": true' in capsys.readouterr().out
