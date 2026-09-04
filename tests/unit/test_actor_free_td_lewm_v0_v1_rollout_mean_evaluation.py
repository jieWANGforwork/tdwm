from __future__ import annotations

import importlib
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tdwm.evaluation import frozen_actor_free_td_v0_common as v0_common
from tdwm.evaluation import frozen_actor_free_td_v1_common as v1_common

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "experiment"
VERSIONS = ("v0", "v1")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")
MODE = "g_only_f_rollout_mean"


def _common(version: str) -> ModuleType:
    return v0_common if version == "v0" else v1_common


def _evaluation_module(version: str, variant: str) -> ModuleType:
    return importlib.import_module(
        f"tdwm.evaluation.actor_free_td_lewm_{version}_{variant}"
    )


def _load(version: str, variant: str, *, mean: bool) -> dict[str, Any]:
    module = _evaluation_module(version, variant)
    loader = getattr(
        module,
        f"load_actor_free_td_lewm_{version}_{variant}_evaluation_protocol",
    )
    suffix = f"_{MODE}" if mean else ""
    path = (
        CONFIG_ROOT
        / f"actor_free_td_lewm_{version}_{variant}_cube_checkpoint_o50{suffix}.yaml"
    )
    return loader(path)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_each_rollout_mean_overlay_is_loadable_and_version_native(
    version: str,
    variant: str,
) -> None:
    common = _common(version)
    protocol = _load(version, variant, mean=True)
    inference = protocol["inference_objective"]
    expected_action_processing = (
        "normalized_raw_25d_action_block"
        if version == "v0"
        else "frozen_shared_lewm_action_encoder_to_192d"
    )

    assert protocol["id"].endswith(f"checkpoint_o50_{MODE}")
    assert protocol["planning"]["horizon"] == 5
    assert inference["score_mode"] == MODE
    assert inference["f_score"] == "none"
    assert inference["f_score_reducer"] == "none"
    assert inference["f_transition_used"] is True
    assert inference["f_goal_distance_used"] is False
    assert inference["score_definition"] == common.ROLLOUT_MEAN_SCORE_DEFINITION
    assert (
        inference["score_definition"]["action_processing"] == expected_action_processing
    )
    assert "g_first_weight" not in inference


@pytest.mark.parametrize("version", VERSIONS)
def test_rollout_mean_configurer_round_trip_restores_legacy_contract(
    version: str,
) -> None:
    common = _common(version)
    module = _evaluation_module(version, "c")
    base = _load(version, "c", mean=False)
    configure = getattr(
        common,
        f"configure_frozen_actor_free_td_{version}_evaluation_mode",
    )
    validate = getattr(
        common,
        f"validate_frozen_actor_free_td_{version}_evaluation_protocol",
    )

    mean = configure(
        base,
        smoke=False,
        pilot=False,
        score_mode=MODE,
    )
    validate(mean, spec=module.METHOD_SPEC)
    assert mean["inference_objective"]["f_goal_distance_used"] is False

    legacy = configure(
        mean,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g",
    )
    validate(legacy, spec=module.METHOD_SPEC)
    inference = legacy["inference_objective"]
    assert inference["f_score"] == common.LEGACY_F_SCORE
    assert inference["f_score_reducer"] == common.LEGACY_F_SCORE_REDUCER
    assert inference["g_score"] == common.LEGACY_G_SCORE
    assert "score_definition" not in inference
    for key in common.ROLLOUT_MEAN_ONLY_INFERENCE_KEYS:
        assert key not in inference

    malformed = deepcopy(mean)
    malformed["inference_objective"]["f_goal_distance_used"] = True
    with pytest.raises(ValueError, match="f_goal_distance_used"):
        validate(malformed, spec=module.METHOD_SPEC)


@pytest.mark.parametrize("version", VERSIONS)
def test_rollout_mean_metadata_records_no_terminal_goal_distance(
    version: str,
) -> None:
    common = _common(version)
    protocol = _load(version, "c", mean=True)

    metadata = common._rollout_mean_output_metadata(
        protocol,
        protocol["planning"],
    )

    assert metadata["planning_horizon"] == 5
    assert metadata["rollout_horizon"] == 5
    assert metadata["f_transition_used"] is True
    assert metadata["f_goal_distance_used"] is False
    assert metadata["score_definition"] == common.ROLLOUT_MEAN_SCORE_DEFINITION


def _load_cli(version: str, variant: str) -> ModuleType:
    path = ROOT / "scripts" / f"evaluate_actor_free_td_lewm_{version}_{variant}.py"
    spec = importlib.util.spec_from_file_location(
        f"_test_{version}_{variant}_rollout_mean_cli",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_all_twelve_cli_entrypoints_forward_rollout_mean_without_alpha(
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
            str(
                CONFIG_ROOT
                / f"actor_free_td_lewm_{version}_{variant}_cube_checkpoint_o50_"
                f"{MODE}.yaml"
            ),
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
