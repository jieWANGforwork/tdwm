from __future__ import annotations

import importlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.adapters.g_weighted_cem import GWeightedCEMConfig
from tdwm.evaluation.g_weighted_cem import (
    configure_g_weighted_cem,
    g_weighted_cem_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ("v0", "v1", "v2", "v2_ema_sg")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def baseline_protocol(version, variant, offset=50):
    stem = f"actor_free_td_lewm_{version}_{variant}"
    module = importlib.import_module(f"tdwm.evaluation.{stem}")
    original = getattr(module, f"load_{stem}_evaluation_protocol")(
        ROOT / "configs" / "experiment" / f"{stem}_cube_checkpoint_o{offset}.yaml"
    )
    return getattr(module, f"configure_{stem}_evaluation_mode")(
        original, smoke=False, pilot=False, score_mode="f_only", g_first_weight=None
    )


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ("path", "action"))
def test_existing_versions_preserve_checkpoints_pairs_planning_and_budget(
    version, variant, mode
):
    original = baseline_protocol(version, variant)
    snapshot = deepcopy(original)
    changed = configure_g_weighted_cem(original, GWeightedCEMConfig(mode))
    assert original == snapshot
    for key, value in original.items():
        if key != "inference_objective":
            assert changed[key] == value
    metadata = g_weighted_cem_metadata(changed)
    assert metadata["score_mode"] == f"g_{mode}_weighted_cem"
    assert metadata["elite_selection_score_mode"] == "f_only"
    assert metadata["score_definition"]["g_calls_per_iteration"] == 150
    assert metadata["score_definition"]["score_normalization"] == "none"
    assert g_weighted_cem_metadata(original) == {}


@pytest.mark.parametrize("offset", (25, 50, 100))
@pytest.mark.parametrize("variant", ("c", "c4"))
def test_o25_o50_o100_use_existing_feedback_and_correct_g_state_source(offset, variant):
    original = baseline_protocol("v1", variant, offset)
    changed = configure_g_weighted_cem(original, GWeightedCEMConfig("action"))
    assert changed["evaluation"] == original["evaluation"]
    assert changed["planning"] == original["planning"]
    assert changed["evaluation"]["goal_offset"] == offset
    definition = g_weighted_cem_metadata(changed)["score_definition"]
    assert definition["g_state_source"] == (
        "f_post_action_state" if variant == "c4" else "f_pre_action_state_and_action"
    )


def test_c2_uses_the_same_new_inference_entrypoint():
    protocol = baseline_protocol("v1", "c2")
    changed = configure_g_weighted_cem(protocol, GWeightedCEMConfig("path"))
    assert changed["source_v1_c"] == protocol["source_v1_c"]


def test_nonbaseline_cost_and_single_block_are_not_silently_redefined():
    protocol = baseline_protocol("v1", "c")
    protocol["inference_objective"]["score_mode"] = "f_plus_g"
    with pytest.raises(ValueError, match="F-only"):
        configure_g_weighted_cem(protocol, GWeightedCEMConfig("path"))
    protocol["inference_objective"]["score_mode"] = "f_only"
    protocol["planning"]["horizon"] = 1
    with pytest.raises(ValueError, match="five"):
        configure_g_weighted_cem(protocol, GWeightedCEMConfig("path"))


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "g_weighted_cem_entrypoint",
        ROOT / "scripts" / "evaluate_actor_free_td_lewm_g_weighted_cem.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("mode", ("path", "action"))
def test_dry_run_resolves_without_checkpoint_or_gpu(version, mode):
    entry = load_entrypoint()
    args = entry.parse_args(
        [
            "--version",
            version,
            "--variant",
            "c",
            "--weight-mode",
            mode,
            "--config",
            str(
                ROOT
                / "configs"
                / "experiment"
                / f"actor_free_td_lewm_{version}_c_cube_checkpoint_o50.yaml"
            ),
            "--dry-run",
        ]
    )
    protocol, _ = entry.resolve_protocol(args)
    assert protocol["inference_objective"]["score_mode"] == f"g_{mode}_weighted_cem"


def test_execution_requires_declared_checkpoint_hash_and_new_output(monkeypatch):
    entry = load_entrypoint()
    monkeypatch.delenv("TDWM_CUBE_DATASET", raising=False)
    with pytest.raises(SystemExit):
        entry.parse_args(
            [
                "--version",
                "v1",
                "--variant",
                "c",
                "--weight-mode",
                "path",
                "--config",
                "unused.yaml",
            ]
        )
