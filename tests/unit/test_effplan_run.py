from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tdwm.training import eff_protocol
from tdwm.training.effplan_run import (
    _planner_settings,
    planner_learning_rate,
    run_effplan_training,
)
from tdwm.training.effplan_runtime import EffPlanTrainSettings

REPO_CONFIG = "configs/experiment/effplan_cube.yaml"

_STAGES = (
    "eff_training",
    "planner_generation",
    "planner_refinement",
    "evaluation",
)


def draft_copy(tmp_path):
    """The shipped config is locked; gate tests need their own draft copy."""
    config = eff_protocol.load_eff_protocol(REPO_CONFIG)
    for stage in _STAGES:
        config[stage] = {**config[stage], "status": "draft"}
    path = tmp_path / "draft_protocol.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def generation_stage(**overrides) -> dict:
    settings = dict(
        phase="generation",
        seed=3072,
        learning_rate=1e-3,
        weight_decay=0.001,
        gradient_clip=1.0,
        epsilon=1e-6,
        target_readout=True,
        trajectory_coefficient=1.0,
        efficiency_coefficient=0.0,
        dynamics_coefficient=0.0,
        search_iterations=[],
        cem_candidates=4,
        cem_elites=2,
        cem_batch_size=2,
        supervision="final",
    )
    settings.update(overrides)
    return {"status": "locked", "settings": settings}


def test_settings_keep_search_budget_ordered_and_hashable():
    settings = _planner_settings(generation_stage())
    assert settings.search_iterations == ()
    refinement = _planner_settings(
        generation_stage(
            phase="refinement",
            efficiency_coefficient=0.01,
            dynamics_coefficient=0.1,
            search_iterations=[4, 3],
        )
    )
    assert refinement.search_iterations == (4, 3)
    assert isinstance(refinement, EffPlanTrainSettings)


def test_learning_rate_starts_at_zero_slope_and_ends_at_zero():
    settings = _planner_settings(generation_stage(learning_rate=1e-3))
    assert planner_learning_rate(settings, 1000, 0) > 0
    assert planner_learning_rate(settings, 1000, 999) < 1e-4
    with pytest.raises(ValueError):
        planner_learning_rate(settings, 1000, 1000)


def test_unlocked_configuration_is_refused_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="not approved for formal execution"):
        run_effplan_training(
            config_path=draft_copy(tmp_path),
            phase="generation",
            latent_store=tmp_path / "store",
            terminal_metadata=tmp_path / "terminal",
            eff_checkpoint=tmp_path / "eff.pt",
            eff_manifest=tmp_path / "eff.json",
            lewm_checkpoint=None,
            output_dir=tmp_path / "out",
            device="cpu",
        )


def test_phase_and_requested_phase_must_agree(tmp_path):
    with pytest.raises(ValueError, match="not approved for formal execution"):
        run_effplan_training(
            config_path=draft_copy(tmp_path),
            phase="refinement",
            latent_store=tmp_path / "store",
            terminal_metadata=tmp_path / "terminal",
            eff_checkpoint=tmp_path / "eff.pt",
            eff_manifest=tmp_path / "eff.json",
            lewm_checkpoint=None,
            output_dir=tmp_path / "out",
            device="cpu",
        )


def test_every_planner_run_key_read_by_the_trainer_is_declared():
    """The protocol gate rejects ``null`` but cannot see a *missing* key.

    ``run_effplan_training`` indexes ``config["planner_<phase>"]["run"]``
    directly, so an absent key only surfaces as a KeyError deep inside
    training -- after the Eff model has already been trained. Scan the
    trainer source for every ``run["..."]`` lookup and require the locked
    config to declare all of them.
    """
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "src/tdwm/training/effplan_run.py").read_text()
    referenced = set(re.findall(r'run\["([a-z_]+)"\]', source))
    assert referenced, "no run[...] lookups found; the scan is broken"
    config = yaml.safe_load((repo_root / REPO_CONFIG).read_text())
    for stage in ("planner_generation", "planner_refinement"):
        declared = set(config[stage]["run"])
        missing = sorted(referenced - declared)
        assert not missing, f"{stage}.run is missing {missing}"
