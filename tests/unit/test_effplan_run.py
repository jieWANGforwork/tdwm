from __future__ import annotations

import pytest

from tdwm.training.effplan_run import (
    _planner_settings,
    planner_learning_rate,
    run_effplan_training,
)
from tdwm.training.effplan_runtime import EffPlanTrainSettings

REPO_CONFIG = "configs/experiment/effplan_cube.yaml"


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
            config_path=REPO_CONFIG,
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
            config_path=REPO_CONFIG,
            phase="refinement",
            latent_store=tmp_path / "store",
            terminal_metadata=tmp_path / "terminal",
            eff_checkpoint=tmp_path / "eff.pt",
            eff_manifest=tmp_path / "eff.json",
            lewm_checkpoint=None,
            output_dir=tmp_path / "out",
            device="cpu",
        )
