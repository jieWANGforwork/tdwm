from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stable_worldmodel as swm
import yaml

from tdwm.training import eff_protocol as module

CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment/effplan_cube.yaml"


def test_draft_config_cannot_start_formal_training_or_evaluation():
    for stage in (
        "eff_training",
        "planner_generation",
        "planner_refinement",
        "evaluation",
    ):
        with pytest.raises(ValueError, match="not approved"):
            module.load_eff_protocol(CONFIG, stage=stage)


def test_shared_sources_can_be_audited_before_numerical_choices_are_locked():
    config = module.load_eff_protocol(CONFIG)
    assert config["data"]["episode_split"] == "rp1_cube_8000_2000"
    reference = module.baseline_reference(config, CONFIG)
    assert reference["dataset"]["expected_episodes"] == 10000
    assert reference["planning"]["horizon"] == 5


def test_grouping_cannot_be_silently_omitted_even_after_numeric_fields_filled(tmp_path):
    config = module.load_eff_protocol(CONFIG)
    config["eff_training"] = {"status": "locked", "settings": {"beta": 5}}
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="efficiency_groups"):
        module.load_eff_protocol(path, stage="eff_training")


def test_wrong_dependency_and_old_clip_split_are_rejected(tmp_path):
    original = module.load_eff_protocol(CONFIG)
    for key, value in (
        ("runtime", {"stable_worldmodel_version": "0.2"}),
        ("data", {"episode_split": "random_clip", "state_stride": 5}),
    ):
        config = copy.deepcopy(original)
        config[key] = value
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError):
            module.load_eff_protocol(path)


def test_public_dataset_metadata_audit_does_not_promote_truncations(
    monkeypatch, tmp_path
):
    terminated = np.zeros((2010000, 1), dtype=np.float32)
    truncated = terminated.copy()
    truncated[200::201] = 1
    columns = {"terminated": terminated, "truncated": truncated}
    dataset = SimpleNamespace(
        lengths=np.full(10000, 201), get_col_data=columns.__getitem__
    )
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(swm.data, "load_dataset", load)
    monkeypatch.setattr(
        module, "_resolve_frozen_dataset_source", lambda *args: {"format": "lance"}
    )
    result = module.prepare_eff_terminal_metadata(
        config_path=CONFIG,
        dataset_path=tmp_path / "dataset.lance",
        output_dir=tmp_path / "metadata",
    )
    assert result["true_terminal_count"] == 0 and result["truncation_count"] == 10000
    assert calls[0]["keys_to_load"] == ["terminated", "truncated"]
    actual = np.load(tmp_path / "metadata/terminal_at_state.npy", allow_pickle=False)
    assert not actual.any() and actual.dtype == np.bool_
    manifest = json.loads((tmp_path / "metadata/manifest.json").read_text())
    assert (
        module.sha256_file(tmp_path / "metadata/terminal_at_state.npy")
        == manifest["files"]["terminal"]["sha256"]
    )
    columns["terminated"][1] = 1
    with pytest.raises(ValueError, match="re-audit"):
        module.prepare_eff_terminal_metadata(
            config_path=CONFIG,
            dataset_path=tmp_path / "dataset.lance",
            output_dir=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()
