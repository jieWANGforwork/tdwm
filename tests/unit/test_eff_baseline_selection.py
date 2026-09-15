"""Baseline fixed pairs are explicit; legacy heldout selection stays distinct."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tdwm.evaluation.eff_action import HISTORICAL_SELECTION_SHA256
from tdwm.evaluation.effplan import prepare_eff_selections, validate_selection
from tdwm.training.eff_protocol import load_eff_protocol

CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment/effplan_cube_same_episode_v1.yaml"


@pytest.mark.parametrize("protocol", ["baseline_fixed", "episode_heldout"])
def test_pair_protocols_roundtrip_and_tampering(tmp_path, protocol):
    source = load_eff_protocol(CONFIG)["source"]["dataset_source_sha256"]
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "manifest.json").write_text(json.dumps(
        dict(dataset_source_sha256=source, episode_lengths=[201] * 10000)))
    paths = prepare_eff_selections(config_path=CONFIG, terminal_metadata=metadata,
                                   output_dir=tmp_path / "pairs",
                                   selection_protocol=protocol)
    for offset in [25, 50, 100]:
        selection = json.loads(Path(paths[f"O{offset}"]["path"]).read_text())
        validate_selection(selection, source_sha256=source)
        ids = np.asarray(selection["pairs"]["episode_indices"])
        if protocol == "baseline_fixed":
            assert selection["baseline_pairs_sha256"] == HISTORICAL_SELECTION_SHA256[offset]
            assert (ids < 8000).any()
        else:
            assert (ids >= 8000).all()
            assert "baseline_pairs_sha256" not in selection
        bad = copy.deepcopy(selection)
        bad["pairs"]["start_steps"][0] += 1
        with pytest.raises(ValueError):
            validate_selection(bad, source_sha256=source)
        with pytest.raises(ValueError):
            validate_selection(selection, source_sha256="wrong")
    assert prepare_eff_selections(config_path=CONFIG, terminal_metadata=metadata,
                                  output_dir=tmp_path / "pairs",
                                  selection_protocol=protocol) == paths
    alternate = "episode_heldout" if protocol == "baseline_fixed" else "baseline_fixed"
    with pytest.raises(FileExistsError):
        prepare_eff_selections(config_path=CONFIG, terminal_metadata=metadata,
                               output_dir=tmp_path / "pairs", selection_protocol=alternate)
