from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tdwm.evaluation.actor_free_td_lewm_v1_c import (
    FORMAL_O100_PLANNING as FORMAL_V1_C_O100_PLANNING,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c import (
    configure_actor_free_td_lewm_v1_c_evaluation_mode,
    load_actor_free_td_lewm_v1_c_evaluation_protocol,
    validate_actor_free_td_lewm_v1_c_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    FORMAL_O100_PLANNING as FORMAL_V1_C3_O100_PLANNING,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    FORMAL_O100_SELECTION_SHA256,
    STATE_V_FIRST_Q2_SCORE_MODE,
    configure_actor_free_td_lewm_v1_c3_evaluation_mode,
    load_actor_free_td_lewm_v1_c3_evaluation_protocol,
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    FORMAL_SELECTION_SHA256_BY_PROTOCOL,
    v1_evaluation_protocol_label,
)
from tdwm.evaluation.lewm_checkpoint import sample_start_goal_pairs

V1_C_O100_CONFIG = Path(
    "configs/experiment/actor_free_td_lewm_v1_c_cube_checkpoint_o100.yaml"
)
V1_C3_O100_CONFIG = Path(
    "configs/experiment/actor_free_td_lewm_v1_c3_cube_checkpoint_o100.yaml"
)
O100_RANKS_SHA256 = (
    "36994b1ab36656666ff91b379a59829c4b2af150b1f4ed23d409deb5cca9654e"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_o100_seed_42_selection_is_locked(tmp_path: Path) -> None:
    episodes, starts, ranks = sample_start_goal_pairs(
        np.full(10_000, 201),
        goal_offset=100,
        episodes=50,
        seed=42,
    )
    selection = {
        "episode_indices": episodes.tolist(),
        "start_steps": starts.tolist(),
        "goal_steps": (starts + 100).tolist(),
        "valid_row_ranks": ranks.tolist(),
    }
    selection_path = tmp_path / "episode_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == (
        FORMAL_O100_SELECTION_SHA256
    )
    assert FORMAL_SELECTION_SHA256_BY_PROTOCOL["o100"] == (
        FORMAL_O100_SELECTION_SHA256
    )
    assert _canonical_sha256(ranks.tolist()) == O100_RANKS_SHA256


@pytest.mark.parametrize(
    "config_path",
    [V1_C_O100_CONFIG, V1_C3_O100_CONFIG],
)
def test_v1_o100_configs_lock_the_formal_envelope(config_path: Path) -> None:
    loader = (
        load_actor_free_td_lewm_v1_c3_evaluation_protocol
        if "_c3_" in config_path.name
        else load_actor_free_td_lewm_v1_c_evaluation_protocol
    )
    protocol = loader(config_path)

    assert v1_evaluation_protocol_label(protocol) == "o100"
    assert protocol["evaluation"]["episodes"] == 50
    assert protocol["evaluation"]["goal_offset"] == 100
    assert protocol["evaluation"]["selection_sha256"] == (
        FORMAL_O100_SELECTION_SHA256
    )
    if "_c3_" in config_path.name:
        assert protocol["planning"] == FORMAL_V1_C3_O100_PLANNING
    else:
        for key, expected in FORMAL_V1_C_O100_PLANNING.items():
            assert protocol["planning"][key] == expected


@pytest.mark.parametrize(
    "score_mode",
    [
        "f_only",
        "g_only",
        "f_plus_g",
        "f_plus_g_first",
        "f_plus_g_first_q2",
        "g_only_f_rollout_mean",
    ],
)
def test_v1_c_o100_modes_use_rh1_and_mode_specific_horizon(
    score_mode: str,
) -> None:
    formal = load_actor_free_td_lewm_v1_c_evaluation_protocol(V1_C_O100_CONFIG)
    configured = configure_actor_free_td_lewm_v1_c_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=score_mode,
        g_first_weight=0.25 if "first" in score_mode else None,
    )

    assert configured["planning"]["episode_budget"] == 200
    assert configured["planning"]["receding_horizon"] == 1
    assert configured["planning"]["horizon"] == (
        1 if score_mode == "g_only" else 5
    )
    assert configured["inference_objective"]["replanning"] == (
        "every_action_block"
    )
    validate_actor_free_td_lewm_v1_c_evaluation_protocol(configured)


def test_v1_c3_o100_first_q2_keeps_formal_rh1() -> None:
    formal = load_actor_free_td_lewm_v1_c3_evaluation_protocol(V1_C3_O100_CONFIG)
    configured = configure_actor_free_td_lewm_v1_c3_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=STATE_V_FIRST_Q2_SCORE_MODE,
        g_first_weight=0.1,
    )

    assert configured["planning"] == FORMAL_V1_C3_O100_PLANNING
    assert configured["inference_objective"]["replanning"] == (
        "every_action_block"
    )
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol(configured)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evaluation", "goal_offset", 50),
        ("evaluation", "episodes", 100),
        ("planning", "episode_budget", 100),
        ("planning", "receding_horizon", 5),
    ],
)
def test_v1_c_o100_rejects_protocol_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    protocol = load_actor_free_td_lewm_v1_c_evaluation_protocol(V1_C_O100_CONFIG)
    changed = deepcopy(protocol)
    changed[section][field] = value

    with pytest.raises(ValueError):
        validate_actor_free_td_lewm_v1_c_evaluation_protocol(changed)
