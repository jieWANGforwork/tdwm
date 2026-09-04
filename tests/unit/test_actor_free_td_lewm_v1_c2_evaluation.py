from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from tdwm.adapters.actor_free_td_lewm_v1_c2 import (
    FIRST_Q_ALIGNMENT_LOCK,
    METHOD_SPEC,
    load_actor_free_td_lewm_v1_c2_checkpoint,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c2 import (
    configure_actor_free_td_lewm_v1_c2_evaluation_mode,
    evaluate_actor_free_td_lewm_v1_c2,
    load_actor_free_td_lewm_v1_c2_evaluation_protocol,
    validate_actor_free_td_lewm_v1_c2_checkpoint_protocol,
    validate_actor_free_td_lewm_v1_c2_evaluation_protocol,
)

CONFIG_PATH = Path(
    "configs/experiment/actor_free_td_lewm_v1_c2_cube_checkpoint_o50.yaml"
)
SCRIPT_PATH = Path("scripts/evaluate_actor_free_td_lewm_v1_c2.py")


def test_c2_o50_protocol_locks_identity_objective_and_provenance() -> None:
    protocol = load_actor_free_td_lewm_v1_c2_evaluation_protocol(CONFIG_PATH)

    assert protocol["method"] == "actor_free_td_lewm_v1_c2"
    assert protocol["method_family"] == "actor_free_td_lewm_v1"
    assert protocol["variant"] == "c2"
    assert protocol["predictor"]["objective_version"] == 0
    assert protocol["joint_objective"]["objective"] == (
        "goal_projected_td_plus_first_q_alignment"
    )
    assert protocol["joint_objective"]["goal_projection_weight"] == 1.0
    assert protocol["joint_objective"]["first_q_alignment"] == FIRST_Q_ALIGNMENT_LOCK
    assert protocol["inference_objective"]["training_only_auxiliary"] == [
        "goal_projection_training_loss",
        "first_q_candidate_ranking_alignment",
    ]
    assert protocol["provenance"]["initialization_method"] == (
        "actor_free_td_lewm_v1_c"
    )
    assert len(protocol["provenance"]["initialization_checkpoint_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("loss", "mean_squared_error"),
        ("candidate_count", 8),
        ("rollout_horizon", 4),
        ("max_goal_examples_per_batch", 16),
        ("score_standardization", "none"),
        ("teacher_temperature", 0.5),
        ("candidate_sampling_seed_offset", 370010),
    ],
)
def test_c2_method_spec_rejects_any_changed_alignment_contract(
    field: str,
    changed: object,
) -> None:
    protocol = load_actor_free_td_lewm_v1_c2_evaluation_protocol(CONFIG_PATH)
    malformed = deepcopy(protocol)
    malformed["joint_objective"]["first_q_alignment"][field] = changed

    with pytest.raises(ValueError, match=field):
        validate_actor_free_td_lewm_v1_c2_evaluation_protocol(malformed)


def test_c2_method_spec_rejects_missing_extra_or_non_boolean_alignment_fields() -> None:
    protocol = load_actor_free_td_lewm_v1_c2_evaluation_protocol(CONFIG_PATH)

    missing = deepcopy(protocol)
    del missing["joint_objective"]["first_q_alignment"]["student"]
    with pytest.raises(ValueError, match="exactly the locked C2 fields"):
        validate_actor_free_td_lewm_v1_c2_evaluation_protocol(missing)

    extra = deepcopy(protocol)
    extra["joint_objective"]["first_q_alignment"]["unlocked"] = True
    with pytest.raises(ValueError, match="exactly the locked C2 fields"):
        validate_actor_free_td_lewm_v1_c2_evaluation_protocol(extra)

    non_boolean = deepcopy(protocol)
    non_boolean["joint_objective"]["first_q_alignment"][
        "force_first_candidate_to_mean"
    ] = 1
    with pytest.raises(ValueError, match="force_first_candidate_to_mean"):
        validate_actor_free_td_lewm_v1_c2_evaluation_protocol(non_boolean)

    wrong_numeric_type = deepcopy(protocol)
    wrong_numeric_type["joint_objective"]["first_q_alignment"]["candidate_count"] = 16.0
    with pytest.raises(ValueError, match="candidate_count"):
        validate_actor_free_td_lewm_v1_c2_evaluation_protocol(wrong_numeric_type)


@pytest.mark.parametrize(
    ("score_mode", "expected_horizon"),
    [
        ("f_only", 5),
        ("g_only", 1),
        ("f_plus_g", 5),
        ("f_plus_g_first", 5),
        ("f_plus_g_first_q2", 5),
        ("g_only_f_rollout_mean", 5),
    ],
)
def test_c2_supports_all_six_shared_v1_score_modes(
    score_mode: str,
    expected_horizon: int,
) -> None:
    protocol = load_actor_free_td_lewm_v1_c2_evaluation_protocol(CONFIG_PATH)
    weight = 0.25 if score_mode in {"f_plus_g_first", "f_plus_g_first_q2"} else None

    configured = configure_actor_free_td_lewm_v1_c2_evaluation_mode(
        protocol,
        smoke=False,
        pilot=False,
        score_mode=score_mode,
        g_first_weight=weight,
    )

    assert configured["inference_objective"]["score_mode"] == score_mode
    assert configured["planning"]["horizon"] == expected_horizon
    validate_actor_free_td_lewm_v1_c2_evaluation_protocol(configured)


def test_c2_adapter_and_evaluator_delegate_with_the_c2_method_spec() -> None:
    sentinel = object()
    with patch(
        "tdwm.adapters.actor_free_td_lewm_v1_c2."
        "load_frozen_actor_free_td_v1_checkpoint",
        return_value=sentinel,
    ) as shared_loader:
        assert load_actor_free_td_lewm_v1_c2_checkpoint("checkpoint.pt") is sentinel
    assert shared_loader.call_args.kwargs["spec"] is METHOD_SPEC

    with patch(
        "tdwm.evaluation.actor_free_td_lewm_v1_c2.evaluate_frozen_actor_free_td_v1",
        return_value={"ok": True},
    ) as shared_evaluator:
        assert evaluate_actor_free_td_lewm_v1_c2(example="value") == {"ok": True}
    assert shared_evaluator.call_args.kwargs["spec"] is METHOD_SPEC
    assert shared_evaluator.call_args.kwargs["checkpoint_validator"] is (
        validate_actor_free_td_lewm_v1_c2_checkpoint_protocol
    )
    assert shared_evaluator.call_args.kwargs["example"] == "value"


def test_c2_runtime_checkpoint_validator_binds_the_exact_v1_c_parent() -> None:
    protocol = load_actor_free_td_lewm_v1_c2_evaluation_protocol(CONFIG_PATH)
    predictor_config = {"source_v1_c": deepcopy(protocol["source_v1_c"])}
    shared_path = (
        "tdwm.evaluation.actor_free_td_lewm_v1_c2."
        "validate_frozen_actor_free_td_v1_checkpoint_protocol"
    )

    with patch(shared_path) as shared_validator:
        validate_actor_free_td_lewm_v1_c2_checkpoint_protocol(
            payload={},
            predictor_config=predictor_config,
            protocol=protocol,
            spec=METHOD_SPEC,
            require_formal_completion=True,
        )
    assert shared_validator.call_args.kwargs["spec"] is METHOD_SPEC

    changed = deepcopy(protocol)
    changed["source_v1_c"]["checkpoint_sha256"] = "f" * 64
    with patch(shared_path):
        with pytest.raises(ValueError, match="source_v1_c"):
            validate_actor_free_td_lewm_v1_c2_checkpoint_protocol(
                payload={},
                predictor_config=predictor_config,
                protocol=changed,
                spec=METHOD_SPEC,
                require_formal_completion=True,
            )


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
def test_c2_evaluation_cli_accepts_every_shared_score_mode(
    monkeypatch: pytest.MonkeyPatch,
    score_mode: str,
) -> None:
    spec = importlib.util.spec_from_file_location("c2_evaluation_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--checkpoint-path",
            "checkpoint.pt",
            "--score-mode",
            score_mode,
        ],
    )

    assert module.parse_args().score_mode == score_mode
