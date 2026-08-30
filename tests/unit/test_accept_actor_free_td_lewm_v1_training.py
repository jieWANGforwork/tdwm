from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/accept_actor_free_td_lewm_v1_training.py"
SPEC = importlib.util.spec_from_file_location("v1_training_acceptance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)


def test_acceptance_defaults_to_its_v1_worktree_and_canonical_action_hash() -> None:
    args = ACCEPTANCE.parse_args(["--output-root", "/tmp/formal-v1-output"])

    assert Path(args.repo_root).resolve() == REPOSITORY_ROOT
    assert ACCEPTANCE.EXPECTED_ACTION_ENCODER_SHA256 == (
        "2657b55140013b4b071cd8cdea63f1eac5c65c498d55331c7499744ef31a9cd3"
    )


def test_pid_map_is_exact_and_unambiguous() -> None:
    mapping = ACCEPTANCE.parse_pid_map("c=11,d=12,f=13,g1=14,g2=15,g3=16")

    assert mapping == {"c": 11, "d": 12, "f": 13, "g1": 14, "g2": 15, "g3": 16}
    with pytest.raises(argparse.ArgumentTypeError):
        ACCEPTANCE.parse_pid_map("c=11,d=12")
    with pytest.raises(argparse.ArgumentTypeError):
        ACCEPTANCE.parse_pid_map("c=11,d=12,f=13,g1=14,g2=15,g2=16")


def test_action_encoder_hash_uses_submodule_local_keys() -> None:
    audit = ACCEPTANCE.AcceptanceAudit()
    seen = {}
    world_state = {
        "action_encoder.projection.weight": torch.arange(6).reshape(2, 3),
        "action_encoder.projection.bias": torch.arange(2),
        "encoder.unrelated": torch.ones(1),
    }

    def record_hash(state):
        seen.update(state)
        return "canonical"

    digest = ACCEPTANCE._canonical_action_encoder_hash(
        world_state,
        record_hash,
        audit=audit,
        context="checkpoint.world_model_state_dict",
    )

    assert audit.errors == []
    assert digest == "canonical"
    assert set(seen) == {"projection.weight", "projection.bias"}
    assert torch.equal(
        seen["projection.weight"],
        world_state["action_encoder.projection.weight"],
    )


def test_atomic_output_and_exit_statuses(tmp_path: Path) -> None:
    destination = tmp_path / "training_acceptance.json"
    destination.write_text("stale")
    value = {"status": "PASS_WITH_WARNINGS", "errors": [], "warnings": ["exit"]}

    ACCEPTANCE._atomic_write_json(destination, value)

    assert json.loads(destination.read_text()) == value
    assert destination.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob(".training_acceptance.json.*.tmp"))
    assert ACCEPTANCE.acceptance_exit_code({"status": "PASS"}) == 0
    assert ACCEPTANCE.acceptance_exit_code({"status": "PASS_WITH_WARNINGS"}) == 0
    assert ACCEPTANCE.acceptance_exit_code({"status": "FAIL"}) == 1


def test_failed_cli_still_atomically_records_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "missing-output"
    output_json = tmp_path / "evidence" / "training_acceptance.json"

    exit_code = ACCEPTANCE.main(
        [
            "--repo-root",
            str(REPOSITORY_ROOT),
            "--output-root",
            str(output_root),
            "--output-json",
            str(output_json),
            "--min-free-gib",
            "0",
        ]
    )

    evidence = json.loads(output_json.read_text())
    assert exit_code == 1
    assert evidence["status"] == "FAIL"
    assert evidence["expected_action_encoder_sha256"] == (
        ACCEPTANCE.EXPECTED_ACTION_ENCODER_SHA256
    )
    assert any("missing training root" in error for error in evidence["errors"])
