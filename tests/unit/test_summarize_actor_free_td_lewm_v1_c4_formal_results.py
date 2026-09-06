from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    C4_ACTION_EFFECT,
    OBJECTIVE_VERSION,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c4 import (
    configure_actor_free_td_lewm_v1_c4_evaluation_mode,
    load_actor_free_td_lewm_v1_c4_evaluation_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "scripts" / "summarize_actor_free_td_lewm_v1_c4_formal_results.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_v1_c4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)
_TEST_LOCKED_PROTOCOLS: set[str] = set()


def _selection(protocol: str, *, rank_offset: int = 0) -> dict[str, list[int]]:
    offset = SUMMARY.GOAL_OFFSET_BY_PROTOCOL[protocol]
    starts = [index * 3 for index in range(50)]
    return {
        "episode_indices": list(range(100, 150)),
        "start_steps": starts,
        "goal_steps": [start + offset for start in starts],
        "valid_row_ranks": [rank_offset + index for index in range(50)],
    }


def _outcomes(method_key: str, score_mode: str) -> list[bool]:
    baseline = {index for index in range(10)}
    if method_key == "c4":
        successes = {
            "f_only": baseline,
            "g_only": (baseline - {0}) | {10},
            "f_plus_g": baseline | {10, 11, 12, 13},
            "f_plus_g_first": (baseline - {0, 1}) | {10, 11, 12},
            "g_only_f_rollout_mean": {0, 2, 4, 6, 8, 10},
            "f_plus_g_first_q2": (baseline - {0}) | {10, 11, 12, 13, 14},
        }[score_mode]
    else:
        successes = {
            "f_only": baseline,
            "g_only": set(range(8)),
            "f_plus_g": set(range(12)),
            "f_plus_g_first": set(range(9)),
            "g_only_f_rollout_mean": {0, 2, 4, 6, 8},
            "f_plus_g_first_q2": set(range(11)),
        }[score_mode]
    return [index in successes for index in range(50)]


def _cell_directory(
    root: Path,
    *,
    protocol: str,
    variant: str,
    score_mode: str,
) -> Path:
    directory = root / "formal" / protocol / "v1" / variant / score_mode
    if score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}:
        directory /= "alpha_0p25"
    return directory


def _write_cell(
    directory: Path,
    *,
    method_key: str,
    protocol: str,
    score_mode: str,
    selection: dict[str, list[int]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    method = SUMMARY.METHODS[method_key]
    outcomes = _outcomes(method_key, score_mode)
    common = {
        "method": method["method"],
        "variant": method["variant"],
        "protocol_label": protocol,
        "evaluation_protocol": protocol.upper(),
        "goal_offset": SUMMARY.GOAL_OFFSET_BY_PROTOCOL[protocol],
        "score_mode": score_mode,
    }
    results = {
        **common,
        "smoke": False,
        "pilot": False,
        "metrics": {
            "episode_successes": outcomes,
            "success_rate": sum(outcomes) * 2.0,
        },
    }
    horizon = 1 if score_mode == "g_only" else 5
    receding_horizon = (
        1 if protocol in {"o50", "o100"} or score_mode == "g_only" else 5
    )
    planning = {
        "solver": "CEM",
        "horizon": horizon,
        "candidates": 300,
        "iterations": 30,
        "elites": 30,
        "action_block": 5,
        "frame_skip": 5,
        "history_len": 1,
        "receding_horizon": receding_horizon,
        "episode_budget": {"o25": 50, "o50": 100, "o100": 200}[protocol],
        "planning_seed": 42,
    }
    if protocol == "o25":
        planning["executed_environment_steps_before_replanning"] = (
            receding_horizon * 5
        )
    inference = {"score_mode": score_mode}
    if score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}:
        inference["g_first_weight"] = 0.25
    protocol_value = {
        "method": method["method"],
        "variant": method["variant"],
        "evaluation": {
            "episodes": 50,
            "goal_offset": SUMMARY.GOAL_OFFSET_BY_PROTOCOL[protocol],
            "start_goal_source": "same_dataset_episode",
        },
        "inference_objective": inference,
        "planning": planning,
    }
    checkpoint = {
        "path": f"/server/{method_key}/epoch_10.pt",
        "sha256": "a" * 64 if method_key == "c4" else "b" * 64,
    }
    c4_metadata: dict[str, object] = {}
    formal_protocol = None
    if method_key == "c4":
        formal_protocol = load_actor_free_td_lewm_v1_c4_evaluation_protocol(
            ROOT
            / "configs"
            / "experiment"
            / f"actor_free_td_lewm_v1_c4_cube_checkpoint_{protocol}.yaml"
        )
        protocol_value = configure_actor_free_td_lewm_v1_c4_evaluation_mode(
            formal_protocol,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
            g_first_weight=(
                0.25
                if score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}
                else None
            ),
        )
        c4_metadata = {
            "objective_version": OBJECTIVE_VERSION,
            "state_only_g": True,
            "action_enters_g": False,
            "action_effect": C4_ACTION_EFFECT,
            "g_state_source": "stopped_f_post_action_ghost_state",
            "score_definition": protocol_value["inference_objective"][
                "score_definition"
            ],
        }
        results.update(c4_metadata)
        checkpoint.update(
            {
                "objective_version": OBJECTIVE_VERSION,
                "g_config": {
                    "objective_version": OBJECTIVE_VERSION,
                    "action_effect": C4_ACTION_EFFECT,
                    "time_alignment": formal_protocol["time_alignment"],
                    "joint_objective": formal_protocol["joint_objective"],
                },
            }
        )
    manifest = {
        **c4_metadata,
        "protocol_label": protocol,
        "evaluation_protocol": protocol.upper(),
        "goal_offset": SUMMARY.GOAL_OFFSET_BY_PROTOCOL[protocol],
        "score_mode": score_mode,
        "selection": selection,
        "protocol": protocol_value,
        "checkpoint": checkpoint,
    }
    if formal_protocol is not None:
        manifest["formal_protocol"] = formal_protocol
    (directory / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (directory / "protocol_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    selection_path = directory / "episode_selection.json"
    selection_path.write_text(
        json.dumps(selection), encoding="utf-8"
    )
    action_path = directory / "action_normalization.json"
    action_path.write_text(
        json.dumps({"normalization": "shared-test-fixture"}), encoding="utf-8"
    )
    if protocol not in _TEST_LOCKED_PROTOCOLS:
        SUMMARY.EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL[protocol] = (
            SUMMARY._file_sha256(selection_path)
        )
        SUMMARY.EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL[protocol] = (
            SUMMARY._canonical_json_sha256(selection["valid_row_ranks"])
        )
        _TEST_LOCKED_PROTOCOLS.add(protocol)
    SUMMARY.EXPECTED_ACTION_NORMALIZATION_SHA256 = SUMMARY._file_sha256(action_path)


def _write_method(
    root: Path,
    *,
    method_key: str,
    protocol: str,
    rank_offset: int = 0,
    score_modes: tuple[str, ...] | None = None,
) -> None:
    variant = SUMMARY.METHODS[method_key]["variant"]
    selection = _selection(protocol, rank_offset=rank_offset)
    for score_mode in score_modes or SUMMARY.SCORE_MODES:
        _write_cell(
            _cell_directory(
                root,
                protocol=protocol,
                variant=variant,
                score_mode=score_mode,
            ),
            method_key=method_key,
            protocol=protocol,
            score_mode=score_mode,
            selection=selection,
        )


def _complete_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    c4_root = tmp_path / "c4"
    v1_c_roots = {
        protocol: tmp_path / f"v1_c_{protocol}" for protocol in SUMMARY.PROTOCOLS
    }
    for protocol in SUMMARY.PROTOCOLS:
        _write_method(c4_root, method_key="c4", protocol=protocol)
        _write_method(
            v1_c_roots[protocol], method_key="v1_c", protocol=protocol
        )
    return c4_root, v1_c_roots


def test_summary_contains_all_cells_paired_counts_and_source_hashes(
    tmp_path: Path,
) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    summary = SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)

    assert len(summary["episode_matrix"]) == 150
    assert set(summary["protocols"]) == set(SUMMARY.PROTOCOLS)
    for protocol in SUMMARY.PROTOCOLS:
        values = summary["protocols"][protocol]
        assert set(values["methods"]["c4"]["scores"]) == set(SUMMARY.SCORE_MODES)
        assert set(values["methods"]["v1_c"]["scores"]) == set(
            SUMMARY.SCORE_MODES
        )
        paired = values["comparisons"]["c4_vs_same_protocol_f_only"]["g_only"]
        assert paired["candidate_successes"] == 10
        assert paired["new"] == 1
        assert paired["lost"] == 1
        assert paired["f_plus_new_successes"] == 11
        assert paired["delta_successes"] == 0
        assert paired["exact_mcnemar_p_two_sided"] == 1.0

        cross = values["comparisons"]["c4_vs_v1_c_same_score_mode"]["g_only"]
        assert cross["reference_successes"] == 8
        assert cross["candidate_successes"] == 10
        assert cross["new"] == 3
        assert cross["lost"] == 1
        assert cross["delta_successes"] == 2
        source = values["methods"]["c4"]["scores"]["g_only"]["source"]
        results_path = Path(source["results"]["path"])
        assert source["results"]["sha256"] == SUMMARY._file_sha256(results_path)
        assert source["checkpoint"]["sha256"] == "a" * 64


def test_writes_one_json_wide_episode_csv_and_markdown(tmp_path: Path) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    summary = SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)
    paths = SUMMARY.write_outputs(summary, tmp_path / "summary")

    assert set(paths) == {"json", "csv", "markdown"}
    stored = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert stored == summary
    rows = list(csv.DictReader(io.StringIO(Path(paths["csv"]).read_text())))
    assert len(rows) == 150
    assert rows[0]["pair_id"] == "O25-P01"
    assert rows[0]["c4__f_only"] == "1"
    assert rows[0]["v1_c__f_plus_g"] == "1"
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "## O25" in markdown
    assert "## O50" in markdown
    assert "## O100" in markdown
    assert "New | Lost | F+New" in markdown
    assert "C4 relative to V1-C" in markdown
    assert "| F-only |" not in markdown.split(
        "### C4 relative to its same-protocol F-only baseline", 1
    )[1].split("## O50", 1)[0]
    assert SUMMARY.EPISODE_CSV_NAME in markdown

    assert SUMMARY.write_outputs(summary, tmp_path / "summary") == paths
    Path(paths["markdown"]).write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        SUMMARY.write_outputs(summary, tmp_path / "summary")


def test_rejects_cross_method_selection_rank_mismatch(tmp_path: Path) -> None:
    c4_root = tmp_path / "c4"
    v1_c_roots = {
        protocol: tmp_path / f"v1_c_{protocol}" for protocol in SUMMARY.PROTOCOLS
    }
    for protocol in SUMMARY.PROTOCOLS:
        _write_method(c4_root, method_key="c4", protocol=protocol)
        _write_method(
            v1_c_roots[protocol],
            method_key="v1_c",
            protocol=protocol,
            rank_offset=1000 if protocol == "o50" else 0,
        )

    with pytest.raises(ValueError, match="locked O50 selection"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)


def test_accepts_one_protocol_split_across_multiple_source_roots(
    tmp_path: Path,
) -> None:
    c4_root = tmp_path / "c4"
    v1_c_roots: dict[str, object] = {}
    for protocol in SUMMARY.PROTOCOLS:
        _write_method(c4_root, method_key="c4", protocol=protocol)
        if protocol != "o50":
            root = tmp_path / f"v1_c_{protocol}"
            _write_method(root, method_key="v1_c", protocol=protocol)
            v1_c_roots[protocol] = root

    split_roots = [tmp_path / f"v1_c_o50_part_{index}" for index in range(4)]
    groups = (
        ("f_only", "g_only", "f_plus_g"),
        ("f_plus_g_first",),
        ("g_only_f_rollout_mean",),
        ("f_plus_g_first_q2",),
    )
    for root, modes in zip(split_roots, groups):
        _write_method(
            root,
            method_key="v1_c",
            protocol="o50",
            score_modes=modes,
        )
    v1_c_roots["o50"] = split_roots

    summary = SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)

    assert summary["input_roots"]["v1_c"]["o50"] == [
        str(path.resolve()) for path in split_roots
    ]
    scores = summary["protocols"]["o50"]["methods"]["v1_c"]["scores"]
    assert {mode: values["success_count"] for mode, values in scores.items()} == {
        "f_only": 10,
        "g_only": 8,
        "f_plus_g": 12,
        "f_plus_g_first": 9,
        "g_only_f_rollout_mean": 5,
        "f_plus_g_first_q2": 11,
    }


def test_accepts_only_authentic_legacy_v1_c_o50_missing_top_level_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy_v1_c_o50"
    directory = _cell_directory(
        root,
        protocol="o50",
        variant="c",
        score_mode="f_plus_g_first",
    )
    _write_cell(
        directory,
        method_key="v1_c",
        protocol="o50",
        score_mode="f_plus_g_first",
        selection=_selection("o50"),
    )
    optional = {"protocol_label", "evaluation_protocol", "goal_offset"}
    for name in ("results.json", "protocol_manifest.json"):
        path = directory / name
        payload = json.loads(path.read_text())
        for key in optional:
            payload.pop(key)
        path.write_text(json.dumps(payload))

    cell = SUMMARY._load_cell(
        directory,
        method_key="v1_c",
        protocol="o50",
        score_mode="f_plus_g_first",
    )

    assert cell.source["top_level_protocol_metadata"] == (
        "legacy_v1_c_o50_validated_from_embedded_protocol"
    )
    assert cell.success_count == 9

    results_path = directory / "results.json"
    manifest_path = directory / "protocol_manifest.json"
    results = json.loads(results_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    for payload in (results, manifest):
        payload.update(
            {
                "protocol_label": "o50",
                "evaluation_protocol": "O50",
                "goal_offset": 50,
            }
        )
    results["goal_offset"] = 25
    results_path.write_text(json.dumps(results))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"results\.json\.goal_offset must be 50"):
        SUMMARY._load_cell(
            directory,
            method_key="v1_c",
            protocol="o50",
            score_mode="f_plus_g_first",
        )

    results["goal_offset"] = 50
    results.pop("evaluation_protocol")
    results_path.write_text(json.dumps(results))
    with pytest.raises(ValueError, match="must be either fully explicit or absent"):
        SUMMARY._load_cell(
            directory,
            method_key="v1_c",
            protocol="o50",
            score_mode="f_plus_g_first",
        )


def test_missing_top_level_protocol_label_is_not_allowed_for_c4(
    tmp_path: Path,
) -> None:
    root = tmp_path / "c4"
    directory = _cell_directory(
        root,
        protocol="o50",
        variant="c4",
        score_mode="f_only",
    )
    _write_cell(
        directory,
        method_key="c4",
        protocol="o50",
        score_mode="f_only",
        selection=_selection("o50"),
    )
    results_path = directory / "results.json"
    results = json.loads(results_path.read_text())
    results.pop("protocol_label")
    results_path.write_text(json.dumps(results))

    with pytest.raises(ValueError, match=r"results\.json\.protocol_label"):
        SUMMARY._load_cell(
            directory,
            method_key="c4",
            protocol="o50",
            score_mode="f_only",
        )


def test_rejects_conflicting_duplicate_cells_across_source_roots(
    tmp_path: Path,
) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    duplicate_root = tmp_path / "v1_c_o50_conflict"
    _write_method(
        duplicate_root,
        method_key="v1_c",
        protocol="o50",
        score_modes=("g_only",),
    )
    cell = _cell_directory(
        duplicate_root,
        protocol="o50",
        variant="c",
        score_mode="g_only",
    )
    result_path = cell / "results.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["episode_successes"][20] = True
    result["metrics"]["success_rate"] = 18.0
    result_path.write_text(json.dumps(result))
    v1_c_roots["o50"] = [v1_c_roots["o50"], duplicate_root]

    with pytest.raises(ValueError, match="Conflicting duplicate v1_c/o50/g_only"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)


def test_rejects_non_boolean_outcome_and_missing_cell(tmp_path: Path) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    cell = _cell_directory(
        c4_root, protocol="o25", variant="c4", score_mode="g_only"
    )
    result_path = cell / "results.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["episode_successes"][0] = 1
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="50 Boolean outcomes"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)

    c4_root, v1_c_roots = _complete_inputs(tmp_path / "missing")
    missing = _cell_directory(
        c4_root, protocol="o100", variant="c4", score_mode="f_plus_g"
    )
    (missing / "results.json").unlink()
    with pytest.raises(FileNotFoundError, match="No c4/o100/f_plus_g result cell"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)


def test_rejects_changed_formal_planning_and_action_normalization(
    tmp_path: Path,
) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    cell = _cell_directory(
        c4_root, protocol="o50", variant="c4", score_mode="f_plus_g"
    )
    manifest_path = cell / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"]["planning"]["planning_seed"] = 43
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="planning_seed must be 42"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)

    c4_root, v1_c_roots = _complete_inputs(tmp_path / "normalization")
    cell = _cell_directory(
        c4_root, protocol="o100", variant="c4", score_mode="g_only"
    )
    (cell / "action_normalization.json").write_text(
        json.dumps({"normalization": "changed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="locked action normalization"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)


@pytest.mark.parametrize(
    ("protocol", "score_mode", "key", "changed"),
    (
        ("o25", "f_plus_g", "solver", "MPPI"),
        ("o25", "f_plus_g", "candidates", 301),
        ("o25", "f_plus_g", "iterations", 29),
        ("o25", "f_plus_g", "elites", 31),
        ("o25", "f_plus_g", "action_block", 4),
        ("o25", "f_plus_g", "horizon", 4),
        ("o25", "f_plus_g", "receding_horizon", 1),
        ("o25", "f_plus_g", "episode_budget", 49),
        (
            "o25",
            "f_plus_g",
            "executed_environment_steps_before_replanning",
            5,
        ),
        ("o50", "g_only", "horizon", 5),
        ("o100", "f_plus_g", "episode_budget", 100),
    ),
)
def test_rejects_each_changed_formal_cem_contract_field(
    tmp_path: Path,
    protocol: str,
    score_mode: str,
    key: str,
    changed: object,
) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    cell = _cell_directory(
        c4_root,
        protocol=protocol,
        variant="c4",
        score_mode=score_mode,
    )
    manifest_path = cell / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"]["planning"][key] = changed
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=rf"planning\.{key} must be"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)


def test_rejects_smoke_or_pilot_result_cell(tmp_path: Path) -> None:
    c4_root, v1_c_roots = _complete_inputs(tmp_path)
    cell = _cell_directory(
        c4_root,
        protocol="o50",
        variant="c4",
        score_mode="f_only",
    )
    result_path = cell / "results.json"
    result = json.loads(result_path.read_text())
    result["smoke"] = True
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=r"results\.json\.smoke must be False"):
        SUMMARY.build_summary(c4_root=c4_root, v1_c_roots=v1_c_roots)


@pytest.mark.parametrize(
    ("left_only", "right_only", "expected"),
    ((0, 0, 1.0), (0, 4, 0.125), (1, 3, 0.625), (2, 2, 1.0)),
)
def test_exact_two_sided_mcnemar(
    left_only: int, right_only: int, expected: float
) -> None:
    assert SUMMARY.exact_mcnemar_p_two_sided(left_only, right_only) == expected


@pytest.mark.parametrize(("left", "right"), ((-1, 0), (0, -1), (True, 0)))
def test_exact_mcnemar_rejects_invalid_counts(left: int, right: int) -> None:
    with pytest.raises(ValueError):
        SUMMARY.exact_mcnemar_p_two_sided(left, right)
