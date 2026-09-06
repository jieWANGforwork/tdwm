from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "scripts" / "summarize_actor_free_td_lewm_v1_c4_formal_results.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_v1_c4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


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
    manifest = {
        "protocol_label": protocol,
        "evaluation_protocol": protocol.upper(),
        "goal_offset": SUMMARY.GOAL_OFFSET_BY_PROTOCOL[protocol],
        "score_mode": score_mode,
        "selection": selection,
        "protocol": {
            "method": method["method"],
            "variant": method["variant"],
            "evaluation": {
                "goal_offset": SUMMARY.GOAL_OFFSET_BY_PROTOCOL[protocol]
            },
            "inference_objective": {"score_mode": score_mode},
        },
        "checkpoint": {
            "path": f"/server/{method_key}/epoch_10.pt",
            "sha256": "a" * 64 if method_key == "c4" else "b" * 64,
        },
    }
    (directory / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (directory / "protocol_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (directory / "episode_selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    (directory / "action_normalization.json").write_text(
        json.dumps({"protocol": protocol}), encoding="utf-8"
    )


def _write_method(
    root: Path,
    *,
    method_key: str,
    protocol: str,
    rank_offset: int = 0,
) -> None:
    variant = SUMMARY.METHODS[method_key]["variant"]
    selection = _selection(protocol, rank_offset=rank_offset)
    for score_mode in SUMMARY.SCORE_MODES:
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

    with pytest.raises(ValueError, match="between C4 and V1-C for O50"):
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
    with pytest.raises(FileNotFoundError, match="No complete six-cell"):
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
