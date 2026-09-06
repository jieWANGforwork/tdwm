#!/usr/bin/env python3
"""Summarize the paired V1-C4/V1-C formal Cube result matrix.

This script is deliberately downstream of evaluation.  It requires all six
predeclared score modes for O25, O50, and O100, reads exactly 50 Boolean
episode outcomes from every cell, and refuses to compare cells whose ordered
start-goal selections differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOLS = ("o25", "o50", "o100")
SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "g_only_f_rollout_mean",
    "f_plus_g_first_q2",
)
SCORE_LABELS = {
    "f_only": "F-only",
    "g_only": "G/C4-only",
    "f_plus_g": "F+G/C4 tail",
    "f_plus_g_first": "First-Q",
    "g_only_f_rollout_mean": "Mean-Q",
    "f_plus_g_first_q2": "First-Q2",
}
METHODS = {
    "c4": {
        "method": "actor_free_td_lewm_v1_c4",
        "variant": "c4",
        "label": "V1-C4",
    },
    "v1_c": {
        "method": "actor_free_td_lewm_v1_c",
        "variant": "c",
        "label": "V1-C",
    },
}
EXPECTED_EPISODES = 50
GOAL_OFFSET_BY_PROTOCOL = {"o25": 25, "o50": 50, "o100": 100}
SUMMARY_JSON_NAME = "actor_free_td_lewm_v1_c4_formal_summary.json"
EPISODE_CSV_NAME = "actor_free_td_lewm_v1_c4_formal_episode_matrix.csv"
SUMMARY_MARKDOWN_NAME = "actor_free_td_lewm_v1_c4_formal_summary.md"


@dataclass(frozen=True)
class Cell:
    method_key: str
    protocol: str
    score_mode: str
    outcomes: tuple[bool, ...]
    selection: dict[str, list[int]]
    source: dict[str, Any]

    @property
    def success_count(self) -> int:
        return sum(self.outcomes)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one JSON object.")
    return dict(value)


def _cell_parts(score_mode: str) -> tuple[str, ...]:
    if score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}:
        return score_mode, "alpha_0p25"
    return (score_mode,)


def _contains_all_cells(method_root: Path) -> bool:
    required_names = ("results.json", "protocol_manifest.json", "episode_selection.json")
    return all(
        all((method_root.joinpath(*_cell_parts(mode)) / name).is_file() for name in required_names)
        for mode in SCORE_MODES
    )


def _resolve_method_root(
    root: str | Path,
    *,
    protocol: str,
    variant: str,
) -> Path:
    supplied = Path(root).expanduser().resolve()
    candidates = (
        supplied / "formal" / protocol / "v1" / variant,
        supplied / protocol / "v1" / variant,
        supplied / "v1" / variant,
        supplied,
    )
    matches: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in matches and _contains_all_cells(resolved):
            matches.append(resolved)
    if not matches:
        expected = candidates[0]
        raise FileNotFoundError(
            f"No complete six-cell {protocol.upper()} V1-{variant.upper()} matrix "
            f"was found below {supplied}; expected a layout such as {expected}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"{supplied} resolves more than one possible {protocol.upper()} "
            f"V1-{variant.upper()} matrix: {[str(path) for path in matches]}."
        )
    return matches[0]


def _validate_selection(
    value: Mapping[str, Any],
    *,
    protocol: str,
    label: str,
) -> dict[str, list[int]]:
    selection: dict[str, list[int]] = {}
    for key in ("episode_indices", "start_steps", "goal_steps", "valid_row_ranks"):
        values = value.get(key)
        if (
            not isinstance(values, list)
            or len(values) != EXPECTED_EPISODES
            or any(type(item) is not int for item in values)
        ):
            raise ValueError(f"{label}.{key} must contain exactly 50 integers.")
        selection[key] = list(values)
    ranks = selection["valid_row_ranks"]
    if any(rank < 0 for rank in ranks) or len(set(ranks)) != EXPECTED_EPISODES:
        raise ValueError(f"{label}.valid_row_ranks must be unique and non-negative.")
    expected_offset = GOAL_OFFSET_BY_PROTOCOL[protocol]
    for start, goal in zip(selection["start_steps"], selection["goal_steps"]):
        if goal - start != expected_offset:
            raise ValueError(
                f"{label} does not encode exact {protocol.upper()} start-goal pairs."
            )
    return selection


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _optional_source(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _load_cell(
    directory: Path,
    *,
    method_key: str,
    protocol: str,
    score_mode: str,
) -> Cell:
    paths = {
        "results": directory / "results.json",
        "manifest": directory / "protocol_manifest.json",
        "selection": directory / "episode_selection.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{method_key}/{protocol}/{score_mode} is incomplete: {missing}."
        )
    results = _read_mapping(paths["results"])
    manifest = _read_mapping(paths["manifest"])
    selection_value = _read_mapping(paths["selection"])
    selection = _validate_selection(
        selection_value,
        protocol=protocol,
        label=f"{method_key}.{protocol}.{score_mode}.selection",
    )

    method = METHODS[method_key]
    expected = {
        "method": method["method"],
        "variant": method["variant"],
        "protocol_label": protocol,
        "evaluation_protocol": protocol.upper(),
        "goal_offset": GOAL_OFFSET_BY_PROTOCOL[protocol],
        "score_mode": score_mode,
        "smoke": False,
        "pilot": False,
    }
    for key, expected_value in expected.items():
        if results.get(key) != expected_value:
            raise ValueError(
                f"{paths['results']}.{key} must be {expected_value!r}, "
                f"found {results.get(key)!r}."
            )
    for key in (
        "protocol_label",
        "evaluation_protocol",
        "goal_offset",
        "score_mode",
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"{paths['manifest']}.{key} must be {expected[key]!r}."
            )
    manifest_protocol = _required_mapping(
        manifest.get("protocol"), label=f"{paths['manifest']}.protocol"
    )
    for key in ("method", "variant"):
        if manifest_protocol.get(key) != expected[key]:
            raise ValueError(
                f"{paths['manifest']}.protocol.{key} must be {expected[key]!r}."
            )
    protocol_evaluation = _required_mapping(
        manifest_protocol.get("evaluation"),
        label=f"{paths['manifest']}.protocol.evaluation",
    )
    protocol_inference = _required_mapping(
        manifest_protocol.get("inference_objective"),
        label=f"{paths['manifest']}.protocol.inference_objective",
    )
    if protocol_evaluation.get("goal_offset") != expected["goal_offset"]:
        raise ValueError(
            f"{paths['manifest']}.protocol.evaluation.goal_offset is incorrect."
        )
    if protocol_inference.get("score_mode") != expected["score_mode"]:
        raise ValueError(
            f"{paths['manifest']}.protocol.inference_objective.score_mode is "
            "incorrect."
        )
    if manifest.get("selection") != selection_value:
        raise ValueError(f"{paths['manifest']} does not embed its selection file.")

    metrics = _required_mapping(
        results.get("metrics"), label=f"{paths['results']}.metrics"
    )
    outcomes_value = metrics.get("episode_successes")
    if (
        not isinstance(outcomes_value, list)
        or len(outcomes_value) != EXPECTED_EPISODES
        or any(type(outcome) is not bool for outcome in outcomes_value)
    ):
        raise ValueError(
            f"{paths['results']}.metrics.episode_successes must contain exactly "
            "50 Boolean outcomes."
        )
    outcomes = tuple(outcomes_value)
    expected_rate = sum(outcomes) * 100.0 / EXPECTED_EPISODES
    try:
        recorded_rate = float(metrics["success_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{paths['results']} has no numeric success_rate.") from error
    if not math.isclose(recorded_rate, expected_rate, abs_tol=1e-12):
        raise ValueError(
            f"{paths['results']} success_rate disagrees with its Boolean outcomes."
        )

    checkpoint = _required_mapping(
        manifest.get("checkpoint"), label=f"{paths['manifest']}.checkpoint"
    )
    checkpoint_sha = checkpoint.get("sha256")
    checkpoint_path = checkpoint.get("path")
    if (
        not isinstance(checkpoint_sha, str)
        or len(checkpoint_sha) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha)
        or not isinstance(checkpoint_path, str)
        or not checkpoint_path
    ):
        raise ValueError(f"{paths['manifest']} has invalid checkpoint provenance.")

    source: dict[str, Any] = {
        "directory": str(directory.resolve()),
        "results": {
            "path": str(paths["results"].resolve()),
            "sha256": _file_sha256(paths["results"]),
        },
        "protocol_manifest": {
            "path": str(paths["manifest"].resolve()),
            "sha256": _file_sha256(paths["manifest"]),
        },
        "episode_selection": {
            "path": str(paths["selection"].resolve()),
            "sha256": _file_sha256(paths["selection"]),
            "valid_row_ranks_sha256": _canonical_json_sha256(
                selection["valid_row_ranks"]
            ),
        },
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha},
    }
    action_normalization = _optional_source(directory / "action_normalization.json")
    if action_normalization is not None:
        source["action_normalization"] = action_normalization
    return Cell(
        method_key=method_key,
        protocol=protocol,
        score_mode=score_mode,
        outcomes=outcomes,
        selection=selection,
        source=source,
    )


def _load_method(
    root: str | Path,
    *,
    method_key: str,
    protocol: str,
) -> dict[str, Cell]:
    variant = str(METHODS[method_key]["variant"])
    method_root = _resolve_method_root(root, protocol=protocol, variant=variant)
    cells = {
        score_mode: _load_cell(
            method_root.joinpath(*_cell_parts(score_mode)),
            method_key=method_key,
            protocol=protocol,
            score_mode=score_mode,
        )
        for score_mode in SCORE_MODES
    }
    reference = cells[SCORE_MODES[0]].selection
    for score_mode, cell in cells.items():
        if cell.selection != reference:
            raise ValueError(
                f"Selection mismatch inside {method_key}/{protocol}: "
                f"{score_mode} differs from {SCORE_MODES[0]}."
            )
    checkpoint_hashes = {cell.source["checkpoint"]["sha256"] for cell in cells.values()}
    if len(checkpoint_hashes) != 1:
        raise ValueError(f"{method_key}/{protocol} uses multiple checkpoints.")
    return cells


def exact_mcnemar_p_two_sided(left_only: int, right_only: int) -> float:
    """Return the exact two-sided McNemar/binomial p-value."""

    if (
        type(left_only) is not int
        or type(right_only) is not int
        or left_only < 0
        or right_only < 0
    ):
        raise ValueError("McNemar discordant counts must be non-negative integers.")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(
    reference: Sequence[bool],
    candidate: Sequence[bool],
) -> dict[str, Any]:
    if len(reference) != EXPECTED_EPISODES or len(candidate) != EXPECTED_EPISODES:
        raise ValueError("Paired comparisons require two 50-outcome vectors.")
    both_success = sum(left and right for left, right in zip(reference, candidate))
    new = sum(not left and right for left, right in zip(reference, candidate))
    lost = sum(left and not right for left, right in zip(reference, candidate))
    both_failure = EXPECTED_EPISODES - both_success - new - lost
    reference_successes = both_success + lost
    candidate_successes = both_success + new
    return {
        "reference_successes": reference_successes,
        "candidate_successes": candidate_successes,
        "both_success": both_success,
        "new": new,
        "lost": lost,
        "both_failure": both_failure,
        "delta_successes": candidate_successes - reference_successes,
        "delta_percentage_points": (
            (candidate_successes - reference_successes) * 100.0 / EXPECTED_EPISODES
        ),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p_two_sided(new, lost),
        "new_episode_positions": [
            index + 1
            for index, (left, right) in enumerate(
                zip(reference, candidate)
            )
            if not left and right
        ],
        "lost_episode_positions": [
            index + 1
            for index, (left, right) in enumerate(
                zip(reference, candidate)
            )
            if left and not right
        ],
    }


def _method_payload(cells: Mapping[str, Cell]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": cells[SCORE_MODES[0]].source["checkpoint"]["sha256"],
        "scores": {
            score_mode: {
                "success_count": cell.success_count,
                "success_rate_percent": (
                    cell.success_count * 100.0 / EXPECTED_EPISODES
                ),
                "episode_successes": list(cell.outcomes),
                "source": cell.source,
            }
            for score_mode, cell in cells.items()
        },
    }


def build_summary(
    *,
    c4_root: str | Path,
    v1_c_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    if set(v1_c_roots) != set(PROTOCOLS):
        raise ValueError("V1-C roots must be provided exactly for O25, O50, and O100.")

    protocols: dict[str, Any] = {}
    episode_matrix: list[dict[str, Any]] = []
    c4_checkpoint_hashes: set[str] = set()
    v1_c_checkpoint_hashes: set[str] = set()
    for protocol in PROTOCOLS:
        c4 = _load_method(c4_root, method_key="c4", protocol=protocol)
        v1_c = _load_method(
            v1_c_roots[protocol], method_key="v1_c", protocol=protocol
        )
        selection = c4[SCORE_MODES[0]].selection
        if v1_c[SCORE_MODES[0]].selection != selection:
            raise ValueError(
                f"Selection mismatch between C4 and V1-C for {protocol.upper()}."
            )

        c4_checkpoint_hashes.add(c4[SCORE_MODES[0]].source["checkpoint"]["sha256"])
        v1_c_checkpoint_hashes.add(
            v1_c[SCORE_MODES[0]].source["checkpoint"]["sha256"]
        )
        f_reference = c4["f_only"].outcomes
        versus_f_only: dict[str, Any] = {}
        versus_v1_c: dict[str, Any] = {}
        for score_mode in SCORE_MODES:
            f_comparison = _paired(f_reference, c4[score_mode].outcomes)
            f_comparison["f_plus_new_successes"] = (
                f_comparison["reference_successes"] + f_comparison["new"]
            )
            versus_f_only[score_mode] = f_comparison
            versus_v1_c[score_mode] = _paired(
                v1_c[score_mode].outcomes,
                c4[score_mode].outcomes,
            )

        ranks = selection["valid_row_ranks"]
        protocols[protocol] = {
            "selection": {
                **selection,
                "valid_row_ranks_sha256": _canonical_json_sha256(ranks),
                "c4_selection_file": c4["f_only"].source["episode_selection"],
                "v1_c_selection_file": v1_c["f_only"].source[
                    "episode_selection"
                ],
            },
            "methods": {
                "c4": _method_payload(c4),
                "v1_c": _method_payload(v1_c),
            },
            "comparisons": {
                "c4_vs_same_protocol_f_only": versus_f_only,
                "c4_vs_v1_c_same_score_mode": versus_v1_c,
            },
        }
        for position in range(EXPECTED_EPISODES):
            episode_matrix.append(
                {
                    "protocol": protocol,
                    "episode_position": position + 1,
                    "pair_id": f"{protocol.upper()}-P{position + 1:02d}",
                    "valid_row_rank": ranks[position],
                    "episode_index": selection["episode_indices"][position],
                    "start_step": selection["start_steps"][position],
                    "goal_step": selection["goal_steps"][position],
                    "c4": {
                        mode: c4[mode].outcomes[position] for mode in SCORE_MODES
                    },
                    "v1_c": {
                        mode: v1_c[mode].outcomes[position] for mode in SCORE_MODES
                    },
                }
            )

    if len(c4_checkpoint_hashes) != 1:
        raise ValueError("The three C4 protocols do not share one checkpoint.")
    if len(v1_c_checkpoint_hashes) != 1:
        raise ValueError("The three V1-C protocols do not share one checkpoint.")
    return {
        "schema_version": 1,
        "study": {
            "method": "actor_free_td_lewm_v1_c4",
            "comparison_method": "actor_free_td_lewm_v1_c",
            "training_seed": 3072,
            "protocols": list(PROTOCOLS),
            "score_modes": list(SCORE_MODES),
            "episodes_per_protocol": EXPECTED_EPISODES,
            "paired_comparison": True,
            "selection_equality_scope": (
                "exact ordered selection within each protocol; protocols are "
                "not compared to one another"
            ),
            "c4_checkpoint_sha256": next(iter(c4_checkpoint_hashes)),
            "v1_c_checkpoint_sha256": next(iter(v1_c_checkpoint_hashes)),
        },
        "input_roots": {
            "c4": str(Path(c4_root).expanduser().resolve()),
            "v1_c": {
                protocol: str(Path(v1_c_roots[protocol]).expanduser().resolve())
                for protocol in PROTOCOLS
            },
        },
        "protocols": protocols,
        "episode_matrix": episode_matrix,
    }


def _format_rate(count: int) -> str:
    return f"{count}/50 ({count * 2}%)"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# V1-C4 formal paired results",
        "",
        (
            "All cells contain 50 Boolean outcomes. Pairing is accepted only "
            "after exact ordered start-goal selection equality is verified "
            "within each protocol."
        ),
        "",
    ]
    protocols = _required_mapping(summary.get("protocols"), label="protocols")
    for protocol in PROTOCOLS:
        values = _required_mapping(protocols[protocol], label=protocol)
        methods = _required_mapping(values["methods"], label=f"{protocol}.methods")
        comparisons = _required_mapping(
            values["comparisons"], label=f"{protocol}.comparisons"
        )
        selection = _required_mapping(
            values["selection"], label=f"{protocol}.selection"
        )
        lines.extend(
            [
                f"## {protocol.upper()}",
                "",
                (
                    "Selection-ranks SHA-256: "
                    f"`{selection['valid_row_ranks_sha256']}`"
                ),
                "",
                "| Method | "
                + " | ".join(SCORE_LABELS[mode] for mode in SCORE_MODES)
                + " |",
                "|---|" + "---:|" * len(SCORE_MODES),
            ]
        )
        for method_key in ("v1_c", "c4"):
            method = _required_mapping(methods[method_key], label=method_key)
            scores = _required_mapping(method["scores"], label=f"{method_key}.scores")
            cells = [
                _format_rate(int(_required_mapping(scores[mode], label=mode)["success_count"]))
                for mode in SCORE_MODES
            ]
            lines.append(f"| {METHODS[method_key]['label']} | " + " | ".join(cells) + " |")

        lines.extend(
            [
                "",
                "### C4 relative to its same-protocol F-only baseline",
                "",
                "| Score | C4 success | New | Lost | F+New | Delta | Exact McNemar p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        versus_f = _required_mapping(
            comparisons["c4_vs_same_protocol_f_only"],
            label=f"{protocol}.versus_f",
        )
        for mode in SCORE_MODES:
            paired = _required_mapping(versus_f[mode], label=mode)
            lines.append(
                f"| {SCORE_LABELS[mode]} | "
                f"{_format_rate(int(paired['candidate_successes']))} | "
                f"{paired['new']} | {paired['lost']} | "
                f"{paired['f_plus_new_successes']} | "
                f"{int(paired['delta_successes']):+d} | "
                f"{float(paired['exact_mcnemar_p_two_sided']):.6g} |"
            )

        lines.extend(
            [
                "",
                "### C4 relative to V1-C under the same score mode",
                "",
                "| Score | V1-C | C4 | New | Lost | Delta | Exact McNemar p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        versus_c = _required_mapping(
            comparisons["c4_vs_v1_c_same_score_mode"],
            label=f"{protocol}.versus_c",
        )
        for mode in SCORE_MODES:
            paired = _required_mapping(versus_c[mode], label=mode)
            lines.append(
                f"| {SCORE_LABELS[mode]} | "
                f"{_format_rate(int(paired['reference_successes']))} | "
                f"{_format_rate(int(paired['candidate_successes']))} | "
                f"{paired['new']} | {paired['lost']} | "
                f"{int(paired['delta_successes']):+d} | "
                f"{float(paired['exact_mcnemar_p_two_sided']):.6g} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Audit note",
            "",
            (
                f"The full {len(PROTOCOLS) * EXPECTED_EPISODES}-row episode "
                f"matrix is stored in `{EPISODE_CSV_NAME}`. The JSON retains "
                "every Boolean outcome plus the absolute source paths and "
                "SHA-256 hashes of the result, protocol, selection, and "
                "available action-normalization files."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_episode_csv(summary: Mapping[str, Any]) -> str:
    rows = summary.get("episode_matrix")
    if not isinstance(rows, list) or len(rows) != len(PROTOCOLS) * EXPECTED_EPISODES:
        raise ValueError("episode_matrix must contain exactly 150 rows.")
    columns = [
        "protocol",
        "episode_position",
        "pair_id",
        "valid_row_rank",
        "episode_index",
        "start_step",
        "goal_step",
        *(f"c4__{mode}" for mode in SCORE_MODES),
        *(f"v1_c__{mode}" for mode in SCORE_MODES),
    ]
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        c4 = _required_mapping(row["c4"], label="episode.c4")
        v1_c = _required_mapping(row["v1_c"], label="episode.v1_c")
        flat = {key: row[key] for key in columns[:7]}
        flat.update({f"c4__{mode}": int(bool(c4[mode])) for mode in SCORE_MODES})
        flat.update(
            {f"v1_c__{mode}": int(bool(v1_c[mode])) for mode in SCORE_MODES}
        )
        writer.writerow(flat)
    return stream.getvalue()


def _write_identical_or_new(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to replace a different existing summary: {path}."
            )
        return
    if path.exists():
        raise FileExistsError(f"Summary output path is not a file: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir).expanduser().resolve()
    paths = {
        "json": directory / SUMMARY_JSON_NAME,
        "csv": directory / EPISODE_CSV_NAME,
        "markdown": directory / SUMMARY_MARKDOWN_NAME,
    }
    payloads = {
        "json": (
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
        "csv": render_episode_csv(summary).encode("utf-8"),
        "markdown": render_markdown(summary).encode("utf-8"),
    }
    for key, path in paths.items():
        _write_identical_or_new(path, payloads[key])
    return {key: str(path) for key, path in paths.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly summarize paired C4 and V1-C O25/O50/O100 formal results."
        )
    )
    parser.add_argument("--c4-root", required=True)
    parser.add_argument("--v1-c-o25-root", required=True)
    parser.add_argument("--v1-c-o50-root", required=True)
    parser.add_argument("--v1-c-o100-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(
        c4_root=args.c4_root,
        v1_c_roots={
            "o25": args.v1_c_o25_root,
            "o50": args.v1_c_o50_root,
            "o100": args.v1_c_o100_root,
        },
    )
    paths = write_outputs(summary, args.output_dir)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
