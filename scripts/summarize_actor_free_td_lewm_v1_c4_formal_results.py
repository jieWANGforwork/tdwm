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
from typing import Any, Union
from typing import Sequence as TypingSequence

from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    C4_ACTION_EFFECT,
    C4_JOINT_OBJECTIVE,
    C4_TIME_ALIGNMENT,
)
from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    OBJECTIVE_VERSION as C4_OBJECTIVE_VERSION,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c4 import (
    validate_actor_free_td_lewm_v1_c4_evaluation_protocol,
)

PROTOCOLS = ("o25", "o50", "o100")
SCORE_MODES = (
    "f_only",
    "g_only",
    "f_plus_g",
    "f_plus_g_first",
    "g_only_f_rollout_mean",
    "f_plus_g_first_q2",
)
NONBASELINE_SCORE_MODES = SCORE_MODES[1:]
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
EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL = {
    "o25": "56546fe8725ce0e4670f308c5b325bd64ff2a792373add8c20ddbcab02da6b37",
    "o50": "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7",
    "o100": "8a87815e8e1816ccb5021af81a5e2307a5b342d094eec3edf221a0e24851d10c",
}
EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL = {
    "o25": "72af45d4bad65a25288c5d405072d18ab5c0b4f0b67ddc970ac3f344b3c22fd9",
    "o50": "88c204770f33c0b0220057d45b187766e3cfc54912e3f5ca49f2aa93d16437e9",
    "o100": "36994b1ab36656666ff91b379a59829c4b2af150b1f4ed23d409deb5cca9654e",
}
EXPECTED_ACTION_NORMALIZATION_SHA256 = (
    "57f4d3c252e1805f4af1f614d20d1d1a064fa0d1d463ed5eb8ecf9dfc2b1a723"
)
SUMMARY_JSON_NAME = "actor_free_td_lewm_v1_c4_formal_summary.json"
EPISODE_CSV_NAME = "actor_free_td_lewm_v1_c4_formal_episode_matrix.csv"
SUMMARY_MARKDOWN_NAME = "actor_free_td_lewm_v1_c4_formal_summary.md"
RootInput = Union[str, Path, TypingSequence[Union[str, Path]]]


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


def _normalise_roots(roots: RootInput, *, label: str) -> tuple[Path, ...]:
    if isinstance(roots, (str, Path)):
        values: Sequence[str | Path] = (roots,)
    elif isinstance(roots, Sequence):
        values = roots
    else:
        raise TypeError(f"{label} must be one path or a sequence of paths.")
    resolved: list[Path] = []
    for value in values:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"Every {label} entry must be a path.")
        path = Path(value).expanduser().resolve()
        if path not in resolved:
            resolved.append(path)
    if not resolved:
        raise ValueError(f"At least one {label} path is required.")
    for path in resolved:
        if not path.is_dir():
            raise FileNotFoundError(path)
    return tuple(resolved)


def _candidate_cell_directories(
    root: Path,
    *,
    protocol: str,
    variant: str,
    score_mode: str,
) -> tuple[Path, ...]:
    bases = (
        root / "formal" / protocol / "v1" / variant,
        root / "formal" / "v1" / variant,
        root / protocol / "v1" / variant,
        root / protocol / variant,
        root / "v1" / variant,
        root / variant,
        root,
    )
    required_names = (
        "results.json",
        "protocol_manifest.json",
        "episode_selection.json",
    )
    matches: list[Path] = []
    candidates = [base.joinpath(*_cell_parts(score_mode)) for base in bases]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in matches:
            continue
        if all((resolved / name).is_file() for name in required_names):
            matches.append(resolved)
    return tuple(matches)


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


def _validate_formal_protocol(
    protocol_value: Mapping[str, Any],
    *,
    protocol: str,
    score_mode: str,
    label: str,
) -> None:
    """Validate the shared V1-C/C4 formal planning contract.

    Historical V1-C manifests and new C4 manifests use different method-
    specific fields, but both retain the same ``evaluation``, ``planning`` and
    ``inference_objective`` mappings.  Restrict this downstream audit to that
    stable common surface so old, genuine V1-C evidence remains readable while
    a run with changed CEM settings cannot be relabelled as formal.
    """

    evaluation = _required_mapping(
        protocol_value.get("evaluation"), label=f"{label}.evaluation"
    )
    expected_evaluation = {
        "episodes": EXPECTED_EPISODES,
        "goal_offset": GOAL_OFFSET_BY_PROTOCOL[protocol],
        "start_goal_source": "same_dataset_episode",
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"{label}.evaluation.{key} must be {expected!r}.")

    planning = _required_mapping(
        protocol_value.get("planning"), label=f"{label}.planning"
    )
    expected_horizon = 1 if score_mode == "g_only" else 5
    expected_receding_horizon = (
        1 if protocol in {"o50", "o100"} or score_mode == "g_only" else 5
    )
    expected_budget = {"o25": 50, "o50": 100, "o100": 200}[protocol]
    expected_planning: dict[str, Any] = {
        "solver": "CEM",
        "horizon": expected_horizon,
        "candidates": 300,
        "iterations": 30,
        "elites": 30,
        "action_block": 5,
        "frame_skip": 5,
        "history_len": 1,
        "receding_horizon": expected_receding_horizon,
        "episode_budget": expected_budget,
        "planning_seed": 42,
    }
    if protocol == "o25":
        expected_planning["executed_environment_steps_before_replanning"] = (
            expected_receding_horizon * 5
        )
    for key, expected in expected_planning.items():
        actual = planning.get(key)
        if actual != expected or (
            type(expected) is int and type(actual) is not int
        ):
            raise ValueError(f"{label}.planning.{key} must be {expected!r}.")

    inference = _required_mapping(
        protocol_value.get("inference_objective"),
        label=f"{label}.inference_objective",
    )
    if inference.get("score_mode") != score_mode:
        raise ValueError(
            f"{label}.inference_objective.score_mode must be {score_mode!r}."
        )
    if score_mode in {"f_plus_g_first", "f_plus_g_first_q2"}:
        try:
            weight = float(inference.get("g_first_weight"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label}.inference_objective.g_first_weight must be 0.25."
            ) from error
        if not math.isclose(weight, 0.25, abs_tol=1e-12):
            raise ValueError(
                f"{label}.inference_objective.g_first_weight must be 0.25."
            )


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
    selection_file_sha = _file_sha256(paths["selection"])
    expected_selection_sha = EXPECTED_SELECTION_FILE_SHA256_BY_PROTOCOL[protocol]
    if selection_file_sha != expected_selection_sha:
        raise ValueError(
            f"{paths['selection']} is not the locked {protocol.upper()} selection: "
            f"expected {expected_selection_sha}, found {selection_file_sha}."
        )
    ranks_sha = _canonical_json_sha256(selection["valid_row_ranks"])
    expected_ranks_sha = EXPECTED_SELECTION_RANKS_SHA256_BY_PROTOCOL[protocol]
    if ranks_sha != expected_ranks_sha:
        raise ValueError(
            f"{paths['selection']} has the wrong ordered valid-row ranks: "
            f"expected {expected_ranks_sha}, found {ranks_sha}."
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
    legacy_v1_c_o50 = method_key == "v1_c" and protocol == "o50"
    legacy_optional_top_level_fields = {
        "protocol_label",
        "evaluation_protocol",
        "goal_offset",
    }
    missing_results_metadata = {
        key for key in legacy_optional_top_level_fields if key not in results
    }
    missing_manifest_metadata = {
        key for key in legacy_optional_top_level_fields if key not in manifest
    }
    legacy_metadata_omitted = (
        legacy_v1_c_o50
        and missing_results_metadata == legacy_optional_top_level_fields
        and missing_manifest_metadata == legacy_optional_top_level_fields
    )
    if legacy_v1_c_o50 and (
        missing_results_metadata or missing_manifest_metadata
    ) and not legacy_metadata_omitted:
        raise ValueError(
            "Legacy V1-C O50 top-level protocol metadata must be either fully "
            "explicit or absent as one historical schema block."
        )
    for key, expected_value in expected.items():
        actual = results.get(key)
        if (
            legacy_metadata_omitted
            and key in legacy_optional_top_level_fields
        ):
            continue
        if actual != expected_value:
            raise ValueError(
                f"{paths['results']}.{key} must be {expected_value!r}, "
                f"found {actual!r}."
            )
    for key in (
        "protocol_label",
        "evaluation_protocol",
        "goal_offset",
        "score_mode",
    ):
        actual = manifest.get(key)
        if (
            legacy_metadata_omitted
            and key in legacy_optional_top_level_fields
        ):
            continue
        if actual != expected[key]:
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
    _validate_formal_protocol(
        manifest_protocol,
        protocol=protocol,
        score_mode=score_mode,
        label=f"{paths['manifest']}.protocol",
    )
    if method_key == "c4":
        for values, values_path in (
            (results, paths["results"]),
            (manifest, paths["manifest"]),
        ):
            for key, expected_value in {
                "objective_version": C4_OBJECTIVE_VERSION,
                "state_only_g": True,
                "action_enters_g": False,
                "action_effect": C4_ACTION_EFFECT,
                "g_state_source": "stopped_f_post_action_ghost_state",
            }.items():
                if values.get(key) != expected_value:
                    raise ValueError(
                        f"{values_path}.{key} must be {expected_value!r}."
                    )
        validate_actor_free_td_lewm_v1_c4_evaluation_protocol(manifest_protocol)
        formal_protocol = _required_mapping(
            manifest.get("formal_protocol"),
            label=f"{paths['manifest']}.formal_protocol",
        )
        validate_actor_free_td_lewm_v1_c4_evaluation_protocol(formal_protocol)
        score_definition = protocol_inference.get("score_definition")
        if (
            results.get("score_definition") != score_definition
            or manifest.get("score_definition") != score_definition
        ):
            raise ValueError(
                f"{paths['manifest']} changed the locked C4 score definition."
            )
    if manifest.get("selection") != selection_value:
        raise ValueError(f"{paths['manifest']} does not embed its selection file.")

    action_normalization_path = directory / "action_normalization.json"
    if not action_normalization_path.is_file():
        raise FileNotFoundError(
            f"{method_key}/{protocol}/{score_mode} has no action_normalization.json."
        )
    action_normalization_sha = _file_sha256(action_normalization_path)
    if action_normalization_sha != EXPECTED_ACTION_NORMALIZATION_SHA256:
        raise ValueError(
            f"{action_normalization_path} does not match the locked action "
            f"normalization: expected {EXPECTED_ACTION_NORMALIZATION_SHA256}, "
            f"found {action_normalization_sha}."
        )

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
    if method_key == "c4":
        if checkpoint.get("objective_version") != C4_OBJECTIVE_VERSION:
            raise ValueError(
                f"{paths['manifest']}.checkpoint.objective_version must be "
                f"{C4_OBJECTIVE_VERSION}."
            )
        g_config = _required_mapping(
            checkpoint.get("g_config"),
            label=f"{paths['manifest']}.checkpoint.g_config",
        )
        if (
            g_config.get("objective_version") != C4_OBJECTIVE_VERSION
            or g_config.get("action_effect") != C4_ACTION_EFFECT
            or g_config.get("time_alignment") != C4_TIME_ALIGNMENT
            or g_config.get("joint_objective") != C4_JOINT_OBJECTIVE
        ):
            raise ValueError(
                f"{paths['manifest']} checkpoint does not implement final C4 "
                "objective v1."
            )

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
            "sha256": selection_file_sha,
            "valid_row_ranks_sha256": ranks_sha,
        },
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha},
        "action_normalization": {
            "path": str(action_normalization_path.resolve()),
            "sha256": action_normalization_sha,
        },
        "top_level_protocol_metadata": (
            "legacy_v1_c_o50_validated_from_embedded_protocol"
            if legacy_metadata_omitted
            else "explicit"
        ),
    }
    return Cell(
        method_key=method_key,
        protocol=protocol,
        score_mode=score_mode,
        outcomes=outcomes,
        selection=selection,
        source=source,
    )


def _cell_content_signature(cell: Cell) -> tuple[Any, ...]:
    source = cell.source
    action = source.get("action_normalization")
    return (
        cell.outcomes,
        _canonical_json_sha256(cell.selection),
        source["results"]["sha256"],
        source["protocol_manifest"]["sha256"],
        source["episode_selection"]["sha256"],
        action["sha256"] if isinstance(action, Mapping) else None,
        source["checkpoint"]["sha256"],
    )


def _load_method(
    roots: RootInput,
    *,
    method_key: str,
    protocol: str,
) -> dict[str, Cell]:
    variant = str(METHODS[method_key]["variant"])
    source_roots = _normalise_roots(
        roots, label=f"{method_key}/{protocol} source root"
    )
    cells: dict[str, Cell] = {}
    used_roots: set[Path] = set()
    for score_mode in SCORE_MODES:
        directories: list[tuple[Path, Path]] = []
        for root in source_roots:
            for directory in _candidate_cell_directories(
                root,
                protocol=protocol,
                variant=variant,
                score_mode=score_mode,
            ):
                if all(existing != directory for _, existing in directories):
                    directories.append((root, directory))
        if not directories:
            raise FileNotFoundError(
                f"No {method_key}/{protocol}/{score_mode} result cell was found "
                f"under {[str(path) for path in source_roots]}."
            )
        loaded = [
            (
                root,
                _load_cell(
                    directory,
                    method_key=method_key,
                    protocol=protocol,
                    score_mode=score_mode,
                ),
            )
            for root, directory in directories
        ]
        signatures = {_cell_content_signature(cell) for _, cell in loaded}
        if len(signatures) != 1:
            raise ValueError(
                f"Conflicting duplicate {method_key}/{protocol}/{score_mode} "
                f"cells were found at {[cell.source['directory'] for _, cell in loaded]}."
            )
        selected_root, selected = loaded[0]
        used_roots.update(root for root, _ in loaded)
        if len(loaded) > 1:
            selected.source["equivalent_source_directories"] = [
                cell.source["directory"] for _, cell in loaded
            ]
        cells[score_mode] = selected
        used_roots.add(selected_root)
    unused_roots = [root for root in source_roots if root not in used_roots]
    if unused_roots:
        raise ValueError(
            f"These {method_key}/{protocol} source roots contributed no result "
            f"cells: {[str(path) for path in unused_roots]}."
        )
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
    c4_root: RootInput,
    v1_c_roots: Mapping[str, RootInput],
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
            "objective_version": C4_OBJECTIVE_VERSION,
            "training_objective": C4_JOINT_OBJECTIVE["objective"],
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
            "c4": [
                str(path)
                for path in _normalise_roots(c4_root, label="C4 source root")
            ],
            "v1_c": {
                protocol: [
                    str(path)
                    for path in _normalise_roots(
                        v1_c_roots[protocol],
                        label=f"V1-C {protocol.upper()} source root",
                    )
                ]
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
        for mode in NONBASELINE_SCORE_MODES:
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
        for mode in NONBASELINE_SCORE_MODES:
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
    parser.add_argument(
        "--c4-root",
        action="append",
        required=True,
        help="C4 matrix source root; repeat only if its cells are split.",
    )
    parser.add_argument(
        "--v1-c-o25-root",
        action="append",
        required=True,
        help="V1-C O25 source root; may be repeated for split cells.",
    )
    parser.add_argument(
        "--v1-c-o50-root",
        action="append",
        required=True,
        help="V1-C O50 source root; repeat for each historical source run.",
    )
    parser.add_argument(
        "--v1-c-o100-root",
        action="append",
        required=True,
        help="V1-C O100 source root; may be repeated for split cells.",
    )
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
