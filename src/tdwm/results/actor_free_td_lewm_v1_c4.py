"""Validated C4 extension for the canonical Results TD report.

The formal evaluator writes one summary that contains three protocols, six
predeclared score modes, and fifty Boolean outcomes per cell.  This module is
the only bridge from that evidence into the existing Results TD DOCX and
Markdown report.  It validates all 18 C4 cells (900 outcomes), the paired
comparisons, the frozen-training manifest, the loss CSV, and the deployment
checkpoint before constructing any document output.

``python-docx`` is imported lazily.  The document must be authored with the
bundled Codex workspace Python and must be rendered and visually reviewed
before the staged file replaces the canonical report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    C4_ACTION_EFFECT,
    C4_JOINT_OBJECTIVE,
    C4_TIME_ALIGNMENT,
    OBJECTIVE_VERSION,
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
    "g_only": "C4-only",
    "f_plus_g": "F+C4 tail",
    "f_plus_g_first": "First-Q",
    "g_only_f_rollout_mean": "Mean-Q",
    "f_plus_g_first_q2": "First-Q2",
}
LOSS_METRICS = (
    "vector_td_loss",
    "goal_projection_loss",
    "c4_total_loss",
)
EXPECTED_EPISODES = 50
EXPECTED_C4_CELLS = 18
EXPECTED_C4_OUTCOMES = 900
EXPECTED_EPOCHS = 10
EXPECTED_STEPS = 127_960
METHOD = "actor_free_td_lewm_v1_c4"
VARIANT = "c4"
SECTION_START = "<!-- RESULTS_TD_V1_C4_FORMAL_START -->"
SECTION_END = "<!-- RESULTS_TD_V1_C4_FORMAL_END -->"
DOCX_END_MARKER = "RESULTS TD / V1-C4 FORMAL EXTENSION END"
HISTORICAL_V0_SECTION_START = (
    "<!-- RESULTS_TD_V1_C4_OBJECTIVE_V0_HISTORY_START -->"
)
HISTORICAL_V0_SECTION_END = (
    "<!-- RESULTS_TD_V1_C4_OBJECTIVE_V0_HISTORY_END -->"
)
HISTORICAL_V0_DOCX_END_MARKER = (
    "RESULTS TD / V1-C4 OBJECTIVE-V0 HISTORICAL RECORD END"
)
HISTORICAL_V0_CHECKPOINT_SHA256 = (
    "28a59d0b07cb2e0ea66b34c57fdc1eb8dce513ca80b8a8700cc36ad9458ef99b"
)

V1_FIXED_COUNTS: dict[str, tuple[int | None, ...]] = {
    "C": (23, 18, 22, 28, 21, 26, None),
    "C2": (23, 18, 23, 26, 22, 26, None),
    "C3": (None, None, None, None, None, None, 26),
    "D": (23, 22, 21, 25, 26, None, None),
    "F": (23, 23, 24, 26, 26, None, None),
    "G1": (23, 21, 24, 26, 25, None, None),
    "G2": (23, 21, 25, 25, 24, None, None),
    "G3": (23, 19, 27, 26, 27, None, None),
}


class C4ResultsUpdateError(ValueError):
    """The supplied evidence or canonical report is incomplete or inconsistent."""


@dataclass(frozen=True)
class LossSeries:
    """One aggregate metric per zero-based training epoch."""

    values: tuple[float, ...]

    @property
    def first(self) -> float:
        return self.values[0]

    @property
    def final(self) -> float:
        return self.values[-1]


@dataclass(frozen=True)
class C4ReportEvidence:
    summary: dict[str, Any]
    summary_path: Path
    summary_sha256: str
    historical_v0_summary: dict[str, Any]
    historical_v0_summary_path: Path
    historical_v0_summary_sha256: str
    training_manifest: dict[str, Any]
    training_manifest_path: Path
    training_manifest_sha256: str
    metrics_path: Path
    metrics_sha256: str
    losses: dict[str, dict[str, LossSeries]]
    checkpoint_path: Path
    checkpoint_sha256: str
    loss_plot_path: Path | None = None
    loss_plot_sha256: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C4ResultsUpdateError(f"{label} must be a JSON object.")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise C4ResultsUpdateError(f"{label} is not valid JSON: {path}") from error
    return dict(_mapping(value, label))


def _validate_png(path: Path) -> None:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise C4ResultsUpdateError("--loss-plot has an invalid PNG signature")
    offset = 8
    seen_ihdr = False
    seen_idat = False
    seen_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise C4ResultsUpdateError("--loss-plot has a truncated PNG chunk")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise C4ResultsUpdateError("--loss-plot has a truncated PNG payload")
        recorded_crc = int.from_bytes(payload[data_end:crc_end], "big")
        computed_crc = zlib.crc32(chunk_type + payload[data_start:data_end]) & 0xFFFFFFFF
        if recorded_crc != computed_crc:
            raise C4ResultsUpdateError("--loss-plot has a PNG CRC mismatch")
        if not seen_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                raise C4ResultsUpdateError("--loss-plot must start with a 13-byte IHDR")
            width = int.from_bytes(payload[data_start : data_start + 4], "big")
            height = int.from_bytes(payload[data_start + 4 : data_start + 8], "big")
            if width <= 0 or height <= 0:
                raise C4ResultsUpdateError("--loss-plot has invalid PNG dimensions")
            seen_ihdr = True
        elif chunk_type == b"IHDR":
            raise C4ResultsUpdateError("--loss-plot contains multiple IHDR chunks")
        if chunk_type == b"IDAT":
            seen_idat = True
        if chunk_type == b"IEND":
            if length != 0 or crc_end != len(payload):
                raise C4ResultsUpdateError("--loss-plot has an invalid final IEND chunk")
            seen_iend = True
            break
        offset = crc_end
    if not (seen_ihdr and seen_idat and seen_iend):
        raise C4ResultsUpdateError("--loss-plot is missing IHDR, IDAT, or IEND")


def _paired(reference: Sequence[bool], candidate: Sequence[bool]) -> dict[str, int]:
    if len(reference) != EXPECTED_EPISODES or len(candidate) != EXPECTED_EPISODES:
        raise C4ResultsUpdateError("paired comparisons require two 50-outcome vectors")
    both_success = sum(left and right for left, right in zip(reference, candidate))
    new = sum((not left) and right for left, right in zip(reference, candidate))
    lost = sum(left and (not right) for left, right in zip(reference, candidate))
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
        "f_plus_new_successes": reference_successes + new,
    }


def _exact_mcnemar(new: int, lost: int) -> float:
    discordant = new + lost
    if discordant == 0:
        return 1.0
    lower = min(new, lost)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _outcomes(score: Mapping[str, Any], label: str) -> tuple[bool, ...]:
    values = score.get("episode_successes")
    if (
        not isinstance(values, list)
        or len(values) != EXPECTED_EPISODES
        or any(type(value) is not bool for value in values)
    ):
        raise C4ResultsUpdateError(f"{label} must contain exactly 50 Boolean outcomes")
    result = tuple(values)
    count = score.get("success_count")
    if type(count) is not int or count != sum(result):
        raise C4ResultsUpdateError(f"{label}.success_count disagrees with its outcomes")
    try:
        rate = float(score["success_rate_percent"])
    except (KeyError, TypeError, ValueError) as error:
        raise C4ResultsUpdateError(f"{label} has no numeric success rate") from error
    if not math.isclose(rate, count * 2.0, abs_tol=1e-12):
        raise C4ResultsUpdateError(f"{label}.success_rate_percent is inconsistent")
    return result


def _validate_comparison(
    recorded: Mapping[str, Any],
    reference: Sequence[bool],
    candidate: Sequence[bool],
    label: str,
    *,
    require_f_plus_new: bool,
) -> None:
    expected = _paired(reference, candidate)
    for key, value in expected.items():
        if key == "f_plus_new_successes" and not require_f_plus_new:
            continue
        if recorded.get(key) != value:
            raise C4ResultsUpdateError(
                f"{label}.{key} must be recomputed as {value}, found {recorded.get(key)!r}"
            )
    expected_new = [
        index + 1
        for index, (left, right) in enumerate(zip(reference, candidate))
        if not left and right
    ]
    expected_lost = [
        index + 1
        for index, (left, right) in enumerate(zip(reference, candidate))
        if left and not right
    ]
    if recorded.get("new_episode_positions") != expected_new:
        raise C4ResultsUpdateError(f"{label}.new_episode_positions is inconsistent")
    if recorded.get("lost_episode_positions") != expected_lost:
        raise C4ResultsUpdateError(f"{label}.lost_episode_positions is inconsistent")
    try:
        delta_pp = float(recorded["delta_percentage_points"])
        p_value = float(recorded["exact_mcnemar_p_two_sided"])
    except (KeyError, TypeError, ValueError) as error:
        raise C4ResultsUpdateError(f"{label} has invalid paired floating metrics") from error
    if not math.isclose(delta_pp, expected["delta_successes"] * 2.0, abs_tol=1e-12):
        raise C4ResultsUpdateError(f"{label}.delta_percentage_points is inconsistent")
    expected_p = _exact_mcnemar(expected["new"], expected["lost"])
    if not math.isclose(p_value, expected_p, abs_tol=1e-12):
        raise C4ResultsUpdateError(f"{label}.exact_mcnemar_p_two_sided is inconsistent")


def validate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete C4/V1-C formal summary and return a plain copy."""

    if summary.get("schema_version") != 1:
        raise C4ResultsUpdateError("C4 formal summary schema_version must be 1")
    study = _mapping(summary.get("study"), "study")
    required_study = {
        "method": METHOD,
        "objective_version": OBJECTIVE_VERSION,
        "training_objective": C4_JOINT_OBJECTIVE["objective"],
        "comparison_method": "actor_free_td_lewm_v1_c",
        "training_seed": 3072,
        "protocols": list(PROTOCOLS),
        "score_modes": list(SCORE_MODES),
        "episodes_per_protocol": EXPECTED_EPISODES,
        "paired_comparison": True,
    }
    for key, expected in required_study.items():
        if study.get(key) != expected:
            raise C4ResultsUpdateError(
                f"study.{key} must be {expected!r}, found {study.get(key)!r}"
            )
    c4_checkpoint_sha = study.get("c4_checkpoint_sha256")
    v1_c_checkpoint_sha = study.get("v1_c_checkpoint_sha256")
    if not _is_sha256(c4_checkpoint_sha) or not _is_sha256(v1_c_checkpoint_sha):
        raise C4ResultsUpdateError("study checkpoint SHA-256 values are invalid")

    protocols = _mapping(summary.get("protocols"), "protocols")
    if set(protocols) != set(PROTOCOLS):
        raise C4ResultsUpdateError("summary must contain exactly O25, O50, and O100")

    nested: dict[tuple[str, str, str], tuple[bool, ...]] = {}
    c4_outcome_count = 0
    c4_cell_count = 0
    for protocol in PROTOCOLS:
        protocol_value = _mapping(protocols[protocol], f"protocols.{protocol}")
        selection = _mapping(
            protocol_value.get("selection"), f"protocols.{protocol}.selection"
        )
        ranks = selection.get("valid_row_ranks")
        if (
            not isinstance(ranks, list)
            or len(ranks) != EXPECTED_EPISODES
            or any(type(rank) is not int for rank in ranks)
            or len(set(ranks)) != EXPECTED_EPISODES
        ):
            raise C4ResultsUpdateError(
                f"protocols.{protocol}.selection.valid_row_ranks must be 50 unique ints"
            )
        if not _is_sha256(selection.get("valid_row_ranks_sha256")):
            raise C4ResultsUpdateError(
                f"protocols.{protocol}.selection has no valid ranks SHA-256"
            )
        methods = _mapping(
            protocol_value.get("methods"), f"protocols.{protocol}.methods"
        )
        if set(methods) != {"c4", "v1_c"}:
            raise C4ResultsUpdateError(
                f"protocols.{protocol}.methods must contain exactly c4 and v1_c"
            )
        for method_key, expected_sha in (
            ("c4", c4_checkpoint_sha),
            ("v1_c", v1_c_checkpoint_sha),
        ):
            method = _mapping(methods[method_key], f"{protocol}.{method_key}")
            if method.get("checkpoint_sha256") != expected_sha:
                raise C4ResultsUpdateError(
                    f"{protocol}.{method_key} checkpoint differs from study"
                )
            scores = _mapping(method.get("scores"), f"{protocol}.{method_key}.scores")
            if set(scores) != set(SCORE_MODES):
                raise C4ResultsUpdateError(
                    f"{protocol}.{method_key} must contain exactly six score modes"
                )
            for mode in SCORE_MODES:
                score = _mapping(scores[mode], f"{protocol}.{method_key}.{mode}")
                values = _outcomes(score, f"{protocol}.{method_key}.{mode}")
                nested[(protocol, method_key, mode)] = values
                if method_key == "c4":
                    c4_cell_count += 1
                    c4_outcome_count += len(values)
                source = _mapping(score.get("source"), f"{protocol}.{method_key}.{mode}.source")
                checkpoint = _mapping(source.get("checkpoint"), f"{protocol}.{method_key}.{mode}.checkpoint")
                if checkpoint.get("sha256") != expected_sha:
                    raise C4ResultsUpdateError(
                        f"{protocol}.{method_key}.{mode} source checkpoint drifted"
                    )
                for source_key in ("results", "protocol_manifest", "episode_selection"):
                    source_item = _mapping(
                        source.get(source_key),
                        f"{protocol}.{method_key}.{mode}.source.{source_key}",
                    )
                    if not _is_sha256(source_item.get("sha256")):
                        raise C4ResultsUpdateError(
                            f"{protocol}.{method_key}.{mode} has invalid {source_key} SHA-256"
                        )

        comparisons = _mapping(
            protocol_value.get("comparisons"),
            f"protocols.{protocol}.comparisons",
        )
        versus_f = _mapping(
            comparisons.get("c4_vs_same_protocol_f_only"),
            f"{protocol}.c4_vs_same_protocol_f_only",
        )
        versus_c = _mapping(
            comparisons.get("c4_vs_v1_c_same_score_mode"),
            f"{protocol}.c4_vs_v1_c_same_score_mode",
        )
        if set(versus_f) != set(SCORE_MODES) or set(versus_c) != set(SCORE_MODES):
            raise C4ResultsUpdateError(f"{protocol} paired tables must cover all six modes")
        f_reference = nested[(protocol, "c4", "f_only")]
        if f_reference != nested[(protocol, "v1_c", "f_only")]:
            raise C4ResultsUpdateError(
                f"{protocol} C4 and V1-C F-only outcomes must be identical"
            )
        for mode in SCORE_MODES:
            _validate_comparison(
                _mapping(versus_f[mode], f"{protocol}.versus_f.{mode}"),
                f_reference,
                nested[(protocol, "c4", mode)],
                f"{protocol}.versus_f.{mode}",
                require_f_plus_new=True,
            )
            _validate_comparison(
                _mapping(versus_c[mode], f"{protocol}.versus_c.{mode}"),
                nested[(protocol, "v1_c", mode)],
                nested[(protocol, "c4", mode)],
                f"{protocol}.versus_c.{mode}",
                require_f_plus_new=False,
            )

    if c4_cell_count != EXPECTED_C4_CELLS or c4_outcome_count != EXPECTED_C4_OUTCOMES:
        raise C4ResultsUpdateError(
            f"C4 summary has {c4_cell_count} cells / {c4_outcome_count} outcomes, "
            "expected 18 / 900"
        )

    episode_matrix = summary.get("episode_matrix")
    if not isinstance(episode_matrix, list) or len(episode_matrix) != 150:
        raise C4ResultsUpdateError("episode_matrix must contain exactly 150 rows")
    seen: set[tuple[str, int]] = set()
    for row in episode_matrix:
        row = _mapping(row, "episode_matrix row")
        protocol = row.get("protocol")
        position = row.get("episode_position")
        if protocol not in PROTOCOLS or type(position) is not int or not 1 <= position <= 50:
            raise C4ResultsUpdateError("episode_matrix has an invalid protocol/position")
        key = (str(protocol), position)
        if key in seen:
            raise C4ResultsUpdateError(f"episode_matrix duplicates {key}")
        seen.add(key)
        for method_key in ("c4", "v1_c"):
            row_outcomes = _mapping(row.get(method_key), f"episode_matrix.{method_key}")
            if set(row_outcomes) != set(SCORE_MODES):
                raise C4ResultsUpdateError("episode_matrix row is missing score modes")
            for mode in SCORE_MODES:
                value = row_outcomes[mode]
                if type(value) is not bool:
                    raise C4ResultsUpdateError("episode_matrix outcomes must be Boolean")
                if value != nested[(str(protocol), method_key, mode)][position - 1]:
                    raise C4ResultsUpdateError(
                        f"episode_matrix disagrees with nested score at {key}/{method_key}/{mode}"
                    )
    if len(seen) != 150:
        raise C4ResultsUpdateError("episode_matrix does not cover all protocol positions")
    return dict(summary)


def validate_historical_v0_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact superseded, pre-versioned C4 objective-v0 summary.

    The historical bundle predates the explicit ``objective_version`` and
    ``training_objective`` study fields.  Its complete 18-cell/900-outcome
    schema is otherwise the same as the current formal summary.  Normalize a
    deep copy only long enough to reuse the strict structural, source-hash,
    episode-matrix, and paired-comparison validation; return the untouched
    historical payload so it cannot be mistaken for objective v1 evidence.
    """

    if summary.get("schema_version") != 1:
        raise C4ResultsUpdateError(
            "historical C4 objective-v0 summary schema_version must be 1"
        )
    study = _mapping(summary.get("study"), "historical_v0.study")
    if "objective_version" in study or "training_objective" in study:
        raise C4ResultsUpdateError(
            "historical C4 objective-v0 summary must use the exact pre-versioned study schema"
        )
    if study.get("c4_checkpoint_sha256") != HISTORICAL_V0_CHECKPOINT_SHA256:
        raise C4ResultsUpdateError(
            "historical C4 objective-v0 summary has an unexpected checkpoint SHA-256"
        )

    normalized = deepcopy(summary)
    normalized_study = dict(
        _mapping(normalized.get("study"), "historical_v0.study")
    )
    normalized_study["objective_version"] = OBJECTIVE_VERSION
    normalized_study["training_objective"] = C4_JOINT_OBJECTIVE["objective"]
    normalized["study"] = normalized_study
    validate_summary(normalized)
    return dict(summary)


def _validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    exact = {
        "method": METHOD,
        "variant": VARIANT,
        "objective_version": OBJECTIVE_VERSION,
        "seed": 3072,
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise C4ResultsUpdateError(f"training manifest {key} must be {value!r}")
    protocol = _mapping(manifest.get("protocol"), "training manifest protocol")
    if protocol.get("stage") != "full_training" or protocol.get("seeds") != [3072]:
        raise C4ResultsUpdateError("training manifest is not the formal seed-3072 run")
    if protocol.get("method") != METHOD or protocol.get("variant") != VARIANT:
        raise C4ResultsUpdateError("training protocol method/variant mismatch")
    g = _mapping(protocol.get("g"), "training protocol g")
    required_g = {
        "state_dim": 192,
        "task_dim": 192,
        "output_dim": 192,
        "action_input": "none",
        "action_effect": C4_ACTION_EFFECT,
        "successor_semantics": "includes_current_input_state",
        "actor": "none",
        "reward": "none",
    }
    for key, value in required_g.items():
        if g.get(key) != value:
            raise C4ResultsUpdateError(f"training protocol g.{key} must be {value!r}")
    alignment = _mapping(protocol.get("time_alignment"), "training time_alignment")
    if dict(alignment) != C4_TIME_ALIGNMENT:
        raise C4ResultsUpdateError(
            "training protocol time_alignment must encode the final post-action "
            "ghost objective"
        )
    objective = _mapping(protocol.get("joint_objective"), "training objective")
    if dict(objective) != C4_JOINT_OBJECTIVE:
        raise C4ResultsUpdateError(
            "training objective must be single_post_action_ghost_goal_projected_td"
        )
    training_protocol = _mapping(protocol.get("training"), "protocol.training")
    if training_protocol.get("epochs") != EXPECTED_EPOCHS:
        raise C4ResultsUpdateError("C4 training must use ten epochs")
    if training_protocol.get("optimizer_steps_per_epoch") != 12_796:
        raise C4ResultsUpdateError("C4 optimizer steps per epoch differ from V1-C")

    model = _mapping(manifest.get("model"), "training manifest model")
    if model.get("trainable_modules") != ["online_g_c4"]:
        raise C4ResultsUpdateError("only online G_C4 may be trainable")
    if model.get("optimizer_scope") != "exact_online_g_parameters_only":
        raise C4ResultsUpdateError("optimizer scope is not exact online G_C4")
    if model.get("trainable_lewm_parameters") != 0:
        raise C4ResultsUpdateError("training manifest reports trainable LeWM parameters")
    run = _mapping(manifest.get("training"), "training manifest training")
    required_run = {
        "formal_optimizer_steps": EXPECTED_STEPS,
        "configured_optimizer_steps": EXPECTED_STEPS,
        "epochs": EXPECTED_EPOCHS,
        "validation_skipped": False,
        "f_output_stop_gradient": True,
        "lewm_prediction_loss": False,
        "sigreg_loss": False,
        "loss_metrics": list(LOSS_METRICS),
    }
    for key, value in required_run.items():
        if run.get(key) != value:
            raise C4ResultsUpdateError(f"training manifest training.{key} must be {value!r}")


def _metric_column(fieldnames: Sequence[str], stage: str, metric: str) -> str:
    candidates = (
        f"{stage}/{metric}_epoch",
        f"{stage}/{metric}",
    )
    matches = [candidate for candidate in candidates if candidate in fieldnames]
    if not matches:
        raise C4ResultsUpdateError(
            f"metrics CSV has no aggregate column for {stage}/{metric}"
        )
    return matches[0]


def load_loss_series(path: Path) -> dict[str, dict[str, LossSeries]]:
    """Load exactly one finite aggregate per stage, metric, and epoch."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise C4ResultsUpdateError("metrics CSV must contain an epoch column")
        rows = list(reader)
        fields = tuple(reader.fieldnames)
    result: dict[str, dict[str, LossSeries]] = {}
    for stage in ("train", "validation"):
        result[stage] = {}
        for metric in LOSS_METRICS:
            column = _metric_column(fields, stage, metric)
            per_epoch: dict[int, list[float]] = {}
            for row in rows:
                raw = row.get(column, "")
                if raw in (None, ""):
                    continue
                try:
                    epoch = int(float(str(row["epoch"])))
                    value = float(raw)
                except (TypeError, ValueError) as error:
                    raise C4ResultsUpdateError(
                        f"metrics CSV has an invalid {column} aggregate"
                    ) from error
                if not math.isfinite(value):
                    raise C4ResultsUpdateError(f"metrics CSV {column} contains NaN/Inf")
                per_epoch.setdefault(epoch, []).append(value)
            if set(per_epoch) != set(range(EXPECTED_EPOCHS)):
                raise C4ResultsUpdateError(
                    f"{column} must cover epochs 0..9 exactly; found {sorted(per_epoch)}"
                )
            values: list[float] = []
            for epoch in range(EXPECTED_EPOCHS):
                epoch_values = per_epoch[epoch]
                if len(epoch_values) != 1:
                    raise C4ResultsUpdateError(
                        f"{column} epoch {epoch} has {len(epoch_values)} aggregates"
                    )
                values.append(epoch_values[0])
            result[stage][metric] = LossSeries(tuple(values))
        for epoch in range(EXPECTED_EPOCHS):
            components = sum(
                result[stage][metric].values[epoch]
                for metric in LOSS_METRICS[:-1]
            )
            total = result[stage]["c4_total_loss"].values[epoch]
            if not math.isclose(components, total, rel_tol=2e-4, abs_tol=2e-3):
                raise C4ResultsUpdateError(
                    f"{stage} epoch {epoch} C4 total is not vector TD plus goal loss"
                )
    return result


def load_report_evidence(
    *,
    summary_path: str | Path,
    historical_v0_summary_path: str | Path,
    training_manifest_path: str | Path,
    metrics_path: str | Path,
    checkpoint_path: str | Path,
    loss_plot_path: str | Path | None = None,
) -> C4ReportEvidence:
    """Validate every result/training input before document construction."""

    summary_file = Path(summary_path).expanduser().resolve()
    historical_v0_summary_file = (
        Path(historical_v0_summary_path).expanduser().resolve()
    )
    manifest_file = Path(training_manifest_path).expanduser().resolve()
    metrics_file = Path(metrics_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    summary = validate_summary(_load_json(summary_file, "C4 formal summary"))
    historical_v0_summary = validate_historical_v0_summary(
        _load_json(
            historical_v0_summary_file,
            "historical C4 objective-v0 formal summary",
        )
    )
    manifest = _load_json(manifest_file, "C4 training manifest")
    _validate_training_manifest(manifest)
    losses = load_loss_series(metrics_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    checkpoint_sha = _sha256(checkpoint_file)
    expected_sha = _mapping(summary["study"], "study")["c4_checkpoint_sha256"]
    if checkpoint_sha != expected_sha:
        raise C4ResultsUpdateError(
            "deployment checkpoint bytes differ from the checkpoint evaluated in the summary"
        )
    plot_file: Path | None = None
    plot_sha: str | None = None
    if loss_plot_path is not None:
        plot_file = Path(loss_plot_path).expanduser().resolve()
        if not plot_file.is_file() or plot_file.suffix.lower() != ".png":
            raise C4ResultsUpdateError("--loss-plot must name an existing PNG file")
        _validate_png(plot_file)
        plot_sha = _sha256(plot_file)
    return C4ReportEvidence(
        summary=summary,
        summary_path=summary_file,
        summary_sha256=_sha256(summary_file),
        historical_v0_summary=historical_v0_summary,
        historical_v0_summary_path=historical_v0_summary_file,
        historical_v0_summary_sha256=_sha256(historical_v0_summary_file),
        training_manifest=manifest,
        training_manifest_path=manifest_file,
        training_manifest_sha256=_sha256(manifest_file),
        metrics_path=metrics_file,
        metrics_sha256=_sha256(metrics_file),
        losses=losses,
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=checkpoint_sha,
        loss_plot_path=plot_file,
        loss_plot_sha256=plot_sha,
    )


def _score(evidence: C4ReportEvidence, protocol: str, mode: str) -> Mapping[str, Any]:
    return _mapping(
        evidence.summary["protocols"][protocol]["methods"]["c4"]["scores"][mode],
        f"{protocol}.c4.{mode}",
    )


def _historical_v0_score(
    evidence: C4ReportEvidence,
    protocol: str,
    mode: str,
) -> Mapping[str, Any]:
    return _mapping(
        evidence.historical_v0_summary["protocols"][protocol]["methods"]["c4"][
            "scores"
        ][mode],
        f"historical_v0.{protocol}.c4.{mode}",
    )


def _comparison(
    evidence: C4ReportEvidence, protocol: str, mode: str, key: str
) -> Mapping[str, Any]:
    return _mapping(
        evidence.summary["protocols"][protocol]["comparisons"][key][mode],
        f"{protocol}.{key}.{mode}",
    )


def _rate(count: int) -> str:
    return f"{count}/50 ({count * 2}%)"


def _best_modes(evidence: C4ReportEvidence, protocol: str) -> tuple[list[str], int]:
    counts = {mode: int(_score(evidence, protocol, mode)["success_count"]) for mode in SCORE_MODES}
    maximum = max(counts.values())
    return [SCORE_LABELS[mode] for mode, count in counts.items() if count == maximum], maximum


def _analysis_lines(evidence: C4ReportEvidence) -> list[str]:
    lines: list[str] = []
    for protocol in PROTOCOLS:
        modes, best = _best_modes(evidence, protocol)
        baseline = int(_score(evidence, protocol, "f_only")["success_count"])
        delta = best - baseline
        lines.append(
            f"{protocol.upper()}: best C4 score is {'/'.join(modes)} at {_rate(best)}, "
            f"{delta:+d}/50 ({delta * 2:+d} pp) versus its unchanged F-only baseline."
        )

    improved = tied = harmed = 0
    for protocol in PROTOCOLS:
        for mode in NONBASELINE_SCORE_MODES:
            delta = int(
                _comparison(
                    evidence, protocol, mode, "c4_vs_same_protocol_f_only"
                )["delta_successes"]
            )
            improved += delta > 0
            tied += delta == 0
            harmed += delta < 0
    lines.append(
        f"Across the 15 non-baseline protocol-score cells, C4 improves {improved}, "
        f"ties {tied}, and harms {harmed} relative to the same-protocol F-only outcome."
    )

    for protocol in PROTOCOLS:
        deltas = {
            mode: int(
                _comparison(
                    evidence, protocol, mode, "c4_vs_v1_c_same_score_mode"
                )["delta_successes"]
            )
            for mode in NONBASELINE_SCORE_MODES
        }
        maximum = max(deltas.values())
        minimum = min(deltas.values())
        best = "/".join(SCORE_LABELS[m] for m, value in deltas.items() if value == maximum)
        worst = "/".join(SCORE_LABELS[m] for m, value in deltas.items() if value == minimum)
        lines.append(
            f"Against V1-C under identical {protocol.upper()} scorers, C4's largest "
            f"change is {best} {maximum:+d}/50 and its smallest is {worst} {minimum:+d}/50."
        )

        higher = sum(value > 0 for value in deltas.values())
        equal = sum(value == 0 for value in deltas.values())
        lower = sum(value < 0 for value in deltas.values())
        state_readouts = ", ".join(
            f"{SCORE_LABELS[mode]} {deltas[mode]:+d}"
            for mode in ("g_only", "g_only_f_rollout_mean")
        )
        mixed_readouts = ", ".join(
            f"{SCORE_LABELS[mode]} {deltas[mode]:+d}"
            for mode in ("f_plus_g", "f_plus_g_first", "f_plus_g_first_q2")
        )
        lines.append(
            f"{protocol.upper()} scorer pattern for state-only/action-through-F C4 versus "
            f"V1-C: higher {higher}/5, tied {equal}/5, lower {lower}/5; state-focused "
            f"readouts [{state_readouts}], mixed F+C4 readouts [{mixed_readouts}] "
            "(all deltas are successes out of 50)."
        )

    lines.append(
        "C4 simultaneously changes the action route and successor time semantics, and "
        "replaces the action-conditioned G interface with a state-only G interface. "
        "Therefore the C4-versus-V1-C scorer pattern is descriptive and cannot isolate "
        "a causal effect of routing action through frozen F or of removing action from G "
        "by itself."
    )

    for protocol in PROTOCOLS:
        comparisons = {
            mode: _comparison(
                evidence, protocol, mode, "c4_vs_same_protocol_f_only"
            )
            for mode in NONBASELINE_SCORE_MODES
        }
        largest_union = max(
            int(paired["f_plus_new_successes"])
            for paired in comparisons.values()
        )
        union_details = []
        for mode, paired in comparisons.items():
            if int(paired["f_plus_new_successes"]) != largest_union:
                continue
            union_details.append(
                f"{SCORE_LABELS[mode]} {_rate(int(paired['candidate_successes']))}, "
                f"New {int(paired['new'])}, Lost {int(paired['lost'])}, "
                f"delta {int(paired['delta_successes']):+d}"
            )
        deployed_best = max(
            int(paired["delta_successes"])
            for paired in comparisons.values()
        )
        deployed_modes = "/".join(
            SCORE_LABELS[mode]
            for mode, paired in comparisons.items()
            if int(paired["delta_successes"]) == deployed_best
        )
        lines.append(
            f"{protocol.upper()} complementarity: the largest F+New oracle union is "
            f"{largest_union}/50, from {'; '.join(union_details)}. The largest deployed "
            f"delta is {deployed_modes} {deployed_best:+d}/50. F+New preserves F successes "
            "only by oracle construction; the deployable score still incurs every Lost case."
        )

    train = evidence.losses["train"]
    validation = evidence.losses["validation"]
    loss_scale_parts: list[str] = []
    for label, stage in (("train", train), ("validation", validation)):
        goal_sum = stage["goal_projection_loss"].final
        vector_sum = stage["vector_td_loss"].final
        loss_scale_parts.append(
            f"{label} goal/vector {goal_sum:.6g}/{vector_sum:.6g} "
            f"({goal_sum / max(vector_sum, 1e-12):.2f}x)"
        )
    lines.append(
        "At E10 with lambda_C=1, "
        + "; ".join(loss_scale_parts)
        + ". This raw-loss dominance measures optimization scale, not usefulness of the "
        "goal signal; it means representation conclusions are confounded by unequal "
        "component magnitudes until the loss scales are balanced."
    )

    q2_differences = []
    for protocol in PROTOCOLS:
        first = int(_score(evidence, protocol, "f_plus_g_first")["success_count"])
        first_q2 = int(
            _score(evidence, protocol, "f_plus_g_first_q2")["success_count"]
        )
        q2_differences.append(f"{protocol.upper()} {first_q2 - first:+d}/50")
    lines.append(
        "First-Q2 minus First-Q is "
        + ", ".join(q2_differences)
        + ". This quantifies sensitivity to F/Q scaling; it does not authorize choosing "
        "a scorer after seeing these formal cells."
    )

    lines.append(
        "Next predeclared experiment 1: keep C4 architecture, checkpoints, protocols, and "
        "six scorer definitions fixed; compare lambda_C or running-scale-normalized vector/goal "
        "losses chosen only on a disjoint development split, then run the locked choice once "
        "on each formal protocol."
    )
    lines.append(
        "Next predeclared experiment 2: fit F/Q calibration or an F-versus-C4 gate only on "
        "separate development pairs, freeze its rule and threshold before formal evaluation, "
        "and report its deployed result alongside New, Lost, and the non-deployable F+New "
        "oracle ceiling."
    )
    lines.append(
        "Confirmation target: repeat every locked comparison with multiple training seeds "
        "and planning seeds, reporting paired uncertainty separately for O25, O50, and O100 "
        "before making any overall superiority claim."
    )
    lines.append(
        "These are paired, single-training-seed and single-planning-seed results. "
        "O25, O50, and O100 use different goal offsets, so their percentages describe "
        "separate protocols and are not pooled as interchangeable episodes. No scorer is "
        "selected post hoc from these formal outcomes."
    )
    return lines


def _loss_change(series: LossSeries) -> str:
    if math.isclose(series.first, 0.0, abs_tol=1e-15):
        return "—"
    return f"{100.0 * (series.final / series.first - 1.0):+.1f}%"


def _loss_rows(evidence: C4ReportEvidence) -> list[tuple[str, str, str, str, str]]:
    labels = {
        "vector_td_loss": "Vector TD",
        "goal_projection_loss": "Goal projection",
        "c4_total_loss": "C4 total",
    }
    rows: list[tuple[str, str, str, str, str]] = []
    for stage in ("train", "validation"):
        for metric in LOSS_METRICS:
            series = evidence.losses[stage][metric]
            rows.append(
                (
                    stage.title(),
                    labels[metric],
                    f"{series.first:.6g}",
                    f"{series.final:.6g}",
                    _loss_change(series),
                )
            )
    return rows


def _loss_interpretation(evidence: C4ReportEvidence) -> str:
    observations: list[str] = []
    for stage in ("train", "validation"):
        losses = evidence.losses[stage]
        goal = losses["goal_projection_loss"].final
        vector = losses["vector_td_loss"].final
        dominant = "goal-projection" if goal > vector else "vector-TD"
        ratio = max(goal, vector) / max(min(goal, vector), 1e-12)
        observations.append(f"{stage} ends {dominant}-dominated ({ratio:.2f}x)")
    return (
        "; ".join(observations)
        + ". Absolute train/validation loss levels diagnose C4 optimization only; "
        "they are not directly comparable to the differently scaled C, C2, or C3 objectives."
    )


def _historical_v0_markdown_section(evidence: C4ReportEvidence) -> str:
    study = _mapping(evidence.historical_v0_summary["study"], "historical_v0.study")
    lines = [
        HISTORICAL_V0_SECTION_START,
        "## V1-C4 objective v0 historical record - superseded",
        "",
        (
            "This is the exact pre-versioned C4 run, retrospectively labelled objective v0. "
            "It used two aligned online branches, `x_real=z_i` and "
            "`x_pred=sg[F(z_(i-1),a_(i-1))]`, with the shared target "
            "`Y_i=sg[z_i+gamma(1-d_i)Gbar_C4(z_(i+1),m)]`. Its loss was "
            "`L_C4=0.5*((L_vector^real+L_goal^real)+"
            "(L_vector^pred+L_goal^pred))`, with lambda_C=1."
        ),
        "",
        (
            "Objective v0 is preserved only as historical evidence. It is superseded by "
            "the objective-v1 post-action-ghost formulation below and is excluded from "
            "the current 511-cell O50 ledger, 25,550-outcome total, master-table row, "
            "winner markers, and objective-v1 conclusions."
        ),
        "",
        "| Protocol | " + " | ".join(SCORE_LABELS[mode] for mode in SCORE_MODES) + " |",
        "|---|" + "---:|" * len(SCORE_MODES),
    ]
    for protocol in PROTOCOLS:
        cells = [
            _rate(
                int(
                    _historical_v0_score(evidence, protocol, mode)[
                        "success_count"
                    ]
                )
            )
            for mode in SCORE_MODES
        ]
        lines.append(f"| {protocol.upper()} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            f"- Historical summary SHA-256: `{evidence.historical_v0_summary_sha256}`",
            (
                "- Historical objective-v0 C4 checkpoint SHA-256: "
                f"`{study['c4_checkpoint_sha256']}`"
            ),
            f"- Historical evidence: `{evidence.historical_v0_summary_path}`",
            "- Historical coverage retained: 18 cells and 900 Boolean outcomes.",
            "",
            HISTORICAL_V0_SECTION_END,
            "",
        ]
    )
    return "\n".join(lines)


def _formal_markdown_section(evidence: C4ReportEvidence) -> str:
    lines = [
        SECTION_START,
        "## V1-C4 objective v1 formal O25 O50 O100 paired evaluation",
        "",
        (
            "C4 keeps the V1 LeWM observation encoder, Action Encoder and world-model "
            "predictor F frozen, stops every F output, and trains only a new online "
            "state-only G_C4 with a frozen EMA target. G_C4 has interface "
            "`G_C4(z_ghost,m)->Psi_i in R^192`; raw action and action embedding never enter G_C4."
        ),
        "",
        (
            "The single online input is `x_i=sg[F(z_i^real,a_i)]`. Its target is "
            "`Y_i=sg[z_(i+1)^real+gamma(1-d_i)Gbar_C4(sg[F(z_(i+1)^real,a_(i+1))],m)]`; "
            "when the transition after a_i terminates, `Y_i=z_(i+1)^real`. The loss is "
            "`L_C4=L_vector+L_goal`, where the full 192-D vector TD term uses every "
            "transition and the goal projection residual uses goal-derived samples only "
            "with lambda_C=1."
        ),
        "",
        "### Protocol by score matrix",
        "",
        "| Protocol | " + " | ".join(SCORE_LABELS[mode] for mode in SCORE_MODES) + " |",
        "|---|" + "---:|" * len(SCORE_MODES),
    ]
    for protocol in PROTOCOLS:
        cells = [
            _rate(int(_score(evidence, protocol, mode)["success_count"]))
            for mode in SCORE_MODES
        ]
        lines.append(f"| {protocol.upper()} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "C4-only first rolls the candidate action through F and evaluates "
                "`-G_C4(z1^F,m)^T m`. F+C4 tail rolls all five actions through F, "
                "then evaluates G_C4 at z5^F; the last action cannot bypass F. First-Q "
                "and First-Q2 read z1^F, while Mean-Q averages aligned state-only Q over "
                "z1^F...z5^F. The two first-action weights were fixed at alpha=0.25 before evaluation."
            ),
            "",
            "### Paired outcomes relative to same-protocol F-only",
            "",
            "| Protocol | Score | C4 result | New | Lost | F+New | Delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in PROTOCOLS:
        for mode in NONBASELINE_SCORE_MODES:
            paired = _comparison(
                evidence, protocol, mode, "c4_vs_same_protocol_f_only"
            )
            lines.append(
                f"| {protocol.upper()} | {SCORE_LABELS[mode]} | "
                f"{_rate(int(paired['candidate_successes']))} | {paired['new']} | "
                f"{paired['lost']} | {paired['f_plus_new_successes']} | "
                f"{int(paired['delta_successes']):+d} |"
            )
    lines.extend(
        [
            "",
            "### Paired outcomes relative to V1-C under the same score",
            "",
            "| Protocol | Score | V1-C | C4 | New | Lost | Delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in PROTOCOLS:
        for mode in NONBASELINE_SCORE_MODES:
            paired = _comparison(
                evidence, protocol, mode, "c4_vs_v1_c_same_score_mode"
            )
            lines.append(
                f"| {protocol.upper()} | {SCORE_LABELS[mode]} | "
                f"{_rate(int(paired['reference_successes']))} | "
                f"{_rate(int(paired['candidate_successes']))} | {paired['new']} | "
                f"{paired['lost']} | {int(paired['delta_successes']):+d} |"
            )
    train = evidence.losses["train"]
    validation = evidence.losses["validation"]
    lines.extend(
        [
            "",
            "### Training loss and evidence",
            "",
            "| Stage | Component | E1 | E10 | Change |",
            "|---|---|---:|---:|---:|",
            *(
                f"| {stage} | {metric} | {first} | {final} | {change} |"
                for stage, metric, first, final, change in _loss_rows(evidence)
            ),
            "",
            (
                "The ten-epoch formal run completed 127,960 optimizer updates. "
                f"Train C4 total changed from {train['c4_total_loss'].first:.6g} to "
                f"{train['c4_total_loss'].final:.6g}; validation C4 total changed from "
                f"{validation['c4_total_loss'].first:.6g} to "
                f"{validation['c4_total_loss'].final:.6g}. Final train components are "
                f"vector TD {train['vector_td_loss'].final:.6g} and goal projection "
                f"{train['goal_projection_loss'].final:.6g}."
            ),
            "",
            _loss_interpretation(evidence),
            "",
            f"- Formal summary SHA-256: `{evidence.summary_sha256}`",
            f"- Training manifest SHA-256: `{evidence.training_manifest_sha256}`",
            f"- Metrics CSV SHA-256: `{evidence.metrics_sha256}`",
            f"- C4 E10 deployment checkpoint SHA-256: `{evidence.checkpoint_sha256}`",
            f"- Checkpoint: `{evidence.checkpoint_path}`",
            "- Evidence coverage: 18 C4 cells and 900 C4 Boolean outcomes; no smoke or pilot cell is included.",
            "",
            "### Result analysis",
            "",
        ]
    )
    if evidence.loss_plot_path is not None:
        lines.insert(
            -3,
            f"- Loss plot: `{evidence.loss_plot_path}` (SHA-256 `{evidence.loss_plot_sha256}`)",
        )
    lines.extend(f"- {line}" for line in _analysis_lines(evidence))
    lines.extend(["", SECTION_END, ""])
    return "\n".join(lines)


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise C4ResultsUpdateError(
            f"expected one canonical Markdown anchor, found {count}: {old[:80]!r}"
        )
    return text.replace(old, new, 1)


def _v1_fixed_counts(
    evidence: C4ReportEvidence,
) -> dict[str, tuple[int | None, ...]]:
    counts = dict(V1_FIXED_COUNTS)
    counts["C4"] = tuple(
        int(_score(evidence, "o50", mode)["success_count"])
        for mode in SCORE_MODES
    ) + (None,)
    return counts


def _v1_column_winners(
    counts: Mapping[str, Sequence[int | None]],
) -> tuple[tuple[str, ...], ...]:
    winners: list[tuple[str, ...]] = []
    for column in range(7):
        available = {
            method: values[column]
            for method, values in counts.items()
            if values[column] is not None
        }
        maximum = max(int(value) for value in available.values())
        winners.append(
            tuple(
                method
                for method, value in available.items()
                if int(value) == maximum
            )
        )
    return tuple(winners)


def _rewrite_markdown_v1_fixed_markers(
    text: str,
    evidence: C4ReportEvidence,
) -> str:
    """Recompute fixed V1 column markers without touching alpha markers."""

    counts = _v1_fixed_counts(evidence)
    winners = _v1_column_winners(counts)
    alpha_cells = {("C", 3), ("C3", 3), ("C3", 5)}
    for method, values in counts.items():
        prefix = f"| V1 | {method} |"
        matches = [line for line in text.splitlines() if line.startswith(prefix)]
        if len(matches) != 1:
            raise C4ResultsUpdateError(
                f"Markdown V1 fixed row {method!r} matched {len(matches)} times"
            )
        original = matches[0]
        cells = original.split(" | ")
        if len(cells) != 10:
            raise C4ResultsUpdateError(
                f"Markdown V1 fixed row {method!r} is not a ten-column row"
            )
        for column, count in enumerate(values):
            if count is None or (method, column) in alpha_cells:
                continue
            cell_index = column + 3
            cells[cell_index] = re.sub(r"^◆\s*", "", cells[cell_index])
            if method in winners[column]:
                cells[cell_index] = "◆ " + cells[cell_index]
        text = _replace_once(text, original, " | ".join(cells))

    winner_cells: list[str] = []
    for column, methods in enumerate(winners):
        maximum = max(
            int(counts[method][column])
            for method in methods
            if counts[method][column] is not None
        )
        winner_cells.append(f"{'/'.join(methods)} {maximum}/50")
    replacement = "| V1 fixed | " + " | ".join(winner_cells) + " |"
    existing = [
        line for line in text.splitlines() if line.startswith("| V1 fixed |")
    ]
    if len(existing) != 1:
        raise C4ResultsUpdateError(
            f"Markdown V1 fixed winner row matched {len(existing)} times"
        )
    return _replace_once(text, existing[0], replacement)


def _rewrite_markdown_fixed_winner_summary(
    text: str,
    evidence: C4ReportEvidence,
) -> str:
    """Keep the prose summary consistent with the updated fixed O50 cells."""

    c4_counts = {
        mode: int(_score(evidence, "o50", mode)["success_count"])
        for mode in SCORE_MODES
    }
    tail_count = c4_counts["f_plus_g"]
    tail_maximum = max(27, tail_count)
    tail_names = ["V1-G3"] if tail_maximum == 27 else []
    if tail_count == tail_maximum:
        tail_names.append("V1-C4")
    tail_summary = f"{', '.join(tail_names)}: {_rate(tail_maximum)}"

    fixed_maximum = max(28, *c4_counts.values())
    fixed_names = ["V1-C + F + first-Q"] if fixed_maximum == 28 else []
    fixed_names.extend(
        f"V1-C4 + {SCORE_LABELS[mode]}"
        for mode, count in c4_counts.items()
        if count == fixed_maximum
    )
    fixed_summary = f"{', '.join(fixed_names)}: {_rate(fixed_maximum)}"

    text = _replace_once(
        text,
        "- **按原先固定的主评分列 F+G，描述性领先配置为 V1-G3: 27/50 (54%)。**",
        f"- **按原先固定的主评分列 F+G，描述性领先配置为 {tail_summary}。**",
    )
    text = _replace_once(
        text,
        "- **所有固定 E10 单格的最高结果为 V1-C + F + first-Q: 28/50 (56%)。**",
        f"- **所有固定 E10 单格的最高结果为 {fixed_summary}。**",
    )
    text = _replace_once(
        text,
        "- **若把五种评分等权平均，描述性领先训练配置为 V1-F, V1-G3（并列 48.8%）。**",
        "- **在原 477 格基础账的 24 个训练配置内，若把五种评分等权平均，描述性领先训练配置为 V1-F, V1-G3（并列 48.8%）。** C4 作为独立正式扩展在文末按相同 O50 评分逐格报告。",
    )
    text = _replace_once(
        text,
        "- **按六个训练方法 × 五种评分的版本均值，V1 action encoder 最高（47.3%）。**",
        "- **在原 477 格基础账的六方法 × 五评分版本均值中，V1 action encoder 最高（47.3%）。** C4 不回填改写该历史聚合口径。",
    )

    old_conclusion = (
        "不存在脱离测试评分定义的唯一训练赢家。按原研究固定的 F+G 主列，领先配置为 "
        "**V1-G3: 27/50 (54%)**；若把五种评分等权平均，则 **V1-F, V1-G3 并列领先（48.8%）**；"
        "固定评分中的最高单格为 **V1-C + F + first-Q: 28/50 (56%)**。"
    )
    new_conclusion = (
        "不存在脱离测试评分定义的唯一训练赢家。按原研究固定的 F+G 主列，领先配置为 "
        f"**{tail_summary}**；在原 477 格基础账的 24 个训练配置内，把五种评分等权平均，"
        "则 **V1-F, V1-G3 并列领先（48.8%）**；加入 C4 后固定评分中的最高单格为 "
        f"**{fixed_summary}**。"
    )
    return _replace_once(text, old_conclusion, new_conclusion)


def update_markdown_text(text: str, evidence: C4ReportEvidence) -> str:
    """Return the canonical Markdown report with one C4 extension."""

    if any(
        marker in text
        for marker in (
            SECTION_START,
            SECTION_END,
            HISTORICAL_V0_SECTION_START,
            HISTORICAL_V0_SECTION_END,
        )
    ):
        raise C4ResultsUpdateError("Markdown report already contains a C4 formal extension")
    text = _replace_once(
        text,
        "20 个 First-Q 权重扫描单元，共 505 格、25,250 个逐-pair outcome",
        "20 个 First-Q 权重扫描单元，以及 6 个 V1-C4 objective-v1 O50 单元，共 511 格、25,550 个逐-pair outcome",
    )
    text = _replace_once(
        text,
        "所以当前文档总覆盖为 505 格、25,250 个逐-pair 布尔结果。",
        "再加入 V1-C4 的 6 个正式 O50 单元后，当前 O50 总覆盖为 511 格、25,550 个逐-pair 布尔结果。",
    )
    coverage_anchor = (
        "| First-Q alpha sweep | V1-C / V1-C3 | C E10 / C3 E12 | "
        "5 original First-Q + 5 C3 Raw First-Q + 10 C3 Z-score First-Q2 | 20 |\n"
        "| **TOTAL** | — | — | same locked O50 selection | **505** |"
    )
    coverage_replacement = (
        "| First-Q alpha sweep | V1-C / V1-C3 | C E10 / C3 E12 | "
        "5 original First-Q + 5 C3 Raw First-Q + 10 C3 Z-score First-Q2 | 20 |\n"
        "| V1-C4 objective-v1 formal O50 | 1 | E10 | six predeclared state-only C4 scores | 6 |\n"
        "| **TOTAL** | — | — | same locked O50 selection | **511** |"
    )
    text = _replace_once(text, coverage_anchor, coverage_replacement)
    method_anchor = next(
        (line for line in text.splitlines() if line.startswith("| C3 (V1 only) |")),
        None,
    )
    if method_anchor is None:
        raise C4ResultsUpdateError("Markdown method table has no V1-C3 row")
    method_row = (
        "| C4 objective v1 (V1 only) | state-only G on stopped F(z_i,a_i) post-action ghost states | "
        "L_C4=L_vector+L_goal, lambda_C=1 | "
        "Freeze encoder, Action Encoder and F; action affects G_C4 only through the F-produced state |"
    )
    text = _replace_once(text, method_anchor, method_anchor + "\n" + method_row)
    text = _replace_once(text, "## 26 个训练方法 × 7 种评分", "## 27 个训练方法 × 7 种评分")
    text = _replace_once(text, "固定 26×7 结果", "固定 27×7 结果")

    c3_master = next(
        (line for line in text.splitlines() if line.startswith("| V1 | C3 |")),
        None,
    )
    if c3_master is None:
        raise C4ResultsUpdateError("Markdown master table has no V1-C3 row")
    o50_counts = {
        mode: int(_score(evidence, "o50", mode)["success_count"])
        for mode in SCORE_MODES
    }
    row_best = max(o50_counts.values())
    existing_v1 = _v1_fixed_counts(evidence)
    column_maxima = tuple(
        max(value[index] for value in existing_v1.values() if value[index] is not None)
        for index in range(7)
    )
    formatted: list[str] = []
    for index, mode in enumerate(SCORE_MODES):
        count = o50_counts[mode]
        value = _rate(count)
        row_winner = count == row_best
        column_winner = count == column_maxima[index]
        if row_winner:
            value = f"**{value}**"
        if column_winner:
            value = "◆ " + value
        formatted.append(value)
    formatted.append("—")
    c4_master = (
        "| V1 | C4 | L_C4=L_vector+L_goal | "
        + " | ".join(formatted)
        + " |"
    )
    text = _replace_once(text, c3_master, c3_master + "\n" + c4_master)
    text = _rewrite_markdown_v1_fixed_markers(text, evidence)
    text = _rewrite_markdown_fixed_winner_summary(text, evidence)

    text = _replace_once(
        text,
        "；总覆盖为 505 格、25,250 个逐-pair outcome。基础账 action normalization",
        "；原 505 格保持其既有指纹，加入 C4 O50 的 6 格后总覆盖为 511 格、25,550 个逐-pair outcome。基础账 action normalization",
    )
    text = _replace_once(
        text,
        "三部分合计 505 格 / 25,250 个 outcomes。",
        "三部分仍为原 505 格 / 25,250 个 outcomes；C4 另增加 6 个 O50 格 / 300 个 outcomes，总计 511 格 / 25,550 个 outcomes。",
    )
    text = (
        text.rstrip()
        + "\n\n"
        + _historical_v0_markdown_section(evidence)
        + "\n"
        + _formal_markdown_section(evidence)
    )
    return text


def _docx_imports() -> tuple[Any, ...]:
    try:
        from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Pt, RGBColor
        from docx.table import _Row
    except ImportError as error:
        raise RuntimeError(
            "python-docx is required; run this updater with the bundled Codex workspace Python"
        ) from error
    return (
        WD_CELL_VERTICAL_ALIGNMENT,
        WD_ALIGN_PARAGRAPH,
        OxmlElement,
        qn,
        Pt,
        RGBColor,
        _Row,
    )


def _shade_cell(cell: Any, fill: str | None) -> None:
    _, _, OxmlElement, qn, *_ = _docx_imports()
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if fill is None:
        if shading is not None:
            properties.remove(shading)
        return
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)


def _set_cell_border(cell: Any, *, edges: Sequence[str], color: str, size: int) -> None:
    _, _, OxmlElement, qn, *_ = _docx_imports()
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in edges:
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def _set_cell_text(
    cell: Any,
    text: str,
    *,
    size: float = 8.5,
    bold: bool = False,
    color: str = "111827",
    center: bool = False,
) -> None:
    _, WD_ALIGN_PARAGRAPH, _, qn, Pt, RGBColor, _ = _docx_imports()
    cell.text = ""
    paragraph = cell.paragraphs[0]
    if center:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Aptos")


def _find_paragraph(document: Any, prefix: str) -> Any:
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text.startswith(prefix)]
    if len(matches) != 1:
        raise C4ResultsUpdateError(
            f"canonical DOCX paragraph prefix {prefix!r} matched {len(matches)} times"
        )
    return matches[0]


def _replace_paragraph(document: Any, prefix: str, text: str) -> None:
    paragraph = _find_paragraph(document, prefix)
    if len(paragraph.runs) != 1:
        raise C4ResultsUpdateError(f"canonical paragraph {prefix!r} no longer has one run")
    paragraph.runs[0].text = text


def _edit_paragraph(
    document: Any,
    prefix: str,
    *,
    replacements: Sequence[tuple[str, str]] = (),
    append: str | None = None,
) -> None:
    """Apply narrow edits while retaining every unaffected source sentence."""

    paragraph = _find_paragraph(document, prefix)
    if len(paragraph.runs) != 1:
        raise C4ResultsUpdateError(f"canonical paragraph {prefix!r} no longer has one run")
    text = paragraph.runs[0].text
    for old, new in replacements:
        if text.count(old) != 1:
            raise C4ResultsUpdateError(
                f"canonical paragraph {prefix!r} does not contain one {old!r}"
            )
        text = text.replace(old, new, 1)
    if append:
        text = text.rstrip() + " " + append.strip()
    paragraph.runs[0].text = text


def _find_table_row(table: Any, key: str, value: str) -> Any:
    matches = [row for row in table.rows if row.cells[0].text == key and row.cells[1].text == value]
    if len(matches) != 1:
        raise C4ResultsUpdateError(f"DOCX table row ({key!r}, {value!r}) matched {len(matches)}")
    return matches[0]


def _find_label_row(table: Any, label: str) -> Any:
    matches = [row for row in table.rows if row.cells[0].text == label]
    if len(matches) != 1:
        raise C4ResultsUpdateError(f"DOCX table label {label!r} matched {len(matches)}")
    return matches[0]


def _insert_cloned_row_after(table: Any, anchor: Any, template: Any) -> Any:
    *_, _Row = _docx_imports()
    element = deepcopy(template._tr)
    anchor._tr.addnext(element)
    return _Row(element, table)


def _insert_cloned_row_before(table: Any, anchor: Any, template: Any) -> Any:
    *_, _Row = _docx_imports()
    element = deepcopy(template._tr)
    anchor._tr.addprevious(element)
    return _Row(element, table)


def _parse_count(text: str) -> int | None:
    match = re.fullmatch(r"\s*(\d+)/50(?:\s*\(\d+%\))?\s*", text)
    return int(match.group(1)) if match else None


def _update_master_docx(document: Any, evidence: C4ReportEvidence) -> None:
    table = document.tables[18]
    headers = [cell.text for cell in table.rows[0].cells]
    expected_headers = [
        "Version", "Method", "Training loss", "F-only", "G-only", "F+G tail",
        "First-Q", "Mean-Q", "First-Q2", "State-V",
    ]
    if headers != expected_headers or len(table.rows) != 28:
        raise C4ResultsUpdateError("DOCX table 18 is not the canonical 28x10 master table")
    if any(row.cells[1].text == "C4" for row in table.rows):
        raise C4ResultsUpdateError("DOCX master table already contains C4")
    c3 = _find_table_row(table, "V1", "C3")
    following_index = next(
        index for index, row in enumerate(table.rows) if row._tr is c3._tr
    ) + 1
    inserted = _insert_cloned_row_after(table, c3, table.rows[following_index])
    _set_cell_text(inserted.cells[1], "C4", bold=True)
    _set_cell_text(
        inserted.cells[2],
        "L_C4=L_vector+L_goal",
        size=7.5,
    )
    for column, mode in enumerate(SCORE_MODES, start=3):
        count = int(_score(evidence, "o50", mode)["success_count"])
        _set_cell_text(inserted.cells[column], _rate(count), center=True)
    _set_cell_text(inserted.cells[9], "—", center=True)
    _shade_cell(inserted.cells[1], "F8FAFC")
    _shade_cell(inserted.cells[2], "F8FAFC")

    v1_rows = [
        row
        for row in table.rows[1:]
        if row.cells[0].text == "V1" and row.cells[1].text in {"C", "C2", "C3", "C4", "D", "F", "G1", "G2", "G3"}
    ]
    if len(v1_rows) != 9:
        raise C4ResultsUpdateError("updated DOCX V1 fixed band must contain nine rows")
    counts = [[_parse_count(row.cells[column].text) for column in range(3, 10)] for row in v1_rows]
    column_maxima = [
        max(value[column] for value in counts if value[column] is not None)
        for column in range(7)
    ]
    for row, row_counts in zip(v1_rows, counts):
        available = [value for value in row_counts if value is not None]
        row_maximum = max(available) if available else None
        for index, count in enumerate(row_counts):
            cell = row.cells[index + 3]
            _shade_cell(cell, None)
            if count is None:
                continue
            row_best = count == row_maximum
            column_best = count == column_maxima[index]
            if row_best and column_best:
                _shade_cell(cell, "B7DEE8")
            elif column_best:
                _shade_cell(cell, "DDEBF7")
            elif row_best:
                _shade_cell(cell, "FFF2CC")
            for run in cell.paragraphs[0].runs:
                run.bold = row_best or column_best
    # Restore the explicit V1 band accent at its new lower edge.
    for row in v1_rows:
        for cell in row.cells:
            # Existing row borders remain authoritative; only C4's inserted top/bottom
            # borders are neutralized to avoid a false version boundary.
            if row.cells[1].text == "C4":
                _set_cell_border(cell, edges=("top", "bottom"), color="D9D9D9", size=4)


def _update_winner_docx(document: Any) -> None:
    master = document.tables[18]
    table = document.tables[19]
    v1_rows = [
        row for row in master.rows[1:]
        if row.cells[0].text == "V1" and row.cells[1].text in {"C", "C2", "C3", "C4", "D", "F", "G1", "G2", "G3"}
    ]
    target = _find_label_row(table, "V1")
    for score_index in range(7):
        available = [
            (row.cells[1].text, _parse_count(row.cells[score_index + 3].text))
            for row in v1_rows
        ]
        available = [(name, count) for name, count in available if count is not None]
        maximum = max(count for _, count in available)
        names = "/".join(name for name, count in available if count == maximum)
        _set_cell_text(target.cells[score_index + 1], f"{names} {maximum}/50", center=True)


def _update_global_fixed_conclusion_docx(
    document: Any,
    evidence: C4ReportEvidence,
) -> None:
    paragraph = _find_paragraph(
        document, "There is no evaluation-independent training winner."
    )
    if len(paragraph.runs) != 1 or "After integrating" not in paragraph.text:
        raise C4ResultsUpdateError("DOCX global-winner paragraph changed shape")
    suffix = "After integrating" + paragraph.text.split("After integrating", 1)[1]
    c4_counts = {
        mode: int(_score(evidence, "o50", mode)["success_count"])
        for mode in SCORE_MODES
    }
    tail_count = c4_counts["f_plus_g"]
    if tail_count > 27:
        tail_text = f"V1-C4 at {_rate(tail_count)}"
    elif tail_count == 27:
        tail_text = "V1-G3 and V1-C4 tied at 27/50 (54%)"
    else:
        tail_text = "V1-G3 at 27/50 (54%)"
    c4_maximum = max(c4_counts.values())
    c4_best = "/".join(
        SCORE_LABELS[mode]
        for mode, count in c4_counts.items()
        if count == c4_maximum
    )
    if c4_maximum > 28:
        fixed_text = f"V1-C4 {c4_best} at {_rate(c4_maximum)}"
    elif c4_maximum == 28:
        fixed_text = (
            f"V1-C First-Q and V1-C4 {c4_best} tied at 28/50 (56%)"
        )
    else:
        fixed_text = "V1-C First-Q at 28/50 (56%)"
    paragraph.runs[0].text = (
        "There is no evaluation-independent training winner. The prespecified "
        f"F+G leader is {tail_text}, and the highest fixed cell is {fixed_text}. "
        + suffix
    )


def _update_coverage_docx(document: Any) -> None:
    table = document.tables[24]
    if len(table.rows) != 9 or table.rows[-1].cells[-1].text != "505":
        raise C4ResultsUpdateError("DOCX table 24 is not the canonical 505-cell coverage table")
    total = table.rows[-1]
    inserted = _insert_cloned_row_before(table, total, table.rows[-2])
    values = (
        "V1-C4 objective-v1 formal O50",
        "1",
        "E10",
        "Six predeclared state-only C4 scores",
        "6",
    )
    for cell, value in zip(inserted.cells, values):
        _set_cell_text(cell, value, center=cell is not inserted.cells[0])
    _set_cell_text(total.cells[-1], "511", bold=True, center=True)


def _update_method_docx(document: Any) -> None:
    table = document.tables[25]
    if any(row.cells[0].text == "C4 objective v1 (V1 only)" for row in table.rows):
        raise C4ResultsUpdateError("DOCX method table already contains C4")
    row = table.add_row()
    values = (
        "C4 objective v1 (V1 only)",
        "x_i=sg[F(z_i^real,a_i)]; Y_i=sg[z_(i+1)^real+gamma(1-d_i)Gbar_C4(sg[F(z_(i+1)^real,a_(i+1))],m)]",
        "L_C4=L_vector+L_goal, lambda_C=1",
        "Freeze encoder, Action Encoder and F; action reaches state-only G_C4 only through F",
    )
    for cell, value in zip(row.cells, values):
        _set_cell_text(cell, value, size=8.0)
        _set_cell_border(cell, edges=("top", "bottom", "left", "right"), color="D9D9D9", size=4)


def _relative_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _update_audit_docx(document: Any, evidence: C4ReportEvidence, repository_root: Path) -> None:
    table = document.tables[39]
    _set_cell_text(
        _find_label_row(table, "Verified O50 cells").cells[1],
        "511 / 511 = original 505 audited cells + 6 formal V1-C4 O50 cells",
    )
    _set_cell_text(
        _find_label_row(table, "Per-pair Boolean outcomes").cells[1],
        "25,550 / 25,550 = original 25,250 outcomes + 300 V1-C4 O50 outcomes",
    )
    _set_cell_text(
        _find_label_row(table, "Grand total").cells[1],
        "511 O50 cells; 25,550 pair-level Boolean outcomes",
    )
    anchor = _find_label_row(table, "Alpha-sweep evidence")
    additions = (
        (
            "C4 objective-v1 formal summary",
            f"{_relative_path(evidence.summary_path, repository_root)}; SHA-256 {evidence.summary_sha256}",
        ),
        (
            "C4 objective-v1 O50 extension",
            "6 / 6 strict cells; 300 pair-level Boolean outcomes",
        ),
        (
            "C4 training evidence",
            f"manifest SHA-256 {evidence.training_manifest_sha256}; metrics SHA-256 {evidence.metrics_sha256}",
        ),
        ("C4 E10 checkpoint", evidence.checkpoint_sha256),
    )
    for label, value in reversed(additions):
        inserted = _insert_cloned_row_after(table, anchor, table.rows[-1])
        _set_cell_text(inserted.cells[0], label, bold=True)
        _set_cell_text(inserted.cells[1], value)


def _style_new_table(table: Any) -> None:
    _, WD_ALIGN_PARAGRAPH, OxmlElement, qn, Pt, RGBColor, _ = _docx_imports()
    table.style = "Table Grid"
    for row_index, row in enumerate(table.rows):
        row_properties = row._tr.get_or_add_trPr()
        cant_split = row_properties.find(qn("w:cantSplit"))
        if cant_split is None:
            cant_split = OxmlElement("w:cantSplit")
            row_properties.append(cant_split)
        if row_index == 0:
            repeat = row_properties.find(qn("w:tblHeader"))
            if repeat is None:
                repeat = OxmlElement("w:tblHeader")
                row_properties.append(repeat)
            repeat.set(qn("w:val"), "true")
        for cell in row.cells:
            _set_cell_border(cell, edges=("top", "bottom", "left", "right"), color="D9D9D9", size=4)
            cell.vertical_alignment = _docx_imports()[0].CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.name = "Aptos"
                    run.font.size = Pt(8.5)
        if row_index == 0:
            for cell in row.cells:
                _shade_cell(cell, "17365D")
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
        elif row_index % 2 == 0:
            for cell in row.cells:
                _shade_cell(cell, "F3F6FA")


def _add_docx_heading(document: Any, text: str, level: int, page_break: bool = False) -> Any:
    _, _, _, qn, Pt, RGBColor, _ = _docx_imports()
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.page_break_before = page_break
    run = paragraph.add_run(text)
    run.bold = True
    run.font.name = "Aptos Display"
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.size = Pt(16 if level == 1 else 12.5)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Aptos Display")
    return paragraph


def _add_docx_body(document: Any, text: str, *, bold: bool = False) -> Any:
    _, _, _, qn, Pt, RGBColor, _ = _docx_imports()
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor.from_string("111827")
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Aptos")
    return paragraph


def _start_labelled_docx_section(
    document: Any,
    *,
    header_text: str,
    footer_text: str,
) -> None:
    """Start a section whose header/footer cannot inherit an earlier label."""

    from docx.enum.section import WD_SECTION_START
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    section = document.add_section(WD_SECTION_START.NEW_PAGE)
    section.different_first_page_header_footer = True

    def write_container(container: Any, value: str, alignment: Any) -> None:
        container.is_linked_to_previous = False
        paragraph = container.paragraphs[0]
        paragraph.text = ""
        paragraph.alignment = alignment
        run = paragraph.add_run(value)
        run.font.name = "Aptos"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string("667085")
        run._element.get_or_add_rPr().get_or_add_rFonts().set(
            qn("w:eastAsia"), "Aptos"
        )

    # The canonical report enables distinct first/odd/even headers.  Populate
    # every variant in the new section so no C4 page can inherit an O100/C3
    # label merely because its physical page parity changes after pagination.
    for header in (
        section.header,
        section.first_page_header,
        section.even_page_header,
    ):
        write_container(
            header,
            header_text,
            WD_ALIGN_PARAGRAPH.LEFT,
        )
    for footer in (
        section.footer,
        section.first_page_footer,
        section.even_page_footer,
    ):
        write_container(
            footer,
            footer_text,
            WD_ALIGN_PARAGRAPH.RIGHT,
        )


def _start_c4_docx_section(document: Any) -> None:
    _start_labelled_docx_section(
        document,
        header_text=(
            "Results TD · V1-C4 objective v1 formal O25 O50 O100 paired evaluation"
        ),
        footer_text="Validated V1-C4 objective v1 paired outcomes",
    )


def _append_historical_v0_docx(
    document: Any,
    evidence: C4ReportEvidence,
    repository_root: Path,
) -> None:
    if any(
        paragraph.text == HISTORICAL_V0_DOCX_END_MARKER
        for paragraph in document.paragraphs
    ):
        raise C4ResultsUpdateError(
            "DOCX already contains a C4 objective-v0 historical record"
        )
    _start_labelled_docx_section(
        document,
        header_text="Results TD · V1-C4 objective v0 historical record",
        footer_text="Superseded V1-C4 objective v0 evidence",
    )
    _add_docx_heading(
        document,
        "V1 C4 objective v0 historical record superseded",
        1,
    )
    _add_docx_body(
        document,
        "This is the exact pre-versioned C4 run, retrospectively labelled objective v0. "
        "It used x_real = z_i and x_pred = stop-gradient F(z_(i-1),a_(i-1)), with "
        "Y_i = stop-gradient[z_i + gamma(1-d_i) Gbar_C4(z_(i+1),m)]. Its loss was "
        "L_C4 = 1/2[(L_vector^real + L_goal^real) + "
        "(L_vector^pred + L_goal^pred)], with lambda_C = 1.",
    )
    _add_docx_body(
        document,
        "Objective v0 is preserved only as historical evidence. It is superseded by "
        "the objective-v1 post-action-ghost formulation and is excluded from the "
        "current 511-cell O50 ledger, 25,550-outcome total, master-table row, winner "
        "markers, and objective-v1 conclusions.",
        bold=True,
    )
    table = document.add_table(rows=1, cols=7)
    headers = ["Protocol", *(SCORE_LABELS[mode] for mode in SCORE_MODES)]
    for cell, value in zip(table.rows[0].cells, headers):
        _set_cell_text(cell, value, bold=True, center=True)
    for protocol in PROTOCOLS:
        row = table.add_row()
        values = [
            protocol.upper(),
            *(
                _rate(
                    int(
                        _historical_v0_score(evidence, protocol, mode)[
                            "success_count"
                        ]
                    )
                )
                for mode in SCORE_MODES
            ),
        ]
        for cell, value in zip(row.cells, values):
            _set_cell_text(cell, value, center=True)
    _style_new_table(table)
    historical_study = _mapping(
        evidence.historical_v0_summary["study"],
        "historical_v0.study",
    )
    _add_docx_body(
        document,
        "Historical evidence: 18 cells / 900 Boolean outcomes; summary SHA-256 "
        f"{evidence.historical_v0_summary_sha256}; objective-v0 checkpoint SHA-256 "
        f"{historical_study['c4_checkpoint_sha256']}; source "
        f"{_relative_path(evidence.historical_v0_summary_path, repository_root)}.",
    )
    _add_docx_body(document, HISTORICAL_V0_DOCX_END_MARKER, bold=True)


def _append_formal_docx(document: Any, evidence: C4ReportEvidence) -> None:
    if any(paragraph.text == DOCX_END_MARKER for paragraph in document.paragraphs):
        raise C4ResultsUpdateError("DOCX already contains a C4 formal extension")
    _start_c4_docx_section(document)
    _add_docx_heading(
        document,
        "V1 C4 objective v1 formal O25 O50 O100 paired evaluation",
        1,
    )
    _add_docx_body(
        document,
        "C4 freezes the V1 observation encoder, Action Encoder and world-model predictor F; every F output is stopped. Only online state-only G_C4 is optimized, with a frozen EMA target. G_C4 accepts state and task only and returns a 192-dimensional successor vector. Raw action and action embedding never enter G_C4.",
        bold=True,
    )
    _add_docx_body(
        document,
        "Single online input: x_i = stopgrad[F(z_i^real,a_i)]. Target: Y_i = stopgrad[z_(i+1)^real + gamma(1-d_i) Gbar_C4(stopgrad[F(z_(i+1)^real,a_(i+1))],m)]; if the transition after a_i terminates, Y_i = z_(i+1)^real. Loss: L_C4 = L_vector + L_goal, with lambda_C = 1 and goal loss only on goal-derived samples.",
    )
    _add_docx_heading(document, "Protocol by score matrix", 2)
    table = document.add_table(rows=1, cols=7)
    headers = ["Protocol", *(SCORE_LABELS[mode] for mode in SCORE_MODES)]
    for cell, value in zip(table.rows[0].cells, headers):
        _set_cell_text(cell, value, bold=True, center=True)
    for protocol in PROTOCOLS:
        row = table.add_row()
        values = [
            protocol.upper(),
            *(
                _rate(int(_score(evidence, protocol, mode)["success_count"]))
                for mode in SCORE_MODES
            ),
        ]
        for cell, value in zip(row.cells, values):
            _set_cell_text(cell, value, center=True)
    _style_new_table(table)
    _add_docx_body(
        document,
        "C4-only first sends the candidate action through F and reads G_C4 at z1^F. F+C4 tail sends the fifth action through F and reads G_C4 at z5^F; no action bypass exists. First-Q and First-Q2 read z1^F, and Mean-Q averages the aligned state-only readout over z1^F through z5^F. First-Q and First-Q2 use the preregistered alpha = 0.25.",
    )
    _add_docx_heading(document, "Paired outcomes relative to same protocol F only", 2)
    paired_table = document.add_table(rows=1, cols=7)
    paired_headers = ("Protocol", "Score", "C4 result", "New", "Lost", "F+New", "Delta")
    for cell, value in zip(paired_table.rows[0].cells, paired_headers):
        _set_cell_text(cell, value, bold=True, center=True)
    for protocol in PROTOCOLS:
        for mode in NONBASELINE_SCORE_MODES:
            paired = _comparison(
                evidence, protocol, mode, "c4_vs_same_protocol_f_only"
            )
            row = paired_table.add_row()
            values = (
                protocol.upper(),
                SCORE_LABELS[mode],
                _rate(int(paired["candidate_successes"])),
                str(paired["new"]),
                str(paired["lost"]),
                str(paired["f_plus_new_successes"]),
                f"{int(paired['delta_successes']):+d}",
            )
            for cell, value in zip(row.cells, values):
                _set_cell_text(cell, value, center=True)
    _style_new_table(paired_table)
    _start_c4_docx_section(document)
    _add_docx_heading(
        document,
        "Paired outcomes relative to V1 C under the same score",
        2,
    )
    v1_c_table = document.add_table(rows=1, cols=7)
    for cell, value in zip(
        v1_c_table.rows[0].cells,
        ("Protocol", "Score", "V1-C", "C4", "New", "Lost", "Delta"),
    ):
        _set_cell_text(cell, value, bold=True, center=True)
    for protocol in PROTOCOLS:
        for mode in NONBASELINE_SCORE_MODES:
            paired = _comparison(
                evidence, protocol, mode, "c4_vs_v1_c_same_score_mode"
            )
            row = v1_c_table.add_row()
            values = (
                protocol.upper(),
                SCORE_LABELS[mode],
                _rate(int(paired["reference_successes"])),
                _rate(int(paired["candidate_successes"])),
                str(paired["new"]),
                str(paired["lost"]),
                f"{int(paired['delta_successes']):+d}",
            )
            for cell, value in zip(row.cells, values):
                _set_cell_text(cell, value, center=True)
    _style_new_table(v1_c_table)
    _start_c4_docx_section(document)
    _add_docx_heading(document, "Training loss and evidence", 2)
    train = evidence.losses["train"]
    validation = evidence.losses["validation"]
    loss_table = document.add_table(rows=1, cols=5)
    for cell, value in zip(
        loss_table.rows[0].cells,
        ("Stage", "Component", "E1", "E10", "Change"),
    ):
        _set_cell_text(cell, value, bold=True, center=True)
    for values in _loss_rows(evidence):
        row = loss_table.add_row()
        for cell, value in zip(row.cells, values):
            _set_cell_text(cell, value, center=True)
    _style_new_table(loss_table)
    _add_docx_body(
        document,
        "The formal ten-epoch run completed 127,960 optimizer updates. "
        f"Train total changed {train['c4_total_loss'].first:.6g} -> {train['c4_total_loss'].final:.6g}; "
        f"validation total changed {validation['c4_total_loss'].first:.6g} -> {validation['c4_total_loss'].final:.6g}. "
        f"Final train components: vector TD {train['vector_td_loss'].final:.6g}, "
        f"goal projection {train['goal_projection_loss'].final:.6g}.",
    )
    _add_docx_body(document, _loss_interpretation(evidence), bold=True)
    if evidence.loss_plot_path is not None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches

        figure = document.add_paragraph()
        figure.alignment = WD_ALIGN_PARAGRAPH.CENTER
        figure.add_run().add_picture(str(evidence.loss_plot_path), width=Inches(8.7))
        caption = _add_docx_body(
            document,
            "Figure. V1-C4 E1-E10 train and validation total and component losses.",
        )
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_docx_body(
        document,
        f"Evidence: 18 formal C4 cells / 900 Boolean outcomes; summary SHA-256 {evidence.summary_sha256}; training manifest SHA-256 {evidence.training_manifest_sha256}; metrics SHA-256 {evidence.metrics_sha256}; E10 checkpoint SHA-256 {evidence.checkpoint_sha256}.",
    )
    _add_docx_heading(document, "Result analysis", 2)
    for line in _analysis_lines(evidence):
        paragraph = _add_docx_body(document, line)
        paragraph.style = "List Bullet"
    _add_docx_body(document, DOCX_END_MARKER, bold=True)


def update_docx_document(document: Any, evidence: C4ReportEvidence, repository_root: Path) -> None:
    """Mutate a validated canonical Results TD Document object in memory."""

    existing_markers = {
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text
        in {DOCX_END_MARKER, HISTORICAL_V0_DOCX_END_MARKER}
    }
    if existing_markers:
        raise C4ResultsUpdateError(
            "DOCX already contains a C4 formal or historical extension"
        )
    if len(document.tables) != 46 or len(document.paragraphs) != 276:
        raise C4ResultsUpdateError(
            "DOCX is not the canonical pre-C4 Results TD artifact (expected 46 tables / 276 paragraphs)"
        )
    _edit_paragraph(
        document,
        "Complete 26-method",
        replacements=(("Complete 26-method", "Complete 27-method"),),
    )
    _replace_paragraph(
        document,
        "Cube · seed 3072 · 477 base",
        "Cube · seed 3072 · 477 base + 8 endpoints + 20 alpha-sweep + 6 C4 cells = 511 O50 cells",
    )
    _edit_paragraph(
        document,
        "The preceding pages preserve",
        replacements=(
            (
                "and all 20 First-Q alpha-sweep cells in one master matrix.",
                "all 20 First-Q alpha-sweep cells, and six formal V1-C4 O50 cells in one master matrix.",
            ),
            (
                "Every result uses the same 50 start-goal pairs; the three audited components retain 25,250 Boolean outcomes.",
                "Every O50 result uses the same 50 start-goal pairs; the original three audited components retain 25,250 Boolean outcomes, and the six V1-C4 O50 cells raise that ledger to 25,550.",
            ),
        ),
    )
    _edit_paragraph(
        document,
        "The original fixed-E10 analysis remains unchanged",
        replacements=(
            (
                "the complete document covers 505 O50 cells and 25,250 pair-level outcomes.",
                "the pre-C4 document covers 505 O50 cells and 25,250 pair-level outcomes; six formal V1-C4 O50 cells extend the ledger to 511 cells and 25,550 pair-level outcomes.",
            ),
        ),
    )
    _edit_paragraph(
        document,
        "This section defines the seven score families",
        replacements=(("26-method matrix", "27-method matrix"),),
        append=(
            "For state-only C4, m uses that same normalized goal representation and "
            "q_C4(z,m)=G_C4(z,m)^T m; candidate actions can reach G_C4 only after "
            "they have changed an F-produced imagined state."
        ),
    )
    _edit_paragraph(
        document,
        "Action input differs only by training version",
        replacements=(("differs only", "differs"),),
        append=(
            "V1-C4 is stricter: state-only G_C4 never sees raw action or action "
            "embedding; a candidate action affects its score only through an "
            "F-produced imagined state."
        ),
    )
    _replace_paragraph(
        document,
        "26-method, seven-score master comparison",
        "27-method, seven-score master comparison with the integrated alpha sweep",
    )
    _edit_paragraph(
        document,
        "Read across a row to compare",
        replacements=(
            ("V1 includes C2/C3 beside C", "V1 includes C2/C3/C4 beside C"),
            (
                "Yellow marks a row best, blue a sweep-column best, and teal both; all ties are retained. Fixed-score and exploratory-sweep winners are kept as separate comparison scopes.",
                "For the fixed 27-method cells, yellow marks a row best, blue a within-version column best, and teal both; all ties are retained. The dedicated alpha submatrix keeps its separate exploratory winner scope and is excluded from the fixed winner recomputation.",
            ),
        ),
    )
    _edit_paragraph(
        document,
        "The Training loss column shows",
        append=(
            "C4 is the only V1 row with a single state-only online branch on "
            "the stopped post-action ghost state x_i = F(z_i,a_i); its training "
            "loss is L_vector + L_goal."
        ),
    )
    _edit_paragraph(
        document,
        "V1-C2 initializes every parameter",
        append=(
            "V1-C4 instead starts a new state-only G_C4 over the same frozen V1 "
            "LeWM: x_i = stop-gradient F(z_i^real,a_i), with target "
            "Y_i = stop-gradient[z_(i+1)^real + gamma(1-d_i)Gbar_C4("
            "stop-gradient F(z_(i+1)^real,a_(i+1)),m)]; only online G_C4 is optimized."
        ),
    )
    _replace_paragraph(
        document,
        "One 26 by 7 master matrix",
        "One 27 by 7 master matrix with all 20 alpha-sweep cells",
    )
    _update_master_docx(document, evidence)
    _update_winner_docx(document)
    _update_global_fixed_conclusion_docx(document, evidence)
    _update_coverage_docx(document)
    _update_method_docx(document)
    _update_audit_docx(document, evidence, repository_root)
    _append_historical_v0_docx(document, evidence, repository_root)
    _append_formal_docx(document, evidence)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_updated_reports(
    *,
    docx_path: str | Path,
    markdown_path: str | Path,
    evidence: C4ReportEvidence,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write staged same-name outputs, or atomically replace the two inputs.

    Validation is complete before this function is called.  When ``output_dir``
    is supplied, the output names are exactly the source basenames; this avoids
    creating a competing Results TD document name.  Without it, temporary
    siblings are constructed and atomically promoted in place.
    """

    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError(
            "python-docx is required; use the bundled Codex workspace Python"
        ) from error
    source_docx = Path(docx_path).expanduser().resolve()
    source_markdown = Path(markdown_path).expanduser().resolve()
    repo = Path(repository_root).expanduser().resolve()
    if not source_docx.is_file() or source_docx.suffix.lower() != ".docx":
        raise FileNotFoundError(source_docx)
    if not source_markdown.is_file() or source_markdown.suffix.lower() != ".md":
        raise FileNotFoundError(source_markdown)
    markdown = update_markdown_text(source_markdown.read_text(encoding="utf-8"), evidence)
    document = Document(source_docx)
    update_docx_document(document, evidence, repo)

    if output_dir is None:
        docx_target = source_docx
        markdown_target = source_markdown
    else:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        docx_target = directory / source_docx.name
        markdown_target = directory / source_markdown.name
    descriptor, name = tempfile.mkstemp(
        prefix=f".{docx_target.name}.", suffix=".docx", dir=docx_target.parent
    )
    os.close(descriptor)
    temporary_docx = Path(name)
    try:
        document.save(temporary_docx)
        if not temporary_docx.is_file() or temporary_docx.stat().st_size == 0:
            raise RuntimeError("python-docx did not produce a non-empty temporary file")
        os.replace(temporary_docx, docx_target)
    finally:
        if temporary_docx.exists():
            temporary_docx.unlink()
    _atomic_text(markdown_target, markdown)
    return {"docx": str(docx_target), "markdown": str(markdown_target)}


__all__ = [
    "C4ReportEvidence",
    "C4ResultsUpdateError",
    "EXPECTED_C4_CELLS",
    "EXPECTED_C4_OUTCOMES",
    "HISTORICAL_V0_CHECKPOINT_SHA256",
    "HISTORICAL_V0_DOCX_END_MARKER",
    "HISTORICAL_V0_SECTION_END",
    "HISTORICAL_V0_SECTION_START",
    "LOSS_METRICS",
    "PROTOCOLS",
    "SCORE_MODES",
    "load_loss_series",
    "load_report_evidence",
    "update_docx_document",
    "update_markdown_text",
    "validate_historical_v0_summary",
    "validate_summary",
    "write_updated_reports",
]
