from __future__ import annotations

import csv
import hashlib
import json
import math
import zlib
from copy import deepcopy
from pathlib import Path

import pytest

from tdwm.results.actor_free_td_lewm_v1_c4 import (
    LOSS_METRICS,
    PROTOCOLS,
    SCORE_MODES,
    C4ReportEvidence,
    C4ResultsUpdateError,
    LossSeries,
    load_loss_series,
    load_report_evidence,
    update_docx_document,
    update_markdown_text,
    validate_summary,
)


def _paired(reference: list[bool], candidate: list[bool], *, f_plus: bool) -> dict:
    both_success = sum(left and right for left, right in zip(reference, candidate))
    new = sum((not left) and right for left, right in zip(reference, candidate))
    lost = sum(left and (not right) for left, right in zip(reference, candidate))
    discordant = new + lost
    exact_p = 1.0
    if discordant:
        lower = min(new, lost)
        exact_p = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(lower + 1))
            / (2**discordant),
        )
    value = {
        "reference_successes": sum(reference),
        "candidate_successes": sum(candidate),
        "both_success": both_success,
        "new": new,
        "lost": lost,
        "both_failure": 50 - both_success - new - lost,
        "delta_successes": sum(candidate) - sum(reference),
        "delta_percentage_points": 2.0 * (sum(candidate) - sum(reference)),
        "exact_mcnemar_p_two_sided": exact_p,
        "new_episode_positions": [
            index + 1
            for index, (left, right) in enumerate(zip(reference, candidate))
            if not left and right
        ],
        "lost_episode_positions": [
            index + 1
            for index, (left, right) in enumerate(zip(reference, candidate))
            if left and not right
        ],
    }
    if f_plus:
        value["f_plus_new_successes"] = sum(reference) + new
    return value


def _outcome_vector(protocol_index: int, mode_index: int, method_index: int) -> list[bool]:
    threshold = 18 + protocol_index * 3 + mode_index + method_index
    return [((index * 7 + mode_index * 3) % 50) < threshold for index in range(50)]


def _summary(checkpoint_sha: str = "c" * 64) -> dict:
    protocols = {}
    episode_matrix = []
    nested = {}
    for protocol_index, protocol in enumerate(PROTOCOLS):
        methods = {}
        for method_index, method_key in enumerate(("c4", "v1_c")):
            scores = {}
            for mode_index, mode in enumerate(SCORE_MODES):
                if mode == "f_only":
                    outcomes = _outcome_vector(protocol_index, 0, 0)
                else:
                    outcomes = _outcome_vector(protocol_index, mode_index, method_index)
                nested[(protocol, method_key, mode)] = outcomes
                method_sha = checkpoint_sha if method_key == "c4" else "d" * 64
                scores[mode] = {
                    "success_count": sum(outcomes),
                    "success_rate_percent": 2.0 * sum(outcomes),
                    "episode_successes": outcomes,
                    "source": {
                        "directory": f"/{protocol}/{method_key}/{mode}",
                        "results": {"path": "/results.json", "sha256": "1" * 64},
                        "protocol_manifest": {"path": "/manifest.json", "sha256": "2" * 64},
                        "episode_selection": {
                            "path": "/selection.json",
                            "sha256": "3" * 64,
                            "valid_row_ranks_sha256": "4" * 64,
                        },
                        "checkpoint": {"path": "/epoch_10.pt", "sha256": method_sha},
                    },
                }
            methods[method_key] = {
                "checkpoint_sha256": checkpoint_sha if method_key == "c4" else "d" * 64,
                "scores": scores,
            }
        f_reference = nested[(protocol, "c4", "f_only")]
        versus_f = {
            mode: _paired(f_reference, nested[(protocol, "c4", mode)], f_plus=True)
            for mode in SCORE_MODES
        }
        versus_c = {
            mode: _paired(
                nested[(protocol, "v1_c", mode)],
                nested[(protocol, "c4", mode)],
                f_plus=False,
            )
            for mode in SCORE_MODES
        }
        protocols[protocol] = {
            "selection": {
                "episode_indices": list(range(50)),
                "start_steps": list(range(50)),
                "goal_steps": [index + (25, 50, 100)[protocol_index] for index in range(50)],
                "valid_row_ranks": list(range(50)),
                "valid_row_ranks_sha256": "4" * 64,
            },
            "methods": methods,
            "comparisons": {
                "c4_vs_same_protocol_f_only": versus_f,
                "c4_vs_v1_c_same_score_mode": versus_c,
            },
        }
        for position in range(50):
            episode_matrix.append(
                {
                    "protocol": protocol,
                    "episode_position": position + 1,
                    "pair_id": f"{protocol.upper()}-P{position + 1:02d}",
                    "valid_row_rank": position,
                    "episode_index": position,
                    "start_step": position,
                    "goal_step": position + (25, 50, 100)[protocol_index],
                    "c4": {mode: nested[(protocol, "c4", mode)][position] for mode in SCORE_MODES},
                    "v1_c": {mode: nested[(protocol, "v1_c", mode)][position] for mode in SCORE_MODES},
                }
            )
    return {
        "schema_version": 1,
        "study": {
            "method": "actor_free_td_lewm_v1_c4",
            "comparison_method": "actor_free_td_lewm_v1_c",
            "training_seed": 3072,
            "protocols": list(PROTOCOLS),
            "score_modes": list(SCORE_MODES),
            "episodes_per_protocol": 50,
            "paired_comparison": True,
            "c4_checkpoint_sha256": checkpoint_sha,
            "v1_c_checkpoint_sha256": "d" * 64,
        },
        "protocols": protocols,
        "episode_matrix": episode_matrix,
    }


def _manifest() -> dict:
    return {
        "method": "actor_free_td_lewm_v1_c4",
        "variant": "c4",
        "seed": 3072,
        "protocol": {
            "method": "actor_free_td_lewm_v1_c4",
            "variant": "c4",
            "stage": "full_training",
            "seeds": [3072],
            "g": {
                "state_dim": 192,
                "task_dim": 192,
                "output_dim": 192,
                "action_input": "none",
                "action_effect": "only_via_f_predicted_state",
                "successor_semantics": "includes_current_input_state",
                "actor": "none",
                "reward": "none",
            },
            "time_alignment": {
                "real_online_input": "real_z_i",
                "predicted_online_input": "stop_gradient_f_of_z_i_minus_1_a_i_minus_1",
                "shared_target_current_feature": "real_z_i",
                "shared_target_bootstrap_input": "real_z_i_plus_1",
                "terminal_semantics": "d_i_true_when_real_z_i_is_terminal",
                "terminal_target": "y_i_equals_z_i",
                "f_output_gradient": "stop_gradient",
            },
            "joint_objective": {
                "vector_td_population": "all_transitions_both_branches",
                "vector_reduction": "mean_of_squared_l2_norm",
                "goal_subset": "goal_derived_tasks_only",
                "goal_projection_weight": 1.0,
                "branch_combination": "one_half_real_plus_predicted",
                "target_gradient": "stop_gradient",
                "trainable_modules": ["online_g_c4"],
                "lewm_prediction_loss": "none",
                "sigreg_loss": "none",
            },
            "training": {"epochs": 10, "optimizer_steps_per_epoch": 12_796},
        },
        "model": {
            "trainable_modules": ["online_g_c4"],
            "optimizer_scope": "exact_online_g_parameters_only",
            "trainable_lewm_parameters": 0,
        },
        "training": {
            "formal_optimizer_steps": 127_960,
            "configured_optimizer_steps": 127_960,
            "epochs": 10,
            "validation_skipped": False,
            "f_output_stop_gradient": True,
            "lewm_prediction_loss": False,
            "sigreg_loss": False,
            "loss_metrics": list(LOSS_METRICS),
        },
    }


def _write_metrics(path: Path) -> None:
    columns = ["epoch"] + [
        f"{stage}/{metric}{'_epoch' if stage == 'train' else ''}"
        for stage in ("train", "validation")
        for metric in LOSS_METRICS
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for epoch in range(10):
            row = {"epoch": epoch}
            for stage_index, stage in enumerate(("train", "validation")):
                component_values = []
                for metric_index, metric in enumerate(LOSS_METRICS[:-1]):
                    value = float(10 + stage_index + metric_index + epoch)
                    component_values.append(value)
                    key = f"{stage}/{metric}{'_epoch' if stage == 'train' else ''}"
                    row[key] = value
                total_key = f"{stage}/c4_total_loss{'_epoch' if stage == 'train' else ''}"
                row[total_key] = sum(component_values) / 2.0
            writer.writerow(row)


def _evidence(summary: dict) -> C4ReportEvidence:
    losses = {
        stage: {
            metric: LossSeries(tuple(float(index + offset) for index in range(1, 11)))
            for offset, metric in enumerate(LOSS_METRICS)
        }
        for stage in ("train", "validation")
    }
    return C4ReportEvidence(
        summary=summary,
        summary_path=Path("/tmp/summary.json"),
        summary_sha256="a" * 64,
        training_manifest=_manifest(),
        training_manifest_path=Path("/tmp/run_manifest.json"),
        training_manifest_sha256="b" * 64,
        metrics_path=Path("/tmp/metrics.csv"),
        metrics_sha256="c" * 64,
        losses=losses,
        checkpoint_path=Path("/tmp/epoch_10.pt"),
        checkpoint_sha256="c" * 64,
    )


def test_validate_summary_requires_exact_18_cell_900_outcome_matrix() -> None:
    value = validate_summary(_summary())
    assert len(value["protocols"]) == 3
    assert sum(
        len(value["protocols"][protocol]["methods"]["c4"]["scores"][mode]["episode_successes"])
        for protocol in PROTOCOLS
        for mode in SCORE_MODES
    ) == 900


def test_validate_summary_rejects_missing_cell_and_paired_drift() -> None:
    missing = _summary()
    del missing["protocols"]["o100"]["methods"]["c4"]["scores"]["g_only"]
    with pytest.raises(C4ResultsUpdateError, match="exactly six"):
        validate_summary(missing)

    drifted = _summary()
    drifted["protocols"]["o50"]["comparisons"]["c4_vs_same_protocol_f_only"]["g_only"]["new"] += 1
    with pytest.raises(C4ResultsUpdateError, match="must be recomputed"):
        validate_summary(drifted)


def test_load_loss_series_requires_ten_complete_consistent_epochs(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics)
    values = load_loss_series(metrics)
    assert values["train"]["c4_total_loss"].first == 23.0
    assert values["validation"]["c4_total_loss"].final == 43.0

    rows = list(csv.DictReader(metrics.read_text().splitlines()))
    rows[-1]["validation/real_goal_loss"] = "nan"
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(C4ResultsUpdateError, match="NaN/Inf"):
        load_loss_series(metrics)


def test_load_report_evidence_binds_checkpoint_and_optional_png(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_10.pt"
    checkpoint.write_bytes(b"c4-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_summary(checkpoint_sha)), encoding="utf-8")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics)
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return len(payload).to_bytes(4, "big") + kind + payload + checksum.to_bytes(4, "big")

    png = tmp_path / "loss.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            (1).to_bytes(4, "big")
            + (1).to_bytes(4, "big")
            + bytes((8, 2, 0, 0, 0)),
        )
        + chunk(b"IDAT", zlib.compress(bytes((0, 0, 0, 0))))
        + chunk(b"IEND", b"")
    )

    evidence = load_report_evidence(
        summary_path=summary,
        training_manifest_path=manifest,
        metrics_path=metrics,
        checkpoint_path=checkpoint,
        loss_plot_path=png,
    )
    assert evidence.checkpoint_sha256 == checkpoint_sha
    assert evidence.loss_plot_path == png

    checkpoint.write_bytes(b"wrong")
    with pytest.raises(C4ResultsUpdateError, match="checkpoint bytes differ"):
        load_report_evidence(
            summary_path=summary,
            training_manifest_path=manifest,
            metrics_path=metrics,
            checkpoint_path=checkpoint,
        )


def test_markdown_update_keeps_one_master_table_and_adds_two_c4_matrices() -> None:
    report = Path(__file__).resolve().parents[2] / "reports" / "actor_free_td_lewm_complete_cube_seed3072.md"
    updated = update_markdown_text(report.read_text(encoding="utf-8"), _evidence(_summary()))
    assert "## 27 个训练方法 × 7 种评分" in updated
    assert updated.count("| V1 | C4 |") == 1
    assert "| V1-C4 formal O50 | 1 | E10 |" in updated
    assert "| **TOTAL** | — | — | same locked O50 selection | **511** |" in updated
    assert updated.count("### Protocol by score matrix") == 1
    assert updated.count("### Paired outcomes relative to same-protocol F-only") == 1
    section = updated.split("<!-- RESULTS_TD_V1_C4_FORMAL_START -->", 1)[1]
    paired_lines = [line for line in section.splitlines() if line.startswith(("| O25 |", "| O50 |", "| O100 |"))]
    assert len(paired_lines) == 21  # 3 protocol-matrix rows + 18 paired rows
    assert "18 C4 cells and 900 C4 Boolean outcomes" in updated


def test_summary_validation_does_not_mutate_input() -> None:
    summary = _summary()
    before = deepcopy(summary)
    validate_summary(summary)
    assert summary == before


def test_docx_update_in_memory_has_one_c4_row_and_preserves_old_audit_hashes() -> None:
    docx = pytest.importorskip("docx")
    from docx.oxml.ns import qn

    repository = Path(__file__).resolve().parents[2]
    source = repository / "reports" / "results_td_actor_free_td_lewm_complete_cube_seed3072.docx"
    document = docx.Document(source)

    update_docx_document(document, _evidence(_summary()), repository)

    assert len(document.tables[18].rows) == 29
    assert sum(row.cells[1].text == "C4" for row in document.tables[18].rows) == 1
    methods = [row.cells[1].text for row in document.tables[18].rows]
    assert methods[7:12] == ["C", "C2", "C3", "C4", "D"]
    c4_row = next(row for row in document.tables[18].rows if row.cells[1].text == "C4")
    assert c4_row.cells[0].text == "V1"
    assert c4_row.cells[8].text == "26/50 (52%)"
    assert c4_row.cells[9].text == "—"
    shading = c4_row.cells[8]._tc.get_or_add_tcPr().find(qn("w:shd"))
    assert shading is not None and shading.get(qn("w:fill")) == "B7DEE8"
    assert document.tables[24].rows[-1].cells[-1].text == "511"
    assert document.tables[39].rows[3].cells[1].text == (
        "0e5b541bdb11cf6d647fc1e679499a02c3aa430d64e37c8819d02c44e1dcb900"
    )
    assert any(paragraph.text == "RESULTS TD / V1-C4 FORMAL EXTENSION END" for paragraph in document.paragraphs)
    assert len(document.tables) == 49
