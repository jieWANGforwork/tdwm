#!/usr/bin/env python3
"""Replace only the O25 summary table with a method-by-score matrix.

This intentionally edits the existing DOCX in place so that sections appended by
other report workflows remain untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPO_DOCX = (
    REPOSITORY_ROOT
    / "reports/results_td_actor_free_td_lewm_complete_cube_seed3072.docx"
)
PROJECT_DOCX = REPOSITORY_ROOT.parents[1] / "Results TD.docx"

OLD_HEADER_PREFIX = ("Checkpoint", "Score", "Loss", "O25")
NEW_HEADER_PREFIX = ("Method / checkpoint",)
NEW_HEADERS = (
    "Method / checkpoint",
    "F-only\nbaseline",
    "G-only",
    "New / Lost / F+New",
    "F+G tail",
    "New / Lost / F+New",
    "First-Q\nalpha=.25",
    "New / Lost / F+New",
    "Mean-Q",
    "New / Lost / F+New",
    "First-Q2\nalpha=.25",
    "New / Lost / F+New",
    "C3 State-V + First-Q2\nalpha=.10",
    "New / Lost / F+New",
)
NEW_WIDTHS = (1800, 1200, 900, 700, 900, 700, 900, 700, 900, 700, 950, 700, 2200, 1150)

NEW_HEADING = "Methods by inference score, relative to F-only"
NEW_EXPLANATION = (
    "Each row is one trained method/checkpoint. Every score column is followed "
    "by one adjacent New / Lost / F+New column. A dash means not evaluated."
)
NEW_LEGEND = (
    "The adjacent column contains counts only: newly successful pairs / lost F "
    "successes / total if all F successes are retained and New is added."
)
NEW_FOOTNOTE = (
    "* C3 freezes and reuses V1-C's F. Its F-only cell is therefore the same "
    "37/50 baseline reference, not a separate C3 F-only rerun. Exact pair-level "
    "records remain in the audit CSV rather than a second results table."
)


def _load_report_module() -> ModuleType:
    path = REPOSITORY_ROOT / "scripts/append_results_td_o25_v1_c_c3.py"
    spec = importlib.util.spec_from_file_location("_o25_report", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _header(table: Any) -> tuple[str, ...]:
    return tuple(cell.text.strip() for cell in table.rows[0].cells)


def _matches_prefix(header: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(header) >= len(prefix) and header[: len(prefix)] == prefix


def _set_paragraph_text(paragraph: Any, text: str, report: ModuleType, **font: Any) -> None:
    paragraph.text = ""
    report.load_docx_helpers()._set_run_font(paragraph.add_run(text), **font)


def _replace_neighbor_copy(document: Any, report: ModuleType) -> None:
    heading_candidates = {
        "Seven scores and paired coverage relative to F-only",
        NEW_HEADING,
    }
    explanation_prefixes = (
        "The first six V1-C E10 rows",
        "Each row is one trained method/checkpoint",
    )
    legend_prefixes = (
        "Yellow fill marks the highest standalone O25 result",
        "New = F-only fails and the score succeeds",
        "The adjacent column contains counts only",
    )

    headings = [p for p in document.paragraphs if p.text in heading_candidates]
    explanations = [
        p for p in document.paragraphs if p.text.startswith(explanation_prefixes)
    ]
    legends = [p for p in document.paragraphs if p.text.startswith(legend_prefixes)]
    if len(headings) != 1 or len(explanations) != 1 or len(legends) != 1:
        raise ValueError(
            "expected one O25 heading, explanation and legend; found "
            f"{len(headings)}, {len(explanations)}, {len(legends)}"
        )
    _set_paragraph_text(headings[0], NEW_HEADING, report, size=14, color="000000", bold=True)
    _set_paragraph_text(explanations[0], NEW_EXPLANATION, report, size=9.5, color="000000")
    _set_paragraph_text(legends[0], NEW_LEGEND, report, size=9.5, color="374151", bold=True)


def _remove_existing_footnote(document: Any) -> None:
    for paragraph in list(document.paragraphs):
        if paragraph.text.startswith("* C3 freezes and reuses V1-C's F."):
            paragraph._element.getparent().remove(paragraph._element)


def _remove_legacy_detail_block(document: Any) -> int:
    """Remove the old transition and P01-P50 result tables from the O25 section."""
    from docx.oxml.ns import qn

    starts = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text == "Exact F-only successes and failures"
    ]
    ends = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text == "Conclusions and the next gating objective"
    ]
    if not starts:
        return 0
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            "expected one O25 legacy-detail start and conclusion heading; found "
            f"{len(starts)} and {len(ends)}"
        )
    body = document._element.body
    children = list(body)
    start_index = children.index(starts[0]._p)
    end_index = children.index(ends[0]._p)
    if end_index <= start_index:
        raise ValueError("O25 legacy-detail block is reversed")
    removed_tables = sum(
        child.tag == qn("w:tbl") for child in children[start_index:end_index]
    )
    for child in children[start_index:end_index]:
        body.remove(child)
    return removed_tables


def _remove_protocol_and_score_tables(document: Any) -> int:
    """Leave the O25 extension with exactly one results table."""
    from docx.oxml.ns import qn

    starts = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text == "Protocol and evidence fingerprints"
    ]
    ends = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text == NEW_HEADING
    ]
    if not starts:
        return 0
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            "expected one O25 protocol start and result heading; found "
            f"{len(starts)} and {len(ends)}"
        )
    body = document._element.body
    children = list(body)
    start_index = children.index(starts[0]._p)
    end_index = children.index(ends[0]._p)
    if end_index <= start_index:
        raise ValueError("O25 protocol block is reversed")
    removed_tables = sum(
        child.tag == qn("w:tbl") for child in children[start_index:end_index]
    )
    for child in children[start_index:end_index]:
        body.remove(child)
    return removed_tables


def reformat(path: Path, report: ModuleType) -> None:
    from docx import Document

    document = Document(str(path))
    prior_table_count = len(document.tables)
    prior_section_count = len(document.sections)
    related_work_was_present = any(
        "相关工作 Cube 结果比较" in paragraph.text for paragraph in document.paragraphs
    )

    candidates = [
        table
        for table in document.tables
        if _matches_prefix(_header(table), OLD_HEADER_PREFIX)
        or _matches_prefix(_header(table), NEW_HEADER_PREFIX)
    ]
    if len(candidates) != 1:
        raise ValueError(f"{path}: expected exactly one O25 summary table, found {len(candidates)}")
    old_table = candidates[0]

    outcomes, selection = report.validate_evidence(report.ARTIFACT_ROOT)
    summary = report.reconcile(outcomes, selection)
    rows = report.result_matrix_rows(summary["cells"])
    helpers = report.load_docx_helpers()

    new_table = helpers._add_table(
        document,
        headers=NEW_HEADERS,
        rows=rows,
        widths=NEW_WIDTHS,
    )
    report._format_table(new_table, helpers, font_size=7.1, centered_from=1)
    helpers._shade_cell(new_table.rows[2].cells[12], "FFF2CC")

    old_table._tbl.addprevious(new_table._tbl)
    old_table._element.getparent().remove(old_table._element)
    _replace_neighbor_copy(document, report)
    _remove_existing_footnote(document)
    removed_tables = _remove_legacy_detail_block(document)
    removed_tables += _remove_protocol_and_score_tables(document)

    footnote = document.add_paragraph()
    helpers._set_run_font(
        footnote.add_run(NEW_FOOTNOTE), size=8.3, color="5C6975"
    )
    new_table._tbl.addnext(footnote._p)

    if len(document.tables) != prior_table_count - removed_tables:
        raise AssertionError("unexpected table-count change while consolidating O25 results")
    if len(document.sections) != prior_section_count:
        raise AssertionError("section count changed while replacing the O25 summary")
    if related_work_was_present and not any(
        "相关工作 Cube 结果比较" in paragraph.text for paragraph in document.paragraphs
    ):
        raise AssertionError("a later related-work section was lost")
    if "New" in new_table.rows[1].cells[1].text or "Lost" in new_table.rows[1].cells[1].text:
        raise AssertionError("F-only baseline must not contain New/Lost counts")
    if new_table.rows[2].cells[13].text != "6/5/43":
        raise AssertionError("C3 paired counts are missing")

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        document.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reformat the existing Results TD O25 summary as a method-by-score matrix"
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[REPO_DOCX, PROJECT_DOCX])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _load_report_module()
    for raw_path in args.paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        reformat(path, report)
        print(f"Updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
