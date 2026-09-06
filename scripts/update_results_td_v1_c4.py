#!/usr/bin/env python3
"""Validate formal V1-C4 evidence and stage/update the canonical Results TD."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tdwm.results.actor_free_td_lewm_v1_c4 import (
    load_report_evidence,
    write_updated_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate 18 objective-v1 C4 formal cells / 900 outcomes, the exact "
            "superseded objective-v0 summary, and training evidence, then update "
            "the existing Results TD DOCX and Markdown. Run the "
            "document-operation marker before invoking a non-dry run, and render "
            "the staged DOCX before promotion."
        )
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--historical-v0-summary",
        required=True,
        help=(
            "Exact pre-versioned objective-v0 formal summary. It is preserved in a "
            "historical section but excluded from the current ledger and winners."
        ),
    )
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--loss-plot",
        help="Optional validated PNG with the C4 E1-E10 train/validation loss curves.",
    )
    parser.add_argument("--docx", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--output-dir",
        help=(
            "Stage same-basename outputs in this directory. Omit only after visual "
            "QA when the two canonical inputs should be atomically replaced in place."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every evidence input without importing or writing DOCX/Markdown.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = load_report_evidence(
        summary_path=args.summary,
        historical_v0_summary_path=args.historical_v0_summary,
        training_manifest_path=args.training_manifest,
        metrics_path=args.metrics,
        checkpoint_path=args.checkpoint,
        loss_plot_path=args.loss_plot,
    )
    validated = {
        "c4_cells": 18,
        "c4_boolean_outcomes": 900,
        "summary_sha256": evidence.summary_sha256,
        "historical_v0_cells": 18,
        "historical_v0_boolean_outcomes": 900,
        "historical_v0_summary_sha256": evidence.historical_v0_summary_sha256,
        "training_manifest_sha256": evidence.training_manifest_sha256,
        "metrics_sha256": evidence.metrics_sha256,
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "loss_plot_sha256": evidence.loss_plot_sha256,
    }
    if args.dry_run:
        print(json.dumps({"validated": validated, "written": False}, indent=2, sort_keys=True))
        return 0
    outputs = write_updated_reports(
        docx_path=args.docx,
        markdown_path=args.markdown,
        evidence=evidence,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"validated": validated, "written": True, "outputs": outputs},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
