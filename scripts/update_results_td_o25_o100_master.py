#!/usr/bin/env python3
"""Validate and add the current O25/O100 master tables to Results TD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdwm.results.actor_free_td_lewm_o25_o100_master import (
    load_master_evidence,
    write_updated_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument(
        "--repository-root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evidence = load_master_evidence(args.repository_root)
    validated = {
        master.protocol: {
            "rows": len(master.rows),
            "formal_cells": sum(
                cell is not None for row in master.rows for cell in row.cells.values()
            ),
            "outcomes": sum(
                len(cell.outcomes)
                for row in master.rows
                for cell in row.cells.values()
                if cell is not None
            ),
        }
        for master in (evidence.o25, evidence.o100)
    }
    if args.dry_run:
        print(json.dumps({"validated": validated, "written": False}, indent=2))
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required unless --dry-run is used")
    outputs = write_updated_reports(
        docx_path=args.docx,
        markdown_path=args.markdown,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"validated": validated, "written": True, "outputs": outputs}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
