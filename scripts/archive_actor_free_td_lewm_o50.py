#!/usr/bin/env python3
"""Validate a complete Actor-Free TD-LeWM bundle and build its audit archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from tdwm.results.actor_free_td_lewm import validate_bundle, write_archive


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPOSITORY_ROOT / "reports/artifacts/actor_free_td_lewm_cube_seed3072"
)
DEFAULT_REPORT = REPOSITORY_ROOT / "reports/actor_free_td_lewm_cube_seed3072.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete 7x3 Actor-Free TD-LeWM Cube O50 result bundle "
            "and generate deterministic auditable reports."
        )
    )
    parser.add_argument("--bundle", required=True, help="Root of the exported bundle.")
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Destination for summary.json, paired CSV, README, and checksums.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT),
        help="Destination for the human-readable Results TD Markdown report.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the bundle and byte-compare generated files without writing.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the complete bundle without reading or writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check and args.validate_only:
        raise SystemExit("--check and --validate-only are mutually exclusive.")
    study = validate_bundle(args.bundle)
    if args.validate_only:
        print("Validated 7 variants x 3 formal O50 score modes (21 runs).")
        return
    paths = write_archive(
        study,
        artifact_dir=args.artifact_dir,
        report_path=args.report_path,
        check=args.check,
    )
    verb = "Verified" if args.check else "Wrote"
    print(f"{verb} {len(paths)} files from 21 validated formal O50 runs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
