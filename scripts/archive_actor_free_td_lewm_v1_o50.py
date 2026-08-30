#!/usr/bin/env python3
"""Validate and archive the formal Actor-Free TD-JEPA V1 6x3 O50 study."""

from __future__ import annotations

import argparse
from pathlib import Path

from tdwm.results.actor_free_td_lewm_v1 import validate_bundle, write_archive

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPOSITORY_ROOT / "reports/artifacts/actor_free_td_lewm_v1_cube_seed3072"
)
DEFAULT_REPORT = REPOSITORY_ROOT / "reports/actor_free_td_lewm_v1_cube_seed3072.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate six Actor-Free TD-JEPA V1 methods across F-only, G-only, "
            "and F+G formal Cube O50 scoring, then build a deterministic archive."
        )
    )
    parser.add_argument(
        "--bundle", required=True, help="Root of the exported V1 bundle."
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Destination for machine-readable archive files.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT),
        help="Destination for the generated Markdown Results TD report.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs and byte-compare all previously generated files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all raw inputs without reading or writing generated outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check and args.validate_only:
        raise SystemExit("--check and --validate-only are mutually exclusive.")
    study = validate_bundle(args.bundle)
    if args.validate_only:
        print("Validated 6 V1 methods x 3 formal O50 score modes (18 runs).")
        return
    paths = write_archive(
        study,
        artifact_dir=args.artifact_dir,
        report_path=args.report_path,
        check=args.check,
    )
    verb = "Verified" if args.check else "Wrote"
    print(f"{verb} {len(paths)} files from 18 validated formal O50 runs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
