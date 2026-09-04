#!/usr/bin/env python3
"""Validate and archive the eight formal V1-C/C2/C3 endpoint O50 cells."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_v1_c2_c3 import (
    reconcile_endpoint_results,
    write_archive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPOSITORY_ROOT / "reports/artifacts/actor_free_td_lewm_v1_c2_c3_cube_seed3072"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c2-launcher-root",
        required=True,
        help="Root containing the formal C/C2 launcher manifest and seven outputs.",
    )
    parser.add_argument(
        "--c2-launcher-manifest",
        required=True,
        help="The successful seven-job formal launcher_manifest.json.",
    )
    parser.add_argument(
        "--c3-output-dir",
        required=True,
        help="Standalone formal V1-C3 State-V O50 output directory.",
    )
    parser.add_argument(
        "--c2-training-run",
        help="Optional completed formal V1-C2 run to preserve its loss metrics.",
    )
    parser.add_argument(
        "--c3-training-run",
        help=(
            "Optional completed formal V1-C3 run to preserve its loss metrics "
            "and offline-validation summary."
        ),
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all eight cells and 400 outcomes without writing an archive.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Revalidate sources and byte-check an existing archive.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only and args.check:
        raise SystemExit("--validate-only and --check are mutually exclusive.")
    study = reconcile_endpoint_results(
        c2_launcher_root=args.c2_launcher_root,
        c2_launcher_manifest=args.c2_launcher_manifest,
        c3_output_dir=args.c3_output_dir,
        c2_training_run=args.c2_training_run,
        c3_training_run=args.c3_training_run,
    )
    if args.validate_only:
        print("Validated 8 endpoint O50 cells with 400 boolean outcomes.")
        return 0
    paths = write_archive(
        study,
        artifact_dir=args.artifact_dir,
        check=args.check,
    )
    verb = "Verified" if args.check else "Wrote"
    print(f"{verb} {len(paths)} endpoint archive files from 8 validated cells:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
