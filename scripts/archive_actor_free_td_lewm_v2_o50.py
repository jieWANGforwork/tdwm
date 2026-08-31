#!/usr/bin/env python3
"""Validate and archive the formal Actor-Free TD-LeWM V2 6 x 3 O50 study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_v2 import validate_bundle, write_archive

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPOSITORY_ROOT / "reports/artifacts/actor_free_td_lewm_v2_cube_seed3072"
)
DEFAULT_REPORT = REPOSITORY_ROOT / "reports/actor_free_td_lewm_v2_cube_seed3072.md"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate six V2 trainings and F-only/G-only/F+G Cube O50 for each "
            "method, then build the deterministic formal archive."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help=(
            "Root containing training_acceptance.json and "
            "evaluations/<variant>/<score_mode>."
        ),
    )
    parser.add_argument(
        "--evaluation-root",
        help="Explicit 6 x 3 evaluation root for an alternate export layout.",
    )
    parser.add_argument(
        "--training-acceptance",
        help="Explicit training_acceptance.json path for an alternate export layout.",
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate raw inputs and byte-compare the existing generated archive.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all raw inputs without reading or writing generated outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check and args.validate_only:
        raise SystemExit("--check and --validate-only are mutually exclusive.")
    study = validate_bundle(
        args.bundle,
        evaluation_root=args.evaluation_root,
        acceptance_path=args.training_acceptance,
    )
    if args.validate_only:
        print("Validated 6 V2 trainings and 18 formal Cube O50 evaluations.")
        return 0
    paths = write_archive(
        study,
        artifact_dir=args.artifact_dir,
        report_path=args.report_path,
        check=args.check,
    )
    verb = "Verified" if args.check else "Wrote"
    print(f"{verb} {len(paths)} files from the complete V2 formal study:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
