#!/usr/bin/env python3
"""Reconcile and archive the split 96-cell V2-EMA new-score sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_v2_ema_new_scores import (
    reconcile_new_score_sweeps,
    write_archive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-epoch3-root",
        required=True,
        help="Dedicated bundle root containing only the strict epoch-3 replacement.",
    )
    parser.add_argument(
        "--strict-epoch3-state",
        required=True,
        help="Final 12-cell scheduler state.json below the strict epoch-3 root.",
    )
    parser.add_argument(
        "--original-epoch4-10-root",
        required=True,
        help="Original EMA bundle root containing the epoch-4-through-10 outputs.",
    )
    parser.add_argument(
        "--original-epoch4-10-state",
        required=True,
        help="Final filtered 84-cell scheduler state.json below the original root.",
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument(
        "--fixed-launcher",
        action="append",
        nargs=2,
        default=None,
        metavar=("ROOT", "MANIFEST"),
        help=(
            "Optionally include the fixed V0/V1/V2 checkpoint comparison. "
            "Repeat exactly three times for the non-C, C-first and V2-C-mean "
            "launcher root/launcher_manifest.json pairs."
        ),
    )
    parser.add_argument(
        "--fixed-evaluation-checkout",
        help=(
            "Exact git checkout recorded as launcher_manifest.repository for "
            "the optional fixed 24-cell comparison. Its HEAD must be the locked "
            "evaluation commit."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all 96 source cells without reading or writing an archive.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Revalidate sources and byte-compare the existing generated archive.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only and args.check:
        raise SystemExit("--validate-only and --check are mutually exclusive.")
    study = reconcile_new_score_sweeps(
        strict_epoch3_root=args.strict_epoch3_root,
        strict_epoch3_state=args.strict_epoch3_state,
        original_epoch4_10_root=args.original_epoch4_10_root,
        original_epoch4_10_state=args.original_epoch4_10_state,
        fixed_launchers=args.fixed_launcher,
        fixed_evaluation_checkout=args.fixed_evaluation_checkout,
    )
    if args.validate_only:
        fixed = (
            f" plus {len(study.fixed_cells)} fixed-checkpoint cells"
            if study.fixed_cells
            else ""
        )
        print(
            "Validated strict epoch 3 (12) + original epochs 4-10 (84) "
            f"= 96 cells{fixed}."
        )
        return 0
    paths = write_archive(study, artifact_dir=args.artifact_dir, check=args.check)
    verb = "Verified" if args.check else "Wrote"
    print(f"{verb} {len(paths)} lightweight archive files from 96 validated cells:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
