#!/usr/bin/env python3
"""Validate and archive all 465 formal Actor-Free TD-LeWM Cube O50 cells."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_complete import (
    reconcile_complete_o50,
    write_archive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
REPORT_ARTIFACTS = REPOSITORY_ROOT / "reports/artifacts"
EMA_ARCHIVE = REPORT_ARTIFACTS / "actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072"
TEMP_EVIDENCE = Path("/private/tmp/td-results-full.y1qTdB")

DEFAULTS = {
    "legacy_bundle_root": (
        PROJECT_ROOT / "artifacts/actor_free_td_lewm_final_lightweight_bundle_fa46ed9"
    ),
    "legacy_summary": (
        REPORT_ARTIFACTS / "actor_free_td_lewm_cube_seed3072/summary.json"
    ),
    "legacy_paired_outcomes": (
        REPORT_ARTIFACTS / "actor_free_td_lewm_cube_seed3072/paired_outcomes.csv"
    ),
    "v0_root": TEMP_EVIDENCE / "v0",
    "v0_summary": (
        REPORT_ARTIFACTS / "actor_free_td_lewm_v0_cube_seed3072/formal_o50_summary.json"
    ),
    "v0_training_root": TEMP_EVIDENCE / "v0/training_metadata",
    "v1_bundle_root": PROJECT_ROOT / "tmp/v1-bundle-3c4e62e",
    "v1_summary": (
        REPORT_ARTIFACTS / "actor_free_td_lewm_v1_cube_seed3072/summary.json"
    ),
    "v1_paired_outcomes": (
        REPORT_ARTIFACTS / "actor_free_td_lewm_v1_cube_seed3072/paired_outcomes.csv"
    ),
    "ema_new_ledger": EMA_ARCHIVE / "new_scores/reconciliation_ledger.json",
    "v2_root": TEMP_EVIDENCE / "v2",
    "v2_training_root": TEMP_EVIDENCE / "v2/formal_metadata",
    "v2_ema_root": TEMP_EVIDENCE / "v2_ema",
    "v2_ema_epoch10_summary": EMA_ARCHIVE / "original_scores/summary.json",
    "v2_ema_epoch10_paired_outcomes": (
        EMA_ARCHIVE / "original_scores/paired_outcomes.csv"
    ),
    "artifact_dir": (REPORT_ARTIFACTS / "actor_free_td_lewm_complete_cube_seed3072"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=str(path))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all sources and 465 cells without writing the archive.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Revalidate sources and byte-compare the existing five-file archive.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only and args.check:
        raise SystemExit("--validate-only and --check are mutually exclusive.")
    study = reconcile_complete_o50(
        legacy_bundle_root=args.legacy_bundle_root,
        legacy_summary=args.legacy_summary,
        legacy_paired_outcomes=args.legacy_paired_outcomes,
        v0_root=args.v0_root,
        v0_summary=args.v0_summary,
        v0_training_root=args.v0_training_root,
        v1_bundle_root=args.v1_bundle_root,
        v1_summary=args.v1_summary,
        v1_paired_outcomes=args.v1_paired_outcomes,
        ema_new_ledger=args.ema_new_ledger,
        v2_root=args.v2_root,
        v2_training_root=args.v2_training_root,
        v2_ema_root=args.v2_ema_root,
        v2_ema_epoch10_summary=args.v2_ema_epoch10_summary,
        v2_ema_epoch10_paired_outcomes=args.v2_ema_epoch10_paired_outcomes,
    )
    if args.validate_only:
        print("Validated 465 cells with 23,250 boolean O50 outcomes.")
        return 0
    paths = write_archive(study, artifact_dir=args.artifact_dir, check=args.check)
    verb = "Verified" if args.check else "Wrote"
    print(f"{verb} {len(paths)} complete-ledger artifacts from 465 validated cells:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
