#!/usr/bin/env python3
"""Build a new 96+36 compact ledger from the sealed 96+24 ledger.

The supplied launcher root may be a byte-preserving local copy of the remote
formal stage root.  Paths recorded inside its manifest remain untouched and
are audited relative to their original remote roots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_mean_q_extension import (
    extend_compact_ledger,
    write_extended_ledger,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_LEDGER = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_v2_ema_sg_new_scores_cube_seed3072"
    / "new_scores/reconciliation_ledger.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "reports/artifacts/actor_free_td_lewm_complete_cube_seed3072"
    / "sources/v2_ema_new_scores_96_plus_36/reconciliation_ledger.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ledger", default=str(DEFAULT_OLD_LEDGER))
    parser.add_argument(
        "--old-ledger-sha256",
        required=True,
        help="Required SHA-256 lock for the historical 96+24 compact ledger.",
    )
    parser.add_argument(
        "--launcher-root",
        required=True,
        help="Local copy of the new launcher's formal stage root.",
    )
    parser.add_argument(
        "--launcher-manifest",
        required=True,
        help="Copied _launcher/launcher_manifest.json below the formal launcher root.",
    )
    parser.add_argument(
        "--evaluation-checkout",
        required=True,
        help=(
            "Clean local checkout of the commit used by manifest.repository; "
            "recorded remote paths are relocated without editing the manifest."
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the deterministic new SHA without writing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = extend_compact_ledger(
        old_ledger_path=args.old_ledger,
        expected_old_ledger_sha256=args.old_ledger_sha256,
        launcher_root=args.launcher_root,
        launcher_manifest=args.launcher_manifest,
        evaluation_checkout=args.evaluation_checkout,
    )
    if args.validate_only:
        print("Validated old 96+24 ledger plus exact V0/V1 Mean-Q 12-cell launcher.")
    else:
        path = write_extended_ledger(
            result,
            output_path=args.output,
            old_ledger_path=args.old_ledger,
        )
        print(f"Wrote new 96+36 compact ledger: {path}")
    print(f"source_ledger_sha256={result.source_ledger_sha256}")
    print(f"launcher_manifest_sha256={result.launcher_manifest_sha256}")
    print(f"evaluation_commit={result.evaluation_commit}")
    print(f"output_ledger_sha256={result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
