#!/usr/bin/env python3
"""Validate and plot one completed formal V1-C4 training run."""

from __future__ import annotations

import argparse

from tdwm.results.actor_free_td_lewm_v1_c4_training import (
    write_training_metrics_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the formal V1-C4 training contract and write a JSON "
            "summary, a ten-row epoch CSV and one loss-curve PNG."
        )
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Completed Lightning metrics.csv (select the completed version explicitly).",
    )
    parser.add_argument("--training-result", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        paths = write_training_metrics_report(
            metrics_path=args.metrics,
            training_result_path=args.training_result,
            training_manifest_path=args.training_manifest,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
