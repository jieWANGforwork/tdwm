#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation.actor_free_td_lewm import evaluate_actor_free_td_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an Actor-Free TD-LeWM variant with policy-free CEM."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/experiment/"
            "actor_free_td_lewm_serial_decoupled_cube_checkpoint_o50.yaml"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited Cube HDF5 or Lance dataset.",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Joint deployment checkpoint containing LeWM and successor weights.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    output_dir = args.output_dir
    if output_dir is None:
        config_stem = Path(args.config).stem
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / config_stem
        )
    result = evaluate_actor_free_td_lewm(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        video=args.video,
        smoke=args.smoke,
        pilot=args.pilot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
