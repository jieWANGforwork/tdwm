#!/usr/bin/env python3
"""Evaluate V1-C3 with full frozen-F rollout and EMA terminal State-V."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    actor_free_td_lewm_v1_c3_output_directory_name,
    evaluate_actor_free_td_lewm_v1_c3,
    load_actor_free_td_lewm_v1_c3_evaluation_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V1-C3 using only EMA State-V after a full five-block "
            "frozen-LeWM rollout."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-epoch", type=int)
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
    if args.checkpoint_epoch is not None and (args.smoke or args.pilot):
        raise SystemExit("--checkpoint-epoch cannot be combined with smoke/pilot.")
    output_dir = args.output_dir
    if output_dir is None:
        protocol = load_actor_free_td_lewm_v1_c3_evaluation_protocol(args.config)
        output_dir = Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / (
            actor_free_td_lewm_v1_c3_output_directory_name(
                protocol, smoke=args.smoke, pilot=args.pilot
            )
        )
    result = evaluate_actor_free_td_lewm_v1_c3(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_epoch=args.checkpoint_epoch,
        video=args.video,
        smoke=args.smoke,
        pilot=args.pilot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
