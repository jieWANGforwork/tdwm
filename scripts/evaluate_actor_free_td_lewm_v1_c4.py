#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.adapters.actor_free_td_lewm_v1_c4 import SCORE_MODES
from tdwm.evaluation.actor_free_td_lewm_v1_c4 import (
    actor_free_td_lewm_v1_c4_output_directory_name,
    evaluate_actor_free_td_lewm_v1_c4,
    load_actor_free_td_lewm_v1_c4_evaluation_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate post-action-ghost, state-only Actor-Free TD-LeWM V1-C4 "
            "with policy-free CEM."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--score-mode", choices=tuple(sorted(SCORE_MODES)), default=None)
    parser.add_argument("--g-first-weight", type=float, default=None)
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
        protocol = load_actor_free_td_lewm_v1_c4_evaluation_protocol(args.config)
        output_dir = Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / (
            actor_free_td_lewm_v1_c4_output_directory_name(
                protocol,
                smoke=args.smoke,
                pilot=args.pilot,
                score_mode=args.score_mode,
                g_first_weight=args.g_first_weight,
            )
        )
    result = evaluate_actor_free_td_lewm_v1_c4(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        video=args.video,
        smoke=args.smoke,
        pilot=args.pilot,
        score_mode=args.score_mode,
        g_first_weight=args.g_first_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
