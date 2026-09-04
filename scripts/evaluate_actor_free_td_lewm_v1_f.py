#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.evaluation.actor_free_td_lewm_v1_f import (
    evaluate_actor_free_td_lewm_v1_f,
    load_actor_free_td_lewm_v1_f_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_v1_common import (
    actor_free_td_v1_output_directory_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Actor-Free TD-LeWM V1 F with policy-free CEM."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--score-mode",
        choices=(
            "f_only",
            "g_only",
            "f_plus_g",
            "f_plus_g_first",
            "f_plus_g_first_q2",
            "g_only_f_rollout_mean",
        ),
        default=None,
    )
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
        protocol = load_actor_free_td_lewm_v1_f_evaluation_protocol(args.config)
        output_dir = Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / (
            actor_free_td_v1_output_directory_name(
                protocol,
                smoke=args.smoke,
                pilot=args.pilot,
                score_mode=args.score_mode,
                g_first_weight=args.g_first_weight,
            )
        )
    result = evaluate_actor_free_td_lewm_v1_f(
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
