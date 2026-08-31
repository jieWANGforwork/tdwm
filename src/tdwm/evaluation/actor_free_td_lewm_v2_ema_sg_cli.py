"""Shared command-line runner for V2-EMA-SG Cube O50 evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_common import (
    actor_free_td_v2_ema_sg_output_directory_name,
)


def run_actor_free_td_lewm_v2_ema_sg_evaluation(variant: str) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate Actor-Free TD-LeWM V2 EMA {variant.upper()} with "
            "policy-free CEM."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--training-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--score-mode",
        choices=(
            "f_only",
            "g_only",
            "f_plus_g",
            "f_plus_g_first",
            "g_only_f_rollout_mean",
        ),
        default=None,
    )
    parser.add_argument("--g-first-weight", type=float, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, choices=range(3, 10))
    parser.add_argument("--video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")

    module = importlib.import_module(
        f"tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_{variant}"
    )
    load_protocol = getattr(
        module,
        f"load_actor_free_td_lewm_v2_ema_sg_{variant}_evaluation_protocol",
    )
    evaluate = getattr(
        module,
        f"evaluate_actor_free_td_lewm_v2_ema_sg_{variant}",
    )
    output_dir = args.output_dir
    if output_dir is None:
        protocol = load_protocol(args.config)
        output_dir = Path(os.environ.get("TDWM_RUN_ROOT", "outputs")) / (
            actor_free_td_v2_ema_sg_output_directory_name(
                protocol,
                smoke=args.smoke,
                pilot=args.pilot,
                score_mode=args.score_mode,
                g_first_weight=args.g_first_weight,
            )
        )
    result = evaluate(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        training_manifest_path=args.training_manifest,
        video=args.video,
        smoke=args.smoke,
        pilot=args.pilot,
        score_mode=args.score_mode,
        g_first_weight=args.g_first_weight,
        checkpoint_epoch=args.checkpoint_epoch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = ["run_actor_free_td_lewm_v2_ema_sg_evaluation"]
