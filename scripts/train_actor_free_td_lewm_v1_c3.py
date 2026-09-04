#!/usr/bin/env python3
"""Train the V1-C3 RP1-style state critic from a frozen V1-C parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tdwm.training.actor_free_td_lewm_v1_c3 import (
    load_actor_free_td_lewm_v1_c3_training_protocol,
    train_actor_free_td_lewm_v1_c3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V1-C3 while freezing every V1-C parent parameter."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--initial-v1-c-checkpoint", required=True)
    parser.add_argument("--frozen-latent-store", required=True)
    parser.add_argument("--split-indices", required=True)
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    if not args.smoke and (args.max_steps is not None or args.skip_validation):
        raise SystemExit("--max-steps/--skip-validation are smoke-only.")
    load_actor_free_td_lewm_v1_c3_training_protocol(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / "actor_free_td_lewm_v1_c3_cube_training"
        )
    result = train_actor_free_td_lewm_v1_c3(
        protocol_path=args.config,
        dataset_path=args.dataset,
        output_dir=output_dir,
        seed=args.seed,
        initial_v1_c_checkpoint_path=args.initial_v1_c_checkpoint,
        frozen_latent_store_path=args.frozen_latent_store,
        split_indices_path=args.split_indices,
        resume=args.resume,
        smoke=args.smoke,
        max_steps=args.max_steps,
        skip_validation=args.skip_validation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
