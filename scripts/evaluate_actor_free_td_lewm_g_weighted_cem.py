#!/usr/bin/env python3
"""Test an existing G through path-weighted or action-weighted CEM updates.

This adds inference experiments only. F selects the elites by its unchanged
five-block terminal cost, then G weights their fitted Gaussian moments.
Historical checkpoint digests, pair selection, rendering configuration and
feedback intervals are retained. Output must be a new directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from tdwm.adapters.g_weighted_cem import GWeightedCEMConfig
from tdwm.evaluation.full_plan_revalidation import require_new_revalidation_output
from tdwm.evaluation.g_weighted_cem import configure_g_weighted_cem
from tdwm.evaluation.lewm_checkpoint import _sha256

VERSIONS = ("v0", "v1", "v2", "v2_ema_sg")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3", "c2", "c4")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight-mode", choices=("path", "action"), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        GWeightedCEMConfig(args.weight_mode, args.temperature)
    except ValueError as error:
        parser.error(str(error))
    if args.variant in {"c2", "c4"} and args.version != "v1":
        parser.error("C2/C4 are V1-only checkpoints.")
    if args.checkpoint_epoch is not None:
        if (
            args.version not in {"v2", "v2_ema_sg"}
            or not 3 <= args.checkpoint_epoch <= 9
        ):
            parser.error(
                "--checkpoint-epoch is only for historical V2/EMA epochs 3--9."
            )
        if args.smoke:
            parser.error("Intermediate-epoch formal validation cannot be a smoke run.")
    if not args.dry_run:
        for name in ("dataset", "checkpoint_path", "checkpoint_sha256", "output_dir"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for execution.")
        digest = args.checkpoint_sha256
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            parser.error("--checkpoint-sha256 must be a lowercase SHA-256 digest.")
    return args


def resolve_protocol(args):
    stem = f"actor_free_td_lewm_{args.version}_{args.variant}"
    module = importlib.import_module(f"tdwm.evaluation.{stem}")
    original = getattr(module, f"load_{stem}_evaluation_protocol")(args.config)
    baseline = getattr(module, f"configure_{stem}_evaluation_mode")(
        original,
        smoke=args.smoke,
        pilot=False,
        score_mode="f_only",
        g_first_weight=None,
    )
    config = GWeightedCEMConfig(args.weight_mode, args.temperature)
    return configure_g_weighted_cem(baseline, config), getattr(
        module, f"evaluate_{stem}"
    )


def main(argv=None):
    args = parse_args(argv)
    protocol, evaluate = resolve_protocol(args)
    if args.dry_run:
        print(
            json.dumps(
                {"dry_run": True, "protocol": protocol}, indent=2, sort_keys=True
            )
        )
        return
    require_new_revalidation_output(args.output_dir)
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != args.checkpoint_sha256:
        raise SystemExit("Checkpoint does not match the declared historical SHA-256.")
    import torch

    if not args.smoke and not torch.cuda.is_available():
        raise SystemExit(
            "Formal G-weighted evaluation requires a GPU; no evaluation was started."
        )
    kwargs = {
        "protocol_path": args.config,
        "dataset_path": args.dataset,
        "checkpoint_path": str(checkpoint),
        "output_dir": args.output_dir,
        "score_mode": "f_only",
        "g_weighted_cem": GWeightedCEMConfig(args.weight_mode, args.temperature),
        "smoke": args.smoke,
        "video": args.video,
    }
    if args.checkpoint_epoch is not None:
        kwargs["checkpoint_epoch"] = args.checkpoint_epoch
    result = evaluate(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
