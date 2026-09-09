#!/usr/bin/env python3
"""Re-evaluate an existing C--G3 checkpoint with 25-step real feedback.

This opt-in entry point retains the historical config and readout, requires a
new result directory, and binds formal runs to an explicit checkpoint digest.
It does not train or choose a new critic weight. Use --dry-run on CPU to inspect
the resolved execution protocol without loading data, weights, or an environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from tdwm.evaluation.full_plan_revalidation import (
    FULL_PLAN_SCORE_MODES,
    configure_full_plan_revalidation,
    require_new_revalidation_output,
)
from tdwm.evaluation.lewm_checkpoint import _sha256

VERSIONS = ("v0", "v1", "v2", "v2_ema_sg")
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--score-mode", choices=sorted(FULL_PLAN_SCORE_MODES), required=True
    )
    parser.add_argument("--g-first-weight", type=float)
    parser.add_argument("--dataset", default=os.environ.get("TDWM_CUBE_DATASET"))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
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
    configured = getattr(module, f"configure_{stem}_evaluation_mode")(
        original,
        smoke=args.smoke,
        pilot=False,
        score_mode=args.score_mode,
        g_first_weight=args.g_first_weight,
    )
    return configure_full_plan_revalidation(configured), getattr(
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
            "Formal revalidation requires a GPU; no evaluation was started."
        )
    kwargs = {
        "protocol_path": args.config,
        "dataset_path": args.dataset,
        "checkpoint_path": str(checkpoint),
        "output_dir": args.output_dir,
        "score_mode": args.score_mode,
        "g_first_weight": args.g_first_weight,
        "smoke": args.smoke,
        "video": args.video,
        "full_plan_revalidation": True,
    }
    if args.checkpoint_epoch is not None:
        kwargs["checkpoint_epoch"] = args.checkpoint_epoch
    result = evaluate(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
