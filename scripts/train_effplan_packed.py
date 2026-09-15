"""Packed-compute variable-length EffPlan training entrypoint; no implicit protocol choices."""

from __future__ import annotations

import argparse
import json

from tdwm.training.effplan_packed_run import run_packed_effplan_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "config",
        "latent-store",
        "terminal-metadata",
        "eff-checkpoint",
        "eff-manifest",
        "output-dir",
        "device",
    ):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--phase", choices=["generation", "refinement"], required=True)
    parser.add_argument("--lewm-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    parser.add_argument("--branch-from", help="Dense-to-sparse continuation retaining optimizer, RNG and step.")
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="Pause at this absolute update with a resumable checkpoint; keep the full LR budget.",
    )
    args = parser.parse_args()
    result = run_packed_effplan_training(
        config_path=args.config,
        phase=args.phase,
        latent_store=args.latent_store,
        terminal_metadata=args.terminal_metadata,
        eff_checkpoint=args.eff_checkpoint,
        eff_manifest=args.eff_manifest,
        lewm_checkpoint=args.lewm_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
        init_from=args.init_from,
        stop_after_updates=args.stop_after_updates,
        branch_from=args.branch_from,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
