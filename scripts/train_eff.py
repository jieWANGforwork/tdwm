"""Eff metadata preparation and full training entrypoint; no implicit protocol choices."""

from __future__ import annotations

import argparse
import json

from tdwm.training.eff_protocol import (
    prepare_eff_terminal_metadata,
    train_eff_from_protocol,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-metadata")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--output-dir", required=True)
    train = sub.add_parser("train")
    for arg in ("config", "latent-store", "terminal-metadata", "output-dir", "device"):
        train.add_argument("--" + arg, required=True)
    train.add_argument("--resume")
    args = parser.parse_args()
    if args.command == "prepare-metadata":
        result = prepare_eff_terminal_metadata(
            config_path=args.config,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
        )
    else:
        result = train_eff_from_protocol(
            config_path=args.config,
            latent_store=args.latent_store,
            terminal_metadata=args.terminal_metadata,
            output_dir=args.output_dir,
            device=args.device,
            resume=args.resume,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
