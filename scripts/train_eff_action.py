#!/usr/bin/env python3
"""Train the explicitly configured EffAction and EffActionPlan stages."""

import argparse
import json

from tdwm.training.eff_action import train_eff_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--latent-store", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-normalization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after", choices=("gv", "stage1", "stage2"), default="stage2")
    args = vars(parser.parse_args())
    args["config_path"] = args.pop("config")
    print(json.dumps(train_eff_action(**args), indent=2))


if __name__ == "__main__":
    main()
