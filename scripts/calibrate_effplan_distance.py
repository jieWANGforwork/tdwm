"""Compute the approved training-only five-step distance P95, without GPU work."""

import argparse
import json

from tdwm.evaluation.effplan_distance_calibration import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-store", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = calibrate(store_path=args.latent_store, config_path=args.config,
                       output_path=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
