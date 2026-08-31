#!/usr/bin/env python3
"""Fail-closed acceptance for six formal V2-EMA-SG trainings."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_v2_ema_sg import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    LOCAL_PREDICTION,
    LOCAL_PREDICTION_TARGET,
    LOCAL_PREDICTION_TARGET_GRADIENT,
    METHOD_FAMILY,
    OBJECTIVE_VERSION,
    PREDICTOR_PARAMETERS,
    TRAINING_INITIALIZATION,
    TRAINING_STAGE,
    WORLD_MODEL_PARAMETERS,
    audit_training,
    write_training_acceptance,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all six formal Actor-Free TD-LeWM V2-EMA-SG runs and "
            "atomically publish training_acceptance.json."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bundle",
        help=(
            "EMA-SG launcher bundle root; trainings are read from "
            "<bundle>/formal."
        ),
    )
    source.add_argument(
        "--output-root",
        help="Root containing c,d,f,g1,g2,g3/seed_3072 EMA-SG runs.",
    )
    parser.add_argument(
        "--evidence-root",
        help=(
            "Optional launcher-evidence root containing <variant>/"
            "execution_evidence.json or <variant>.json."
        ),
    )
    parser.add_argument(
        "--output-json",
        help="Explicit acceptance path; overrides the layout-derived default.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_root = Path(args.bundle).expanduser().resolve() if args.bundle else None
    output_root = (
        bundle_root / "formal"
        if bundle_root is not None
        else Path(args.output_root).expanduser().resolve()
    )
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (bundle_root or output_root) / "training_acceptance.json"
    )
    try:
        acceptance = audit_training(
            output_root=output_root,
            evidence_root=args.evidence_root,
        )
    except Exception as error:  # preserve a machine-readable FAIL record
        acceptance = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output_root),
            "method_family": METHOD_FAMILY,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "stage": TRAINING_STAGE,
            "initialization": TRAINING_INITIALIZATION,
            "local_prediction": LOCAL_PREDICTION,
            "local_prediction_target": LOCAL_PREDICTION_TARGET,
            "local_prediction_target_gradient": (
                LOCAL_PREDICTION_TARGET_GRADIENT
            ),
            "training_revision": None,
            "seed": 3072,
            "expected_epoch": 10,
            "expected_global_step": 127_960,
            "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
            "predictor_parameter_count": PREDICTOR_PARAMETERS,
            "stable_worldmodel_version": "0.1.1",
            "variants": {},
            "warnings": [],
            "errors": [f"internal acceptance error: {error}"],
            "status": "FAIL",
        }
    try:
        write_training_acceptance(acceptance, output_json)
    except Exception as error:
        print(f"Could not publish {output_json}: {error}", file=sys.stderr)
        return 1
    print(f"V2-EMA-SG TRAINING ACCEPTANCE: {acceptance['status']} -> {output_json}")
    for warning in acceptance.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in acceptance.get("errors", []):
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if acceptance.get("status") in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
