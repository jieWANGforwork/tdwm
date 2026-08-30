#!/usr/bin/env python3
"""Fail-closed acceptance for six formal Actor-Free TD-LeWM V2 trainings."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tdwm.results.actor_free_td_lewm_v2 import (
    PREDICTOR_PARAMETERS,
    WORLD_MODEL_PARAMETERS,
    audit_training,
    write_training_acceptance,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all six formal Actor-Free TD-LeWM V2 training runs and "
            "atomically publish training_acceptance.json."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bundle",
        help=(
            "Launcher bundle root; trainings are read from <bundle>/formal and "
            "acceptance defaults to <bundle>/training_acceptance.json."
        ),
    )
    source.add_argument(
        "--output-root",
        help="Root containing c,d,f,g1,g2,g3/seed_3072 training directories.",
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
    print(f"V2 TRAINING ACCEPTANCE: {acceptance['status']} -> {output_json}")
    for warning in acceptance.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in acceptance.get("errors", []):
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if acceptance.get("status") in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
