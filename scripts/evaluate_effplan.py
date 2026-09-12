"""Prepare shared episode pairs or run a full Eff/EffPlan formal evaluation."""

from __future__ import annotations

import argparse
import json

from tdwm.evaluation.effplan import evaluate_effplan, prepare_eff_selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("prepare-selections")
    for name in ("config", "terminal-metadata", "output-dir"):
        select.add_argument("--" + name, required=True)
    evaluate = sub.add_parser("evaluate")
    for name in (
        "config",
        "dataset",
        "lewm-checkpoint",
        "selection",
        "output-dir",
        "device",
    ):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument(
        "--method", choices=["F-only", "Eff", "EffPlan"], required=True
    )
    for name in (
        "eff-checkpoint",
        "eff-manifest",
        "planner-checkpoint",
        "planner-manifest",
    ):
        evaluate.add_argument("--" + name)
    evaluate.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-selections":
        result = prepare_eff_selections(
            config_path=args.config,
            terminal_metadata=args.terminal_metadata,
            output_dir=args.output_dir,
        )
    else:
        result = evaluate_effplan(
            config_path=args.config,
            dataset_path=args.dataset,
            lewm_checkpoint=args.lewm_checkpoint,
            selection_path=args.selection,
            output_dir=args.output_dir,
            method=args.method,
            device=args.device,
            eff_checkpoint=args.eff_checkpoint,
            eff_manifest=args.eff_manifest,
            planner_checkpoint=args.planner_checkpoint,
            planner_manifest=args.planner_manifest,
            video=args.video,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
