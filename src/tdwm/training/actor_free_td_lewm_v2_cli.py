"""CLI harness shared by the six V2 coupled-Hybrid fine-tuning entries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

ProtocolLoader = Callable[[str | Path], dict[str, Any]]
Trainer = Callable[..., dict[str, Any]]


def build_actor_free_td_lewm_v2_parser(
    *,
    method_label: str,
    requires_neighbor_index: bool,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Fine-tune Actor-Free TD-JEPA {method_label} with the V2 "
            "coupled-Hybrid objective."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="V2 variant overlay protocol; its common protocol is resolved by extends.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited raw-clip Cube HDF5 or Lance dataset.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--initial-v1-checkpoint",
        required=True,
        help="Matching completed V1 deployment checkpoint for this variant.",
    )
    parser.add_argument(
        "--split-indices",
        required=True,
        help="Exact split_indices.npz shared with the corresponding V1 run.",
    )
    if requires_neighbor_index:
        parser.add_argument(
            "--neighbor-index",
            required=True,
            help="Audited training-only state-neighbor index required by G1.",
        )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def run_actor_free_td_lewm_v2_cli(
    *,
    method_label: str,
    requires_neighbor_index: bool,
    load_protocol: ProtocolLoader,
    train: Trainer,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    parser = build_actor_free_td_lewm_v2_parser(
        method_label=method_label,
        requires_neighbor_index=requires_neighbor_index,
    )
    args = parser.parse_args(argv)
    if not args.dataset:
        raise SystemExit("Pass --dataset or set TDWM_CUBE_DATASET.")
    if not args.smoke and args.max_steps is not None:
        parser.error("--max-steps is smoke-only; pass --smoke.")
    if not args.smoke and args.skip_validation:
        parser.error("--skip-validation is smoke-only; pass --smoke.")

    protocol = load_protocol(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(os.environ.get("TDWM_RUN_ROOT", "outputs"))
            / f"{protocol['method']}_cube_finetuning"
        )
    training_kwargs: dict[str, Any] = {
        "protocol_path": args.config,
        "dataset_path": args.dataset,
        "output_dir": output_dir,
        "seed": args.seed,
        "initial_v1_checkpoint_path": args.initial_v1_checkpoint,
        "split_indices_path": args.split_indices,
        "smoke": args.smoke,
        "resume": args.resume,
        "max_steps": args.max_steps,
        "skip_validation": args.skip_validation,
    }
    if requires_neighbor_index:
        training_kwargs["neighbor_index_path"] = args.neighbor_index
    result = train(**training_kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


__all__ = [
    "build_actor_free_td_lewm_v2_parser",
    "run_actor_free_td_lewm_v2_cli",
]
