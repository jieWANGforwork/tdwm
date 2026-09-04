"""Small CLI harness shared by the standalone frozen methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

ProtocolLoader = Callable[[str | Path], dict[str, Any]]
Trainer = Callable[..., dict[str, Any]]


def build_frozen_actor_free_td_parser(
    *,
    method_label: str,
    requires_neighbor_index: bool,
    requires_v1_c_checkpoint: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train standalone Actor-Free TD-LeWM method {method_label}."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Explicit resolved protocol for the completed pretrained checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TDWM_CUBE_DATASET"),
        help="Path to the audited Cube Lance dataset.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--initial-world-model-checkpoint",
        required=True,
        help="Completed epoch-10 Stable World Model public export.",
    )
    parser.add_argument(
        "--frozen-latent-store",
        required=True,
        help="Immutable global-row LeWM latent/action store.",
    )
    if requires_v1_c_checkpoint:
        parser.add_argument(
            "--initial-v1-c-checkpoint",
            required=True,
            help=(
                "Completed V1-C epoch-10 deployment checkpoint whose complete "
                "model parameter state initializes V1-C2."
            ),
        )
    parser.add_argument(
        "--split-indices",
        required=True,
        help=(
            "Prebuilt split_indices.npz used by this run and any dependent "
            "state-bank/index artifacts."
        ),
    )
    if requires_neighbor_index:
        parser.add_argument(
            "--neighbor-index",
            required=True,
            help="Audited training-only state-neighbor index required by G1.",
        )
    return parser


def run_frozen_actor_free_td_cli(
    *,
    method_label: str,
    requires_neighbor_index: bool,
    requires_v1_c_checkpoint: bool = False,
    load_protocol: ProtocolLoader,
    train: Trainer,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    parser = build_frozen_actor_free_td_parser(
        method_label=method_label,
        requires_neighbor_index=requires_neighbor_index,
        requires_v1_c_checkpoint=requires_v1_c_checkpoint,
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
            / f"{protocol['method']}_cube_training"
        )
    training_kwargs: dict[str, Any] = {
        "protocol_path": args.config,
        "dataset_path": args.dataset,
        "output_dir": output_dir,
        "seed": args.seed,
        "smoke": args.smoke,
        "resume": args.resume,
        "max_steps": args.max_steps,
        "skip_validation": args.skip_validation,
        "initial_world_model_checkpoint_path": args.initial_world_model_checkpoint,
        "frozen_latent_store_path": args.frozen_latent_store,
        "split_indices_path": args.split_indices,
    }
    if requires_neighbor_index:
        training_kwargs["neighbor_index_path"] = args.neighbor_index
    if requires_v1_c_checkpoint:
        training_kwargs["initial_v1_c_checkpoint_path"] = args.initial_v1_c_checkpoint
    result = train(**training_kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


__all__ = [
    "build_frozen_actor_free_td_parser",
    "run_frozen_actor_free_td_cli",
]
