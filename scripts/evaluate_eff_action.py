#!/usr/bin/env python3
"""Evaluate EffAction CEM or EffActionPlan on an explicit Cube pair protocol."""

import argparse
import json
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--action-normalization", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--goal-offset", type=int, choices=(25, 50, 100))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument(
        "--planning-seed",
        type=int,
        choices=(42, 43, 44),
        help="Override planning.planning_seed only; leaves the pair list untouched.",
    )
    parser.add_argument("--allow-intermediate-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if not args.smoke and config.get("protocol_status") != "user_locked":
        raise ValueError("Formal evaluation requires the unresolved protocol choices to be fixed first.")
    if args.goal_offset is not None:
        config["selection"]["goal_offset"] = args.goal_offset
        config["planning"]["episode_budget"] = 2 * args.goal_offset
    if args.seed is not None:
        config["selection"]["seed"] = args.seed
        config["planning"]["planning_seed"] = args.seed
    if args.planning_seed is not None:
        config["planning"]["planning_seed"] = args.planning_seed
    if args.smoke:
        config["protocol_status"] = "provisional"
        config["run_mode"] = "smoke"
        # historical_cg3 reproduces a locked pair list under an exact hash
        # check, so its 50-pair count cannot be reduced for a smoke run.
        if config["selection"]["protocol"] != "historical_cg3":
            config["selection"]["episodes"] = 1
        config["planning"]["episode_budget"] = 5
        if config["method"] == "EffAction":
            config["planning"].update(candidates=4, elites=2, iterations=2)
    renderer = config["runtime"]["renderer"]
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        if os.environ.get(key) not in (None, "", renderer):
            raise ValueError(f"{key} disagrees with the configured renderer.")
        os.environ[key] = renderer
    # The renderer must be set before SWM can import MuJoCo/OpenGL.
    from tdwm.evaluation.eff_action import evaluate_eff_action
    from tdwm.training.eff_action import (
        load_eff_action_deployment,
        load_eff_action_normalization,
    )

    world, successor, value, planner, metadata = load_eff_action_deployment(
        args.checkpoint, pretrained=args.pretrained, method=config["method"], device=args.device,
        allow_intermediate=args.allow_intermediate_checkpoint,
    )
    train_config = metadata["training_config"]["planner"]
    if config["epsilon"] != train_config["epsilon"]:
        raise ValueError("Evaluation epsilon differs from training.")
    if not args.smoke and metadata["training_provenance"].get("run_mode") != "formal":
        raise ValueError("A smoke checkpoint cannot be used for a formal reported experiment.")
    if config["method"] == "EffActionPlan":
        for key in ("iterations", "lower_bound", "upper_bound"):
            if config["planning"][key] != train_config[key]:
                raise ValueError(f"Planner evaluation {key} differs from its training contract.")
        init = train_config["initialization"]
        if config["planning"]["initialization"] != init["distribution"]:
            raise ValueError("Planner reference initialization differs from training.")
        if init["distribution"] == "normal" and (init["mean"] != 0 or config["planning"]["initial_std"] != init["std"]):
            raise ValueError("Planner normal reference distribution differs from training.")
    normalization = load_eff_action_normalization(args.action_normalization, expected_sha256=metadata["training_provenance"]["column_normalization_sha256"])
    result = evaluate_eff_action(
        world_model=world, successor=successor, value=value, planner=planner,
        config=config, dataset_path=args.dataset, output_dir=args.output_dir,
        checkpoint_metadata=metadata, action_normalization=normalization,
        device=args.device, video=args.video,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
