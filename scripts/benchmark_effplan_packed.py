"""Read-only paired timing/gradient check on an immutable P checkpoint.

Never writes a checkpoint or steps an optimizer. Both backends use the same
sampled real batch, weights and solver RNG. Calibration budget stays unchanged.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tdwm.adapters.effplan_adaptive import AdaptiveTrackingCost
from tdwm.methods.effplan import StatePlanner
from tdwm.training.eff_protocol import load_eff_protocol, load_eff_replays, sha256_file
from tdwm.training.eff_runtime import load_eff_model
from tdwm.training.effplan_run import _planner_settings, load_frozen_world_model
from tdwm.training.effplan_packed_runtime import PackedEffPlanTrainer
from tdwm.training.effplan_variable_runtime import VariableEffPlanTrainer
from tdwm.training.effplan_variable_data import sample_variable_planner_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "latent-store", "terminal-metadata", "eff-checkpoint",
                 "eff-manifest", "planner-checkpoint", "output", "device"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--lewm-checkpoint")
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(args.planner_checkpoint, weights_only=False, map_location="cpu")
    phase = payload["settings"]["phase"]
    config = load_eff_protocol(args.config, stage="planner_"+phase)
    settings = _planner_settings(config["planner_"+phase])
    replay, _, source = load_eff_replays(config=config, latent_store=args.latent_store,
                                        terminal_metadata=args.terminal_metadata)
    emeta = json.loads(Path(args.eff_manifest).read_text())
    assert source == emeta["identity"]["source"]
    assert sha256_file(args.eff_checkpoint) == payload["identity"]["source"]["eff_checkpoint_sha256"]
    eff, _ = load_eff_model(args.eff_checkpoint, expected_identity=emeta["identity"],
        expected_global_step=emeta["completed_updates"], device=args.device,
        expected_v_parameterization=config["eff_training"]["settings"].get("v_parameterization", "total_work"))
    tracking = None
    if phase == "refinement":
        assert sha256_file(args.lewm_checkpoint) == source["lewm_checkpoint_sha256"]
        world, _ = load_frozen_world_model(args.lewm_checkpoint, args.device, output.parent)
        tracking = AdaptiveTrackingCost(world, eff, target=settings.target_readout)
    trainers = {}
    for name, cls in (("old", VariableEffPlanTrainer), ("packed", PackedEffPlanTrainer)):
        t = cls(planner=StatePlanner(hidden_dim=payload["planner_hidden_dim"]), eff=eff,
                settings=settings, identity=payload["identity"], device=args.device,
                tracking_model=tracking)
        t.resume(args.planner_checkpoint)
        trainers[name] = t
    batch = sample_variable_planner_paths(replay,
        batch_size=config["planner_"+phase]["run"]["batch_size"],
        rng=trainers["old"].rng,
        backup_primitive_steps=config["eff_training"]["settings"]["backup_primitive_steps"],
        epsilon=settings.epsilon)
    cuda = torch.device(args.device).type == "cuda"
    sync = lambda: torch.cuda.synchronize(args.device) if cuda else None
    report = {"phase": phase, "source_step": payload["global_step"],
              "checkpoint_sha256": sha256_file(args.planner_checkpoint), "cases": {}}
    for case in (["ordinary", "calibration"] if phase=="refinement" else ["ordinary"]):
        times = {"old": [], "packed": []}
        metrics, gradients, peaks = {}, {}, {}
        repeats = 1 if case=="calibration" else args.repeats
        for repeat in range(repeats+1):
            for name, t in trainers.items():
                t.global_step = 0 if case=="ordinary" else settings.calibration_interval-1
                if t.solver is not None:
                    t.solver.torch_gen.set_state(payload["cem_rng"])
                t.optimizer.zero_grad(set_to_none=True)
                sync()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(args.device)
                started = time.perf_counter()
                metrics[name], _ = t._aggregate(batch, validation=False)
                sync()
                elapsed = time.perf_counter()-started
                if repeat:
                    times[name].append(elapsed)
                gradients[name] = torch.cat([x.grad.detach().flatten() for x in t.planner.parameters()]).cpu()
                peaks[name] = torch.cuda.max_memory_allocated(args.device) if cuda else None
        relative = float((gradients["old"]-gradients["packed"]).norm()/gradients["old"].norm().clamp_min(1e-12))
        old_loss, new_loss = metrics["old"]["total_loss"], metrics["packed"]["total_loss"]
        assert abs(old_loss-new_loss) <= 1e-4 * max(abs(old_loss), 1), (old_loss, new_loss)
        assert relative < 1e-3, relative
        result = dict(seconds={k:float(np.mean(v)) for k,v in times.items()},
            speedup=float(np.mean(times["old"])/np.mean(times["packed"])),
            old_loss=old_loss, packed_loss=new_loss, relative_gradient_error=relative,
            peak_allocated_bytes=peaks)
        report["cases"][case] = result
        print(json.dumps({case: result}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
