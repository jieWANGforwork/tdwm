"""Opt-in primitive-time adapter for the installed public SWM iCEM.

Only the search representation changes: [H, 5*D] <-> [5*H, D].
F, state tracking, P refinement, inverse normalization and execution retain
their original block interfaces. No optimizer or dependency code is copied.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class EffPlanICEMSettings:
    noise_beta: float = 2.0
    alpha: float = 0.1
    n_elite_keep: int = 5
    return_mean: bool = False

    def __post_init__(self):
        if (isinstance(self.noise_beta, bool)
                or not math.isfinite(self.noise_beta) or self.noise_beta < 0):
            raise ValueError("noise_beta must be finite and nonnegative")
        if (isinstance(self.alpha, bool)
                or not math.isfinite(self.alpha) or not 0 <= self.alpha < 1):
            raise ValueError("alpha must be finite and in [0,1)")
        if type(self.n_elite_keep) is not int or self.n_elite_keep < 0:
            raise ValueError("n_elite_keep must be a nonnegative integer")
        if type(self.return_mean) is not bool or self.return_mean:
            raise ValueError("This predeclared iCEM comparison returns an evaluated best candidate")

    def manifest(self):
        return dict(
            **asdict(self),
            implementation="stable_worldmodel.solver.ICEMSolver", version="0.1.1",
            sampling_layout="25 primitive timesteps x 5 action dimensions",
            world_model_layout="5 action blocks x 25 ordered components",
            action_units="unchanged dataset-standardized coordinates",
            action_clipping="none; infinite search bounds preserve original P+CEM support",
            score="unchanged state_path_tracking; no perturbation-risk term",
            population_decay=False, cross_solve_elite_reuse=False,
            iteration_elites="reuse within each fixed-node search only",
            feedback="reroll the returned best sequence before updating P",
            checkpoints_retrained=False,
            rng_note="seed 42; different sampler, not identical candidate random numbers",
        )


def load_effplan_icem(path: str | Path) -> EffPlanICEMSettings:
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("iCEM settings must be a JSON object")
    return EffPlanICEMSettings(**values)


def blocks_to_primitives(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim < 2 or actions.shape[-1] != 25:
        raise ValueError("Cube block actions must end in [H,25]")
    return actions.reshape(*actions.shape[:-2], actions.shape[-2] * 5, 5)


def primitives_to_blocks(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim < 2 or actions.shape[-1] != 5 or actions.shape[-2] % 5:
        raise ValueError("Cube primitive actions must end in [5*H,5]")
    return actions.reshape(*actions.shape[:-2], actions.shape[-2] // 5, 25)


class PrimitiveTrackingCost(nn.Module):
    """Only reshape candidates; never change values, context, scores or F."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_cost(self, info_dict, action_candidates):
        return self.model.get_cost(info_dict, primitives_to_blocks(action_candidates))


class PrimitiveICEMSolver:
    """Expose the original block solver interface around public SWM iCEM."""

    def __init__(self, *, model, settings: EffPlanICEMSettings, **kwargs):
        import stable_worldmodel as swm

        if settings.n_elite_keep > min(kwargs["topk"], kwargs["num_samples"] - 1):
            raise ValueError("Retained elites must fit both the elite set and candidate pool")
        self.settings = settings
        self.model = model
        self.optimizer = swm.solver.ICEMSolver(
            model=PrimitiveTrackingCost(model), **asdict(settings), **kwargs,
        )

    def configure(self, *, action_space, n_envs, config):
        import stable_worldmodel as swm
        from gymnasium.spaces import Box

        if (config.horizon, config.action_block, config.history_len) != (5, 5, 1):
            raise ValueError("This iCEM comparison requires H5 and five-action Cube blocks")
        if not isinstance(action_space, Box) or action_space.shape != (n_envs, 5):
            raise ValueError("Expected the existing batched five-dimensional Cube action space")
        self.block_config = config
        # This is the SEARCH space in standardized coordinates, not the real
        # environment's action_space. Do not accidentally clamp normalized
        # candidates to raw [-1,1], or add the previous action-bounds ablation.
        search_space = Box(-np.inf, np.inf, shape=(n_envs, 5), dtype=action_space.dtype)
        primitive_config = swm.PlanConfig(
            horizon=25, receding_horizon=config.receding_horizon * 5,
            history_len=1, action_block=1, warm_start=config.warm_start,
        )
        self.optimizer.configure(action_space=search_space, n_envs=n_envs, config=primitive_config)

    @property
    def n_steps(self):
        return self.optimizer.n_steps

    @n_steps.setter
    def n_steps(self, value):
        self.optimizer.n_steps = value

    @property
    def n_envs(self):
        return self.optimizer.n_envs

    @property
    def horizon(self):
        return self.block_config.horizon

    @property
    def action_dim(self):
        return 25

    def solve(self, info_dict, init_action=None):
        # SWM updates its initialized mean in place when returning actions.
        # Do not let that alias the caller's previous block action sequence.
        initial = None if init_action is None else blocks_to_primitives(init_action).clone()
        output = self.optimizer.solve(info_dict, init_action=initial)
        output["actions"] = primitives_to_blocks(output["actions"])
        for key in ("mean", "var"):
            output[key] = [primitives_to_blocks(value) for value in output[key]]
        return output

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)
