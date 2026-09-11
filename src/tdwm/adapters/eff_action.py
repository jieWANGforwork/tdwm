"""Public SWM policies for the provisional single-block EffAction protocol.

The environment buffer, normalization, and CEM implementation remain in
stable-worldmodel. EffActionPlan implements only its new learned solver.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import project_tasks_to_sphere_v1
from tdwm.methods.eff_action import (
    EFF_ACTION_RAW_ACTION_DIM,
    EFF_ACTION_STATE_DIM,
    eff_action_cost,
    require_frozen_eff_action_module,
)
from tdwm.methods.eff_action_plan import iterate_eff_action_plan


def canonical_eff_action_method(method: str) -> str:
    aliases = {
        "EffAction": "EffAction",
        "eff_action": "EffAction",
        "EffActionPlan": "EffActionPlan",
        "eff_action_plan": "EffActionPlan",
    }
    if method not in aliases:
        raise ValueError("method must be EffAction or EffActionPlan.")
    return aliases[method]


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_eff_action_planning(planning: Mapping[str, Any], method: str) -> None:
    """Reject an unagreed multi-block plan instead of silently changing it."""

    method = canonical_eff_action_method(method)
    for key, expected in {
        "horizon": 1,
        "receding_horizon": 1,
        "action_block": 5,
        "history_len": 1,
    }.items():
        if type(planning.get(key)) is not int or planning[key] != expected:
            raise ValueError(f"Current {method} requires planning.{key}={expected}.")
    if planning.get("warm_start") is not False:
        raise ValueError("A fully executed single block requires warm_start=false.")
    if type(planning.get("planning_seed")) is not int or planning["planning_seed"] < 0:
        raise ValueError("planning_seed must be an explicit nonnegative integer.")
    _positive_integer(planning.get("iterations"), "iterations")
    if method == "EffAction":
        candidates = _positive_integer(planning.get("candidates"), "candidates")
        elites = _positive_integer(planning.get("elites"), "elites")
        if elites < 2 or elites > candidates:
            raise ValueError("CEM requires 2 <= elites <= candidates for finite std.")
        _positive_integer(planning.get("solver_batch_size"), "solver_batch_size")
        scale = planning.get("initial_variance")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (float, int))
            or not (math.isfinite(scale) and scale > 0)
        ):
            raise ValueError("initial_variance must be finite and positive.")
        if any(planning.get(key) is not None for key in ("lower_bound", "upper_bound")):
            raise ValueError(
                "Public CEM uses unbounded normalized candidates; action bounds "
                "cannot be claimed by clamping only inside the cost model."
            )
    else:
        if planning.get("initialization") not in {"zeros", "normal"}:
            raise ValueError("Planner initialization must be zeros or normal.")
        std = planning.get("initial_std")
        if (
            isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not (math.isfinite(std) and std >= 0)
        ):
            raise ValueError("Planner initial_std must be explicit, finite and >= 0.")
        if planning["initialization"] == "normal" and std <= 0:
            raise ValueError("Normal initialization requires initial_std > 0.")
        for key in ("lower_bound", "upper_bound"):
            if key not in planning:
                raise ValueError(f"Planner requires explicit normalized {key}.")
        lower = torch.as_tensor(planning["lower_bound"], dtype=torch.float32)
        upper = torch.as_tensor(planning["upper_bound"], dtype=torch.float32)
        try:
            lower = torch.broadcast_to(lower, (EFF_ACTION_RAW_ACTION_DIM,))
            upper = torch.broadcast_to(upper, (EFF_ACTION_RAW_ACTION_DIM,))
        except RuntimeError as error:
            raise ValueError(
                "Bounds must be scalar or 25D normalized bounds."
            ) from error
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("Planner action bounds must be finite.")
        if bool((lower > upper).any()):
            raise ValueError("Planner lower_bound must not exceed upper_bound.")


class EffActionCostModel(nn.Module):
    """J=-distance/(V(G(z,E_A(a),m),m)+epsilon), without any F rollout."""

    def __init__(
        self,
        world_model: nn.Module,
        successor: nn.Module,
        value: nn.Module,
        *,
        epsilon: float,
    ) -> None:
        super().__init__()
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        self.world_model = world_model
        self.successor = successor
        self.value = value
        self.epsilon = float(epsilon)
        for name, module in (
            ("world_model", world_model),
            ("successor", successor),
            ("value", value),
        ):
            require_frozen_eff_action_module(module, name)

    def encode_state_goal(
        self, info: dict[str, Any], *, samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the single real state and goal; cache only in this solve dict.

        Without a candidate axis tensors have shape [B,1,...]. Public CEM
        expands them to [B,S,1,...]. Cached emb/goal_emb follow these same axes.
        Historical actions and privileged environment state never enter G/V.
        """

        parameter = next(self.successor.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        dtype = parameter.dtype if parameter is not None else torch.float32

        def latent(image_key: str, embedding_key: str) -> torch.Tensor:
            embedding = info.get(embedding_key)
            if embedding is None:
                pixels = info.get(image_key)
                expected_ndim = 6 if samples is not None else 5
                if not torch.is_tensor(pixels) or pixels.ndim != expected_ndim:
                    raise ValueError(
                        f"{image_key} must contain one batched image frame."
                    )
                if samples is not None:
                    if pixels.shape[1] != samples:
                        raise ValueError(
                            f"{image_key} candidate axis differs from CEM."
                        )
                    pixels = pixels[:, 0]
                if pixels.shape[1] != 1:
                    raise ValueError(
                        "EffAction accepts exactly one real current frame."
                    )
                with torch.no_grad():
                    embedding = self.world_model.encode(
                        {"pixels": pixels.to(device=device, dtype=dtype)}
                    )["emb"]
                if samples is not None:
                    embedding = embedding.unsqueeze(1).expand(-1, samples, -1, -1)
                info[embedding_key] = embedding
            expected_ndim = 4 if samples is not None else 3
            if not torch.is_tensor(embedding) or embedding.ndim != expected_ndim:
                raise ValueError(f"{embedding_key} has incompatible batch/time axes.")
            if samples is not None:
                if embedding.shape[1] != samples:
                    raise ValueError(
                        f"{embedding_key} has an incompatible candidate axis."
                    )
                embedding = embedding[:, 0]
            if embedding.shape[1:] != (1, EFF_ACTION_STATE_DIM):
                raise ValueError(f"{embedding_key} must encode one 192D frame.")
            embedding = embedding[:, 0].to(device=device, dtype=dtype).detach()
            if not bool(torch.isfinite(embedding).all()):
                raise ValueError(f"{embedding_key} must be finite.")
            return embedding

        state = latent("pixels", "emb")
        goal = latent("goal", "goal_emb")
        if state.shape != goal.shape:
            raise ValueError("Current state and goal batches must align.")
        task = project_tasks_to_sphere_v1(goal)
        return state, goal, task

    def get_cost(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        if action_candidates.ndim != 4 or action_candidates.shape[2:] != (1, 25):
            raise ValueError("EffAction candidates must have shape [B,S,1,25].")
        batch, samples = action_candidates.shape[:2]
        if batch <= 0 or samples <= 0:
            raise ValueError("Candidate batch and sample axes must be nonempty.")
        state, goal, task = self.encode_state_goal(info_dict, samples=samples)
        if state.shape[0] != batch:
            raise ValueError("Candidate and observation batches differ.")

        def expand(x: torch.Tensor) -> torch.Tensor:
            return x[:, None].expand(batch, samples, EFF_ACTION_STATE_DIM)

        return eff_action_cost(
            self.successor,
            self.value,
            self.world_model.action_encoder,
            state=expand(state),
            raw_action=action_candidates[:, :, 0],
            task=expand(task),
            goal=expand(goal),
            epsilon=self.epsilon,
        )

    def criterion(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.get_cost(info_dict, action_candidates)


class EffActionPlanSolver:
    """Learned action updater implementing the public SWM Solver protocol."""

    def __init__(
        self,
        model: EffActionCostModel,
        planner: nn.Module,
        *,
        planning: Mapping[str, Any],
        device: str | torch.device,
    ) -> None:
        validate_eff_action_planning(planning, "EffActionPlan")
        require_frozen_eff_action_module(planner, "planner")
        self.model, self.planner = model, planner
        self.planning = dict(planning)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(
            planning["planning_seed"]
        )
        self._n_envs: int | None = None

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        for key in (
            "horizon",
            "receding_horizon",
            "action_block",
            "history_len",
            "warm_start",
        ):
            if getattr(config, key) != self.planning[key]:
                raise ValueError(
                    f"SWM PlanConfig.{key} differs from the learned solver."
                )
        if not hasattr(action_space, "shape") or len(action_space.shape) < 2:
            raise ValueError("SWM must provide a batched continuous action space.")
        if int(np.prod(action_space.shape[1:])) != 5 or action_space.shape[0] != n_envs:
            raise ValueError("EffActionPlan requires the batched Cube 5D action space.")
        self._n_envs = _positive_integer(n_envs, "n_envs")

    @property
    def n_envs(self) -> int:
        if self._n_envs is None:
            raise RuntimeError("The learned solver must be configured by SWM first.")
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return EFF_ACTION_RAW_ACTION_DIM

    @property
    def horizon(self) -> int:
        return 1

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        _ = self.n_envs
        if init_action is not None:
            raise ValueError(
                "The one-block protocol has no unexecuted warm-start tail."
            )
        started = time.perf_counter()
        state, goal, task = self.model.encode_state_goal(info_dict)
        shape = (state.shape[0], EFF_ACTION_RAW_ACTION_DIM)
        if self.planning["initialization"] == "zeros":
            initial = torch.zeros(shape, device=self.device, dtype=state.dtype)
        else:
            initial = (
                torch.randn(
                    shape,
                    device=self.device,
                    dtype=state.dtype,
                    generator=self.generator,
                )
                * self.planning["initial_std"]
            )
        plan = iterate_eff_action_plan(
            self.planner,
            self.model.successor,
            self.model.value,
            self.model.world_model.action_encoder,
            state=state,
            initial_action=initial,
            goal=goal,
            task=task,
            iterations=self.planning["iterations"],
            epsilon=self.model.epsilon,
            lower_bound=self.planning["lower_bound"],
            upper_bound=self.planning["upper_bound"],
            track_grad=False,
        )
        return {
            "actions": plan.action.detach().unsqueeze(1).cpu(),
            "costs": plan.final_cost.detach().cpu().tolist(),
            "iterations": self.planning["iterations"],
            "elapsed_seconds": time.perf_counter() - started,
        }


def make_eff_action_policy(
    *,
    world_model: nn.Module,
    successor: nn.Module,
    value: nn.Module,
    planning: Mapping[str, Any],
    epsilon: float,
    method: str = "EffAction",
    planner: nn.Module | None = None,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
):
    """Assemble CEM or the learned updater with SWM's common action policy."""

    method = canonical_eff_action_method(method)
    validate_eff_action_planning(planning, method)
    if (method == "EffActionPlan") != (planner is not None):
        raise ValueError("Exactly EffActionPlan requires a planner module.")
    for module in (world_model, successor, value, planner):
        if module is not None:
            module.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    model = EffActionCostModel(world_model, successor, value, epsilon=epsilon)
    import stable_worldmodel as swm

    if method == "EffAction":
        solver = swm.solver.CEMSolver(
            model=model,
            batch_size=planning["solver_batch_size"],
            num_samples=planning["candidates"],
            n_steps=planning["iterations"],
            topk=planning["elites"],
            var_scale=planning["initial_variance"],
            device=device,
            seed=planning["planning_seed"],
        )
    else:
        solver = EffActionPlanSolver(model, planner, planning=planning, device=device)
    config = swm.PlanConfig(
        horizon=1, receding_horizon=1, history_len=1, action_block=5, warm_start=False
    )
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )


__all__ = [
    "EffActionCostModel",
    "EffActionPlanSolver",
    "canonical_eff_action_method",
    "make_eff_action_policy",
    "validate_eff_action_planning",
]
