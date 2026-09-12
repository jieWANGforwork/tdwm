"""Public SWM CEM integration for Eff terminal cost and EffPlan state paths.

Reuse the audited C3 observation/goal/LeWM-rollout interface, but substitute
Eff's G->V readout. No C/G checkpoints or training losses are modified.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v1_c3 import ActorFreeTDLeWMV1C3
from tdwm.methods.eff import EffModel
from tdwm.methods.effplan import (
    StatePlanner,
    generate_state_path,
    refine_state_path,
)


class EffReadout(nn.Module):
    """Adapt the state/goal Costable readout to V(G(state, task), task)."""

    def __init__(self, eff: EffModel, *, target: bool) -> None:
        super().__init__()
        self.eff = eff
        self.target = target

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.eff.value(state, goal, target=self.target)


class EffCEMCost(ActorFreeTDLeWMV1C3):
    """Eff: full five-block F rollout, then terminal G->V, no action in G."""

    def __init__(self, world_model: nn.Module, eff: EffModel, *, target: bool) -> None:
        if any(p.requires_grad for p in world_model.parameters()):
            raise ValueError("Eff planning requires a frozen LeWM.")
        if any(p.requires_grad for p in eff.parameters()):
            raise ValueError("Eff planning requires frozen G/V parameters.")
        super().__init__(
            world_model,
            EffReadout(eff, target=target),
            run_constant_shift_sanity=False,
        )

    def future_states(self, info: dict, actions: torch.Tensor) -> torch.Tensor:
        """Return aligned z1..z5; every candidate action goes through F."""
        if actions.ndim != 4 or actions.shape[-2:] != (5, 25):
            raise ValueError("Eff uses five normalized 25D action blocks.")
        history = self._observed_frames(info)
        result = self.world_model.rollout(dict(info), actions, history_size=3)
        future = result["predicted_emb"][..., history:, :]
        if future.shape != (*actions.shape[:2], 5, 192):
            raise ValueError("LeWM future states do not align with five actions.")
        return future.detach()

    def cached_context(self, info: dict, *, device: torch.device) -> dict:
        """Encode the observed anchor and goal once, before any state planning."""
        tensors = {
            key: value.to(device)
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        if "pixels" not in tensors or tensors["pixels"].shape[1] != 1:
            raise ValueError("Eff protocol uses exactly one observed history frame.")
        batch = len(tensors["pixels"])
        expanded = {key: value.unsqueeze(1) for key, value in tensors.items()}
        reference = torch.zeros(batch, 1, 5, 25, device=device)
        with torch.no_grad():
            current = self._current_state_for_samples(
                expanded, batch=batch, samples=1, reference=reference
            )
            goal = self._goal_for_samples(
                expanded, batch=batch, samples=1, reference=reference
            )
        result = dict(info)
        result.update(tensors)
        result["emb"] = current.detach()  # [B, one observed frame, 192]
        result["goal_emb"] = goal.detach()
        return result


class EffPlanTrackingCost(EffCEMCost):
    """CEM tracks ALL planned nodes, including the goal, in one F rollout."""

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        nodes = info_dict.get("effplan_nodes")
        future = self.future_states(info_dict, action_candidates)
        if not torch.is_tensor(nodes) or nodes.shape != (*future.shape[:2], 6, 192):
            raise ValueError("State tracking requires z0..z5 for each candidate.")
        return (future - nodes[..., 1:, :]).square().sum(-1).mean(-1)


def distribute_cem_iterations(
    *, total_iterations: int, searches: int
) -> tuple[int, ...]:
    """Deterministic predeclared budget; final search receives any remainder."""
    if searches < 1 or total_iterations < searches:
        raise ValueError("Every CEM search needs at least one iteration.")
    base, extra = divmod(total_iterations, searches)
    return tuple(base + (i >= searches - extra) for i in range(searches))


class EffPlanSolver:
    """Learned state planning alternating with public SWM action CEM.

    CEM is not differentiated. After each search, reroll the RETURNED mean
    actions for dynamics feedback, never reuse the best candidate's states.
    After the last state update there is a final action search for that path.
    """

    def __init__(
        self,
        *,
        model: EffPlanTrackingCost,
        planner: StatePlanner,
        search_iterations: tuple[int, ...],
        candidates: int,
        elites: int,
        batch_size: int,
        seed: int,
        device: str | torch.device,
        epsilon: float,
        dynamics_coefficient: float,
    ) -> None:
        if len(search_iterations) < 2 or any(n < 1 for n in search_iterations):
            raise ValueError(
                "EffPlan requires improvement searches and a final search."
            )
        if any(p.requires_grad for p in planner.parameters()):
            raise ValueError("The deployed state planner must be frozen.")
        import stable_worldmodel as swm

        self.model, self.planner = model, planner
        self.search_iterations = search_iterations
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.dynamics_coefficient = dynamics_coefficient
        self.inner = swm.solver.CEMSolver(
            model=model,
            batch_size=batch_size,
            num_samples=candidates,
            topk=elites,
            n_steps=search_iterations[0],
            var_scale=1.0,
            device=device,
            seed=seed,
        )
        self.searches_per_solve = len(search_iterations)
        self.rollouts_per_solve = candidates * sum(search_iterations)
        self.last_diagnostics: dict[str, Any] = {}

    def configure(self, *, action_space, n_envs, config) -> None:
        if config.horizon != 5 or config.action_block != 5 or config.history_len != 1:
            raise ValueError("EffPlan requires H=5, action_block=5, history_len=1.")
        self.inner.configure(action_space=action_space, n_envs=n_envs, config=config)

    @property
    def n_envs(self) -> int:
        return self.inner.n_envs

    @property
    def action_dim(self) -> int:
        return self.inner.action_dim

    @property
    def horizon(self) -> int:
        return self.inner.horizon

    def __call__(self, *args, **kwargs) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        with torch.inference_mode(False), torch.no_grad():
            info = self.model.cached_context(info_dict, device=self.device)
            start = info["emb"][:, -1].clone()
            goal = info["goal_emb"][:, -1].clone()
            value = self.model.target_critic
            nodes = generate_state_path(
                self.planner, start, goal, value, horizon=5, epsilon=self.epsilon
            )
            actions = init_action
            output: dict[str, Any] = {}
            for index, iterations in enumerate(self.search_iterations):
                self.inner.n_steps = iterations
                search_info = dict(info, effplan_nodes=nodes)
                output = self.inner.solve(search_info, init_action=actions)
                actions = output["actions"].to(self.device)
                if index + 1 == len(self.search_iterations):
                    break
                expanded = {
                    key: val.unsqueeze(1)
                    for key, val in search_info.items()
                    if torch.is_tensor(val)
                }
                # Mean action sequence is not generally the best sampled plan.
                future = self.model.future_states(expanded, actions[:, None])[
                    :, 0
                ].clone()
                nodes = refine_state_path(
                    self.planner,
                    nodes,
                    value,
                    predicted_future=future,
                    epsilon=self.epsilon,
                    dynamics_coefficient=self.dynamics_coefficient,
                )
            self.last_diagnostics = {
                "search_iterations": list(self.search_iterations),
                "candidate_rollouts": self.rollouts_per_solve,
                "returned_action_rerolls": len(self.search_iterations) - 1,
                "state_updates": len(self.search_iterations) - 1,
                "final_search_after_last_state_update": True,
                "final_state_nodes": nodes.detach().cpu(),
            }
            return output
