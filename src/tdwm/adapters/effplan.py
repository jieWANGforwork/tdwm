"""Public SWM CEM integration for Eff terminal cost and EffPlan state paths.

Reuse the audited C3 observation/goal/LeWM-rollout interface, but substitute
Eff's G->V readout. No C/G checkpoints or training losses are modified.
"""

from __future__ import annotations

import math
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

EFF_TERMINAL_SCORE_MODE = "terminal_eff_cost"
EFF_LATENT_PATH_SCORE_MODE = "latent_path_eff_cost"
EFF_VALUE_PATH_SCORE_MODE = "value_path_eff_cost"
EFF_GOAL_DISTANCE_SCORE_MODE = "goal_distance_eff_cost"
EFF_VALUE_TO_GOAL_SUM_SCORE_MODE = "value_to_goal_sum_eff_cost"
EFF_VALUE_TO_GOAL_MEAN_SCORE_MODE = "value_to_goal_mean_eff_cost"

# Added on top of a separate terminal V(G(z_H, m_g), m_g) term.
EFF_CUMULATIVE_SCORE_MODES = frozenset(
    {
        EFF_LATENT_PATH_SCORE_MODE,
        EFF_VALUE_PATH_SCORE_MODE,
        EFF_GOAL_DISTANCE_SCORE_MODE,
    }
)
# V(z_k -> z_g) at every post-action state. The k=H term IS the terminal cost,
# so these modes are self-contained and add no separate terminal term.
EFF_SELF_CONTAINED_SCORE_MODES = frozenset(
    {
        EFF_VALUE_TO_GOAL_SUM_SCORE_MODE,
        EFF_VALUE_TO_GOAL_MEAN_SCORE_MODE,
    }
)
# Modes that need the pre-rollout anchor z_0.
EFF_START_ANCHORED_SCORE_MODES = frozenset(
    {EFF_LATENT_PATH_SCORE_MODE, EFF_VALUE_PATH_SCORE_MODE}
)
EFF_ACCUMULATED_SCORE_MODES = (
    EFF_CUMULATIVE_SCORE_MODES | EFF_SELF_CONTAINED_SCORE_MODES
)
EFF_SCORE_MODES = frozenset({EFF_TERMINAL_SCORE_MODE, *EFF_ACCUMULATED_SCORE_MODES})


def cumulative_eff_cost(
    *,
    future: torch.Tensor,
    goal: torch.Tensor,
    eff: EffModel,
    mode: str,
    target: bool,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulated rollout term for one Eff scoring mode.

    Two families, distinguished by whether a separate terminal term is added:

    Cumulative (a terminal V(G(z_H, m_g), m_g) is added separately):
      ``latent_path_eff_cost``       sum of geometric chords ||z_{k+1} - z_k||
      ``value_path_eff_cost``        sum of per-segment G -> V costs, i.e. the
                                     denominator of ``path_efficiency``
      ``goal_distance_eff_cost``     the endpoint chord ||z_H - z_g|| only

    Self-contained (the k=H term already is the terminal cost):
      ``value_to_goal_sum_eff_cost``   sum   of V(z_k -> z_g) over k = 1..H
      ``value_to_goal_mean_eff_cost``  mean  of V(z_k -> z_g) over k = 1..H

    The first family asks "how much latent change did the rollout burn"; the
    second asks "how far from the goal was it at every step".
    """
    if mode not in EFF_ACCUMULATED_SCORE_MODES:
        raise ValueError(
            f"Unsupported Eff cumulative score mode {mode!r}; expected one of "
            f"{sorted(EFF_ACCUMULATED_SCORE_MODES)}."
        )
    if mode in EFF_SELF_CONTAINED_SCORE_MODES:
        flat = future.reshape(-1, future.shape[-1])
        flat_goal = (
            goal.unsqueeze(-2).expand_as(future).reshape(-1, future.shape[-1])
        )
        costs = eff.value(flat, flat_goal, target=target)
        if costs.shape != (flat.shape[0],):
            raise ValueError("G -> V must return one scalar per action step.")
        costs = costs.reshape(*future.shape[:2], future.shape[-2])
        if mode == EFF_VALUE_TO_GOAL_MEAN_SCORE_MODE:
            return costs.mean(-1)
        return costs.sum(-1)
    if mode in EFF_START_ANCHORED_SCORE_MODES and start is None:
        raise ValueError(f"{mode} requires the pre-rollout anchor state z_0.")
    if mode == EFF_GOAL_DISTANCE_SCORE_MODE:
        return torch.linalg.vector_norm(future[..., -1, :] - goal, dim=-1)

    states = torch.cat([start.unsqueeze(-2), future], dim=-2)
    if mode == EFF_LATENT_PATH_SCORE_MODE:
        delta = states[..., 1:, :] - states[..., :-1, :]
        return torch.linalg.vector_norm(delta, dim=-1).sum(-1)
    segments = states.shape[-2] - 1
    costs = eff.value(
        states[..., :-1, :].reshape(-1, states.shape[-1]),
        states[..., 1:, :].reshape(-1, states.shape[-1]),
        target=target,
    )
    if costs.shape != (start.shape[0] * start.shape[1] * segments,):
        raise ValueError("G -> V must return one scalar per traversed segment.")
    return costs.reshape(*future.shape[:2], segments).sum(-1)


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

    def __init__(
        self,
        world_model: nn.Module,
        eff: EffModel,
        *,
        target: bool,
        score_mode: str = EFF_TERMINAL_SCORE_MODE,
        cumulative_weight: float = 1.0,
    ) -> None:
        if any(p.requires_grad for p in world_model.parameters()):
            raise ValueError("Eff planning requires a frozen LeWM.")
        if any(p.requires_grad for p in eff.parameters()):
            raise ValueError("Eff planning requires frozen G/V parameters.")
        if score_mode not in EFF_SCORE_MODES:
            raise ValueError(
                f"Unsupported Eff score mode {score_mode!r}; expected one of "
                f"{sorted(EFF_SCORE_MODES)}."
            )
        if not math.isfinite(cumulative_weight) or cumulative_weight < 0:
            raise ValueError("cumulative_weight must be finite and nonnegative.")
        super().__init__(
            world_model,
            EffReadout(eff, target=target),
            run_constant_shift_sanity=False,
        )
        # Deliberately not `self.score_mode`: the base class owns that attribute
        # and branches on it (STATE_V_SCORE_MODE vs first-action modes), so Eff
        # keeps its own mode under a separate name.
        self.eff_score_mode = score_mode
        self.cumulative_weight = float(cumulative_weight)

    @property
    def eff(self) -> EffModel:
        """The frozen G/V model, reached through the registered readout."""
        return self.target_critic.eff

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Terminal G -> V, optionally plus an accumulated rollout term."""
        if self.eff_score_mode == EFF_TERMINAL_SCORE_MODE:
            return super().get_cost(info_dict, action_candidates)

        future = self.future_states(info_dict, action_candidates)
        batch, samples = future.shape[:2]
        goal = self._goal_for_samples(
            info_dict, batch=batch, samples=samples, reference=action_candidates
        )
        accumulated = cumulative_eff_cost(
            future=future,
            goal=goal,
            eff=self.eff,
            mode=self.eff_score_mode,
            target=self.target_critic.target,
            start=(
                self._current_state_for_samples(
                    dict(info_dict),
                    batch=batch,
                    samples=samples,
                    reference=action_candidates,
                )
                if self.eff_score_mode in EFF_START_ANCHORED_SCORE_MODES
                else None
            ),
        )
        if self.eff_score_mode in EFF_SELF_CONTAINED_SCORE_MODES:
            # V(z_H -> z_g) is already the last term; adding it twice would
            # silently double-weight the terminal cost.
            total = accumulated
        else:
            terminal = self.target_critic(future[..., -1, :], goal)
            if terminal.ndim == 3 and terminal.shape[-1] == 1:
                terminal = terminal.squeeze(-1)
            if terminal.shape != (batch, samples):
                raise ValueError(
                    "Eff terminal G -> V must return one cost per candidate."
                )
            total = terminal + self.cumulative_weight * accumulated
        if total.shape != (batch, samples):
            raise ValueError("Eff accumulated cost must return one cost per candidate.")
        if not bool(torch.isfinite(total).all()):
            raise ValueError("Eff accumulated cost returned NaN or Inf.")
        if bool((total < 0).any()):
            raise ValueError("Eff accumulated cost must be nonnegative.")
        return total

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
