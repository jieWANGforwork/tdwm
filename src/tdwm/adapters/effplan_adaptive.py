"""Variable-length, one-shot EffPlan via public SWM CEM/policy/wrapper APIs."""

import time

import gymnasium as gym
import numpy as np
import torch

from tdwm.adapters.effplan import EffCEMCost
from tdwm.methods.effplan import refine_state_path
from tdwm.methods.effplan_adaptive import adaptive_state_path
from tdwm.methods.effplan_safety import PlannerSafetyRuntime, require_finite


class AdaptiveTrackingCost(EffCEMCost):
    """No fixed H=5 restriction; F rolls out every actual proposed action."""

    def future_states(self, info, actions):
        if actions.ndim != 4 or actions.shape[-1] != 25 or actions.shape[-2] < 1:
            raise ValueError("Expected [B, candidates, H, 25] action blocks.")
        history = self._observed_frames(info)
        result = self.world_model.rollout(dict(info), actions, history_size=3)
        future = result["predicted_emb"][..., history:, :]
        if future.shape != (*actions.shape[:3], 192):
            raise ValueError("Variable-horizon LeWM rollout is misaligned.")
        require_finite(future, "adaptive F rollout")
        return future.detach()

    def get_cost(self, info_dict, action_candidates):
        future = self.future_states(info_dict, action_candidates)
        nodes = info_dict["effplan_nodes"]
        if nodes.shape != (*future.shape[:2], future.shape[-2]+1, 192):
            raise ValueError("K intermediate nodes require K+1 action blocks.")
        score = (future-nodes[..., 1:, :]).square().sum(-1).mean(-1)
        require_finite(score, "adaptive tracking score")
        return score


class PlanExecutionLimits:
    """Share chosen lengths with public gym wrappers; exhaustion is FAILURE.

    SWM policy transports a rectangular action tensor. Padding is never stepped:
    wrappers truncate each environment after its own actual plan length.
    """

    def __init__(self, budget):
        self.budget = budget
        self.wrappers = []

    def wrap(self, env):
        wrapper = PlanLengthLimit(env, self.budget)
        self.wrappers.append(wrapper)
        return wrapper

    def set_lengths(self, lengths):
        if len(lengths) != len(self.wrappers):
            raise ValueError("Plans must cover every evaluation environment once.")
        for wrapper, length in zip(self.wrappers, lengths, strict=True):
            wrapper.set_length(length)


class PlanLengthLimit(gym.Wrapper):
    """End at success or plan exhaustion, never replan or execute padded actions."""

    def __init__(self, env, budget):
        super().__init__(env)
        self.budget = budget
        self.limit = None
        self.steps = 0
        self.ended = False

    def reset(self, **kwargs):
        self.limit = None
        self.steps = 0
        self.ended = False
        return self.env.reset(**kwargs)

    def set_length(self, length):
        if self.limit is not None or type(length) is not int or not 1 <= length <= self.budget:
            raise ValueError("Invalid or repeated one-shot execution length.")
        self.limit = length

    def step(self, action):
        if self.limit is None or self.ended or self.steps >= self.limit:
            raise RuntimeError("Attempt to execute without a plan or past its end.")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        truncated = bool(truncated or (self.steps == self.limit and not terminated))
        self.ended = bool(terminated or truncated)
        return obs, reward, terminated, truncated, info


class AdaptiveEffPlanSolver:
    """One solve per episode, arbitrary K; internal refinement is pre-execution."""

    def __init__(self, *, model, planner, limits, safety, budget,
                 search_iterations=(3, 3, 3, 3, 3, 3, 4, 4, 4),
                 candidates=300, elites=30, seed=42, device="cuda",
                 epsilon=1e-6, minimum_relative_gain=1e-6,
                 dynamics_coefficient=0.1, efficiency_threshold=None, local_distance_limit=None):
        if local_distance_limit is not None:
            from tdwm.methods.effplan_efficiency import validate_local_distance_limit
            validate_local_distance_limit(local_distance_limit)
            if efficiency_threshold is None:
                raise ValueError("Distance gate requires an efficiency threshold.")
        self.local_distance_limit = local_distance_limit
        if efficiency_threshold is not None:
            from tdwm.methods.effplan_efficiency import validate_efficiency_threshold
            validate_efficiency_threshold(efficiency_threshold)
        self.efficiency_threshold = efficiency_threshold
        if budget % 5 or budget < 5 or len(search_iterations) < 1:
            raise ValueError("Budget must contain whole five-primitive-action blocks.")
        if any(p.requires_grad for p in planner.parameters()):
            raise ValueError("Adaptive evaluation requires frozen P.")
        self.model, self.planner, self.limits = model, planner, limits
        self.safety, self.budget = safety, budget
        self.search_iterations = tuple(search_iterations)
        self.candidates, self.elites, self.seed = candidates, elites, seed
        self.device = torch.device(device)
        self.epsilon, self.minimum_relative_gain = epsilon, minimum_relative_gain
        self.dynamics_coefficient = dynamics_coefficient
        self.solve_calls = 0
        self.records = []
        self.artifacts = []

    def configure(self, *, action_space, n_envs, config):
        if (config.horizon, config.receding_horizon, config.action_block, config.history_len) != (
            self.budget//5, self.budget//5, 5, 1
        ):
            raise ValueError("One-shot policy must buffer the full execution cap.")
        self.action_space, self.n_envs = action_space, n_envs
        self.horizon, self.action_dim = config.horizon, 25

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        import stable_worldmodel as swm

        if self.solve_calls:
            raise RuntimeError("One-shot evaluation must never replan after execution.")
        self.solve_calls += 1
        with torch.inference_mode(False), torch.no_grad():
            info = self.model.cached_context(info_dict, device=self.device)
            batch = len(info["emb"])
            if batch != self.n_envs:
                raise ValueError("First solve must plan for all paired episodes.")
            # Only transport padding. Truncation wrapper prevents its execution.
            padded = torch.zeros(batch, self.horizon, 25, device=self.device)
            lengths = []
            for i in range(batch):
                started = time.monotonic()
                sample = {k: v[i:i+1] for k, v in info.items() if torch.is_tensor(v)}
                safety = PlannerSafetyRuntime(self.safety)
                path_builder = adaptive_state_path
                criterion_kwargs = dict(minimum_relative_gain=self.minimum_relative_gain)
                if self.efficiency_threshold is not None:
                    from tdwm.methods.effplan_efficiency import efficiency_state_path
                    path_builder = efficiency_state_path
                    criterion_kwargs = dict(efficiency_threshold=self.efficiency_threshold,
                                            local_distance_limit=self.local_distance_limit)
                nodes, record = path_builder(
                    self.planner, sample["emb"][:, -1].clone(),
                    sample["goal_emb"][:, -1].clone(), self.model.target_critic,
                    max_blocks=self.horizon, safety=safety, epsilon=self.epsilon,
                    **criterion_kwargs,
                )
                initial_nodes = nodes.clone()
                horizon = nodes.shape[1]-1
                inner = swm.solver.CEMSolver(
                    model=self.model, batch_size=1, num_samples=self.candidates,
                    topk=self.elites, n_steps=self.search_iterations[0], var_scale=1.0,
                    device=self.device, seed=self.seed,
                )
                space = gym.spaces.Box(
                    low=np.asarray(self.action_space.low)[i:i+1],
                    high=np.asarray(self.action_space.high)[i:i+1], dtype=np.float32,
                )
                inner.configure(action_space=space, n_envs=1, config=swm.PlanConfig(
                    horizon=horizon, receding_horizon=horizon, action_block=5,
                    history_len=1, warm_start=False,
                ))
                actions = None
                for j, iterations in enumerate(self.search_iterations):
                    inner.n_steps = iterations
                    search_info = dict(sample, effplan_nodes=nodes)
                    actions = inner.solve(search_info, init_action=actions)["actions"].to(self.device)
                    if j+1 < len(self.search_iterations):
                        expanded = {k: v.unsqueeze(1) for k, v in search_info.items()}
                        future = self.model.future_states(expanded, actions[:, None])[:, 0].clone()
                        nodes = refine_state_path(
                            self.planner, nodes, self.model.target_critic,
                            predicted_future=future, epsilon=self.epsilon,
                            dynamics_coefficient=self.dynamics_coefficient, safety=safety,
                        )
                if actions.shape != (1, horizon, 25):
                    raise ValueError("Returned action count differs from adaptive path.")
                require_finite(actions, "adaptive final actions")
                padded[i, :horizon] = actions[0]
                lengths.append(horizon*5)
                record.update(
                    index=i, planning_calls=1, planned_primitive_steps=horizon*5,
                    search_iterations=list(self.search_iterations),
                    candidate_plans=self.candidates*sum(self.search_iterations),
                    candidate_macro_transitions=self.candidates*sum(self.search_iterations)*horizon,
                    returned_action_rerolls=len(self.search_iterations)-1,
                    planning_seconds=time.monotonic()-started, safety=safety.metrics(),
                )
                self.records.append(record)
                self.artifacts.append(dict(initial_nodes=initial_nodes.cpu(),
                                           final_nodes=nodes.cpu(), actions=actions.cpu()))
                print(f"Adaptive plan {i+1}/{batch}: {horizon} blocks ({horizon*5} steps)", flush=True)
            self.limits.set_lengths(lengths)
            return {"actions": padded}
