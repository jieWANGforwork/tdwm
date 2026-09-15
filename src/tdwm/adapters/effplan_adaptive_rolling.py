"""Adaptive-length decisions, repeated from real observations until total budget.

Delegate preprocessing, action normalization and buffering to public SWM
WorldModelPolicy. A separate single-environment delegate handles each ragged plan.
No environment wrapper truncates at a plan boundary in this mode.
"""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch

from tdwm.adapters.effplan_adaptive import AdaptiveEffPlanSolver


class DecisionLength:
    """A decision length, NOT an environment termination condition."""

    def set_lengths(self, lengths):
        if len(lengths) != 1:
            raise ValueError("A ragged decision must belong to exactly one episode.")
        self.steps = lengths[0]


class AdaptiveDecisionSolver(AdaptiveEffPlanSolver):
    """Reuse the existing adaptive search, return actual actions without padding."""

    def solve(self, info_dict, init_action=None):
        output = super().solve(info_dict, init_action)
        blocks = self.records[0]["action_blocks"]
        # Public policy reads cfg AFTER solve, before enqueuing returned actions.
        self.policy.cfg = swm.PlanConfig(
            horizon=blocks, receding_horizon=blocks, action_block=5,
            history_len=1, warm_start=False,
        )
        return dict(output, actions=output["actions"][:, :blocks])


class AdaptiveRollingPolicy:
    """Execute all actions of each adaptive decision, then replan if unfinished."""

    def __init__(self, *, model, planner, safety, budget, device,
                 search_iterations, epsilon, dynamics_coefficient,
                 process=None, transform=None, efficiency_threshold=None, local_distance_limit=None,
                 distance_only=False):
        from tdwm.methods.effplan_efficiency import validate_distance_only
        validate_distance_only(distance_only, local_distance_limit, efficiency_threshold)
        if budget < 5 or budget % 5:
            raise ValueError("Episode budget must be whole five-step blocks.")
        self.budget = budget
        self.solver_kwargs = dict(
            model=model, planner=planner, safety=safety, device=device,
            search_iterations=tuple(search_iterations), epsilon=epsilon,
            dynamics_coefficient=dynamics_coefficient, minimum_relative_gain=1e-6,
        )
        if efficiency_threshold is not None:
            from tdwm.methods.effplan_efficiency import validate_efficiency_threshold
            validate_efficiency_threshold(efficiency_threshold)
            self.solver_kwargs["efficiency_threshold"] = efficiency_threshold
        self.process, self.transform = process or {}, transform or {}
        if distance_only:
            self.solver_kwargs["distance_only"] = True
        if local_distance_limit is not None:
            from tdwm.methods.effplan_efficiency import validate_local_distance_limit
            validate_local_distance_limit(local_distance_limit)
            if efficiency_threshold is None and not distance_only:
                raise ValueError("Distance gate requires an efficiency threshold.")
            self.solver_kwargs["local_distance_limit"] = local_distance_limit

    def set_env(self, env):
        self.env = env
        n = env.num_envs
        self.executed_steps = np.zeros(n, dtype=int)
        self.pending_steps = np.zeros(n, dtype=int)
        self.policies = [None] * n
        self.records = [[] for _ in range(n)]
        self.artifacts = [[] for _ in range(n)]

    def _decision(self, index):
        remaining = self.budget - int(self.executed_steps[index])
        if remaining < 5 or remaining % 5:
            raise RuntimeError("A new decision requires remaining whole action blocks.")
        length = DecisionLength()
        solver = AdaptiveDecisionSolver(
            **self.solver_kwargs, limits=length, budget=remaining,
        )
        delegate = swm.policy.WorldModelPolicy(
            solver, swm.PlanConfig(remaining//5, remaining//5, action_block=5,
                                  history_len=1, warm_start=False),
            process=self.process, transform=self.transform,
        )
        # Public policy only needs these public EnvPool attributes. No rollout,
        # simulator, normalization or action-buffer implementation is duplicated.
        space = self.env.action_space
        view = SimpleNamespace(
            num_envs=1, single_action_space=self.env.single_action_space,
            action_space=gym.spaces.Box(
                low=np.asarray(space.low)[index:index+1],
                high=np.asarray(space.high)[index:index+1], dtype=np.float32,
            ),
        )
        delegate.set_env(view)
        solver.policy = delegate
        self.policies[index] = delegate
        return solver

    def get_action(self, info_dict, **kwargs):
        n = self.env.num_envs
        dead = np.zeros(n, dtype=bool)
        for key in ("terminated", "truncated"):
            if key in info_dict:
                dead |= np.asarray(info_dict[key], dtype=bool).reshape(n, -1).any(-1)
        actions = np.full(self.env.action_space.shape, np.nan, dtype=np.float32)
        for i in range(n):
            if dead[i] or self.executed_steps[i] >= self.budget:
                continue
            new = self.pending_steps[i] == 0
            solver = self._decision(i) if new else None
            sample = {}
            for key, val in info_dict.items():
                if isinstance(val, (np.ndarray, torch.Tensor, list)) and len(val) == n:
                    sample[key] = val[i:i+1]
                else:
                    sample[key] = val
            # At a decision boundary this encodes the CURRENT real observation.
            # Otherwise the public policy returns the next buffered action.
            action = self.policies[i].get_action(sample)[0]
            if not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite action for an active episode.")
            actions[i] = action
            if new:
                if solver.solve_calls != 1:
                    raise RuntimeError("Each decision must contain one adaptive solve.")
                record = solver.records[0]
                steps = record["planned_primitive_steps"]
                remaining = self.budget - int(self.executed_steps[i])
                if steps != 5*(record["intermediate_nodes"]+1) or steps > remaining:
                    raise RuntimeError("Adaptive decision violates its remaining budget.")
                record.update(
                    index=i, decision_index=len(self.records[i]),
                    start_primitive_step=int(self.executed_steps[i]),
                    remaining_budget_before=remaining, executed_primitive_steps=0,
                )
                self.records[i].append(record)
                self.artifacts[i].append(solver.artifacts[0])
                self.pending_steps[i] = steps
                print(f"Adaptive rolling pair={i} round={record['decision_index']} "
                      f"start={record['start_primitive_step']} remaining={remaining} "
                      f"blocks={record['action_blocks']}", flush=True)
            self.executed_steps[i] += 1
            self.pending_steps[i] -= 1
            self.records[i][-1]["executed_primitive_steps"] += 1
        return actions
