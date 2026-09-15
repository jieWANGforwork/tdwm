"""Same binary P construction, batching independent splits at each tree depth."""

import torch

from tdwm.methods.effplan import binary_midpoint_order, state_feedback


def generate_variable_state_path(planner, start, goal, value, *, horizon, epsilon, safety=None):
    if start.ndim != 2 or start.shape != goal.shape or start.shape[-1] != 192:
        raise ValueError("start/goal must match [batch, 192].")
    order = binary_midpoint_order(horizon)
    depth = {(0, horizon): 0}
    levels = {}
    for left, middle, right in order:
        d = depth[left, right]
        levels.setdefault(d, []).append((left, middle, right))
        depth[left, middle] = depth[middle, right] = d + 1
    nodes = {0: start, horizon: goal}
    for splits in levels.values():
        left = torch.cat([nodes[l] for l, _, _ in splits])
        right = torch.cat([nodes[r] for _, _, r in splits])
        candidate = (left + right) / 2
        local = torch.stack((left, candidate, right), dim=1)
        objective, gradient = state_feedback(local, value, epsilon=epsilon, safety=safety)
        delta = planner(left, candidate, right, objective, gradient[:, 1])
        result = candidate + (delta if safety is None else safety.delta(delta))
        for (_, middle, _), state in zip(splits, result.split(len(start)), strict=True):
            nodes[middle] = state
    return torch.stack([nodes[i] for i in range(horizon + 1)], dim=1)
