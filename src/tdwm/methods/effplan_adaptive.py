"""Inference-only, critic-gated binary subdivision using the existing P/G/V."""

from collections import deque

import torch

from tdwm.methods.effplan import state_feedback
from tdwm.methods.effplan_safety import require_finite


def adaptive_state_path(planner, start, goal, value, *, max_blocks, safety,
                        epsilon=1e-6, minimum_relative_gain=1e-6):
    """Breadth-first subdivision, accepting only lower predicted total work.

    One sample, no actions/observed midpoints. A rejected parent is a leaf.
    max_blocks is the ENVIRONMENT budget // 5, not a required path length.
    Scores are heuristic critic estimates, not reachability certificates.
    """
    if start.shape != (1, 192) or goal.shape != start.shape:
        raise ValueError("Adaptive subdivision needs one 192D start/goal pair.")
    if type(max_blocks) is not int or max_blocks < 1:
        raise ValueError("max_blocks must be a positive integer.")
    if epsilon <= 0 or not 0 <= minimum_relative_gain < 1:
        raise ValueError("Invalid subdivision tolerances.")
    if any(p.requires_grad for p in planner.parameters()):
        raise ValueError("Adaptive evaluation requires frozen P.")
    nodes = [start, goal]
    # References to tensors remain stable when the ordered leaf list changes.
    pending = deque([(start, goal, 0)])
    records = []
    accepted = 0
    with torch.inference_mode(False), torch.no_grad():
        while pending and accepted < max_blocks - 1:
            left, right, depth = pending.popleft()
            candidate = (left + right) / 2
            local = torch.stack((left, candidate, right), dim=1)
            objective, gradient = state_feedback(
                local, value, epsilon=epsilon, safety=safety,
            )
            delta = planner(left, candidate, right, objective, gradient[:, 1])
            midpoint = candidate + safety.delta(delta)
            require_finite(midpoint, "adaptive midpoint")
            unsplit = torch.stack((left, right), dim=1)
            split = torch.stack((left, midpoint, right), dim=1)
            before = safety.costs(unsplit, value(unsplit[:, :-1], unsplit[:, 1:])).sum()
            after = safety.costs(split, value(split[:, :-1], split[:, 1:])).sum()
            gain = (before - after) / (before + epsilon)
            require_finite(gain, "adaptive split gain")
            # Do not retain a duplicate node even if an imperfect V rewards it.
            distinct = bool((torch.linalg.vector_norm(midpoint-left) > epsilon)
                            & (torch.linalg.vector_norm(midpoint-right) > epsilon))
            keep = distinct and float(gain) > minimum_relative_gain
            records.append(dict(depth=depth, before=float(before), after=float(after),
                                relative_gain=float(gain), accepted=keep))
            if keep:
                index = next(i for i, node in enumerate(nodes) if node is left)
                nodes.insert(index+1, midpoint)
                accepted += 1
                pending.extend(((left, midpoint, depth+1), (midpoint, right, depth+1)))
        path = torch.stack(nodes, dim=1).detach()
    return path, dict(
        intermediate_nodes=accepted, action_blocks=accepted+1,
        max_action_blocks=max_blocks, split_attempts=records,
        stop_reason="budget_cap" if pending and accepted == max_blocks-1 else "no_improving_leaves",
    )
