"""Independent local-efficiency stopping rule; no action search during subdivision."""

import math
from collections import deque

import torch

from tdwm.methods.effplan import state_feedback
from tdwm.methods.effplan_safety import require_finite


def validate_efficiency_threshold(threshold):
    # With the existing geometric floor and +epsilon, nondegenerate eta < 1.
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("efficiency_threshold must be finite and strictly between 0 and 1.")


def efficiency_state_path(planner, start, goal, value, *, max_blocks, safety,
                          efficiency_threshold, epsilon=1e-6):
    """Split each leaf iff D/(max(predicted work,D)+epsilon) < threshold.

    Breadth-first order is only scheduling, never a left/right score comparison.
    A valid midpoint is retained regardless of the sum of its children's work.
    This is an efficiency heuristic, NOT a one-action reachability certificate.
    """
    validate_efficiency_threshold(efficiency_threshold)
    if start.shape != (1, 192) or goal.shape != start.shape:
        raise ValueError("Efficiency subdivision needs one 192D start/goal pair.")
    if type(max_blocks) is not int or max_blocks < 1:
        raise ValueError("max_blocks must be a positive integer.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if any(p.requires_grad for p in planner.parameters()):
        raise ValueError("Efficiency evaluation requires frozen P.")
    require_finite(start, "efficiency start")
    require_finite(goal, "efficiency goal")
    nodes = [start, goal]
    pending = deque([(start, goal, "")])
    records = []
    accepted = 0
    with torch.inference_mode(False), torch.no_grad():
        while pending:
            left, right, branch = pending.popleft()
            distance = torch.linalg.vector_norm(right-left)
            require_finite(distance, "efficiency segment distance")
            record = dict(branch=branch, depth=len(branch), distance=float(distance),
                          accepted=False, threshold=float(efficiency_threshold))
            records.append(record)
            if float(distance) <= epsilon:
                record.update(efficiency=None, predicted_work=None,
                              stop_reason="degenerate_segment")
                continue
            segment = torch.stack((left, right), dim=1)
            raw = value(segment[:, :-1], segment[:, 1:])
            if raw.shape != (1, 1):
                raise ValueError("G -> V must return one scalar per segment.")
            work = safety.costs(segment, raw).sum()
            efficiency = distance / (work + epsilon)
            require_finite(efficiency, "segment efficiency")
            record.update(predicted_work=float(work), raw_work=float(raw.item()),
                          efficiency=float(efficiency))
            if float(efficiency) >= efficiency_threshold:
                record["stop_reason"] = "efficiency_sufficient"
                continue
            if accepted == max_blocks-1:
                record["stop_reason"] = "budget_cap"
                continue
            candidate = (left+right)/2
            local = torch.stack((left, candidate, right), dim=1)
            objective, gradient = state_feedback(local, value, epsilon=epsilon, safety=safety)
            delta = planner(left, candidate, right, objective, gradient[:, 1])
            midpoint = candidate+safety.delta(delta)
            require_finite(midpoint, "efficiency midpoint")
            distinct = bool((torch.linalg.vector_norm(midpoint-left) > epsilon)
                            & (torch.linalg.vector_norm(midpoint-right) > epsilon))
            if not distinct:
                record["stop_reason"] = "duplicate_midpoint"
                continue
            index = next(i for i, node in enumerate(nodes) if node is left)
            nodes.insert(index+1, midpoint)
            accepted += 1
            record.update(accepted=True, stop_reason=None)
            pending.extend(((left, midpoint, branch+"L"), (midpoint, right, branch+"R")))
        path = torch.stack(nodes, dim=1).detach()
    reasons = {r["stop_reason"] for r in records if not r["accepted"]}
    return path, dict(
        criterion="local_efficiency", efficiency_threshold=float(efficiency_threshold),
        intermediate_nodes=accepted, action_blocks=accepted+1, max_action_blocks=max_blocks,
        split_attempts=records, stop_reason="budget_cap" if "budget_cap" in reasons else "leaves_stopped",
    )
