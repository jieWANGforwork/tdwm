"""Batch independent P work across path lengths, without padded fake states."""
from dataclasses import dataclass

import torch

from tdwm.methods.effplan import binary_midpoint_order, state_feedback


@dataclass
class PackedPaths:
    real: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    interior: torch.Tensor
    interior_owner: torch.Tensor
    interior_time: torch.Tensor
    segment_left: torch.Tensor
    segment_owner: torch.Tensor
    segment_time: torch.Tensor
    counts: torch.Tensor
    levels: tuple
    batch_count: int
    max_horizon: int

    @classmethod
    def from_samples(cls, samples, device):
        active = [s for s in samples if s.real_states.shape[1] > 2]
        if not active:
            return None
        if any(s.state_stride != 5 or not bool(s.trajectory_valid.all()) for s in active):
            raise ValueError("Packed P requires valid stride-5 real paths.")
        starts, ends, interior, io, it, sl, so, st, counts = ([] for _ in range(9))
        levels = {}
        offset = 0
        for owner, sample in enumerate(active):
            length = sample.real_states.shape[1]
            starts.append(offset)
            ends.append(offset + length - 1)
            interior.extend(range(offset + 1, offset + length - 1))
            io.extend([owner] * (length - 2))
            it.extend(range(length - 2))
            sl.extend(range(offset, offset + length - 1))
            so.extend([owner] * (length - 1))
            st.extend(range(length - 1))
            counts.append(length - 2)
            depths = {(0, length - 1): 0}
            for left, mid, right in binary_midpoint_order(length - 1):
                depth = depths[left, right]
                levels.setdefault(depth, []).append((offset+left, offset+mid, offset+right))
                depths[left, mid] = depths[mid, right] = depth + 1
            offset += length
        tensor = lambda x: torch.tensor(x, device=device, dtype=torch.long)
        return cls(
            torch.cat([s.real_states[0] for s in active]).to(device).detach(),
            *(tensor(x) for x in (starts, ends, interior, io, it, sl, so, st, counts)),
            tuple(tensor(levels[d]).unbind(1) for d in sorted(levels)),
            len(samples), max(counts)+1,
        )

    def sum_segments(self, values):
        # Unique indices: no atomics, no cross-path accumulation, no fake edges.
        dense = values.new_zeros(len(self.starts), self.max_horizon)
        return dense.index_put((self.segment_owner, self.segment_time), values).sum(1)

    def efficiency(self, nodes, value, epsilon, safety):
        left, right = nodes[self.segment_left], nodes[self.segment_left + 1]
        costs = value(left, right)
        if safety is not None:
            edges = torch.stack((left, right), 1)
            costs = safety.costs(edges, costs[:, None])[:, 0]
        net = torch.linalg.vector_norm(nodes[self.ends] - nodes[self.starts], dim=-1)
        eta = net / (self.sum_segments(costs) + epsilon)
        if safety is not None:
            safety.efficiency(eta)
        return eta

    def trajectory(self, nodes):
        errors = (nodes[self.interior] - self.real[self.interior]).square().sum(-1)
        dense = errors.new_zeros(len(self.starts), self.max_horizon-1)
        sums = dense.index_put((self.interior_owner, self.interior_time), errors).sum(1)
        return (sums / self.counts).sum() / self.batch_count


def packed_generate(planner, paths, value, *, epsilon, safety):
    nodes = torch.zeros_like(paths.real)
    nodes = nodes.index_copy(0, paths.starts, paths.real[paths.starts])
    nodes = nodes.index_copy(0, paths.ends, paths.real[paths.ends])
    for left_ids, mid_ids, right_ids in paths.levels:
        left, right = nodes[left_ids], nodes[right_ids]
        candidate = (left + right) / 2
        objective, gradient = state_feedback(
            torch.stack((left, candidate, right), 1), value, epsilon=epsilon, safety=safety,
        )
        delta = planner(left, candidate, right, objective, gradient[:, 1])
        nodes = nodes.index_copy(0, mid_ids, candidate + (delta if safety is None else safety.delta(delta)))
    return nodes


def packed_refine(planner, nodes, paths, value, *, epsilon, safety):
    with torch.inference_mode(False), torch.enable_grad():
        candidates = nodes.detach().clone().requires_grad_(True)
        objective = -paths.efficiency(candidates, value, epsilon, safety)
        gradient = torch.autograd.grad(objective.sum(), candidates)[0]
        if safety is not None:
            gradient = safety.gradient(gradient)
    ids = paths.interior
    delta = planner(nodes[ids-1], nodes[ids], nodes[ids+1],
                    objective.detach()[paths.interior_owner], gradient.detach()[ids])
    delta = delta if safety is None else safety.delta(delta)
    return nodes.index_copy(0, ids, nodes[ids] + delta)
