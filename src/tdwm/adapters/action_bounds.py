"""Opt-in execution-space box constraints for the public SWM CEM solver.

No dependency patch or replacement CEM. The cost adapter projects the shared
candidate tensor in place, so scoring AND elite fitting use the same actions.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ActionBox:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    raw_low: float = -1.0
    raw_high: float = 1.0

    def __post_init__(self):
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("Action mean and scale must have equal nonzero length.")
        if not all(math.isfinite(v) for v in (*self.mean, *self.scale,
                                             self.raw_low, self.raw_high)):
            raise ValueError("Action bounds and normalization must be finite.")
        if min(self.scale) <= 0 or self.raw_low >= self.raw_high:
            raise ValueError("Positive scales and ordered bounds are required.")

    def limits(self, actions):
        dim = len(self.mean)
        if actions.shape[-1] % dim or not actions.is_floating_point():
            raise ValueError("Action blocks must concatenate whole floating-point actions.")
        repeats = actions.shape[-1] // dim
        # Compute with saved full-precision statistics, then move one ULP inward
        # to keep inverse_transform's float32 rounding inside execution bounds.
        lower = [(self.raw_low - m) / s for m, s in zip(self.mean, self.scale)]
        upper = [(self.raw_high - m) / s for m, s in zip(self.mean, self.scale)]
        lo = actions.new_tensor(lower).repeat(repeats)
        hi = actions.new_tensor(upper).repeat(repeats)
        return torch.nextafter(lo, hi), torch.nextafter(hi, lo)

    @torch.inference_mode()
    def project_(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite CEM action; bounds cannot repair NaNs.")
        lo, hi = self.limits(actions)
        actions.clamp_(min=lo, max=hi)
        return actions

    def manifest(self):
        return dict(
            name="execution_box_v1", raw_low=self.raw_low, raw_high=self.raw_high,
            mean=list(self.mean), scale=list(self.scale),
            standardized_low=[(self.raw_low-m)/s for m, s in zip(self.mean, self.scale)],
            standardized_high=[(self.raw_high-m)/s for m, s in zip(self.mean, self.scale)],
            projection="each candidate in-place before F scoring and elite fitting; returned means also bounded",
            grouping="repeat primitive bounds across every action in each block",
            roundoff_guard="one representable step inward in candidate dtype",
            execution_only_clipping=False, changes_training=False,
        )


class BoundedCEMCost(nn.Module):
    """Costable wrapper: mutate SWM candidates before it fits elites to them."""

    def __init__(self, model, bounds: ActionBox):
        super().__init__()
        self.model = model
        self.bounds = bounds
        self.calls = 0
        self.changed_components = 0
        self.total_components = 0

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        lo, hi = self.bounds.limits(action_candidates)
        changed = ((action_candidates < lo) | (action_candidates > hi)).sum()
        self.bounds.project_(action_candidates)
        self.calls += 1
        self.changed_components += int(changed)
        self.total_components += action_candidates.numel()
        return self.model.get_cost(info_dict, action_candidates)

    def diagnostics(self):
        return dict(**self.bounds.manifest(), cost_calls=self.calls,
                    projected_candidate_components=self.changed_components,
                    evaluated_candidate_components=self.total_components)
