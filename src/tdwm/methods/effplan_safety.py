"""Opt-in P-only safeguards; never change the frozen G/V or Eff score modes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlannerSafety:
    state_gradient_max_norm: float
    state_delta_max_norm: float
    geometric_lower_bound: bool = True

    def __post_init__(self):
        if self.geometric_lower_bound is not True:
            raise ValueError("Stable P requires the geometric lower bound.")
        for limit in (self.state_gradient_max_norm, self.state_delta_max_norm):
            if isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
                raise ValueError("P safety norm limits must be finite and positive.")


def require_finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Nonfinite {label}; do not continue P update.")


def limit_vector_norm(tensor: torch.Tensor, limit: float) -> torch.Tensor:
    """Per-node L2 projection, not elementwise clipping. Zero has finite grad.

    Accumulate norms in float64 so finite float32 entries do not overflow just
    while measuring a norm. Nonfinite entries are rejected, never hidden.
    """
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Norm limit must be finite and positive.")
    require_finite(tensor, "state vector")
    precise = tensor.double()
    norm = torch.linalg.vector_norm(precise, dim=-1, keepdim=True)
    return (precise * (limit / norm.clamp_min(limit))).to(tensor.dtype)


class PlannerSafetyRuntime:
    """One loss/solve's detached diagnostics and shared train/deploy rules."""

    def __init__(self, settings: PlannerSafety):
        self.settings = settings
        self.records: dict[str, list[torch.Tensor]] = {}

    def _record(self, name, tensor):
        self.records.setdefault(name, []).append(tensor.detach().reshape(-1))

    def costs(self, states, raw_costs):
        require_finite(states, "candidate states")
        require_finite(raw_costs, "raw V costs")
        distance = torch.linalg.vector_norm(states[:, 1:] - states[:, :-1], dim=-1)
        require_finite(distance, "segment distance")
        self._record("value_below_distance_fraction", (raw_costs < distance).float())
        self._record("raw_value", raw_costs)
        return torch.maximum(raw_costs, distance)

    def efficiency(self, efficiency):
        require_finite(efficiency, "protected efficiency")
        self._record("protected_efficiency", efficiency)

    def _limit(self, tensor, limit, name):
        result = limit_vector_norm(tensor, limit)
        norm = torch.linalg.vector_norm(tensor.detach().double(), dim=-1)
        self._record(name + "_raw_norm", norm)
        self._record(
            name + "_capped_norm",
            torch.linalg.vector_norm(result.detach().double(), dim=-1),
        )
        self._record(name + "_cap_fraction", (norm > limit).float())
        return result

    def gradient(self, tensor):
        return self._limit(
            tensor, self.settings.state_gradient_max_norm, "candidate_gradient"
        )

    def delta(self, tensor):
        return self._limit(tensor, self.settings.state_delta_max_norm, "state_delta")

    def metrics(self):
        result = {}
        for name, values in self.records.items():
            value = torch.cat(values).double()
            result["safety/" + name + "/mean"] = value.mean().item()
            if not name.endswith("fraction"):
                result["safety/" + name + "/max"] = value.max().item()
                result["safety/" + name + "/min"] = value.min().item()
        return result
