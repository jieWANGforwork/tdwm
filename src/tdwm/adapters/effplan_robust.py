"""Opt-in action-perturbation reranking of existing EffPlan CEM candidates.

This measures sensitivity inside frozen F, NOT calibrated real-world error.
No environment calls, learned modules, or changes to P refinement are added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch

from tdwm.adapters.effplan import EffPlanTrackingCost


@dataclass(frozen=True)
class ActionRobustness:
    samples: int = 4
    sigma: float = 0.05
    weight: float = 1.0
    seed: int = 43017
    shortlist: int | None = 60

    def __post_init__(self):
        if type(self.samples) is not int or self.samples < 2 or self.samples % 2:
            raise ValueError("samples must be a positive even integer >= 2")
        for name in ("sigma", "weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.shortlist is not None and (
            type(self.shortlist) is not int or self.shortlist < 1
        ):
            raise ValueError("shortlist must be null (all candidates) or a positive integer")

    @property
    def active(self):
        return self.weight > 0 and self.sigma > 0

    def manifest(self):
        return dict(
            **asdict(self),
            formula="J(A) + weight * mean_j relu(J(A + delta_j) - J(A))",
            nominal="existing mean squared state-path tracking cost, including goal",
            noise="fixed seeded antithetic Gaussian bank, each component clipped at +/-3 sigma",
            units="dataset-standardized actions before the unchanged inverse_transform",
            action_clipping="none, matching the installed baseline CEM scoring path",
            reference="same real start, same fixed P nodes, same F for all perturbations",
            shortlist_rule="nominal top-K, robust rerank within K; others excluded from elites",
            zero_weight_or_sigma="exact nominal scores, no shortlist restriction or extra F calls",
            confidence_claim=False,
            matched_environment_budget=True,
            matched_model_compute=False,
        )


def load_action_robustness(path: str | Path) -> ActionRobustness:
    with Path(path).open() as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("Action robustness config must be a JSON object")
    return ActionRobustness(**values)


class RobustEffPlanTrackingCost(EffPlanTrackingCost):
    def __init__(self, world_model, eff, *, target, robustness: ActionRobustness):
        super().__init__(world_model, eff, target=target)
        self.robustness = robustness
        self.risk_records: list[dict] = []

    def perturbations(self, actions):
        # A separate CPU generator never advances CEM's generator/global RNG.
        # Common random numbers make identical candidates comparable across
        # candidate permutations and CEM environment batch sizes.
        rng = torch.Generator(device="cpu").manual_seed(self.robustness.seed)
        half = torch.randn(
            self.robustness.samples // 2, *actions.shape[-2:], generator=rng
        ).clamp(-3, 3) * self.robustness.sigma
        return torch.cat((half, -half)).to(device=actions.device, dtype=actions.dtype)

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        base = super().get_cost(info_dict, action_candidates)
        if not torch.isfinite(base).all():
            raise FloatingPointError("Nonfinite nominal EffPlan cost")
        settings = self.robustness
        if not settings.active:
            return base
        batch, count = base.shape
        k = min(settings.shortlist or count, count)
        indices = base.argsort(dim=1, stable=True)[:, :k]
        rows = torch.arange(batch, device=base.device)[:, None]
        chosen = action_candidates[rows, indices]
        chosen_base = base[rows, indices]
        # SWM expands every tensor context to [batch, candidates, ...]. Keep
        # the original context unchanged, including all P nodes and z_start.
        chosen_info = {
            key: value[rows, indices]
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == base.shape
            else value
            for key, value in info_dict.items()
        }
        risk = torch.zeros_like(chosen_base)
        for delta in self.perturbations(chosen):
            perturbed = super().get_cost(chosen_info, chosen + delta)
            if not torch.isfinite(perturbed).all():
                raise FloatingPointError("Nonfinite perturbed EffPlan cost")
            risk += (perturbed - chosen_base).clamp_min(0)
        risk /= settings.samples
        robust = chosen_base + settings.weight * risk
        if not torch.isfinite(robust).all():
            raise FloatingPointError("Nonfinite robust EffPlan cost")
        scores = torch.full_like(base, float("inf"))
        scores[rows, indices] = robust
        self.risk_records.append(dict(
            batch=batch, candidates=count, shortlisted=k,
            nominal_candidate_rollouts=batch * count,
            perturbation_candidate_rollouts=batch * k * settings.samples,
            nominal_shortlist_mean=float(chosen_base.mean()),
            risk_mean=float(risk.mean()), risk_max=float(risk.max()),
            weighted_risk_mean=float((settings.weight * risk).mean()),
        ))
        return scores

    def diagnostics(self):
        records = self.risk_records
        return dict(
            settings=self.robustness.manifest(), scoring_calls=len(records),
            nominal_candidate_rollouts=sum(r["nominal_candidate_rollouts"] for r in records),
            perturbation_candidate_rollouts=sum(r["perturbation_candidate_rollouts"] for r in records),
            records=records,
        )
