"""No-data checks for the optional perturbation scoring rule."""

import json

import pytest
import torch
from torch import nn

from tdwm.adapters.effplan_robust import (
    ActionRobustness, RobustEffPlanTrackingCost, load_action_robustness,
)
from tdwm.adapters.effplan import EffPlanTrackingCost
from tdwm.methods.eff import EffModel


class LinearWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def rollout(self, info, actions, history_size=None):
        assert history_size == 3
        self.calls.append((info, actions.clone()))
        future = info["emb"][..., -1:, :] + nn.functional.pad(actions, (0, 167)).cumsum(2)
        return dict(info, predicted_emb=torch.cat((info["emb"], future), dim=2))


def make_cost(**settings):
    world = LinearWorld()
    eff = EffModel(g_hidden_dim=8, v_hidden_dim=8).requires_grad_(False).eval()
    return RobustEffPlanTrackingCost(world, eff, target=True, robustness=ActionRobustness(**settings)), world


def inputs(batch=2, candidates=4):
    info = dict(
        emb=torch.zeros(batch, candidates, 1, 192),
        goal_emb=torch.ones(batch, candidates, 1, 192),
        effplan_nodes=torch.zeros(batch, candidates, 6, 192),
    )
    return info, torch.zeros(batch, candidates, 5, 25)


@pytest.mark.parametrize("settings", [
    {"samples": 0}, {"samples": 3}, {"samples": True},
    {"sigma": -1}, {"sigma": float("nan")}, {"weight": float("inf")},
    {"weight": -1}, {"seed": -1}, {"seed": True}, {"shortlist": 0},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        ActionRobustness(**settings)


def test_json_config_roundtrip_and_unknown_keys(tmp_path):
    path = tmp_path / "risk.json"
    path.write_text(json.dumps({"samples": 2, "shortlist": None}))
    assert load_action_robustness(path) == ActionRobustness(samples=2, shortlist=None)
    path.write_text('{"typo": 2}')
    with pytest.raises(TypeError):
        load_action_robustness(path)


@pytest.mark.parametrize("setting", [{"weight": 0}, {"sigma": 0}])
def test_disabled_is_exact_old_cost_without_extra_rollouts_or_shortlisting(setting):
    cost, world = make_cost(shortlist=1, **setting)
    info, actions = inputs()
    actions[:, 1:] = 0.3
    expected = EffPlanTrackingCost.get_cost(cost, info, actions)
    world.calls.clear()
    actual = cost.get_cost(info, actions)
    assert torch.equal(actual, expected)
    assert len(world.calls) == 1
    assert cost.risk_records == []


def test_positive_part_is_taken_before_mean_and_same_start_nodes_are_used():
    cost, world = make_cost(samples=2, weight=2, shortlist=None)
    info, actions = inputs()
    actions[..., 0] = 1
    # One positive, one negative perturbation of known size.
    delta = torch.zeros(5, 25)
    delta[:, 0] = 0.25
    cost.perturbations = lambda _: torch.stack((delta, -delta))
    nominal = EffPlanTrackingCost.get_cost(cost, info, actions)
    worse = EffPlanTrackingCost.get_cost(cost, info, actions + delta)
    world.calls.clear()
    result = cost.get_cost(info, actions.requires_grad_())
    torch.testing.assert_close(result, nominal + (worse - nominal))
    assert len(world.calls) == 3
    for context, _ in world.calls:
        assert torch.equal(context["emb"], info["emb"])
        assert torch.equal(context["effplan_nodes"], info["effplan_nodes"])
    assert not result.requires_grad
    assert actions.grad is None
    assert all(p.grad is None for p in cost.parameters())


def test_noise_is_antithetic_bounded_seeded_without_advancing_global_rng():
    cost, _ = make_cost()
    _, actions = inputs()
    before = torch.get_rng_state()
    noise = cost.perturbations(actions)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(noise[:2], -noise[2:])
    assert noise.abs().max() <= 3 * cost.robustness.sigma
    assert torch.equal(noise, cost.perturbations(actions))
    other, _ = make_cost(seed=43018)
    assert not torch.equal(noise, other.perturbations(actions))


def test_candidate_permutation_and_environment_batching_invariance():
    cost, _ = make_cost(shortlist=None)
    info, actions = inputs()
    actions = torch.randn_like(actions)
    original = cost.get_cost(info, actions)
    perm = torch.tensor([3, 0, 2, 1])
    permuted = cost.get_cost({k: v[:, perm] for k, v in info.items()}, actions[:, perm])
    torch.testing.assert_close(permuted, original[:, perm])
    single = torch.cat([
        cost.get_cost({k: v[i:i+1] for k, v in info.items()}, actions[i:i+1])
        for i in range(2)
    ])
    torch.testing.assert_close(single, original)


def test_shortlist_excludes_unchecked_candidates_from_elites_and_counts_compute():
    cost, world = make_cost(shortlist=2)
    info, actions = inputs()
    actions[:, 2:] = 10
    result = cost.get_cost(info, actions)
    assert torch.isfinite(result[:, :2]).all()
    assert torch.isposinf(result[:, 2:]).all()
    assert [a.shape[1] for _, a in world.calls] == [4, 2, 2, 2, 2]
    stats = cost.diagnostics()
    assert stats["nominal_candidate_rollouts"] == 8
    assert stats["perturbation_candidate_rollouts"] == 16
    assert not stats["settings"]["confidence_claim"]


def test_nonfinite_predictions_fail_instead_of_looking_safe():
    cost, _ = make_cost()
    info, actions = inputs()
    actions[..., 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nominal"):
        cost.get_cost(info, actions)


@pytest.mark.parametrize("kwargs", [
    {"method": "Eff"}, {"method": "F-only"},
    {"method": "EffPlan", "offset_window": True},
    {"method": "EffPlan", "adaptive_rolling": True},
])
def test_evaluation_rejects_unpaired_protocol_combinations(kwargs):
    from tdwm.evaluation.effplan import evaluate_effplan
    with pytest.raises(ValueError, match="fixed H5/RH5"):
        evaluate_effplan(
            config_path="configs/experiment/effplan_cube_same_episode_v1.yaml",
            action_robustness_path="configs/experiment/effplan_action_robustness_v1.json",
            dataset_path="unused", lewm_checkpoint="unused", selection_path="unused",
            output_dir="unused", device="cpu", **kwargs,
        )
