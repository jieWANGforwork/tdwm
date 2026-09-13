"""Extra-work V: geometry, full-cost TD, deployment and legacy invariance."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from tdwm.adapters.effplan import EffReadout
from tdwm.methods.eff import EffModel, eff_loss, goal_task
from tdwm.training.eff_run import EffRunSettings, eff_settings_payload


def model(mode="extra_work"):
    result = EffModel(g_hidden_dim=8, v_hidden_dim=8, v_parameterization=mode)
    for p in result.parameters():
        with torch.no_grad():
            p.zero_()
    return result


@pytest.mark.parametrize("target", [False, True])
def test_total_is_raw_endpoint_distance_plus_residual_and_adapter_matches(target):
    m = model()
    z, goal = torch.zeros(2, 192), torch.zeros(2, 192)
    goal[:, :2] = torch.tensor([3., 4.])
    expected = torch.full((2,), 5.) + torch.log(torch.tensor(2.))
    torch.testing.assert_close(m.value(z, goal, target=target), expected)
    torch.testing.assert_close(EffReadout(m, target=target)(z, goal), expected)


def test_exact_goal_boundary_and_candidate_derivatives_are_finite():
    m = model().requires_grad_(False)
    z = torch.zeros(2, 192, requires_grad=True)
    goal = torch.zeros_like(z)
    goal[1, 0] = 1e-8
    values = m.value(z, goal)
    assert values[0] == 0 and values[1] >= 1e-8
    grad = torch.autograd.grad(values.sum(), z)[0]
    assert torch.isfinite(grad).all()
    assert grad[0].eq(0).all()


def test_full_cost_mc_and_td_target_and_detached_g():
    m = model()
    z = torch.zeros(3, 192, requires_grad=True)
    goal = torch.zeros_like(z)
    goal[:, 0] = 10
    boot = torch.zeros_like(z)
    boot[:, 0] = 4
    losses = eff_loss(
        m, state=z, next_state=boot, bootstrap_state=boot, goal=goal,
        terminal_after_transition=torch.zeros(3, dtype=torch.bool),
        observed_cost=torch.tensor([12., 4., 4.]),
        goal_reached=torch.tensor([True, False, False]),
        continuation_valid=torch.tensor([True, True, False]),
        vector_valid=torch.ones(3, dtype=torch.bool), path_ids=torch.arange(3),
        path_weights=torch.ones(3), gamma_g=.98, critic_coefficient=1.,
    )
    # MC stays 12; TD is prefix 4 + remaining distance 6 + target residual.
    expected = torch.tensor([12., 10. + torch.log(torch.tensor(2.)).item(), 0.])
    torch.testing.assert_close(losses.critic_target, expected)
    assert not losses.critic_target.requires_grad
    losses.critic.backward()
    assert z.grad is None
    assert all(p.grad is None for p in m.g.parameters())
    assert all(p.grad is None for p in m.target_g.parameters())
    assert all(p.grad is None for p in m.target_v.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.v.parameters())


def test_legacy_numerics_and_state_dict_layout_are_unchanged():
    m = model("total_work")
    z, goal = torch.randn(4, 192), torch.randn(4, 192)
    task = goal_task(goal)
    torch.testing.assert_close(m.value(z, goal), m.v(m.g(z, task), task), rtol=0, atol=0)
    assert m.state_dict().keys() == model().state_dict().keys()


def test_extra_work_config_changes_only_v_and_preserves_legacy_settings_hash_input():
    root = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    old = yaml.safe_load((root / "effplan_cube_stable_p_v1.yaml").read_text())
    new = yaml.safe_load((root / "effplan_cube_extra_work_v1.yaml").read_text())
    original = copy.deepcopy(new)
    assert new["eff_training"]["settings"].pop("v_parameterization") == "extra_work"
    assert new == old
    legacy = EffRunSettings(**old["eff_training"]["settings"])
    assert eff_settings_payload(legacy) == old["eff_training"]["settings"]
    extra = EffRunSettings(**original["eff_training"]["settings"])
    assert eff_settings_payload(extra) == original["eff_training"]["settings"]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="parameterization"):
        model("unknown")
