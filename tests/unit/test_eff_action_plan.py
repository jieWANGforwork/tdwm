"""CPU-only checks of the complete residual planner computation graph."""

import pytest
import torch
from torch import nn

from tdwm.methods.eff_action import EffActionSuccessor, EffActionValue
from tdwm.methods.eff_action_plan import (
    EffActionPlanner,
    EffActionPlanOutput,
    build_eff_action_plan_loss,
    iterate_eff_action_plan,
    project_eff_action,
)


class ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(25, 192)

    def forward(self, action):
        return self.linear(action)


def _setup(shape=(3,)):
    torch.manual_seed(43)
    g = EffActionSuccessor(hidden_dim=8, hidden_layers=1, embedding_layers=2)
    v = EffActionValue(hidden_dim=8, hidden_layers=1, output_activation="softplus")
    e = ActionEncoder()
    for module in (g, v, e):
        module.requires_grad_(False).eval()
    p = EffActionPlanner(raw_action_dim=25, hidden_dim=12, hidden_layers=2)
    batch = dict(
        state=torch.randn(*shape, 192),
        initial_action=torch.randn(*shape, 25) * 0.1,
        goal=torch.randn(*shape, 192),
        task=torch.randn(*shape, 192),
        iterations=3,
        epsilon=0.1,
        lower_bound=-100.0,
        upper_bound=100.0,
        track_grad=True,
    )
    return p, g, v, e, batch


def test_planner_uses_all_six_inputs_and_detaches_only_feedback_and_context():
    torch.manual_seed(47)
    p = EffActionPlanner(raw_action_dim=7, hidden_dim=11, hidden_layers=1)
    state = torch.randn(2, 192, requires_grad=True)
    goal = torch.randn(2, 192, requires_grad=True)
    reference = torch.randn(2, 7, requires_grad=True)
    current = torch.randn(2, 7, requires_grad=True)
    cost = torch.randn(2, requires_grad=True)
    gradient = torch.randn(2, 7, requires_grad=True)
    assert p.network[0].in_features == 2 * 192 + 3 * 7 + 1
    seen = []
    handle = p.network[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    delta = p(state, reference, current, goal, cost, gradient)
    handle.remove()
    assert delta.shape == current.shape
    expected = torch.cat(
        (state, reference, current, goal, cost[:, None], gradient), dim=-1
    )
    assert torch.equal(seen[0], expected)
    delta.square().sum().backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert all(x.grad is None for x in (state, goal, reference, cost, gradient))


def test_joint_loss_reaches_first_update_through_all_k_steps_with_frozen_heads():
    p, g, v, e, batch = _setup()
    before = {
        prefix + "." + key: tensor.clone()
        for prefix, module in (("g", g), ("v", v), ("e", e))
        for key, tensor in module.state_dict().items()
    }
    plan = iterate_eff_action_plan(p, g, v, e, **batch)
    assert len(plan.actions) == len(plan.costs) == 4
    assert len(plan.deltas) == len(plan.action_gradients) == 3
    for delta in plan.deltas:
        delta.retain_grad()
    assert all(not tensor.requires_grad for tensor in plan.costs[:-1])
    assert all(not tensor.requires_grad for tensor in plan.action_gradients)
    assert plan.final_cost.requires_grad
    dataset_action = torch.randn(3, 25, requires_grad=True)
    result = build_eff_action_plan_loss(
        plan,
        dataset_action=dataset_action,
        stage=2,
        lambda_traj=1.0,
        lambda_eff=0.3,
        valid_mask=torch.ones(3, dtype=torch.bool),
    )
    result.loss.backward()
    assert all(
        delta.grad is not None and delta.grad.abs().sum() > 0 for delta in plan.deltas
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in p.parameters()
    )
    assert all(
        parameter.grad is None
        for module in (g, v, e)
        for parameter in module.parameters()
    )
    assert dataset_action.grad is None
    for name, module in (("g", g), ("v", v), ("e", e)):
        assert all(
            torch.equal(tensor, before[name + "." + key])
            for key, tensor in module.state_dict().items()
        )


def test_efficiency_term_alone_differentiates_final_action_through_frozen_e_g_v():
    p, g, v, e, batch = _setup()
    plan = iterate_eff_action_plan(p, g, v, e, **batch)
    plan.action.retain_grad()
    plan.deltas[0].retain_grad()
    result = build_eff_action_plan_loss(
        plan,
        dataset_action=torch.zeros(3, 25),
        stage=2,
        lambda_traj=0,
        lambda_eff=1,
        valid_mask=torch.ones(3, dtype=torch.bool),
    )
    result.loss.backward()
    assert plan.action.grad is not None and plan.action.grad.abs().sum() > 0
    assert plan.deltas[0].grad is not None and plan.deltas[0].grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in p.parameters()
    )
    assert all(
        parameter.grad is None
        for module in (g, v, e)
        for parameter in module.parameters()
    )


@pytest.mark.parametrize("shape", [(3,), (2, 3), ()])
@pytest.mark.parametrize("outer_mode", ["no_grad", "inference_mode"])
def test_inference_handles_outer_disabled_autograd_and_preserves_training_iteration(
    shape, outer_mode
):
    p, g, v, e, batch = _setup(shape)
    p.eval()
    reference = iterate_eff_action_plan(p, g, v, e, **batch)
    batch["track_grad"] = False
    mode = getattr(torch, outer_mode)
    with mode():
        # Specifically exercise inputs created while inference_mode is active.
        for key in ("state", "initial_action", "goal", "task"):
            batch[key] = batch[key].clone()
        inference = iterate_eff_action_plan(p, g, v, e, **batch)
    assert inference.action.shape == shape + (25,)
    assert torch.equal(reference.action, inference.action)
    assert torch.equal(reference.final_cost, inference.final_cost)
    assert all(
        not tensor.requires_grad
        for tensor in inference.actions
        + inference.costs
        + inference.deltas
        + inference.action_gradients
    )


def test_training_can_reenable_graph_inside_outer_inference_mode():
    p, g, v, e, batch = _setup()
    with torch.inference_mode():
        plan = iterate_eff_action_plan(p, g, v, e, **batch)
    plan.final_cost.mean().backward()
    assert any(parameter.grad is not None for parameter in p.parameters())


def test_bfloat16_autocast_preserves_action_gradients_and_original_action_dtype():
    p, g, v, e, batch = _setup()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        plan = iterate_eff_action_plan(p, g, v, e, **batch)
    assert plan.action.dtype == batch["initial_action"].dtype == torch.float32
    assert plan.final_cost.dtype == torch.float32
    plan.final_cost.mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in p.parameters()
    )
    assert all(
        parameter.grad is None
        for module in (g, v, e)
        for parameter in module.parameters()
    )


def test_projection_applies_to_initial_and_every_updated_action():
    p, g, v, e, batch = _setup()
    batch["lower_bound"] = torch.linspace(-0.2, -0.1, 25)
    batch["upper_bound"] = torch.linspace(0.1, 0.2, 25)
    batch["initial_action"].fill_(20)
    with torch.no_grad():
        for parameter in p.parameters():
            parameter.zero_()
        p.network[-1].bias.fill_(-30)
    plan = iterate_eff_action_plan(p, g, v, e, **batch)
    assert torch.equal(plan.reference_action, batch["upper_bound"].expand(3, -1))
    for action in plan.actions:
        assert torch.all(action >= batch["lower_bound"])
        assert torch.all(action <= batch["upper_bound"])
    assert torch.equal(plan.action, batch["lower_bound"].expand(3, -1))
    with pytest.raises(ValueError, match="lower_bound"):
        project_eff_action(plan.action, lower_bound=1, upper_bound=-1)
    with pytest.raises(ValueError, match="finite"):
        project_eff_action(plan.action, lower_bound=float("nan"), upper_bound=1)


def test_two_stage_losses_use_vector_norm_and_external_mask_exactly():
    action = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    cost = torch.tensor([-5.0, -7.0], requires_grad=True)
    plan = EffActionPlanOutput(action, action.detach(), (action,), (cost,), (), ())
    target = torch.tensor([[0.0, 0.0], [float("nan"), float("nan")]])
    valid = torch.tensor([True, False])
    stage1 = build_eff_action_plan_loss(
        plan,
        dataset_action=target,
        stage=1,
        lambda_traj=3,
        lambda_eff=100,
        valid_mask=valid,
    )
    assert stage1.trajectory_loss.item() == stage1.loss.item() == 5
    stage1.loss.backward(retain_graph=True)
    assert cost.grad is None
    assert torch.equal(action.grad, torch.tensor([[2.0, 4.0], [0.0, 0.0]]))
    stage2 = build_eff_action_plan_loss(
        plan,
        dataset_action=target,
        stage=2,
        lambda_traj=3,
        lambda_eff=2,
        valid_mask=valid,
    )
    assert stage2.loss.item() == 3 * 5 + 2 * -5
    assert torch.equal(stage2.trajectory_per_example_loss, torch.tensor([5.0, 0.0]))


def test_frozen_mode_nonfinite_feedback_and_action_dimension_checks():
    p, g, v, e, batch = _setup()
    g.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        iterate_eff_action_plan(p, g, v, e, **batch)
    g.requires_grad_(False)
    batch["initial_action"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        iterate_eff_action_plan(p, g, v, e, **batch)
    batch["initial_action"].zero_()
    with torch.no_grad():
        p.network[-1].bias[0] = torch.inf
    with pytest.raises(FloatingPointError, match="non-finite"):
        iterate_eff_action_plan(p, g, v, e, **batch)
    p7 = EffActionPlanner(raw_action_dim=7, hidden_dim=8, hidden_layers=1)
    with pytest.raises(ValueError, match="25D"):
        iterate_eff_action_plan(p7, g, v, e, **batch)


def test_planner_state_restore_reproduces_exact_actions_and_feedback(tmp_path):
    p, g, v, e, batch = _setup()
    p.eval()
    batch["track_grad"] = False
    original = iterate_eff_action_plan(p, g, v, e, **batch)
    path = tmp_path / "planner.pt"
    torch.save(p.state_dict(), path)
    restored = EffActionPlanner(
        raw_action_dim=25, hidden_dim=12, hidden_layers=2
    ).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    replay = iterate_eff_action_plan(restored, g, v, e, **batch)
    assert torch.equal(replay.action, original.action)
    assert torch.equal(replay.final_cost, original.final_cost)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            replay.action_gradients, original.action_gradients, strict=True
        )
    )
