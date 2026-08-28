from __future__ import annotations

import inspect

import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm import (
    DirectGoalCriticLeWM,
    load_actor_free_td_checkpoint,
)
from tdwm.methods.direct_goal_critic_lewm import (
    DirectGoalCriticHead,
    direct_goal_critic_td_objective,
)


class ActionEchoCritic(nn.Module):
    embed_dim = 1
    action_dim = 1
    history_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.queried_actions: torch.Tensor | None = None

    def forward(self, latent_history, previous_actions, current_action, goal):
        del latent_history, previous_actions, goal
        self.queried_actions = current_action.detach().clone()
        return self.scale * current_action.squeeze(-1)


class FixedRolloutWorld(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.predicted = predicted

    def rollout(self, info, action_sequence, history_size=None):
        del info, history_size
        return {"predicted_emb": self.predicted.to(action_sequence)}


class RecordingDirectCritic(nn.Module):
    embed_dim = 2
    action_dim = 1
    history_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.goal: torch.Tensor | None = None
        self.current_action: torch.Tensor | None = None

    def forward(self, latent_history, previous_actions, current_action, goal):
        del latent_history, previous_actions
        self.goal = goal.detach().clone()
        self.current_action = current_action.detach().clone()
        return current_action.new_full(current_action.shape[:-1], 4.0)


def test_direct_head_accepts_goal_and_outputs_one_scalar():
    critic = DirectGoalCriticHead(
        embed_dim=4,
        action_dim=2,
        history_size=3,
        hidden_dim=8,
    )
    signature = inspect.signature(DirectGoalCriticHead.forward)

    output = critic(
        torch.randn(5, 3, 4),
        torch.randn(5, 2, 2),
        torch.randn(5, 2),
        torch.randn(5, 4),
    )

    assert list(signature.parameters) == [
        "self",
        "latent_history",
        "previous_actions",
        "current_action",
        "goal",
    ]
    assert output.shape == (5,)
    assert not hasattr(critic, "successor")
    assert not hasattr(critic, "actor")


def test_direct_td_target_uses_real_ema_next_state_dataset_next_action_and_goal():
    online = ActionEchoCritic()
    target = ActionEchoCritic().requires_grad_(False)
    real = torch.zeros(1, 6, 1, requires_grad=True)
    predicted = torch.zeros_like(real, requires_grad=True)
    real_ema = torch.tensor(
        [[[0.0], [0.0], [0.0], [2.0], [5.0], [9.0]]],
        requires_grad=True,
    )
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)

    output = direct_goal_critic_td_objective(
        online,
        target,
        real,
        predicted,
        real_ema,
        actions,
        gamma=0.5,
        goal_offsets=torch.tensor([[2, 1, 1]]),
    )

    # At t=2, goal=z4=5: y=.5*(z3-goal)^2 + .5*Cbar(a3)=6.
    # For offsets one at t=3,4, the real next state is the goal and y=0.
    expected_branch = torch.tensor(((2.0 - 6.0) ** 2 + 3.0**2 + 4.0**2) / 3.0)
    assert target.queried_actions is not None
    assert target.queried_actions.flatten().tolist() == [3.0, 4.0, 5.0]
    assert torch.allclose(output.predicted_td_loss, expected_branch)
    assert torch.allclose(output.real_td_loss, expected_branch)
    assert torch.allclose(output.td_loss, 2.0 * expected_branch)
    assert torch.allclose(output.target_mean, torch.tensor(2.0))
    assert torch.allclose(output.terminal_fraction, torch.tensor(2.0 / 3.0))
    assert output.pair_count.item() == 3

    output.td_loss.backward()
    assert online.scale.grad is not None and torch.count_nonzero(online.scale.grad)
    assert real_ema.grad is None
    assert all(parameter.grad is None for parameter in target.parameters())


def test_direct_hybrid_couples_predicted_branch_and_trains_real_branch():
    torch.manual_seed(5)
    critic = DirectGoalCriticHead(
        embed_dim=3,
        action_dim=2,
        history_size=2,
        hidden_dim=11,
    )
    target = critic.make_target()
    real = torch.randn(2, 6, 3, requires_grad=True)
    predicted = torch.randn(2, 6, 3, requires_grad=True)
    real_ema = torch.randn(2, 6, 3, requires_grad=True)
    actions = torch.randn(2, 6, 2)

    output = direct_goal_critic_td_objective(
        critic,
        target,
        real,
        predicted,
        real_ema,
        actions,
        gamma=0.95,
        goal_offsets=torch.ones(2, 3, dtype=torch.int64),
    )
    output.td_loss.backward()

    assert predicted.grad is not None and torch.count_nonzero(predicted.grad)
    assert real.grad is not None and torch.count_nonzero(real.grad)
    assert real_ema.grad is None
    assert any(parameter.grad is not None for parameter in critic.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())


def test_direct_critic_planning_tail_receives_goal_without_successor_readout():
    predicted = torch.tensor(
        [[[[99.0, 99.0], [98.0, 98.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]
    )
    critic = RecordingDirectCritic()
    adapter = DirectGoalCriticLeWM(
        FixedRolloutWorld(predicted),
        critic,
        gamma=0.5,
        clamp_tail_cost=False,
    )
    actions = torch.tensor([[[[0.1], [0.2], [0.3]]]])
    info = {
        "pixels": torch.zeros(1, 1, 2, 3, 2, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }

    cost = adapter.get_cost(info, actions)

    # Same splice as SF Hybrid: two explicit costs (.75), then gamma^2*C (1).
    assert torch.allclose(cost, torch.tensor([[1.75]]))
    assert critic.current_action is not None
    assert torch.allclose(critic.current_action, actions[..., -1, :])
    assert critic.goal is not None
    assert torch.equal(critic.goal, torch.zeros(1, 1, 2))


def test_direct_checkpoint_loader_restores_critic_not_successor(tmp_path):
    world_model = nn.Linear(2, 2)
    critic = DirectGoalCriticHead(
        embed_dim=2,
        action_dim=1,
        history_size=2,
        hidden_dim=5,
    )
    critic_config = {
        "method": "actor_free_td_lewm",
        "variant": "direct_goal_hybrid",
        "objective_version": 3,
        "deployment_checkpoint_version": 1,
        "embed_dim": 2,
        "action_dim": 1,
        "history_size": 2,
        "hidden_dim": 5,
        "gamma": 0.95,
        "architecture": "direct_goal_critic_head",
        "goal_conditioning": "direct_latent_input",
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "goal_source": "uniform_reachable_future_ema_latent_same_clip",
        "goal_offset_weighting": "uniform_per_transition",
        "goal_terminal_condition": "dataset_terminal_or_next_state_is_goal",
        "td_branches": ["real_context", "predicted_context"],
        "goal_cost": "normalized_discounted_latent_mse",
        "goal_enters_critic_head": True,
        "predicted_context_detach": False,
        "predicted_critic_td_weight": 1.0,
        "real_critic_td_weight": 1.0,
        "actor": "none",
        "reward": "none",
    }
    checkpoint = tmp_path / "direct_goal_critic.pt"
    torch.save(
        {
            "method": "actor_free_td_lewm",
            "variant": "direct_goal_hybrid",
            "objective_version": 3,
            "deployment_checkpoint_version": 1,
            "critic_state_dict": critic.state_dict(),
            "critic_config": critic_config,
            "world_model_state_dict": world_model.state_dict(),
            "world_model_config": {
                "_target_": "torch.nn.Linear",
                "in_features": 2,
                "out_features": 2,
            },
        },
        checkpoint,
    )

    restored_world, restored_critic, restored_config, payload = (
        load_actor_free_td_checkpoint(checkpoint)
    )

    assert isinstance(restored_critic, DirectGoalCriticHead)
    assert restored_config == critic_config
    assert payload["variant"] == "direct_goal_hybrid"
    assert not any(parameter.requires_grad for parameter in restored_world.parameters())
    assert not any(
        parameter.requires_grad for parameter in restored_critic.parameters()
    )
