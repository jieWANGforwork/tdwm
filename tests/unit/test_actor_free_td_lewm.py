from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm import (
    ActorFreeTDLeWM,
    load_actor_free_td_checkpoint,
)
from tdwm.methods.actor_free_td_lewm import (
    ActorFreeSuccessorHead,
    actor_free_td_objective,
)
from tdwm.methods.successor_geometry import successor_feature_basis


class ActionEchoSuccessor(nn.Module):
    """Expose the queried action in the first successor coordinate."""

    def __init__(self, *, history_size: int = 2) -> None:
        super().__init__()
        self.embed_dim = 1
        self.action_dim = 1
        self.history_size = history_size
        self.output_dim = 3
        self.scale = nn.Parameter(torch.ones(()))
        self.queried_actions: torch.Tensor | None = None

    def forward(self, latent_history, previous_actions, current_action):
        del latent_history, previous_actions
        self.queried_actions = current_action.detach().clone()
        zeros = torch.zeros_like(current_action)
        return torch.cat((self.scale * current_action, zeros, zeros), dim=-1)


class FixedRolloutWorld(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.predicted = predicted
        self.requested_history_size: int | None = None

    def rollout(self, info, action_sequence, history_size=None):
        del info
        self.requested_history_size = history_size
        return {"predicted_emb": self.predicted.to(action_sequence)}


class RecordingTailSuccessor(nn.Module):
    embed_dim = 2
    action_dim = 1
    history_size = 2
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.latent_history: torch.Tensor | None = None
        self.previous_actions: torch.Tensor | None = None
        self.current_action: torch.Tensor | None = None

    def forward(self, latent_history, previous_actions, current_action):
        self.latent_history = latent_history.detach().clone()
        self.previous_actions = previous_actions.detach().clone()
        self.current_action = current_action.detach().clone()
        value = current_action.new_zeros(current_action.shape[:-1] + (4,))
        value[..., 2] = 4.0
        return value


def test_successor_head_is_goal_free_action_conditioned_and_d_plus_two():
    head = ActorFreeSuccessorHead(
        embed_dim=4,
        action_dim=2,
        history_size=3,
        hidden_dim=8,
    )
    signature = inspect.signature(ActorFreeSuccessorHead.forward)

    output = head(
        torch.randn(5, 3, 4),
        torch.randn(5, 2, 2),
        torch.randn(5, 2),
    )

    assert list(signature.parameters) == [
        "self",
        "latent_history",
        "previous_actions",
        "current_action",
    ]
    assert output.shape == (5, 6)
    assert "goal" not in signature.parameters
    assert not hasattr(head, "actor")
    assert not hasattr(head, "policy")
    assert not hasattr(ActorFreeTDLeWM, "get_action")


def test_td_target_uses_next_real_ema_latent_and_dataset_next_action():
    online = ActionEchoSuccessor()
    target_head = ActionEchoSuccessor().requires_grad_(False)
    real = torch.zeros(1, 6, 1)
    predicted = torch.zeros_like(real)
    real_ema = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    actions = 10.0 * torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    terminals = torch.zeros(1, 6, dtype=torch.bool)
    terminals[:, 3] = True
    gamma = 0.5

    output = actor_free_td_objective(
        online,
        target_head,
        real,
        predicted,
        real_ema,
        actions,
        gamma=gamma,
        variant="serial_decoupled",
        terminals=terminals,
    )

    # Default first_current_index is history_size=2: current indices are
    # [2, 3, 4], and the bootstrap must query dataset actions [3, 4, 5].
    assert target_head.queried_actions is not None
    assert target_head.queried_actions.flatten().tolist() == [30.0, 40.0, 50.0]
    bootstrap = torch.zeros(1, 3, 3)
    bootstrap[..., 0] = actions[:, 3:, 0]
    continuation = torch.tensor([[[1.0], [0.0], [1.0]]])
    expected_target = (1.0 - gamma) * successor_feature_basis(
        real_ema[:, 3:]
    ) + gamma * continuation * bootstrap
    prediction = torch.zeros_like(expected_target)
    prediction[..., 0] = actions[:, 2:-1, 0]

    assert output.pair_count == 3
    assert torch.allclose(
        output.td_loss, (prediction - expected_target).square().mean()
    )
    assert torch.allclose(output.target_mean, expected_target.mean())
    assert torch.allclose(output.terminal_fraction, torch.tensor(1.0 / 3.0))

    with pytest.raises(ValueError, match="at least history_size"):
        actor_free_td_objective(
            online,
            target_head,
            real,
            predicted,
            real_ema,
            actions,
            gamma=gamma,
            variant="serial_decoupled",
            first_current_index=1,
        )


@pytest.mark.parametrize(
    ("variant", "predicted_has_grad", "real_has_grad"),
    [
        ("parallel_real", False, True),
        ("serial_decoupled", False, False),
        ("serial_coupled", True, False),
        ("hybrid", True, True),
    ],
)
def test_td_variants_have_the_intended_world_model_gradient_paths(
    variant: str,
    predicted_has_grad: bool,
    real_has_grad: bool,
):
    torch.manual_seed(31)
    head = ActorFreeSuccessorHead(
        embed_dim=3,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
    )
    target = head.make_target()
    real = torch.randn(2, 6, 3, requires_grad=True)
    predicted = torch.randn(2, 6, 3, requires_grad=True)
    real_ema = torch.randn(2, 6, 3, requires_grad=True)
    actions = torch.randn(2, 6, 2)

    output = actor_free_td_objective(
        head,
        target,
        real,
        predicted,
        real_ema,
        actions,
        gamma=0.9,
        variant=variant,
    )
    output.td_loss.backward()

    assert (predicted.grad is not None) is predicted_has_grad
    if predicted_has_grad:
        assert torch.count_nonzero(predicted.grad) > 0
    assert (real.grad is not None) is real_has_grad
    if real_has_grad:
        assert torch.count_nonzero(real.grad) > 0
    assert real_ema.grad is None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in head.parameters()
    )
    assert all(parameter.grad is None for parameter in target.parameters())
    assert (output.real_td_loss is not None) is (
        variant in {"parallel_real", "hybrid"}
    )
    if variant == "hybrid":
        assert torch.allclose(
            output.td_loss,
            output.predicted_td_loss + output.real_td_loss,
        )
    elif variant == "parallel_real":
        assert output.real_td_loss is not None
        assert torch.equal(output.predicted_td_loss, torch.zeros(()))
        assert torch.equal(output.td_loss, output.real_td_loss)
    else:
        assert torch.equal(output.td_loss, output.predicted_td_loss)


def test_parallel_real_does_not_read_predicted_context_values():
    torch.manual_seed(32)
    head = ActorFreeSuccessorHead(
        embed_dim=3,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
    )
    real = torch.randn(2, 6, 3)
    predicted = torch.full_like(real, torch.nan)

    output = actor_free_td_objective(
        head,
        head.make_target(),
        real,
        predicted,
        torch.randn_like(real),
        torch.randn(2, 6, 2),
        gamma=0.9,
        variant="parallel_real",
    )

    assert torch.isfinite(output.td_loss)
    assert torch.isfinite(output.prediction_mean)
    assert torch.equal(output.predicted_td_loss, torch.zeros(()))


def test_cem_cost_splices_h_minus_one_explicit_steps_and_final_action_tail():
    predicted = torch.tensor(
        [
            [
                [
                    [99.0, 99.0],
                    [98.0, 98.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                ]
            ]
        ]
    )
    world_model = FixedRolloutWorld(predicted)
    successor = RecordingTailSuccessor()
    adapter = ActorFreeTDLeWM(
        world_model,
        successor,
        gamma=0.5,
        clamp_tail_cost=False,
    )
    actions = torch.tensor([[[[0.1], [0.2], [0.3]]]])
    info = {
        "pixels": torch.zeros(1, 1, 2, 3, 2, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }

    cost = adapter.get_cost(info, actions)

    # z1 and z2 are explicit: .5 * (c(z1) + .5 c(z2)) = .75.
    # The head starts from [z1, z2] with final action .3 and contributes
    # .5**2 * 4 = 1.  z3 is therefore represented only inside the Q tail.
    assert torch.allclose(cost, torch.tensor([[1.75]]))
    assert world_model.requested_history_size == 2
    assert successor.latent_history is not None
    assert torch.equal(successor.latent_history, predicted[..., 2:4, :])
    assert successor.previous_actions is not None
    assert torch.allclose(successor.previous_actions, actions[..., 1:2, :])
    assert successor.current_action is not None
    assert torch.allclose(successor.current_action, actions[..., 2, :])


@pytest.mark.parametrize("variant", ["serial_coupled", "parallel_real"])
def test_joint_checkpoint_loader_restores_goal_free_successor_and_world_model(
    tmp_path,
    variant,
):
    world_model = nn.Linear(2, 2)
    successor = ActorFreeSuccessorHead(
        embed_dim=2,
        action_dim=1,
        history_size=2,
        hidden_dim=5,
    )
    successor_config = {
        "method": "actor_free_td_lewm",
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "embed_dim": 2,
        "action_dim": 1,
        "history_size": 2,
        "hidden_dim": 5,
        "gamma": 0.95,
        "variant": variant,
        "architecture": "actor_free_successor_head",
        "feature_basis": "augmented_latent_squared_distance",
        "goal_conditioning": "none",
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "actor": "none",
        "reward": "none",
        "predicted_context_detach": False,
    }
    checkpoint = tmp_path / "actor_free_td.pt"
    torch.save(
        {
            "method": "actor_free_td_lewm",
            "variant": variant,
            "objective_version": 1,
            "deployment_checkpoint_version": 1,
            "successor_state_dict": successor.state_dict(),
            "successor_config": successor_config,
            "world_model_state_dict": world_model.state_dict(),
            "world_model_config": {
                "_target_": "torch.nn.Linear",
                "in_features": 2,
                "out_features": 2,
            },
        },
        checkpoint,
    )

    restored_world, restored_head, restored_config, payload = (
        load_actor_free_td_checkpoint(checkpoint)
    )

    assert restored_config == successor_config
    assert payload["variant"] == variant
    assert not any(parameter.requires_grad for parameter in restored_world.parameters())
    assert not any(parameter.requires_grad for parameter in restored_head.parameters())
    for key, value in world_model.state_dict().items():
        assert torch.equal(restored_world.state_dict()[key], value)
    for key, value in successor.state_dict().items():
        assert torch.equal(restored_head.state_dict()[key], value)
