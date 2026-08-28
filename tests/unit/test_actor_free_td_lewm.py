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
    actor_free_goal_future_offset_limits,
    actor_free_td_objective,
    sample_actor_free_goal_offsets,
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


class CoefficientEchoSuccessor(nn.Module):
    """Put the queried action on the goal-independent quadratic coefficient."""

    def __init__(self, *, history_size: int = 2) -> None:
        super().__init__()
        self.embed_dim = 1
        self.action_dim = 1
        self.history_size = history_size
        self.output_dim = 3
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, latent_history, previous_actions, current_action):
        del latent_history, previous_actions
        zeros = torch.zeros_like(current_action)
        return torch.cat((zeros, self.scale * current_action, zeros), dim=-1)


class HistoryRecordingSuccessor(nn.Module):
    """Record target histories while returning a trainable zero prediction."""

    def __init__(self, *, history_size: int = 2) -> None:
        super().__init__()
        self.embed_dim = 1
        self.action_dim = 1
        self.history_size = history_size
        self.output_dim = 3
        self.scale = nn.Parameter(torch.zeros(()))
        self.queried_histories: torch.Tensor | None = None

    def forward(self, latent_history, previous_actions, current_action):
        del previous_actions
        self.queried_histories = latent_history.detach().clone()
        return self.scale.expand(current_action.shape[:-1] + (self.output_dim,))


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
        ("goal_hybrid", True, True),
        ("imaginary_hybrid", True, True),
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
    imagined = None
    if variant == "imaginary_hybrid":
        imagined = torch.randn(2, 3, 3, requires_grad=True)

    output = actor_free_td_objective(
        head,
        target,
        real,
        predicted,
        real_ema,
        actions,
        gamma=0.9,
        variant=variant,
        imagined_ema_next_latents=imagined,
    )
    output.td_loss.backward()

    assert (predicted.grad is not None) is predicted_has_grad
    if predicted_has_grad:
        assert torch.count_nonzero(predicted.grad) > 0
    assert (real.grad is not None) is real_has_grad
    if real_has_grad:
        assert torch.count_nonzero(real.grad) > 0
    assert real_ema.grad is None
    if imagined is not None:
        assert imagined.grad is None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in head.parameters()
    )
    assert all(parameter.grad is None for parameter in target.parameters())
    assert (output.real_td_loss is not None) is (
        variant in {"parallel_real", "hybrid", "goal_hybrid", "imaginary_hybrid"}
    )
    if variant in {"hybrid", "goal_hybrid", "imaginary_hybrid"}:
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


def test_imaginary_hybrid_replaces_only_bootstrap_history_next_latent():
    online = HistoryRecordingSuccessor()
    target = HistoryRecordingSuccessor().requires_grad_(False)
    real = torch.zeros(1, 6, 1, requires_grad=True)
    predicted = torch.zeros_like(real, requires_grad=True)
    real_ema = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    imagined = torch.tensor([[[30.0], [40.0], [50.0]]], requires_grad=True)

    output = actor_free_td_objective(
        online,
        target,
        real,
        predicted,
        real_ema,
        actions,
        gamma=0.5,
        variant="imaginary_hybrid",
        imagined_ema_next_latents=imagined,
    )

    assert target.queried_histories is not None
    expected_histories = torch.tensor(
        [[[[2.0], [30.0]], [[3.0], [40.0]], [[4.0], [50.0]]]]
    )
    assert torch.equal(target.queried_histories, expected_histories)
    expected_target = 0.5 * successor_feature_basis(real_ema[:, 3:])
    expected_branch = expected_target.square().mean()
    assert torch.allclose(output.predicted_td_loss, expected_branch)
    assert torch.allclose(output.real_td_loss, expected_branch)
    assert torch.allclose(output.td_loss, 2.0 * expected_branch)
    assert torch.allclose(
        output.imaginary_next_mse,
        (imagined.detach() - real_ema[:, 3:]).square().mean(),
    )

    output.td_loss.backward()
    assert imagined.grad is None
    assert all(parameter.grad is None for parameter in target.parameters())

    with pytest.raises(ValueError, match="requires EMA-predicted next latents"):
        actor_free_td_objective(
            online,
            target,
            real,
            predicted,
            real_ema,
            actions,
            gamma=0.5,
            variant="imaginary_hybrid",
        )
    with pytest.raises(ValueError, match="only valid for imaginary_hybrid"):
        actor_free_td_objective(
            online,
            target,
            real,
            predicted,
            real_ema,
            actions,
            gamma=0.5,
            variant="hybrid",
            imagined_ema_next_latents=imagined,
        )


def test_goal_hybrid_uses_hindsight_goal_bellman_target_without_clamping():
    online = CoefficientEchoSuccessor()
    target = CoefficientEchoSuccessor().requires_grad_(False)
    real = torch.zeros(1, 6, 1, requires_grad=True)
    predicted = torch.zeros_like(real, requires_grad=True)
    real_ema = torch.tensor(
        [[[0.0], [0.0], [0.0], [2.0], [5.0], [9.0]]],
        requires_grad=True,
    )
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    offsets = torch.tensor([[2, 1, 1]])

    output = actor_free_td_objective(
        online,
        target,
        real,
        predicted,
        real_ema,
        actions,
        gamma=0.5,
        variant="goal_hybrid",
        goal_offsets=offsets,
    )

    # At t=2, g=z4=5: y=.5*(2-5)^2 + .5*Gbar(a3)^T w(g)=6.
    # At t=3 and t=4 the sampled goal is the real next state, hence y=0.
    expected_branch = torch.tensor(((2.0 - 6.0) ** 2 + 3.0**2 + 4.0**2) / 3.0)
    assert torch.allclose(output.predicted_goal_td_loss, expected_branch)
    assert torch.allclose(output.real_goal_td_loss, expected_branch)
    assert torch.allclose(output.goal_td_loss, 2.0 * expected_branch)
    assert torch.allclose(output.goal_target_mean, torch.tensor(2.0))
    assert torch.allclose(output.goal_prediction_mean, torch.tensor(3.0))
    assert torch.allclose(output.goal_terminal_fraction, torch.tensor(2.0 / 3.0))
    assert output.goal_pair_count.item() == 3

    output.goal_td_loss.backward()
    assert online.scale.grad is not None and torch.count_nonzero(online.scale.grad)
    assert real_ema.grad is None
    assert all(parameter.grad is None for parameter in target.parameters())


def test_goal_validation_is_exact_per_transition_uniform_expectation():
    online = CoefficientEchoSuccessor()
    target = CoefficientEchoSuccessor().requires_grad_(False)
    real = torch.zeros(1, 5, 1)
    real_ema = torch.tensor([[[0.0], [0.0], [0.0], [2.0], [5.0]]])
    actions = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)

    output = actor_free_td_objective(
        online,
        target,
        real,
        real,
        real_ema,
        actions,
        gamma=0.5,
        variant="goal_hybrid",
    )

    # t=2 has two equiprobable goals: z3 gives y=0, z4 gives y=6.
    # t=3 has one goal z4 and y=0. Average goals within each t first.
    expected_branch = torch.tensor(
        ((((2.0 - 0.0) ** 2 + (2.0 - 6.0) ** 2) / 2.0) + 3.0**2) / 2.0
    )
    assert torch.allclose(output.predicted_goal_td_loss, expected_branch)
    assert torch.allclose(output.real_goal_td_loss, expected_branch)
    assert output.goal_pair_count.item() == 3


def test_hindsight_goal_sampling_stops_at_first_terminal_and_is_reproducible():
    terminals = torch.zeros(1, 8, dtype=torch.bool)
    terminals[:, 4] = True
    limits = actor_free_goal_future_offset_limits(terminals, first_current_index=2)

    # t=2..4 may reach only through z5; t>4 is outside the terminated episode.
    assert limits.tolist() == [[3, 2, 1, 0, 0]]
    first = sample_actor_free_goal_offsets(
        terminals,
        first_current_index=2,
        generator=torch.Generator().manual_seed(77),
    )
    second = sample_actor_free_goal_offsets(
        terminals,
        first_current_index=2,
        generator=torch.Generator().manual_seed(77),
    )
    assert torch.equal(first, second)
    assert torch.all(first[:, :3] <= limits[:, :3])
    assert torch.all(first >= 1)


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


@pytest.mark.parametrize(
    "variant",
    ["serial_coupled", "parallel_real", "goal_hybrid", "imaginary_hybrid"],
)
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
        "objective_version": (
            2 if variant == "goal_hybrid" else 3 if variant == "imaginary_hybrid" else 1
        ),
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
    if variant == "goal_hybrid":
        successor_config.update(
            {
                "goal_readout_training": True,
                "goal_source": "uniform_reachable_future_ema_latent_same_clip",
                "goal_offset_weighting": "uniform_per_transition",
                "goal_terminal_condition": ("dataset_terminal_or_next_state_is_goal"),
                "goal_readout_branches": ["real_context", "predicted_context"],
                "goal_readout_precision": "float32",
                "goal_cost": "normalized_discounted_latent_mse",
                "goal_enters_successor_head": False,
                "predicted_goal_td_weight": 1.0,
                "real_goal_td_weight": 1.0,
            }
        )
    if variant == "imaginary_hybrid":
        successor_config.update(
            {
                "immediate_feature_source": "real_ema_next_latent",
                "bootstrap_state_source": (
                    "ema_lewm_predicted_next_from_real_ema_history"
                ),
                "imaginary_horizon": 1,
                "imaginary_predictor_gradient": "target_ema_stop_gradient",
            }
        )
    checkpoint = tmp_path / "actor_free_td.pt"
    torch.save(
        {
            "method": "actor_free_td_lewm",
            "variant": variant,
            "objective_version": (
                2
                if variant == "goal_hybrid"
                else 3 if variant == "imaginary_hybrid" else 1
            ),
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
