from __future__ import annotations

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1
from tdwm.methods.actor_free_td_lewm_v2 import (
    V2_ACTION_DIM,
    V2_ACTION_EMBEDDING_DIM,
    V2_OUTPUT_DIM,
    V2_RAW_ACTION_DIM,
    V2_STATE_DIM,
    V2_TASK_DIM,
    build_hybrid_tdjepa_td_batch_v2,
    encode_trainable_action_blocks_v2,
)


class RecordingActionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 25
        self.emb_dim = 192
        self.projection = nn.Linear(25, 192)
        self.seen_shapes: list[torch.Size] = []
        self.grad_enabled: list[bool] = []

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(raw_action.shape)
        self.grad_enabled.append(torch.is_grad_enabled())
        return torch.tanh(self.projection(raw_action))


def _predictor(*, hidden_dim: int = 16) -> ActorFreeTDJEPAPredictorV1:
    return ActorFreeTDJEPAPredictorV1(hidden_dim=hidden_dim)


def _hybrid_inputs(
    *,
    batch_size: int = 5,
) -> dict[str, torch.Tensor]:
    return {
        "real_state": torch.randn(batch_size, 192, requires_grad=True),
        "predicted_state": torch.randn(batch_size, 192, requires_grad=True),
        "task": torch.randn(batch_size, 192, requires_grad=True),
        "ema_next_state": torch.randn(batch_size, 192, requires_grad=True),
        "target_next_action_embedding": torch.randn(
            batch_size,
            192,
            requires_grad=True,
        ),
    }


def test_v2_reuses_v1_predictor_and_locked_lewm_dimensions():
    predictor = _predictor()

    assert isinstance(predictor, ActorFreeTDJEPAPredictorV1)
    assert V2_RAW_ACTION_DIM == 25
    assert V2_ACTION_DIM == V2_ACTION_EMBEDDING_DIM == 192
    assert V2_STATE_DIM == V2_TASK_DIM == V2_OUTPUT_DIM == 192


def test_trainable_action_helper_restores_axes_and_keeps_gradients():
    torch.manual_seed(101)
    encoder = RecordingActionEncoder().train()
    raw_action = torch.randn(2, 3, 4, 25, requires_grad=True)
    reference = torch.randn(2, 3, 4, 192)

    embedding = encode_trainable_action_blocks_v2(
        encoder,
        raw_action,
        reference,
    )
    embedding.square().mean().backward()

    assert embedding.shape == (2, 3, 4, 192)
    assert embedding.dtype == reference.dtype
    assert embedding.device == reference.device
    assert embedding.requires_grad
    assert encoder.seen_shapes == [torch.Size((24, 1, 25))]
    assert encoder.grad_enabled == [True]
    assert raw_action.grad is not None
    assert torch.count_nonzero(raw_action.grad) > 0
    assert encoder.projection.weight.grad is not None
    assert torch.count_nonzero(encoder.projection.weight.grad) > 0

    vector = encode_trainable_action_blocks_v2(
        encoder,
        torch.randn(25, requires_grad=True),
        torch.randn(192),
    )
    assert vector.shape == (192,)
    assert vector.requires_grad


def test_trainable_action_helper_does_not_require_train_mode_or_parameters():
    encoder = RecordingActionEncoder().eval().requires_grad_(False)
    raw_action = torch.randn(3, 25, requires_grad=True)

    result = encode_trainable_action_blocks_v2(
        encoder,
        raw_action,
        torch.randn(3, 192),
    )
    result.sum().backward()

    assert raw_action.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert encoder.grad_enabled == [True]


def test_trainable_action_helper_rejects_invalid_interfaces_and_shapes():
    encoder = RecordingActionEncoder()
    raw_action = torch.randn(2, 25)
    reference = torch.randn(2, 192)

    encoder.input_dim = 24
    with pytest.raises(ValueError, match="25D action block"):
        encode_trainable_action_blocks_v2(encoder, raw_action, reference)

    encoder.input_dim = 25
    with pytest.raises(ValueError, match="identical leading axes"):
        encode_trainable_action_blocks_v2(
            encoder,
            torch.randn(2, 3, 25),
            reference,
        )

    class WrongOutput(nn.Module):
        input_dim = 25
        emb_dim = 192

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    with pytest.raises(ValueError, match="must return shape"):
        encode_trainable_action_blocks_v2(
            WrongOutput(),
            raw_action,
            reference,
        )


def test_hybrid_batch_uses_one_g_twice_and_one_detached_ema_target():
    torch.manual_seed(103)
    online = _predictor()
    target = online.make_target()
    encoder = RecordingActionEncoder()
    raw_action = torch.randn(5, 25, requires_grad=True)
    inputs = _hybrid_inputs()
    action_embedding = encode_trainable_action_blocks_v2(
        encoder,
        raw_action,
        inputs["real_state"],
    )
    online_inputs: list[tuple[torch.Tensor, ...]] = []
    target_inputs: list[tuple[torch.Tensor, ...]] = []

    def record_online(
        module: nn.Module,
        values: tuple[torch.Tensor, ...],
    ) -> None:
        del module
        online_inputs.append(values)

    def record_target(
        module: nn.Module,
        values: tuple[torch.Tensor, ...],
    ) -> None:
        del module
        target_inputs.append(values)

    online.register_forward_pre_hook(record_online)
    target.register_forward_pre_hook(record_target)
    batch = build_hybrid_tdjepa_td_batch_v2(
        online,
        target,
        inputs["real_state"],
        inputs["predicted_state"],
        action_embedding,
        inputs["task"],
        inputs["ema_next_state"],
        inputs["target_next_action_embedding"],
        gamma=0.97,
        terminal=torch.tensor([False, False, True, False, False]),
    )

    assert len(online_inputs) == 2
    assert online_inputs[0][0] is inputs["real_state"]
    assert online_inputs[1][0] is inputs["predicted_state"]
    assert online_inputs[0][1] is action_embedding
    assert online_inputs[1][1] is action_embedding
    assert len(target_inputs) == 1
    assert torch.equal(target_inputs[0][0], inputs["ema_next_state"])
    assert torch.equal(
        target_inputs[0][1],
        inputs["target_next_action_embedding"],
    )
    assert batch.real_prediction.shape == (5, 192)
    assert batch.predicted_prediction.shape == (5, 192)
    assert batch.target.shape == (5, 192)
    assert batch.real_per_transition_td_loss.shape == (5,)
    assert batch.predicted_per_transition_td_loss.shape == (5,)
    assert torch.equal(
        batch.per_transition_td_loss,
        batch.real_per_transition_td_loss
        + batch.predicted_per_transition_td_loss,
    )
    assert batch.hybrid_td_loss == batch.real_td_loss + batch.predicted_td_loss
    assert not batch.target.requires_grad


def test_hybrid_batch_backpropagates_both_branches_and_online_action_path():
    torch.manual_seed(107)
    online = _predictor()
    target = online.make_target()
    encoder = RecordingActionEncoder()
    raw_action = torch.randn(4, 25, requires_grad=True)
    inputs = _hybrid_inputs(batch_size=4)
    action_embedding = encode_trainable_action_blocks_v2(
        encoder,
        raw_action,
        inputs["real_state"],
    )

    batch = build_hybrid_tdjepa_td_batch_v2(
        online,
        target,
        inputs["real_state"],
        inputs["predicted_state"],
        action_embedding,
        inputs["task"],
        inputs["ema_next_state"],
        inputs["target_next_action_embedding"],
        gamma=0.95,
    )
    batch.hybrid_td_loss.backward()

    for name in ("real_state", "predicted_state"):
        assert inputs[name].grad is not None
        assert torch.count_nonzero(inputs[name].grad) > 0
    assert raw_action.grad is not None
    assert torch.count_nonzero(raw_action.grad) > 0
    assert encoder.projection.weight.grad is not None
    assert torch.count_nonzero(encoder.projection.weight.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in online.parameters()
    )
    assert inputs["task"].grad is None
    assert inputs["ema_next_state"].grad is None
    assert inputs["target_next_action_embedding"].grad is None
    assert all(parameter.grad is None for parameter in target.parameters())


def test_hybrid_batch_sums_feature_mse_across_real_and_predicted_branches():
    online = _predictor()
    target = online.make_target()
    for predictor in (online, target):
        for parameter in predictor.parameters():
            parameter.data.zero_()
    ema_next_state = torch.stack(
        (torch.ones(192), torch.full((192,), 2.0))
    )

    batch = build_hybrid_tdjepa_td_batch_v2(
        online,
        target,
        torch.zeros(2, 192),
        torch.zeros(2, 192),
        torch.zeros(2, 192),
        torch.ones(2, 192),
        ema_next_state,
        torch.zeros(2, 192),
        gamma=0.95,
    )

    assert torch.equal(
        batch.real_per_transition_td_loss,
        torch.tensor([192.0, 768.0]),
    )
    assert torch.equal(
        batch.predicted_per_transition_td_loss,
        torch.tensor([192.0, 768.0]),
    )
    assert torch.equal(
        batch.per_transition_td_loss,
        torch.tensor([384.0, 1536.0]),
    )
    assert batch.real_td_loss == 480.0
    assert batch.predicted_td_loss == 480.0
    assert batch.hybrid_td_loss == 960.0


def test_hybrid_batch_accepts_no_grad_predicted_state_without_detaching_policy():
    online = _predictor()
    target = online.make_target()

    batch = build_hybrid_tdjepa_td_batch_v2(
        online,
        target,
        torch.randn(2, 192),
        torch.randn(2, 192),
        torch.randn(2, 192),
        torch.randn(2, 192),
        torch.randn(2, 192),
        torch.randn(2, 192),
        gamma=0.9,
    )

    assert batch.predicted_prediction.requires_grad
    assert batch.hybrid_td_loss.requires_grad


def test_hybrid_batch_strictly_rejects_invalid_target_and_alignment():
    online = _predictor()
    inputs = _hybrid_inputs(batch_size=2)
    action_embedding = torch.randn(2, 192, requires_grad=True)

    with pytest.raises(ValueError, match="distinct predictor"):
        build_hybrid_tdjepa_td_batch_v2(
            online,
            online,
            inputs["real_state"],
            inputs["predicted_state"],
            action_embedding,
            inputs["task"],
            inputs["ema_next_state"],
            inputs["target_next_action_embedding"],
            gamma=0.9,
        )

    trainable_target = _predictor()
    with pytest.raises(ValueError, match="must be frozen"):
        build_hybrid_tdjepa_td_batch_v2(
            online,
            trainable_target,
            inputs["real_state"],
            inputs["predicted_state"],
            action_embedding,
            inputs["task"],
            inputs["ema_next_state"],
            inputs["target_next_action_embedding"],
            gamma=0.9,
        )

    training_target = online.make_target().train()
    with pytest.raises(ValueError, match="eval mode"):
        build_hybrid_tdjepa_td_batch_v2(
            online,
            training_target,
            inputs["real_state"],
            inputs["predicted_state"],
            action_embedding,
            inputs["task"],
            inputs["ema_next_state"],
            inputs["target_next_action_embedding"],
            gamma=0.9,
        )

    target = online.make_target()
    with pytest.raises(ValueError, match="leading axes"):
        build_hybrid_tdjepa_td_batch_v2(
            online,
            target,
            inputs["real_state"],
            torch.randn(2, 3, 192),
            action_embedding,
            inputs["task"],
            inputs["ema_next_state"],
            inputs["target_next_action_embedding"],
            gamma=0.9,
        )

    with pytest.raises(ValueError, match="terminal must have shape"):
        build_hybrid_tdjepa_td_batch_v2(
            online,
            target,
            inputs["real_state"],
            inputs["predicted_state"],
            action_embedding,
            inputs["task"],
            inputs["ema_next_state"],
            inputs["target_next_action_embedding"],
            gamma=0.9,
            terminal=torch.zeros(2, 1),
        )
