from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_ACTION_DIM,
    V1_ACTION_EMBEDDING_DIM,
    V1_RAW_ACTION_DIM,
    ActorFreeTDJEPAPredictorV1,
    build_tdjepa_td_batch_v1,
    ema_update_target_v1,
    encode_frozen_action_blocks_v1,
    project_tasks_to_sphere_v1,
    sample_mixed_tasks_v1,
    tdjepa_goal_score_v1,
    tdjepa_successor_td_target_v1,
)


class RecordingActionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 25
        self.emb_dim = 192
        self.projection = nn.Linear(25, 192)
        self.seen: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        self.output_dtypes: list[torch.dtype] = []

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        self.seen.append(raw_action.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        output = torch.tanh(self.projection(raw_action) + 0.125)
        self.output_dtypes.append(output.dtype)
        return output


def _frozen_encoder() -> RecordingActionEncoder:
    encoder = RecordingActionEncoder()
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _predictor(*, hidden_dim: int = 16) -> ActorFreeTDJEPAPredictorV1:
    return ActorFreeTDJEPAPredictorV1(hidden_dim=hidden_dim)


def test_v1_predictor_has_locked_embedding_dimensions_and_no_encoder_member():
    predictor = _predictor()
    signature = inspect.signature(ActorFreeTDJEPAPredictorV1.forward)

    output = predictor(
        torch.randn(3, 4, 192),
        torch.randn(3, 4, 192),
        torch.randn(3, 4, 192),
    )

    assert list(signature.parameters) == [
        "self",
        "state",
        "action_embedding",
        "task",
    ]
    assert output.shape == (3, 4, 192)
    assert V1_RAW_ACTION_DIM == predictor.raw_action_dim == 25
    assert V1_ACTION_DIM == predictor.action_dim == 192
    assert V1_ACTION_EMBEDDING_DIM == predictor.action_embedding_dim == 192
    assert predictor.state_dim == predictor.task_dim == predictor.output_dim == 192
    assert not hasattr(predictor, "action_encoder")
    assert all("action_encoder" not in name for name, _ in predictor.named_modules())
    assert all("action_encoder" not in key for key in predictor.state_dict())
    with pytest.raises(ValueError, match="dimension 192"):
        predictor(
            torch.randn(2, 192),
            torch.randn(2, 25),
            torch.randn(2, 192),
        )


def test_default_v1_predictor_exactly_matches_forward_map_widths():
    predictor = ActorFreeTDJEPAPredictorV1()

    for branch in (predictor.embed_task, predictor.embed_state_action):
        assert isinstance(branch[0], nn.Linear)
        assert (branch[0].in_features, branch[0].out_features) == (384, 256)
        assert isinstance(branch[1], nn.LayerNorm)
        assert isinstance(branch[2], nn.Tanh)
        assert isinstance(branch[3], nn.Linear)
        assert (branch[3].in_features, branch[3].out_features) == (256, 128)
        assert isinstance(branch[4], nn.ReLU)
    assert isinstance(predictor.output[0], nn.Linear)
    assert (predictor.output[0].in_features, predictor.output[0].out_features) == (
        256,
        256,
    )
    assert isinstance(predictor.output[1], nn.ReLU)
    assert isinstance(predictor.output[2], nn.Linear)
    assert (predictor.output[2].in_features, predictor.output[2].out_features) == (
        256,
        192,
    )
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 379_072


def test_action_helper_flattens_and_restores_arbitrary_leading_axes():
    torch.manual_seed(11)
    encoder = _frozen_encoder()
    raw_action = torch.randn(2, 3, 4, 25, requires_grad=True)
    reference = torch.randn(2, 3, 4, 192, dtype=torch.float64)

    encoded = encode_frozen_action_blocks_v1(encoder, raw_action, reference)

    assert encoded.shape == (2, 3, 4, 192)
    assert encoded.dtype == reference.dtype
    assert encoded.device == reference.device
    assert not encoded.requires_grad
    assert encoder.grad_enabled == [False]
    assert encoder.seen[0].shape == (24, 1, 25)
    assert torch.equal(
        encoder.seen[0], raw_action.detach().reshape(24, 1, 25)
    )
    assert raw_action.grad is None
    assert all(parameter.grad is None for parameter in encoder.parameters())

    vector_result = encode_frozen_action_blocks_v1(
        encoder,
        torch.randn(25),
        torch.randn(192),
    )
    assert vector_result.shape == (192,)


def test_action_helper_rejects_non_eval_trainable_or_wrong_shape_encoders():
    raw_action = torch.randn(2, 25)
    reference = torch.randn(2, 192)
    encoder = RecordingActionEncoder()
    encoder.requires_grad_(False)
    with pytest.raises(ValueError, match="eval mode"):
        encode_frozen_action_blocks_v1(encoder, raw_action, reference)

    encoder.eval()
    encoder.projection.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="must be frozen"):
        encode_frozen_action_blocks_v1(encoder, raw_action, reference)

    class WrongOutput(nn.Module):
        input_dim = 25
        emb_dim = 192

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    wrong = WrongOutput().eval()
    with pytest.raises(ValueError, match="must return shape"):
        encode_frozen_action_blocks_v1(wrong, raw_action, reference)

    encoder.requires_grad_(False)
    encoder.eval()
    with pytest.raises(ValueError, match="leading axes"):
        encode_frozen_action_blocks_v1(
            encoder,
            torch.randn(2, 3, 25),
            torch.randn(2, 192),
        )


def test_td_batch_uses_encoded_dataset_next_action_and_only_online_gradients():
    torch.manual_seed(17)
    online = _predictor()
    target = online.make_target()
    state = torch.randn(5, 192, requires_grad=True)
    action_embedding = torch.randn(5, 192, requires_grad=True)
    task = torch.randn(5, 192, requires_grad=True)
    next_state = torch.randn(5, 192, requires_grad=True)
    next_action_embedding_leaf = torch.arange(
        5 * 192, dtype=torch.float32, requires_grad=True
    )
    next_action_embedding = next_action_embedding_leaf.reshape(5, 192)
    recorded: dict[str, torch.Tensor] = {}

    def record_target_input(
        module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        del module
        recorded["next_action_embedding"] = inputs[1].detach().clone()

    target.register_forward_pre_hook(record_target_input)
    batch = build_tdjepa_td_batch_v1(
        online,
        target,
        state,
        action_embedding,
        task,
        next_state,
        next_action_embedding,
        gamma=0.97,
        terminal=torch.tensor([False, False, True, False, False]),
    )
    batch.td_loss.backward()

    assert torch.equal(recorded["next_action_embedding"], next_action_embedding)
    assert batch.prediction.shape == batch.target.shape == (5, 192)
    assert batch.per_transition_td_loss.shape == (5,)
    assert not batch.target.requires_grad
    assert all(
        value.grad is None
        for value in (state, action_embedding, task, next_state)
    )
    assert next_action_embedding_leaf.grad is None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in online.parameters()
    )
    assert all(parameter.grad is None for parameter in target.parameters())


@pytest.mark.skipif(
    not hasattr(torch, "autocast"), reason="torch.autocast requires modern PyTorch"
)
def test_bfloat16_helper_and_td_keep_predictor_context_and_fp32_reductions():
    torch.manual_seed(23)
    encoder = _frozen_encoder()
    online = _predictor()
    target = online.make_target()
    state = torch.randn(4, 192)
    raw_action = torch.randn(4, 25, requires_grad=True)
    task = torch.randn(4, 192)
    next_state = torch.randn(4, 192)
    next_raw_action = torch.randn(4, 25, requires_grad=True)
    terminal = torch.tensor([False, True, False, False])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        action_embedding = encode_frozen_action_blocks_v1(
            encoder, raw_action, state
        )
        next_action_embedding = encode_frozen_action_blocks_v1(
            encoder, next_raw_action, next_state
        )
        batch = build_tdjepa_td_batch_v1(
            online,
            target,
            state,
            action_embedding,
            task,
            next_state,
            next_action_embedding,
            gamma=0.97,
            terminal=terminal,
        )
        goal_score = tdjepa_goal_score_v1(batch.prediction, batch.task)
        loss = batch.td_loss - 0.01 * goal_score.mean()

    assert encoder.output_dtypes == [torch.bfloat16, torch.bfloat16]
    assert action_embedding.dtype == state.dtype == torch.float32
    assert next_action_embedding.dtype == next_state.dtype == torch.float32
    assert batch.prediction.dtype == torch.bfloat16
    assert batch.target.dtype == torch.float32
    assert batch.per_transition_td_loss.dtype == torch.float32
    assert batch.td_loss.dtype == torch.float32
    assert goal_score.dtype == torch.float32
    assert torch.equal(
        goal_score,
        (batch.prediction.float() * batch.task.float()).sum(dim=-1),
    )
    loss.backward()
    assert raw_action.grad is None
    assert next_raw_action.grad is None
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in online.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())


def test_v1_task_sampling_goal_score_and_td_target_preserve_v0_semantics():
    goals = torch.arange(1, 1 + 40 * 192, dtype=torch.float32).reshape(40, 192)
    first = sample_mixed_tasks_v1(
        goals,
        generator=torch.Generator().manual_seed(91),
    )
    second = sample_mixed_tasks_v1(
        goals,
        generator=torch.Generator().manual_seed(91),
    )

    assert torch.equal(first.goal_mask, second.goal_mask)
    assert torch.equal(first.task, second.task)
    assert first.goal_count + first.random_count == 40
    assert torch.allclose(
        torch.norm(first.task, dim=-1),
        torch.full((40,), math.sqrt(192)),
    )
    assert torch.allclose(
        first.task[first.goal_mask],
        project_tasks_to_sphere_v1(goals)[first.goal_mask],
    )
    prediction = torch.zeros(2, 192)
    task = torch.zeros(2, 192)
    prediction[:, 0] = torch.tensor([1.0, 5.0])
    task[:, 0] = torch.tensor([2.0, 3.0])
    assert torch.equal(tdjepa_goal_score_v1(prediction, task), torch.tensor([2.0, 15.0]))

    next_state = torch.full((3, 192), 2.0, requires_grad=True)
    target_next = torch.full((3, 192), 4.0, requires_grad=True)
    result = tdjepa_successor_td_target_v1(
        next_state,
        target_next,
        gamma=0.5,
        terminal=torch.tensor([False, True, False]),
    )
    expected = torch.full((3, 192), 4.0)
    expected[1].fill_(2.0)
    assert torch.equal(result, expected)
    assert not result.requires_grad


def test_v1_td_reduction_and_ema_update_match_v0():
    online = _predictor(hidden_dim=8)
    target = online.make_target()
    assert not target.training
    assert all(not parameter.requires_grad for parameter in target.parameters())

    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        for parameter in online.parameters():
            parameter.fill_(10.0)
    ema_update_target_v1(target, online, decay=0.9)
    assert all(
        torch.allclose(parameter, torch.ones_like(parameter))
        for parameter in target.parameters()
    )
    assert all(not parameter.requires_grad for parameter in target.parameters())

    zero_online = _predictor()
    zero_target = zero_online.make_target()
    for predictor in (zero_online, zero_target):
        for parameter in predictor.parameters():
            parameter.data.zero_()
    next_state = torch.stack((torch.ones(192), torch.full((192,), 2.0)))
    batch = build_tdjepa_td_batch_v1(
        zero_online,
        zero_target,
        torch.zeros(2, 192),
        torch.zeros(2, 192),
        torch.ones(2, 192),
        next_state,
        torch.zeros(2, 192),
        gamma=0.95,
    )
    assert torch.equal(batch.per_transition_td_loss, torch.tensor([192.0, 768.0]))
    assert batch.td_loss == 480.0
