from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v2_common import (
    V2_SCORE_MODES,
    ActorFreeTDLeWMV2,
)
from tdwm.adapters.frozen_actor_free_td_v1_common import (
    SCORE_MODES as V1_SCORE_MODES,
)
from tdwm.adapters.frozen_actor_free_td_v1_common import ActorFreeTDLeWMV1

ROLLOUT_MEAN_MODE = "g_only_f_rollout_mean"


class RecordingActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.seen: list[torch.Tensor] = []
        self.eval()

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        self.seen.append(raw_action.detach().clone())
        embedding = raw_action.new_zeros(*raw_action.shape[:-1], 192)
        embedding[..., 0] = raw_action[..., 0]
        return embedding + self.anchor


class RecordingWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.action_encoder = RecordingActionEncoder()
        self.rollout_horizons: list[int] = []
        self.eval()

    def encode(self, info):
        return info

    def rollout(self, info, action_sequence, history_size=None):
        del history_size
        self.rollout_horizons.append(int(action_sequence.shape[-2]))
        batch, samples, horizon = action_sequence.shape[:3]
        observed = int(info["emb"].shape[-2])
        predicted = action_sequence.new_zeros(
            batch,
            samples,
            observed + horizon,
            192,
        )
        predicted[..., :observed, :] = info["emb"].to(action_sequence)
        for step in range(horizon):
            predicted[..., observed + step, 0] = 11.0 + step
        return {"predicted_emb": predicted + self.anchor}


class RecordingPredictor(nn.Module):
    state_dim = 192
    raw_action_dim = 25
    action_dim = 192
    action_embedding_dim = 192
    task_dim = 192
    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.seen_state: list[torch.Tensor] = []
        self.seen_action: list[torch.Tensor] = []
        self.seen_task: list[torch.Tensor] = []

    def forward(self, state, action_embedding, task):
        self.seen_state.append(state.detach().clone())
        self.seen_action.append(action_embedding.detach().clone())
        self.seen_task.append(task.detach().clone())
        output = torch.zeros_like(state) + self.anchor
        output[..., 0] = state[..., 0] + action_embedding[..., 0]
        return output


def _info(*, samples: int = 1) -> dict[str, torch.Tensor]:
    current = torch.zeros(1, samples, 1, 192)
    current[..., 0] = 7.0
    goal = torch.zeros(1, 1, 192)
    goal[..., 0] = 1.0
    return {"emb": current, "goal_emb": goal}


def _actions(horizon: int, *, samples: int = 1) -> torch.Tensor:
    actions = torch.zeros(1, samples, horizon, 25)
    actions[..., 0] = torch.arange(1, horizon + 1, dtype=actions.dtype)
    return actions


def _adapter(
    *,
    score_mode: str,
    gamma: float = 0.95,
    g_first_weight: float | None = None,
) -> tuple[ActorFreeTDLeWMV2, RecordingWorld, RecordingPredictor]:
    world = RecordingWorld()
    predictor = RecordingPredictor()
    adapter = ActorFreeTDLeWMV2(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=gamma,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    return adapter, world, predictor


def test_rollout_mean_aligns_five_predecessor_states_actions_and_tasks() -> None:
    adapter, world, predictor = _adapter(score_mode=ROLLOUT_MEAN_MODE)
    actions = _actions(5)

    cost = adapter.get_cost(_info(), actions)

    assert world.rollout_horizons == [5]
    assert predictor.seen_state[-1].shape == (1, 1, 5, 192)
    assert torch.equal(
        predictor.seen_state[-1][0, 0, :, 0],
        torch.tensor([7.0, 11.0, 12.0, 13.0, 14.0]),
    )
    assert torch.equal(
        world.action_encoder.seen[-1],
        actions.reshape(-1, 1, 25),
    )
    assert torch.equal(
        predictor.seen_action[-1][0, 0, :, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    expected_task = torch.zeros(5, 192)
    expected_task[:, 0] = math.sqrt(192.0)
    assert torch.equal(predictor.seen_task[-1][0, 0], expected_task)
    expected_score = -math.sqrt(192.0) * torch.tensor(
        [[(8.0 + 13.0 + 15.0 + 17.0 + 19.0) / 5.0]]
    )
    assert torch.allclose(cost, expected_score)


def test_rollout_mean_horizon_one_helper_equals_existing_g_only() -> None:
    direct, direct_world, _ = _adapter(score_mode="g_only")
    rollout_mean, rollout_world, _ = _adapter(score_mode=ROLLOUT_MEAN_MODE)
    actions = _actions(1, samples=2)
    actions[0, 1, 0, 0] = -3.0

    direct_cost = direct.get_cost(_info(samples=2), actions)
    helper_cost = rollout_mean._g_only_f_rollout_mean_cost(
        _info(samples=2),
        actions,
    )

    assert torch.equal(helper_cost, direct_cost)
    assert direct_world.rollout_horizons == []
    assert rollout_world.rollout_horizons == [1]


def test_rollout_mean_uses_f_only_for_states_and_ignores_gamma() -> None:
    low_gamma, _, _ = _adapter(score_mode=ROLLOUT_MEAN_MODE, gamma=0.0)
    high_gamma, _, _ = _adapter(score_mode=ROLLOUT_MEAN_MODE, gamma=0.95)
    actions = _actions(5)

    with patch.object(
        ActorFreeTDLeWMV2,
        "_explicit_terminal_cost",
        side_effect=AssertionError("terminal F cost must not be used"),
    ):
        low_cost = low_gamma.get_cost(_info(), actions)
        high_cost = high_gamma.get_cost(_info(), actions)

    assert torch.equal(low_cost, high_cost)


def test_rollout_mean_public_mode_requires_horizon_five() -> None:
    adapter, world, _ = _adapter(score_mode=ROLLOUT_MEAN_MODE)

    with pytest.raises(ValueError, match="horizon=5"):
        adapter.get_cost(_info(), _actions(1))

    assert world.rollout_horizons == []


@pytest.mark.parametrize(
    ("score_mode", "g_first_weight"),
    (
        ("f_only", None),
        ("g_only", None),
        ("f_plus_g", None),
        ("f_plus_g_first", 1.0),
    ),
)
def test_existing_v2_modes_still_delegate_unchanged_to_v1(
    score_mode: str,
    g_first_weight: float | None,
) -> None:
    adapter, _, _ = _adapter(
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    sentinel = torch.tensor([[123.0]])
    with patch.object(ActorFreeTDLeWMV1, "get_cost", return_value=sentinel) as parent:
        result = adapter.get_cost({}, torch.empty(0))

    assert result is sentinel
    parent.assert_called_once()


def test_rollout_mean_is_v2_only() -> None:
    assert ROLLOUT_MEAN_MODE in V2_SCORE_MODES
    assert ROLLOUT_MEAN_MODE not in V1_SCORE_MODES
    with pytest.raises(ValueError, match="Unsupported"):
        ActorFreeTDLeWMV1(
            RecordingWorld(),
            RecordingPredictor(),  # type: ignore[arg-type]
            gamma=0.95,
            score_mode=ROLLOUT_MEAN_MODE,
        )
