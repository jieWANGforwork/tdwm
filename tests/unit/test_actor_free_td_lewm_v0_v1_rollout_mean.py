from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tdwm.adapters.frozen_actor_free_td_v0_common import ActorFreeTDLeWMV0
from tdwm.adapters.frozen_actor_free_td_v1_common import ActorFreeTDLeWMV1

MODE = "g_only_f_rollout_mean"


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
        self.rollout_actions: list[torch.Tensor] = []
        self.eval()

    def encode(self, info):
        return info

    def rollout(self, info, action_sequence, history_size=None):
        del history_size
        self.rollout_horizons.append(int(action_sequence.shape[-2]))
        self.rollout_actions.append(action_sequence.detach().clone())
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


class RawActionPredictor(nn.Module):
    state_dim = 192
    action_dim = 25
    task_dim = 192
    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.seen_state: list[torch.Tensor] = []
        self.seen_action: list[torch.Tensor] = []
        self.seen_task: list[torch.Tensor] = []

    def forward(self, state, action, task):
        self.seen_state.append(state.detach().clone())
        self.seen_action.append(action.detach().clone())
        self.seen_task.append(task.detach().clone())
        output = torch.zeros_like(state) + self.anchor
        output[..., 0] = state[..., 0] + action[..., 0]
        return output


class EmbeddedActionPredictor(RawActionPredictor):
    raw_action_dim = 25
    action_dim = 192
    action_embedding_dim = 192


@dataclass(frozen=True)
class VersionCase:
    name: str
    adapter_type: type[nn.Module]
    embedded_action: bool


CASES = (
    VersionCase("v0", ActorFreeTDLeWMV0, False),
    VersionCase("v1", ActorFreeTDLeWMV1, True),
)


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
    case: VersionCase,
    *,
    score_mode: str,
    gamma: float = 0.95,
    g_first_weight: float | None = None,
) -> tuple[nn.Module, RecordingWorld, RawActionPredictor]:
    world = RecordingWorld()
    predictor: RawActionPredictor
    predictor = (
        EmbeddedActionPredictor() if case.embedded_action else RawActionPredictor()
    )
    adapter = case.adapter_type(
        world,
        predictor,
        gamma=gamma,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    return adapter, world, predictor


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rollout_mean_aligns_five_states_actions_and_native_action_path(
    case: VersionCase,
) -> None:
    adapter, world, predictor = _adapter(case, score_mode=MODE)
    actions = _actions(5)

    cost = adapter.get_cost(_info(), actions)  # type: ignore[attr-defined]

    assert world.rollout_horizons == [5]
    assert torch.equal(world.rollout_actions[0], actions)
    assert predictor.seen_state[-1].shape == (1, 1, 5, 192)
    assert torch.equal(
        predictor.seen_state[-1][0, 0, :, 0],
        torch.tensor([7.0, 11.0, 12.0, 13.0, 14.0]),
    )
    if case.embedded_action:
        assert torch.equal(world.action_encoder.seen[-1], actions.reshape(-1, 1, 25))
        assert predictor.seen_action[-1].shape == (1, 1, 5, 192)
        assert torch.equal(
            predictor.seen_action[-1][0, 0, :, 0],
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
    else:
        assert world.action_encoder.seen == []
        assert torch.equal(predictor.seen_action[-1], actions)
    expected_task = torch.zeros(5, 192)
    expected_task[:, 0] = math.sqrt(192.0)
    assert torch.equal(predictor.seen_task[-1][0, 0], expected_task)
    expected = -math.sqrt(192.0) * torch.tensor(
        [[(8.0 + 13.0 + 15.0 + 17.0 + 19.0) / 5.0]]
    )
    assert torch.allclose(cost, expected)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rollout_mean_horizon_one_helper_equals_native_g_only(
    case: VersionCase,
) -> None:
    direct, direct_world, _ = _adapter(case, score_mode="g_only")
    mean, mean_world, _ = _adapter(case, score_mode=MODE)
    actions = _actions(1, samples=2)
    actions[0, 1, 0, 0] = -3.0

    direct_cost = direct.get_cost(_info(samples=2), actions)  # type: ignore[attr-defined]
    helper_cost = mean._g_only_f_rollout_mean_cost(  # type: ignore[attr-defined]
        _info(samples=2),
        actions,
    )

    assert torch.equal(helper_cost, direct_cost)
    assert direct_world.rollout_horizons == []
    assert mean_world.rollout_horizons == [1]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rollout_mean_uses_f_only_for_transitions_not_goal_distance_or_gamma(
    case: VersionCase,
) -> None:
    low_gamma, _, _ = _adapter(case, score_mode=MODE, gamma=0.0)
    high_gamma, _, _ = _adapter(case, score_mode=MODE, gamma=0.95)
    actions = _actions(5)

    with patch.object(
        case.adapter_type,
        "_explicit_terminal_cost",
        side_effect=AssertionError("terminal F goal-distance must not be used"),
    ):
        low_cost = low_gamma.get_cost(_info(), actions)  # type: ignore[attr-defined]
        high_cost = high_gamma.get_cost(_info(), actions)  # type: ignore[attr-defined]

    assert torch.equal(low_cost, high_cost)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rollout_mean_public_mode_requires_exactly_five_blocks(
    case: VersionCase,
) -> None:
    adapter, world, _ = _adapter(case, score_mode=MODE)

    with pytest.raises(ValueError, match="horizon=5"):
        adapter.get_cost(_info(), _actions(1))  # type: ignore[attr-defined]

    assert world.rollout_horizons == []


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("score_mode", "horizon", "g_first_weight"),
    (
        ("f_only", 5, None),
        ("g_only", 1, None),
        ("f_plus_g", 5, None),
        ("f_plus_g_first", 5, 0.25),
    ),
)
def test_existing_score_modes_never_route_through_rollout_mean(
    case: VersionCase,
    score_mode: str,
    horizon: int,
    g_first_weight: float | None,
) -> None:
    adapter, _, _ = _adapter(
        case,
        score_mode=score_mode,
        g_first_weight=g_first_weight,
    )
    with patch.object(
        adapter,
        "_g_only_f_rollout_mean_cost",
        side_effect=AssertionError("legacy mode entered rollout-mean path"),
    ):
        cost = adapter.get_cost(_info(), _actions(horizon))  # type: ignore[attr-defined]

    assert cost.shape == (1, 1)
