from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v2_common import ActorFreeTDLeWMV2
from tdwm.adapters.frozen_actor_free_td_v0_common import ActorFreeTDLeWMV0
from tdwm.adapters.frozen_actor_free_td_v1_common import (
    FIRST_Q2_SCORE_MODE,
    FIRST_Q2_STD_EPSILON,
    ActorFreeTDLeWMV1,
    _normalize_cem_candidate_scores,
)

FIRST_ACTION_MODE = "f_plus_g_first"


class RecordingActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)
        self.seen: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            self.projection.weight[:25, :25].copy_(torch.eye(25))
        self.requires_grad_(False).eval()

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        self.seen.append(raw_action.detach().clone())
        output = self.projection(raw_action)
        self.outputs.append(output.detach().clone())
        return output


class RecordingWorld(nn.Module):
    def __init__(
        self,
        predicted: torch.Tensor | None = None,
        *,
        mutate_info_on_rollout: bool = False,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_encoder = RecordingActionEncoder()
        self.predicted = predicted
        self.rollout_calls = 0
        self.rollout_history_sizes: list[int | None] = []
        self.rollout_horizons: list[int] = []
        self.rollout_actions: list[torch.Tensor] = []
        self.mutate_info_on_rollout = mutate_info_on_rollout

    def encode(self, info):
        if "emb" not in info:
            batch = info["pixels"].shape[0]
            info["emb"] = self.anchor.new_zeros(batch, 1, 192)
        return info

    def rollout(self, info, action_sequence, history_size=None):
        self.rollout_calls += 1
        self.rollout_history_sizes.append(history_size)
        self.rollout_horizons.append(int(action_sequence.shape[-2]))
        self.rollout_actions.append(action_sequence.detach().clone())
        if self.mutate_info_on_rollout:
            # A planner must snapshot the real online z0 before F rollout.  This
            # emulates a rollout implementation that refreshes its info cache.
            info["emb"] = torch.full_like(info["emb"], 123.0)
        if self.predicted is None:
            raise AssertionError("rollout must not be called")
        return {"predicted_emb": self.predicted.to(action_sequence)}


class RawActionScorePredictor(nn.Module):
    state_dim = 192
    action_dim = 25
    task_dim = 192
    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0
        self.last_state: torch.Tensor | None = None
        self.last_action: torch.Tensor | None = None

    def forward(self, state, action, task):
        del task
        self.calls += 1
        self.last_state = state.detach().clone()
        self.last_action = action.detach().clone()
        output = torch.zeros_like(state) + self.anchor
        output[..., 0] = action[..., 0]
        return output


class EmbeddedActionScorePredictor(nn.Module):
    state_dim = 192
    raw_action_dim = 25
    action_dim = 192
    action_embedding_dim = 192
    task_dim = 192
    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0
        self.last_state: torch.Tensor | None = None
        self.last_action: torch.Tensor | None = None

    def forward(self, state, action_embedding, task):
        del task
        self.calls += 1
        self.last_state = state.detach().clone()
        self.last_action = action_embedding.detach().clone()
        output = torch.zeros_like(state) + self.anchor
        output[..., 0] = action_embedding[..., 0]
        return output


@dataclass(frozen=True)
class VersionCase:
    name: str
    adapter_type: type[nn.Module]
    embeds_action: bool


VERSION_CASES = (
    VersionCase("v0", ActorFreeTDLeWMV0, False),
    VersionCase("v1", ActorFreeTDLeWMV1, True),
    VersionCase("v2", ActorFreeTDLeWMV2, True),
)


def _predictor(case: VersionCase) -> nn.Module:
    if case.embeds_action:
        return EmbeddedActionScorePredictor()
    return RawActionScorePredictor()


def _adapter(
    case: VersionCase,
    world: RecordingWorld,
    predictor: nn.Module,
    *,
    score_mode: str,
    g_first_weight: float | None = None,
    pass_weight: bool = False,
) -> nn.Module:
    kwargs = {
        "gamma": 0.5,
        "score_mode": score_mode,
    }
    if pass_weight:
        kwargs["g_first_weight"] = g_first_weight
    return case.adapter_type(world, predictor, **kwargs)  # type: ignore[arg-type]


def _one_axis_goal(batch: int = 1) -> torch.Tensor:
    goal = torch.zeros(batch, 1, 192)
    goal[..., 0] = 1.0
    return goal


def _current_state(samples: int = 1) -> torch.Tensor:
    state = torch.zeros(1, samples, 1, 192)
    state[0, :, 0, 11] = torch.arange(7, 7 + samples, dtype=state.dtype)
    return state


def _assert_action_path(
    case: VersionCase,
    *,
    world: RecordingWorld,
    predictor: nn.Module,
    expected_raw_action: torch.Tensor,
) -> None:
    assert predictor.last_action is not None  # type: ignore[attr-defined]
    if not case.embeds_action:
        assert predictor.last_action.shape[-1] == 25  # type: ignore[attr-defined]
        assert torch.equal(  # type: ignore[attr-defined]
            predictor.last_action,
            expected_raw_action,
        )
        assert world.action_encoder.seen == []
        return

    assert predictor.last_action.shape[-1] == 192  # type: ignore[attr-defined]
    expected_flat = expected_raw_action.reshape(-1, 1, 25)
    assert torch.equal(world.action_encoder.seen[-1], expected_flat)
    assert torch.equal(  # type: ignore[attr-defined]
        predictor.last_action,
        world.action_encoder.outputs[-1].reshape(*expected_raw_action.shape[:-1], 192),
    )


@pytest.mark.parametrize("case", VERSION_CASES, ids=lambda case: case.name)
def test_first_action_alpha_zero_is_bit_equal_to_full_horizon_f_only(
    case: VersionCase,
) -> None:
    predicted = torch.zeros(1, 1, 6, 192)
    predicted[..., 1:5, :] = 1000.0
    predicted[..., -1, :] = 0.0
    predicted[..., -1, 7] = 2.0
    actions = torch.arange(125, dtype=torch.float32).reshape(1, 1, 5, 25) / 100.0
    info = {"emb": _current_state(), "goal_emb": _one_axis_goal()}

    f_world = RecordingWorld(predicted.clone())
    f_predictor = _predictor(case)
    f_only = _adapter(
        case,
        f_world,
        f_predictor,
        score_mode="f_only",
    )
    f_cost = f_only.get_cost(dict(info), actions)  # type: ignore[attr-defined]

    first_world = RecordingWorld(predicted.clone())
    first_predictor = _predictor(case)
    first = _adapter(
        case,
        first_world,
        first_predictor,
        score_mode=FIRST_ACTION_MODE,
        g_first_weight=0.0,
        pass_weight=True,
    )
    first_cost = first.get_cost(dict(info), actions)  # type: ignore[attr-defined]

    assert torch.equal(first_cost, f_cost)
    assert torch.equal(first_cost, torch.tensor([[5.0]]))
    assert f_world.rollout_horizons == [5]
    assert first_world.rollout_horizons == [5]
    assert torch.equal(f_world.rollout_actions[0], actions)
    assert torch.equal(first_world.rollout_actions[0], actions)


@pytest.mark.parametrize("case", VERSION_CASES, ids=lambda case: case.name)
def test_first_action_critic_uses_z0_and_a1_after_full_five_block_rollout(
    case: VersionCase,
) -> None:
    samples = 2
    predicted = torch.zeros(1, samples, 6, 192)
    predicted[..., 4, 11] = 99.0  # z_hat4: deliberately unlike the online z0.
    predicted[..., 5, 7] = 2.0  # z_hat5 determines terminal summed MSE.
    actions = torch.zeros(1, samples, 5, 25)
    actions[0, :, 0, 0] = torch.tensor([2.0, -3.0])
    actions[0, :, 4, 0] = torch.tensor([40.0, 50.0])
    current = _current_state(samples)
    info = {"emb": current, "goal_emb": _one_axis_goal()}
    alpha = 0.25

    world = RecordingWorld(predicted, mutate_info_on_rollout=True)
    predictor = _predictor(case)
    adapter = _adapter(
        case,
        world,
        predictor,
        score_mode=FIRST_ACTION_MODE,
        g_first_weight=alpha,
        pass_weight=True,
    )
    cost = adapter.get_cost(info, actions)  # type: ignore[attr-defined]

    expected = torch.full((1, samples), 5.0) - alpha * math.sqrt(192.0) * torch.tensor(
        [[2.0, -3.0]]
    )
    assert torch.allclose(cost, expected)
    assert world.rollout_horizons == [5]
    assert torch.equal(world.rollout_actions[0], actions)
    assert predictor.calls == 1  # type: ignore[attr-defined]
    assert predictor.last_state is not None  # type: ignore[attr-defined]
    assert torch.equal(predictor.last_state, current[..., -1, :])  # type: ignore[attr-defined]
    assert not torch.equal(predictor.last_state[..., 11], predicted[..., 4, 11])  # type: ignore[attr-defined]
    _assert_action_path(
        case,
        world=world,
        predictor=predictor,
        expected_raw_action=actions[..., 0, :],
    )


@pytest.mark.parametrize("case", VERSION_CASES, ids=lambda case: case.name)
def test_existing_tail_f_plus_g_still_uses_zhat4_a5_and_gamma_four(
    case: VersionCase,
) -> None:
    # The legacy H=5 branch rolls only A1..A4, so observed+future has length 5.
    predicted = torch.zeros(1, 1, 5, 192)
    predicted[..., 1:4, :] = 1000.0
    predicted[..., 4, 7] = 9.0
    actions = torch.zeros(1, 1, 5, 25)
    actions[..., 0, 0] = 2.0
    actions[..., 4, 0] = 40.0
    current = _current_state()
    info = {"emb": current, "goal_emb": _one_axis_goal()}

    world = RecordingWorld(predicted)
    predictor = _predictor(case)
    adapter = _adapter(
        case,
        world,
        predictor,
        score_mode="f_plus_g",
    )
    cost = adapter.get_cost(info, actions)  # type: ignore[attr-defined]

    # Reproduce the legacy float32 operation order, not merely the same real
    # number, so this locks the old tail score bit-for-bit.
    legacy_q = torch.tensor([[40.0]]) * torch.tensor(math.sqrt(192.0))
    expected = torch.tensor([[1.0 + 9.0**2]]) - (0.5**4) * legacy_q
    assert torch.equal(cost, expected)
    assert world.rollout_horizons == [4]
    assert torch.equal(world.rollout_actions[0], actions[..., :4, :])
    assert predictor.last_state is not None  # type: ignore[attr-defined]
    assert predictor.last_state[0, 0, 7].item() == 9.0  # type: ignore[attr-defined]
    assert not torch.equal(predictor.last_state, current[..., -1, :])  # type: ignore[attr-defined]
    _assert_action_path(
        case,
        world=world,
        predictor=predictor,
        expected_raw_action=actions[..., 4, :],
    )


@pytest.mark.parametrize("case", VERSION_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("bad_weight", (None, True, -0.01, float("nan"), float("inf")))
def test_first_action_weight_must_be_explicit_finite_and_nonnegative(
    case: VersionCase,
    bad_weight: float | bool | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _adapter(
            case,
            RecordingWorld(),
            _predictor(case),
            score_mode=FIRST_ACTION_MODE,
            g_first_weight=bad_weight,  # type: ignore[arg-type]
            pass_weight=True,
        )


@pytest.mark.parametrize("case", VERSION_CASES, ids=lambda case: case.name)
def test_old_score_modes_reject_first_action_weight(case: VersionCase) -> None:
    with pytest.raises((TypeError, ValueError)):
        _adapter(
            case,
            RecordingWorld(),
            _predictor(case),
            score_mode="f_only",
            g_first_weight=1.0,
            pass_weight=True,
        )


@pytest.mark.parametrize("case", VERSION_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("horizon", (1, 4, 6))
def test_first_action_mode_requires_exactly_five_blocks(
    case: VersionCase,
    horizon: int,
) -> None:
    world = RecordingWorld()
    adapter = _adapter(
        case,
        world,
        _predictor(case),
        score_mode=FIRST_ACTION_MODE,
        g_first_weight=1.0,
        pass_weight=True,
    )
    actions = torch.zeros(1, 1, horizon, 25)
    info = {"emb": _current_state(), "goal_emb": _one_axis_goal()}

    with pytest.raises(ValueError, match="horizon=5"):
        adapter.get_cost(info, actions)  # type: ignore[attr-defined]
    assert world.rollout_calls == 0


def _population_zscore(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    return centered / centered.square().mean(dim=1, keepdim=True).sqrt()


def test_first_q2_normalizes_f_and_q_independently_per_cem_candidate_set() -> None:
    batch, samples = 2, 3
    terminal_axis = torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 7.0]])
    first_actions = torch.tensor([[1.0, 2.0, 10.0], [100.0, 110.0, 130.0]])
    predicted = torch.zeros(batch, samples, 6, 192)
    predicted[..., -1, 7] = terminal_axis
    actions = torch.zeros(batch, samples, 5, 25)
    actions[..., 0, 0] = first_actions
    current = torch.zeros(batch, samples, 1, 192)
    current[..., 0, 11] = torch.tensor([[7.0, 8.0, 9.0], [17.0, 18.0, 19.0]])
    goal = _one_axis_goal(batch)
    alpha = 0.25

    world = RecordingWorld(predicted, mutate_info_on_rollout=True)
    predictor = EmbeddedActionScorePredictor()
    adapter = ActorFreeTDLeWMV1(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=0.5,
        score_mode=FIRST_Q2_SCORE_MODE,
        g_first_weight=alpha,
    )
    cost = adapter.get_cost(
        {"emb": current, "goal_emb": goal},
        actions,
    )

    f_cost = 1.0 + terminal_axis.square()
    q_first = math.sqrt(192.0) * first_actions
    expected = _population_zscore(f_cost) - alpha * _population_zscore(q_first)
    torch.testing.assert_close(cost, expected)
    torch.testing.assert_close(cost.mean(dim=1), torch.zeros(batch), atol=1e-6, rtol=0)
    assert world.rollout_horizons == [5]
    assert torch.equal(world.rollout_actions[0], actions)
    assert predictor.calls == 1
    assert predictor.last_state is not None
    assert torch.equal(predictor.last_state, current[..., -1, :])
    _assert_action_path(
        VERSION_CASES[1],
        world=world,
        predictor=predictor,
        expected_raw_action=actions[..., 0, :],
    )


def test_first_q2_constant_and_singleton_candidate_signals_are_zero_not_nan() -> None:
    constant = torch.tensor([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    singleton = torch.tensor([[4.0], [-2.0]])

    assert torch.equal(
        _normalize_cem_candidate_scores(constant), torch.zeros_like(constant)
    )
    assert torch.equal(
        _normalize_cem_candidate_scores(singleton),
        torch.zeros_like(singleton),
    )

    almost_constant = torch.tensor([[1.0, 1.0 + FIRST_Q2_STD_EPSILON / 4.0]])
    assert torch.equal(
        _normalize_cem_candidate_scores(almost_constant),
        torch.zeros_like(almost_constant),
    )


@pytest.mark.parametrize(
    "bad",
    (
        torch.tensor([1.0]),
        torch.ones(1, 0),
        torch.ones(1, 2, dtype=torch.int64),
        torch.tensor([[1.0, math.inf]]),
    ),
)
def test_first_q2_normalization_rejects_invalid_cem_scores(bad: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        _normalize_cem_candidate_scores(bad)


@pytest.mark.parametrize("bad_weight", (None, True, -0.01, math.nan, math.inf))
def test_first_q2_requires_explicit_finite_nonnegative_weight(
    bad_weight: float | bool | None,
) -> None:
    with pytest.raises((TypeError, ValueError), match="g_first_weight"):
        ActorFreeTDLeWMV1(
            RecordingWorld(),
            EmbeddedActionScorePredictor(),  # type: ignore[arg-type]
            gamma=0.5,
            score_mode=FIRST_Q2_SCORE_MODE,
            g_first_weight=bad_weight,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("horizon", (1, 4, 6))
def test_first_q2_requires_exactly_five_action_blocks(horizon: int) -> None:
    world = RecordingWorld()
    adapter = ActorFreeTDLeWMV1(
        world,
        EmbeddedActionScorePredictor(),  # type: ignore[arg-type]
        gamma=0.5,
        score_mode=FIRST_Q2_SCORE_MODE,
        g_first_weight=0.25,
    )
    actions = torch.zeros(1, 2, horizon, 25)

    with pytest.raises(ValueError, match="horizon=5"):
        adapter.get_cost(
            {"emb": _current_state(2), "goal_emb": _one_axis_goal()},
            actions,
        )
    assert world.rollout_calls == 0
