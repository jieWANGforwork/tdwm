from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.adapters.g_weighted_cem import (
    GWeightedCEMConfig,
    GWeightedEliteUpdate,
    GWeightedPlanningModel,
    elite_g_weights,
    weighted_elite_moments,
)


def test_import_and_default_formula():
    config = GWeightedCEMConfig("path")
    assert config.temperature == 1.0
    assert config.evaluation_mode == "g_path_weighted_cem"


@pytest.mark.parametrize("mode", ("path", "action"))
def test_weights_normalize_across_paths_not_time(mode):
    q = torch.tensor([[[1.0, 8.0], [3.0, 2.0], [2.0, 5.0]]])
    weights = elite_g_weights(q, GWeightedCEMConfig(mode, temperature=2.0))
    expected = q.mean(-1, keepdim=True).expand_as(q) if mode == "path" else q
    torch.testing.assert_close(weights, torch.softmax(expected / 2.0, dim=1))
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 2))
    if mode == "path":
        assert torch.equal(weights[..., 0], weights[..., 1])
    else:
        assert not torch.equal(weights[..., 0], weights[..., 1])


@pytest.mark.parametrize("mode", ("path", "action"))
def test_equal_q_exactly_recovers_baseline_mean_and_sample_std(mode):
    actions = torch.randn(2, 30, 5, 25, generator=torch.Generator().manual_seed(7))
    q = torch.full((2, 30, 5), 100.0)
    mean, std = weighted_elite_moments(
        actions, elite_g_weights(q, GWeightedCEMConfig(mode))
    )
    assert torch.equal(mean, actions.mean(1))
    assert torch.equal(std, actions.std(1))


def test_weighted_moments_use_each_positions_weights_and_sample_correction():
    actions = torch.tensor([[[[1.0], [10.0]], [[3.0], [20.0]]]])
    weights = torch.tensor([[[0.25, 0.8], [0.75, 0.2]]])
    mean, std = weighted_elite_moments(actions, weights)
    torch.testing.assert_close(mean, torch.tensor([[[2.5], [12.0]]]))
    numerator = (weights[..., None] * (actions - mean[:, None]).square()).sum(1)
    denominator = 1.0 - weights.square().sum(1)
    torch.testing.assert_close(std, (numerator / denominator[..., None]).sqrt())


def test_concentrated_weights_do_not_produce_nan():
    q = torch.tensor([[[1.0e6], [-1.0e6]]])
    weights = elite_g_weights(q, GWeightedCEMConfig("action"))
    mean, std = weighted_elite_moments(torch.tensor([[[[2.0]], [[9.0]]]]), weights)
    assert mean.item() == 2.0
    assert std.item() == 0.0
    assert torch.isfinite(std).all()


@pytest.mark.parametrize("temperature", (0, -1, float("inf"), float("nan"), True))
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        GWeightedCEMConfig("path", temperature)


def test_invalid_mode_and_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError):
        GWeightedCEMConfig("first")
    with pytest.raises(ValueError):
        elite_g_weights(
            torch.full((1, 30, 5), float("nan")), GWeightedCEMConfig("path")
        )


class RecordingAdapter(nn.Module):
    score_mode = "f_only"

    def __init__(self, constant_q=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.constant_q = constant_q
        self.rollouts = []
        self.g_calls = []
        self.eval()

    def _current_state_for_samples(self, info, *, batch, samples, reference):
        return info["emb"][..., -1, :].to(reference)

    def _goal_for_samples(self, info, *, batch, samples, reference):
        return info["goal_emb"][..., -1, :].to(reference)

    def _rollout_future(self, info, actions, *, batch, samples, horizon):
        self.rollouts.append(actions.detach().clone())
        initial = info["emb"][..., -1, :]
        future = initial.unsqueeze(-2).expand(batch, samples, horizon, 192).clone()
        future[..., 0] += actions[..., 0].cumsum(-1)
        return future

    def _explicit_terminal_cost(self, future, goal):
        return (future[..., -1, :] - goal).square().sum(-1)

    def _goal_score(self, state, actions, task):
        self.g_calls.append((state.detach().clone(), actions.detach().clone(), task))
        if self.constant_q:
            return torch.zeros_like(state[..., 0])
        return state[..., 0] + actions[..., 0] + task[..., 0]

    def get_cost(self, info, actions):
        batch, samples, horizon = actions.shape[:3]
        goal = self._goal_for_samples(
            info, batch=batch, samples=samples, reference=actions
        )
        future = self._rollout_future(
            info, actions, batch=batch, samples=samples, horizon=horizon
        )
        return self._explicit_terminal_cost(future, goal)


def test_only_elites_reach_g_with_their_own_prefix_states():
    base = RecordingAdapter()
    model = GWeightedPlanningModel(base)
    actions = torch.zeros(1, 3, 5, 25)
    actions[0, :, :, 0] = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ]
    )
    info = {"emb": torch.zeros(1, 3, 1, 192), "goal_emb": torch.ones(1, 3, 1, 192)}
    expected = base.get_cost(info, actions)
    assert torch.equal(model.get_cost(info, actions), expected)
    assert base.g_calls == []
    inds = torch.tensor([[2, 0]])
    elites = actions[:, [2, 0]]
    scores = model.score_elites(actions, inds, elites)
    states, supplied_actions, tasks = base.g_calls[-1]
    assert scores.shape == (1, 2, 5)
    assert torch.equal(supplied_actions, elites)
    torch.testing.assert_close(
        states[0, :, :, 0],
        torch.tensor([[0.0, 3.0, 7.0, 12.0, 18.0], [0.0, 1.0, 3.0, 6.0, 10.0]]),
    )
    assert tasks.shape == states.shape
    assert not scores.requires_grad
    with pytest.raises(RuntimeError, match="cached"):
        model.score_elites(actions.clone(), inds, elites)


@pytest.mark.parametrize("mode", ("path", "action"))
@pytest.mark.parametrize("constant_q", (True, False))
def test_installed_cem_callback_changes_next_population_and_final_mean(
    mode, constant_q
):
    import numpy as np
    import stable_worldmodel as swm
    from gymnasium.spaces import Box

    raw = RecordingAdapter(constant_q=constant_q)
    wrapped = GWeightedPlanningModel(RecordingAdapter(constant_q=constant_q))
    callback = GWeightedEliteUpdate(wrapped, GWeightedCEMConfig(mode))
    args = dict(batch_size=1, num_samples=8, n_steps=2, topk=3, seed=42)
    baseline = swm.solver.CEMSolver(model=raw, **args)
    weighted = swm.solver.CEMSolver(model=wrapped, callbacks=[callback], **args)
    config = SimpleNamespace(horizon=5, action_block=5)
    for solver in (baseline, weighted):
        solver.configure(
            action_space=Box(-1.0, 1.0, (2, 5), dtype=np.float32),
            n_envs=2,
            config=config,
        )
    info = {"emb": torch.zeros(2, 1, 192), "goal_emb": torch.ones(2, 1, 192)}
    before = baseline.solve(info)
    after = weighted.solve(info)
    assert torch.equal(raw.rollouts[0], wrapped.base.rollouts[0])
    assert len(wrapped.base.g_calls) == 4  # 2 environment batches, 2 iterations.
    assert all(call[0].shape == (1, 3, 5, 192) for call in wrapped.base.g_calls)
    assert len(after["callbacks"][callback.output_key]) == 2
    if constant_q:
        assert torch.equal(before["actions"], after["actions"])
        assert torch.equal(before["var"][0], after["var"][0])
        assert all(
            torch.equal(a, b) for a, b in zip(raw.rollouts, wrapped.base.rollouts)
        )
    else:
        assert not torch.equal(raw.rollouts[1], wrapped.base.rollouts[1])
        assert not torch.equal(before["actions"], after["actions"])
    assert all(parameter.grad is None for parameter in wrapped.parameters())


@pytest.mark.parametrize("version", ("v0", "v1"))
def test_real_action_conditioned_adapter_keeps_native_action_encoding(version):
    from test_actor_free_td_lewm_v0_v1_rollout_mean import (
        ActorFreeTDLeWMV0,
        ActorFreeTDLeWMV1,
        EmbeddedActionPredictor,
        RawActionPredictor,
        RecordingWorld,
        _actions,
        _info,
    )

    world = RecordingWorld()
    predictor = EmbeddedActionPredictor() if version == "v1" else RawActionPredictor()
    adapter_type = ActorFreeTDLeWMV1 if version == "v1" else ActorFreeTDLeWMV0
    base = adapter_type(world, predictor, gamma=0.95, score_mode="f_only")
    base.eval().requires_grad_(False)
    model = GWeightedPlanningModel(base)
    actions = _actions(5, samples=3)
    info = _info(samples=3)
    expected = base.get_cost(dict(info), actions)
    assert torch.equal(model.get_cost(dict(info), actions), expected)
    assert predictor.seen_state == []
    indices = torch.tensor([[2, 0]])
    scores = model.score_elites(actions, indices, actions[:, [2, 0]])
    assert scores.shape == (1, 2, 5)
    assert predictor.seen_action[-1].shape[-1] == (192 if version == "v1" else 25)
    torch.testing.assert_close(
        predictor.seen_state[-1][..., 0],
        torch.tensor([[[7.0, 11.0, 12.0, 13.0, 14.0], [7.0, 11.0, 12.0, 13.0, 14.0]]]),
    )
    assert world.rollout_horizons == [5, 5]


def test_real_c4_adapter_uses_post_action_states_and_final_action_goes_through_f():
    from test_actor_free_td_lewm_v1_c4_evaluation import (
        RecordingStateOnlyG,
        RecurrentRecordingWorld,
        _actions,
        _adapter,
        _info,
    )

    world = RecurrentRecordingWorld()
    g = RecordingStateOnlyG()
    base = _adapter(world, g, score_mode="f_only").eval().requires_grad_(False)
    model = GWeightedPlanningModel(base)
    actions = _actions(5, samples=2)
    actions[0, 0, :, 0] = 1
    actions[0, 1, :, 0] = 2
    expected = base.get_cost(_info(samples=2), actions)
    assert torch.equal(model.get_cost(_info(samples=2), actions), expected)
    assert g.states == []
    scores = model.score_elites(actions, torch.tensor([[1, 0]]), actions[:, [1, 0]])
    assert scores.shape == (1, 2, 5)
    assert len(g.states) == 1
    torch.testing.assert_close(
        g.states[-1][..., 0],
        torch.tensor(
            [[[2.0, 22.0, 222.0, 2222.0, 22222.0], [1.0, 11.0, 111.0, 1111.0, 11111.0]]]
        ),
    )
    assert not g.state_requires_grad[-1]
    assert all(torch.equal(recorded, actions) for recorded in world.rollout_actions)
