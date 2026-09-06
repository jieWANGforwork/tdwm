from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v1_c4 import (
    FIRST_ACTION_SCORE_MODES,
    FORMAL_HORIZON_BY_SCORE_MODE,
    SCORE_MODES,
    ActorFreeTDLeWMV1C4,
    load_actor_free_td_lewm_v1_c4_checkpoint,
    validate_actor_free_td_lewm_v1_c4_payload,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c4 import (
    FORMAL_SELECTION_SHA256_BY_PROTOCOL,
    METHOD_SPEC,
    configure_actor_free_td_lewm_v1_c4_evaluation_mode,
    evaluate_actor_free_td_lewm_v1_c4,
    load_actor_free_td_lewm_v1_c4_evaluation_protocol,
    validate_actor_free_td_lewm_v1_c4_checkpoint_protocol,
    validate_actor_free_td_lewm_v1_c4_evaluation_protocol,
)
from tdwm.methods.actor_free_td_lewm_v1_c4 import ActorFreeTDJEPAPredictorV1C4
from tdwm.training.actor_free_td_lewm_v1_c4 import (
    _deployment_payload,
    _state_dict_sha256,
    load_actor_free_td_lewm_v1_c4_training_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    label: ROOT
    / "configs"
    / "experiment"
    / f"actor_free_td_lewm_v1_c4_cube_checkpoint_{label}.yaml"
    for label in ("o25", "o50", "o100")
}


class FrozenActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return action.new_zeros(*action.shape[:-1], 192) + self.anchor


class RecordingWorld(nn.Module):
    def __init__(self, fixed_future: torch.Tensor | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.action_encoder = FrozenActionEncoder()
        self.fixed_future = fixed_future
        self.rollout_actions: list[torch.Tensor] = []
        self.rollout_history_sizes: list[int | None] = []
        self.eval()

    def encode(self, info):
        return info

    def predict(self, emb, act_emb):
        return emb + 0.0 * act_emb[..., : emb.shape[-1]]

    def rollout(self, info, actions, history_size=None):
        self.rollout_actions.append(actions.detach().clone())
        self.rollout_history_sizes.append(history_size)
        batch, samples, horizon = actions.shape[:3]
        observed = int(info["emb"].shape[-2])
        predicted = actions.new_zeros(batch, samples, observed + horizon, 192)
        predicted[..., :observed, :] = info["emb"].to(actions)
        if self.fixed_future is None:
            predicted[..., observed:, 0] = actions[..., 0]
        else:
            predicted[..., observed:, :] = self.fixed_future.to(actions)
        return {"predicted_emb": predicted + self.anchor}


class RecordingStateOnlyG(nn.Module):
    state_dim = 192
    task_dim = 192
    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.states: list[torch.Tensor] = []

    def forward(self, state, task):
        del task
        self.states.append(state.detach().clone())
        output = torch.zeros_like(state) + self.anchor
        output[..., 0] = state[..., 0]
        return output


def _info(samples: int = 1) -> dict[str, torch.Tensor]:
    state = torch.zeros(1, samples, 1, 192)
    goal = torch.zeros(1, 1, 192)
    goal[..., 0] = 1.0
    return {"emb": state, "goal_emb": goal}


def _actions(horizon: int, samples: int = 1) -> torch.Tensor:
    return torch.zeros(1, samples, horizon, 25)


def _adapter(
    world: RecordingWorld,
    predictor: RecordingStateOnlyG,
    *,
    score_mode: str,
    gamma: float = 0.5,
) -> ActorFreeTDLeWMV1C4:
    return ActorFreeTDLeWMV1C4(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=gamma,
        score_mode=score_mode,
        g_first_weight=(0.25 if score_mode in FIRST_ACTION_SCORE_MODES else None),
    )


def test_c4_only_routes_action_through_f_and_g_receives_only_zhat1() -> None:
    world = RecordingWorld()
    predictor = RecordingStateOnlyG()
    adapter = _adapter(world, predictor, score_mode="g_only")
    actions = _actions(1, samples=2)
    actions[0, :, 0, 0] = torch.tensor([2.0, -3.0])

    cost = adapter.get_cost(_info(samples=2), actions)

    assert torch.equal(world.rollout_actions[0], actions)
    assert world.rollout_history_sizes == [3]
    assert len(predictor.states) == 1
    assert predictor.states[0].shape == (1, 2, 192)
    assert torch.equal(predictor.states[0][..., 0], torch.tensor([[2.0, -3.0]]))
    torch.testing.assert_close(
        cost, -math.sqrt(192.0) * torch.tensor([[2.0, -3.0]])
    )


def test_f_plus_c4_uses_full_rollout_z4_cost_and_zhat5_state_only_tail() -> None:
    future = torch.zeros(1, 1, 5, 192)
    future[..., 3, 7] = 9.0
    future[..., 4, 0] = 4.0
    world = RecordingWorld(future)
    predictor = RecordingStateOnlyG()
    adapter = _adapter(world, predictor, score_mode="f_plus_g")
    actions = _actions(5)
    actions[..., -1, 0] = 123.0

    cost = adapter.get_cost(_info(), actions)

    assert torch.equal(world.rollout_actions[0], actions)
    assert world.rollout_actions[0].shape[-2] == 5
    assert len(predictor.states) == 1
    assert torch.equal(predictor.states[0], future[..., 4, :])
    expected = 1.0 + 9.0**2 - (0.5**4) * 4.0 * math.sqrt(192.0)
    torch.testing.assert_close(cost, torch.tensor([[expected]]))


@pytest.mark.parametrize("score_mode", ("f_plus_g_first", "f_plus_g_first_q2"))
def test_c4_first_q_uses_exact_raw_or_zscore_formula_after_full_f_rollout(
    score_mode: str,
) -> None:
    future = torch.zeros(1, 3, 5, 192)
    future[0, :, 0, 0] = torch.tensor([1.0, 2.0, 10.0])
    future[0, :, 4, 7] = torch.tensor([0.0, 2.0, 4.0])
    world = RecordingWorld(future)
    predictor = RecordingStateOnlyG()
    adapter = _adapter(world, predictor, score_mode=score_mode)
    actions = _actions(5, samples=3)

    cost = adapter.get_cost(_info(samples=3), actions)

    assert torch.equal(world.rollout_actions[0], actions)
    assert torch.equal(predictor.states[0], future[..., 0, :])
    f_cost = torch.tensor([[1.0, 5.0, 17.0]])
    q_first = math.sqrt(192.0) * torch.tensor([[1.0, 2.0, 10.0]])
    if score_mode == "f_plus_g_first_q2":
        f_cost = (f_cost - f_cost.mean(dim=1, keepdim=True)) / f_cost.std(
            dim=1,
            correction=0,
            keepdim=True,
        )
        q_first = (q_first - q_first.mean(dim=1, keepdim=True)) / q_first.std(
            dim=1,
            correction=0,
            keepdim=True,
        )
    torch.testing.assert_close(cost, f_cost - 0.25 * q_first)


def test_c4_mean_q_reads_all_five_f_successor_states() -> None:
    future = torch.zeros(1, 1, 5, 192)
    future[..., 0] = torch.arange(1.0, 6.0)
    world = RecordingWorld(future)
    predictor = RecordingStateOnlyG()
    adapter = _adapter(world, predictor, score_mode="g_only_f_rollout_mean")

    cost = adapter.get_cost(_info(), _actions(5))

    assert torch.equal(predictor.states[0], future)
    expected = -3.0 * math.sqrt(192.0)
    torch.testing.assert_close(cost, torch.tensor([[expected]]))


def test_c4_f_only_is_unchanged_full_f_rollout_and_never_calls_g() -> None:
    future = torch.zeros(1, 1, 5, 192)
    future[..., -1, 7] = 2.0
    world = RecordingWorld(future)
    predictor = RecordingStateOnlyG()
    adapter = _adapter(world, predictor, score_mode="f_only")
    actions = _actions(5)

    cost = adapter.get_cost(_info(), actions)

    assert torch.equal(world.rollout_actions[0], actions)
    assert predictor.states == []
    assert torch.equal(cost, torch.tensor([[5.0]]))


def test_changing_action_can_affect_c4_only_through_changed_f_output() -> None:
    world = RecordingWorld()
    predictor = RecordingStateOnlyG()
    adapter = _adapter(world, predictor, score_mode="g_only")
    low = _actions(1)
    high = _actions(1)
    high[..., 0] = 3.0

    low_cost = adapter.get_cost(_info(), low)
    high_cost = adapter.get_cost(_info(), high)

    assert not torch.equal(low_cost, high_cost)
    assert [state.shape[-1] for state in predictor.states] == [192, 192]
    assert tuple(type(predictor).forward.__code__.co_varnames[:3]) == (
        "self",
        "state",
        "task",
    )


@pytest.mark.parametrize("protocol_label", ("o25", "o50", "o100"))
@pytest.mark.parametrize("score_mode", tuple(sorted(SCORE_MODES)))
def test_c4_protocols_lock_all_six_f_through_state_only_g_scores(
    protocol_label: str,
    score_mode: str,
) -> None:
    formal = load_actor_free_td_lewm_v1_c4_evaluation_protocol(
        CONFIGS[protocol_label]
    )
    configured = configure_actor_free_td_lewm_v1_c4_evaluation_mode(
        formal,
        smoke=False,
        pilot=False,
        score_mode=score_mode,
        g_first_weight=(0.25 if score_mode in FIRST_ACTION_SCORE_MODES else None),
    )

    validate_actor_free_td_lewm_v1_c4_evaluation_protocol(configured)
    assert configured["planning"]["horizon"] == FORMAL_HORIZON_BY_SCORE_MODE[
        score_mode
    ]
    inference = configured["inference_objective"]
    assert inference["action_enters_g"] is False
    assert inference["action_effect"] == "only_via_f_predicted_state"
    assert inference["score_definition"]["action_enters_g"] is False
    if score_mode == "f_plus_g":
        assert inference["score_definition"]["final_action_path"] == (
            "A5_to_frozen_f_to_zhat5_to_state_only_g"
        )
    if score_mode in FIRST_ACTION_SCORE_MODES:
        assert inference["score_definition"]["q_first_state"].endswith("zhat1")
    if score_mode == "g_only_f_rollout_mean":
        assert inference["state_source_for_q1"].endswith("zhat1")
        assert inference["state_source_for_q2_to_q5"].endswith("zhat2_to_zhat5")

    assert len(FORMAL_SELECTION_SHA256_BY_PROTOCOL[protocol_label]) == 64


def _g_config() -> dict:
    return {
        "method": "actor_free_td_lewm_v1_c4",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c4",
        "implementation_version": "v1",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "architecture": "td_jepa_state_only_forward_map_v1_c4",
        "state_dim": 192,
        "task_dim": 192,
        "output_dim": 192,
        "hidden_dim": 256,
        "hidden_layers": 1,
        "embedding_layers": 2,
        "num_parallel": 1,
        "action_input": "none",
        "action_effect": "only_via_f_predicted_state",
        "goal_conditioning": "task_input",
        "successor_semantics": "includes_current_input_state",
        "actor": "none",
        "reward": "none",
        "gamma": 0.95,
        "target_ema_decay": 0.995,
        "task_sampling": {},
        "joint_objective": {"goal_projection_weight": 1.0},
        "time_alignment": {},
        "pretrained_world_model": {"frozen": True},
    }


def test_c4_checkpoint_roundtrip_restores_state_only_online_g(tmp_path: Path) -> None:
    world = RecordingWorld()
    online = ActorFreeTDJEPAPredictorV1C4()
    payload = {
        "method": "actor_free_td_lewm_v1_c4",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c4",
        "implementation_version": "v1",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "world_model_state_dict": world.state_dict(),
        "world_model_config": {"_target_": "tests.RecordingWorld"},
        "online_g_state_dict": online.state_dict(),
        "target_g_state_dict": online.make_target().state_dict(),
        "g_config": _g_config(),
        "pretrained_world_model_provenance": {
            "source_checkpoint_sha256": "a" * 64
        },
    }
    checkpoint = tmp_path / "c4.pt"
    torch.save(payload, checkpoint)

    with patch("hydra.utils.instantiate", return_value=RecordingWorld()):
        restored_world, restored_g, config, restored = (
            load_actor_free_td_lewm_v1_c4_checkpoint(checkpoint)
        )

    assert isinstance(restored_world, RecordingWorld)
    assert isinstance(restored_g, ActorFreeTDJEPAPredictorV1C4)
    assert config["action_input"] == "none"
    assert restored["variant"] == "c4"
    output = restored_g(torch.zeros(2, 192), torch.ones(2, 192))
    assert output.shape == (2, 192)
    with pytest.raises(TypeError):
        restored_g(torch.zeros(2, 192), torch.zeros(2, 25), torch.ones(2, 192))


def test_real_training_payload_contract_passes_formal_eval_validator() -> None:
    training = load_actor_free_td_lewm_v1_c4_training_protocol(
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v1_c4_cube_train.yaml"
    )

    class Module:
        model = RecordingWorld()
        online_g = ActorFreeTDJEPAPredictorV1C4()
        target_g = online_g.make_target()

    source_sha = training["pretrained_world_model"]["checkpoint_sha256"]
    module = Module()
    payload = _deployment_payload(
        module,
        protocol=training,
        model_config={"_target_": "tests.RecordingWorld"},
        initialization_info={"source_checkpoint_sha256": source_sha},
        frozen_world_model_sha256=_state_dict_sha256(module.model.state_dict()),
        epoch=10,
        global_step=127_960,
    )
    evaluation = load_actor_free_td_lewm_v1_c4_evaluation_protocol(CONFIGS["o50"])

    assert validate_actor_free_td_lewm_v1_c4_payload(payload) == payload["g_config"]
    validate_actor_free_td_lewm_v1_c4_checkpoint_protocol(
        payload=payload,
        predictor_config=payload["g_config"],
        protocol=evaluation,
        spec=METHOD_SPEC,
        require_formal_completion=True,
    )


def test_evaluation_manifest_uses_g_config_not_old_predictor_config(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "results.json"
    manifest_path = tmp_path / "protocol_manifest.json"
    result_path.write_text(json.dumps({"score_mode": "f_only"}))
    manifest_path.write_text(
        json.dumps({"checkpoint": {"predictor_config": {"action_input": "none"}}})
    )
    with patch(
        "tdwm.evaluation.actor_free_td_lewm_v1_c4."
        "evaluate_actor_free_td_predictor_runtime",
        return_value={"score_mode": "f_only"},
    ):
        result = evaluate_actor_free_td_lewm_v1_c4(output_dir=tmp_path)

    stored_manifest = json.loads(manifest_path.read_text())
    assert "predictor_config" not in stored_manifest["checkpoint"]
    assert stored_manifest["checkpoint"]["g_config"]["action_input"] == "none"
    assert result["state_only_g"] is True
    assert result["action_enters_g"] is False
