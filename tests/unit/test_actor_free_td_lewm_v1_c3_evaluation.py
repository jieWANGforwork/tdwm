from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v1_c3 import (
    METHOD,
    STATE_V_SCORE_MODE,
    ActorFreeTDLeWMV1C3,
    assert_constant_shift_preserves_selection,
    load_actor_free_td_lewm_v1_c3_checkpoint,
    validate_actor_free_td_lewm_v1_c3_payload,
)
from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    FORMAL_O50_PLANNING,
    FORMAL_SELECTION_SHA256,
    STATE_V_SCORE_DEFINITION,
    load_actor_free_td_lewm_v1_c3_evaluation_protocol,
    validate_actor_free_td_lewm_v1_c3_checkpoint_protocol,
    validate_actor_free_td_lewm_v1_c3_evaluation_protocol,
)
from tdwm.methods.actor_free_td_lewm_v1 import ActorFreeTDJEPAPredictorV1
from tdwm.methods.actor_free_td_lewm_v1_c3 import RP1StateValueV1C3

CONFIG_PATH = Path(
    "configs/experiment/actor_free_td_lewm_v1_c3_cube_checkpoint_o50.yaml"
)
SCRIPT_PATH = Path("scripts/evaluate_actor_free_td_lewm_v1_c3.py")


class _ActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 192)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.projection(action)


class _RecordingWorld(nn.Module):
    def __init__(self, terminal: torch.Tensor) -> None:
        super().__init__()
        self.action_encoder = _ActionEncoder()
        self.register_buffer("terminal", terminal.clone())
        self.seen_actions: torch.Tensor | None = None
        self.seen_history_size: int | None = None

    def rollout(
        self,
        info: dict[str, torch.Tensor],
        actions: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self.seen_actions = actions.detach().clone()
        self.seen_history_size = history_size
        batch, samples, horizon = actions.shape[:3]
        predicted = actions.new_zeros(batch, samples, 3 + horizon, 192)
        predicted[..., -1, :] = self.terminal.to(actions)
        return {"predicted_emb": predicted}


class _RecordingCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_state: torch.Tensor | None = None
        self.seen_goal: torch.Tensor | None = None

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        self.seen_state = state.detach().clone()
        self.seen_goal = goal.detach().clone()
        return (state[..., 0] - goal[..., 0]).abs()


class _ForbiddenG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("V1-C3 deployment must never call G.")


def _protocol() -> dict:
    return load_actor_free_td_lewm_v1_c3_evaluation_protocol(CONFIG_PATH)


def _predictor_config(protocol: dict) -> dict:
    return {
        "method": "actor_free_td_lewm_v1_c",
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c",
        "implementation_version": "v1",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        **deepcopy(protocol["predictor"]),
        "task_sampling": deepcopy(protocol["task_sampling"]),
        "joint_objective": deepcopy(protocol["joint_objective"]),
        "pretrained_world_model": deepcopy(protocol["pretrained_world_model"]),
    }


def _critic_config(protocol: dict) -> dict:
    return {
        "method": METHOD,
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c3",
        "implementation_version": "v1",
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        **deepcopy(protocol["state_critic"]),
        "goal_sampling": deepcopy(protocol["goal_sampling"]),
        "objective": deepcopy(protocol["objective"]),
    }


def _payload(protocol: dict) -> dict:
    predictor = ActorFreeTDJEPAPredictorV1()
    target_predictor = predictor.make_target()
    critic = RP1StateValueV1C3()
    target_critic = critic.make_target()
    world = _RecordingWorld(torch.zeros(1, 1, 192))
    predictor_config = _predictor_config(protocol)
    world_state = world.state_dict()
    online_g_state = predictor.state_dict()
    target_g_state = target_predictor.state_dict()

    def state_hash(state_dict: dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for key in sorted(state_dict):
            tensor = state_dict[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    parent_hashes = {
        "world_model_state_sha256": state_hash(world_state),
        "online_g_state_sha256": state_hash(online_g_state),
        "target_g_state_sha256": state_hash(target_g_state),
    }
    predictor_config_sha = hashlib.sha256(
        json.dumps(
            predictor_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return {
        "method": METHOD,
        "method_family": "actor_free_td_lewm_v1",
        "variant": "c3",
        "implementation_version": "v1",
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "epoch": 12,
        "logical_epoch": 12,
        "global_step": 12_000,
        "world_model_state_dict": world_state,
        "world_model_config": {"action_encoder": {"input_dim": 25, "emb_dim": 192}},
        "predictor_state_dict": online_g_state,
        "target_predictor_state_dict": target_g_state,
        "predictor_config": predictor_config,
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target_critic.state_dict(),
        "critic_config": _critic_config(protocol),
        "pretrained_world_model_provenance": {
            "source_checkpoint_sha256": protocol["pretrained_world_model"][
                "checkpoint_sha256"
            ]
        },
        "source_v1_c_provenance": deepcopy(protocol["source_v1_c"]),
        "source_v1_c_runtime_provenance": {
            "strategy": "strict_frozen_v1_c_epoch_10_parent",
            "parent_checkpoint_path": "/audited/parent.pt",
            "parent_checkpoint_sha256": protocol["source_v1_c"]["checkpoint_sha256"],
            "parent_method": "actor_free_td_lewm_v1_c",
            "parent_epoch": 10,
            "parent_global_step": 127_960,
            "predictor_config_sha256": predictor_config_sha,
            **parent_hashes,
        },
        "parent_state_hashes": parent_hashes,
    }


def test_state_v_adapter_uses_full_f_terminal_and_only_ema_critic() -> None:
    terminal = torch.zeros(1, 4, 192)
    terminal[0, :, 0] = torch.tensor([4.0, 1.0, 3.0, 2.0])
    # Make latent L2 prefer a different candidate from the critic's first-axis cost.
    terminal[0, 1, 1:] = 100.0
    world = _RecordingWorld(terminal)
    critic = _RecordingCritic()
    forbidden_online_g = _ForbiddenG()
    forbidden_target_g = _ForbiddenG()
    adapter = ActorFreeTDLeWMV1C3(world, critic)
    adapter.parent_predictor = forbidden_online_g
    adapter.parent_target_predictor = forbidden_target_g
    actions = torch.randn(1, 4, 5, 25)
    goal = torch.zeros(1, 192)

    costs = adapter.get_cost(
        {"pixels": torch.zeros(1, 4, 3, 1), "goal_emb": goal}, actions
    )

    assert torch.equal(costs, torch.tensor([[4.0, 1.0, 3.0, 2.0]]))
    assert costs.argmin(dim=1).item() == 1
    assert torch.linalg.vector_norm(terminal, dim=-1).argmin(dim=1).item() != 1
    assert world.seen_actions is not None
    assert torch.equal(world.seen_actions, actions)
    assert world.seen_actions.shape == (1, 4, 5, 25)
    assert world.seen_history_size == 3
    assert critic.seen_state is not None and torch.equal(critic.seen_state, terminal)
    assert critic.seen_goal is not None
    assert torch.equal(critic.seen_goal, goal.unsqueeze(1).expand_as(terminal))
    assert forbidden_online_g.calls == forbidden_target_g.calls == 0
    assert adapter.constant_shift_sanity_checked is True


def test_state_v_adapter_rejects_partial_rollout_and_bad_critic_output() -> None:
    world = _RecordingWorld(torch.zeros(1, 2, 192))
    adapter = ActorFreeTDLeWMV1C3(world, _RecordingCritic())
    info = {"pixels": torch.zeros(1, 2, 3, 1), "goal_emb": torch.zeros(1, 192)}

    with pytest.raises(ValueError, match="full five-block"):
        adapter.get_cost(info, torch.zeros(1, 2, 4, 25))

    class _Negative(nn.Module):
        def forward(self, state, goal):
            return state.new_full(state.shape[:-1], -1.0)

    with pytest.raises(ValueError, match="nonnegative"):
        ActorFreeTDLeWMV1C3(world, _Negative()).get_cost(info, torch.zeros(1, 2, 5, 25))


def test_constant_25_sanity_preserves_exact_ranking_and_selected_action() -> None:
    costs = torch.tensor([[3.0, 0.0, 2.0, 1.0], [1.5, 1.5, 2.0, 0.5]])
    shifted = costs.double() + 25.0

    assert_constant_shift_preserves_selection(costs)
    assert torch.equal(
        torch.argsort(costs.double(), dim=1, stable=True),
        torch.argsort(shifted, dim=1, stable=True),
    )
    assert torch.equal(costs.argmin(dim=1), shifted.argmin(dim=1))


def test_v1_c3_protocol_locks_state_v_only_formal_o50() -> None:
    protocol = _protocol()

    assert protocol["method"] == METHOD
    assert protocol["planning"] == FORMAL_O50_PLANNING
    assert protocol["evaluation"]["selection_sha256"] == FORMAL_SELECTION_SHA256
    assert protocol["inference_objective"]["score_mode"] == STATE_V_SCORE_MODE
    assert protocol["inference_objective"]["score_definition"] == (
        STATE_V_SCORE_DEFINITION
    )
    assert protocol["inference_objective"]["parent_g_used"] is False
    assert protocol["inference_objective"]["terminal_goal_distance_used"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("planning", "horizon"), 4, "planning.horizon"),
        (("planning", "candidates"), 301, "planning.candidates"),
        (("planning", "planning_seed"), 43, "planning.planning_seed"),
        (("evaluation", "selection_sha256"), "f" * 64, "selection_sha256"),
        (("inference_objective", "parent_g_used"), True, "parent_g_used"),
        (
            ("inference_objective", "terminal_goal_distance_used"),
            True,
            "terminal_goal_distance_used",
        ),
    ],
)
def test_v1_c3_protocol_rejects_any_changed_formal_lock(path, value, message) -> None:
    changed = deepcopy(_protocol())
    changed[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_actor_free_td_lewm_v1_c3_evaluation_protocol(changed)


def test_v1_c3_payload_and_checkpoint_protocol_bind_exact_parent() -> None:
    protocol = _protocol()
    payload = _payload(protocol)

    predictor_config, critic_config = validate_actor_free_td_lewm_v1_c3_payload(payload)
    validate_actor_free_td_lewm_v1_c3_checkpoint_protocol(
        payload=payload,
        predictor_config=predictor_config,
        critic_config=critic_config,
        protocol=protocol,
        require_formal_completion=True,
    )

    wrong_parent = deepcopy(payload)
    wrong_parent["source_v1_c_provenance"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_actor_free_td_lewm_v1_c3_payload(wrong_parent)
    wrong_method = deepcopy(payload)
    wrong_method["method"] = "actor_free_td_lewm_v1_c2"
    with pytest.raises(ValueError, match="checkpoint.method"):
        validate_actor_free_td_lewm_v1_c3_payload(wrong_method)


def test_v1_c3_checkpoint_loader_restores_and_freezes_every_module(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    payload = _payload(protocol)
    checkpoint_path = tmp_path / "v1_c3.pt"
    torch.save(payload, checkpoint_path)
    restored_world = _RecordingWorld(torch.zeros(1, 1, 192))

    with patch("hydra.utils.instantiate", return_value=restored_world):
        restored = load_actor_free_td_lewm_v1_c3_checkpoint(checkpoint_path)

    assert restored.payload["method"] == METHOD
    assert restored.target_critic is not restored.critic
    assert restored.target_predictor is not restored.predictor
    for module in (
        restored.world_model,
        restored.predictor,
        restored.target_predictor,
        restored.critic,
        restored.target_critic,
    ):
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())


def test_v1_c3_cli_has_one_state_v_score_and_no_g_or_l2_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("v1_c3_evaluation_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--checkpoint-path",
            "checkpoint.pt",
        ],
    )

    args = module.parse_args()
    assert not hasattr(args, "score_mode")
    assert not hasattr(args, "g_first_weight")
