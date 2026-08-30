from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_v0_c import (
    METHOD_SPEC as C_SPEC,
)
from tdwm.adapters.actor_free_td_lewm_v0_c import (
    load_actor_free_td_lewm_v0_c_checkpoint,
)
from tdwm.adapters.frozen_actor_free_td_v0_common import ActorFreeTDLeWMV0
from tdwm.methods.actor_free_td_lewm_v0 import ActorFreeTDJEPAPredictorV0


class RecordingWorld(nn.Module):
    def __init__(self, predicted: torch.Tensor | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.predicted = predicted
        self.rollout_calls = 0
        self.rollout_history_sizes: list[int | None] = []
        self.rollout_horizons: list[int] = []
        self.encode_calls = 0

    def encode(self, info):
        self.encode_calls += 1
        if "emb" not in info:
            batch = info["pixels"].shape[0]
            info["emb"] = self.anchor.new_zeros(batch, 1, 192)
        return info

    def rollout(self, info, action_sequence, history_size=None):
        del info
        self.rollout_calls += 1
        self.rollout_history_sizes.append(history_size)
        self.rollout_horizons.append(int(action_sequence.shape[-2]))
        if self.predicted is None:
            raise AssertionError("rollout must not be called")
        return {"predicted_emb": self.predicted.to(action_sequence)}


class ActionScorePredictor(nn.Module):
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


def _one_axis_goal(batch: int = 1) -> torch.Tensor:
    goal = torch.zeros(batch, 1, 192)
    goal[..., 0] = 1.0
    return goal


def test_g_only_is_direct_one_block_scoring_without_lewm_rollout():
    world = RecordingWorld()
    predictor = ActionScorePredictor()
    adapter = ActorFreeTDLeWMV0(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=0.95,
        score_mode="g_only",
    )
    actions = torch.zeros(1, 2, 1, 25)
    actions[0, :, 0, 0] = torch.tensor([2.0, -3.0])
    expanded_goal = torch.zeros(1, 2, 2, 192)
    expanded_goal[0, 0, 0, 1] = 1.0
    expanded_goal[0, 0, 1, 0] = 1.0
    expanded_goal[0, 1, :, 1] = 1.0
    info = {
        "emb": torch.zeros(1, 2, 1, 192),
        "goal_emb": expanded_goal,
    }

    cost = adapter.get_cost(info, actions)

    assert world.rollout_calls == 0
    assert predictor.calls == 1
    assert torch.allclose(
        cost,
        -math.sqrt(192.0) * torch.tensor([[2.0, -3.0]]),
    )
    assert predictor.last_action is not None
    assert torch.equal(predictor.last_action, actions[..., 0, :])

    with pytest.raises(ValueError, match="horizon=1"):
        adapter.get_cost(info, actions.expand(-1, -1, 2, -1))
    assert world.rollout_calls == 0


def test_g_only_visual_state_encoding_excludes_live_primitive_action():
    class StateOnlyEncodingWorld(RecordingWorld):
        def encode(self, info):
            if "action" in info:
                raise AssertionError(
                    "V0 state encoding must not invoke LeWM's 25D action encoder."
                )
            return super().encode(info)

    world = StateOnlyEncodingWorld()
    predictor = ActionScorePredictor()
    adapter = ActorFreeTDLeWMV0(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=0.95,
        score_mode="g_only",
    )
    actions = torch.zeros(1, 2, 1, 25)
    actions[0, :, 0, 0] = torch.tensor([2.0, -3.0])
    info = {
        "pixels": torch.zeros(1, 2, 1, 3, 2, 2),
        # Stable World Model exposes the latest primitive Cube action here.
        # It is 5D and must not be confused with G's 25D candidate block.
        "action": torch.zeros(1, 2, 1, 5),
        "goal_emb": _one_axis_goal(),
    }

    cost = adapter.get_cost(info, actions)

    assert world.encode_calls == 1
    assert world.rollout_calls == 0
    assert torch.allclose(
        cost,
        -math.sqrt(192.0) * torch.tensor([[2.0, -3.0]]),
    )
    assert predictor.last_action is not None
    assert torch.equal(predictor.last_action, actions[..., 0, :])


def test_f_only_rolls_full_horizon_but_scores_only_final_state_sum():
    predicted = torch.zeros(1, 1, 4, 192)
    predicted[..., 1:3, :] = 1000.0
    predicted[..., -1, 7] = 2.0
    world = RecordingWorld(predicted)
    predictor = ActionScorePredictor()
    adapter = ActorFreeTDLeWMV0(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=0.5,
        score_mode="f_only",
    )
    actions = torch.zeros(1, 1, 3, 25)
    info = {
        "pixels": torch.zeros(1, 1, 1, 3, 2, 2),
        "goal_emb": _one_axis_goal(),
    }

    cost = adapter.get_cost(info, actions)

    # Original LeWM criterion is final-only summed feature MSE.  The enormous
    # intermediate states must not contribute: (0-1)^2 + (2-0)^2 = 5.
    expected = 5.0
    assert torch.allclose(cost, torch.tensor([[expected]]))
    assert world.rollout_calls == 1
    assert world.rollout_history_sizes == [3]
    assert world.rollout_horizons == [3]
    assert predictor.calls == 0


def test_f_plus_g_splices_first_h_minus_one_states_with_unclamped_negative_q():
    predicted = torch.zeros(1, 1, 3, 192)
    predicted[..., 1, :] = 1000.0
    predicted[..., 2, 7] = 9.0  # State after the second of three actions.
    world = RecordingWorld(predicted)
    predictor = ActionScorePredictor()
    adapter = ActorFreeTDLeWMV0(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=0.5,
        score_mode="f_plus_g",
    )
    actions = torch.zeros(1, 1, 3, 25)
    actions[..., -1, 0] = 40.0
    info = {
        "pixels": torch.zeros(1, 1, 1, 3, 2, 2),
        "goal_emb": _one_axis_goal(),
    }

    cost = adapter.get_cost(info, actions)

    # The F prefix is also final-only summed MSE at z_(H-1).  Its earlier state
    # is ignored, while the prefix terminal contributes 1 + 9**2 = 82.
    explicit = 1.0 + 9.0**2
    expected = explicit - 0.25 * 40.0 * math.sqrt(192.0)
    assert torch.allclose(cost, torch.tensor([[expected]]))
    assert cost.item() < 0.0  # Regression: V0 must not clamp -Q.
    assert world.rollout_history_sizes == [3]
    assert world.rollout_horizons == [2]
    assert predictor.last_state is not None
    assert predictor.last_state[0, 0, 7].item() == 9.0
    assert predictor.last_action is not None
    assert predictor.last_action[0, 0, 0].item() == 40.0


def test_f_plus_g_horizon_one_skips_f_and_scores_from_current_state():
    world = RecordingWorld()
    predictor = ActionScorePredictor()
    adapter = ActorFreeTDLeWMV0(
        world,
        predictor,  # type: ignore[arg-type]
        gamma=0.5,
        score_mode="f_plus_g",
    )
    actions = torch.zeros(1, 2, 1, 25)
    actions[0, :, 0, 0] = torch.tensor([1.5, -2.0])
    current = torch.zeros(1, 2, 1, 192)
    current[..., 11] = 7.0
    info = {"emb": current, "goal_emb": _one_axis_goal()}

    cost = adapter.get_cost(info, actions)

    assert world.rollout_calls == 0
    assert torch.allclose(
        cost,
        -math.sqrt(192.0) * torch.tensor([[1.5, -2.0]]),
    )
    assert predictor.last_state is not None
    assert torch.equal(predictor.last_state, current[..., -1, :])


def _c_predictor_config() -> dict:
    return {
        "method": C_SPEC.method,
        "method_family": "actor_free_td_lewm_v0",
        "variant": C_SPEC.variant,
        "implementation_version": "v0",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "architecture": "td_jepa_forward_map_v0",
        "state_dim": 192,
        "action_dim": 25,
        "task_dim": 192,
        "output_dim": 192,
        "hidden_dim": 256,
        "hidden_layers": 1,
        "embedding_layers": 2,
        "num_parallel": 1,
        "action_processing": "normalized_raw_lewm_action_block",
        "shared_lewm_action_encoder": False,
        "state_parameterization": "symmetric_shared_frozen_lewm_latent",
        "goal_conditioning": "task_input",
        "bootstrap_action": "dataset_next_action",
        "actor": "none",
        "reward": "none",
        "gamma": 0.95,
        "target_ema_decay": 0.995,
        "task_sampling": {
            "goal_probability": 0.5,
            "sampling": "per_transition_bernoulli",
            "random_source": "isotropic_gaussian_sphere",
            "goal_source": "uniform_reachable_future_frozen_latent_same_clip",
            "normalization": "sqrt_dim_l2_sphere",
            "mix_unit": "transition_minibatch",
        },
        "joint_objective": {
            "base_td_population": "all_transitions",
            "random_task_weight": 1.0,
            "goal_subset": "goal_derived_tasks_only",
            "final_weight_normalization": "mean_one_over_all_transitions",
            "weight_gradient": "stop_gradient",
            "candidate_td_targets": "none",
            "goal_projection_weight": 1.0,
        },
        "pretrained_world_model": {"frozen": True},
    }


def test_v0_checkpoint_loader_restores_single_predictor_and_rejects_parallel(
    tmp_path,
):
    world_model = nn.Linear(2, 2)
    predictor = ActorFreeTDJEPAPredictorV0()
    config = _c_predictor_config()
    payload = {
        "method": C_SPEC.method,
        "method_family": "actor_free_td_lewm_v0",
        "variant": C_SPEC.variant,
        "implementation_version": "v0",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "world_model_state_dict": world_model.state_dict(),
        "world_model_config": {
            "_target_": "torch.nn.Linear",
            "in_features": 2,
            "out_features": 2,
        },
        "predictor_state_dict": predictor.state_dict(),
        "target_predictor_state_dict": predictor.make_target().state_dict(),
        "predictor_config": config,
    }
    checkpoint = tmp_path / "v0-c.pt"
    torch.save(payload, checkpoint)

    restored_world, restored_g, restored_config, restored_payload = (
        load_actor_free_td_lewm_v0_c_checkpoint(checkpoint)
    )

    assert restored_config == config
    assert restored_payload["implementation_version"] == "v0"
    assert restored_g(torch.zeros(2, 192), torch.zeros(2, 25), torch.ones(2, 192)).shape == (
        2,
        192,
    )
    assert not any(parameter.requires_grad for parameter in restored_world.parameters())
    assert not any(parameter.requires_grad for parameter in restored_g.parameters())

    missing_target = deepcopy(payload)
    missing_target.pop("target_predictor_state_dict")
    torch.save(missing_target, checkpoint)
    with pytest.raises(ValueError, match="target_predictor_state_dict"):
        load_actor_free_td_lewm_v0_c_checkpoint(checkpoint)

    malformed_target = deepcopy(payload)
    malformed_target["target_predictor_state_dict"] = {
        "not_a_predictor_parameter": torch.zeros(1)
    }
    torch.save(malformed_target, checkpoint)
    with pytest.raises(RuntimeError):
        load_actor_free_td_lewm_v0_c_checkpoint(checkpoint)

    parallel = deepcopy(payload)
    parallel["predictor_config"] = deepcopy(config)
    parallel["predictor_config"]["num_parallel"] = 2
    torch.save(parallel, checkpoint)
    with pytest.raises(ValueError, match="num_parallel"):
        load_actor_free_td_lewm_v0_c_checkpoint(checkpoint)
