from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm import (
    ActorFreeSuccessorHead,
    actor_free_td_objective,
)
from tdwm.training.actor_free_td_lewm import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    FORMAL_OPTIMIZER_UPDATES,
    GOAL_OBJECTIVE_VERSION,
    METHOD,
    OBJECTIVE_VERSION,
    _build_generator_callback,
    _deployment_payload,
    build_actor_free_td_inputs,
    load_actor_free_td_training_protocol,
    resolve_actor_free_training_schedule,
    validate_actor_free_td_training_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    variant: ROOT
    / "configs"
    / "experiment"
    / f"actor_free_td_lewm_{variant}_cube_train.yaml"
    for variant in (
        "parallel_real",
        "serial_decoupled",
        "serial_coupled",
        "hybrid",
        "goal_hybrid",
    )
}


def test_five_variants_share_data_seed_budget_and_original_lewm_loss():
    protocols = {
        variant: load_actor_free_td_training_protocol(path)
        for variant, path in CONFIGS.items()
    }
    reference = protocols["serial_decoupled"]

    for variant, protocol in protocols.items():
        assert protocol["method"] == METHOD
        assert protocol["variant"] == variant
        assert protocol["seeds"] == [0, 42, 3072]
        assert protocol["dataset"] == reference["dataset"]
        assert protocol["split"] == reference["split"]
        assert protocol["sequence"] == reference["sequence"]
        assert protocol["loader"] == reference["loader"]
        assert protocol["loader"]["validation_workers"] == 8
        assert protocol["optimizer"] == reference["optimizer"]
        assert protocol["scheduler"] == reference["scheduler"]
        assert protocol["training"] == reference["training"]
        assert (
            protocol["training"]["epochs"]
            * protocol["training"]["optimizer_steps_per_epoch"]
            == FORMAL_OPTIMIZER_UPDATES
        )
        assert protocol["loss"]["prediction"] == "mse"
        assert protocol["loss"]["sigreg"]["weight"] == 0.09
        assert protocol["joint_objective"]["goal_conditioning"] == "none"
        assert protocol["joint_objective"]["actor"] == "none"
        assert protocol["joint_objective"]["reward"] == "none"

    assert reference["joint_objective"]["predicted_context_detach"] is True
    assert (
        protocols["serial_coupled"]["joint_objective"]["predicted_context_detach"]
        is False
    )
    assert protocols["hybrid"]["joint_objective"]["real_td_weight"] == 1.0
    goal = protocols["goal_hybrid"]["joint_objective"]
    assert goal["real_td_weight"] == goal["predicted_td_weight"] == 1.0
    assert goal["real_goal_td_weight"] == goal["predicted_goal_td_weight"] == 1.0
    assert goal["goal_enters_successor_head"] is False
    assert protocols["goal_hybrid"]["successor"]["objective_version"] == 2
    assert protocols["serial_coupled"]["joint_objective"]["real_td_weight"] == 0.0
    parallel = protocols["parallel_real"]["joint_objective"]
    assert parallel["real_td_weight"] == 1.0
    assert parallel["predicted_td_weight"] == 0.0
    assert parallel["predicted_context_detach"] is False


def test_variant_protocol_rejects_a_mismatched_gradient_path():
    protocol = load_actor_free_td_training_protocol(CONFIGS["serial_decoupled"])
    protocol["joint_objective"]["predicted_context_detach"] = False

    with pytest.raises(ValueError, match="predicted_context_detach"):
        validate_actor_free_td_training_protocol(protocol)


def test_parallel_real_protocol_rejects_a_predicted_td_branch():
    protocol = load_actor_free_td_training_protocol(CONFIGS["parallel_real"])
    protocol["joint_objective"]["predicted_td_weight"] = 1.0

    with pytest.raises(ValueError, match="predicted_td_weight"):
        validate_actor_free_td_training_protocol(protocol)


def test_goal_hybrid_protocol_rejects_goal_semantic_drift():
    protocol = load_actor_free_td_training_protocol(CONFIGS["goal_hybrid"])
    protocol["joint_objective"]["goal_source"] = "final_frame_only"
    with pytest.raises(ValueError, match="goal_source"):
        validate_actor_free_td_training_protocol(protocol)

    protocol = load_actor_free_td_training_protocol(CONFIGS["goal_hybrid"])
    protocol["joint_objective"]["real_goal_td_weight"] = 0.5
    with pytest.raises(ValueError, match="real_goal_td_weight"):
        validate_actor_free_td_training_protocol(protocol)

    protocol = load_actor_free_td_training_protocol(CONFIGS["goal_hybrid"])
    protocol["successor"]["objective_version"] = 1
    with pytest.raises(ValueError, match="objective_version"):
        validate_actor_free_td_training_protocol(protocol)


def test_teacher_forced_predictions_replace_only_indices_history_and_later():
    real = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    actions = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    # Four start-major local windows.  Only their last tokens are z_3..z_6.
    local = torch.zeros(4, 3, 1)
    local[:, -1, 0] = torch.tensor([30.0, 40.0, 50.0, 60.0])

    inputs = build_actor_free_td_inputs(real, actions, local, history_size=3)

    assert inputs.predicted_context[0, :, 0].tolist() == [
        0.0,
        1.0,
        2.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ]
    assert inputs.one_step_predictions[0, :, 0].tolist() == [
        30.0,
        40.0,
        50.0,
        60.0,
    ]


def test_next_action_nan_sets_terminal_before_actions_are_zeroed():
    real = torch.zeros(1, 7, 2)
    actions = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    actions[:, 5] = torch.nan
    local = torch.zeros(4, 3, 2)

    inputs = build_actor_free_td_inputs(real, actions, local, history_size=3)

    # action[5] is the bootstrap action for current t=4.
    assert inputs.terminals[0].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert torch.isfinite(inputs.actions).all()
    assert inputs.actions[0, 5, 0] == 0.0


def _variant_backward(variant: str):
    torch.manual_seed(11)
    batch, time, embed_dim, action_dim, history = 2, 7, 3, 2, 3
    real = torch.randn(batch, time, embed_dim, requires_grad=True)
    ema_real = torch.randn(batch, time, embed_dim)
    actions = torch.randn(batch, time, action_dim)
    local = torch.randn(
        (time - history) * batch, history, embed_dim, requires_grad=True,
    )
    inputs = build_actor_free_td_inputs(real, actions, local, history_size=history)
    successor = ActorFreeSuccessorHead(
        embed_dim=embed_dim, action_dim=action_dim, history_size=history, hidden_dim=13,
    )
    target_successor = successor.make_target()
    output = actor_free_td_objective(
        successor,
        target_successor,
        real,
        inputs.predicted_context,
        ema_real,
        inputs.actions,
        gamma=0.9,
        variant=variant,
        terminals=inputs.terminals,
        first_current_index=history,
    )
    backward_loss = output.td_loss
    if variant == "goal_hybrid":
        backward_loss = backward_loss + output.goal_td_loss
    backward_loss.backward()
    return output, local, real, successor, target_successor


def test_serial_decoupled_td_does_not_reach_teacher_forced_prediction():
    output, local, real, successor, target_successor = _variant_backward(
        "serial_decoupled"
    )

    assert output.pair_count == 2 * (7 - 3 - 1)
    assert output.real_td_loss is None
    assert local.grad is None
    assert real.grad is None
    assert any(parameter.grad is not None for parameter in successor.parameters())
    assert all(parameter.grad is None for parameter in target_successor.parameters())


def test_parallel_real_td_uses_encoder_latents_without_reaching_prediction():
    output, local, real, successor, target_successor = _variant_backward(
        "parallel_real"
    )

    assert output.real_td_loss is not None
    assert torch.equal(output.predicted_td_loss, torch.zeros(()))
    assert torch.equal(output.td_loss, output.real_td_loss)
    assert local.grad is None
    assert real.grad is not None and torch.count_nonzero(real.grad) > 0
    assert any(parameter.grad is not None for parameter in successor.parameters())
    assert all(parameter.grad is None for parameter in target_successor.parameters())


def test_serial_coupled_td_reaches_teacher_forced_world_prediction():
    output, local, real, _, _ = _variant_backward("serial_coupled")

    assert output.real_td_loss is None
    assert local.grad is not None and torch.count_nonzero(local.grad) > 0
    assert real.grad is not None and torch.count_nonzero(real.grad) > 0


def test_hybrid_adds_real_and_predicted_td_losses():
    output, local, real, _, _ = _variant_backward("hybrid")

    assert output.real_td_loss is not None
    assert torch.allclose(
        output.td_loss, output.predicted_td_loss + output.real_td_loss
    )
    assert local.grad is not None and torch.count_nonzero(local.grad) > 0
    assert real.grad is not None and torch.count_nonzero(real.grad) > 0


def test_goal_hybrid_retains_hybrid_gradients_and_adds_two_goal_losses():
    output, local, real, _, target_successor = _variant_backward("goal_hybrid")

    assert output.real_td_loss is not None
    assert output.real_goal_td_loss is not None
    assert torch.allclose(
        output.td_loss, output.predicted_td_loss + output.real_td_loss
    )
    assert torch.allclose(
        output.goal_td_loss,
        output.predicted_goal_td_loss + output.real_goal_td_loss,
    )
    assert local.grad is not None and torch.count_nonzero(local.grad) > 0
    assert real.grad is not None and torch.count_nonzero(real.grad) > 0
    assert all(parameter.grad is None for parameter in target_successor.parameters())


def test_td_current_index_cannot_start_at_history_minus_one():
    history = 3
    real = torch.randn(1, 7, 2)
    actions = torch.randn(1, 7, 1)
    local = torch.randn(4, history, 2)
    inputs = build_actor_free_td_inputs(real, actions, local, history_size=history)
    successor = ActorFreeSuccessorHead(
        embed_dim=2, action_dim=1, history_size=history, hidden_dim=8
    )

    with pytest.raises(ValueError, match="at least history_size"):
        actor_free_td_objective(
            successor,
            successor.make_target(),
            real,
            inputs.predicted_context,
            real,
            actions,
            gamma=0.9,
            variant="serial_coupled",
            first_current_index=history - 1,
        )


def test_smoke_resume_runs_a_second_epoch_on_one_four_step_schedule():
    protocol = load_actor_free_td_training_protocol(CONFIGS["serial_decoupled"])

    first = resolve_actor_free_training_schedule(
        protocol, smoke=True, resume="never", max_steps=None, train_limit=2,
    )
    resumed = resolve_actor_free_training_schedule(
        protocol, smoke=True, resume="required", max_steps=None, train_limit=2,
    )

    assert first.total_scheduler_steps == resumed.total_scheduler_steps == 4
    assert first.max_epochs == 1
    assert resumed.max_epochs == 2


def test_goal_hybrid_resume_restores_data_and_goal_rng_streams():
    data_generator = torch.Generator().manual_seed(7)
    goal_generator = torch.Generator().manual_seed(8)
    callback = _build_generator_callback(
        data_generator,
        variant="goal_hybrid",
        goal_generator=goal_generator,
    )
    state = callback.state_dict()
    expected_data = torch.rand(5, generator=data_generator)
    expected_goal = torch.rand(5, generator=goal_generator)

    torch.rand(11, generator=data_generator)
    torch.rand(13, generator=goal_generator)
    callback.load_state_dict(state)

    assert torch.equal(torch.rand(5, generator=data_generator), expected_data)
    assert torch.equal(torch.rand(5, generator=goal_generator), expected_goal)

    missing_goal_state = {"generator_state": state["generator_state"]}
    with pytest.raises(RuntimeError, match="goal sampler RNG state"):
        callback.load_state_dict(missing_goal_state)


def test_joint_checkpoint_pairs_online_and_ema_world_and_successor():
    protocol = load_actor_free_td_training_protocol(CONFIGS["serial_coupled"])
    module = SimpleNamespace(
        model=nn.Linear(2, 3),
        target_model=nn.Linear(2, 3),
        successor=nn.Linear(3, 4),
        target_successor=nn.Linear(3, 4),
    )
    model_config = {"_target_": "example.WorldModel"}
    payload = _deployment_payload(
        module,
        protocol=protocol,
        model_config=model_config,
        action_block_dim=25,
        epoch=2,
        global_step=4,
        base_export_run_name="epoch_02",
        base_checkpoint_sha256="0" * 64,
    )

    assert payload["method"] == METHOD
    assert payload["variant"] == "serial_coupled"
    assert payload["objective_version"] == OBJECTIVE_VERSION
    assert payload["deployment_checkpoint_version"] == DEPLOYMENT_CHECKPOINT_VERSION
    assert payload["world_model_config"] is model_config
    assert payload["successor_config"]["action_dim"] == 25
    assert payload["successor_config"]["history_size"] == 3
    assert payload["successor_config"]["predicted_context_detach"] is False
    for key in (
        "world_model_state_dict",
        "target_world_model_state_dict",
        "successor_state_dict",
        "target_successor_state_dict",
    ):
        assert key in payload and payload[key]


def test_goal_hybrid_checkpoint_records_trained_readout_semantics():
    protocol = load_actor_free_td_training_protocol(CONFIGS["goal_hybrid"])
    module = SimpleNamespace(
        model=nn.Linear(2, 3),
        target_model=nn.Linear(2, 3),
        successor=nn.Linear(3, 4),
        target_successor=nn.Linear(3, 4),
    )
    payload = _deployment_payload(
        module,
        protocol=protocol,
        model_config={"_target_": "example.WorldModel"},
        action_block_dim=25,
        epoch=1,
        global_step=2,
        base_export_run_name="epoch_01",
        base_checkpoint_sha256="0" * 64,
    )

    assert payload["objective_version"] == GOAL_OBJECTIVE_VERSION
    config = payload["successor_config"]
    assert config["goal_readout_training"] is True
    assert config["goal_enters_successor_head"] is False
    assert config["goal_readout_branches"] == [
        "real_context",
        "predicted_context",
    ]
    assert config["real_goal_td_weight"] == 1.0
    assert config["predicted_goal_td_weight"] == 1.0
