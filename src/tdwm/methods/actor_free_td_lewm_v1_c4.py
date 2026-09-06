"""State-only successor primitives for Actor-Free TD-LeWM V1-C4.

C4 compares V1-C's action-conditioned successor with one state-only online
successor evaluated after the logged action.  Both G inputs are frozen-LeWM
post-action ghost states, while the immediate TD feature stays real::

    prediction_i = G(stop_gradient(F(z_i, a_i)), m)
    Y_i = stop_gradient(
        z_(i+1)_real
        + gamma * (1 - d_i) * G_bar(stop_gradient(F(z_(i+1)_real, a_(i+1))), m)
    )
    L_C4 = L_vector(prediction_i, Y_i) + L_goal(prediction_i, Y_i, m)

There is one online branch and no online call on a real state.  Neither the
online nor EMA C4 successor accepts an action argument, and every LeWM output
is detached.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    V1_TASK_DIM,
    encode_frozen_action_blocks_v1,
)
from tdwm.methods.actor_free_td_lewm_v1_objectives import goal_projected_v1_loss

C4_OUTPUT_DIM = V1_STATE_DIM
C4_F_HISTORY_STATES = 3
C4_F_PREVIOUS_ACTIONS = 2


def _validate_floating_vector(
    name: str,
    value: torch.Tensor,
    *,
    final_dim: int,
    minimum_ndim: int = 2,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if value.ndim < minimum_ndim:
        raise ValueError(f"{name} must have at least {minimum_ndim} dimensions.")
    if value.shape[-1] != final_dim:
        raise ValueError(f"{name} must end with dimension {final_dim}.")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _require_same_context(
    reference_name: str,
    reference: torch.Tensor,
    other_name: str,
    other: torch.Tensor,
) -> None:
    if other.shape[:-1] != reference.shape[:-1]:
        raise ValueError(
            f"{other_name} must match the leading axes of {reference_name}."
        )
    if other.device != reference.device or other.dtype != reference.dtype:
        raise ValueError(
            f"{other_name} must share the device and dtype of {reference_name}."
        )


def _simple_embedding(
    input_dim: int,
    hidden_dim: int,
    embedding_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
    ]
    for _ in range(embedding_layers - 2):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
    layers.extend((nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()))
    return nn.Sequential(*layers)


class ActorFreeTDJEPAPredictorV1C4(nn.Module):
    """One state/task-only successor with no action-shaped API or parameters."""

    state_dim = V1_STATE_DIM
    task_dim = V1_TASK_DIM
    output_dim = C4_OUTPUT_DIM

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        embedding_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer.")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative.")
        if embedding_layers < 2:
            raise ValueError("embedding_layers must be at least two.")
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.embedding_layers = int(embedding_layers)

        # Match V1-C's two embedding-path capacity while replacing its
        # state/action path with a strictly state-only path.
        self.embed_state_task = _simple_embedding(
            V1_STATE_DIM + V1_TASK_DIM,
            self.hidden_dim,
            self.embedding_layers,
        )
        self.embed_state = _simple_embedding(
            V1_STATE_DIM,
            self.hidden_dim,
            self.embedding_layers,
        )
        output_layers: list[nn.Module] = []
        for _ in range(self.hidden_layers):
            output_layers.extend(
                (nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
            )
        output_layers.append(nn.Linear(self.hidden_dim, C4_OUTPUT_DIM))
        self.output = nn.Sequential(*output_layers)
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        _validate_floating_vector("state", state, final_dim=self.state_dim)
        _validate_floating_vector("task", task, final_dim=self.task_dim)
        _require_same_context("state", state, "task", task)
        state_task = self.embed_state_task(torch.cat((state, task), dim=-1))
        state_only = self.embed_state(state)
        prediction = self.output(torch.cat((state_only, state_task), dim=-1))
        if prediction.shape != (*state.shape[:-1], self.output_dim):
            raise RuntimeError("C4 predictor returned an unexpected shape.")
        if not bool(torch.isfinite(prediction.detach()).all()):
            raise FloatingPointError("C4 predictor produced a non-finite output.")
        return prediction

    def make_target(self) -> "ActorFreeTDJEPAPredictorV1C4":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


def _validate_predictor_pair(
    online: ActorFreeTDJEPAPredictorV1C4,
    target: ActorFreeTDJEPAPredictorV1C4,
) -> None:
    if not isinstance(online, ActorFreeTDJEPAPredictorV1C4):
        raise TypeError("online must be ActorFreeTDJEPAPredictorV1C4.")
    if not isinstance(target, ActorFreeTDJEPAPredictorV1C4):
        raise TypeError("target must be ActorFreeTDJEPAPredictorV1C4.")
    if online is target:
        raise ValueError("online and target must be distinct C4 predictors.")
    for name in ("hidden_dim", "hidden_layers", "embedding_layers"):
        if getattr(online, name) != getattr(target, name):
            raise ValueError("online and target C4 predictors must share an architecture.")
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("the C4 target predictor must be frozen.")


def _normalize_terminal(
    terminal: torch.Tensor | bool,
    *,
    leading_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(terminal, bool):
        return torch.full(leading_shape, terminal, device=device, dtype=torch.bool)
    if not isinstance(terminal, torch.Tensor):
        raise TypeError("terminal must be a torch.Tensor or bool.")
    if terminal.shape != leading_shape:
        raise ValueError(f"terminal must have shape {tuple(leading_shape)}.")
    on_device = terminal.to(device=device)
    as_bool = on_device.to(dtype=torch.bool)
    if bool((on_device != as_bool).any()):
        raise ValueError("terminal must contain only binary values.")
    return as_bool


def _validate_frozen_world_model(world_model: Any) -> nn.Module:
    if not isinstance(world_model, nn.Module):
        raise TypeError("world_model must be a torch.nn.Module.")
    if any(parameter.requires_grad for parameter in world_model.parameters()):
        raise ValueError("C4 requires every frozen LeWM parameter to be frozen.")
    if any(module.training for module in world_model.modules()):
        raise ValueError("C4 requires every frozen LeWM module to remain in eval mode.")
    action_encoder = getattr(world_model, "action_encoder", None)
    if not isinstance(action_encoder, nn.Module):
        raise ValueError("C4 requires world_model.action_encoder.")
    return action_encoder


def predict_frozen_lewm_ghost_next_state_v1_c4(
    world_model: Any,
    state_history: torch.Tensor,
    previous_raw_actions: torch.Tensor,
    current_raw_action: torch.Tensor,
) -> torch.Tensor:
    """Predict the detached ghost ``z_(i+1)`` from history ending at ``z_i``.

    ``state_history`` is ``[z_(i-2), z_(i-1), z_i]`` and
    ``previous_raw_actions`` contains the first two corresponding macro-action
    blocks.  ``current_raw_action`` is ``a_i``.  LeWM predicts a shifted
    three-state sequence; its final element is therefore the ghost next state.
    """

    action_encoder = _validate_frozen_world_model(world_model)
    _validate_floating_vector(
        "state_history", state_history, final_dim=V1_STATE_DIM, minimum_ndim=3
    )
    if state_history.ndim != 3 or state_history.shape[1] != C4_F_HISTORY_STATES:
        raise ValueError("state_history must have shape [batch, 3, 192].")
    _validate_floating_vector(
        "previous_raw_actions",
        previous_raw_actions,
        final_dim=V1_RAW_ACTION_DIM,
        minimum_ndim=3,
    )
    if (
        previous_raw_actions.ndim != 3
        or previous_raw_actions.shape[1] != C4_F_PREVIOUS_ACTIONS
    ):
        raise ValueError("previous_raw_actions must have shape [batch, 2, 25].")
    _validate_floating_vector(
        "current_raw_action",
        current_raw_action,
        final_dim=V1_RAW_ACTION_DIM,
    )
    batch = state_history.shape[0]
    if previous_raw_actions.shape[0] != batch or current_raw_action.shape != (
        batch,
        V1_RAW_ACTION_DIM,
    ):
        raise ValueError("C4 ghost-state inputs must share one batch axis.")
    for name, value in (
        ("previous_raw_actions", previous_raw_actions),
        ("current_raw_action", current_raw_action),
    ):
        if value.device != state_history.device or value.dtype != state_history.dtype:
            raise ValueError(f"{name} must match state_history device and dtype.")

    raw_actions = torch.cat(
        (previous_raw_actions.detach(), current_raw_action.detach().unsqueeze(1)),
        dim=1,
    )
    action_embeddings = encode_frozen_action_blocks_v1(
        action_encoder,
        raw_actions,
        reference=state_history,
    )
    with torch.no_grad():
        predicted_sequence = world_model.predict(
            state_history.detach(), action_embeddings.detach()
        )
    if not isinstance(predicted_sequence, torch.Tensor) or predicted_sequence.shape != (
        batch,
        C4_F_HISTORY_STATES,
        V1_STATE_DIM,
    ):
        raise ValueError(
            "frozen LeWM predict must return a shifted [batch, 3, 192] sequence."
        )
    if not bool(torch.isfinite(predicted_sequence).all()):
        raise FloatingPointError("frozen LeWM produced a non-finite C4 state.")
    # Lightning's bf16 autocast may make the frozen LeWM prediction bfloat16
    # even though the immutable latent store is float32.  Restore the ghost to
    # that latent space before the EMA bootstrap validates and consumes it.
    return predicted_sequence[:, -1, :].to(
        device=state_history.device,
        dtype=state_history.dtype,
    ).detach()


def successor_td_target_v1_c4(
    target: ActorFreeTDJEPAPredictorV1C4,
    immediate_next_state: torch.Tensor,
    ghost_next_next_state: torch.Tensor,
    task: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> torch.Tensor:
    """Build detached ``z_(i+1)^real + gamma*(1-d_i)*G_bar(z_(i+2)^F,m)``."""

    if not isinstance(target, ActorFreeTDJEPAPredictorV1C4):
        raise TypeError("target must be ActorFreeTDJEPAPredictorV1C4.")
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("the C4 target predictor must be frozen.")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must be finite and lie in [0, 1].")
    _validate_floating_vector(
        "immediate_next_state", immediate_next_state, final_dim=V1_STATE_DIM
    )
    _validate_floating_vector(
        "ghost_next_next_state", ghost_next_next_state, final_dim=V1_STATE_DIM
    )
    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    for name, value in (
        ("ghost_next_next_state", ghost_next_next_state),
        ("task", task),
    ):
        _require_same_context(
            "immediate_next_state", immediate_next_state, name, value
        )
    terminal_bool = _normalize_terminal(
        terminal,
        leading_shape=immediate_next_state.shape[:-1],
        device=immediate_next_state.device,
    )
    with torch.no_grad():
        flat_immediate = immediate_next_state.detach().float().reshape(
            -1, V1_STATE_DIM
        )
        flat_ghost = ghost_next_next_state.detach().reshape(-1, V1_STATE_DIM)
        flat_task = task.detach().reshape(-1, V1_TASK_DIM)
        flat_terminal = terminal_bool.reshape(-1)
        result = flat_immediate.clone()
        continuation_indices = torch.nonzero(
            ~flat_terminal, as_tuple=False
        ).flatten()
        if continuation_indices.numel():
            bootstrap = target(
                flat_ghost.index_select(0, continuation_indices),
                flat_task.index_select(0, continuation_indices),
            ).float()
            continued = flat_immediate.index_select(0, continuation_indices)
            continued = continued + gamma_value * bootstrap
            result.index_copy_(0, continuation_indices, continued)
    return result.reshape_as(immediate_next_state).detach()


@dataclass(frozen=True)
class C4TDOutput:
    target: torch.Tensor
    prediction: torch.Tensor
    per_transition_vector_loss: torch.Tensor
    vector_loss: torch.Tensor
    prediction_score: torch.Tensor
    target_score: torch.Tensor
    score_residual: torch.Tensor
    goal_loss: torch.Tensor
    total_loss: torch.Tensor
    goal_indices: torch.Tensor
    terminal: torch.Tensor


def build_td_loss_v1_c4(
    online: ActorFreeTDJEPAPredictorV1C4,
    target: ActorFreeTDJEPAPredictorV1C4,
    online_ghost_next_state: torch.Tensor,
    immediate_next_state: torch.Tensor,
    target_ghost_next_next_state: torch.Tensor,
    task: torch.Tensor,
    goal_mask: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
    goal_projection_weight: float = 1.0,
) -> C4TDOutput:
    """Compute C4's one post-action online branch and ghost EMA bootstrap."""

    _validate_predictor_pair(online, target)
    _validate_floating_vector(
        "online_ghost_next_state", online_ghost_next_state, final_dim=V1_STATE_DIM
    )
    _validate_floating_vector(
        "immediate_next_state", immediate_next_state, final_dim=V1_STATE_DIM
    )
    _validate_floating_vector(
        "target_ghost_next_next_state",
        target_ghost_next_next_state,
        final_dim=V1_STATE_DIM,
    )
    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    for name, value in (
        ("immediate_next_state", immediate_next_state),
        ("target_ghost_next_next_state", target_ghost_next_next_state),
        ("task", task),
    ):
        _require_same_context(
            "online_ghost_next_state", online_ghost_next_state, name, value
        )
    if online_ghost_next_state.ndim != 2:
        raise ValueError("C4 training states must have shape [batch, 192].")
    if not isinstance(goal_mask, torch.Tensor) or goal_mask.dtype != torch.bool:
        raise TypeError("goal_mask must be a boolean torch.Tensor.")
    if (
        goal_mask.shape != online_ghost_next_state.shape[:-1]
        or goal_mask.device != online_ghost_next_state.device
    ):
        raise ValueError("goal_mask must be an aligned [batch] tensor.")
    coefficient = float(goal_projection_weight)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("goal_projection_weight must be finite and non-negative.")
    terminal_bool = _normalize_terminal(
        terminal,
        leading_shape=online_ghost_next_state.shape[:-1],
        device=online_ghost_next_state.device,
    )
    frozen_online_ghost = online_ghost_next_state.detach()
    frozen_immediate = immediate_next_state.detach()
    frozen_target_ghost = target_ghost_next_next_state.detach()
    frozen_task = task.detach()
    td_target = successor_td_target_v1_c4(
        target,
        frozen_immediate,
        frozen_target_ghost,
        frozen_task,
        gamma=gamma,
        terminal=terminal_bool,
    )
    prediction = online(frozen_online_ghost, frozen_task)
    per_transition = (
        prediction.float() - td_target.float()
    ).square().sum(dim=-1)
    objective = goal_projected_v1_loss(
        prediction,
        td_target,
        frozen_task,
        goal_mask,
        per_transition,
        projection_coefficient=coefficient,
    )
    if not bool(torch.isfinite(objective.loss.detach())):
        raise FloatingPointError("C4 TD loss became non-finite.")
    return C4TDOutput(
        target=td_target,
        prediction=prediction,
        per_transition_vector_loss=objective.per_transition_td_loss,
        vector_loss=objective.base_td_loss,
        prediction_score=objective.prediction_score,
        target_score=objective.target_score,
        score_residual=objective.score_residual,
        goal_loss=objective.projection_loss,
        total_loss=objective.loss,
        goal_indices=objective.goal_indices,
        terminal=terminal_bool,
    )


@torch.no_grad()
def ema_update_target_v1_c4(
    target: ActorFreeTDJEPAPredictorV1C4,
    online: ActorFreeTDJEPAPredictorV1C4,
    *,
    decay: float,
) -> None:
    _validate_predictor_pair(online, target)
    decay_value = float(decay)
    if not math.isfinite(decay_value) or not 0.0 <= decay_value <= 1.0:
        raise ValueError("decay must be finite and lie in [0, 1].")
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.mul_(decay_value).add_(
            online_parameter, alpha=1.0 - decay_value
        )
    for target_buffer, online_buffer in zip(
        target.buffers(), online.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(decay_value).add_(
                online_buffer, alpha=1.0 - decay_value
            )
        else:
            target_buffer.copy_(online_buffer)
    target.requires_grad_(False).eval()


__all__ = [
    "ActorFreeTDJEPAPredictorV1C4",
    "C4TDOutput",
    "C4_F_HISTORY_STATES",
    "C4_F_PREVIOUS_ACTIONS",
    "C4_OUTPUT_DIM",
    "build_td_loss_v1_c4",
    "ema_update_target_v1_c4",
    "predict_frozen_lewm_ghost_next_state_v1_c4",
    "successor_td_target_v1_c4",
]
