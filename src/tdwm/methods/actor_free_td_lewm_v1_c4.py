"""State-only successor primitives for Actor-Free TD-LeWM V1-C4.

C4 compares the action-conditioned V1-C successor with a successor that sees
actions only through the frozen LeWM transition model.  At replay anchor
``r = i - 1`` the two online inputs are aligned to the same macro time ``i``::

    x_i_real = z_i
    x_i_pred = stop_gradient(F(z_{i-1}, a_{i-1}))

Both branches share the real, current-state-including TD target::

    Y_i = stop_gradient(z_i + gamma * (1 - d_i) * G_bar(z_{i+1}, m)).

Neither the online nor EMA C4 successor accepts an action argument.
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

        # Match V1-C's two-branch capacity while replacing its state/action
        # branch with a strictly state-only branch.
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


def predict_frozen_lewm_aligned_state_v1_c4(
    world_model: Any,
    state_history: torch.Tensor,
    previous_raw_actions: torch.Tensor,
    predecessor_raw_action: torch.Tensor,
) -> torch.Tensor:
    """Predict ``z_i`` from the frozen LeWM history ending at ``i-1``.

    ``state_history`` is ``[z_(i-3), z_(i-2), z_(i-1)]`` and
    ``previous_raw_actions`` contains the first two corresponding macro-action
    blocks.  ``predecessor_raw_action`` is ``a_(i-1)``.  LeWM predicts a shifted
    three-state sequence; its final element is therefore the prediction at the
    same time as the C4 real input ``z_i``.
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
        "predecessor_raw_action",
        predecessor_raw_action,
        final_dim=V1_RAW_ACTION_DIM,
    )
    batch = state_history.shape[0]
    if previous_raw_actions.shape[0] != batch or predecessor_raw_action.shape != (
        batch,
        V1_RAW_ACTION_DIM,
    ):
        raise ValueError("C4 frozen-F histories must share one batch axis.")
    for name, value in (
        ("previous_raw_actions", previous_raw_actions),
        ("predecessor_raw_action", predecessor_raw_action),
    ):
        if value.device != state_history.device or value.dtype != state_history.dtype:
            raise ValueError(f"{name} must match state_history device and dtype.")

    raw_actions = torch.cat(
        (previous_raw_actions.detach(), predecessor_raw_action.detach().unsqueeze(1)),
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
    # even though the immutable latent store is float32.  C4 treats both
    # online branches as two views of the same latent space, so restore the
    # frozen prediction to the cache tensor's exact device/dtype before the
    # shared loss validates and consumes it.
    return predicted_sequence[:, -1, :].to(
        device=state_history.device,
        dtype=state_history.dtype,
    ).detach()


def successor_td_target_v1_c4(
    target: ActorFreeTDJEPAPredictorV1C4,
    current_state: torch.Tensor,
    next_state: torch.Tensor,
    task: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> torch.Tensor:
    """Build detached ``z_i + gamma*(1-d_i)*G_bar(z_(i+1),m)``."""

    if not isinstance(target, ActorFreeTDJEPAPredictorV1C4):
        raise TypeError("target must be ActorFreeTDJEPAPredictorV1C4.")
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("the C4 target predictor must be frozen.")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must be finite and lie in [0, 1].")
    _validate_floating_vector("current_state", current_state, final_dim=V1_STATE_DIM)
    _validate_floating_vector("next_state", next_state, final_dim=V1_STATE_DIM)
    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    for name, value in (("next_state", next_state), ("task", task)):
        _require_same_context("current_state", current_state, name, value)
    terminal_bool = _normalize_terminal(
        terminal,
        leading_shape=current_state.shape[:-1],
        device=current_state.device,
    )
    with torch.no_grad():
        bootstrap = target(next_state.detach(), task.detach()).float()
        continuation = (~terminal_bool).to(dtype=torch.float32).unsqueeze(-1)
        result = current_state.detach().float() + gamma_value * continuation * bootstrap
    return result.detach()


@dataclass(frozen=True)
class C4BranchLoss:
    prediction: torch.Tensor
    per_transition_vector_loss: torch.Tensor
    vector_loss: torch.Tensor
    prediction_score: torch.Tensor
    target_score: torch.Tensor
    score_residual: torch.Tensor
    goal_loss: torch.Tensor
    loss: torch.Tensor


@dataclass(frozen=True)
class C4TDOutput:
    target: torch.Tensor
    real: C4BranchLoss
    predicted: C4BranchLoss
    total_loss: torch.Tensor
    goal_indices: torch.Tensor
    terminal: torch.Tensor


def _branch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: torch.Tensor,
    goal_indices: torch.Tensor,
    *,
    goal_projection_weight: float,
) -> C4BranchLoss:
    per_vector = (prediction.float() - target.float()).square().sum(dim=-1)
    vector_loss = per_vector.mean()
    detached_task = task.detach().float()
    prediction_score = (prediction.float() * detached_task).sum(dim=-1)
    with torch.no_grad():
        target_score = (target.detach().float() * detached_task).sum(dim=-1)
    residual = prediction_score - target_score
    if goal_indices.numel():
        goal_loss = residual.index_select(0, goal_indices).square().mean()
    else:
        goal_loss = prediction_score.sum() * 0.0
    loss = vector_loss + float(goal_projection_weight) * goal_loss
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("C4 branch loss became non-finite.")
    return C4BranchLoss(
        prediction=prediction,
        per_transition_vector_loss=per_vector,
        vector_loss=vector_loss,
        prediction_score=prediction_score,
        target_score=target_score,
        score_residual=residual,
        goal_loss=goal_loss,
        loss=loss,
    )


def build_two_branch_td_loss_v1_c4(
    online: ActorFreeTDJEPAPredictorV1C4,
    target: ActorFreeTDJEPAPredictorV1C4,
    real_state: torch.Tensor,
    predicted_state: torch.Tensor,
    next_state: torch.Tensor,
    task: torch.Tensor,
    goal_mask: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
    goal_projection_weight: float = 1.0,
) -> C4TDOutput:
    """Compute the predeclared equal-weight real/predicted C4 objective."""

    _validate_predictor_pair(online, target)
    _validate_floating_vector("real_state", real_state, final_dim=V1_STATE_DIM)
    _validate_floating_vector(
        "predicted_state", predicted_state, final_dim=V1_STATE_DIM
    )
    _validate_floating_vector("next_state", next_state, final_dim=V1_STATE_DIM)
    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    for name, value in (
        ("predicted_state", predicted_state),
        ("next_state", next_state),
        ("task", task),
    ):
        _require_same_context("real_state", real_state, name, value)
    if real_state.ndim != 2:
        raise ValueError("C4 training states must have shape [batch, 192].")
    if not isinstance(goal_mask, torch.Tensor) or goal_mask.dtype != torch.bool:
        raise TypeError("goal_mask must be a boolean torch.Tensor.")
    if goal_mask.shape != real_state.shape[:-1] or goal_mask.device != real_state.device:
        raise ValueError("goal_mask must be an aligned [batch] tensor.")
    coefficient = float(goal_projection_weight)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("goal_projection_weight must be finite and non-negative.")
    terminal_bool = _normalize_terminal(
        terminal,
        leading_shape=real_state.shape[:-1],
        device=real_state.device,
    )
    frozen_real = real_state.detach()
    frozen_predicted = predicted_state.detach()
    frozen_next = next_state.detach()
    frozen_task = task.detach()
    shared_target = successor_td_target_v1_c4(
        target,
        frozen_real,
        frozen_next,
        frozen_task,
        gamma=gamma,
        terminal=terminal_bool,
    )
    real_prediction = online(frozen_real, frozen_task)
    predicted_prediction = online(frozen_predicted, frozen_task)
    goal_indices = torch.nonzero(goal_mask, as_tuple=False).flatten()
    real_output = _branch_loss(
        real_prediction,
        shared_target,
        frozen_task,
        goal_indices,
        goal_projection_weight=coefficient,
    )
    predicted_output = _branch_loss(
        predicted_prediction,
        shared_target,
        frozen_task,
        goal_indices,
        goal_projection_weight=coefficient,
    )
    total = 0.5 * (real_output.loss + predicted_output.loss)
    return C4TDOutput(
        target=shared_target,
        real=real_output,
        predicted=predicted_output,
        total_loss=total,
        goal_indices=goal_indices,
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
    "C4BranchLoss",
    "C4TDOutput",
    "C4_F_HISTORY_STATES",
    "C4_F_PREVIOUS_ACTIONS",
    "C4_OUTPUT_DIM",
    "build_two_branch_td_loss_v1_c4",
    "ema_update_target_v1_c4",
    "predict_frozen_lewm_aligned_state_v1_c4",
    "successor_td_target_v1_c4",
]
