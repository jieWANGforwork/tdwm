"""EffAction: undiscounted successor and path-cost supervision.

The caller supplies the direct/TD branch and its validity mask. This module
does not infer efficiency labels, sample episodes, or add another objective.
The frozen LeWM encoders remain external to the two independent heads.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from tdwm.methods.actor_free_td_lewm_v1 import (
    ActorFreeTDJEPAPredictorV1,
    encode_frozen_action_blocks_v1,
    validate_frozen_lewm_action_encoder_v1,
)
from tdwm.methods.actor_free_td_lewm_v2 import encode_trainable_action_blocks_v2

EFF_ACTION_STATE_DIM = 192
EFF_ACTION_TASK_DIM = 192
EFF_ACTION_RAW_ACTION_DIM = 25
EFF_ACTION_EMBEDDING_DIM = 192
ValueOutput = Literal["softplus", "relu", "square"]


def _vector(
    name: str,
    value: torch.Tensor,
    dimension: int,
    *,
    finite: bool = True,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if value.ndim < 1 or value.shape[-1] != dimension or value.numel() == 0:
        raise ValueError(f"{name} must have nonempty shape (..., {dimension}).")
    if finite and not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _aligned(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    if value.shape[:-1] != reference.shape[:-1]:
        raise ValueError(f"{name} must have the same leading shape as state.")
    if value.device != reference.device:
        raise ValueError(f"{name} and state must share a device.")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} and state must share a dtype.")


def _mask(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean torch.Tensor.")
    if value.shape != reference.shape[:-1]:
        raise ValueError(f"{name} must match the state leading shape.")
    if value.device != reference.device:
        raise ValueError(f"{name} and state must share a device.")


def require_frozen_eff_action_module(module: nn.Module, name: str) -> None:
    """Check weights and mutable training state before planner use."""

    if not isinstance(module, nn.Module):
        raise TypeError(f"{name} must be a torch.nn.Module.")
    if any(part.training for part in module.modules()):
        raise ValueError(f"{name} and all submodules must be in eval mode.")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"{name} parameters must be frozen.")


class EffActionSuccessor(ActorFreeTDJEPAPredictorV1):
    """Independent G(z, e(a), m) using the existing project G topology.

    Only the architecture is reused; this instance owns independent weights
    and is trained exclusively by the EffAction objective. All architecture
    choices must be provided in the experiment configuration.
    """

    def __init__(
        self, *, hidden_dim: int, hidden_layers: int, embedding_layers: int
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            embedding_layers=embedding_layers,
        )

    def forward(
        self,
        state: torch.Tensor,
        action_embedding: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        # The inherited topology accepts arbitrary batch axes but requires at
        # least one. Handle an individual candidate without changing it.
        if isinstance(state, torch.Tensor) and state.ndim == 1:
            return (
                super()
                .forward(
                    state.unsqueeze(0), action_embedding.unsqueeze(0), task.unsqueeze(0)
                )
                .squeeze(0)
            )
        return super().forward(state, action_embedding, task)


class EffActionValue(nn.Module):
    """Nonnegative scalar path cost V(Ψ, m), with no input state bypass."""

    state_dim = EFF_ACTION_STATE_DIM
    task_dim = EFF_ACTION_TASK_DIM

    def __init__(
        self, *, hidden_dim: int, hidden_layers: int, output_activation: ValueOutput
    ) -> None:
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if not isinstance(hidden_layers, int) or hidden_layers < 1:
            raise ValueError("hidden_layers must be a positive integer.")
        if output_activation not in {"softplus", "relu", "square"}:
            raise ValueError("output_activation must be softplus, relu, or square.")
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_activation = output_activation
        layers: list[nn.Module] = []
        input_dim = self.state_dim + self.task_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, successor: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        _vector("successor", successor, self.state_dim)
        _vector("task", task, self.task_dim)
        _aligned("task", task, successor)
        raw = self.network(torch.cat((successor, task), dim=-1)).squeeze(-1)
        # Cost arithmetic stays float32 when the head uses mixed precision.
        raw = raw.float()
        if self.output_activation == "softplus":
            value = F.softplus(raw)
        elif self.output_activation == "relu":
            value = F.relu(raw)
        else:
            value = raw.square()
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError("EffAction V produced a non-finite value.")
        return value

    def make_target(self) -> EffActionValue:
        return copy.deepcopy(self).requires_grad_(False).eval()


def encode_eff_action_blocks(
    action_encoder: nn.Module,
    raw_action: torch.Tensor,
    reference: torch.Tensor,
    *,
    retain_action_grad: bool,
) -> torch.Tensor:
    """Reuse the shared LeWM 25-to-192 encoder; optionally keep da gradients.

    Frozen encoder weights do not imply a detached action input. The planner
    needs the differentiable project helper, while G/V training uses the
    existing detached helper.
    """

    validate_frozen_lewm_action_encoder_v1(action_encoder)
    if retain_action_grad:
        return encode_trainable_action_blocks_v2(action_encoder, raw_action, reference)
    return encode_frozen_action_blocks_v1(action_encoder, raw_action, reference)


@dataclass(frozen=True)
class EffActionLossOutput:
    g_loss: torch.Tensor
    v_loss: torch.Tensor
    g_direct_loss: torch.Tensor
    g_td_loss: torch.Tensor
    v_direct_loss: torch.Tensor
    v_td_loss: torch.Tensor
    g_prediction: torch.Tensor
    v_prediction: torch.Tensor
    g_target: torch.Tensor
    v_target: torch.Tensor
    g_per_example_loss: torch.Tensor
    v_per_example_loss: torch.Tensor
    valid_mask: torch.Tensor
    direct_branch: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        """Convenience sum; V's input detachment keeps its gradient separate."""

        return self.g_loss + self.v_loss


def _optional_vector(
    name: str,
    value: torch.Tensor | None,
    dimension: int,
    reference: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required for the selected branch.")
    _vector(name, value, dimension, finite=False)
    _aligned(name, value, reference)
    subset = value.reshape(-1, dimension)[selected].detach()
    _vector(name, subset, dimension)
    return subset


def build_eff_action_loss(
    successor: EffActionSuccessor,
    value: EffActionValue,
    target_successor: EffActionSuccessor,
    target_value: EffActionValue,
    action_encoder: nn.Module,
    *,
    state: torch.Tensor,
    raw_action: torch.Tensor,
    task: torch.Tensor,
    goal: torch.Tensor,
    direct_branch: torch.Tensor,
    valid_mask: torch.Tensor,
    goal_terminal: torch.Tensor,
    direct_successor_target: torch.Tensor | None,
    next_state: torch.Tensor | None,
    next_raw_action: torch.Tensor | None,
) -> EffActionLossOutput:
    """Compute the exact two-branch G/V losses, with no discount factor.

    ``direct_branch`` is the externally supplied boolean b. It selects both
    direct terms only for the caller's high-efficiency examples. The direct G
    target must be the *sum* of real future z through the goal, not their mean.
    On TD examples the target is z_next + target_G(...), and the scalar target
    is ||z_next-z||₂ + target_V(target_G(...), m). At a goal terminal both
    bootstraps are zero; no next action is read. ``valid_mask`` removes padded
    examples before any head or encoder is called.

    Every loss is an expectation over all valid examples. Branch loss fields
    are contributions to that expectation (they sum to g_loss/v_loss), not
    conditional branch means. Invalid entries in returned arrays are zero.
    """

    _vector("state", state, EFF_ACTION_STATE_DIM, finite=False)
    for name, tensor, dimension in (
        ("raw_action", raw_action, EFF_ACTION_RAW_ACTION_DIM),
        ("task", task, EFF_ACTION_TASK_DIM),
        ("goal", goal, EFF_ACTION_STATE_DIM),
    ):
        _vector(name, tensor, dimension, finite=False)
        _aligned(name, tensor, state)
    for name, tensor in (
        ("direct_branch", direct_branch),
        ("valid_mask", valid_mask),
        ("goal_terminal", goal_terminal),
    ):
        _mask(name, tensor, state)
    selected = valid_mask.reshape(-1)
    if not bool(selected.any()):
        raise ValueError("valid_mask must select at least one example.")
    if successor is target_successor or value is target_value:
        raise ValueError("Online and target heads must be independent modules.")
    require_frozen_eff_action_module(target_successor, "target_successor")
    require_frozen_eff_action_module(target_value, "target_value")
    validate_frozen_lewm_action_encoder_v1(action_encoder)

    selected_state = state.reshape(-1, EFF_ACTION_STATE_DIM)[selected].detach()
    selected_action = raw_action.reshape(-1, EFF_ACTION_RAW_ACTION_DIM)[
        selected
    ].detach()
    selected_task = task.reshape(-1, EFF_ACTION_TASK_DIM)[selected].detach()
    selected_goal = goal.reshape(-1, EFF_ACTION_STATE_DIM)[selected].detach()
    for name, tensor, dimension in (
        ("state", selected_state, EFF_ACTION_STATE_DIM),
        ("raw_action", selected_action, EFF_ACTION_RAW_ACTION_DIM),
        ("task", selected_task, EFF_ACTION_TASK_DIM),
        ("goal", selected_goal, EFF_ACTION_STATE_DIM),
    ):
        _vector(name, tensor, dimension)
    embedding = encode_eff_action_blocks(
        action_encoder, selected_action, selected_state, retain_action_grad=False
    )
    predicted_g = successor(selected_state, embedding, selected_task)
    # The cast is differentiable for V parameters, while Ψ remains detached.
    predicted_v = value(predicted_g.detach().to(selected_task.dtype), selected_task)
    selected_direct = direct_branch.reshape(-1)[selected]
    direct_rows = selected & direct_branch.reshape(-1)
    td_rows = selected & ~direct_branch.reshape(-1)
    selected_terminal = goal_terminal.reshape(-1)[selected]
    g_target = torch.zeros_like(selected_state, dtype=torch.float32)
    v_target = torch.zeros_like(predicted_v, dtype=torch.float32)

    with torch.no_grad():
        if bool(selected_direct.any()):
            direct_g = _optional_vector(
                "direct_successor_target",
                direct_successor_target,
                EFF_ACTION_STATE_DIM,
                state,
                direct_rows,
            )
            g_target[selected_direct] = direct_g.float()
            v_target[selected_direct] = torch.linalg.vector_norm(
                selected_goal[selected_direct].float()
                - selected_state[selected_direct].float(),
                dim=-1,
            )
        if bool((~selected_direct).any()):
            td_next_state = _optional_vector(
                "next_state", next_state, EFF_ACTION_STATE_DIM, state, td_rows
            )
            td_g = td_next_state.float().clone()
            td_v = torch.linalg.vector_norm(
                td_next_state.float() - selected_state[~selected_direct].float(),
                dim=-1,
            )
            continuation = ~selected_terminal[~selected_direct]
            if bool(continuation.any()):
                continuation_rows = td_rows & ~goal_terminal.reshape(-1)
                next_actions = _optional_vector(
                    "next_raw_action",
                    next_raw_action,
                    EFF_ACTION_RAW_ACTION_DIM,
                    state,
                    continuation_rows,
                )
                continuing_state = td_next_state[continuation]
                continuing_task = selected_task[~selected_direct][continuation]
                next_embedding = encode_eff_action_blocks(
                    action_encoder,
                    next_actions,
                    continuing_state,
                    retain_action_grad=False,
                )
                bootstrap_g = target_successor(
                    continuing_state, next_embedding, continuing_task
                )
                bootstrap_v = target_value(
                    bootstrap_g.to(continuing_task.dtype), continuing_task
                )
                td_g[continuation] += bootstrap_g.float()
                td_v[continuation] += bootstrap_v.float()
            g_target[~selected_direct] = td_g
            v_target[~selected_direct] = td_v
    if not bool(torch.isfinite(g_target).all() and torch.isfinite(v_target).all()):
        raise FloatingPointError("EffAction target contains a non-finite value.")
    g_error = (predicted_g.float() - g_target).square().sum(dim=-1)
    v_error = (predicted_v.float() - v_target).square()
    if not bool(torch.isfinite(g_error.detach()).all()):
        raise FloatingPointError("EffAction G loss is non-finite.")
    if not bool(torch.isfinite(v_error.detach()).all()):
        raise FloatingPointError("EffAction V loss is non-finite.")
    denominator = selected.sum().to(torch.float32)
    direct_float = selected_direct.to(torch.float32)
    g_direct_loss = (g_error * direct_float).sum() / denominator
    g_td_loss = (g_error * (1.0 - direct_float)).sum() / denominator
    v_direct_loss = (v_error * direct_float).sum() / denominator
    v_td_loss = (v_error * (1.0 - direct_float)).sum() / denominator

    def restore(tensor: torch.Tensor, *, vector: bool = False) -> torch.Tensor:
        shape = (
            (selected.numel(), EFF_ACTION_STATE_DIM) if vector else (selected.numel(),)
        )
        output = tensor.new_zeros(shape)
        output[selected] = tensor
        final_shape = state.shape if vector else state.shape[:-1]
        return output.reshape(final_shape)

    return EffActionLossOutput(
        g_loss=g_direct_loss + g_td_loss,
        v_loss=v_direct_loss + v_td_loss,
        g_direct_loss=g_direct_loss,
        g_td_loss=g_td_loss,
        v_direct_loss=v_direct_loss,
        v_td_loss=v_td_loss,
        g_prediction=restore(predicted_g, vector=True),
        v_prediction=restore(predicted_v),
        g_target=restore(g_target, vector=True),
        v_target=restore(v_target),
        g_per_example_loss=restore(g_error),
        v_per_example_loss=restore(v_error),
        valid_mask=valid_mask.detach().clone(),
        direct_branch=direct_branch.detach().clone(),
    )


def eff_action_cost(
    successor: EffActionSuccessor,
    value: EffActionValue,
    action_encoder: nn.Module,
    *,
    state: torch.Tensor,
    raw_action: torch.Tensor,
    task: torch.Tensor,
    goal: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return J(a) = -||z_goal-z||₂ / (V(G(z,e(a),m),m) + epsilon).

    This is a minimization cost. It retains the action derivative and adds no
    imagined rollouts, reward query, preference weight, or consistency term.
    The caller owns any rollout-prefix protocol needed by its CEM adapter.
    """

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    _vector("state", state, EFF_ACTION_STATE_DIM)
    for name, tensor, dimension in (
        ("raw_action", raw_action, EFF_ACTION_RAW_ACTION_DIM),
        ("task", task, EFF_ACTION_TASK_DIM),
        ("goal", goal, EFF_ACTION_STATE_DIM),
    ):
        _vector(name, tensor, dimension)
        _aligned(name, tensor, state)
    state = state.detach()
    task = task.detach()
    goal = goal.detach()
    embedding = encode_eff_action_blocks(
        action_encoder, raw_action, state, retain_action_grad=True
    )
    psi = successor(state, embedding, task)
    path_cost = value(psi.to(task.dtype), task).float()
    if bool((path_cost.detach() < 0.0).any()):
        raise ValueError("EffAction V must return nonnegative path costs.")
    distance = torch.linalg.vector_norm(goal.float() - state.float(), dim=-1)
    cost = -distance / (path_cost + epsilon)
    if not bool(torch.isfinite(cost.detach()).all()):
        raise FloatingPointError("EffAction J is non-finite.")
    return cost


@torch.no_grad()
def ema_update_eff_action(target: nn.Module, online: nn.Module, *, rate: float) -> None:
    """Apply target <- (1-rate) target + rate online to matching heads."""

    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("EMA rate must lie in [0, 1].")
    if target is online:
        raise ValueError("Online and target heads must be independent modules.")
    require_frozen_eff_action_module(target, "target")
    target_state = target.state_dict()
    online_state = online.state_dict()
    if target_state.keys() != online_state.keys():
        raise ValueError(
            "Online and target state dictionaries must have matching keys."
        )
    for name, target_tensor in target_state.items():
        online_tensor = online_state[name]
        if target_tensor.shape != online_tensor.shape:
            raise ValueError(f"EMA shape mismatch for {name}.")
        if target_tensor.dtype != online_tensor.dtype:
            raise ValueError(f"EMA dtype mismatch for {name}.")
        if target_tensor.device != online_tensor.device:
            raise ValueError(f"EMA device mismatch for {name}.")
        if not bool(torch.isfinite(online_tensor).all()):
            raise ValueError(f"EMA online state {name} must contain finite values.")
    for name, target_tensor in target_state.items():
        if target_tensor.is_floating_point():
            target_tensor.lerp_(online_state[name], rate)
        else:
            target_tensor.copy_(online_state[name])


__all__ = [
    "EFF_ACTION_EMBEDDING_DIM",
    "EFF_ACTION_RAW_ACTION_DIM",
    "EFF_ACTION_STATE_DIM",
    "EFF_ACTION_TASK_DIM",
    "EffActionLossOutput",
    "EffActionSuccessor",
    "EffActionValue",
    "build_eff_action_loss",
    "eff_action_cost",
    "ema_update_eff_action",
    "encode_eff_action_blocks",
    "require_frozen_eff_action_module",
]
