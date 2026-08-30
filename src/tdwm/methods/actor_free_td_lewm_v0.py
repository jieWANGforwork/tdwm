"""V0 actor-free TD-JEPA predictor and frozen-LeWM tensor objectives.

This module intentionally lives beside, rather than replacing, the V-1
history-conditioned successor implementation.  V0 consumes exactly one frozen
LeWM state, one normalized raw five-step Cube action block, and one task vector:

``G(state[192], action[25], task[192]) -> successor[192]``.

The predictor follows the symmetric TD-JEPA ``ForwardMap`` topology with one
predictor, not a parallel estimator ensemble.  It owns one state-task branch,
one state-action branch, and one output MLP.  No LeWM action encoder, latent
history, previous action, actor, or policy is part of this module.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

V0_STATE_DIM = 192
V0_TASK_DIM = 192
V0_ACTION_DIM = 25
V0_OUTPUT_DIM = 192


def _validate_floating_vector(
    name: str,
    value: torch.Tensor,
    *,
    final_dim: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if value.ndim < 2:
        raise ValueError(f"{name} must contain a transition and vector axis.")
    if value.shape[-1] != final_dim:
        raise ValueError(
            f"{name} must end with dimension {final_dim}, found {value.shape[-1]}."
        )
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _require_matching_tensor_context(
    reference_name: str,
    reference: torch.Tensor,
    other_name: str,
    other: torch.Tensor,
) -> None:
    if other.shape[:-1] != reference.shape[:-1]:
        raise ValueError(
            f"{other_name} must match the {reference_name} leading axes, found "
            f"{tuple(other.shape[:-1])} and {tuple(reference.shape[:-1])}."
        )
    if other.device != reference.device:
        raise ValueError(f"{other_name} and {reference_name} must share a device.")
    if other.dtype != reference.dtype:
        raise ValueError(f"{other_name} and {reference_name} must share a dtype.")


def _simple_embedding(
    input_dim: int,
    hidden_dim: int,
    embedding_layers: int,
) -> nn.Sequential:
    """Return the embedding topology used by TD-JEPA ``ForwardMap``."""

    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
    ]
    for _ in range(embedding_layers - 2):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
    layers.extend((nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()))
    return nn.Sequential(*layers)


class ActorFreeTDJEPAPredictorV0(nn.Module):
    """Single symmetric TD-JEPA predictor for frozen LeWM latents.

    The action is already the normalized raw 25-dimensional LeWM action block.
    It is consumed by the predictor's state-action branch directly; the frozen
    LeWM action encoder is deliberately not reused in V0.
    """

    state_dim = V0_STATE_DIM
    task_dim = V0_TASK_DIM
    action_dim = V0_ACTION_DIM
    output_dim = V0_OUTPUT_DIM

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
        self.embed_task = _simple_embedding(
            V0_STATE_DIM + V0_TASK_DIM,
            self.hidden_dim,
            self.embedding_layers,
        )
        self.embed_state_action = _simple_embedding(
            V0_STATE_DIM + V0_ACTION_DIM,
            self.hidden_dim,
            self.embedding_layers,
        )
        output_layers: list[nn.Module] = []
        for _ in range(self.hidden_layers):
            output_layers.extend(
                (nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
            )
        output_layers.append(nn.Linear(self.hidden_dim, V0_OUTPUT_DIM))
        self.output = nn.Sequential(*output_layers)
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        # TD-JEPA orthogonally initializes its model after construction.
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        _validate_floating_vector("state", state, final_dim=self.state_dim)
        _validate_floating_vector("action", action, final_dim=self.action_dim)
        _validate_floating_vector("task", task, final_dim=self.task_dim)
        _require_matching_tensor_context("state", state, "action", action)
        _require_matching_tensor_context("state", state, "task", task)

        task_embedding = self.embed_task(torch.cat((state, task), dim=-1))
        state_action_embedding = self.embed_state_action(
            torch.cat((state, action), dim=-1)
        )
        output = self.output(
            torch.cat((state_action_embedding, task_embedding), dim=-1)
        )
        if not bool(torch.isfinite(output.detach()).all()):
            raise FloatingPointError("V0 predictor produced a non-finite output.")
        return output

    def make_target(self) -> "ActorFreeTDJEPAPredictorV0":
        """Return an initialized, frozen EMA target copy."""

        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


@dataclass(frozen=True)
class MixedTasksV0:
    """Per-transition mixed tasks and their detached source diagnostics.

    ``goal_mask`` is true exactly where the sampled task came from the matched
    real goal.  False entries came from the isotropic random sphere.
    """

    task: torch.Tensor
    goal_mask: torch.Tensor
    goal_task: torch.Tensor
    random_task: torch.Tensor
    goal_count: int
    random_count: int


def project_tasks_to_sphere_v0(task: torch.Tensor) -> torch.Tensor:
    """Detach and project 192D task vectors onto radius ``sqrt(192)``."""

    _validate_floating_vector("task", task, final_dim=V0_TASK_DIM)
    detached = task.detach()
    norm = torch.norm(detached, dim=-1, keepdim=True)
    minimum_norm = torch.finfo(detached.dtype).eps
    if bool((norm <= minimum_norm).any()):
        raise ValueError("task vectors must have non-zero finite norm.")
    return math.sqrt(V0_TASK_DIM) * detached / norm


def sample_mixed_tasks_v0(
    matched_real_goal: torch.Tensor,
    *,
    goal_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> MixedTasksV0:
    """Draw the official per-transition Bernoulli random/goal task mixture."""

    if not 0.0 <= goal_probability <= 1.0:
        raise ValueError("goal_probability must lie in [0, 1].")
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator.")
    goal_task = project_tasks_to_sphere_v0(matched_real_goal)
    # Generate on CPU even when training is on an accelerator.  A dedicated CPU
    # generator can then be checkpointed once and resumes identically regardless
    # of the device hosting the frozen LeWM tensors.
    random_task_cpu = project_tasks_to_sphere_v0(
        torch.randn(
            matched_real_goal.shape,
            dtype=matched_real_goal.dtype,
            device="cpu",
            generator=generator,
        )
    )
    random_task = random_task_cpu.to(device=matched_real_goal.device)
    transition_count = matched_real_goal.numel() // V0_TASK_DIM
    goal_mask = (
        torch.rand(
            matched_real_goal.shape[:-1],
            device="cpu",
            generator=generator,
        )
        .lt(float(goal_probability))
        .to(device=matched_real_goal.device)
    )
    goal_count = int(goal_mask.sum().item())
    random_count = transition_count - goal_count
    task = torch.where(goal_mask.unsqueeze(-1), goal_task, random_task)
    return MixedTasksV0(
        task=task.detach(),
        goal_mask=goal_mask,
        goal_task=goal_task.detach(),
        random_task=random_task.detach(),
        goal_count=goal_count,
        random_count=random_count,
    )


def tdjepa_goal_score_v0(
    prediction: torch.Tensor,
    task: torch.Tensor,
) -> torch.Tensor:
    """Return the single predictor's ``G(state, action, task) dot task``.

    Predictor outputs may be bfloat16 under the formal autocast policy while
    frozen tasks remain float32.  Cast both operands to float32 before the
    reduction so the 192-term dot product is accumulated stably without
    breaking the gradient from the score back to the online prediction.
    """

    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch.Tensor.")
    if not prediction.is_floating_point():
        raise TypeError("prediction must have a floating-point dtype.")
    if prediction.ndim < 2:
        raise ValueError("prediction must contain transition and feature axes.")
    if prediction.shape[-1] != V0_OUTPUT_DIM:
        raise ValueError("prediction must end with the 192D V0 output dimension.")
    if not bool(torch.isfinite(prediction.detach()).all()):
        raise ValueError("prediction must contain only finite values.")
    _validate_floating_vector("task", task, final_dim=V0_TASK_DIM)
    if prediction.shape != task.shape:
        raise ValueError("task must have the same shape as prediction.")
    if prediction.device != task.device:
        raise ValueError("prediction and task must share a device.")
    return (prediction.float() * task.float()).sum(dim=-1)


def _normalize_terminal_v0(
    terminal: torch.Tensor | bool,
    *,
    leading_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(terminal, bool):
        return torch.full(leading_shape, terminal, dtype=torch.bool, device=device)
    if not isinstance(terminal, torch.Tensor):
        raise TypeError("terminal must be a torch.Tensor or bool.")
    if terminal.shape != leading_shape:
        raise ValueError(
            f"terminal must have shape {tuple(leading_shape)}, found "
            f"{tuple(terminal.shape)}."
        )
    terminal_device = terminal.to(device=device)
    terminal_bool = terminal_device.to(dtype=torch.bool)
    if bool((terminal_device != terminal_bool).any()):
        raise ValueError("terminal must contain only binary values.")
    return terminal_bool


def tdjepa_successor_td_target_v0(
    next_state: torch.Tensor,
    target_next_prediction: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> torch.Tensor:
    """Build ``s_next + gamma * (1-terminal) * target_prediction``."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    _validate_floating_vector("next_state", next_state, final_dim=V0_STATE_DIM)
    if not isinstance(target_next_prediction, torch.Tensor):
        raise TypeError("target_next_prediction must be a torch.Tensor.")
    if target_next_prediction.shape != next_state.shape:
        raise ValueError("target_next_prediction must match next_state.")
    if not target_next_prediction.is_floating_point():
        raise TypeError("target_next_prediction must have a floating-point dtype.")
    if target_next_prediction.device != next_state.device:
        raise ValueError(
            "target_next_prediction and next_state must share a device."
        )
    if not bool(torch.isfinite(target_next_prediction.detach()).all()):
        raise ValueError("target_next_prediction must contain only finite values.")

    terminal_bool = _normalize_terminal_v0(
        terminal,
        leading_shape=next_state.shape[:-1],
        device=next_state.device,
    )
    # The EMA prediction is bfloat16 under autocast whereas frozen LeWM states
    # are float32.  Form the complete detached bootstrap in float32 so neither
    # the feature sum nor the Bellman target is rounded to bfloat16.
    continuation = (~terminal_bool).to(dtype=torch.float32).unsqueeze(-1)
    discounted_bootstrap = (
        float(gamma) * continuation * target_next_prediction.detach().float()
    )
    return next_state.detach().float() + discounted_bootstrap


@dataclass(frozen=True)
class TDJEPATDBatchV0:
    """Aligned V0 prediction, detached TD target, and transition losses."""

    prediction: torch.Tensor
    target: torch.Tensor
    per_transition_td_loss: torch.Tensor
    td_loss: torch.Tensor
    task: torch.Tensor
    terminal: torch.Tensor


def _validate_predictor_pair_v0(
    online: ActorFreeTDJEPAPredictorV0,
    target: ActorFreeTDJEPAPredictorV0,
) -> None:
    if not isinstance(online, ActorFreeTDJEPAPredictorV0):
        raise TypeError("online must be an ActorFreeTDJEPAPredictorV0.")
    if not isinstance(target, ActorFreeTDJEPAPredictorV0):
        raise TypeError("target must be an ActorFreeTDJEPAPredictorV0.")
    if online is target:
        raise ValueError("online and target must be distinct predictor instances.")
    architecture = (
        "hidden_dim",
        "hidden_layers",
        "embedding_layers",
    )
    if any(getattr(online, name) != getattr(target, name) for name in architecture):
        raise ValueError("online and target predictors must share an architecture.")
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("target predictor parameters must be frozen.")


def build_tdjepa_td_batch_v0(
    online: ActorFreeTDJEPAPredictorV0,
    target: ActorFreeTDJEPAPredictorV0,
    state: torch.Tensor,
    action: torch.Tensor,
    task: torch.Tensor,
    next_state: torch.Tensor,
    next_action: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> TDJEPATDBatchV0:
    """Build the V0 TD batch using the dataset's actual next action block.

    Frozen LeWM latents, normalized raw actions, and tasks are detached at this
    boundary.  Back-propagation therefore updates only the online predictor.
    """

    _validate_predictor_pair_v0(online, target)
    _validate_floating_vector("state", state, final_dim=V0_STATE_DIM)
    _validate_floating_vector("action", action, final_dim=V0_ACTION_DIM)
    _validate_floating_vector("task", task, final_dim=V0_TASK_DIM)
    _validate_floating_vector("next_state", next_state, final_dim=V0_STATE_DIM)
    _validate_floating_vector("next_action", next_action, final_dim=V0_ACTION_DIM)
    for other_name, other in (
        ("action", action),
        ("task", task),
        ("next_state", next_state),
        ("next_action", next_action),
    ):
        _require_matching_tensor_context("state", state, other_name, other)

    terminal_bool = _normalize_terminal_v0(
        terminal,
        leading_shape=state.shape[:-1],
        device=state.device,
    )
    frozen_state = state.detach()
    frozen_action = action.detach()
    frozen_task = task.detach()
    frozen_next_state = next_state.detach()
    frozen_next_action = next_action.detach()

    prediction = online(frozen_state, frozen_action, frozen_task)
    with torch.no_grad():
        target_next_prediction = target(
            frozen_next_state,
            frozen_next_action,
            frozen_task,
        )
        td_target = tdjepa_successor_td_target_v0(
            frozen_next_state,
            target_next_prediction,
            gamma=gamma,
            terminal=terminal_bool,
        )

    # Match the symmetric TD-JEPA reduction: sum feature error per transition,
    # then average transitions in the scalar objective.
    # Autocast returns bfloat16 predictor features.  Promote before subtracting
    # and summing the 192 squared residuals; ``float()`` remains differentiable
    # and therefore preserves the online predictor's gradient path.
    per_transition = (prediction.float() - td_target.float()).square().sum(dim=-1)
    if not bool(torch.isfinite(per_transition.detach()).all()):
        raise FloatingPointError("V0 TD loss produced a non-finite value.")
    return TDJEPATDBatchV0(
        prediction=prediction,
        target=td_target,
        per_transition_td_loss=per_transition,
        td_loss=per_transition.mean(),
        task=frozen_task,
        terminal=terminal_bool,
    )


@torch.no_grad()
def ema_update_target_v0(
    target: ActorFreeTDJEPAPredictorV0,
    online: ActorFreeTDJEPAPredictorV0,
    *,
    decay: float,
) -> None:
    """Update a frozen V0 target as ``decay*target + (1-decay)*online``."""

    _validate_predictor_pair_v0(online, target)
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must lie in [0, 1].")
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.mul_(float(decay)).add_(
            online_parameter,
            alpha=1.0 - float(decay),
        )
    for target_buffer, online_buffer in zip(
        target.buffers(), online.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(float(decay)).add_(
                online_buffer,
                alpha=1.0 - float(decay),
            )
        else:
            target_buffer.copy_(online_buffer)
    target.requires_grad_(False)
    target.eval()


__all__ = [
    "ActorFreeTDJEPAPredictorV0",
    "MixedTasksV0",
    "TDJEPATDBatchV0",
    "V0_ACTION_DIM",
    "V0_OUTPUT_DIM",
    "V0_STATE_DIM",
    "V0_TASK_DIM",
    "build_tdjepa_td_batch_v0",
    "ema_update_target_v0",
    "project_tasks_to_sphere_v0",
    "sample_mixed_tasks_v0",
    "tdjepa_goal_score_v0",
    "tdjepa_successor_td_target_v0",
]
