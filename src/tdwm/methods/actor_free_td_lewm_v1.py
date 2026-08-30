"""V1 actor-free TD-JEPA predictor with shared frozen LeWM actions.

V1 keeps the V0 TD, task-sampling, and EMA semantics, but changes the action
input to the embedding produced by the *existing* pretrained LeWM action
encoder::

    E_A(raw_action[..., 25]) -> action_embedding[..., 192]
    G(state[..., 192], action_embedding[..., 192], task[..., 192])
      -> successor[..., 192]

The action encoder is intentionally not owned by the predictor.  Callers pass
the world model's frozen encoder to :func:`encode_frozen_action_blocks_v1` and
then pass the resulting embeddings to the online and EMA predictors.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

V1_STATE_DIM = 192
V1_TASK_DIM = 192
V1_RAW_ACTION_DIM = 25
V1_ACTION_EMBEDDING_DIM = 192
# ``action_dim`` always describes the predictor input.  Raw dataset and CEM
# blocks must use ``V1_RAW_ACTION_DIM`` instead.
V1_ACTION_DIM = V1_ACTION_EMBEDDING_DIM
V1_OUTPUT_DIM = 192


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
        raise ValueError(
            f"{name} must have at least {minimum_ndim} dimensions."
        )
    if value.shape[-1] != final_dim:
        raise ValueError(
            f"{name} must end with dimension {final_dim}, found "
            f"{value.shape[-1]}."
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


def validate_frozen_lewm_action_encoder_v1(action_encoder: nn.Module) -> None:
    """Fail closed unless this is the shared frozen LeWM 25-to-192 encoder."""

    if not isinstance(action_encoder, nn.Module):
        raise TypeError("action_encoder must be a torch.nn.Module.")
    if getattr(action_encoder, "input_dim", None) != V1_RAW_ACTION_DIM:
        raise ValueError(
            "action_encoder.input_dim must match the normalized 25D action block."
        )
    if getattr(action_encoder, "emb_dim", None) != V1_ACTION_EMBEDDING_DIM:
        raise ValueError("action_encoder.emb_dim must be the 192D LeWM embedding.")
    if any(module.training for module in action_encoder.modules()):
        raise ValueError("action_encoder and all its submodules must be in eval mode.")
    if any(parameter.requires_grad for parameter in action_encoder.parameters()):
        raise ValueError("action_encoder parameters must be frozen.")


def encode_frozen_action_blocks_v1(
    action_encoder: nn.Module,
    raw_action: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Encode raw 25D blocks through one external frozen LeWM encoder.

    LeWM's action encoder accepts ``[batch, time, 25]``.  This helper supports
    arbitrary leading axes by flattening them to ``[-1, 1, 25]`` and restoring
    them after an exact ``[-1, 1, 192]`` result.  The returned tensor is always
    detached and aligned to ``reference`` after encoding.  That final cast is
    necessary under bf16 autocast, where the frozen encoder can return bf16
    while the frozen state latent remains float32.
    """

    validate_frozen_lewm_action_encoder_v1(action_encoder)
    _validate_floating_vector(
        "raw_action",
        raw_action,
        final_dim=V1_RAW_ACTION_DIM,
        minimum_ndim=1,
    )
    _validate_floating_vector(
        "reference",
        reference,
        final_dim=V1_STATE_DIM,
        minimum_ndim=1,
    )
    if raw_action.shape[:-1] != reference.shape[:-1]:
        raise ValueError(
            "raw_action and reference must have identical leading axes."
        )
    if raw_action.device != reference.device:
        raise ValueError("raw_action and reference must share a device.")

    leading_shape = raw_action.shape[:-1]
    flat_raw_action = raw_action.detach().reshape(-1, 1, V1_RAW_ACTION_DIM)
    with torch.no_grad():
        flat_embedding = action_encoder(flat_raw_action)
    if not isinstance(flat_embedding, torch.Tensor):
        raise TypeError("action_encoder must return a torch.Tensor.")
    expected_shape = (flat_raw_action.shape[0], 1, V1_ACTION_EMBEDDING_DIM)
    if flat_embedding.shape != expected_shape:
        raise ValueError(
            "action_encoder must return shape "
            f"{expected_shape}, found {tuple(flat_embedding.shape)}."
        )
    if not flat_embedding.is_floating_point():
        raise TypeError("action_encoder output must have a floating-point dtype.")
    if not bool(torch.isfinite(flat_embedding.detach()).all()):
        raise ValueError("action_encoder output must contain only finite values.")

    embedding = flat_embedding.detach().reshape(
        *leading_shape, V1_ACTION_EMBEDDING_DIM
    )
    return embedding.to(device=reference.device, dtype=reference.dtype).detach()


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


class ActorFreeTDJEPAPredictorV1(nn.Module):
    """Single V1 TD-JEPA predictor consuming a frozen action embedding."""

    state_dim = V1_STATE_DIM
    task_dim = V1_TASK_DIM
    raw_action_dim = V1_RAW_ACTION_DIM
    action_dim = V1_ACTION_DIM
    action_embedding_dim = V1_ACTION_EMBEDDING_DIM
    output_dim = V1_OUTPUT_DIM

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
            V1_STATE_DIM + V1_TASK_DIM,
            self.hidden_dim,
            self.embedding_layers,
        )
        self.embed_state_action = _simple_embedding(
            V1_STATE_DIM + V1_ACTION_EMBEDDING_DIM,
            self.hidden_dim,
            self.embedding_layers,
        )
        output_layers: list[nn.Module] = []
        for _ in range(self.hidden_layers):
            output_layers.extend(
                (nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
            )
        output_layers.append(nn.Linear(self.hidden_dim, V1_OUTPUT_DIM))
        self.output = nn.Sequential(*output_layers)
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        action_embedding: torch.Tensor,
        task: torch.Tensor,
    ) -> torch.Tensor:
        _validate_floating_vector("state", state, final_dim=self.state_dim)
        _validate_floating_vector(
            "action_embedding",
            action_embedding,
            final_dim=self.action_embedding_dim,
        )
        _validate_floating_vector("task", task, final_dim=self.task_dim)
        _require_matching_tensor_context(
            "state", state, "action_embedding", action_embedding
        )
        _require_matching_tensor_context("state", state, "task", task)

        task_embedding = self.embed_task(torch.cat((state, task), dim=-1))
        state_action_embedding = self.embed_state_action(
            torch.cat((state, action_embedding), dim=-1)
        )
        output = self.output(
            torch.cat((state_action_embedding, task_embedding), dim=-1)
        )
        if not bool(torch.isfinite(output.detach()).all()):
            raise FloatingPointError("V1 predictor produced a non-finite output.")
        return output

    def make_target(self) -> "ActorFreeTDJEPAPredictorV1":
        """Return a frozen EMA copy containing predictor parameters only."""

        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


@dataclass(frozen=True)
class MixedTasksV1:
    """Per-transition mixed tasks and detached source diagnostics."""

    task: torch.Tensor
    goal_mask: torch.Tensor
    goal_task: torch.Tensor
    random_task: torch.Tensor
    goal_count: int
    random_count: int


def project_tasks_to_sphere_v1(task: torch.Tensor) -> torch.Tensor:
    """Detach and project 192D task vectors onto radius ``sqrt(192)``."""

    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    detached = task.detach()
    norm = torch.norm(detached, dim=-1, keepdim=True)
    minimum_norm = torch.finfo(detached.dtype).eps
    if bool((norm <= minimum_norm).any()):
        raise ValueError("task vectors must have non-zero finite norm.")
    return math.sqrt(V1_TASK_DIM) * detached / norm


def sample_mixed_tasks_v1(
    matched_real_goal: torch.Tensor,
    *,
    goal_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> MixedTasksV1:
    """Draw the official per-transition Bernoulli random/goal task mixture."""

    if not 0.0 <= goal_probability <= 1.0:
        raise ValueError("goal_probability must lie in [0, 1].")
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator.")
    goal_task = project_tasks_to_sphere_v1(matched_real_goal)
    random_task_cpu = project_tasks_to_sphere_v1(
        torch.randn(
            matched_real_goal.shape,
            dtype=matched_real_goal.dtype,
            device="cpu",
            generator=generator,
        )
    )
    random_task = random_task_cpu.to(device=matched_real_goal.device)
    transition_count = matched_real_goal.numel() // V1_TASK_DIM
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
    return MixedTasksV1(
        task=task.detach(),
        goal_mask=goal_mask,
        goal_task=goal_task.detach(),
        random_task=random_task.detach(),
        goal_count=goal_count,
        random_count=random_count,
    )


def tdjepa_goal_score_v1(
    prediction: torch.Tensor,
    task: torch.Tensor,
) -> torch.Tensor:
    """Return ``G(state, E_A(action), task) dot task`` in float32."""

    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch.Tensor.")
    if not prediction.is_floating_point():
        raise TypeError("prediction must have a floating-point dtype.")
    if prediction.ndim < 2:
        raise ValueError("prediction must contain transition and feature axes.")
    if prediction.shape[-1] != V1_OUTPUT_DIM:
        raise ValueError("prediction must end with the 192D V1 output dimension.")
    if not bool(torch.isfinite(prediction.detach()).all()):
        raise ValueError("prediction must contain only finite values.")
    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    if prediction.shape != task.shape:
        raise ValueError("task must have the same shape as prediction.")
    if prediction.device != task.device:
        raise ValueError("prediction and task must share a device.")
    return (prediction.float() * task.float()).sum(dim=-1)


def _normalize_terminal_v1(
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


def tdjepa_successor_td_target_v1(
    next_state: torch.Tensor,
    target_next_prediction: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> torch.Tensor:
    """Build ``s_next + gamma * (1-terminal) * target_prediction``."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    _validate_floating_vector("next_state", next_state, final_dim=V1_STATE_DIM)
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

    terminal_bool = _normalize_terminal_v1(
        terminal,
        leading_shape=next_state.shape[:-1],
        device=next_state.device,
    )
    continuation = (~terminal_bool).to(dtype=torch.float32).unsqueeze(-1)
    discounted_bootstrap = (
        float(gamma) * continuation * target_next_prediction.detach().float()
    )
    return next_state.detach().float() + discounted_bootstrap


@dataclass(frozen=True)
class TDJEPATDBatchV1:
    """Aligned V1 prediction, detached TD target, and transition losses."""

    prediction: torch.Tensor
    target: torch.Tensor
    per_transition_td_loss: torch.Tensor
    td_loss: torch.Tensor
    task: torch.Tensor
    terminal: torch.Tensor


def _validate_predictor_pair_v1(
    online: ActorFreeTDJEPAPredictorV1,
    target: ActorFreeTDJEPAPredictorV1,
) -> None:
    if not isinstance(online, ActorFreeTDJEPAPredictorV1):
        raise TypeError("online must be an ActorFreeTDJEPAPredictorV1.")
    if not isinstance(target, ActorFreeTDJEPAPredictorV1):
        raise TypeError("target must be an ActorFreeTDJEPAPredictorV1.")
    if online is target:
        raise ValueError("online and target must be distinct predictor instances.")
    architecture = ("hidden_dim", "hidden_layers", "embedding_layers")
    if any(getattr(online, name) != getattr(target, name) for name in architecture):
        raise ValueError("online and target predictors must share an architecture.")
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("target predictor parameters must be frozen.")


def build_tdjepa_td_batch_v1(
    online: ActorFreeTDJEPAPredictorV1,
    target: ActorFreeTDJEPAPredictorV1,
    state: torch.Tensor,
    action_embedding: torch.Tensor,
    task: torch.Tensor,
    next_state: torch.Tensor,
    next_action_embedding: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> TDJEPATDBatchV1:
    """Build V1 TD using already-encoded current and dataset-next actions.

    The encoder stays outside this function so online and target predictors
    share exactly the world model's one frozen action encoder without
    registering or checkpointing it again.
    """

    _validate_predictor_pair_v1(online, target)
    _validate_floating_vector("state", state, final_dim=V1_STATE_DIM)
    _validate_floating_vector(
        "action_embedding",
        action_embedding,
        final_dim=V1_ACTION_EMBEDDING_DIM,
    )
    _validate_floating_vector("task", task, final_dim=V1_TASK_DIM)
    _validate_floating_vector("next_state", next_state, final_dim=V1_STATE_DIM)
    _validate_floating_vector(
        "next_action_embedding",
        next_action_embedding,
        final_dim=V1_ACTION_EMBEDDING_DIM,
    )
    for other_name, other in (
        ("action_embedding", action_embedding),
        ("task", task),
        ("next_state", next_state),
        ("next_action_embedding", next_action_embedding),
    ):
        _require_matching_tensor_context("state", state, other_name, other)

    terminal_bool = _normalize_terminal_v1(
        terminal,
        leading_shape=state.shape[:-1],
        device=state.device,
    )
    frozen_state = state.detach()
    frozen_action_embedding = action_embedding.detach()
    frozen_task = task.detach()
    frozen_next_state = next_state.detach()
    frozen_next_action_embedding = next_action_embedding.detach()

    prediction = online(frozen_state, frozen_action_embedding, frozen_task)
    with torch.no_grad():
        target_next_prediction = target(
            frozen_next_state,
            frozen_next_action_embedding,
            frozen_task,
        )
        td_target = tdjepa_successor_td_target_v1(
            frozen_next_state,
            target_next_prediction,
            gamma=gamma,
            terminal=terminal_bool,
        )

    per_transition = (prediction.float() - td_target.float()).square().sum(dim=-1)
    if not bool(torch.isfinite(per_transition.detach()).all()):
        raise FloatingPointError("V1 TD loss produced a non-finite value.")
    return TDJEPATDBatchV1(
        prediction=prediction,
        target=td_target,
        per_transition_td_loss=per_transition,
        td_loss=per_transition.mean(),
        task=frozen_task,
        terminal=terminal_bool,
    )


@torch.no_grad()
def ema_update_target_v1(
    target: ActorFreeTDJEPAPredictorV1,
    online: ActorFreeTDJEPAPredictorV1,
    *,
    decay: float,
) -> None:
    """Update a frozen V1 target as ``decay*target + (1-decay)*online``."""

    _validate_predictor_pair_v1(online, target)
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
    "ActorFreeTDJEPAPredictorV1",
    "MixedTasksV1",
    "TDJEPATDBatchV1",
    "V1_ACTION_DIM",
    "V1_ACTION_EMBEDDING_DIM",
    "V1_OUTPUT_DIM",
    "V1_RAW_ACTION_DIM",
    "V1_STATE_DIM",
    "V1_TASK_DIM",
    "build_tdjepa_td_batch_v1",
    "ema_update_target_v1",
    "encode_frozen_action_blocks_v1",
    "project_tasks_to_sphere_v1",
    "sample_mixed_tasks_v1",
    "tdjepa_goal_score_v1",
    "tdjepa_successor_td_target_v1",
    "validate_frozen_lewm_action_encoder_v1",
]
