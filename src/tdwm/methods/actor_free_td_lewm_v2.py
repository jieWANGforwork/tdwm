"""V2 coupled-Hybrid TD-JEPA method primitives.

V2 keeps the V1 successor predictor and its 192D state/action/task contract,
but evaluates that *one shared predictor* on two online contexts:

``real_state``
    The state embedding produced by the online LeWM encoder.

``predicted_state``
    The state embedding produced by the online LeWM predictor.  This tensor is
    deliberately not detached, so its TD loss can update the coupled world
    model during V2 fine-tuning.

Both branches share one detached EMA target built from the EMA next state, the
EMA-encoded dataset next action, and the frozen EMA copy of the V1 predictor.
Action encoding performed by the normal online/EMA LeWM forward pass therefore
stays outside :func:`build_hybrid_tdjepa_td_batch_v2`.  The separate
:func:`encode_trainable_action_blocks_v2` helper is for raw-action paths such
as neighbour and prefix candidates, where V2 must retain gradients through the
online LeWM action encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_ACTION_EMBEDDING_DIM,
    V1_OUTPUT_DIM,
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    V1_TASK_DIM,
    ActorFreeTDJEPAPredictorV1,
    tdjepa_successor_td_target_v1,
)

V2_STATE_DIM = V1_STATE_DIM
V2_TASK_DIM = V1_TASK_DIM
V2_RAW_ACTION_DIM = V1_RAW_ACTION_DIM
V2_ACTION_EMBEDDING_DIM = V1_ACTION_EMBEDDING_DIM
V2_ACTION_DIM = V2_ACTION_EMBEDDING_DIM
V2_OUTPUT_DIM = V1_OUTPUT_DIM


def _validate_floating_vector_v2(
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


def _require_matching_tensor_context_v2(
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


def validate_lewm_action_encoder_v2(action_encoder: nn.Module) -> None:
    """Validate the LeWM 25-to-192 action interface without freezing it.

    Unlike V1's frozen helper, this check deliberately accepts train/eval mode
    and any ``requires_grad`` setting.  The caller controls whether it is the
    online, trainable action encoder or an EMA encoder used under ``no_grad``.
    """

    if not isinstance(action_encoder, nn.Module):
        raise TypeError("action_encoder must be a torch.nn.Module.")
    if getattr(action_encoder, "input_dim", None) != V2_RAW_ACTION_DIM:
        raise ValueError(
            "action_encoder.input_dim must match the normalized 25D action block."
        )
    if getattr(action_encoder, "emb_dim", None) != V2_ACTION_EMBEDDING_DIM:
        raise ValueError("action_encoder.emb_dim must be the 192D LeWM embedding.")


def encode_trainable_action_blocks_v2(
    action_encoder: nn.Module,
    raw_action: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Encode raw 25D action blocks while retaining the online gradient path.

    LeWM's action encoder consumes ``[batch, time, 25]``.  Arbitrary leading
    axes (including no leading axis) are flattened to ``[-1, 1, 25]`` and then
    restored after an exact ``[-1, 1, 192]`` output.  No input, output, or
    encoder parameter is detached, and this helper does not enter
    ``torch.no_grad``.  The final device/dtype alignment to ``reference`` is a
    differentiable cast.
    """

    validate_lewm_action_encoder_v2(action_encoder)
    _validate_floating_vector_v2(
        "raw_action",
        raw_action,
        final_dim=V2_RAW_ACTION_DIM,
        minimum_ndim=1,
    )
    _validate_floating_vector_v2(
        "reference",
        reference,
        final_dim=V2_STATE_DIM,
        minimum_ndim=1,
    )
    if raw_action.shape[:-1] != reference.shape[:-1]:
        raise ValueError(
            "raw_action and reference must have identical leading axes."
        )
    if raw_action.device != reference.device:
        raise ValueError("raw_action and reference must share a device.")

    leading_shape = raw_action.shape[:-1]
    flat_raw_action = raw_action.reshape(-1, 1, V2_RAW_ACTION_DIM)
    flat_embedding = action_encoder(flat_raw_action)
    if not isinstance(flat_embedding, torch.Tensor):
        raise TypeError("action_encoder must return a torch.Tensor.")
    expected_shape = (
        flat_raw_action.shape[0],
        1,
        V2_ACTION_EMBEDDING_DIM,
    )
    if flat_embedding.shape != expected_shape:
        raise ValueError(
            "action_encoder must return shape "
            f"{expected_shape}, found {tuple(flat_embedding.shape)}."
        )
    if not flat_embedding.is_floating_point():
        raise TypeError("action_encoder output must have a floating-point dtype.")
    if flat_embedding.device != reference.device:
        raise ValueError("action_encoder output and reference must share a device.")
    if not bool(torch.isfinite(flat_embedding.detach()).all()):
        raise ValueError("action_encoder output must contain only finite values.")

    embedding = flat_embedding.reshape(
        *leading_shape,
        V2_ACTION_EMBEDDING_DIM,
    )
    return embedding.to(device=reference.device, dtype=reference.dtype)


def _validate_predictor_pair_v2(
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
    if any(module.training for module in target.modules()):
        raise ValueError("target predictor and all its submodules must be in eval mode.")


@dataclass(frozen=True)
class HybridTDJEPATDBatchV2:
    """Two online V2 branches, one detached EMA target, and summed losses."""

    real_prediction: torch.Tensor
    predicted_prediction: torch.Tensor
    target: torch.Tensor
    real_per_transition_td_loss: torch.Tensor
    predicted_per_transition_td_loss: torch.Tensor
    per_transition_td_loss: torch.Tensor
    real_td_loss: torch.Tensor
    predicted_td_loss: torch.Tensor
    hybrid_td_loss: torch.Tensor
    task: torch.Tensor
    terminal: torch.Tensor


def _normalize_terminal_v2(
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


def build_hybrid_tdjepa_td_batch_v2(
    online: ActorFreeTDJEPAPredictorV1,
    target: ActorFreeTDJEPAPredictorV1,
    real_state: torch.Tensor,
    predicted_state: torch.Tensor,
    action_embedding: torch.Tensor,
    task: torch.Tensor,
    ema_next_state: torch.Tensor,
    target_next_action_embedding: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> HybridTDJEPATDBatchV2:
    """Build the V2 real/predicted coupled-Hybrid TD batch.

    ``action_embedding`` is the current action embedding already returned by
    the online LeWM forward pass.  It is shared by both online calls and is not
    detached, allowing both losses to update the online action encoder.
    ``target_next_action_embedding`` is the dataset-next action embedding from
    the EMA LeWM action encoder.  It is used only on the detached target side.

    ``real_state`` and ``predicted_state`` are likewise passed to the online G
    unchanged.  In particular, this function never detaches
    ``predicted_state``; when it came from the online LeWM predictor, the
    predicted-branch TD gradient remains coupled to that predictor.  Constant
    or no-grad states remain valid for evaluation and synthetic tests.

    The two branches share

    ``Y = ema_next_state + gamma * (1-terminal) * target_G(...)``

    and the V2 per-transition loss is the *sum* of their feature-summed MSEs.
    """

    _validate_predictor_pair_v2(online, target)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")

    for name, value, final_dim in (
        ("real_state", real_state, V2_STATE_DIM),
        ("predicted_state", predicted_state, V2_STATE_DIM),
        ("action_embedding", action_embedding, V2_ACTION_EMBEDDING_DIM),
        ("task", task, V2_TASK_DIM),
        ("ema_next_state", ema_next_state, V2_STATE_DIM),
        (
            "target_next_action_embedding",
            target_next_action_embedding,
            V2_ACTION_EMBEDDING_DIM,
        ),
    ):
        _validate_floating_vector_v2(name, value, final_dim=final_dim)
    for other_name, other in (
        ("predicted_state", predicted_state),
        ("action_embedding", action_embedding),
        ("task", task),
        ("ema_next_state", ema_next_state),
        ("target_next_action_embedding", target_next_action_embedding),
    ):
        _require_matching_tensor_context_v2(
            "real_state",
            real_state,
            other_name,
            other,
        )

    terminal_bool = _normalize_terminal_v2(
        terminal,
        leading_shape=real_state.shape[:-1],
        device=real_state.device,
    )
    detached_task = task.detach()

    # One shared online G and one shared online action embedding are evaluated
    # on the real and predicted states.  Neither state nor action is detached.
    real_prediction = online(real_state, action_embedding, detached_task)
    predicted_prediction = online(
        predicted_state,
        action_embedding,
        detached_task,
    )

    with torch.no_grad():
        detached_ema_next_state = ema_next_state.detach()
        detached_target_next_action = target_next_action_embedding.detach()
        target_next_prediction = target(
            detached_ema_next_state,
            detached_target_next_action,
            detached_task,
        )
        td_target = tdjepa_successor_td_target_v1(
            detached_ema_next_state,
            target_next_prediction,
            gamma=gamma,
            terminal=terminal_bool,
        )

    real_per_transition = (
        real_prediction.float() - td_target.float()
    ).square().sum(dim=-1)
    predicted_per_transition = (
        predicted_prediction.float() - td_target.float()
    ).square().sum(dim=-1)
    per_transition = real_per_transition + predicted_per_transition
    if not bool(torch.isfinite(per_transition.detach()).all()):
        raise FloatingPointError("V2 Hybrid TD loss produced a non-finite value.")

    return HybridTDJEPATDBatchV2(
        real_prediction=real_prediction,
        predicted_prediction=predicted_prediction,
        target=td_target,
        real_per_transition_td_loss=real_per_transition,
        predicted_per_transition_td_loss=predicted_per_transition,
        per_transition_td_loss=per_transition,
        real_td_loss=real_per_transition.mean(),
        predicted_td_loss=predicted_per_transition.mean(),
        hybrid_td_loss=per_transition.mean(),
        task=detached_task,
        terminal=terminal_bool,
    )


__all__ = [
    "HybridTDJEPATDBatchV2",
    "V2_ACTION_DIM",
    "V2_ACTION_EMBEDDING_DIM",
    "V2_OUTPUT_DIM",
    "V2_RAW_ACTION_DIM",
    "V2_STATE_DIM",
    "V2_TASK_DIM",
    "build_hybrid_tdjepa_td_batch_v2",
    "encode_trainable_action_blocks_v2",
    "validate_lewm_action_encoder_v2",
]
