"""Planner-aligned extension of the frozen V1-C successor objective.

C2 keeps the complete V1-C TD and goal-projection objective.  Its only new
training signal asks the online successor predictor to rank CEM-like first
actions in the same order as the frozen V1 LeWM terminal goal cost.  The
teacher path is fully detached, no Actor is introduced, and only ``G`` is
optimized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from tdwm.methods.actor_free_td_lewm_v1 import (
    V1_RAW_ACTION_DIM,
    V1_STATE_DIM,
    V1_TASK_DIM,
    ActorFreeTDJEPAPredictorV1,
    encode_frozen_action_blocks_v1,
    tdjepa_goal_score_v1,
    validate_frozen_lewm_action_encoder_v1,
)

C2_OBJECTIVE_VERSION = 1


@dataclass(frozen=True)
class FirstQAlignmentV1C2Output:
    """C2 ranking loss and detached diagnostics for one minibatch."""

    loss: torch.Tensor
    per_example_loss: torch.Tensor
    teacher_cost: torch.Tensor
    student_score: torch.Tensor
    teacher_probability: torch.Tensor
    student_probability: torch.Tensor
    eligible_mask: torch.Tensor
    top1_agreement: torch.Tensor


def sample_first_q_candidates_v1_c2(
    reference: torch.Tensor,
    *,
    candidate_count: int,
    rollout_horizon: int,
    initial_variance: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the exact zero-mean first-iteration CEM distribution.

    Stable World Model 0.1.1 initializes CEM with zero mean and unit variance,
    samples independent Gaussian normalized action blocks, and forces sample
    zero to the current mean.  C2 mirrors that distribution without an Actor.
    The dedicated CPU generator makes the stream checkpointable and independent
    of CUDA/dropout RNG.
    """

    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor.")
    if reference.ndim < 1 or reference.shape[-1] != V1_STATE_DIM:
        raise ValueError("reference must end with the 192D state dimension.")
    if not reference.is_floating_point():
        raise TypeError("reference must be floating point.")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference must contain only finite values.")
    if isinstance(candidate_count, bool) or int(candidate_count) < 2:
        raise ValueError("candidate_count must be an integer of at least two.")
    if isinstance(rollout_horizon, bool) or int(rollout_horizon) <= 0:
        raise ValueError("rollout_horizon must be a positive integer.")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator.")
    if torch.device(generator.device).type != "cpu":
        raise ValueError("The C2 candidate generator must be a CPU generator.")
    variance = float(initial_variance)
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("initial_variance must be finite and positive.")

    batch = int(reference.shape[0])
    candidates = torch.randn(
        batch,
        int(candidate_count),
        int(rollout_horizon),
        V1_RAW_ACTION_DIM,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(
        device=reference.device,
        dtype=reference.dtype,
        non_blocking=True,
    )
    candidates.mul_(variance)
    candidates[:, 0].zero_()
    return candidates.detach()


def rollout_frozen_lewm_candidates_v1_c2(
    world_model: Any,
    state_history: torch.Tensor,
    raw_action_history: torch.Tensor,
    candidate_action_sequences: torch.Tensor,
) -> torch.Tensor:
    """Roll every candidate through frozen V1 LeWM and return its terminal latent."""

    if any(parameter.requires_grad for parameter in world_model.parameters()):
        raise ValueError("C2 requires a completely frozen LeWM teacher.")
    if any(module.training for module in world_model.modules()):
        raise ValueError("C2 requires the LeWM teacher to remain in eval mode.")
    action_encoder = getattr(world_model, "action_encoder", None)
    validate_frozen_lewm_action_encoder_v1(action_encoder)
    if state_history.ndim != 3 or state_history.shape[-1] != V1_STATE_DIM:
        raise ValueError("state_history must have shape [batch,history,192].")
    if (
        raw_action_history.ndim != 3
        or raw_action_history.shape[-1] != V1_RAW_ACTION_DIM
    ):
        raise ValueError("raw_action_history must have shape [batch,history,25].")
    if (
        candidate_action_sequences.ndim != 4
        or candidate_action_sequences.shape[-1] != V1_RAW_ACTION_DIM
    ):
        raise ValueError(
            "candidate_action_sequences must have shape [batch,candidates,horizon,25]."
        )
    batch, candidates, horizon, _ = candidate_action_sequences.shape
    if horizon <= 0:
        raise ValueError("C2 alignment horizon must be positive.")
    if raw_action_history.shape[1] + 1 != state_history.shape[1]:
        raise ValueError(
            "C2 requires two previous actions for its three-state history."
        )
    if state_history.shape[0] != batch:
        raise ValueError("Histories and candidates must share the batch axis.")
    if state_history.shape[1] <= 0:
        raise ValueError("C2 requires a non-empty LeWM history.")
    reference = candidate_action_sequences
    for name, value in (
        ("state_history", state_history),
        ("raw_action_history", raw_action_history),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point.")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(f"{name} must match candidate tensor device and dtype.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values.")

    history_size = int(state_history.shape[1])
    history = (
        state_history.detach()
        .unsqueeze(1)
        .expand(batch, candidates, history_size, V1_STATE_DIM)
    )
    previous_actions = (
        raw_action_history.detach()
        .unsqueeze(1)
        .expand(batch, candidates, history_size - 1, V1_RAW_ACTION_DIM)
    )
    action_blocks = torch.cat(
        (previous_actions, candidate_action_sequences.detach()), dim=2
    )

    with torch.no_grad():
        rollout_info = world_model.rollout(
            {
                "pixels": state_history.new_empty(batch, candidates, history_size, 1),
                "emb": history,
            },
            action_blocks,
            history_size=history_size,
        )
        predicted = rollout_info.get("predicted_emb")
    expected_time = history_size + horizon
    if not isinstance(predicted, torch.Tensor) or predicted.shape != (
        batch,
        candidates,
        expected_time,
        V1_STATE_DIM,
    ):
        raise ValueError(
            "Frozen LeWM rollout must return the observed history plus every "
            "candidate future state."
        )
    if not bool(torch.isfinite(predicted).all()):
        raise FloatingPointError("Frozen LeWM candidate rollout became non-finite.")
    return predicted[:, :, -1, :].detach()


def _candidate_zscore(values: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    variance = centered.square().mean(dim=1, keepdim=True)
    return centered * torch.rsqrt(variance + float(epsilon) ** 2)


def first_q_alignment_v1_c2_loss(
    predictor: ActorFreeTDJEPAPredictorV1,
    world_model: Any,
    state_history: torch.Tensor,
    raw_action_history: torch.Tensor,
    candidate_action_sequences: torch.Tensor,
    goal_latent: torch.Tensor,
    task: torch.Tensor,
    goal_mask: torch.Tensor,
    rollout_valid: torch.Tensor,
    *,
    teacher_temperature: float,
    student_temperature: float,
    standardization_epsilon: float,
) -> FirstQAlignmentV1C2Output:
    """Match First-Q candidate rankings to frozen-F terminal goal rankings."""

    if not isinstance(predictor, ActorFreeTDJEPAPredictorV1):
        raise TypeError("predictor must be ActorFreeTDJEPAPredictorV1.")
    batch, candidates = candidate_action_sequences.shape[:2]
    if goal_latent.shape != (batch, V1_STATE_DIM):
        raise ValueError("goal_latent must have shape [batch,192].")
    if task.shape != (batch, V1_TASK_DIM):
        raise ValueError("task must have shape [batch,192].")
    for name, value in (("goal_mask", goal_mask), ("rollout_valid", rollout_valid)):
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [batch] tensor.")
    if goal_latent.device != candidate_action_sequences.device:
        raise ValueError("goal_latent and candidates must share a device.")
    if task.device != candidate_action_sequences.device:
        raise ValueError("task and candidates must share a device.")
    teacher_tau = float(teacher_temperature)
    student_tau = float(student_temperature)
    epsilon = float(standardization_epsilon)
    for name, value in (
        ("teacher_temperature", teacher_tau),
        ("student_temperature", student_tau),
        ("standardization_epsilon", epsilon),
    ):
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    terminal = rollout_frozen_lewm_candidates_v1_c2(
        world_model,
        state_history,
        raw_action_history,
        candidate_action_sequences,
    )
    with torch.no_grad():
        teacher_cost = (
            (terminal.float() - goal_latent.detach().float().unsqueeze(1))
            .square()
            .sum(dim=-1)
        )
        teacher_logits = -_candidate_zscore(teacher_cost, epsilon=epsilon) / teacher_tau
        teacher_probability = torch.softmax(teacher_logits, dim=1)

    current = (
        state_history[:, -1, :].unsqueeze(1).expand(batch, candidates, V1_STATE_DIM)
    )
    first_action = candidate_action_sequences[:, :, 0, :]
    candidate_tasks = task.detach().unsqueeze(1).expand(batch, candidates, V1_TASK_DIM)
    first_action_embedding = encode_frozen_action_blocks_v1(
        world_model.action_encoder,
        first_action,
        reference=current,
    )
    prediction = predictor(current.detach(), first_action_embedding, candidate_tasks)
    if prediction.shape != current.shape:
        raise ValueError("C2 First-Q prediction shape is misaligned.")
    student_score = tdjepa_goal_score_v1(prediction, candidate_tasks)
    student_logits = (
        _candidate_zscore(student_score.float(), epsilon=epsilon) / student_tau
    )
    student_log_probability = torch.log_softmax(student_logits, dim=1)
    student_probability = student_log_probability.exp()
    per_example = -(teacher_probability.detach() * student_log_probability).sum(dim=1)
    eligible = goal_mask.to(device=per_example.device) & rollout_valid.to(
        device=per_example.device
    )
    if bool(eligible.any()):
        loss = per_example[eligible].mean()
        with torch.no_grad():
            agreement = (
                (
                    teacher_probability[eligible].argmax(dim=1)
                    == student_probability[eligible].argmax(dim=1)
                )
                .float()
                .mean()
            )
    else:
        loss = student_score.sum() * 0.0
        agreement = loss.detach()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("C2 First-Q alignment loss became non-finite.")
    return FirstQAlignmentV1C2Output(
        loss=loss,
        per_example_loss=per_example.detach(),
        teacher_cost=teacher_cost.detach(),
        student_score=student_score.detach(),
        teacher_probability=teacher_probability.detach(),
        student_probability=student_probability.detach(),
        eligible_mask=eligible.detach(),
        top1_agreement=agreement.detach(),
    )


__all__ = [
    "C2_OBJECTIVE_VERSION",
    "FirstQAlignmentV1C2Output",
    "first_q_alignment_v1_c2_loss",
    "rollout_frozen_lewm_candidates_v1_c2",
    "sample_first_q_candidates_v1_c2",
]
