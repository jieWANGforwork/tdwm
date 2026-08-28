"""Goal-free, actor-free TD successor features for LeWM.

The learned successor is conditioned on an observed/predicted latent history and
an externally supplied action.  Goals never enter the successor head: they are
fixed linear queries of its output, optionally Bellman-trained by Goal Hybrid
and reused unchanged at planning time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from tdwm.methods.successor_geometry import (
    goal_cost_weights,
    latent_goal_cost,
    successor_feature_basis,
)

ActorFreeTDVariant = Literal[
    "parallel_real",
    "serial_decoupled",
    "serial_coupled",
    "hybrid",
    "goal_hybrid",
    "imaginary_hybrid",
]
SUPPORTED_VARIANTS = frozenset(
    {
        "parallel_real",
        "serial_decoupled",
        "serial_coupled",
        "hybrid",
        "goal_hybrid",
        "imaginary_hybrid",
    }
)


class ActorFreeSuccessorHead(nn.Module):
    """Predict goal-free successor features ``G(history, action)``.

    ``previous_actions`` contains the ``history_size - 1`` actions joining the
    latent history. ``current_action`` is the externally supplied dataset or
    planner action whose successor is being queried.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        history_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(embed_dim, action_dim, history_size, hidden_dim) <= 0:
            raise ValueError("Successor-head dimensions must be positive.")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = self.embed_dim + 2

        context_dim = (
            self.history_size * self.embed_dim + self.history_size * self.action_dim
        )
        self.network = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        previous_actions: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        expected_history = (self.history_size, self.embed_dim)
        if latent_history.shape[-2:] != expected_history:
            raise ValueError(
                "latent_history must end with "
                f"{expected_history}, found {latent_history.shape[-2:]}."
            )
        leading = latent_history.shape[:-2]
        expected_previous = (max(0, self.history_size - 1), self.action_dim)
        if previous_actions.shape != leading + expected_previous:
            raise ValueError(
                "previous_actions must have shape "
                f"{leading + expected_previous}, found {previous_actions.shape}."
            )
        if current_action.shape != leading + (self.action_dim,):
            raise ValueError("current_action must match the history leading axes.")

        inputs = torch.cat(
            (
                latent_history.flatten(start_dim=-2),
                previous_actions.flatten(start_dim=-2),
                current_action,
            ),
            dim=-1,
        )
        return self.network(inputs)

    def make_target(self) -> "ActorFreeSuccessorHead":
        """Return a frozen initial target-network copy."""

        return copy.deepcopy(self).requires_grad_(False)


@dataclass(frozen=True)
class ActorFreeTDOutput:
    """Losses and diagnostics returned by :func:`actor_free_td_objective`."""

    td_loss: torch.Tensor
    real_td_loss: torch.Tensor | None
    predicted_td_loss: torch.Tensor
    prediction_mean: torch.Tensor
    target_mean: torch.Tensor
    terminal_fraction: torch.Tensor
    pair_count: int
    goal_td_loss: torch.Tensor
    real_goal_td_loss: torch.Tensor | None
    predicted_goal_td_loss: torch.Tensor
    goal_prediction_mean: torch.Tensor
    goal_target_mean: torch.Tensor
    goal_terminal_fraction: torch.Tensor
    goal_negative_prediction_fraction: torch.Tensor
    goal_pair_count: torch.Tensor
    imaginary_next_mse: torch.Tensor


@dataclass(frozen=True)
class _GoalTDContext:
    """Detached hindsight-goal targets shared by the two online branches."""

    weights: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    denominator: torch.Tensor
    transition_valid: torch.Tensor
    terminal: torch.Tensor
    pair_count: torch.Tensor


def actor_free_successor_td_target(
    real_ema_next_latent: torch.Tensor,
    bootstrap: torch.Tensor,
    *,
    gamma: float,
    terminal: torch.Tensor | bool = False,
) -> torch.Tensor:
    """Return ``(1-gamma) chi(z') + gamma (1-terminal) G_target``."""

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    immediate = (1.0 - gamma) * successor_feature_basis(real_ema_next_latent.detach())
    if bootstrap.shape != immediate.shape:
        raise ValueError("bootstrap must match the lifted next-latent shape.")
    if isinstance(terminal, bool):
        continuation = 0.0 if terminal else 1.0
    else:
        if terminal.shape != immediate.shape[:-1]:
            raise ValueError("terminal must match the target leading axes.")
        terminal_bool = terminal.to(device=immediate.device, dtype=torch.bool)
        if torch.any(terminal.to(device=immediate.device) != terminal_bool):
            raise ValueError("terminal must contain only binary values.")
        continuation = (~terminal_bool).to(dtype=immediate.dtype).unsqueeze(-1)
    return immediate + gamma * continuation * bootstrap.detach()


def _latent_histories(
    latents: torch.Tensor, *, history_size: int, current_count: int, shift: int
) -> torch.Tensor:
    return torch.stack(
        [
            latents[:, shift + index : shift + index + history_size]
            for index in range(current_count)
        ],
        dim=1,
    )


def _previous_action_histories(
    actions: torch.Tensor, *, history_size: int, current_count: int, shift: int
) -> torch.Tensor:
    previous = history_size - 1
    if previous == 0:
        return actions.new_empty(actions.shape[0], current_count, 0, actions.shape[-1])
    return torch.stack(
        [
            actions[:, shift + index : shift + index + previous]
            for index in range(current_count)
        ],
        dim=1,
    )


def _validate_latent_inputs(
    successor: ActorFreeSuccessorHead,
    target_successor: ActorFreeSuccessorHead,
    real_latents: torch.Tensor,
    predicted_latents: torch.Tensor,
    real_ema_latents: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    if real_latents.ndim != 3:
        raise ValueError("real_latents must have shape (batch, time, latent_dim).")
    if predicted_latents.shape != real_latents.shape:
        raise ValueError("predicted_latents must match real_latents.")
    if real_ema_latents.shape != real_latents.shape:
        raise ValueError("real_ema_latents must match real_latents.")
    if actions.ndim != 3 or actions.shape[:2] != real_latents.shape[:2]:
        raise ValueError("actions must share the latent batch and time axes.")
    if real_latents.shape[-1] != successor.embed_dim:
        raise ValueError("Latents have the wrong embedding dimension.")
    if actions.shape[-1] != successor.action_dim:
        raise ValueError("Actions have the wrong action dimension.")
    dimensions = (
        "embed_dim",
        "action_dim",
        "history_size",
        "output_dim",
    )
    if any(
        getattr(successor, name) != getattr(target_successor, name)
        for name in dimensions
    ):
        raise ValueError("Online and target successors must share dimensions.")
    if real_latents.shape[1] <= successor.history_size:
        raise ValueError(
            "The clip must contain a full history and a dataset next action."
        )


def _normalize_terminals(
    terminals: torch.Tensor | None,
    *,
    real_latents: torch.Tensor,
) -> torch.Tensor:
    if terminals is None:
        return torch.zeros(
            real_latents.shape[:2],
            device=real_latents.device,
            dtype=torch.bool,
        )
    if terminals.shape != real_latents.shape[:2]:
        raise ValueError("terminals must have shape (batch, time).")
    terminal_bool = terminals.to(device=real_latents.device, dtype=torch.bool)
    if torch.any(terminals.to(device=real_latents.device) != terminal_bool):
        raise ValueError("terminals must contain only binary values.")
    return terminal_bool


def actor_free_goal_future_offset_limits(
    terminals: torch.Tensor,
    *,
    first_current_index: int,
) -> torch.Tensor:
    """Return each transition's largest reachable within-clip goal offset.

    A terminal transition may still use its real next state as the goal, but no
    goal may be sampled after that state.  In the normal episode-aware Cube
    clips a terminal can only occur at the final transition; handling it here
    also makes the boundary rule explicit and independently testable.
    """

    if terminals.ndim != 2:
        raise ValueError("terminals must have shape (batch, time).")
    first_current = int(first_current_index)
    if first_current < 0 or first_current >= terminals.shape[1] - 1:
        raise ValueError("first_current_index must leave at least one future state.")
    terminal_bool = terminals.to(dtype=torch.bool)
    if torch.any(terminals != terminal_bool):
        raise ValueError("terminals must contain only binary values.")

    current_count = terminals.shape[1] - first_current - 1
    aligned = terminal_bool[:, first_current:-1]
    limits: list[torch.Tensor] = []
    for index in range(current_count):
        remaining = aligned[:, index:]
        active = ~aligned[:, :index].any(dim=-1)
        has_terminal = remaining.any(dim=-1)
        first_terminal = remaining.to(dtype=torch.int64).argmax(dim=-1) + 1
        clip_limit = torch.full_like(first_terminal, current_count - index)
        reachable_limit = torch.where(has_terminal, first_terminal, clip_limit)
        limits.append(
            torch.where(active, reachable_limit, torch.zeros_like(clip_limit))
        )
    return torch.stack(limits, dim=1)


def sample_actor_free_goal_offsets(
    terminals: torch.Tensor,
    *,
    first_current_index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniformly sample one reachable future-goal offset per transition.

    Sampling happens on CPU with a dedicated generator so it neither consumes
    the model RNG (notably predictor dropout) nor depends on CUDA RNG state.
    """

    limits = actor_free_goal_future_offset_limits(
        terminals, first_current_index=first_current_index
    )
    uniform = torch.rand(limits.shape, generator=generator, device="cpu")
    sampled = (
        torch.floor(uniform * limits.clamp_min(1).detach().cpu()).to(torch.int64) + 1
    )
    return sampled.to(device=terminals.device)


@torch.no_grad()
def _build_goal_td_context(
    real_ema_latents: torch.Tensor,
    bootstrap: torch.Tensor,
    aligned_terminal: torch.Tensor,
    *,
    gamma: float,
    first_current_index: int,
    offset_limits: torch.Tensor,
    goal_offsets: torch.Tensor | None,
) -> _GoalTDContext:
    """Build sampled or exactly enumerated hindsight-goal Bellman targets."""

    batch, time, embed_dim = real_ema_latents.shape
    current_count = time - first_current_index - 1
    if offset_limits.shape != (batch, current_count):
        raise ValueError("offset_limits must match the aligned transitions.")

    if goal_offsets is None:
        offsets = (
            torch.arange(
                1,
                current_count + 1,
                device=real_ema_latents.device,
                dtype=torch.int64,
            )
            .reshape(1, 1, current_count)
            .expand(batch, current_count, -1)
        )
        valid = (offsets <= offset_limits.unsqueeze(-1)) & offset_limits.unsqueeze(
            -1
        ).gt(0)
        denominator = offset_limits.clamp_min(1).to(dtype=torch.float32)
    else:
        if goal_offsets.shape != (batch, current_count):
            raise ValueError(
                "goal_offsets must have shape (batch, aligned transitions)."
            )
        offsets_2d = goal_offsets.to(device=real_ema_latents.device, dtype=torch.int64)
        if torch.any(goal_offsets.to(device=real_ema_latents.device) != offsets_2d):
            raise ValueError("goal_offsets must contain integer values.")
        transition_valid = offset_limits.gt(0)
        if torch.any(offsets_2d < 1) or torch.any(
            (offsets_2d > offset_limits) & transition_valid
        ):
            raise ValueError("goal_offsets must select reachable future states.")
        offsets = offsets_2d.unsqueeze(-1)
        valid = transition_valid.unsqueeze(-1)
        denominator = torch.ones(
            (batch, current_count),
            device=real_ema_latents.device,
            dtype=torch.float32,
        )

    current_indices = torch.arange(
        first_current_index,
        first_current_index + current_count,
        device=real_ema_latents.device,
        dtype=torch.int64,
    ).reshape(1, current_count, 1)
    goal_indices = (current_indices + offsets).clamp_max(time - 1)
    flat_indices = goal_indices.reshape(batch, -1)
    goals = (
        real_ema_latents.detach()
        .gather(
            1,
            flat_indices.unsqueeze(-1).expand(batch, flat_indices.shape[1], embed_dim),
        )
        .reshape(batch, current_count, offsets.shape[-1], embed_dim)
    )

    # All scalar goal geometry is intentionally FP32 under bf16 mixed training.
    goals = goals.float()
    weights = goal_cost_weights(goals)
    next_latent = real_ema_latents[:, first_current_index + 1 :].detach().float()
    immediate = (1.0 - gamma) * latent_goal_cost(next_latent.unsqueeze(-2), goals)
    bootstrap_cost = (bootstrap.detach().float().unsqueeze(-2) * weights).sum(dim=-1)
    continuation = (~aligned_terminal).unsqueeze(-1) & offsets.gt(1) & valid
    target = immediate + gamma * continuation.to(torch.float32) * bootstrap_cost
    terminal = (~continuation) & valid
    return _GoalTDContext(
        weights=weights,
        target=target,
        valid=valid,
        denominator=denominator,
        transition_valid=offset_limits.gt(0),
        terminal=terminal,
        pair_count=valid.sum().to(dtype=torch.float32),
    )


def _goal_td_branch(
    prediction: torch.Tensor,
    context: _GoalTDContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return conditional-uniform goal loss and prediction diagnostics."""

    cost = (prediction.float().unsqueeze(-2) * context.weights).sum(dim=-1)
    valid = context.valid.to(dtype=torch.float32)
    squared_error = (cost - context.target).square()
    transition_count = context.transition_valid.to(torch.float32).sum().clamp_min(1.0)
    loss = (
        (squared_error * valid).sum(dim=-1) / context.denominator
    ).sum() / transition_count
    prediction_mean = (
        (cost.detach() * valid).sum(dim=-1) / context.denominator
    ).sum() / transition_count
    negative_fraction = (
        ((cost.detach() < 0.0).to(torch.float32) * valid).sum(dim=-1)
        / context.denominator
    ).sum() / transition_count
    return loss, prediction_mean, negative_fraction


def actor_free_td_objective(
    successor: ActorFreeSuccessorHead,
    target_successor: ActorFreeSuccessorHead,
    real_latents: torch.Tensor,
    predicted_latents: torch.Tensor,
    real_ema_latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    gamma: float,
    variant: ActorFreeTDVariant,
    terminals: torch.Tensor | None = None,
    first_current_index: int | None = None,
    goal_offsets: torch.Tensor | None = None,
    imagined_ema_next_latents: torch.Tensor | None = None,
) -> ActorFreeTDOutput:
    """Apply one-step TD to every aligned transition in a clip.

    The target side always uses detached real EMA latents and the *dataset next
    action*. Clip boundaries are not treated as terminals: the final usable
    pair bootstraps from ``actions[:, -1]`` unless its explicit episode-terminal
    mask is true.

    ``parallel_real`` trains only from the real online-encoder context and is
    parallel to the LeWM predictor. ``serial_decoupled`` stops the
    predicted-context gradient, ``serial_coupled`` lets it update the world
    predictor, and ``hybrid`` adds a real-context TD branch to the coupled
    predicted-context branch. ``goal_hybrid`` retains both hybrid successor-
    feature losses and additionally Bellman-trains their fixed linear goal
    readouts.  Passing ``goal_offsets`` samples one hindsight goal per aligned
    transition; omitting it computes the exact conditional-uniform expectation
    over every reachable future goal (used for deterministic validation).
    ``imaginary_hybrid`` keeps the real EMA next latent as the immediate feature
    but bootstraps the target successor from a history whose newest latent is
    predicted by the EMA LeWM predictor.
    """

    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported actor-free TD variant: {variant!r}.")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    _validate_latent_inputs(
        successor,
        target_successor,
        real_latents,
        predicted_latents,
        real_ema_latents,
        actions,
    )
    terminal_mask = _normalize_terminals(terminals, real_latents=real_latents)
    if variant != "goal_hybrid" and goal_offsets is not None:
        raise ValueError("goal_offsets are only valid for the goal_hybrid variant.")
    history_size = successor.history_size
    first_current = (
        history_size if first_current_index is None else int(first_current_index)
    )
    if first_current < history_size:
        raise ValueError(
            "first_current_index must be at least history_size so the latest "
            "predicted current latent is rollout-aligned."
        )
    if first_current >= real_latents.shape[1] - 1:
        raise ValueError(
            "first_current_index must leave a next state and dataset next action."
        )
    current_count = real_latents.shape[1] - first_current - 1
    history_shift = first_current - history_size + 1
    if variant == "imaginary_hybrid":
        expected_imagined = (
            real_latents.shape[0],
            current_count,
            real_latents.shape[-1],
        )
        if imagined_ema_next_latents is None:
            raise ValueError("imaginary_hybrid requires EMA-predicted next latents.")
        if imagined_ema_next_latents.shape != expected_imagined:
            raise ValueError(
                "imagined_ema_next_latents must have shape "
                f"{expected_imagined}, found {imagined_ema_next_latents.shape}."
            )
    elif imagined_ema_next_latents is not None:
        raise ValueError(
            "imagined_ema_next_latents are only valid for imaginary_hybrid."
        )

    previous_actions = _previous_action_histories(
        actions,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    current_actions = actions[:, first_current:-1]
    with torch.no_grad():
        ema_latents = real_ema_latents.detach()
        if variant == "imaginary_hybrid":
            current_ema_history = _latent_histories(
                ema_latents,
                history_size=history_size,
                current_count=current_count,
                shift=history_shift,
            )
            assert imagined_ema_next_latents is not None
            next_history = torch.cat(
                (
                    current_ema_history[..., 1:, :],
                    imagined_ema_next_latents.detach().unsqueeze(-2),
                ),
                dim=-2,
            )
        else:
            next_history = _latent_histories(
                ema_latents,
                history_size=history_size,
                current_count=current_count,
                shift=history_shift + 1,
            )
        next_previous_actions = _previous_action_histories(
            actions.detach(),
            history_size=history_size,
            current_count=current_count,
            shift=history_shift + 1,
        )
        # This is deliberately the recorded dataset action, never an actor.
        dataset_next_action = actions[:, first_current + 1 :].detach()
        bootstrap = target_successor(
            next_history,
            next_previous_actions,
            dataset_next_action,
        )
        next_latent = ema_latents[:, first_current + 1 :]
        aligned_terminal = terminal_mask[:, first_current:-1]
        target = actor_free_successor_td_target(
            next_latent,
            bootstrap,
            gamma=gamma,
            terminal=aligned_terminal,
        )

    predicted_td_loss = real_latents.new_zeros(())
    real_td_loss: torch.Tensor | None = None
    predictions: list[torch.Tensor] = []
    predicted: torch.Tensor | None = None
    real_prediction: torch.Tensor | None = None

    if variant != "parallel_real":
        predicted_history = _latent_histories(
            predicted_latents,
            history_size=history_size,
            current_count=current_count,
            shift=history_shift,
        )
        if variant == "serial_decoupled":
            predicted_history = predicted_history.detach()
        predicted = successor(
            predicted_history,
            previous_actions,
            current_actions,
        )
        predicted_td_loss = (predicted - target).square().mean()
        predictions.append(predicted.detach())

    if variant in {
        "parallel_real",
        "hybrid",
        "goal_hybrid",
        "imaginary_hybrid",
    }:
        real_history = _latent_histories(
            real_latents,
            history_size=history_size,
            current_count=current_count,
            shift=history_shift,
        )
        real_prediction = successor(
            real_history,
            previous_actions,
            current_actions,
        )
        real_td_loss = (real_prediction - target).square().mean()
        predictions.append(real_prediction.detach())

    if variant == "parallel_real":
        assert real_td_loss is not None
        td_loss = real_td_loss
    elif variant in {"hybrid", "goal_hybrid", "imaginary_hybrid"}:
        assert real_td_loss is not None
        td_loss = predicted_td_loss + real_td_loss
    else:
        td_loss = predicted_td_loss

    zero = real_latents.new_zeros((), dtype=torch.float32)
    goal_td_loss = zero
    predicted_goal_td_loss = zero
    real_goal_td_loss: torch.Tensor | None = None
    goal_prediction_mean = zero
    goal_target_mean = zero
    goal_terminal_fraction = zero
    goal_negative_prediction_fraction = zero
    goal_pair_count = zero
    imaginary_next_mse = zero
    if imagined_ema_next_latents is not None:
        imaginary_next_mse = (
            (
                imagined_ema_next_latents.detach().float()
                - ema_latents[:, first_current + 1 :].float()
            )
            .square()
            .mean()
        )
    if variant == "goal_hybrid":
        assert predicted is not None and real_prediction is not None
        offset_limits = actor_free_goal_future_offset_limits(
            terminal_mask, first_current_index=first_current
        )
        goal_context = _build_goal_td_context(
            ema_latents,
            bootstrap,
            aligned_terminal,
            gamma=gamma,
            first_current_index=first_current,
            offset_limits=offset_limits,
            goal_offsets=goal_offsets,
        )
        (
            predicted_goal_td_loss,
            predicted_goal_mean,
            predicted_negative_fraction,
        ) = _goal_td_branch(predicted, goal_context)
        (
            real_goal_td_loss,
            real_goal_mean,
            real_negative_fraction,
        ) = _goal_td_branch(real_prediction, goal_context)
        goal_td_loss = predicted_goal_td_loss + real_goal_td_loss
        goal_prediction_mean = 0.5 * (predicted_goal_mean + real_goal_mean)
        goal_transition_count = (
            goal_context.transition_valid.to(torch.float32).sum().clamp_min(1.0)
        )
        goal_target_mean = (
            (
                goal_context.target.detach()
                * goal_context.valid.to(dtype=torch.float32)
            ).sum(dim=-1)
            / goal_context.denominator
        ).sum() / goal_transition_count
        goal_terminal_fraction = (
            goal_context.terminal.to(dtype=torch.float32).sum(dim=-1)
            / goal_context.denominator
        ).sum() / goal_transition_count
        goal_negative_prediction_fraction = 0.5 * (
            predicted_negative_fraction + real_negative_fraction
        )
        goal_pair_count = goal_context.pair_count

    return ActorFreeTDOutput(
        td_loss=td_loss,
        real_td_loss=real_td_loss,
        predicted_td_loss=predicted_td_loss,
        prediction_mean=torch.cat(
            [item.reshape(-1, item.shape[-1]) for item in predictions], dim=0
        ).mean(),
        target_mean=target.detach().mean(),
        terminal_fraction=aligned_terminal.float().mean(),
        pair_count=int(real_latents.shape[0] * current_count),
        goal_td_loss=goal_td_loss,
        real_goal_td_loss=real_goal_td_loss,
        predicted_goal_td_loss=predicted_goal_td_loss,
        goal_prediction_mean=goal_prediction_mean,
        goal_target_mean=goal_target_mean,
        goal_terminal_fraction=goal_terminal_fraction,
        goal_negative_prediction_fraction=goal_negative_prediction_fraction,
        goal_pair_count=goal_pair_count,
        imaginary_next_mse=imaginary_next_mse,
    )


@torch.no_grad()
def ema_update(
    target: nn.Module,
    source: nn.Module,
    *,
    decay: float,
) -> None:
    """EMA-update target parameters and buffers from the online successor."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(
        target.buffers(), source.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
        else:
            target_buffer.copy_(source_buffer)


__all__ = [
    "ActorFreeSuccessorHead",
    "ActorFreeTDOutput",
    "ActorFreeTDVariant",
    "SUPPORTED_VARIANTS",
    "actor_free_goal_future_offset_limits",
    "actor_free_successor_td_target",
    "actor_free_td_objective",
    "ema_update",
    "sample_actor_free_goal_offsets",
]
