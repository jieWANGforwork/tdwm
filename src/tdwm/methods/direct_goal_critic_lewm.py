"""Direct goal-conditioned scalar TD critic for Actor-Free TD-LeWM.

Unlike successor-feature variants, this head receives the hindsight/planning
goal directly and predicts one normalized discounted latent goal cost.  The
Hybrid objective shares one critic between an online real-latent branch and a
coupled teacher-forced predicted-latent branch; its bootstrap always uses real
EMA latents and the recorded dataset next action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from tdwm.methods.actor_free_td_lewm import actor_free_goal_future_offset_limits
from tdwm.methods.successor_geometry import latent_goal_cost


class DirectGoalCriticHead(nn.Module):
    """Predict ``C(history, current_action, goal)`` as a scalar future cost."""

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
            raise ValueError("Direct-critic dimensions must be positive.")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)

        # The previous actions join the latent history; current_action is the
        # action whose immediate next-state cost belongs to this Bellman value.
        # current-goal is a deterministic interaction feature, not a separate
        # factorized readout: the network still learns an unrestricted scalar.
        input_dim = (
            self.history_size * self.embed_dim
            + self.history_size * self.action_dim
            + 2 * self.embed_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        previous_actions: torch.Tensor,
        current_action: torch.Tensor,
        goal: torch.Tensor,
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
        if goal.shape != leading + (self.embed_dim,):
            raise ValueError("goal must match the history leading axes.")

        current_latent = latent_history[..., -1, :]
        inputs = torch.cat(
            (
                latent_history.flatten(start_dim=-2),
                previous_actions.flatten(start_dim=-2),
                current_action,
                goal,
                current_latent - goal,
            ),
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)

    def make_target(self) -> "DirectGoalCriticHead":
        """Return a frozen initial target-network copy."""

        return copy.deepcopy(self).requires_grad_(False)


@dataclass(frozen=True)
class DirectGoalCriticTDOutput:
    """Hybrid branch losses and Bellman-target diagnostics."""

    td_loss: torch.Tensor
    real_td_loss: torch.Tensor
    predicted_td_loss: torch.Tensor
    prediction_mean: torch.Tensor
    target_mean: torch.Tensor
    terminal_fraction: torch.Tensor
    negative_prediction_fraction: torch.Tensor
    pair_count: torch.Tensor


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


def _normalize_terminals(
    terminals: torch.Tensor | None, *, real_latents: torch.Tensor
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


def _validate_inputs(
    critic: DirectGoalCriticHead,
    target_critic: DirectGoalCriticHead,
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
    if real_latents.shape[-1] != critic.embed_dim:
        raise ValueError("Latents have the wrong embedding dimension.")
    if actions.shape[-1] != critic.action_dim:
        raise ValueError("Actions have the wrong action dimension.")
    for name in ("embed_dim", "action_dim", "history_size"):
        if getattr(critic, name) != getattr(target_critic, name):
            raise ValueError("Online and target critics must share dimensions.")
    if real_latents.shape[1] <= critic.history_size:
        raise ValueError("The clip must contain a full history and a next action.")


def _goal_context(
    real_ema_latents: torch.Tensor,
    terminal_mask: torch.Tensor,
    *,
    first_current_index: int,
    goal_offsets: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return goals, offsets, valid mask, denominator and valid transitions."""

    batch, time, embed_dim = real_ema_latents.shape
    current_count = time - first_current_index - 1
    limits = actor_free_goal_future_offset_limits(
        terminal_mask, first_current_index=first_current_index
    )
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
        valid = (offsets <= limits.unsqueeze(-1)) & limits.unsqueeze(-1).gt(0)
        denominator = limits.clamp_min(1).to(torch.float32)
    else:
        if goal_offsets.shape != (batch, current_count):
            raise ValueError(
                "goal_offsets must have shape (batch, aligned transitions)."
            )
        offsets_2d = goal_offsets.to(device=real_ema_latents.device, dtype=torch.int64)
        if torch.any(goal_offsets.to(device=real_ema_latents.device) != offsets_2d):
            raise ValueError("goal_offsets must contain integer values.")
        transition_valid = limits.gt(0)
        if torch.any(offsets_2d < 1) or torch.any(
            (offsets_2d > limits) & transition_valid
        ):
            raise ValueError("goal_offsets must select reachable future states.")
        offsets = offsets_2d.unsqueeze(-1)
        valid = transition_valid.unsqueeze(-1)
        denominator = torch.ones_like(limits, dtype=torch.float32)

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
        .float()
    )
    return goals, offsets, valid, denominator, limits.gt(0)


def _expand_for_goals(
    histories: torch.Tensor,
    previous_actions: torch.Tensor,
    current_actions: torch.Tensor,
    *,
    goal_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        histories.unsqueeze(2).expand(
            *histories.shape[:2], goal_count, *histories.shape[2:]
        ),
        previous_actions.unsqueeze(2).expand(
            *previous_actions.shape[:2], goal_count, *previous_actions.shape[2:]
        ),
        current_actions.unsqueeze(2).expand(
            *current_actions.shape[:2], goal_count, current_actions.shape[-1]
        ),
    )


def _branch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    denominator: torch.Tensor,
    transition_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_float = valid.to(torch.float32)
    transition_count = transition_valid.to(torch.float32).sum().clamp_min(1.0)
    loss = (
        ((prediction.float() - target).square() * valid_float).sum(dim=-1)
        / denominator
    ).sum() / transition_count
    prediction_mean = (
        (prediction.detach().float() * valid_float).sum(dim=-1) / denominator
    ).sum() / transition_count
    negative_fraction = (
        ((prediction.detach() < 0).to(torch.float32) * valid_float).sum(dim=-1)
        / denominator
    ).sum() / transition_count
    return loss, prediction_mean, negative_fraction


def direct_goal_critic_td_objective(
    critic: DirectGoalCriticHead,
    target_critic: DirectGoalCriticHead,
    real_latents: torch.Tensor,
    predicted_latents: torch.Tensor,
    real_ema_latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    gamma: float,
    terminals: torch.Tensor | None = None,
    first_current_index: int | None = None,
    goal_offsets: torch.Tensor | None = None,
) -> DirectGoalCriticTDOutput:
    """Train real and coupled predicted branches against one detached TD target.

    For every aligned transition and hindsight goal, the target is

    ``(1-gamma)c(z_ema[t+1], goal) + gamma(1-d) C_bar(H_ema[t+1], a[t+1], goal)``.

    ``d`` is true at a recorded episode boundary or when the real next state is
    the sampled hindsight goal.  Passing no offsets exactly enumerates the
    conditional-uniform future-goal expectation for deterministic validation.
    """

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1).")
    _validate_inputs(
        critic,
        target_critic,
        real_latents,
        predicted_latents,
        real_ema_latents,
        actions,
    )
    terminal_mask = _normalize_terminals(terminals, real_latents=real_latents)
    history_size = critic.history_size
    first_current = (
        history_size if first_current_index is None else int(first_current_index)
    )
    if first_current < history_size:
        raise ValueError(
            "first_current_index must be at least history_size so the latest "
            "predicted current latent is rollout-aligned."
        )
    if first_current >= real_latents.shape[1] - 1:
        raise ValueError("first_current_index must leave a next state and action.")

    current_count = real_latents.shape[1] - first_current - 1
    history_shift = first_current - history_size + 1
    previous_actions = _previous_action_histories(
        actions,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    current_actions = actions[:, first_current:-1]
    goals, offsets, valid, denominator, transition_valid = _goal_context(
        real_ema_latents,
        terminal_mask,
        first_current_index=first_current,
        goal_offsets=goal_offsets,
    )
    goal_count = goals.shape[2]

    with torch.no_grad():
        ema_latents = real_ema_latents.detach()
        next_histories = _latent_histories(
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
        dataset_next_action = actions[:, first_current + 1 :].detach()
        expanded_next = _expand_for_goals(
            next_histories,
            next_previous_actions,
            dataset_next_action,
            goal_count=goal_count,
        )
        bootstrap = target_critic(*expanded_next, goals)
        next_latent = ema_latents[:, first_current + 1 :].float()
        immediate = (1.0 - gamma) * latent_goal_cost(
            next_latent.unsqueeze(2), goals
        )
        aligned_terminal = terminal_mask[:, first_current:-1]
        continuation = (
            (~aligned_terminal).unsqueeze(-1) & offsets.gt(1) & valid
        )
        target = immediate + gamma * continuation.to(torch.float32) * bootstrap.float()

    predicted_histories = _latent_histories(
        predicted_latents,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    real_histories = _latent_histories(
        real_latents,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    expanded_previous = _expand_for_goals(
        predicted_histories,
        previous_actions,
        current_actions,
        goal_count=goal_count,
    )
    predicted = critic(*expanded_previous, goals)
    expanded_real = _expand_for_goals(
        real_histories,
        previous_actions,
        current_actions,
        goal_count=goal_count,
    )
    real = critic(*expanded_real, goals)

    predicted_loss, predicted_mean, predicted_negative = _branch_loss(
        predicted, target, valid, denominator, transition_valid
    )
    real_loss, real_mean, real_negative = _branch_loss(
        real, target, valid, denominator, transition_valid
    )
    transition_count = transition_valid.to(torch.float32).sum().clamp_min(1.0)
    valid_float = valid.to(torch.float32)
    target_mean = (
        (target.detach() * valid_float).sum(dim=-1) / denominator
    ).sum() / transition_count
    terminal = (~continuation) & valid
    terminal_fraction = (
        terminal.to(torch.float32).sum(dim=-1) / denominator
    ).sum() / transition_count
    return DirectGoalCriticTDOutput(
        td_loss=predicted_loss + real_loss,
        real_td_loss=real_loss,
        predicted_td_loss=predicted_loss,
        prediction_mean=0.5 * (predicted_mean + real_mean),
        target_mean=target_mean,
        terminal_fraction=terminal_fraction,
        negative_prediction_fraction=0.5 * (
            predicted_negative + real_negative
        ),
        pair_count=valid.sum().to(torch.float32),
    )


__all__ = [
    "DirectGoalCriticHead",
    "DirectGoalCriticTDOutput",
    "direct_goal_critic_td_objective",
]
