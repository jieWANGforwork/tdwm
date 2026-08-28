"""Goal-free, actor-free TD successor features for LeWM.

The learned successor is conditioned on an observed/predicted latent history and
an externally supplied action.  Goals never enter this module: they are linear
queries of the learned successor features at planning time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from tdwm.methods.successor_geometry import successor_feature_basis

ActorFreeTDVariant = Literal[
    "parallel_real",
    "serial_decoupled",
    "serial_coupled",
    "hybrid",
]
SUPPORTED_VARIANTS = frozenset(
    {"parallel_real", "serial_decoupled", "serial_coupled", "hybrid"}
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
    predicted-context branch.
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

    previous_actions = _previous_action_histories(
        actions,
        history_size=history_size,
        current_count=current_count,
        shift=history_shift,
    )
    current_actions = actions[:, first_current:-1]
    with torch.no_grad():
        ema_latents = real_ema_latents.detach()
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

    if variant in {"parallel_real", "hybrid"}:
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
    elif variant == "hybrid":
        assert real_td_loss is not None
        td_loss = predicted_td_loss + real_td_loss
    else:
        td_loss = predicted_td_loss

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
    "actor_free_successor_td_target",
    "actor_free_td_objective",
    "ema_update",
]
