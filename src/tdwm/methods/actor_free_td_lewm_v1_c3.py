"""RP1-style goal-conditioned state critic for the V1-C3 ablation.

V1-C3 leaves the complete V1-C model frozen and trains only this critic.  The
critic estimates temporal *cost* to a goal, so smaller values are better.  Its
MRN quasimetric form is non-negative and makes ``V(g, g) == 0`` by construction::

    V(z, g) = ||u(z) - u(g)||_2 + max_j relu(v_j(g) - v_j(z)).

The cached LeWM data advances in five-primitive-action blocks.  Helpers in this
module make that conversion explicit and apply discounting in primitive-step
units, never in block units.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

V1C3_STATE_DIM = 192
V1C3_BLOCK_STEPS = 5
V1C3_BACKUP_PRIMITIVE_STEPS = 50


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_probability(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


@dataclass(frozen=True)
class V1C3Config:
    """Default RP1 state-critic architecture and optimization constants."""

    hidden_dim: int = 256
    embedding_dim: int = 128
    depth: int = 2
    gamma: float = 0.98
    expectile_tau: float = 0.03
    huber_beta: float = 1.0
    ema_rate: float = 0.005
    block_steps: int = V1C3_BLOCK_STEPS
    backup_primitive_steps: int = V1C3_BACKUP_PRIMITIVE_STEPS

    def __post_init__(self) -> None:
        hidden_dim = _positive_integer("hidden_dim", self.hidden_dim)
        embedding_dim = _positive_integer("embedding_dim", self.embedding_dim)
        depth = _positive_integer("depth", self.depth)
        block_steps = _positive_integer("block_steps", self.block_steps)
        backup_steps = _positive_integer(
            "backup_primitive_steps", self.backup_primitive_steps
        )
        if backup_steps % block_steps:
            raise ValueError(
                "backup_primitive_steps must be divisible by block_steps."
            )
        gamma = _finite_probability("gamma", self.gamma)
        tau = float(self.expectile_tau)
        if not math.isfinite(tau) or not 0.0 < tau < 0.5:
            raise ValueError(
                "expectile_tau must be finite and lie in (0, 0.5) for a "
                "cost critic."
            )
        beta = float(self.huber_beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("huber_beta must be finite and positive.")
        ema_rate = _finite_probability("ema_rate", self.ema_rate)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "embedding_dim", embedding_dim)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "expectile_tau", tau)
        object.__setattr__(self, "huber_beta", beta)
        object.__setattr__(self, "ema_rate", ema_rate)
        object.__setattr__(self, "block_steps", block_steps)
        object.__setattr__(self, "backup_primitive_steps", backup_steps)

    @property
    def backup_blocks(self) -> int:
        return self.backup_primitive_steps // self.block_steps


def _validate_latent(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if value.ndim < 2 or value.shape[-1] != V1C3_STATE_DIM:
        raise ValueError(f"{name} must have shape [..., {V1C3_STATE_DIM}].")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_latent_pair(
    reference_name: str,
    reference: torch.Tensor,
    other_name: str,
    other: torch.Tensor,
) -> None:
    _validate_latent(reference_name, reference)
    _validate_latent(other_name, other)
    if other.shape != reference.shape:
        raise ValueError(f"{other_name} must have the same shape as {reference_name}.")
    if other.device != reference.device or other.dtype != reference.dtype:
        raise ValueError(
            f"{other_name} must share the device and dtype of {reference_name}."
        )


class RP1StateValueV1C3(nn.Module):
    """MRN quasimetric over frozen 192D LeWM state latents."""

    state_dim = V1C3_STATE_DIM

    def __init__(
        self,
        *,
        state_dim: int = V1C3_STATE_DIM,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if state_dim != V1C3_STATE_DIM:
            raise ValueError(f"state_dim must be exactly {V1C3_STATE_DIM}.")
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.embedding_dim = _positive_integer("embedding_dim", embedding_dim)
        self.depth = _positive_integer("depth", depth)

        layers: list[nn.Module] = []
        input_dim = self.state_dim
        for _ in range(self.depth):
            layers.extend((nn.Linear(input_dim, self.hidden_dim), nn.ReLU()))
            input_dim = self.hidden_dim
        layers.append(nn.Linear(input_dim, 2 * self.embedding_dim))
        self.head = nn.Sequential(*layers)
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def components(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared head's Euclidean and directed embeddings."""

        _validate_latent("latent", latent)
        encoded = self.head(latent)
        expected_shape = (*latent.shape[:-1], 2 * self.embedding_dim)
        if encoded.shape != expected_shape:
            raise RuntimeError(
                f"MRN head returned {tuple(encoded.shape)}, expected {expected_shape}."
            )
        if not bool(torch.isfinite(encoded.detach()).all()):
            raise FloatingPointError("V1-C3 MRN head produced a non-finite embedding.")
        return encoded.split(self.embedding_dim, dim=-1)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        _validate_latent_pair("state", state, "goal", goal)
        state_u, state_v = self.components(state)
        goal_u, goal_v = self.components(goal)
        euclidean = torch.linalg.vector_norm(
            state_u.float() - goal_u.float(), ord=2, dim=-1
        )
        directed = torch.relu(goal_v.float() - state_v.float()).amax(dim=-1)
        value = euclidean + directed
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError("V1-C3 state critic produced a non-finite value.")
        if bool((value.detach() < 0.0).any()):
            raise FloatingPointError("V1-C3 state critic produced a negative cost.")
        return value

    def make_target(self) -> "RP1StateValueV1C3":
        """Return a distinct, frozen EMA copy of this critic."""

        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


@dataclass(frozen=True)
class PrimitiveStepWindowV1C3:
    """Same-episode goal and backup offsets expressed in primitive steps."""

    delta_primitive: torch.Tensor
    n_eff_primitive: torch.Tensor
    exact_mask: torch.Tensor


def _integer_tensor_like(
    name: str,
    value: torch.Tensor,
    *,
    shape: torch.Size,
    device: torch.device,
    allow_zero: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}.")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}.")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError(f"{name} must contain real integer-valued offsets.")
    if value.is_floating_point():
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f"{name} must contain only finite values.")
        if not bool((value.detach() == value.detach().round()).all()):
            raise ValueError(f"{name} must contain only integer-valued offsets.")
    converted = value.detach().to(dtype=torch.int64)
    minimum = 0 if allow_zero else 1
    if bool((converted < minimum).any()):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain only {qualifier} offsets.")
    return converted


def rp1_block_window_v1_c3(
    goal_delta_blocks: torch.Tensor,
    remaining_blocks: torch.Tensor,
    *,
    block_steps: int = V1C3_BLOCK_STEPS,
    backup_primitive_steps: int = V1C3_BACKUP_PRIMITIVE_STEPS,
) -> PrimitiveStepWindowV1C3:
    """Convert block offsets to the primitive-step RP1 backup window.

    ``goal_delta_blocks`` and ``remaining_blocks`` are measured from the same
    anchor.  Goals must be in that episode's remaining suffix.  The effective
    bootstrap horizon is capped at 50 primitive steps (ten five-step blocks by
    default).
    """

    block_steps = _positive_integer("block_steps", block_steps)
    backup_primitive_steps = _positive_integer(
        "backup_primitive_steps", backup_primitive_steps
    )
    if backup_primitive_steps % block_steps:
        raise ValueError("backup_primitive_steps must be divisible by block_steps.")
    if not isinstance(goal_delta_blocks, torch.Tensor):
        raise TypeError("goal_delta_blocks must be a torch.Tensor.")
    delta_blocks = _integer_tensor_like(
        "goal_delta_blocks",
        goal_delta_blocks,
        shape=goal_delta_blocks.shape,
        device=goal_delta_blocks.device,
        allow_zero=True,
    )
    remaining = _integer_tensor_like(
        "remaining_blocks",
        remaining_blocks,
        shape=goal_delta_blocks.shape,
        device=goal_delta_blocks.device,
        allow_zero=False,
    )
    if bool((delta_blocks > remaining).any()):
        raise ValueError("Every goal must lie inside the same episode suffix.")

    backup_blocks = backup_primitive_steps // block_steps
    n_eff_blocks = torch.clamp(remaining, max=backup_blocks)
    delta_primitive = delta_blocks * block_steps
    n_eff_primitive = n_eff_blocks * block_steps
    return PrimitiveStepWindowV1C3(
        delta_primitive=delta_primitive,
        n_eff_primitive=n_eff_primitive,
        exact_mask=delta_primitive <= n_eff_primitive,
    )


def discounted_primitive_cost_v1_c3(
    n_eff_primitive: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Return ``sum_{k=0}^{n_eff-1} gamma**k`` in primitive-step units."""

    gamma = _finite_probability("gamma", gamma)
    if not isinstance(n_eff_primitive, torch.Tensor):
        raise TypeError("n_eff_primitive must be a torch.Tensor.")
    steps = _integer_tensor_like(
        "n_eff_primitive",
        n_eff_primitive,
        shape=n_eff_primitive.shape,
        device=n_eff_primitive.device,
        allow_zero=True,
    )
    steps_float = steps.to(dtype=torch.float32)
    if gamma == 1.0:
        return steps_float
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32, device=steps.device)
    return (1.0 - torch.pow(gamma_tensor, steps_float)) / (1.0 - gamma)


@dataclass(frozen=True)
class RP1TDTargetV1C3:
    """Detached RP1 target and its primitive-step diagnostics."""

    target: torch.Tensor
    bootstrap_value: torch.Tensor
    prefix_cost: torch.Tensor
    delta_primitive: torch.Tensor
    n_eff_primitive: torch.Tensor
    exact_mask: torch.Tensor


def rp1_temporal_td_target_v1_c3(
    delta_primitive: torch.Tensor,
    n_eff_primitive: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float = 0.98,
    block_steps: int = V1C3_BLOCK_STEPS,
    backup_primitive_steps: int = V1C3_BACKUP_PRIMITIVE_STEPS,
) -> RP1TDTargetV1C3:
    """Build the exact-within-window or primitive-step bootstrap target."""

    gamma = _finite_probability("gamma", gamma)
    block_steps = _positive_integer("block_steps", block_steps)
    backup_primitive_steps = _positive_integer(
        "backup_primitive_steps", backup_primitive_steps
    )
    if backup_primitive_steps % block_steps:
        raise ValueError("backup_primitive_steps must be divisible by block_steps.")
    if not isinstance(bootstrap_value, torch.Tensor):
        raise TypeError("bootstrap_value must be a torch.Tensor.")
    if not bootstrap_value.is_floating_point():
        raise TypeError("bootstrap_value must have a floating-point dtype.")
    if bootstrap_value.ndim < 1 or bootstrap_value.numel() == 0:
        raise ValueError("bootstrap_value must be a non-empty batched scalar tensor.")
    if not bool(torch.isfinite(bootstrap_value.detach()).all()):
        raise ValueError("bootstrap_value must contain only finite values.")
    if bool((bootstrap_value.detach() < 0.0).any()):
        raise ValueError("bootstrap_value must be a non-negative temporal cost.")
    shape = bootstrap_value.shape
    delta = _integer_tensor_like(
        "delta_primitive",
        delta_primitive,
        shape=shape,
        device=bootstrap_value.device,
        allow_zero=True,
    )
    n_eff = _integer_tensor_like(
        "n_eff_primitive",
        n_eff_primitive,
        shape=shape,
        device=bootstrap_value.device,
        allow_zero=False,
    )
    if bool((delta.remainder(block_steps) != 0).any()):
        raise ValueError("delta_primitive must be aligned to complete action blocks.")
    if bool((n_eff.remainder(block_steps) != 0).any()):
        raise ValueError("n_eff_primitive must be aligned to complete action blocks.")
    if bool((n_eff > backup_primitive_steps).any()):
        raise ValueError("n_eff_primitive exceeds the configured backup window.")

    exact_mask = delta <= n_eff
    prefix_cost = discounted_primitive_cost_v1_c3(n_eff, gamma=gamma)
    discount = torch.pow(
        torch.tensor(gamma, dtype=torch.float32, device=bootstrap_value.device),
        n_eff.to(dtype=torch.float32),
    )
    bootstrapped = prefix_cost + discount * bootstrap_value.detach().float()
    target = torch.where(exact_mask, delta.float(), bootstrapped).detach()
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("V1-C3 RP1 target became non-finite.")
    return RP1TDTargetV1C3(
        target=target,
        bootstrap_value=bootstrap_value.detach().float(),
        prefix_cost=prefix_cost.detach(),
        delta_primitive=delta,
        n_eff_primitive=n_eff,
        exact_mask=exact_mask,
    )


@dataclass(frozen=True)
class RP1TDLossV1C3:
    """Asymmetric expectile-Huber loss and detached diagnostics."""

    loss: torch.Tensor
    per_example_loss: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor


def expectile_huber_td_loss_v1_c3(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    tau: float = 0.03,
    huber_beta: float = 1.0,
) -> RP1TDLossV1C3:
    """Penalize cost overestimates with ``1-tau`` and underestimates with tau."""

    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 < tau < 0.5:
        raise ValueError("tau must be finite and lie in (0, 0.5) for cost values.")
    beta = float(huber_beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("huber_beta must be finite and positive.")
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances.")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must have floating-point dtypes.")
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("prediction and target must share a non-empty shape.")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device.")
    if not bool(torch.isfinite(prediction.detach()).all()) or not bool(
        torch.isfinite(target.detach()).all()
    ):
        raise ValueError("prediction and target must contain only finite values.")

    residual = prediction.float() - target.detach().float()
    absolute = residual.abs()
    huber = torch.where(
        absolute <= beta,
        0.5 * residual.square(),
        beta * (absolute - 0.5 * beta),
    )
    weight = torch.where(residual > 0.0, 1.0 - tau, tau)
    per_example = weight * huber
    loss = per_example.mean()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("V1-C3 expectile-Huber loss became non-finite.")
    return RP1TDLossV1C3(
        loss=loss,
        per_example_loss=per_example,
        residual=residual.detach(),
        weight=weight.detach(),
    )


@dataclass(frozen=True)
class RP1TDBatchV1C3:
    """Complete V1-C3 critic prediction, TD target, loss, and diagnostics."""

    prediction: torch.Tensor
    bootstrap_value: torch.Tensor
    target: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor
    per_example_loss: torch.Tensor
    loss: torch.Tensor
    delta_primitive: torch.Tensor
    n_eff_primitive: torch.Tensor
    exact_mask: torch.Tensor


def _validate_critic_pair_v1_c3(
    critic: RP1StateValueV1C3,
    target_critic: RP1StateValueV1C3,
) -> None:
    if not isinstance(critic, RP1StateValueV1C3):
        raise TypeError("critic must be RP1StateValueV1C3.")
    if not isinstance(target_critic, RP1StateValueV1C3):
        raise TypeError("target_critic must be RP1StateValueV1C3.")
    if critic is target_critic:
        raise ValueError("critic and target_critic must be distinct instances.")
    architecture = ("hidden_dim", "embedding_dim", "depth")
    if any(getattr(critic, key) != getattr(target_critic, key) for key in architecture):
        raise ValueError("critic and target_critic must share an architecture.")
    if any(parameter.requires_grad for parameter in target_critic.parameters()):
        raise ValueError("target_critic parameters must be frozen.")
    if any(module.training for module in target_critic.modules()):
        raise ValueError("target_critic must remain in eval mode.")


def build_rp1_td_loss_v1_c3(
    critic: RP1StateValueV1C3,
    target_critic: RP1StateValueV1C3,
    anchor: torch.Tensor,
    successor: torch.Tensor,
    goal: torch.Tensor,
    delta_primitive: torch.Tensor,
    n_eff_primitive: torch.Tensor,
    *,
    gamma: float = 0.98,
    tau: float = 0.03,
    huber_beta: float = 1.0,
) -> RP1TDBatchV1C3:
    """Build one critic-only RP1 update from frozen V1-C state latents."""

    _validate_critic_pair_v1_c3(critic, target_critic)
    _validate_latent_pair("anchor", anchor, "successor", successor)
    _validate_latent_pair("anchor", anchor, "goal", goal)
    frozen_anchor = anchor.detach()
    frozen_successor = successor.detach()
    frozen_goal = goal.detach()

    prediction = critic(frozen_anchor, frozen_goal)
    with torch.no_grad():
        bootstrap_value = target_critic(frozen_successor, frozen_goal)
        target_output = rp1_temporal_td_target_v1_c3(
            delta_primitive,
            n_eff_primitive,
            bootstrap_value,
            gamma=gamma,
        )
    loss_output = expectile_huber_td_loss_v1_c3(
        prediction,
        target_output.target,
        tau=tau,
        huber_beta=huber_beta,
    )
    return RP1TDBatchV1C3(
        prediction=prediction,
        bootstrap_value=bootstrap_value.detach(),
        target=target_output.target,
        residual=loss_output.residual,
        weight=loss_output.weight,
        per_example_loss=loss_output.per_example_loss,
        loss=loss_output.loss,
        delta_primitive=target_output.delta_primitive,
        n_eff_primitive=target_output.n_eff_primitive,
        exact_mask=target_output.exact_mask,
    )


@torch.no_grad()
def ema_update_target_v1_c3(
    target_critic: RP1StateValueV1C3,
    critic: RP1StateValueV1C3,
    *,
    rate: float = 0.005,
) -> None:
    """Apply ``target=(1-rate)*target + rate*online`` and keep it frozen."""

    _validate_critic_pair_v1_c3(critic, target_critic)
    rate = _finite_probability("rate", rate)
    for target_parameter, online_parameter in zip(
        target_critic.parameters(), critic.parameters(), strict=True
    ):
        target_parameter.mul_(1.0 - rate).add_(online_parameter, alpha=rate)
    for target_buffer, online_buffer in zip(
        target_critic.buffers(), critic.buffers(), strict=True
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(1.0 - rate).add_(online_buffer, alpha=rate)
        else:
            target_buffer.copy_(online_buffer)
    target_critic.requires_grad_(False)
    target_critic.eval()


__all__ = [
    "PrimitiveStepWindowV1C3",
    "RP1StateValueV1C3",
    "RP1TDBatchV1C3",
    "RP1TDLossV1C3",
    "RP1TDTargetV1C3",
    "V1C3_BACKUP_PRIMITIVE_STEPS",
    "V1C3_BLOCK_STEPS",
    "V1C3_STATE_DIM",
    "V1C3Config",
    "build_rp1_td_loss_v1_c3",
    "discounted_primitive_cost_v1_c3",
    "ema_update_target_v1_c3",
    "expectile_huber_td_loss_v1_c3",
    "rp1_block_window_v1_c3",
    "rp1_temporal_td_target_v1_c3",
]
