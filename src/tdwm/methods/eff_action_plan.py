"""EffActionPlan's learned residual action updates and two-stage loss.

The iterative planner consumes all six specified inputs. Feedback J and dJ/da
are detached; the residual action chain is not detached during training.
Frozen e/G/V retain their derivative with respect to the final action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from tdwm.methods.eff_action import (
    EFF_ACTION_RAW_ACTION_DIM,
    EFF_ACTION_STATE_DIM,
    EffActionSuccessor,
    EffActionValue,
    _aligned,
    _mask,
    _vector,
    eff_action_cost,
    require_frozen_eff_action_module,
)


class EffActionPlanner(nn.Module):
    """P(z_t, a_ref, a_current, z_goal, J, grad_a J) -> delta_a.

    ``raw_action_dim`` describes one action object. The current LeWM adapter
    uses one 25D block; configuring another size does not silently create a
    sequence model or change the encoder's contract.
    """

    state_dim = EFF_ACTION_STATE_DIM

    def __init__(
        self, *, raw_action_dim: int, hidden_dim: int, hidden_layers: int
    ) -> None:
        super().__init__()
        for name, value in (
            ("raw_action_dim", raw_action_dim),
            ("hidden_dim", hidden_dim),
            ("hidden_layers", hidden_layers),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.raw_action_dim = raw_action_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = 2 * self.state_dim + 3 * self.raw_action_dim + 1
        input_dim = self.input_dim
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, raw_action_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        reference_action: torch.Tensor,
        current_action: torch.Tensor,
        goal: torch.Tensor,
        cost: torch.Tensor,
        action_gradient: torch.Tensor,
    ) -> torch.Tensor:
        _vector("state", state, self.state_dim)
        for name, tensor, dimension in (
            ("reference_action", reference_action, self.raw_action_dim),
            ("current_action", current_action, self.raw_action_dim),
            ("goal", goal, self.state_dim),
            ("action_gradient", action_gradient, self.raw_action_dim),
        ):
            _vector(name, tensor, dimension)
            _aligned(name, tensor, state)
        if not isinstance(cost, torch.Tensor) or not cost.is_floating_point():
            raise TypeError("cost must be a floating-point torch.Tensor.")
        if cost.shape != state.shape[:-1] or cost.device != state.device:
            raise ValueError("cost must match the state leading shape and device.")
        if not bool(torch.isfinite(cost.detach()).all()):
            raise ValueError("cost must contain only finite values.")
        inputs = torch.cat(
            (
                state.detach(),
                reference_action.detach(),
                current_action,
                goal.detach(),
                cost.detach().to(state.dtype).unsqueeze(-1),
                action_gradient.detach(),
            ),
            dim=-1,
        )
        delta = self.network(inputs)
        if not bool(torch.isfinite(delta.detach()).all()):
            raise FloatingPointError("EffActionPlan P produced a non-finite update.")
        return delta


def project_eff_action(
    action: torch.Tensor,
    *,
    lower_bound: float | torch.Tensor,
    upper_bound: float | torch.Tensor,
) -> torch.Tensor:
    """Differentiably clip actions to the explicitly supplied action bounds."""

    if not isinstance(action, torch.Tensor) or action.ndim < 1:
        raise ValueError("action must have shape (..., raw_action_dim).")
    _vector("action", action, action.shape[-1])
    lower = torch.as_tensor(lower_bound, device=action.device, dtype=action.dtype)
    upper = torch.as_tensor(upper_bound, device=action.device, dtype=action.dtype)
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("Action bounds must contain only finite values.")
    try:
        lower = torch.broadcast_to(lower, action.shape)
        upper = torch.broadcast_to(upper, action.shape)
    except RuntimeError as error:
        raise ValueError("Action bounds must broadcast to the action shape.") from error
    if bool((lower > upper).any()):
        raise ValueError("Action lower_bound must not exceed upper_bound.")
    return torch.minimum(torch.maximum(action, lower), upper)


@dataclass(frozen=True)
class EffActionPlanOutput:
    action: torch.Tensor
    reference_action: torch.Tensor
    actions: tuple[torch.Tensor, ...]
    costs: tuple[torch.Tensor, ...]
    action_gradients: tuple[torch.Tensor, ...]
    deltas: tuple[torch.Tensor, ...]

    @property
    def final_cost(self) -> torch.Tensor:
        return self.costs[-1]


def iterate_eff_action_plan(
    planner: EffActionPlanner,
    successor: EffActionSuccessor,
    value: EffActionValue,
    action_encoder: nn.Module,
    *,
    state: torch.Tensor,
    initial_action: torch.Tensor,
    goal: torch.Tensor,
    task: torch.Tensor,
    iterations: int,
    epsilon: float,
    lower_bound: float | torch.Tensor,
    upper_bound: float | torch.Tensor,
    track_grad: bool,
) -> EffActionPlanOutput:
    """Apply the same K residual updates in training and inference.

    ``track_grad=True`` retains every residual connection and the final J
    graph. ``False`` still computes the action-gradient feedback at each step
    but returns detached tensors. Both modes work inside outer ``no_grad`` or
    ``inference_mode`` contexts. Initialization is supplied by the caller and
    projected once; no CEM, teacher action, or hidden random initialization is
    used here. All e/G/V parameters must already be frozen and in eval mode.

    ``actions`` has K+1 entries, starting at a_0; ``costs`` has K detached
    feedback values followed by J(a_K), which is differentiable in training.
    """

    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer.")
    if not isinstance(track_grad, bool):
        raise TypeError("track_grad must be boolean.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if planner.raw_action_dim != EFF_ACTION_RAW_ACTION_DIM:
        raise ValueError(
            "The current frozen LeWM encoder requires one 25D action block."
        )
    for name, module in (
        ("successor", successor),
        ("value", value),
        ("action_encoder", action_encoder),
    ):
        require_frozen_eff_action_module(module, name)
    if not track_grad and any(part.training for part in planner.modules()):
        raise ValueError("The planner must be in eval mode for inference.")

    # Re-enable autograd even when the public evaluation API wraps policy
    # calls in inference_mode. Cloning there converts inference tensors into
    # regular tensors on which action derivatives are legal.
    with torch.inference_mode(False), torch.enable_grad():
        state = state.detach().clone()
        goal = goal.detach().clone()
        task = task.detach().clone()
        initial_action = initial_action.detach().clone()
        lower = (
            torch.as_tensor(
                lower_bound, device=initial_action.device, dtype=initial_action.dtype
            )
            .detach()
            .clone()
        )
        upper = (
            torch.as_tensor(
                upper_bound, device=initial_action.device, dtype=initial_action.dtype
            )
            .detach()
            .clone()
        )
        reference = project_eff_action(
            initial_action, lower_bound=lower, upper_bound=upper
        ).detach()
        current = reference.clone().requires_grad_(track_grad)
        actions = [current]
        costs: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        for _ in range(iterations):
            probe = current.detach().clone().requires_grad_(True)
            feedback_cost = eff_action_cost(
                successor,
                value,
                action_encoder,
                state=state,
                raw_action=probe,
                task=task,
                goal=goal,
                epsilon=epsilon,
            )
            if feedback_cost.requires_grad:
                gradient = torch.autograd.grad(
                    feedback_cost.sum(), probe, create_graph=False, allow_unused=True
                )[0]
            else:
                gradient = None
            if gradient is None:
                gradient = torch.zeros_like(probe)
            gradient = gradient.detach()
            feedback_cost = feedback_cost.detach()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("EffActionPlan action feedback is non-finite.")
            with torch.set_grad_enabled(track_grad):
                delta = planner(
                    state, reference, current, goal, feedback_cost, gradient
                )
                current = project_eff_action(
                    current + delta.to(current.dtype),
                    lower_bound=lower,
                    upper_bound=upper,
                )
            actions.append(current)
            costs.append(feedback_cost)
            gradients.append(gradient)
            deltas.append(delta)
        with torch.set_grad_enabled(track_grad):
            final_cost = eff_action_cost(
                successor,
                value,
                action_encoder,
                state=state,
                raw_action=current,
                task=task,
                goal=goal,
                epsilon=epsilon,
            )
        costs.append(final_cost)
        return EffActionPlanOutput(
            action=current,
            reference_action=reference,
            actions=tuple(actions),
            costs=tuple(costs),
            action_gradients=tuple(gradients),
            deltas=tuple(deltas),
        )


@dataclass(frozen=True)
class EffActionPlanLossOutput:
    loss: torch.Tensor
    trajectory_loss: torch.Tensor
    efficiency_loss: torch.Tensor
    trajectory_per_example_loss: torch.Tensor
    stage: int


def build_eff_action_plan_loss(
    plan: EffActionPlanOutput,
    *,
    dataset_action: torch.Tensor,
    stage: int,
    lambda_traj: float,
    lambda_eff: float,
    valid_mask: torch.Tensor,
) -> EffActionPlanLossOutput:
    """Stage 1: L_traj. Stage 2: lambda_traj L_traj + lambda_eff E[J].

    L_traj is the squared vector norm, averaged over valid examples, rather
    than the mean over action coordinates. The demonstrated action is always
    detached and belongs only in this terminal loss, never in P's inputs.
    """

    if stage not in (1, 2) or isinstance(stage, bool):
        raise ValueError("stage must be 1 (trajectory) or 2 (joint).")
    for name, coefficient in (("lambda_traj", lambda_traj), ("lambda_eff", lambda_eff)):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    _vector("final action", plan.action, plan.action.shape[-1])
    _vector("dataset_action", dataset_action, plan.action.shape[-1], finite=False)
    _aligned("dataset_action", dataset_action, plan.action)
    _mask("valid_mask", valid_mask, plan.action)
    if not bool(valid_mask.any()):
        raise ValueError("valid_mask must select at least one example.")
    if plan.final_cost.shape != plan.action.shape[:-1]:
        raise ValueError("final_cost must match the action leading shape.")
    selected_target = dataset_action[valid_mask].detach().float()
    if not bool(torch.isfinite(selected_target).all()):
        raise ValueError("Valid dataset actions must contain only finite values.")
    selected_cost = plan.final_cost[valid_mask].float()
    if not bool(torch.isfinite(selected_cost.detach()).all()):
        raise ValueError("Valid final costs must contain only finite values.")
    error = (plan.action[valid_mask].float() - selected_target).square().sum(dim=-1)
    if not bool(torch.isfinite(error.detach()).all()):
        raise FloatingPointError("EffActionPlan trajectory loss is non-finite.")
    trajectory = error.mean()
    efficiency = selected_cost.mean()
    loss = (
        trajectory if stage == 1 else lambda_traj * trajectory + lambda_eff * efficiency
    )
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("EffActionPlan total loss is non-finite.")
    per_example = plan.action.new_zeros(plan.action.shape[:-1], dtype=torch.float32)
    per_example[valid_mask] = error
    return EffActionPlanLossOutput(loss, trajectory, efficiency, per_example, stage)


__all__ = [
    "EffActionPlanLossOutput",
    "EffActionPlanOutput",
    "EffActionPlanner",
    "build_eff_action_plan_loss",
    "iterate_eff_action_plan",
    "project_eff_action",
]
