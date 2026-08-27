"""Reward-free, action-prefix successor supervision for LeWM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from tdwm.methods.successor_geometry import successor_feature_basis


def discounted_prefix_mass(
    horizon: int,
    *,
    gamma: float,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-step discount powers and their inclusive prefix sums."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    powers = torch.pow(
        reference.new_tensor(gamma),
        torch.arange(horizon, device=reference.device),
    )
    return powers, powers.cumsum(dim=0)


def finite_horizon_successor_targets(
    future_latents: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Build normalized direct-MC successor targets for every prefix horizon.

    ``future_latents[..., h, :]`` is the latent reached after applying the
    first ``h + 1`` candidate actions. No goal, reward, policy, or bootstrap is
    involved in this target.
    """

    if future_latents.ndim < 2 or future_latents.shape[-2] <= 0:
        raise ValueError("future_latents must contain a non-empty time axis.")
    return finite_horizon_successor_from_moments(
        successor_feature_basis(future_latents), gamma=gamma
    )


def finite_horizon_successor_from_moments(
    moments: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Convert per-horizon future moments into normalized prefix successors."""

    if moments.ndim < 2 or moments.shape[-2] <= 0 or moments.shape[-1] < 3:
        raise ValueError("moments must contain non-empty time and feature axes.")
    powers, mass = discounted_prefix_mass(
        moments.shape[-2], gamma=gamma, reference=moments
    )
    view_shape = (1,) * (moments.ndim - 2) + (-1, 1)
    weighted = moments * powers.view(view_shape)
    return weighted.cumsum(dim=-2) / mass.view(view_shape)


def successor_moments_from_sequence(
    successor: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Invert an all-horizon successor sequence into future moments."""

    if successor.ndim < 2 or successor.shape[-2] <= 0 or successor.shape[-1] < 3:
        raise ValueError("successor must contain non-empty time and feature axes.")
    powers, mass = discounted_prefix_mass(
        successor.shape[-2], gamma=gamma, reference=successor
    )
    if torch.any(powers == 0):
        raise ValueError("Recovering future moments requires gamma > 0.")
    view_shape = (1,) * (successor.ndim - 2) + (-1, 1)
    weighted = successor * mass.view(view_shape)
    previous = torch.cat(
        (torch.zeros_like(weighted[..., :1, :]), weighted[..., :-1, :]),
        dim=-2,
    )
    return (weighted - previous) / powers.view(view_shape)


def latent_sequence_from_successor(
    successor: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Recover the latent coordinates represented by successor increments."""

    moments = successor_moments_from_sequence(successor, gamma=gamma)
    latent_dim = moments.shape[-1] - 2
    return moments[..., :latent_dim] * math.sqrt(latent_dim)


def successor_recurrence_residual(
    successor: torch.Tensor,
    predicted_future: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Measure whether successor increments equal the matching latent feature."""

    if successor.ndim < 2 or predicted_future.ndim != successor.ndim:
        raise ValueError("successor and predicted_future must have matching ranks.")
    if successor.shape[:-1] != predicted_future.shape[:-1]:
        raise ValueError("successor and predicted_future leading shapes must match.")
    if successor.shape[-1] != predicted_future.shape[-1] + 2:
        raise ValueError("successor must use the lifted latent feature dimension.")

    horizon = predicted_future.shape[-2]
    powers, mass = discounted_prefix_mass(
        horizon, gamma=gamma, reference=predicted_future
    )
    view_shape = (1,) * (predicted_future.ndim - 2) + (-1, 1)
    weighted_successor = successor * mass.view(view_shape)
    previous = torch.cat(
        (
            torch.zeros_like(weighted_successor[..., :1, :]),
            weighted_successor[..., :-1, :],
        ),
        dim=-2,
    )
    expected_increment = (
        successor_feature_basis(predicted_future) * powers.view(view_shape)
    )
    return weighted_successor - previous - expected_increment


def balanced_successor_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    vector_reduction: str = "coordinate_mean",
) -> torch.Tensor:
    """Combine vector, squared-norm, and constant feature errors.

    ``coordinate_mean`` preserves the objective used by the first two method
    versions. ``group_sum`` treats the complete scaled latent vector as one
    feature group. Because the vector is scaled by ``sqrt(d)``, summing that
    group's coordinate errors recovers latent MSE instead of dividing it by
    ``d`` a second time.
    """

    if prediction.shape != target.shape or prediction.shape[-1] < 3:
        raise ValueError("prediction and target must be matching lifted features.")
    vector_error = (prediction[..., :-2] - target[..., :-2]).square()
    if vector_reduction == "coordinate_mean":
        vector = vector_error.mean()
    elif vector_reduction == "group_sum":
        vector = vector_error.sum(dim=-1).mean()
    else:
        raise ValueError(
            "vector_reduction must be 'coordinate_mean' or 'group_sum'."
        )
    squared_norm = (prediction[..., -2] - target[..., -2]).square().mean()
    constant = (prediction[..., -1] - target[..., -1]).square().mean()
    return (vector + squared_norm + constant) / 3.0


def left_pad_latent_history(
    latent_history: torch.Tensor,
    *,
    history_size: int,
    history_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a non-empty latent suffix and return its binary validity mask."""

    if latent_history.ndim < 3 or latent_history.shape[-1] <= 0:
        raise ValueError("latent_history must contain time and latent dimensions.")
    if history_size <= 0:
        raise ValueError("history_size must be positive.")
    observed = int(latent_history.shape[-2])
    if not 0 < observed <= history_size:
        raise ValueError(
            "latent_history time must lie between one and the configured maximum."
        )
    expected_mask_shape = latent_history.shape[:-1]
    if history_mask is None:
        valid = torch.ones(
            expected_mask_shape,
            dtype=torch.bool,
            device=latent_history.device,
        )
    else:
        if history_mask.shape != expected_mask_shape:
            raise ValueError("history_mask must match the latent history time axes.")
        valid = history_mask.to(device=latent_history.device, dtype=torch.bool)
        if torch.any(history_mask.to(device=latent_history.device) != valid):
            raise ValueError("history_mask must contain only binary values.")
        invalid_order = valid[..., :-1] & ~valid[..., 1:]
        if torch.any(~valid[..., -1]) or torch.any(invalid_order):
            raise ValueError("history_mask must be a non-empty right-aligned suffix.")

    missing = history_size - observed
    if missing:
        latent_padding = latent_history.new_zeros(
            *latent_history.shape[:-2], missing, latent_history.shape[-1]
        )
        mask_padding = torch.zeros(
            *valid.shape[:-1], missing, dtype=torch.bool, device=valid.device
        )
        latent_history = torch.cat((latent_padding, latent_history), dim=-2)
        valid = torch.cat((mask_padding, valid), dim=-1)
    padded = torch.where(
        valid.unsqueeze(-1), latent_history, torch.zeros_like(latent_history)
    )
    return padded, valid.to(dtype=latent_history.dtype)


class ActionPrefixSuccessorHead(nn.Module):
    """Causally summarize each supplied action prefix without seeing a goal."""

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        history_size: int,
        hidden_dim: int,
        masked_history: bool = False,
    ) -> None:
        super().__init__()
        if min(embed_dim, action_dim, history_size, hidden_dim) <= 0:
            raise ValueError("Successor-head dimensions must be positive.")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.hidden_dim = int(hidden_dim)
        self.masked_history = bool(masked_history)
        self.output_dim = self.embed_dim + 2

        history_dim = self.history_size * self.embed_dim
        if self.masked_history:
            history_dim += self.history_size
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.prefix_encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        # The constant successor coordinate is fixed to one after horizon
        # normalization, so the network only predicts the nontrivial moments.
        self.readout = nn.Linear(self.hidden_dim, self.embed_dim + 1)

    def forward(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
        *,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent_history.ndim < 3 or action_prefix.ndim != latent_history.ndim:
            raise ValueError("history and action prefix must have matching ranks.")
        if latent_history.shape[-1] != self.embed_dim:
            raise ValueError("latent_history has the wrong latent dimension.")
        if not self.masked_history and latent_history.shape[-2] != self.history_size:
            raise ValueError(
                "latent_history must end with "
                f"({self.history_size}, {self.embed_dim})."
            )
        if not self.masked_history and history_mask is not None:
            raise ValueError("This legacy successor head does not use a history mask.")
        if action_prefix.shape[:-2] != latent_history.shape[:-2]:
            raise ValueError("history and action prefix leading shapes must match.")
        if action_prefix.shape[-2] <= 0 or action_prefix.shape[-1] != self.action_dim:
            raise ValueError("action_prefix has an invalid time or action dimension.")

        leading = latent_history.shape[:-2]
        flat_batch = math.prod(leading) if leading else 1
        if self.masked_history:
            padded, valid = left_pad_latent_history(
                latent_history,
                history_size=self.history_size,
                history_mask=history_mask,
            )
            history = torch.cat(
                (
                    padded.reshape(flat_batch, -1),
                    valid.reshape(flat_batch, -1),
                ),
                dim=-1,
            )
        else:
            history = latent_history.reshape(flat_batch, -1)
        actions = action_prefix.reshape(flat_batch, action_prefix.shape[-2], -1)
        initial = self.history_encoder(history).unsqueeze(0)
        encoded_actions = self.action_encoder(actions)
        states, _ = self.prefix_encoder(encoded_actions, initial)
        raw = self.readout(states)
        linear = raw[..., : self.embed_dim]
        squared_norm = functional.softplus(raw[..., self.embed_dim :])
        constant = torch.ones_like(squared_norm)
        successor = torch.cat((linear, squared_norm, constant), dim=-1)
        return successor.reshape(*leading, action_prefix.shape[-2], self.output_dim)


class ActionPrefixMomentHead(ActionPrefixSuccessorHead):
    """Predict future moments and construct successors by exact accumulation."""

    def __init__(self, *, gamma: float, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("ActionPrefixMomentHead requires gamma in (0, 1].")
        self.gamma = float(gamma)

    def predict_moments(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(latent_history, action_prefix)

    def forward(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        moments = self.predict_moments(latent_history, action_prefix)
        return finite_horizon_successor_from_moments(moments, gamma=self.gamma)


class _ConditionalResidualBlock(nn.Module):
    """Pointwise residual predictor conditioned on one action-prefix token."""

    def __init__(self, *, embed_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 3 * embed_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized = self.norm(state) * (1.0 + scale) + shift
        return state + gate * self.mlp(normalized)


class ManifoldTransformerMomentHead(nn.Module):
    """Predict all future latents with a strong causal action-prefix backbone.

    Only latent coordinates are learned. Their squared norm and constant
    successor coordinates are constructed analytically, so every prediction
    remains on the exact lifted-latent manifold used by the goal query.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        history_size: int,
        gamma: float,
        prefix_depth: int,
        prefix_heads: int,
        prefix_mlp_dim: int,
        predictor_depth: int,
        predictor_mlp_dim: int,
        fusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = (
            embed_dim,
            action_dim,
            history_size,
            prefix_depth,
            prefix_heads,
            prefix_mlp_dim,
            predictor_depth,
            predictor_mlp_dim,
            fusion_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("Manifold-prefix dimensions must be positive.")
        if embed_dim % prefix_heads:
            raise ValueError("embed_dim must be divisible by prefix_heads.")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("ManifoldTransformerMomentHead requires gamma in (0, 1].")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.output_dim = self.embed_dim + 2
        self.gamma = float(gamma)
        self.prefix_depth = int(prefix_depth)
        self.prefix_heads = int(prefix_heads)
        self.prefix_mlp_dim = int(prefix_mlp_dim)
        self.predictor_depth = int(predictor_depth)
        self.predictor_mlp_dim = int(predictor_mlp_dim)
        self.fusion_dim = int(fusion_dim)
        self.dropout = float(dropout)

        self.state_token = nn.Sequential(
            nn.Linear(self.embed_dim, self.prefix_mlp_dim),
            nn.GELU(),
            nn.Linear(self.prefix_mlp_dim, self.embed_dim),
        )
        self.action_token = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.prefix_heads,
            dim_feedforward=self.prefix_mlp_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prefix_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.prefix_depth,
            norm=nn.LayerNorm(self.embed_dim),
            enable_nested_tensor=False,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * self.embed_dim),
            nn.Linear(3 * self.embed_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.embed_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.predictor = nn.ModuleList(
            [
                _ConditionalResidualBlock(
                    embed_dim=self.embed_dim,
                    mlp_dim=self.predictor_mlp_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.predictor_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)

    @staticmethod
    def _position_encoding(
        length: int,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        dimension = reference.shape[-1]
        positions = torch.arange(
            length,
            device=reference.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                dimension,
                2,
                device=reference.device,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(
            length,
            dimension,
            device=reference.device,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        cosine_width = encoding[:, 1::2].shape[-1]
        if cosine_width:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[:cosine_width]
            )
        return encoding.to(dtype=reference.dtype).unsqueeze(0)

    def _flatten_inputs(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        if latent_history.ndim < 3 or action_prefix.ndim != latent_history.ndim:
            raise ValueError("history and action prefix must have matching ranks.")
        if latent_history.shape[-2:] != (self.history_size, self.embed_dim):
            raise ValueError(
                "latent_history must end with "
                f"({self.history_size}, {self.embed_dim})."
            )
        if action_prefix.shape[:-2] != latent_history.shape[:-2]:
            raise ValueError("history and action prefix leading shapes must match.")
        if action_prefix.shape[-2] <= 0 or action_prefix.shape[-1] != self.action_dim:
            raise ValueError("action_prefix has an invalid time or action dimension.")

        leading = latent_history.shape[:-2]
        flat_batch = math.prod(leading) if leading else 1
        anchor = latent_history[..., -1, :].reshape(flat_batch, self.embed_dim)
        actions = action_prefix.reshape(flat_batch, action_prefix.shape[-2], -1)
        return leading, anchor, actions

    def _prefix_features(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        leading, anchor, actions = self._flatten_inputs(
            latent_history, action_prefix
        )
        state_token = self.state_token(anchor).unsqueeze(1)
        action_tokens = self.action_token(actions)
        tokens = torch.cat((state_token, action_tokens), dim=1)
        tokens = tokens + self._position_encoding(
            tokens.shape[1], reference=tokens
        )
        causal_mask = torch.triu(
            torch.ones(
                tokens.shape[1],
                tokens.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        prefixes = self.prefix_encoder(tokens, mask=causal_mask)[:, 1:]
        return leading, anchor, prefixes

    def predict_latents(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        leading, anchor, prefixes = self._prefix_features(
            latent_history, action_prefix
        )

        anchored = anchor.unsqueeze(1).expand_as(prefixes)
        fused = torch.cat((anchored, prefixes, anchored * prefixes), dim=-1)
        state = anchored + self.fusion(fused)
        for block in self.predictor:
            state = block(state, prefixes)
        prediction = self.output_norm(state)
        return prediction.reshape(
            *leading,
            action_prefix.shape[-2],
            self.embed_dim,
        )

    def predict_moments(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        return successor_feature_basis(
            self.predict_latents(latent_history, action_prefix)
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
    ) -> torch.Tensor:
        moments = self.predict_moments(latent_history, action_prefix)
        return finite_horizon_successor_from_moments(moments, gamma=self.gamma)


class LeWMResidualTransformerHead(ManifoldTransformerMomentHead):
    """Correct a frozen LeWM rollout without replacing its latent geometry."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        del self.fusion
        self.base_fusion = nn.Sequential(
            nn.LayerNorm(5 * self.embed_dim),
            nn.Linear(5 * self.embed_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.embed_dim),
        )
        self.correction_out = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.zeros_(self.correction_out.weight)
        nn.init.zeros_(self.correction_out.bias)

    def _validate_base_future(
        self,
        base_future: torch.Tensor,
        *,
        leading: tuple[int, ...],
        horizon: int,
    ) -> torch.Tensor:
        expected = (*leading, horizon, self.embed_dim)
        if base_future.shape != expected:
            raise ValueError(f"base_future must have shape {expected}.")
        flat_batch = math.prod(leading) if leading else 1
        return base_future.reshape(flat_batch, horizon, self.embed_dim)

    def predict_correction(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
        base_future: torch.Tensor,
    ) -> torch.Tensor:
        leading, anchor, prefixes = self._prefix_features(
            latent_history, action_prefix
        )
        base = self._validate_base_future(
            base_future,
            leading=leading,
            horizon=action_prefix.shape[-2],
        )
        anchored = anchor.unsqueeze(1).expand_as(prefixes)
        fused = torch.cat(
            (
                anchored,
                prefixes,
                base,
                base - anchored,
                base * prefixes,
            ),
            dim=-1,
        )
        state = self.base_fusion(fused)
        for block in self.predictor:
            state = block(state, prefixes)
        correction = self.correction_out(self.output_norm(state))
        return correction.reshape(
            *leading,
            action_prefix.shape[-2],
            self.embed_dim,
        )

    def predict_latents(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
        base_future: torch.Tensor,
    ) -> torch.Tensor:
        return base_future + self.predict_correction(
            latent_history,
            action_prefix,
            base_future,
        )

    def predict_moments(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
        base_future: torch.Tensor,
    ) -> torch.Tensor:
        return successor_feature_basis(
            self.predict_latents(latent_history, action_prefix, base_future)
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        action_prefix: torch.Tensor,
        base_future: torch.Tensor,
    ) -> torch.Tensor:
        moments = self.predict_moments(
            latent_history,
            action_prefix,
            base_future,
        )
        return finite_horizon_successor_from_moments(moments, gamma=self.gamma)


@dataclass(frozen=True)
class MultiHorizonSuccessorOutput:
    """Joint losses for the two descriptions of one future trajectory."""

    prediction: torch.Tensor
    target: torch.Tensor
    latent_loss: torch.Tensor
    successor_loss: torch.Tensor
    recurrence_loss: torch.Tensor
    latent_mse_by_horizon: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recurrence_mse_by_horizon: torch.Tensor


@dataclass(frozen=True)
class SuccessorSequenceOutput:
    """The single predictive objective used by the S-only method."""

    prediction: torch.Tensor
    target: torch.Tensor
    moments: torch.Tensor
    recovered_future: torch.Tensor
    successor_loss: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recovered_latent_mse_by_horizon: torch.Tensor


@dataclass(frozen=True)
class MomentSequenceOutput:
    """Direct all-horizon moment supervision with derived successors."""

    moments: torch.Tensor
    target_moments: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    recovered_future: torch.Tensor
    moment_loss: torch.Tensor
    moment_mse_by_horizon: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recovered_latent_mse_by_horizon: torch.Tensor


@dataclass(frozen=True)
class ManifoldSequenceOutput:
    """Dense latent prediction with an exact derived successor sequence."""

    predicted_future: torch.Tensor
    target_future: torch.Tensor
    moments: torch.Tensor
    target_moments: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    recovered_future: torch.Tensor
    latent_loss: torch.Tensor
    latent_mse_by_horizon: torch.Tensor
    successor_mse_by_horizon: torch.Tensor
    recovered_latent_mse_by_horizon: torch.Tensor


@dataclass(frozen=True)
class ResidualManifoldSequenceOutput(ManifoldSequenceOutput):
    """Frozen LeWM rollout plus one learned all-horizon residual."""

    base_future: torch.Tensor
    correction: torch.Tensor
    base_latent_mse_by_horizon: torch.Tensor


def _mse_by_horizon(error: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(error.ndim - 2)) + (error.ndim - 1,)
    return error.square().mean(dim=dimensions)


def multi_horizon_successor_objective(
    head: ActionPrefixSuccessorHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    predicted_future: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
    history_mask: torch.Tensor | None = None,
) -> MultiHorizonSuccessorOutput:
    """Supervise latent rollout, direct successor, and their exact overlap."""

    if predicted_future.shape != target_future.shape:
        raise ValueError("predicted_future and target_future must match.")
    if predicted_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if predicted_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    detached_target = target_future.detach()
    target = finite_horizon_successor_targets(detached_target, gamma=gamma)
    prediction = head(
        latent_history,
        action_prefix,
        history_mask=history_mask,
    )
    recurrence = successor_recurrence_residual(
        prediction, predicted_future, gamma=gamma
    )
    zeros = torch.zeros_like(recurrence)
    return MultiHorizonSuccessorOutput(
        prediction=prediction,
        target=target,
        latent_loss=(predicted_future - detached_target).square().mean(),
        successor_loss=balanced_successor_mse(prediction, target),
        recurrence_loss=balanced_successor_mse(recurrence, zeros),
        latent_mse_by_horizon=_mse_by_horizon(predicted_future - detached_target),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recurrence_mse_by_horizon=_mse_by_horizon(recurrence),
    )


def successor_sequence_objective(
    head: ActionPrefixMomentHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
    vector_reduction: str = "coordinate_mean",
) -> SuccessorSequenceOutput:
    """Train one successor sequence without latent or recurrence losses."""

    if target_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if target_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    if target_future.shape[-1] != head.embed_dim:
        raise ValueError("future latents and the successor head must share a dimension.")
    if not math.isclose(float(gamma), head.gamma):
        raise ValueError("The objective gamma differs from the head gamma.")

    moments = head.predict_moments(latent_history, action_prefix)
    prediction = finite_horizon_successor_from_moments(moments, gamma=gamma)
    target = finite_horizon_successor_targets(target_future, gamma=gamma)
    recovered_future = latent_sequence_from_successor(prediction, gamma=gamma)
    return SuccessorSequenceOutput(
        prediction=prediction,
        target=target,
        moments=moments,
        recovered_future=recovered_future,
        successor_loss=balanced_successor_mse(
            prediction,
            target,
            vector_reduction=vector_reduction,
        ),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recovered_latent_mse_by_horizon=_mse_by_horizon(
            recovered_future - target_future
        ),
    )


def moment_sequence_objective(
    head: ActionPrefixMomentHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
    vector_reduction: str = "group_sum",
    detach_target: bool = True,
) -> MomentSequenceOutput:
    """Supervise every future lifted moment with an optional detached target."""

    if target_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if target_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    if target_future.shape[-1] != head.embed_dim:
        raise ValueError("future latents and the successor head must share a dimension.")
    if not math.isclose(float(gamma), head.gamma):
        raise ValueError("The objective gamma differs from the head gamma.")

    objective_target = target_future.detach() if detach_target else target_future
    target_moments = successor_feature_basis(objective_target)
    moments = head.predict_moments(latent_history, action_prefix)
    prediction = finite_horizon_successor_from_moments(moments, gamma=gamma)
    target = finite_horizon_successor_from_moments(target_moments, gamma=gamma)
    recovered_future = latent_sequence_from_successor(prediction, gamma=gamma)
    return MomentSequenceOutput(
        moments=moments,
        target_moments=target_moments,
        prediction=prediction,
        target=target,
        recovered_future=recovered_future,
        moment_loss=balanced_successor_mse(
            moments,
            target_moments,
            vector_reduction=vector_reduction,
        ),
        moment_mse_by_horizon=_mse_by_horizon(moments - target_moments),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recovered_latent_mse_by_horizon=_mse_by_horizon(
            recovered_future - objective_target
        ),
    )


def manifold_sequence_objective(
    head: ManifoldTransformerMomentHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
    detach_target: bool = False,
) -> ManifoldSequenceOutput:
    """Train one dense latent sequence and derive all successor values exactly."""

    if target_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if target_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    if target_future.shape[-1] != head.embed_dim:
        raise ValueError("future latents and the prefix head must share a dimension.")
    if not math.isclose(float(gamma), head.gamma):
        raise ValueError("The objective gamma differs from the head gamma.")

    objective_target = target_future.detach() if detach_target else target_future
    predicted_future = head.predict_latents(latent_history, action_prefix)
    moments = successor_feature_basis(predicted_future)
    target_moments = successor_feature_basis(objective_target)
    prediction = finite_horizon_successor_from_moments(moments, gamma=gamma)
    target = finite_horizon_successor_from_moments(target_moments, gamma=gamma)
    recovered_future = latent_sequence_from_successor(prediction, gamma=gamma)
    return ManifoldSequenceOutput(
        predicted_future=predicted_future,
        target_future=objective_target,
        moments=moments,
        target_moments=target_moments,
        prediction=prediction,
        target=target,
        recovered_future=recovered_future,
        latent_loss=(predicted_future - objective_target).square().mean(),
        latent_mse_by_horizon=_mse_by_horizon(
            predicted_future - objective_target
        ),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recovered_latent_mse_by_horizon=_mse_by_horizon(
            recovered_future - objective_target
        ),
    )


def residual_manifold_sequence_objective(
    head: LeWMResidualTransformerHead,
    latent_history: torch.Tensor,
    action_prefix: torch.Tensor,
    base_future: torch.Tensor,
    target_future: torch.Tensor,
    *,
    gamma: float,
) -> ResidualManifoldSequenceOutput:
    """Learn only the correction to a frozen LeWM multi-step rollout."""

    if base_future.shape != target_future.shape:
        raise ValueError("base_future and target_future must match.")
    if target_future.shape[:-2] != latent_history.shape[:-2]:
        raise ValueError("future and history leading shapes must match.")
    if target_future.shape[-2] != action_prefix.shape[-2]:
        raise ValueError("future and action-prefix horizons must match.")
    if target_future.shape[-1] != head.embed_dim:
        raise ValueError("future latents and the residual head must share a dimension.")
    if not math.isclose(float(gamma), head.gamma):
        raise ValueError("The objective gamma differs from the head gamma.")

    detached_base = base_future.detach()
    detached_target = target_future.detach()
    correction = head.predict_correction(
        latent_history.detach(),
        action_prefix,
        detached_base,
    )
    predicted_future = detached_base + correction
    moments = successor_feature_basis(predicted_future)
    target_moments = successor_feature_basis(detached_target)
    prediction = finite_horizon_successor_from_moments(moments, gamma=gamma)
    target = finite_horizon_successor_from_moments(target_moments, gamma=gamma)
    recovered_future = latent_sequence_from_successor(prediction, gamma=gamma)
    return ResidualManifoldSequenceOutput(
        predicted_future=predicted_future,
        target_future=detached_target,
        moments=moments,
        target_moments=target_moments,
        prediction=prediction,
        target=target,
        recovered_future=recovered_future,
        latent_loss=(predicted_future - detached_target).square().mean(),
        latent_mse_by_horizon=_mse_by_horizon(
            predicted_future - detached_target
        ),
        successor_mse_by_horizon=_mse_by_horizon(prediction - target),
        recovered_latent_mse_by_horizon=_mse_by_horizon(
            recovered_future - detached_target
        ),
        base_future=detached_base,
        correction=correction,
        base_latent_mse_by_horizon=_mse_by_horizon(
            detached_base - detached_target
        ),
    )


__all__ = [
    "ActionPrefixMomentHead",
    "ActionPrefixSuccessorHead",
    "ManifoldSequenceOutput",
    "ManifoldTransformerMomentHead",
    "LeWMResidualTransformerHead",
    "MultiHorizonSuccessorOutput",
    "MomentSequenceOutput",
    "ResidualManifoldSequenceOutput",
    "SuccessorSequenceOutput",
    "balanced_successor_mse",
    "discounted_prefix_mass",
    "finite_horizon_successor_from_moments",
    "finite_horizon_successor_targets",
    "latent_sequence_from_successor",
    "left_pad_latent_history",
    "manifold_sequence_objective",
    "multi_horizon_successor_objective",
    "moment_sequence_objective",
    "residual_manifold_sequence_objective",
    "successor_moments_from_sequence",
    "successor_recurrence_residual",
    "successor_sequence_objective",
]
