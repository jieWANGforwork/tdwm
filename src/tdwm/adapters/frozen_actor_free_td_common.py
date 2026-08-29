"""Shared checkpoint mechanics for independently named frozen TD methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm import make_actor_free_td_policy
from tdwm.methods.actor_free_td_lewm import ActorFreeSuccessorHead

METHOD_FAMILY = "actor_free_td_lewm"
OBJECTIVE_VERSION = 1
DEPLOYMENT_CHECKPOINT_VERSION = 1
FORMAL_DEPLOYMENT_EPOCH = 10
FORMAL_DEPLOYMENT_GLOBAL_STEP = 127_960
SOURCE_METHOD = "lewm"
SOURCE_SEED = 3072
SOURCE_EPOCH = 10
SOURCE_FINAL_EPOCH = 10
SOURCE_GLOBAL_STEP = 127_960


@dataclass(frozen=True)
class FrozenActorFreeTDMethodSpec:
    """Identity and method-owned validation for one frozen TD method."""

    method: str
    variant: str
    display_name: str
    objective_keys: tuple[str, ...]
    validate_method_config: Callable[[Mapping[str, Any]], None]
    objective_version: int = OBJECTIVE_VERSION
    deployment_checkpoint_version: int = DEPLOYMENT_CHECKPOINT_VERSION


def is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_exact_values(
    values: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"{label}.{key} must be {expected_value!r}.")


def require_positive_float(
    values: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> float:
    try:
        value = float(values.get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.{key} must be finite and positive.") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label}.{key} must be finite and positive.")
    return value


def validate_weight_clip(values: Mapping[str, Any], *, label: str) -> None:
    weight_clip = values.get("weight_clip")
    if weight_clip is None:
        return
    if (
        not isinstance(weight_clip, Sequence)
        or isinstance(weight_clip, (str, bytes))
        or len(weight_clip) != 2
    ):
        raise ValueError(f"{label}.weight_clip must be [minimum, maximum] or null.")
    try:
        minimum, maximum = (float(item) for item in weight_clip)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label}.weight_clip bounds must be finite, non-negative and ordered."
        ) from error
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or maximum <= 0.0
        or minimum > maximum
    ):
        raise ValueError(
            f"{label}.weight_clip bounds must be finite, non-negative and ordered."
        )


def _validate_source_provenance(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = payload.get("pretrained_world_model_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "Frozen checkpoint is missing pretrained_world_model_provenance."
        )
    require_exact_values(
        provenance,
        {
            "strategy": "frozen_pretrained_lewm",
            "source_method": SOURCE_METHOD,
            "source_seed": SOURCE_SEED,
            "source_epoch": SOURCE_EPOCH,
            "source_final_epoch": SOURCE_FINAL_EPOCH,
            "source_global_step": SOURCE_GLOBAL_STEP,
            "frozen": True,
        },
        label="pretrained_world_model_provenance",
    )
    for key in (
        "source_checkpoint_sha256",
        "source_training_result_sha256",
        "source_training_manifest_sha256",
    ):
        if not is_lower_sha256(provenance.get(key)):
            raise ValueError(
                f"pretrained_world_model_provenance.{key} must be lowercase SHA-256."
            )
    if provenance["source_checkpoint_sha256"] != config.get(
        "pretrained_world_model_sha256"
    ):
        raise ValueError(
            "Pretrained source checkpoint SHA differs from successor_config."
        )
    return provenance


def validate_frozen_actor_free_td_payload(
    payload: Mapping[str, Any],
    *,
    spec: FrozenActorFreeTDMethodSpec,
) -> dict[str, Any]:
    """Validate one method's deployment payload without instantiating modules."""

    require_exact_values(
        payload,
        {
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "objective_version": spec.objective_version,
            "deployment_checkpoint_version": spec.deployment_checkpoint_version,
        },
        label="checkpoint",
    )
    required = {
        "epoch",
        "global_step",
        "successor_state_dict",
        "successor_config",
        "world_model_state_dict",
        "world_model_config",
        "pretrained_world_model_provenance",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"{spec.display_name} checkpoint is missing {sorted(missing)}."
        )
    for key in ("epoch", "global_step"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint.{key} must be a positive integer.")
    config_value = payload["successor_config"]
    if not isinstance(config_value, Mapping):
        raise ValueError("checkpoint.successor_config must be a mapping.")
    config = dict(config_value)
    required_config = {
        "embed_dim",
        "action_dim",
        "history_size",
        "hidden_dim",
        "gamma",
        *spec.objective_keys,
    }
    missing_config = required_config - config.keys()
    if missing_config:
        raise ValueError(f"successor_config is missing {sorted(missing_config)}.")
    require_exact_values(
        config,
        {
            "method": spec.method,
            "method_family": METHOD_FAMILY,
            "variant": spec.variant,
            "objective_version": spec.objective_version,
            "deployment_checkpoint_version": spec.deployment_checkpoint_version,
            "architecture": "actor_free_successor_head",
            "feature_basis": "augmented_latent_squared_distance",
            "goal_conditioning": "none",
            "action_conditioning": "dataset_current_action",
            "bootstrap_action": "dataset_next_action",
            "terminal_source": "next_action_nan_invalid",
            "actor": "none",
            "reward": "none",
            "predicted_context_detach": True,
            "pretrained_world_model_frozen": True,
            "pretrained_world_model_source_method": SOURCE_METHOD,
            "pretrained_world_model_source_seed": SOURCE_SEED,
            "pretrained_world_model_source_epoch": SOURCE_EPOCH,
            "training_branches": ["real_context"],
        },
        label="successor_config",
    )
    for key in ("embed_dim", "action_dim", "history_size", "hidden_dim"):
        try:
            positive = int(config[key]) > 0
        except (TypeError, ValueError) as error:
            raise ValueError(f"successor_config.{key} must be positive.") from error
        if not positive:
            raise ValueError(f"successor_config.{key} must be positive.")
    try:
        gamma = float(config["gamma"])
    except (TypeError, ValueError) as error:
        raise ValueError("successor_config.gamma must lie in [0, 1).") from error
    if not 0.0 <= gamma < 1.0:
        raise ValueError("successor_config.gamma must lie in [0, 1).")
    source_sha = config.get("pretrained_world_model_sha256")
    if not is_lower_sha256(source_sha):
        raise ValueError(
            "successor_config.pretrained_world_model_sha256 must be lowercase SHA-256."
        )
    if config.get("base_checkpoint_sha256") != source_sha:
        raise ValueError(
            "successor_config.base_checkpoint_sha256 must match the pretrained "
            "world-model SHA-256."
        )
    spec.validate_method_config(config)
    _validate_source_provenance(payload, config)
    return config


def load_frozen_actor_free_td_checkpoint(
    checkpoint_path: str | Path,
    *,
    spec: FrozenActorFreeTDMethodSpec,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, ActorFreeSuccessorHead, dict[str, Any], dict[str, Any]]:
    """Restore one independent frozen method's LeWM and successor head."""

    payload_value = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload_value, Mapping):
        raise ValueError("Deployment checkpoint must contain a mapping payload.")
    payload = dict(payload_value)
    config = validate_frozen_actor_free_td_payload(payload, spec=spec)
    successor = ActorFreeSuccessorHead(
        embed_dim=int(config["embed_dim"]),
        action_dim=int(config["action_dim"]),
        history_size=int(config["history_size"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    successor.load_state_dict(payload["successor_state_dict"], strict=True)

    import hydra
    from omegaconf import OmegaConf

    world_model = hydra.utils.instantiate(
        OmegaConf.create(payload["world_model_config"])
    )
    world_model.load_state_dict(payload["world_model_state_dict"], strict=True)
    world_model.eval().requires_grad_(False)
    successor.eval().requires_grad_(False)
    return world_model, successor, config, payload


def make_frozen_actor_free_td_policy(
    *,
    world_model: nn.Module,
    successor: ActorFreeSuccessorHead,
    planning: dict[str, Any],
    gamma: float,
    process: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    clamp_tail_cost: bool = True,
    score_mode: str | None = None,
):
    """Use the established actor-free planner without widening its loader."""

    return make_actor_free_td_policy(
        world_model=world_model,
        successor=successor,
        planning=planning,
        gamma=gamma,
        process=process,
        transform=transform,
        device=device,
        clamp_tail_cost=clamp_tail_cost,
        score_mode=score_mode,
    )


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "FORMAL_DEPLOYMENT_EPOCH",
    "FORMAL_DEPLOYMENT_GLOBAL_STEP",
    "FrozenActorFreeTDMethodSpec",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "SOURCE_EPOCH",
    "SOURCE_FINAL_EPOCH",
    "SOURCE_GLOBAL_STEP",
    "SOURCE_METHOD",
    "SOURCE_SEED",
    "is_lower_sha256",
    "load_frozen_actor_free_td_checkpoint",
    "make_frozen_actor_free_td_policy",
    "require_exact_values",
    "require_positive_float",
    "validate_frozen_actor_free_td_payload",
    "validate_weight_clip",
]
