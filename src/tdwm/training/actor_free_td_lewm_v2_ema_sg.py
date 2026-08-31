"""Training identity for V2 coupled-Hybrid with EMA latent MSE targets.

This experiment family reuses the V2 training runtime and changes only the
one-step LeWM prediction target.  Each method starts from its matching V1
deployment checkpoint with fresh optimizer state; an existing V2 checkpoint
is not an initialization source for this family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.training.actor_free_td_lewm_v2 import (
    ActorFreeTDLeWMV2Spec,
    load_actor_free_td_lewm_v2_training_protocol,
    train_actor_free_td_lewm_v2,
    validate_actor_free_td_lewm_v2_training_protocol,
)

METHOD_FAMILY = "actor_free_td_lewm_v2_ema_sg"
IMPLEMENTATION_VERSION = "v2_ema_sg"
STAGE = "coupled_hybrid_ema_target_finetuning"
INITIALIZATION = "corresponding_v1_deployment_finetune"
DEPLOYMENT_CHECKPOINT_VERSION = 1
LOCAL_PREDICTION = "ema_target_lewm_one_step_mse"
LOCAL_PREDICTION_TARGET = "ema_world_model_next_latent"
LOCAL_PREDICTION_TARGET_GRADIENT = "stop_gradient"
INITIALIZATION_CONTRACT = {
    "required_checkpoint_family": "actor_free_td_lewm_v1",
    "required_checkpoint_epoch": 10,
    "v2_checkpoint_as_initialization": "prohibited",
    "optimizer_state": "fresh",
}
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")

V2_EMA_SG_SPECS = {
    variant: ActorFreeTDLeWMV2Spec(
        method=f"{METHOD_FAMILY}_{variant}",
        variant=variant,
        requires_neighbor_index=variant == "g1",
        method_family=METHOD_FAMILY,
        implementation_version=IMPLEMENTATION_VERSION,
        stage=STAGE,
        initialization=INITIALIZATION,
        deployment_checkpoint_version=DEPLOYMENT_CHECKPOINT_VERSION,
        local_prediction=LOCAL_PREDICTION,
        local_prediction_target=LOCAL_PREDICTION_TARGET,
        local_prediction_target_gradient=LOCAL_PREDICTION_TARGET_GRADIENT,
        initialization_contract=INITIALIZATION_CONTRACT,
    )
    for variant in VARIANTS
}


def load_actor_free_td_lewm_v2_ema_sg_training_protocol(
    path: str | Path,
    *,
    spec: ActorFreeTDLeWMV2Spec,
) -> dict[str, Any]:
    """Load and validate one EMA-target variant protocol."""

    return load_actor_free_td_lewm_v2_training_protocol(path, spec=spec)


def validate_actor_free_td_lewm_v2_ema_sg_training_protocol(
    protocol: dict[str, Any],
    *,
    spec: ActorFreeTDLeWMV2Spec,
) -> None:
    """Validate one EMA-target variant against the shared strict runtime."""

    validate_actor_free_td_lewm_v2_training_protocol(protocol, spec=spec)


def train_actor_free_td_lewm_v2_ema_sg(
    *,
    spec: ActorFreeTDLeWMV2Spec,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train one EMA-target variant from its matching V1 deployment."""

    return train_actor_free_td_lewm_v2(spec=spec, **kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "INITIALIZATION",
    "INITIALIZATION_CONTRACT",
    "LOCAL_PREDICTION",
    "LOCAL_PREDICTION_TARGET",
    "LOCAL_PREDICTION_TARGET_GRADIENT",
    "METHOD_FAMILY",
    "STAGE",
    "V2_EMA_SG_SPECS",
    "VARIANTS",
    "load_actor_free_td_lewm_v2_ema_sg_training_protocol",
    "train_actor_free_td_lewm_v2_ema_sg",
    "validate_actor_free_td_lewm_v2_ema_sg_training_protocol",
]
