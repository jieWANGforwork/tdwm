"""Deployment support for the Actor-Free TD-LeWM V2-EMA-SG family.

V2-EMA-SG has the same deployed online LeWM/G topology as V2.  Its separate
identity records the one training change: local LeWM prediction is supervised
by a stop-gradient next latent from the EMA world model.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tdwm.adapters.actor_free_td_lewm_v2_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    OBJECTIVE_VERSION,
    ActorFreeTDLeWMV2,
    ActorFreeTDV2MethodSpec,
    load_actor_free_td_v2_checkpoint,
    make_actor_free_td_v2_policy,
)

METHOD_FAMILY = "actor_free_td_lewm_v2_ema_sg"
IMPLEMENTATION_VERSION = "v2_ema_sg"
VERSION_KEY = "v2_ema"
VERSION_DISPLAY_NAME = "V2 EMA"
TRAINING_COMMIT = "18cd574d522515f20f4103509b1e660b2fc89ea6"
TRAINING_STAGE = "coupled_hybrid_ema_target_finetuning"
EVALUATION_STAGE = "planner_evaluation"
INITIALIZATION = "corresponding_v1_deployment_finetune"
LOCAL_PREDICTION = "ema_target_lewm_one_step_mse"
LOCAL_PREDICTION_TARGET = "ema_world_model_next_latent"
LOCAL_PREDICTION_TARGET_GRADIENT = "stop_gradient"
INITIALIZATION_CONTRACT = {
    "required_checkpoint_family": "actor_free_td_lewm_v1",
    "required_checkpoint_epoch": 10,
    "v2_checkpoint_as_initialization": "prohibited",
    "optimizer_state": "fresh",
}


class ActorFreeTDLeWMV2EMASG(ActorFreeTDLeWMV2):
    """Explicit deployment identity for V2-EMA-SG online modules."""

    implementation_version = IMPLEMENTATION_VERSION
    method_family = METHOD_FAMILY


def make_actor_free_td_v2_ema_sg_spec(
    base_spec: ActorFreeTDV2MethodSpec,
) -> ActorFreeTDV2MethodSpec:
    """Reuse a V2 method objective validator under the EMA-SG identity."""

    return ActorFreeTDV2MethodSpec(
        method=f"{METHOD_FAMILY}_{base_spec.variant}",
        variant=base_spec.variant,
        display_name=base_spec.display_name.replace("V2", VERSION_DISPLAY_NAME, 1),
        objective_keys=base_spec.objective_keys,
        validate_method_config=base_spec.validate_method_config,
        method_family=METHOD_FAMILY,
        implementation_version=IMPLEMENTATION_VERSION,
        evaluation_stage=EVALUATION_STAGE,
        initialization=INITIALIZATION,
        local_prediction=LOCAL_PREDICTION,
        local_prediction_target=LOCAL_PREDICTION_TARGET,
        local_prediction_target_gradient=LOCAL_PREDICTION_TARGET_GRADIENT,
        inference_g_score=("negative_goal_projection_of_v2_ema_sg_online_predictor"),
        deployed_world_model="online_v2_ema_sg_world_model",
        deployed_predictor="online_v2_ema_sg_predictor",
        training_stage=TRAINING_STAGE,
        initialization_contract=INITIALIZATION_CONTRACT,
    )


def load_actor_free_td_v2_ema_sg_checkpoint(
    checkpoint_path: str | Path,
    *,
    spec: ActorFreeTDV2MethodSpec,
    map_location: str | torch.device = "cpu",
):
    """Strictly restore a V2-EMA-SG deployment checkpoint."""

    return load_actor_free_td_v2_checkpoint(
        checkpoint_path,
        spec=spec,
        map_location=map_location,
    )


def make_actor_free_td_v2_ema_sg_policy(**kwargs):
    """Build CEM around the online V2-EMA-SG LeWM and G."""

    return make_actor_free_td_v2_policy(
        **kwargs,
        wrapper_class=ActorFreeTDLeWMV2EMASG,
    )


__all__ = [
    "ActorFreeTDLeWMV2EMASG",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "EVALUATION_STAGE",
    "IMPLEMENTATION_VERSION",
    "INITIALIZATION",
    "INITIALIZATION_CONTRACT",
    "LOCAL_PREDICTION",
    "LOCAL_PREDICTION_TARGET",
    "LOCAL_PREDICTION_TARGET_GRADIENT",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "TRAINING_COMMIT",
    "TRAINING_STAGE",
    "VERSION_DISPLAY_NAME",
    "VERSION_KEY",
    "load_actor_free_td_v2_ema_sg_checkpoint",
    "make_actor_free_td_v2_ema_sg_policy",
    "make_actor_free_td_v2_ema_sg_spec",
]
