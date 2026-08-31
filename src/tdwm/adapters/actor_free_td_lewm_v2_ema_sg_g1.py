"""Deployment adapter for Actor-Free TD-LeWM V2-EMA-SG G1."""

from __future__ import annotations

from pathlib import Path

import torch

from tdwm.adapters.actor_free_td_lewm_v2_ema_sg_common import (
    DEPLOYMENT_CHECKPOINT_VERSION,
    IMPLEMENTATION_VERSION,
    OBJECTIVE_VERSION,
    load_actor_free_td_v2_ema_sg_checkpoint,
    make_actor_free_td_v2_ema_sg_policy,
    make_actor_free_td_v2_ema_sg_spec,
)
from tdwm.adapters.actor_free_td_lewm_v2_g1 import METHOD_SPEC as V2_METHOD_SPEC

VARIANT = "g1"
METHOD_SPEC = make_actor_free_td_v2_ema_sg_spec(V2_METHOD_SPEC)
METHOD = METHOD_SPEC.method


def load_actor_free_td_lewm_v2_ema_sg_g1_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    return load_actor_free_td_v2_ema_sg_checkpoint(
        checkpoint_path, spec=METHOD_SPEC, map_location=map_location
    )


def make_actor_free_td_lewm_v2_ema_sg_g1_policy(**kwargs):
    return make_actor_free_td_v2_ema_sg_policy(**kwargs)


__all__ = [
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "METHOD_SPEC",
    "OBJECTIVE_VERSION",
    "VARIANT",
    "load_actor_free_td_lewm_v2_ema_sg_g1_checkpoint",
    "make_actor_free_td_lewm_v2_ema_sg_g1_policy",
]
