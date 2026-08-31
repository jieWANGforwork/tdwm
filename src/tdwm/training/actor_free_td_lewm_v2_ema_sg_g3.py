"""Training entry for V2-EMA-SG method G3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.training.actor_free_td_lewm_v2_ema_sg import (
    V2_EMA_SG_SPECS,
    load_actor_free_td_lewm_v2_ema_sg_training_protocol,
    train_actor_free_td_lewm_v2_ema_sg,
    validate_actor_free_td_lewm_v2_ema_sg_training_protocol,
)

METHOD = "actor_free_td_lewm_v2_ema_sg_g3"
VARIANT = "g3"
SPEC = V2_EMA_SG_SPECS[VARIANT]


def load_actor_free_td_lewm_v2_ema_sg_g3_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_actor_free_td_lewm_v2_ema_sg_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_v2_ema_sg_g3_training_protocol(
    protocol: dict[str, Any],
) -> None:
    validate_actor_free_td_lewm_v2_ema_sg_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_v2_ema_sg_g3(**kwargs: Any) -> dict[str, Any]:
    return train_actor_free_td_lewm_v2_ema_sg(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_v2_ema_sg_g3_training_protocol",
    "train_actor_free_td_lewm_v2_ema_sg_g3",
    "validate_actor_free_td_lewm_v2_ema_sg_g3_training_protocol",
]
