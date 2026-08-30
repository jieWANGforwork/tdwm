"""Training entry for V1 D (goal-value-weighted TD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.training.frozen_actor_free_td_v1 import (
    V1_SPECS,
    load_actor_free_td_lewm_v1_training_protocol,
    train_actor_free_td_lewm_v1,
    validate_actor_free_td_lewm_v1_training_protocol,
)

METHOD = "actor_free_td_lewm_v1_d"
VARIANT = "d"
SPEC = V1_SPECS[VARIANT]


def load_actor_free_td_lewm_v1_d_training_protocol(path: str | Path) -> dict[str, Any]:
    return load_actor_free_td_lewm_v1_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_v1_d_training_protocol(protocol: dict[str, Any]) -> None:
    validate_actor_free_td_lewm_v1_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_v1_d(**kwargs: Any) -> dict[str, Any]:
    return train_actor_free_td_lewm_v1(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_v1_d_training_protocol",
    "train_actor_free_td_lewm_v1_d",
    "validate_actor_free_td_lewm_v1_d_training_protocol",
]
