"""Training entry for V0 F (same-future/different-goal advantage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.training.frozen_actor_free_td_v0 import (
    V0_SPECS,
    load_actor_free_td_lewm_v0_training_protocol,
    train_actor_free_td_lewm_v0,
    validate_actor_free_td_lewm_v0_training_protocol,
)

METHOD = "actor_free_td_lewm_v0_f"
VARIANT = "f"
SPEC = V0_SPECS[VARIANT]


def load_actor_free_td_lewm_v0_f_training_protocol(path: str | Path) -> dict[str, Any]:
    return load_actor_free_td_lewm_v0_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_v0_f_training_protocol(protocol: dict[str, Any]) -> None:
    validate_actor_free_td_lewm_v0_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_v0_f(**kwargs: Any) -> dict[str, Any]:
    return train_actor_free_td_lewm_v0(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_v0_f_training_protocol",
    "train_actor_free_td_lewm_v0_f",
    "validate_actor_free_td_lewm_v0_f_training_protocol",
]
