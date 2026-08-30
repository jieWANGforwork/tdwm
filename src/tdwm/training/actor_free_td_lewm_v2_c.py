"""Training entry for V2 C coupled-Hybrid fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tdwm.training.actor_free_td_lewm_v2 import (
    V2_SPECS,
    load_actor_free_td_lewm_v2_training_protocol,
    train_actor_free_td_lewm_v2,
    validate_actor_free_td_lewm_v2_training_protocol,
)

METHOD = "actor_free_td_lewm_v2_c"
VARIANT = "c"
SPEC = V2_SPECS[VARIANT]


def load_actor_free_td_lewm_v2_c_training_protocol(
    path: str | Path,
) -> dict[str, Any]:
    return load_actor_free_td_lewm_v2_training_protocol(path, spec=SPEC)


def validate_actor_free_td_lewm_v2_c_training_protocol(
    protocol: dict[str, Any],
) -> None:
    validate_actor_free_td_lewm_v2_training_protocol(protocol, spec=SPEC)


def train_actor_free_td_lewm_v2_c(**kwargs: Any) -> dict[str, Any]:
    return train_actor_free_td_lewm_v2(spec=SPEC, **kwargs)


__all__ = [
    "METHOD",
    "SPEC",
    "VARIANT",
    "load_actor_free_td_lewm_v2_c_training_protocol",
    "train_actor_free_td_lewm_v2_c",
    "validate_actor_free_td_lewm_v2_c_training_protocol",
]
