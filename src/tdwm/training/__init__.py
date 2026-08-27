"""Training orchestration for auditable baseline reproductions."""

from .actor_free_td_lewm import (
    load_actor_free_td_training_protocol,
    train_actor_free_td_lewm,
)
from .cube_data import validate_cube_training_dataset
from .lewm import load_training_protocol, train_lewm
from .local_successor import load_ls_training_protocol, train_ls_lewm
from .rf_successor_lewm import (
    load_rf_successor_training_protocol,
    train_rf_successor_lewm,
)
from .successor_geometry_lewm import (
    load_successor_geometry_training_protocol,
    train_successor_geometry_lewm,
)
from .td_jepa import apply_tdjepa_cube_overrides, train_tdjepa_cube

__all__ = [
    "apply_tdjepa_cube_overrides",
    "load_actor_free_td_training_protocol",
    "load_ls_training_protocol",
    "load_training_protocol",
    "load_rf_successor_training_protocol",
    "load_successor_geometry_training_protocol",
    "train_tdjepa_cube",
    "train_actor_free_td_lewm",
    "train_lewm",
    "train_ls_lewm",
    "train_rf_successor_lewm",
    "train_successor_geometry_lewm",
    "validate_cube_training_dataset",
]
