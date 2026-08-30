#!/usr/bin/env python3
"""Train Actor-Free TD-JEPA V0 method D."""

from tdwm.training.actor_free_td_lewm_v0_d import (
    load_actor_free_td_lewm_v0_d_training_protocol,
    train_actor_free_td_lewm_v0_d,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="V0 D",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v0_d_training_protocol,
        train=train_actor_free_td_lewm_v0_d,
    )


if __name__ == "__main__":
    main()
