#!/usr/bin/env python3
"""Train Actor-Free TD-JEPA V1 method F."""

from tdwm.training.actor_free_td_lewm_v1_f import (
    load_actor_free_td_lewm_v1_f_training_protocol,
    train_actor_free_td_lewm_v1_f,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="V1 F",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v1_f_training_protocol,
        train=train_actor_free_td_lewm_v1_f,
    )


if __name__ == "__main__":
    main()
