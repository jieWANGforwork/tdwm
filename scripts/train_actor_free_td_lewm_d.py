#!/usr/bin/env python3
"""Train standalone method D."""

from tdwm.training.actor_free_td_lewm_d import (
    load_actor_free_td_lewm_d_training_protocol,
    train_actor_free_td_lewm_d,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="D",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_d_training_protocol,
        train=train_actor_free_td_lewm_d,
    )


if __name__ == "__main__":
    main()
