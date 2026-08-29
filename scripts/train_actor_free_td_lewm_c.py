#!/usr/bin/env python3
"""Train standalone method C."""

from tdwm.training.actor_free_td_lewm_c import (
    load_actor_free_td_lewm_c_training_protocol,
    train_actor_free_td_lewm_c,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="C",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_c_training_protocol,
        train=train_actor_free_td_lewm_c,
    )


if __name__ == "__main__":
    main()
