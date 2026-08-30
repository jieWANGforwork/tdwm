#!/usr/bin/env python3
"""Train Actor-Free TD-JEPA V0 method C."""

from tdwm.training.actor_free_td_lewm_v0_c import (
    load_actor_free_td_lewm_v0_c_training_protocol,
    train_actor_free_td_lewm_v0_c,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="V0 C",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v0_c_training_protocol,
        train=train_actor_free_td_lewm_v0_c,
    )


if __name__ == "__main__":
    main()
