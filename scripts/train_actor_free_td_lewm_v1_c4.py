#!/usr/bin/env python3
"""Train the independent V1-C4 state-only successor."""

from tdwm.training.actor_free_td_lewm_v1_c4 import (
    load_actor_free_td_lewm_v1_c4_training_protocol,
    train_actor_free_td_lewm_v1_c4,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="V1 C4",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v1_c4_training_protocol,
        train=train_actor_free_td_lewm_v1_c4,
    )


if __name__ == "__main__":
    main()
