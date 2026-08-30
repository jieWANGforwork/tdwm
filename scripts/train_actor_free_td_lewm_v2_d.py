#!/usr/bin/env python3
"""Fine-tune Actor-Free TD-JEPA V2 method D."""

from tdwm.training.actor_free_td_lewm_v2_cli import run_actor_free_td_lewm_v2_cli
from tdwm.training.actor_free_td_lewm_v2_d import (
    load_actor_free_td_lewm_v2_d_training_protocol,
    train_actor_free_td_lewm_v2_d,
)


def main() -> None:
    run_actor_free_td_lewm_v2_cli(
        method_label="V2 D",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v2_d_training_protocol,
        train=train_actor_free_td_lewm_v2_d,
    )


if __name__ == "__main__":
    main()
