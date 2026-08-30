#!/usr/bin/env python3
"""Fine-tune Actor-Free TD-JEPA V2 method C."""

from tdwm.training.actor_free_td_lewm_v2_c import (
    load_actor_free_td_lewm_v2_c_training_protocol,
    train_actor_free_td_lewm_v2_c,
)
from tdwm.training.actor_free_td_lewm_v2_cli import run_actor_free_td_lewm_v2_cli


def main() -> None:
    run_actor_free_td_lewm_v2_cli(
        method_label="V2 C",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v2_c_training_protocol,
        train=train_actor_free_td_lewm_v2_c,
    )


if __name__ == "__main__":
    main()
