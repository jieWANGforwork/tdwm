#!/usr/bin/env python3
"""Train V1-C2 from every model parameter in V1-C epoch 10."""

from tdwm.training.actor_free_td_lewm_v1_c2 import (
    load_actor_free_td_lewm_v1_c2_training_protocol,
    train_actor_free_td_lewm_v1_c2,
)
from tdwm.training.frozen_actor_free_td_cli import run_frozen_actor_free_td_cli


def main() -> None:
    run_frozen_actor_free_td_cli(
        method_label="V1 C2",
        requires_neighbor_index=False,
        requires_v1_c_checkpoint=True,
        load_protocol=load_actor_free_td_lewm_v1_c2_training_protocol,
        train=train_actor_free_td_lewm_v1_c2,
    )


if __name__ == "__main__":
    main()
