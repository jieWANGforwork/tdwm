#!/usr/bin/env python3
"""Fine-tune Actor-Free TD-JEPA V2-EMA-SG method D."""

from tdwm.training.actor_free_td_lewm_v2_cli import run_actor_free_td_lewm_v2_cli
from tdwm.training.actor_free_td_lewm_v2_ema_sg_d import (
    load_actor_free_td_lewm_v2_ema_sg_d_training_protocol,
    train_actor_free_td_lewm_v2_ema_sg_d,
)


def main() -> None:
    run_actor_free_td_lewm_v2_cli(
        method_label="V2-EMA-SG D",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v2_ema_sg_d_training_protocol,
        train=train_actor_free_td_lewm_v2_ema_sg_d,
    )


if __name__ == "__main__":
    main()
