#!/usr/bin/env python3
"""Fine-tune Actor-Free TD-JEPA V2-EMA-SG method G3."""

from tdwm.training.actor_free_td_lewm_v2_cli import run_actor_free_td_lewm_v2_cli
from tdwm.training.actor_free_td_lewm_v2_ema_sg_g3 import (
    load_actor_free_td_lewm_v2_ema_sg_g3_training_protocol,
    train_actor_free_td_lewm_v2_ema_sg_g3,
)


def main() -> None:
    run_actor_free_td_lewm_v2_cli(
        method_label="V2-EMA-SG G3",
        requires_neighbor_index=False,
        load_protocol=load_actor_free_td_lewm_v2_ema_sg_g3_training_protocol,
        train=train_actor_free_td_lewm_v2_ema_sg_g3,
    )


if __name__ == "__main__":
    main()
