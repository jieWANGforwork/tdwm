from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tdwm.training.actor_free_td_lewm_c import SPEC as C_SPEC
from tdwm.training.actor_free_td_lewm_d import SPEC as D_SPEC
from tdwm.training.actor_free_td_lewm_f import SPEC as F_SPEC
from tdwm.training.actor_free_td_lewm_g1 import SPEC as G1_SPEC
from tdwm.training.actor_free_td_lewm_g2 import SPEC as G2_SPEC
from tdwm.training.actor_free_td_lewm_g3 import SPEC as G3_SPEC
from tdwm.training.frozen_actor_free_td import (
    _build_training_module,
    load_bound_training_split,
)
from tdwm.training.frozen_actor_free_td_cli import (
    build_frozen_actor_free_td_parser,
    run_frozen_actor_free_td_cli,
)

METHOD_CASES = [
    (C_SPEC, {"goal_projection_weight": 0.37}),
    (D_SPEC, {"weight_temperature": 0.73, "weight_clip": None}),
    (F_SPEC, {"weight_temperature": 1.25, "weight_clip": None}),
    (
        G1_SPEC,
        {"weight_temperature": 0.83, "neighbor_temperature": 0.41},
    ),
    (
        G2_SPEC,
        {
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "candidate_td_targets": "none",
            "prefix_slots": 5,
            "suffix_fill": "normalized_zero_mean_action",
            "advantage_reducer": "full_score_minus_all_prefix_mean",
            "weight_temperature": 0.83,
            "weight_gradient": "stop_gradient",
        },
    ),
    (
        G3_SPEC,
        {
            "candidate_source": (
                "same_transition_normalized_action_zero_mean_suffix_prefixes"
            ),
            "candidate_td_targets": "none",
            "prefix_slots": 5,
            "suffix_fill": "normalized_zero_mean_action",
            "advantage_reducer": "mean_adjacent_prefix_score_deltas",
            "weight_temperature": 0.83,
            "weight_gradient": "stop_gradient",
        },
    ),
]


def _required_cli_args() -> list[str]:
    return [
        "--config",
        "resolved.yaml",
        "--dataset",
        "cube.lance",
        "--seed",
        "0",
        "--initial-world-model-checkpoint",
        "epoch_10",
        "--frozen-latent-store",
        "frozen-store",
        "--split-indices",
        "split_indices.npz",
    ]


def test_training_cli_requires_explicit_protocol_and_split_artifact():
    parser = build_frozen_actor_free_td_parser(
        method_label="C",
        requires_neighbor_index=False,
    )
    args = parser.parse_args(_required_cli_args())
    assert args.config == "resolved.yaml"
    assert args.split_indices == "split_indices.npz"

    without_config = _required_cli_args()[2:]
    with pytest.raises(SystemExit):
        parser.parse_args(without_config)
    without_split = _required_cli_args()[:-2]
    with pytest.raises(SystemExit):
        parser.parse_args(without_split)


@pytest.mark.parametrize("unsafe_flag", ["--max-steps", "--skip-validation"])
def test_training_cli_rejects_non_smoke_shortcuts(unsafe_flag):
    argv = _required_cli_args()
    argv.extend([unsafe_flag, "1"] if unsafe_flag == "--max-steps" else [unsafe_flag])

    with pytest.raises(SystemExit):
        run_frozen_actor_free_td_cli(
            method_label="C",
            requires_neighbor_index=False,
            load_protocol=lambda path: pytest.fail(f"unexpected load: {path}"),
            train=lambda **kwargs: pytest.fail(f"unexpected train: {kwargs}"),
            argv=argv,
        )


def test_training_consumes_exact_external_split_without_regeneration(tmp_path):
    split_path = tmp_path / "split_indices.npz"
    train = np.asarray([4, 0, 3, 1], dtype=np.int64)
    validation = np.asarray([2], dtype=np.int64)
    np.savez_compressed(
        split_path,
        train_indices=train,
        val_indices=validation,
    )

    loaded_train, loaded_validation, manifest = load_bound_training_split(
        split_path,
        dataset_size=5,
        train_fraction=0.8,
        validation_fraction=0.2,
    )

    np.testing.assert_array_equal(loaded_train, train)
    np.testing.assert_array_equal(loaded_validation, validation)
    assert manifest["path"] == str(split_path.resolve())
    assert manifest["binding"] == "externally_supplied_exact_artifact"
    assert manifest["validation_array_key"] == "val_indices"


def test_training_rejects_split_that_does_not_partition_dataset(tmp_path):
    split_path = tmp_path / "split_indices.npz"
    np.savez_compressed(
        split_path,
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        validation_indices=np.asarray([4], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="partition every dataset clip"):
        load_bound_training_split(
            split_path,
            dataset_size=5,
            train_fraction=0.8,
            validation_fraction=0.2,
        )


@pytest.mark.parametrize(("spec", "objective"), METHOD_CASES)
def test_c_d_f_g1_g2_g3_train_directly_from_frozen_latent_batch(spec, objective):
    class NeverEncodingWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.unused_scale = nn.Parameter(torch.ones(()))
            self.encode_calls = 0

        def encode(self, data):
            del data
            self.encode_calls += 1
            raise AssertionError("Frozen-cache training must never call encode().")

    class NeighborIndex:
        def lookup(self, global_rows, *, device, dtype):
            batch, transitions = global_rows.shape
            return SimpleNamespace(
                actions=torch.randn(
                    batch,
                    transitions,
                    2,
                    25,
                    device=device,
                    dtype=dtype,
                ),
                distances=torch.ones(batch, transitions, 2, device=device),
                neighbor_rows=torch.zeros(
                    batch,
                    transitions,
                    2,
                    device=device,
                    dtype=torch.int64,
                ),
            )

    protocol = {
        "method": spec.method,
        "method_family": "actor_free_td_lewm",
        "variant": spec.variant,
        "model": {"embed_dim": 4},
        "sequence": {"history_frames": 3, "num_steps": 7, "frame_skip": 5},
        "successor": {
            "hidden_dim": 8,
            "gamma": 0.95,
            "target_world_ema_decay": 0.0,
            "target_successor_ema_decay": 0.995,
            "loss_warmup_fraction": 0.0,
        },
        "joint_objective": {
            "local_prediction_weight": 0.0,
            "goal_sampling_seed_offset": 1,
            **objective,
        },
        "loss": {"sigreg": {"knots": 3, "num_projections": 4, "weight": 0.0}},
        "image_preprocessing": {
            "mean": [0.0] * 3,
            "std": [1.0] * 3,
            "size": 2,
        },
    }
    neighbor_index = NeighborIndex() if spec.requires_neighbor_index else None
    module = _build_training_module(
        NeverEncodingWorld(),
        protocol,
        total_steps=1,
        spec=spec,
        action_block_dim=25,
        device_image_preprocessing=False,
        goal_generator=torch.Generator().manual_seed(7),
        neighbor_index=neighbor_index,
    )
    batch = {
        "_tdwm_frozen_latents": torch.randn(2, 7, 4),
        "action": torch.randn(2, 7, 25),
        "_tdwm_global_start": torch.tensor([100, 200]),
    }

    loss = module._forward_loss(batch, "train")
    loss.backward()

    assert torch.isfinite(loss)
    assert module.model.encode_calls == 0
    assert module.target_model.encode_calls == 0
    assert all(parameter.grad is None for parameter in module.model.parameters())
    assert any(
        parameter.grad is not None for parameter in module.successor.parameters()
    )
