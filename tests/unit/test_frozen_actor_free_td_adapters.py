from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from tdwm.adapters.actor_free_td_lewm_c import (
    METHOD_SPEC as C_SPEC,
)
from tdwm.adapters.actor_free_td_lewm_c import (
    load_actor_free_td_lewm_c_checkpoint,
)
from tdwm.adapters.actor_free_td_lewm_d import (
    METHOD_SPEC as D_SPEC,
)
from tdwm.adapters.actor_free_td_lewm_d import (
    load_actor_free_td_lewm_d_checkpoint,
)
from tdwm.adapters.actor_free_td_lewm_f import (
    METHOD_SPEC as F_SPEC,
)
from tdwm.adapters.actor_free_td_lewm_f import (
    load_actor_free_td_lewm_f_checkpoint,
)
from tdwm.adapters.actor_free_td_lewm_g1 import (
    METHOD_SPEC as G1_SPEC,
)
from tdwm.adapters.actor_free_td_lewm_g1 import (
    load_actor_free_td_lewm_g1_checkpoint,
)
from tdwm.methods.actor_free_td_lewm import ActorFreeSuccessorHead


def _frozen_source_provenance(source_sha: str) -> dict:
    return {
        "strategy": "frozen_pretrained_lewm",
        "source_method": "lewm",
        "source_seed": 3072,
        "source_epoch": 10,
        "source_checkpoint_sha256": source_sha,
        "source_training_result_sha256": "b" * 64,
        "source_training_manifest_sha256": "c" * 64,
        "source_final_epoch": 10,
        "source_global_step": 127_960,
        "frozen": True,
    }


METHOD_CASES = [
    (C_SPEC, load_actor_free_td_lewm_c_checkpoint, {"goal_projection_weight": 0.37}),
    (
        D_SPEC,
        load_actor_free_td_lewm_d_checkpoint,
        {
            "weight_temperature": 0.73,
            "weight_clip": None,
            "weight_gradient": "stop_gradient",
        },
    ),
    (
        F_SPEC,
        load_actor_free_td_lewm_f_checkpoint,
        {
            "weight_temperature": 1.25,
            "weight_clip": None,
            "weight_gradient": "stop_gradient",
        },
    ),
    (
        G1_SPEC,
        load_actor_free_td_lewm_g1_checkpoint,
        {
            "candidate_source": (
                "other_episode_frozen_latent_knn_real_action_blocks"
            ),
            "candidate_td_targets": "none",
            "neighbor_temperature": 0.41,
            "weight_temperature": 0.83,
            "weight_gradient": "stop_gradient",
            "neighbors_per_anchor": 2,
        },
    ),
]


@pytest.mark.parametrize(("spec", "loader", "objective"), METHOD_CASES)
def test_standalone_frozen_checkpoint_loader_restores_one_method_and_provenance(
    tmp_path,
    spec,
    loader,
    objective,
):
    source_sha = "a" * 64
    world_model = nn.Linear(2, 2)
    successor = ActorFreeSuccessorHead(
        embed_dim=2,
        action_dim=25,
        history_size=2,
        hidden_dim=5,
    )
    successor_config = {
        "method": spec.method,
        "method_family": "actor_free_td_lewm",
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "embed_dim": 2,
        "action_dim": 25,
        "history_size": 2,
        "hidden_dim": 5,
        "gamma": 0.95,
        "variant": spec.variant,
        "architecture": "actor_free_successor_head",
        "feature_basis": "augmented_latent_squared_distance",
        "goal_conditioning": "none",
        "action_conditioning": "dataset_current_action",
        "bootstrap_action": "dataset_next_action",
        "terminal_source": "next_action_nan_invalid",
        "actor": "none",
        "reward": "none",
        "predicted_context_detach": True,
        "pretrained_world_model_frozen": True,
        "pretrained_world_model_source_method": "lewm",
        "pretrained_world_model_source_seed": 3072,
        "pretrained_world_model_source_epoch": 10,
        "pretrained_world_model_sha256": source_sha,
        "base_checkpoint_sha256": source_sha,
        "training_branches": ["real_context"],
        **objective,
    }
    checkpoint_payload = {
        "method": spec.method,
        "method_family": "actor_free_td_lewm",
        "variant": spec.variant,
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "successor_state_dict": successor.state_dict(),
        "successor_config": successor_config,
        "world_model_state_dict": world_model.state_dict(),
        "world_model_config": {
            "_target_": "torch.nn.Linear",
            "in_features": 2,
            "out_features": 2,
        },
        "pretrained_world_model_provenance": _frozen_source_provenance(source_sha),
    }
    checkpoint = tmp_path / f"{spec.method}.pt"
    torch.save(checkpoint_payload, checkpoint)

    restored_world, restored_head, restored_config, payload = loader(checkpoint)

    assert restored_config == successor_config
    assert payload["method"] == spec.method
    assert payload["variant"] == spec.variant
    assert not any(parameter.requires_grad for parameter in restored_world.parameters())
    assert not any(parameter.requires_grad for parameter in restored_head.parameters())

    mismatched = deepcopy(checkpoint_payload)
    mismatched["successor_config"] = deepcopy(successor_config)
    mismatched["successor_config"]["base_checkpoint_sha256"] = "d" * 64
    torch.save(mismatched, checkpoint)
    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        loader(checkpoint)

    incomplete = deepcopy(checkpoint_payload)
    incomplete["pretrained_world_model_provenance"]["source_final_epoch"] = 1
    torch.save(incomplete, checkpoint)
    with pytest.raises(ValueError, match="source_final_epoch"):
        loader(checkpoint)
