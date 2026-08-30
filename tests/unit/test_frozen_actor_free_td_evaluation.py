from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import yaml

from tdwm.adapters.actor_free_td_lewm_c import METHOD_SPEC as C_SPEC
from tdwm.adapters.actor_free_td_lewm_d import METHOD_SPEC as D_SPEC
from tdwm.adapters.actor_free_td_lewm_f import METHOD_SPEC as F_SPEC
from tdwm.adapters.actor_free_td_lewm_g1 import METHOD_SPEC as G1_SPEC
from tdwm.adapters.actor_free_td_lewm_g2 import METHOD_SPEC as G2_SPEC
from tdwm.adapters.actor_free_td_lewm_g3 import METHOD_SPEC as G3_SPEC
from tdwm.adapters.frozen_actor_free_td_common import (
    validate_frozen_actor_free_td_payload,
)
from tdwm.evaluation.actor_free_td_lewm import (
    load_actor_free_td_evaluation_protocol,
    validate_actor_free_td_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_c import (
    load_actor_free_td_lewm_c_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_d import (
    load_actor_free_td_lewm_d_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_f import (
    load_actor_free_td_lewm_f_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_g1 import (
    load_actor_free_td_lewm_g1_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_g2 import (
    load_actor_free_td_lewm_g2_evaluation_protocol,
)
from tdwm.evaluation.actor_free_td_lewm_g3 import (
    load_actor_free_td_lewm_g3_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_common import (
    _resolve_frozen_dataset_source,
    configure_frozen_actor_free_td_evaluation_mode,
    frozen_actor_free_td_output_directory_name,
    validate_frozen_actor_free_td_checkpoint_protocol,
    validate_frozen_actor_free_td_evaluation_protocol,
)

BASE_CONFIG = (
    "configs/experiment/"
    "actor_free_td_lewm_parallel_real_cube_checkpoint_o50.yaml"
)
FROZEN_SOURCE_SHA = "a" * 64
FROZEN_RESULT_SHA = "b" * 64
FROZEN_MANIFEST_SHA = "c" * 64
FROZEN_DATASET_SHA = "d" * 64
FROZEN_LANCE_MANIFEST_SHA = "e" * 64

METHOD_CASES = [
    (C_SPEC, {"goal_projection_weight": 0.37}),
    (
        D_SPEC,
        {
            "weight_temperature": 0.73,
            "weight_clip": None,
            "weight_gradient": "stop_gradient",
        },
    ),
    (
        F_SPEC,
        {
            "weight_temperature": 1.25,
            "weight_clip": None,
            "weight_gradient": "stop_gradient",
        },
    ),
    (
        G1_SPEC,
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

PROTOCOL_LOADERS = [
    load_actor_free_td_lewm_c_evaluation_protocol,
    load_actor_free_td_lewm_d_evaluation_protocol,
    load_actor_free_td_lewm_f_evaluation_protocol,
    load_actor_free_td_lewm_g1_evaluation_protocol,
    load_actor_free_td_lewm_g2_evaluation_protocol,
    load_actor_free_td_lewm_g3_evaluation_protocol,
]


def _frozen_protocol(spec, objective: dict) -> dict:
    protocol = load_actor_free_td_evaluation_protocol(BASE_CONFIG)
    protocol.update(
        {
            "id": f"{spec.method}_cube_checkpoint_o50",
            "method": spec.method,
            "method_family": "actor_free_td_lewm",
            "variant": spec.variant,
            "display_name": spec.display_name,
            "pretrained_world_model": {
                "source_method": "lewm",
                "source_seed": 3072,
                "source_epoch": 10,
                "checkpoint_sha256": FROZEN_SOURCE_SHA,
                "frozen": True,
            },
            "joint_objective": deepcopy(objective),
        }
    )
    protocol["successor"].update(
        {
            "objective_version": 1,
            "pretrained_world_model_frozen": True,
            "training_branches": ["real_context"],
            "target_world_ema_decay": 0.0,
        }
    )
    protocol["inference_objective"]["goal_usage"] = (
        "training_objective_and_planning_linear_readout"
    )
    protocol["inference_objective"]["score_mode"] = "f_plus_g"
    protocol["dataset"]["source"] = {
        "file": "cube_single_expert_chunk1.h5",
        "size_bytes": 74_104_077_358,
        "sha256": FROZEN_DATASET_SHA,
    }
    protocol["dataset"]["lance"][
        "manifest_sha256"
    ] = FROZEN_LANCE_MANIFEST_SHA
    validate_frozen_actor_free_td_evaluation_protocol(protocol, spec=spec)
    return protocol


@pytest.fixture(params=list(zip(PROTOCOL_LOADERS, METHOD_CASES, strict=True)))
def standalone_protocol_fixture(request, tmp_path):
    loader, (spec, objective) = request.param
    protocol_path = tmp_path / f"{spec.method}_checkpoint.yaml"
    protocol_path.write_text(yaml.safe_dump(_frozen_protocol(spec, objective)))
    return loader, protocol_path, spec


def _checkpoint_for(protocol: dict, spec, objective: dict):
    successor_config = {
        "method": spec.method,
        "method_family": "actor_free_td_lewm",
        "variant": spec.variant,
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "embed_dim": protocol["model"]["embed_dim"],
        "action_dim": 25,
        "history_size": protocol["successor"]["history_size"],
        "hidden_dim": protocol["successor"]["hidden_dim"],
        "gamma": protocol["successor"]["gamma"],
        "feature_basis": protocol["successor"]["feature_basis"],
        "architecture": "actor_free_successor_head",
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
        "pretrained_world_model_sha256": FROZEN_SOURCE_SHA,
        "base_checkpoint_sha256": FROZEN_SOURCE_SHA,
        "training_branches": ["real_context"],
        **objective,
    }
    payload = {
        "method": spec.method,
        "method_family": "actor_free_td_lewm",
        "variant": spec.variant,
        "objective_version": 1,
        "deployment_checkpoint_version": 1,
        "epoch": 10,
        "global_step": 127_960,
        "world_model_state_dict": {},
        "successor_state_dict": {},
        "world_model_config": {"_target_": "fake.WorldModel"},
        "successor_config": successor_config,
        "pretrained_world_model_provenance": {
            "strategy": "frozen_pretrained_lewm",
            "source_method": "lewm",
            "source_seed": 3072,
            "source_epoch": 10,
            "source_checkpoint_sha256": FROZEN_SOURCE_SHA,
            "source_training_result_sha256": FROZEN_RESULT_SHA,
            "source_training_manifest_sha256": FROZEN_MANIFEST_SHA,
            "source_final_epoch": 10,
            "source_global_step": 127_960,
            "frozen": True,
        },
    }
    return successor_config, payload


def test_legacy_protocol_keeps_accepting_unrelated_extension_metadata():
    protocol = load_actor_free_td_evaluation_protocol(BASE_CONFIG)
    protocol["pretrained_world_model"] = {"legacy_extension": True}

    validate_actor_free_td_evaluation_protocol(protocol)


def test_each_standalone_o50_protocol_loads_with_its_method_specific_loader(
    standalone_protocol_fixture,
):
    loader, path, spec = standalone_protocol_fixture
    protocol = loader(path)

    assert protocol["method"] == spec.method
    assert protocol["variant"] == spec.variant
    assert protocol["successor"]["objective_version"] == 1


@pytest.mark.parametrize(("spec", "objective"), METHOD_CASES)
def test_frozen_methods_lock_pretraining_and_keep_successor_planning_modes(
    spec,
    objective,
):
    protocol = _frozen_protocol(spec, objective)
    config, payload = _checkpoint_for(protocol, spec, objective)

    assert protocol["successor"]["objective_version"] == 1
    assert protocol["successor"]["goal_conditioning"] == "none"
    assert protocol["successor"]["training_branches"] == ["real_context"]
    assert protocol["pretrained_world_model"]["frozen"] is True
    assert protocol["inference_objective"]["goal_enters_successor_head"] is False
    assert protocol["inference_objective"]["score_mode"] == "f_plus_g"
    validate_frozen_actor_free_td_payload(payload, spec=spec)
    validate_frozen_actor_free_td_checkpoint_protocol(
        payload=payload,
        successor_config=config,
        protocol=protocol,
        spec=spec,
    )

    for score_mode in ("f_only", "g_only", "f_plus_g"):
        configured = configure_frozen_actor_free_td_evaluation_mode(
            protocol,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
        )
        assert configured["inference_objective"]["score_mode"] == score_mode


def test_formal_o50_requires_the_completed_deployment_checkpoint():
    spec, objective = METHOD_CASES[0]
    protocol = _frozen_protocol(spec, objective)
    config, payload = _checkpoint_for(protocol, spec, objective)

    early_epoch = deepcopy(payload)
    early_epoch["epoch"] = 1
    with pytest.raises(ValueError, match=r"checkpoint\.epoch must be 10"):
        validate_frozen_actor_free_td_checkpoint_protocol(
            payload=early_epoch,
            successor_config=config,
            protocol=protocol,
            spec=spec,
        )

    early_step = deepcopy(payload)
    early_step["global_step"] = 12_796
    with pytest.raises(ValueError, match=r"checkpoint\.global_step must be 127960"):
        validate_frozen_actor_free_td_checkpoint_protocol(
            payload=early_step,
            successor_config=config,
            protocol=protocol,
            spec=spec,
        )

    validate_frozen_actor_free_td_checkpoint_protocol(
        payload=early_epoch,
        successor_config=config,
        protocol=protocol,
        spec=spec,
        require_formal_completion=False,
    )


def test_default_output_names_separate_score_and_run_modes():
    spec, objective = METHOD_CASES[2]
    protocol = _frozen_protocol(spec, objective)

    names = {
        frozen_actor_free_td_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="f_only",
        ),
        frozen_actor_free_td_output_directory_name(
            protocol,
            smoke=True,
            pilot=False,
            score_mode="g_only",
        ),
        frozen_actor_free_td_output_directory_name(
            protocol,
            smoke=False,
            pilot=True,
            score_mode="f_plus_g",
        ),
    }

    assert names == {
        f"{spec.method}_cube_o50_f_only_formal",
        f"{spec.method}_cube_o50_g_only_smoke",
        f"{spec.method}_cube_o50_f_plus_g_pilot",
    }


def test_frozen_protocol_requires_dataset_source_and_lance_manifest_hashes():
    spec, objective = METHOD_CASES[1]
    protocol = _frozen_protocol(spec, objective)

    missing_source = deepcopy(protocol)
    del missing_source["dataset"]["source"]
    with pytest.raises(ValueError, match=r"protocol\.dataset\.source"):
        validate_frozen_actor_free_td_evaluation_protocol(
            missing_source,
            spec=spec,
        )

    uppercase_source_sha = deepcopy(protocol)
    uppercase_source_sha["dataset"]["source"]["sha256"] = "A" * 64
    with pytest.raises(ValueError, match=r"dataset\.source\.sha256"):
        validate_frozen_actor_free_td_evaluation_protocol(
            uppercase_source_sha,
            spec=spec,
        )

    missing_manifest_sha = deepcopy(protocol)
    del missing_manifest_sha["dataset"]["lance"]["manifest_sha256"]
    with pytest.raises(ValueError, match=r"dataset\.lance\.manifest_sha256"):
        validate_frozen_actor_free_td_evaluation_protocol(
            missing_manifest_sha,
            spec=spec,
        )


def test_hdf5_evaluation_input_is_bound_to_the_protocol_source_hash(tmp_path):
    dataset_path = tmp_path / "cube_single_expert_chunk1.h5"
    dataset_path.write_bytes(b"audited")
    source_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset_config = {
        "expected_size_bytes": dataset_path.stat().st_size,
        "accepted_size_bytes": [dataset_path.stat().st_size],
        "source": {
            "file": dataset_path.name,
            "size_bytes": dataset_path.stat().st_size,
            "sha256": source_sha,
        },
        "lance": {
            "manifest_suffix": ".manifest.json",
            "manifest_sha256": "f" * 64,
            "image_codec": "jpeg",
            "jpeg_quality": 100,
        },
    }

    resolved = _resolve_frozen_dataset_source(dataset_path, dataset_config)

    assert resolved["sha256"] == source_sha
    assert resolved["source_sha256"] == source_sha
    assert resolved["source_size_bytes"] == dataset_path.stat().st_size
    assert resolved["conversion_manifest_sha256"] is None

    dataset_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="HDF5 SHA-256"):
        _resolve_frozen_dataset_source(dataset_path, dataset_config)


def test_lance_evaluation_input_binds_manifest_and_hdf5_source_hashes(tmp_path):
    dataset_path = tmp_path / "cube.lance"
    dataset_path.mkdir()
    source_sha = "a" * 64
    source_size = 123
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "cube_single_expert_chunk1.h5",
            "size_bytes": source_size,
            "sha256": source_sha,
        },
        "destination": {
            "name": dataset_path.name,
            "format": "lance",
            "size_bytes": 456,
            "image_codec": "jpeg",
            "jpeg_quality": 100,
        },
        "conversion": {
            "api": "swm.data.convert",
            "stable_worldmodel_version": "0.1.1",
            "mode": "error",
        },
    }
    manifest_path = tmp_path / "cube.lance.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    dataset_config = {
        "expected_size_bytes": source_size,
        "accepted_size_bytes": [source_size],
        "source": {
            "file": manifest["source"]["name"],
            "size_bytes": source_size,
            "sha256": source_sha,
        },
        "lance": {
            "manifest_suffix": ".manifest.json",
            "manifest_sha256": manifest_sha,
            "image_codec": "jpeg",
            "jpeg_quality": 100,
        },
    }

    resolved = _resolve_frozen_dataset_source(dataset_path, dataset_config)

    assert resolved["conversion_manifest_sha256"] == manifest_sha
    assert resolved["source_sha256"] == source_sha
    assert resolved["source_size_bytes"] == source_size

    wrong_manifest = deepcopy(dataset_config)
    wrong_manifest["lance"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Lance manifest SHA-256"):
        _resolve_frozen_dataset_source(dataset_path, wrong_manifest)

    wrong_source = deepcopy(dataset_config)
    wrong_source["source"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Lance source SHA-256"):
        _resolve_frozen_dataset_source(dataset_path, wrong_source)


def test_frozen_protocol_rejects_sha_freeze_and_training_branch_drift():
    spec, objective = METHOD_CASES[0]
    protocol = _frozen_protocol(spec, objective)

    bad_sha = deepcopy(protocol)
    bad_sha["pretrained_world_model"]["checkpoint_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_frozen_actor_free_td_evaluation_protocol(bad_sha, spec=spec)

    trainable = deepcopy(protocol)
    trainable["pretrained_world_model"]["frozen"] = False
    with pytest.raises(ValueError, match="pretrained_world_model.frozen"):
        validate_frozen_actor_free_td_evaluation_protocol(trainable, spec=spec)

    extra_branch = deepcopy(protocol)
    extra_branch["successor"]["training_branches"] = [
        "real_context",
        "predicted_context",
    ]
    with pytest.raises(ValueError, match="successor.training_branches"):
        validate_frozen_actor_free_td_evaluation_protocol(extra_branch, spec=spec)


def test_frozen_checkpoint_sha_must_match_protocol_and_base_checkpoint():
    spec, objective = METHOD_CASES[2]
    protocol = _frozen_protocol(spec, objective)
    config, payload = _checkpoint_for(protocol, spec, objective)

    wrong_pretrained_sha = deepcopy(config)
    wrong_pretrained_sha["pretrained_world_model_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="pretrained_world_model_sha256"):
        validate_frozen_actor_free_td_checkpoint_protocol(
            payload=payload,
            successor_config=wrong_pretrained_sha,
            protocol=protocol,
            spec=spec,
        )

    wrong_base_sha = deepcopy(config)
    wrong_base_sha["base_checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        validate_frozen_actor_free_td_checkpoint_protocol(
            payload=payload,
            successor_config=wrong_base_sha,
            protocol=protocol,
            spec=spec,
        )


def test_frozen_checkpoint_rejects_incomplete_or_unverified_source_training():
    spec, objective = METHOD_CASES[1]
    protocol = _frozen_protocol(spec, objective)
    config, payload = _checkpoint_for(protocol, spec, objective)

    incomplete = deepcopy(payload)
    incomplete["pretrained_world_model_provenance"]["source_final_epoch"] = 1
    with pytest.raises(ValueError, match="source_final_epoch"):
        validate_frozen_actor_free_td_payload(incomplete, spec=spec)

    wrong_step = deepcopy(payload)
    wrong_step["pretrained_world_model_provenance"]["source_global_step"] = 12_796
    with pytest.raises(ValueError, match="source_global_step"):
        validate_frozen_actor_free_td_payload(wrong_step, spec=spec)

    missing_manifest_sha = deepcopy(payload)
    del missing_manifest_sha["pretrained_world_model_provenance"][
        "source_training_manifest_sha256"
    ]
    with pytest.raises(ValueError, match="source_training_manifest_sha256"):
        validate_frozen_actor_free_td_payload(missing_manifest_sha, spec=spec)

    wrong_source_sha = deepcopy(payload)
    wrong_source_sha["pretrained_world_model_provenance"][
        "source_checkpoint_sha256"
    ] = "d" * 64
    with pytest.raises(ValueError, match="differs from successor_config"):
        validate_frozen_actor_free_td_payload(wrong_source_sha, spec=spec)

    validate_frozen_actor_free_td_checkpoint_protocol(
        payload=payload,
        successor_config=config,
        protocol=protocol,
        spec=spec,
    )
