from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import tdwm.training.rf_successor_lewm as rf_successor_training
from tdwm.adapters.rf_successor_lewm import (
    RewardFreeSuccessorLeWM,
    load_rf_successor_checkpoint,
)
from tdwm.evaluation.rf_successor_lewm import (
    load_rf_successor_evaluation_protocol,
)
from tdwm.methods.rf_successor_lewm import (
    ActionPrefixMomentHead,
    ActionPrefixSuccessorHead,
    finite_horizon_successor_targets,
    left_pad_latent_history,
    multi_horizon_successor_objective,
    successor_recurrence_residual,
)
from tdwm.methods.successor_geometry import (
    goal_cost_weights,
    latent_goal_cost,
    successor_feature_basis,
)
from tdwm.training.rf_successor_lewm import (
    DECODED_FRAME_STORE_ENV,
    _encode_online_and_target,
    _prepare_decoded_frame_store,
    build_history_context_batch,
    build_multi_horizon_windows,
    load_rf_successor_training_protocol,
)


class FakeWorldModel(nn.Module):
    def __init__(self, predicted: torch.Tensor) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.predicted = predicted

    def rollout(self, info, action_sequence, history_size=None):
        del info, history_size
        return {"predicted_emb": self.predicted.to(action_sequence)}

    def encode(self, info):
        return {"emb": info["pixels"]}


class FixedSuccessor(nn.Module):
    def __init__(self, latent: torch.Tensor, *, history_size: int, action_dim: int):
        super().__init__()
        self.embed_dim = int(latent.numel())
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.register_buffer("value", successor_feature_basis(latent))

    def forward(self, history, actions):
        return self.value.to(actions).expand(
            *actions.shape[:-1], self.value.shape[-1]
        )


class _FakeDecodedFrameStore:
    def __init__(
        self,
        manifest_path: Path,
        *,
        row_count: int = 4,
        shape: tuple[int, ...] = (4, 3, 2, 2),
        dtype: object = np.uint8,
        source: dict | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = "b" * 64
        self.data_path = manifest_path.with_suffix(".bin")
        self.sha256 = "a" * 64
        self.row_count = row_count
        self.shape = shape
        self.frame_shape = shape[1:]
        self.dtype = np.dtype(dtype)
        self.source = source or {}
        self.metadata = {
            "decoder": {
                "api": "torchvision.io.decode_jpeg",
                "mode": "RGB",
            },
            "source_pixel_verification": {
                "method": "full_redecode_audit",
                "row_count": row_count,
                "decoded_sha256": self.sha256,
                "data_sha256": self.sha256,
                "matches_data_sha256": True,
                "decoder": {
                    "api": "torchvision.io.decode_jpeg",
                    "mode": "RGB",
                },
                "completed_at_utc": "2026-08-27T00:00:00+00:00",
            },
        }
        self.preload_calls = 0
        self.preload_verify_sha256 = None
        self.sha256_verified = False
        self.page_cache_warmed = False

    def preload(self, *, verify_sha256=False):
        self.preload_calls += 1
        self.preload_verify_sha256 = verify_sha256
        self.sha256_verified = bool(verify_sha256)
        self.page_cache_warmed = True
        return self


class _FakeUnwrappedLanceDataset:
    def __init__(self) -> None:
        self.lengths = np.array([2, 2], dtype=np.int32)
        self.offsets = np.array([0, 2], dtype=np.int32)


def _decoded_store_protocol() -> dict:
    return {
        "dataset": {"expected_transitions": 4, "expected_episodes": 2},
        "image_preprocessing": {"size": 2},
    }


def _valid_decoded_store_inputs(monkeypatch, tmp_path):
    conversion_manifest = tmp_path / "cube.lance.manifest.json"
    conversion_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    conversion_sha256 = hashlib.sha256(conversion_manifest.read_bytes()).hexdigest()
    dataset = _FakeUnwrappedLanceDataset()
    dataset_source = {
        "path": str(tmp_path / "current-copy.lance"),
        "format": "lance",
        "conversion_manifest_path": str(conversion_manifest),
    }
    source = {
        "path": str(tmp_path / "builder-machine.lance"),
        "format": "lance",
        "row_count": 4,
        "manifest_sha256": conversion_sha256,
        "pixel_column": "pixels",
        "row_mapping": "binary frame i is decoded from Lance pixels row i",
        "episode_count": 2,
        "episode_lengths_sha256": rf_successor_training._canonical_int64_sha256(
            dataset.lengths,
            label="fixture.lengths",
        ),
        "episode_offsets_sha256": rf_successor_training._canonical_int64_sha256(
            dataset.offsets,
            label="fixture.offsets",
        ),
        "episode_jpeg_payload_bytes": [100, 120],
    }
    manifest_path = (tmp_path / "frames.json").resolve()
    store = _FakeDecodedFrameStore(manifest_path, source=source)
    monkeypatch.setenv(DECODED_FRAME_STORE_ENV, str(manifest_path))
    monkeypatch.setattr(
        rf_successor_training.DecodedFrameStore,
        "from_manifest",
        lambda path: store if path == manifest_path else None,
    )
    return dataset_source, dataset, store, conversion_sha256


def test_decoded_frame_store_is_disabled_without_environment_variable(monkeypatch):
    monkeypatch.delenv(DECODED_FRAME_STORE_ENV, raising=False)

    store, metadata = _prepare_decoded_frame_store(
        _decoded_store_protocol(),
        {},
        object(),
    )

    assert store is None
    assert metadata is None


def test_decoded_frame_store_is_strictly_validated_preloaded_and_recorded(
    monkeypatch, tmp_path
):
    dataset_source, dataset, store, conversion_sha256 = (
        _valid_decoded_store_inputs(monkeypatch, tmp_path)
    )

    loaded, metadata = _prepare_decoded_frame_store(
        _decoded_store_protocol(),
        dataset_source,
        dataset,
    )

    assert loaded is store
    assert store.preload_calls == 1
    assert store.preload_verify_sha256 is True
    assert metadata == {
        "manifest_path": str(store.manifest_path),
        "manifest_sha256": "b" * 64,
        "data_path": str(store.data_path),
        "data_sha256": "a" * 64,
        "data_sha256_verified": True,
        "row_count": 4,
        "shape": [4, 3, 2, 2],
        "dtype": "uint8",
        "preloaded": True,
        "page_cache_warmed": True,
        "source_binding": {
            "verified": True,
            "conversion_manifest_path": str(
                Path(dataset_source["conversion_manifest_path"]).resolve()
            ),
            "conversion_manifest_sha256": conversion_sha256,
            "dataset_path": str(Path(dataset_source["path"]).resolve()),
            "store_source_path": store.source["path"],
            "path_match": False,
            "row_count": 4,
            "episode_count": 2,
            "episode_lengths_sha256": store.source[
                "episode_lengths_sha256"
            ],
            "episode_offsets_sha256": store.source[
                "episode_offsets_sha256"
            ],
            "episode_payload_count": 2,
            "pixel_column": "pixels",
            "row_mapping": "binary frame i is decoded from Lance pixels row i",
            "decoder": {
                "api": "torchvision.io.decode_jpeg",
                "mode": "RGB",
            },
            "source_pixel_verification": store.metadata[
                "source_pixel_verification"
            ],
        },
    }


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("row_count", 3, "row_count differs"),
        ("shape", (4, 3, 3, 3), "shape differs"),
        ("dtype", np.dtype(np.float32), "dtype must be uint8"),
    ],
)
def test_decoded_frame_store_rejects_protocol_mismatches(
    monkeypatch, tmp_path, attribute, value, message
):
    dataset_source, dataset, store, _ = _valid_decoded_store_inputs(
        monkeypatch,
        tmp_path,
    )
    setattr(store, attribute, value)
    if attribute == "shape":
        store.frame_shape = value[1:]

    with pytest.raises(ValueError, match=message):
        _prepare_decoded_frame_store(
            _decoded_store_protocol(),
            dataset_source,
            dataset,
        )

    assert store.preload_calls == 0


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    [
        ("source", "format", "hdf5", "source.format"),
        ("source", "manifest_sha256", "0" * 64, "manifest_sha256"),
        ("source", "pixel_column", "pixels_alt", "pixel_column"),
        ("source", "row_mapping", "episode-local rows", "row_mapping"),
        ("source", "episode_lengths_sha256", "1" * 64, "episode_lengths"),
        ("source", "episode_offsets_sha256", "2" * 64, "episode_offsets"),
        ("source", "episode_jpeg_payload_bytes", [100], "episode_jpeg_payload"),
        ("decoder", "api", "PIL.Image.open", "decoder.api"),
        ("decoder", "mode", "BGR", "decoder.mode"),
        (
            "source_pixel_verification",
            "decoded_sha256",
            "0" * 64,
            "source_pixel_verification.decoded_sha256",
        ),
        (
            "source_pixel_verification",
            "matches_data_sha256",
            False,
            "matches_data_sha256",
        ),
    ],
)
def test_decoded_frame_store_rejects_wrong_source_binding(
    monkeypatch,
    tmp_path,
    location,
    field,
    value,
    message,
):
    dataset_source, dataset, store, _ = _valid_decoded_store_inputs(
        monkeypatch,
        tmp_path,
    )
    target = store.source if location == "source" else store.metadata[location]
    target[field] = value

    with pytest.raises(ValueError, match=message):
        _prepare_decoded_frame_store(
            _decoded_store_protocol(),
            dataset_source,
            dataset,
        )

    assert store.preload_calls == 0


def test_direct_successor_targets_include_single_and_all_multi_step_values():
    torch.manual_seed(1)
    future = torch.randn(2, 4, 5)
    goal = torch.randn(2, 5)
    gamma = 0.8

    targets = finite_horizon_successor_targets(future, gamma=gamma)
    weights = goal_cost_weights(goal).unsqueeze(1)
    queried_cost = (targets * weights).sum(dim=-1)
    powers = gamma ** torch.arange(4, dtype=future.dtype)
    stage_cost = latent_goal_cost(future, goal.unsqueeze(1))
    expected = (stage_cost * powers).cumsum(dim=1) / powers.cumsum(dim=0)

    assert torch.allclose(targets[:, 0], successor_feature_basis(future[:, 0]))
    assert torch.allclose(queried_cost, expected, atol=1e-6)


def test_successor_target_exactly_satisfies_latent_increment_recurrence():
    torch.manual_seed(2)
    future = torch.randn(3, 5, 7)
    target = finite_horizon_successor_targets(future, gamma=0.95)

    residual = successor_recurrence_residual(target, future, gamma=0.95)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_action_prefix_head_is_causal_and_has_no_goal_or_policy_api():
    torch.manual_seed(3)
    head = ActionPrefixSuccessorHead(
        embed_dim=4,
        action_dim=2,
        history_size=3,
        hidden_dim=8,
    ).eval()
    history = torch.randn(2, 3, 4)
    actions = torch.randn(2, 5, 2)
    changed = actions.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])

    original = head(history, actions)
    perturbed = head(history, changed)

    assert original.shape == (2, 5, 6)
    assert torch.allclose(original[:, :3], perturbed[:, :3])
    assert torch.equal(original[..., -1], torch.ones_like(original[..., -1]))
    assert not hasattr(head, "policy")


def test_masked_history_head_left_pads_without_repeating_observations():
    torch.manual_seed(31)
    head = ActionPrefixSuccessorHead(
        embed_dim=2,
        action_dim=1,
        history_size=3,
        hidden_dim=8,
        masked_history=True,
    ).eval()
    current = torch.tensor([[[2.0, -1.0]]])
    actions = torch.randn(1, 2, 1)
    padded, mask = left_pad_latent_history(current, history_size=3)

    inferred = head(current, actions)
    explicit = head(padded, actions, history_mask=mask)

    assert padded.tolist() == [[[0.0, 0.0], [0.0, 0.0], [2.0, -1.0]]]
    assert mask.tolist() == [[0.0, 0.0, 1.0]]
    assert torch.allclose(inferred, explicit)
    assert head.history_encoder[0].in_features == 3 * 2 + 3


def test_masked_history_requires_a_binary_right_aligned_validity_suffix():
    history = torch.randn(1, 3, 2)

    with pytest.raises(ValueError, match="right-aligned suffix"):
        left_pad_latent_history(
            history,
            history_size=3,
            history_mask=torch.tensor([[1.0, 0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="binary"):
        left_pad_latent_history(
            history,
            history_size=3,
            history_mask=torch.tensor([[0.0, 0.5, 1.0]]),
        )


def test_joint_objective_updates_world_rollout_and_not_target_latents():
    torch.manual_seed(4)
    head = ActionPrefixSuccessorHead(
        embed_dim=3,
        action_dim=2,
        history_size=2,
        hidden_dim=7,
    )
    history = torch.randn(2, 2, 3, requires_grad=True)
    actions = torch.randn(2, 4, 2)
    predicted = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3, requires_grad=True)

    output = multi_horizon_successor_objective(
        head,
        history,
        actions,
        predicted,
        target,
        gamma=0.9,
    )
    loss = output.latent_loss + output.successor_loss + output.recurrence_loss
    loss.backward()

    assert history.grad is not None and torch.count_nonzero(history.grad) > 0
    assert predicted.grad is not None and torch.count_nonzero(predicted.grad) > 0
    assert target.grad is None
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_multi_horizon_windows_align_history_actions_and_targets():
    latents = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    target = latents + 100.0
    actions = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)

    windows = build_multi_horizon_windows(
        latents,
        target,
        actions,
        history_size=3,
        horizon=2,
    )

    assert windows.count_per_clip == 5
    assert torch.equal(windows.history[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(
        windows.rollout_actions[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
    )
    assert torch.equal(windows.action_prefix[0, :, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(windows.target_future[0, :, 0], torch.tensor([103.0, 104.0]))
    assert torch.equal(windows.history[-1, :, 0], torch.tensor([4.0, 5.0, 6.0]))


def test_history_context_batch_aligns_h1_h2_h3_with_the_same_future():
    latents = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    windows = build_multi_horizon_windows(
        latents,
        latents + 100.0,
        latents + 200.0,
        history_size=3,
        horizon=2,
    )
    one_window = type(windows)(
        history=windows.history[:1],
        rollout_actions=windows.rollout_actions[:1],
        action_prefix=windows.action_prefix[:1],
        target_future=windows.target_future[:1],
        count_per_clip=1,
    )

    contexts = build_history_context_batch(one_window)

    assert contexts.padded_history.squeeze(-1).tolist() == [
        [0.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ]
    assert contexts.history_mask.tolist() == [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    assert contexts.action_prefix[:, :, 0].tolist() == [
        [202.0, 203.0],
        [202.0, 203.0],
        [202.0, 203.0],
    ]
    assert contexts.target_future[:, :, 0].tolist() == [
        [103.0, 104.0],
        [103.0, 104.0],
        [103.0, 104.0],
    ]


class _MutatingEncoder(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.action_scale = nn.Parameter(torch.tensor(scale))
        self.last_input = None

    def encode(self, info):
        self.last_input = info
        info["emb"] = info["pixels"] * self.action_scale
        info["act_emb"] = info["action"] * self.action_scale
        return info


def test_online_and_target_encoders_cannot_overwrite_online_action_embeddings():
    online = _MutatingEncoder(2.0)
    target = _MutatingEncoder(7.0)
    encoder_input = {
        "pixels": torch.ones(1, 2, 1),
        "action": torch.ones(1, 2, 1),
    }

    embeddings, action_embeddings, target_embeddings = _encode_online_and_target(
        online,
        target,
        encoder_input,
    )
    (embeddings.sum() + action_embeddings.sum()).backward()

    assert torch.equal(action_embeddings, torch.full_like(action_embeddings, 2.0))
    assert torch.equal(target_embeddings, torch.full_like(target_embeddings, 7.0))
    assert online.last_input is not target.last_input
    assert online.last_input is not encoder_input
    assert target.last_input is not encoder_input
    assert "emb" not in encoder_input and "act_emb" not in encoder_input
    assert online.action_scale.grad is not None
    assert target.action_scale.grad is None


def test_planner_queries_supplied_prefix_without_an_actor():
    # Three observed latents followed by two future latents.
    predicted = torch.tensor(
        [[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]]]
    )
    successor = FixedSuccessor(
        torch.tensor([2.0, 0.0]), history_size=3, action_dim=1
    )
    adapter = RewardFreeSuccessorLeWM(
        FakeWorldModel(predicted),
        successor,
        max_horizon=2,
        successor_weight=1.0,
        terminal_weight=0.5,
        clamp_successor_cost=False,
    )
    info = {
        "pixels": torch.zeros(1, 1, 3, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }
    actions = torch.zeros(1, 1, 2, 1)

    cost = adapter.get_cost(info, actions)

    # Successor point [2, 0] has mean squared cost 2; terminal [4, 0] has 8.
    assert torch.allclose(cost, torch.tensor([[6.0]]))
    assert not hasattr(adapter, "get_action")


def test_planner_rejects_missing_history_instead_of_repeating_one_frame():
    predicted = torch.zeros(1, 1, 3, 2)
    successor = FixedSuccessor(
        torch.tensor([0.0, 0.0]), history_size=3, action_dim=1
    )
    adapter = RewardFreeSuccessorLeWM(
        FakeWorldModel(predicted),
        successor,
        max_horizon=2,
        terminal_weight=0.5,
    )
    info = {
        "pixels": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }

    with pytest.raises(RuntimeError, match="expected 3 latent frames, found 1"):
        adapter.get_cost(info, torch.zeros(1, 1, 2, 1))


def test_planner_can_query_the_terminal_predicted_moment():
    head = ActionPrefixMomentHead(
        embed_dim=2,
        action_dim=1,
        history_size=1,
        hidden_dim=4,
        gamma=1.0,
    )

    def fixed_moments(history, actions):
        del history
        latent = torch.tensor([[2.0, 0.0], [4.0, 0.0]], device=actions.device)
        moments = successor_feature_basis(latent).to(actions)
        return moments.expand(*actions.shape[:-2], -1, -1)

    head.predict_moments = fixed_moments
    adapter = RewardFreeSuccessorLeWM(
        FakeWorldModel(torch.zeros(1, 1, 1, 2)),
        head,
        max_horizon=2,
        clamp_successor_cost=False,
        planning_query="terminal_moment",
    )
    info = {
        "pixels": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }

    cost = adapter.get_cost(info, torch.zeros(1, 1, 2, 1))

    # The terminal predicted latent is [4, 0], whose mean squared cost is 8.
    assert torch.allclose(cost, torch.tensor([[8.0]]))


def test_planner_can_project_predicted_moments_onto_exact_latent_geometry():
    head = ActionPrefixMomentHead(
        embed_dim=2,
        action_dim=1,
        history_size=1,
        hidden_dim=4,
        gamma=1.0,
    )

    def inconsistent_moments(history, actions):
        del history
        latent = torch.tensor([[2.0, 0.0], [4.0, 0.0]], device=actions.device)
        moments = successor_feature_basis(latent).to(actions)
        moments[..., -2] = 0.0
        return moments.expand(*actions.shape[:-2], -1, -1)

    head.predict_moments = inconsistent_moments
    adapter = RewardFreeSuccessorLeWM(
        FakeWorldModel(torch.zeros(1, 1, 1, 2)),
        head,
        max_horizon=2,
        clamp_successor_cost=False,
        planning_query="manifold_projected_successor",
    )
    info = {
        "pixels": torch.zeros(1, 1, 1, 2),
        "goal_emb": torch.zeros(1, 1, 2),
    }

    cost = adapter.get_cost(info, torch.zeros(1, 1, 2, 1))

    # Reprojection ignores the inconsistent norm and averages costs 2 and 8.
    assert torch.allclose(cost, torch.tensor([[5.0]]))


def test_reward_free_successor_checkpoint_round_trip(tmp_path):
    head = ActionPrefixSuccessorHead(
        embed_dim=4, action_dim=3, history_size=2, hidden_dim=6
    )
    config = {
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "max_horizon": 5,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
    }
    checkpoint = tmp_path / "rf_successor.pt"
    torch.save(
        {
            "method": "rf_successor_lewm",
            "objective_version": 1,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert restored_config == config
    assert payload["method"] == "rf_successor_lewm"
    for name, value in head.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_masked_history_successor_checkpoint_round_trip(tmp_path):
    head = ActionPrefixSuccessorHead(
        embed_dim=4,
        action_dim=3,
        history_size=2,
        hidden_dim=6,
        masked_history=True,
    )
    config = {
        "objective_version": 12,
        "architecture": "masked_history_causal_gru_action_prefix",
        "embed_dim": 4,
        "action_dim": 3,
        "history_size": 2,
        "hidden_dim": 6,
        "max_horizon": 5,
        "goal_conditioning": "none",
        "action_conditioning": "causal_prefix",
        "history_padding": "left_zero",
        "history_masking": "explicit_validity",
        "history_supervision": "all_prefix_lengths",
    }
    checkpoint = tmp_path / "rf_successor_masked.pt"
    torch.save(
        {
            "method": "rf_successor_lewm",
            "objective_version": 12,
            "deployment_checkpoint_version": 1,
            "world_model_state_dict": {},
            "successor_state_dict": head.state_dict(),
            "successor_config": config,
        },
        checkpoint,
    )

    restored, restored_config, payload = load_rf_successor_checkpoint(checkpoint)

    assert restored_config == config
    assert payload["objective_version"] == 12
    assert restored.masked_history is True
    for name, value in head.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_training_protocol_locks_reward_free_multi_horizon_semantics():
    protocol = load_rf_successor_training_protocol(
        "configs/experiment/rf_successor_lewm_cube_train.yaml"
    )

    assert protocol["method"] == "rf_successor_lewm"
    assert protocol["sequence"]["rollout_horizon"] == 5
    assert protocol["successor"]["goal_conditioning"] == "none"
    assert protocol["successor"]["continuation_policy"] == "none"
    assert protocol["successor"]["td_bootstrap"] is False
    assert protocol["successor"]["objective_version"] == 12
    assert protocol["successor"]["history_supervision"] == "all_prefix_lengths"
    assert protocol["joint_objective"]["multi_step_prediction"] == (
        "open_loop_latent_mse_all_horizons"
    )


def test_evaluation_protocol_uses_cem_candidates_without_an_actor():
    protocol = load_rf_successor_evaluation_protocol(
        "configs/experiment/rf_successor_lewm_cube_checkpoint_o50.yaml"
    )

    assert protocol["method"] == "rf_successor_lewm"
    assert protocol["planning"]["horizon"] == protocol["successor"]["max_horizon"]
    assert protocol["planning"]["initial_distribution"] == (
        "cem_gaussian_no_actor"
    )
    assert protocol["inference_objective"]["learned_action_policy"] is False
