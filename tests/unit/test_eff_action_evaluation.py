"""No-download protocol and public-CEM adapter checks for EffAction."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from tdwm.adapters.eff_action import EffActionCostModel, validate_eff_action_planning
from tdwm.evaluation import eff_action as evaluation


class FrozenActionEncoder(nn.Module):
    input_dim = 25
    emb_dim = 192

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()

    def forward(self, action):
        return torch.nn.functional.pad(action, (0, 167)) + self.anchor


class FrozenWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = FrozenActionEncoder()
        self.encode_calls = 0
        self.eval()

    def encode(self, info):
        assert set(info) == {"pixels"}
        self.encode_calls += 1
        pixels = info["pixels"]
        z = pixels.flatten(start_dim=2).mean(dim=-1, keepdim=True).expand(-1, -1, 192)
        return {"emb": z}

    def rollout(self, *args, **kwargs):
        raise AssertionError("EffAction must not call F rollout.")

    def predict(self, *args, **kwargs):
        raise AssertionError("EffAction must not call F predict.")


class FrozenG(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()

    def forward(self, state, action_embedding, task):
        return action_embedding + state * 0 + task * 0 + self.anchor


class FrozenV(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()

    def forward(self, psi, task):
        return 1.0 + (psi[..., :25] - 0.3).square().sum(dim=-1) + self.anchor


def planning(method="EffAction"):
    common = dict(
        horizon=1,
        receding_horizon=1,
        action_block=5,
        history_len=1,
        warm_start=False,
        planning_seed=42,
        iterations=2,
        episode_budget=50,
    )
    if method == "EffAction":
        return {
            **common,
            "candidates": 4,
            "elites": 2,
            "initial_variance": 1.0,
            "solver_batch_size": 2,
        }
    return {
        **common,
        "initialization": "zeros",
        "initial_std": 0.0,
        "lower_bound": -2.0,
        "upper_bound": 2.0,
    }


def protocol():
    return dict(
        method="EffAction",
        planning=planning(),
        epsilon=1e-6,
        selection=dict(protocol="rp1_heldout", goal_offset=25, episodes=4, seed=42),
        runtime=dict(
            stable_worldmodel_version="0.1.1", precision="fp32", renderer="osmesa"
        ),
        world=dict(
            env_name="swm/OGBCube-v0",
            env_type="single",
            ob_type="states",
            image_size=224,
            multiview=False,
            visualize_info=False,
            terminate_at_goal=True,
        ),
        image_preprocessing=dict(
            size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        action_normalization=dict(
            mean=[1.0] * 5, scale=[2.0] * 5, variance=[4.0] * 5, samples=2000000
        ),
        dataset=dict(
            format="lance",
            manifest_sha256=evaluation.CUBE_MANIFEST_SHA256,
            source_sha256=evaluation.CUBE_SOURCE_SHA256,
        ),
    )


def audited_dataset_source(tmp_path, monkeypatch):
    source = tmp_path / "cube.lance"
    source.mkdir()
    sidecar = tmp_path / "cube.lance.manifest.json"
    sidecar.write_text(
        json.dumps(
            {
                "destination": {
                    "format": "lance",
                    "image_codec": "jpeg",
                    "jpeg_quality": 100,
                    "size_bytes": 123,
                },
                "source": {
                    "sha256": evaluation.CUBE_SOURCE_SHA256,
                    "size_bytes": evaluation.CUBE_SOURCE_BYTES,
                },
                "conversion": {"stable_worldmodel_version": "0.1.1"},
                "verification": {"episodes": 10000, "transitions": 2010000},
            }
        )
    )
    # Synthetic CPU fixture represents the audited sidecar; corrupt hash and
    # source checks are covered independently below, without the 74GB dataset.
    monkeypatch.setattr(
        evaluation, "_sha256", lambda path: evaluation.CUBE_MANIFEST_SHA256
    )
    return source, sidecar


@pytest.mark.parametrize("offset", [25, 50, 100])
def test_historical_pairs_match_the_archived_exact_locks(offset):
    pairs, metadata = evaluation.select_eff_action_episodes(
        np.full(10000, 201),
        dict(protocol="historical_cg3", goal_offset=offset, episodes=50, seed=42),
    )
    assert (
        metadata["selection_sha256"] == evaluation.HISTORICAL_SELECTION_SHA256[offset]
    )
    assert np.array_equal(
        pairs["goal_steps"] - pairs["start_steps"], np.full(50, offset)
    )
    assert metadata["heldout_from_episode_range_0_8000"] is False


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_heldout_pairs_are_episode_disjoint_deterministic_and_in_bounds(seed):
    cfg = dict(protocol="rp1_heldout", goal_offset=100, episodes=50, seed=seed)
    first, meta = evaluation.select_eff_action_episodes(np.full(10000, 201), cfg)
    second, other = evaluation.select_eff_action_episodes(np.full(10000, 201), cfg)
    assert meta == other
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert np.all(
        (first["episode_indices"] >= 8000) & (first["episode_indices"] < 10000)
    )
    assert np.all(first["goal_steps"] < 201)
    assert len(np.unique(first["valid_row_ranks"])) == 50
    assert meta["rp1_sampler_bitwise_reproduction_claimed"] is False


def test_heldout_sampler_can_reach_its_last_valid_row():
    lengths = np.full(10000, 25)
    lengths[-1] = 26
    pairs, _ = evaluation.select_eff_action_episodes(
        lengths, dict(protocol="rp1_heldout", goal_offset=25, episodes=1, seed=42)
    )
    assert pairs["episode_indices"].tolist() == [9999]
    assert pairs["start_steps"].tolist() == [0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("horizon", 5),
        ("receding_horizon", 5),
        ("action_block", 1),
        ("history_len", 3),
        ("warm_start", True),
    ],
)
def test_unagreed_execution_granularity_is_rejected(key, value):
    config = planning()
    config[key] = value
    with pytest.raises(ValueError):
        validate_eff_action_planning(config, "EffAction")


def test_cem_cannot_claim_only_cost_side_action_clamping():
    config = planning()
    config.update(lower_bound=-1.0, upper_bound=1.0)
    with pytest.raises(ValueError, match="clamping only"):
        validate_eff_action_planning(config, "EffAction")


def test_cost_matches_ratio_uses_all_action_coordinates_and_never_rolls_f():
    world, g, v = FrozenWorld(), FrozenG(), FrozenV()
    model = EffActionCostModel(world, g, v, epsilon=0.01)
    actions = torch.zeros(2, 3, 1, 25)
    actions[:, 1, 0] = 0.3
    actions[:, 2, 0, -1] = 2.0
    info = {
        "pixels": torch.ones(2, 3, 1, 3, 2, 2),
        "goal": torch.full((2, 3, 1, 3, 2, 2), 2.0),
    }
    costs = model.get_cost(info, actions)
    expected_v = 1 + (actions[:, :, 0] - 0.3).square().sum(-1)
    expected = -(192.0**0.5) / (expected_v + 0.01)
    torch.testing.assert_close(costs, expected)
    assert torch.all(costs[:, 1] < costs[:, 0])
    assert torch.all(costs[:, 2] > costs[:, 0])
    model.get_cost(info, actions)
    assert world.encode_calls == 2


def test_zero_state_goal_distance_has_zero_finite_cost():
    model = EffActionCostModel(FrozenWorld(), FrozenG(), FrozenV(), epsilon=1e-6)
    state = torch.ones(1, 4, 1, 192)
    result = model.get_cost({"emb": state, "goal_emb": state}, torch.randn(1, 4, 1, 25))
    assert torch.equal(result, torch.zeros_like(result))


def test_training_normalization_is_restored_without_refitting():
    processor, stats = evaluation.build_eff_action_action_processor(
        protocol()["action_normalization"]
    )
    np.testing.assert_array_equal(
        processor.inverse_transform(np.zeros((3, 5))), np.ones((3, 5))
    )
    np.testing.assert_array_equal(
        processor.transform(np.full((3, 5), 3.0)), np.ones((3, 5))
    )
    assert stats["samples"] == 2000000


def test_evaluator_calls_public_world_with_exact_pairs_budget_and_records_evidence(
    tmp_path, monkeypatch
):
    import stable_worldmodel as swm

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    source, _ = audited_dataset_source(tmp_path, monkeypatch)
    dataset = type(
        "Data", (), {"lengths": np.full(10000, 201), "get_dim": lambda self, key: 5}
    )()
    monkeypatch.setattr(swm.data, "load_dataset", lambda *args, **kwargs: dataset)
    captured = {}

    class World:
        def __init__(self, env_name, **kwargs):
            captured["world"] = (env_name, kwargs)

        def set_policy(self, policy):
            captured["policy"] = policy

        def evaluate(self, **kwargs):
            captured["evaluate"] = kwargs
            return {
                "success_rate": 50.0,
                "episode_successes": np.array([True, False, True, False]),
                "seeds": None,
            }

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(swm, "World", World)
    result = evaluation.evaluate_eff_action(
        world_model=FrozenWorld(),
        successor=FrozenG(),
        value=FrozenV(),
        config=protocol(),
        dataset_path=source,
        output_dir=tmp_path / "result",
        checkpoint_metadata={"method": "EffAction", "sha256": "a" * 64},
    )
    assert captured["world"][1]["max_episode_steps"] == 50
    assert captured["evaluate"]["eval_budget"] == 50
    assert captured["evaluate"]["goal_offset"] == 25
    assert min(captured["evaluate"]["episodes_idx"]) >= 8000
    assert captured["closed"]
    assert result["selection_protocol"] == "rp1_heldout"
    manifest = json.loads((tmp_path / "result" / "protocol_manifest.json").read_text())
    assert manifest["formal_protocol_claimed"] is False
    assert manifest["runtime"]["MUJOCO_GL"] == "osmesa"
    assert manifest["execution"]["candidate_bound_policy"] == "unbounded_normalized"
    assert manifest["execution"]["selected_action"] == "final_elite_mean"
    assert (
        manifest["dataset"]["conversion_manifest_sha256"]
        == evaluation.CUBE_MANIFEST_SHA256
    )
    assert manifest["dataset"]["source_sha256"] == evaluation.CUBE_SOURCE_SHA256
    assert manifest["dataset"]["rows"] == 2010000
    assert manifest["dataset"]["source_bytes_rehashed_this_run"] is False
    assert result["metrics"]["episode_successes"] == [True, False, True, False]
    assert result["success_count"] == 2
    assert result["success_rate_percent"] == 50.0
    assert result["per_episode"][0]["success"] is True
    assert (
        result["per_episode"][0]["episode_index"]
        == captured["evaluate"]["episodes_idx"][0]
    )
    assert (tmp_path / "result" / "action_normalization.json").is_file()
    assert (tmp_path / "result" / "episode_selection.json").is_file()


def test_evaluator_rejects_renderer_mismatch_before_world_creation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(ValueError, match="disagrees"):
        evaluation.evaluate_eff_action(
            world_model=FrozenWorld(),
            successor=FrozenG(),
            value=FrozenV(),
            config=protocol(),
            dataset_path=tmp_path / "none.lance",
            output_dir=tmp_path / "out",
            checkpoint_metadata={"method": "EffAction", "sha256": "a" * 64},
        )


def test_historical_label_refuses_another_seed_or_population():
    config = dict(protocol="historical_cg3", goal_offset=25, episodes=50, seed=43)
    with pytest.raises(ValueError, match="seed42"):
        evaluation.select_eff_action_episodes(np.full(10000, 201), config)
    other = deepcopy(config)
    other["seed"] = 42
    with pytest.raises(ValueError, match="10000 x 201"):
        evaluation.select_eff_action_episodes(np.full(2000, 201), other)


def test_wrong_dataset_manifest_is_rejected_even_with_declared_correct_source(
    tmp_path, monkeypatch
):
    source, _ = audited_dataset_source(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "_sha256", lambda path: "b" * 64)
    with pytest.raises(ValueError, match="manifest SHA256 differs"):
        evaluation._validate_dataset_source(source, protocol()["dataset"])


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("source", "sha256", "b" * 64, "source SHA256 differs"),
        ("source", "size_bytes", 10, "source size differs"),
        ("verification", "transitions", 2000000, "2010000"),
        ("destination", "jpeg_quality", 90, "JPEG quality"),
    ],
)
def test_dataset_identity_and_conversion_properties_are_required(
    tmp_path, monkeypatch, section, key, value, match
):
    source, sidecar = audited_dataset_source(tmp_path, monkeypatch)
    payload = json.loads(sidecar.read_text())
    payload[section][key] = value
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        evaluation._validate_dataset_source(source, protocol()["dataset"])


@pytest.mark.parametrize("dtype", [bool, np.float64, np.uint8])
def test_swm_binary_outcomes_and_percentage_unit_are_preserved(dtype):
    values = np.array([1, 0, 1, 0], dtype=dtype)
    outcomes, rate = evaluation._validate_metrics(
        {"episode_successes": values, "success_rate": 50.0}, 4
    )
    assert outcomes.dtype == values.dtype
    assert rate == 50.0


@pytest.mark.parametrize(
    "values,rate",
    [([1, 0, 1, 0], 0.5), ([1, 0.5, 1, 0], 62.5), ([1, float("nan"), 1, 0], 50)],
)
def test_swm_outcome_validation_rejects_ratios_nonbinary_or_nonfinite(values, rate):
    with pytest.raises(ValueError, match="binary outcomes|binary outcome"):
        evaluation._validate_metrics(
            {"episode_successes": values, "success_rate": rate}, 4
        )
