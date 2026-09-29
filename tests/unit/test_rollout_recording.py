"""Executed actions, temporal alignment, interruption and lossless images."""

import json

import gymnasium as gym
import numpy as np
import pytest
from PIL import Image

from tdwm.evaluation.rollout_recording import RolloutRecorder, _rgb


class PixelsEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)

    def __init__(self, *, stop=2, mutate_action=False):
        self.stop, self.mutate_action = stop, mutate_action
        self.calls = 0
        self.pixels = np.zeros((4, 6, 3), dtype=np.uint8)

    def step(self, action):
        self.calls += 1
        if self.mutate_action:
            action[:] = 0
        self.pixels.fill(10 + self.calls)
        return None, 0.25, self.calls == self.stop, False, {
            "pixels": self.pixels, "qpos": np.array([self.calls], dtype=np.float64),
        }


def make_recorder(tmp_path, *, count=1, budget=6):
    return RolloutRecorder(
        tmp_path / "rollouts", pairs=dict(episode_indices=list(range(count)),
            start_steps=[10]*count, goal_steps=[35]*count),
        metadata=dict(method="EffPlan", score_mode="state_path_tracking", protocol="O25"),
        budget=budget,
    )


def begin(recorder):
    count = len(recorder.episodes)
    recorder.begin(dict(
        pixels=np.full((count, 1, 4, 6, 3), 10, dtype=np.uint8),
        goal=np.full((count, 1, 4, 6, 3), 80, dtype=np.uint8),
        qpos=np.full((count, 1, 1), 100, dtype=np.float64),
    ))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_action_frame_alignment_and_mutating_environment(tmp_path, dtype):
    recorder = make_recorder(tmp_path)
    env = recorder.wrap_environment(PixelsEnv(mutate_action=True))
    begin(recorder)
    for _ in range(2):
        outcome = env.step(np.array([0.125, -0.75], dtype=dtype))
    assert outcome[2]
    summary = recorder.finish([True])
    assert summary["total_executed_actions"] == 2
    folder = recorder.root / "episode_0000"
    trajectory = json.loads((folder / "trajectory.json").read_text())
    streamed = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    assert streamed == trajectory["steps"]
    assert trajectory["initial"]["state"]["qpos"] == [[100.0]]
    for i, row in enumerate(streamed):
        assert row["step"] == i
        assert row["action"] == [0.125, -0.75]
        assert row["action_dtype"] == np.dtype(dtype).name
        assert row["after_state"]["qpos"] == [i+1]
        assert np.asarray(Image.open(folder / row["before_frame"])).min() == 10+i
        assert np.asarray(Image.open(folder / row["after_frame"])).min() == 11+i
    assert len(list((folder / "frames").glob("*.png"))) == 3
    assert np.asarray(Image.open(folder / "goal.png")).min() == 80
    assert summary["episodes"][0]["observation_frames"] == 3


def test_dead_environments_are_not_padded_and_large_actions_group_by_five(tmp_path):
    recorder = make_recorder(tmp_path, count=2)
    first = recorder.wrap_environment(PixelsEnv(stop=2))
    second = recorder.wrap_environment(PixelsEnv(stop=6))
    begin(recorder)
    for i in range(6):
        if i < 2:
            first.step(np.zeros(2))
        second.step(np.ones(2))
    summary = recorder.finish([True, True])
    assert [row["executed_primitive_steps"] for row in summary["episodes"]] == [2, 6]
    assert recorder.records[1][5]["action_block_index"] == 1
    assert recorder.records[1][5]["primitive_index_in_block"] == 0


def test_partial_recording_is_readable_but_never_complete(tmp_path):
    recorder = make_recorder(tmp_path)
    env = recorder.wrap_environment(PixelsEnv())
    begin(recorder)
    env.step(np.zeros(2))
    recorder.abort("deliberate interruption")
    manifest = json.loads((recorder.root / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["error"] == "deliberate interruption"
    assert manifest["total_executed_actions"] == 1
    assert manifest["episodes"][0]["success"] is None
    assert (recorder.root / "episode_0000/frames/000001.png").exists()


def test_recording_refuses_overwrite_and_outcome_mismatch(tmp_path):
    recorder = make_recorder(tmp_path)
    with pytest.raises(FileExistsError):
        make_recorder(tmp_path)
    env = recorder.wrap_environment(PixelsEnv(stop=1))
    begin(recorder)
    env.step(np.zeros(2))
    with pytest.raises(ValueError, match="termination differs"):
        recorder.finish([False])
    assert json.loads((recorder.root / "manifest.json").read_text())["status"] == "running"


@pytest.mark.parametrize("pixels", [
    np.zeros((3, 4, 6), dtype=np.uint8),
    np.zeros((4, 6, 3), dtype=np.float32),
    np.zeros((2, 4, 6, 3), dtype=np.uint8),
])
def test_never_silently_convert_normalized_or_ambiguous_pixels(pixels):
    with pytest.raises(ValueError, match="unnormalized"):
        _rgb(pixels)


def test_policy_is_delegated_once_without_changing_rng_or_inputs(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.wrap_environment(PixelsEnv())

    class Policy:
        marker = "original-policy"
        calls = 0

        def set_env(self, env):
            self.env = env

        def get_action(self, infos):
            self.calls += 1
            return infos["expected_action"]

    original = Policy()
    wrapped = recorder.wrap_policy(original)
    wrapped.set_env("unchanged")
    infos = dict(pixels=np.zeros((1, 1, 4, 6, 3), dtype=np.uint8),
                 goal=np.ones((1, 1, 4, 6, 3), dtype=np.uint8),
                 expected_action=np.array([[0.25, -0.5]], dtype=np.float32))
    np.random.seed(42)
    before = np.random.get_state()
    assert wrapped.get_action(infos) is infos["expected_action"]
    assert wrapped.get_action(infos) is infos["expected_action"]
    after = np.random.get_state()
    assert original.calls == 2 and original.env == "unchanged"
    assert wrapped.marker == "original-policy"
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
