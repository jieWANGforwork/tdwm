"""Actual SWM World.evaluate: restored starts, final frames and alive masks."""

import json

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image

from tdwm.evaluation.rollout_recording import RolloutRecorder


class TinyRenderedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
        self.frames = self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=None if seed is None else int(seed))
        self.position, self.steps, self.stop = 0, 0, 0
        return self._observation(), self._info()

    def set_state(self, qpos, qvel):
        self.position, self.stop = int(qpos[0]), int(qvel[0])

    def _observation(self):
        return np.array([self.position], dtype=np.float32)

    def _info(self):
        return dict(qpos=np.array([self.position], dtype=np.float64),
                    qvel=np.array([self.stop], dtype=np.float64))

    def step(self, action):
        self.steps += 1
        self.position += 1
        return self._observation(), 0.0, bool(self.stop and self.steps == self.stop), False, self._info()

    def render(self):
        self.frames += 1
        return np.full((8, 8, 3), self.position, dtype=np.uint8)


class TinyDataset:
    column_names = ["pixels", "qpos", "qvel", "seed"]

    def load_chunk(self, episodes, starts, ends):
        result = []
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            count = int(end-start)
            positions = torch.arange(count, dtype=torch.float64) + (10 + 10*i)
            result.append(dict(
                pixels=positions.to(torch.uint8)[:, None, None, None].expand(-1, 3, 8, 8).clone(),
                qpos=positions[:, None], qvel=torch.full((count, 1), 2.0 if i == 0 else 0.0),
                seed=torch.full((count,), 42+i, dtype=torch.int64),
            ))
        return result


class ConstantPolicy:
    def set_env(self, env):
        self.env = env

    def get_action(self, infos):
        return np.full((self.env.num_envs, 1), 0.25, dtype=np.float32)


def test_recording_does_not_change_public_world_execution_or_extra_render(tmp_path):
    name = "TDWMRecordingFixture-v0"
    if name not in gym.registry:
        gym.register(name, entry_point=TinyRenderedEnv)
    pairs = dict(episode_indices=[12, 13], start_steps=[5, 9], goal_steps=[8, 12])
    recorder = RolloutRecorder(tmp_path / "rollouts", pairs=pairs, budget=5,
                               metadata=dict(method="EffPlan", protocol="fixture"))

    def evaluate(record):
        world = swm.World(name, num_envs=2, image_shape=(8, 8), max_episode_steps=5,
                          **({"extra_wrappers": [recorder.wrap_environment]} if record else {}))
        policy = ConstantPolicy()
        world.set_policy(recorder.wrap_policy(policy) if record else policy)
        result = world.evaluate(
            dataset=TinyDataset(), episodes_idx=pairs["episode_indices"],
            start_steps=pairs["start_steps"], goal_offset=3, eval_budget=5,
            callables=[dict(method="set_state", args={
                "qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}})],
        )
        counts = [(env.unwrapped.steps, env.unwrapped.frames) for env in world.envs.envs]
        world.close()
        return result, counts

    plain, plain_counts = evaluate(False)
    recorded, recorded_counts = evaluate(True)
    assert plain_counts == recorded_counts == [(2, 3), (5, 6)]
    np.testing.assert_array_equal(plain["episode_successes"], recorded["episode_successes"])
    assert plain["success_rate"] == recorded["success_rate"] == 50
    summary = recorder.finish(recorded["episode_successes"])
    assert summary["total_executed_actions"] == 7
    for i, expected in enumerate([2, 5]):
        folder = recorder.root / f"episode_{i:04d}"
        trajectory = json.loads((folder / "trajectory.json").read_text())
        assert trajectory["executed_primitive_steps"] == expected
        # reset() rendered zero; the saved start must instead be the state
        # restored from the selected dataset episode, before the first action.
        assert np.asarray(Image.open(folder / "frames/000000.png")).min() == 10 + 10*i
        assert np.asarray(Image.open(folder / "goal.png")).min() == 13 + 10*i
        assert np.asarray(Image.open(folder / f"frames/{expected:06d}.png")).min() == 10 + 10*i + expected
        assert all(row["action"] == [0.25] for row in trajectory["steps"])
        assert trajectory["steps"][-1]["terminated"] == (i == 0)
        assert trajectory["steps"][-1]["truncated"] == (i == 1)
