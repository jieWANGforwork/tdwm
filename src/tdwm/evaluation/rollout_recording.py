"""Lossless, passive recording through public Gym/SWM extension points.

Only environment commands actually passed to ``step`` are recorded. No extra
render, model prediction, random sample, reset, or physics step is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def _rgb(value):
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Rollout recording requires unnormalized uint8 HWC RGB pixels.")
    return image.copy()


def _state(info):
    result = {}
    for key, value in info.items():
        if key not in {"qpos", "qvel", "observation", "state"} and not key.startswith("privileged_"):
            continue
        array = np.asarray(value)
        if array.dtype.kind in "biuf" and array.size <= 4096:
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite recorded environment state: {key}")
            result[key] = array.tolist()
    return result


class RolloutRecorder:
    """One new directory per evaluation; one stable index per selected pair."""

    def __init__(self, directory, *, pairs, metadata, budget, action_block=5):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.budget, self.action_block = int(budget), int(action_block)
        count = len(pairs["episode_indices"])
        if count == 0 or any(len(pairs[k]) != count for k in ("start_steps", "goal_steps")):
            raise ValueError("Incomplete recording pair selection.")
        self.episodes = [dict(
            index=i, episode=int(pairs["episode_indices"][i]),
            start=int(pairs["start_steps"][i]), goal=int(pairs["goal_steps"][i]),
        ) for i in range(count)]
        self.records = [[] for _ in self.episodes]
        self.initial = [None for _ in self.episodes]
        self.spaces = []
        self.begun = self.finished = False
        self._write_manifest("running")

    def wrap_environment(self, env):
        index = len(self.spaces)
        if index >= len(self.episodes):
            raise ValueError("More environments than selected recording pairs.")
        self.spaces.append(dict(low=np.asarray(env.action_space.low).tolist(),
                                high=np.asarray(env.action_space.high).tolist()))
        return _RecordedEnvironment(env, self, index)

    def wrap_policy(self, policy):
        return _RecordedPolicy(policy, self)

    def _directory(self, index):
        return self.root / f"episode_{index:04d}"

    def begin(self, infos):
        if self.begun:
            return
        count = len(self.episodes)
        if len(self.spaces) != count or len(infos["pixels"]) != count:
            raise ValueError("Recording environment/pair count mismatch.")
        # Called on the FIRST policy request, after SWM has restored dataset
        # states and goals. reset() pixels are deliberately never saved here.
        for index, episode in enumerate(self.episodes):
            directory = self._directory(index)
            (directory / "frames").mkdir(parents=True, exist_ok=False)
            Image.fromarray(_rgb(infos["pixels"][index])).save(
                directory / "frames/000000.png", compress_level=1)
            Image.fromarray(_rgb(infos["goal"][index])).save(directory / "goal.png", compress_level=1)
            state = _state({k: v[index] for k, v in infos.items()
                            if isinstance(v, np.ndarray) and v.ndim and len(v) == count})
            self.initial[index] = dict(
                frame="frames/000000.png", state=state,
                source="dataset_start_observation_used_by_policy_after_state_restoration",
            )
            _write_json(directory / "episode.json", dict(
                **episode, **self.metadata, initial=self.initial[index], goal_frame="goal.png",
                action_space=self.spaces[index], episode_budget=self.budget,
                action_block=self.action_block,
                action_units="environment-space command passed to env.step; not internal actuator controls",
            ))
        self.begun = True

    def transition(self, index, action, reward, terminated, truncated, info):
        if not self.begun or self.finished:
            raise RuntimeError("Environment step outside an active recording.")
        rows = self.records[index]
        step = len(rows)
        if step >= self.budget or (rows and (rows[-1]["terminated"] or rows[-1]["truncated"])):
            raise RuntimeError("Attempt to record execution past terminal/budget.")
        action = np.asarray(action)
        if action.dtype.kind not in "fiu" or not np.isfinite(action).all():
            raise ValueError("Nonfinite/unsupported executed environment action.")
        directory = self._directory(index)
        before, after = f"frames/{step:06d}.png", f"frames/{step+1:06d}.png"
        Image.fromarray(_rgb(info["pixels"])).save(directory / after, compress_level=1)
        row = dict(
            step=step, action=action.tolist(), action_dtype=str(action.dtype),
            action_block_index=step // self.action_block,
            primitive_index_in_block=step % self.action_block,
            before_frame=before, after_frame=after, reward=float(reward),
            terminated=bool(terminated), truncated=bool(truncated),
            after_state=_state(info),
        )
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        # Stream every transition, so interruption does not lose the action
        # history already written. PNG and JSON paths are website-relative.
        with (directory / "steps.jsonl").open("a") as stream:
            stream.write(line + "\n")
        rows.append(row)

    def _write_manifest(self, status, *, error=None):
        entries = []
        for i, episode in enumerate(self.episodes):
            rows = self.records[i]
            entries.append(dict(
                **episode, trajectory=f"episode_{i:04d}/trajectory.json",
                steps_jsonl=f"episode_{i:04d}/steps.jsonl",
                goal_frame=f"episode_{i:04d}/goal.png",
                executed_primitive_steps=len(rows), observation_frames=len(rows)+int(self.initial[i] is not None),
                success=(any(r["terminated"] for r in rows) if status == "complete" else None),
            ))
        manifest = dict(
            format="tdwm-executed-rollouts-v1", status=status, **self.metadata,
            action_block=self.action_block, episode_budget=self.budget,
            episodes=entries, total_executed_actions=sum(map(len, self.records)),
            observation_source="initial policy dataset frame; thereafter actual environment step pixels, never F predictions",
            error=error,
        )
        _write_json(self.root / "manifest.json", manifest)
        return manifest

    def finish(self, successes):
        if not self.begun or len(successes) != len(self.episodes):
            raise ValueError("Missing recording initial observations or outcomes.")
        for i, success in enumerate(successes):
            rows = self.records[i]
            if not rows or len(rows) > self.budget:
                raise ValueError("Incomplete/over-budget executed rollout.")
            if bool(success) != any(row["terminated"] for row in rows):
                raise ValueError("Recorded environment termination differs from formal success.")
            if not (rows[-1]["terminated"] or rows[-1]["truncated"] or len(rows) == self.budget):
                raise ValueError("Recorded episode ended before success or budget.")
        self._write_trajectories("complete", successes)
        self.finished = True
        return self._write_manifest("complete")

    def _write_trajectories(self, status, successes=None):
        for i, episode in enumerate(self.episodes):
            if self.initial[i] is None:
                continue
            _write_json(self._directory(i) / "trajectory.json", dict(
                format="tdwm-executed-episode-v1", status=status,
                **self.metadata, **episode, initial=self.initial[i], goal_frame="goal.png",
                success=None if successes is None else bool(successes[i]),
                executed_primitive_steps=len(self.records[i]), steps=self.records[i],
            ))

    def abort(self, error):
        if not self.finished:
            self._write_trajectories("incomplete")
            self._write_manifest("incomplete", error=str(error))
            self.finished = True


class _RecordedEnvironment(gym.Wrapper):
    def __init__(self, env, recorder, index):
        super().__init__(env)
        self.recorder, self.index = recorder, index

    def step(self, action):
        command = np.asarray(action).copy()  # preserve even if env mutates input
        outcome = self.env.step(action)
        _, reward, terminated, truncated, info = outcome
        self.recorder.transition(self.index, command, reward, terminated, truncated, info)
        return outcome


class _RecordedPolicy:
    def __init__(self, policy, recorder):
        self.policy, self.recorder = policy, recorder

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def set_env(self, env):
        return self.policy.set_env(env)

    def get_action(self, infos):
        self.recorder.begin(infos)
        return self.policy.get_action(infos)
