"""Additive P sampling: use the same episode/offset/anchor draws as frozen V.

Length always counts STATES: L states, L-1 macro actions, L-2 interior labels.
No interpolation, padding, temporal rescaling, or cross-episode labels.
"""

from dataclasses import dataclass, fields

import numpy as np
import torch

from tdwm.training.eff_data import EffEpisodeReplay, EffPlanReplayBatch


@dataclass(frozen=True)
class VariablePlannerBatch:
    samples: tuple[EffPlanReplayBatch, ...]

    def groups(self, limit=None):
        """Bucket IID draws by length, preserving equal per-path loss weight."""
        buckets = {}
        for sample in self.samples[:limit]:
            buckets.setdefault(sample.real_states.shape[1], []).append(sample)
        for length in sorted(buckets):
            samples = buckets[length]
            yield EffPlanReplayBatch(**{
                field.name: samples[0].state_stride if field.name == "state_stride"
                else torch.cat([getattr(s, field.name) for s in samples], dim=0)
                for field in fields(EffPlanReplayBatch)
            })


def sample_variable_planner_paths(
    replay: EffEpisodeReplay, *, batch_size: int, rng: np.random.Generator,
    backup_primitive_steps: int, epsilon: float,
) -> VariablePlannerBatch:
    # Reuse V's actual sampler, including its RNG ordering. Its n-step target
    # boundary does NOT truncate the full P path from anchor to hindsight goal.
    draws = replay.sample(
        batch_size=batch_size, rng=rng,
        backup_primitive_steps=backup_primitive_steps,
        cross_episode_probability=0.0, epsilon=epsilon,
    )
    samples = []
    for start, goal, episode in zip(
        draws.anchor_rows.tolist(), draws.goal_rows.tolist(),
        draws.anchor_episodes.tolist(), strict=True,
    ):
        if replay._crosses_terminal(start, goal):
            raise ValueError("P cannot supervise a path crossing an environment terminal.")
        rows = np.arange(start, goal + 1, replay.stride, dtype=np.int64)[None]
        samples.append(EffPlanReplayBatch(
            real_states=torch.from_numpy(np.array(replay.store.latents[rows], dtype=np.float32)),
            trajectory_valid=torch.ones(1, dtype=torch.bool),
            rows=torch.from_numpy(rows), anchor_episodes=torch.tensor([episode]),
            goal_episodes=torch.tensor([episode]), state_stride=replay.stride,
        ))
    return VariablePlannerBatch(tuple(samples))
