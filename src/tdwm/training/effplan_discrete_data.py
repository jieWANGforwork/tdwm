"""Fixed four-node P supervision from discrete 5/10/20-block trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tdwm.training.eff_data import EffEpisodeReplay, EffPlanReplayBatch


def discrete_sampling_identity(action_spans) -> dict:
    if not isinstance(action_spans, (list, tuple)) or any(
        type(n) is not int for n in action_spans
    ) or tuple(action_spans) != (5, 10, 20):
        raise ValueError("planner_action_spans must be exactly [5, 10, 20].")
    return dict(
        mode="discrete_spans_fixed_four_nodes_v1",
        action_spans=[5, 10, 20], action_block_primitive_steps=5,
        output_horizon=5, interior_nodes=4,
        sampling="uniform_span_then_uniform_eligible_episode_then_uniform_legal_anchor",
        node_offsets_blocks={str(n): [j*n//5 for j in range(6)] for n in action_spans},
        labels="actual_encoded_states_at_uniform_positions_not_latent_averages",
        cross_episode_probability=0.0,
    )


@dataclass(frozen=True)
class DiscretePlannerBatch(EffPlanReplayBatch):
    action_spans: torch.Tensor
    node_offsets_blocks: torch.Tensor


class DiscretePlannerSampler:
    """Read only six selected rows, after checking the ENTIRE source interval.

    ``state_stride`` remains the replay's primitive stride (5), not the spacing
    between selected labels. That spacing is explicitly recorded per sample.
    Dynamics calibration still searches five NEW action blocks to track the
    compressed path; it does not replay or relabel the source's longer actions.
    """

    def __init__(self, replay: EffEpisodeReplay, action_spans=(5, 10, 20)):
        self.identity = discrete_sampling_identity(action_spans)
        if replay.stride != 5:
            raise ValueError("Discrete P sampling requires five-primitive-step blocks.")
        self.replay = replay
        self.eligible = {}
        for span in action_spans:
            entries = []
            for episode in replay.episodes:
                rows = replay.rows[int(episode)]
                if len(rows) <= span:
                    continue
                anchors = np.arange(len(rows)-span, dtype=np.int64)
                first, last = rows[anchors], rows[anchors+span]
                # Termination at arrival is legal; earlier terminal states are not.
                legal = replay.terminal_prefix[last] == replay.terminal_prefix[first]
                if legal.any():
                    entries.append((int(episode), anchors[legal]))
            if not entries:
                raise ValueError(f"No legal same-episode path for {span} action blocks.")
            self.eligible[span] = entries

    def sample(self, *, batch_size: int, rng: np.random.Generator) -> DiscretePlannerBatch:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive.")
        spans = rng.choice([5, 10, 20], size=batch_size)
        offsets = spans[:, None] * np.arange(6, dtype=np.int64)[None, :] // 5
        selected, episodes = [], []
        for span, relative in zip(spans, offsets):
            entries = self.eligible[int(span)]
            episode, anchors = entries[int(rng.integers(len(entries)))]
            anchor = int(anchors[int(rng.integers(len(anchors)))])
            selected.append(self.replay.rows[episode][anchor+relative])
            episodes.append(episode)
        rows = np.stack(selected)
        ids = torch.tensor(episodes, dtype=torch.long)
        return DiscretePlannerBatch(
            real_states=torch.from_numpy(np.array(self.replay.store.latents[rows], dtype=np.float32)),
            trajectory_valid=torch.ones(batch_size, dtype=torch.bool),
            rows=torch.from_numpy(rows), anchor_episodes=ids, goal_episodes=ids.clone(),
            state_stride=self.replay.stride, action_spans=torch.from_numpy(spans),
            node_offsets_blocks=torch.from_numpy(offsets),
        )
