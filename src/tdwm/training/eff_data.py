"""Episode-disjoint replay and hindsight sampling for Eff / EffPlan.

RP1 Appendix C.1 uses episodes 0..7999 for learning and 8000..9999 for
evaluation. This is intentionally NOT the older C-family random clip split.
This module consumes an existing audited FrozenLatentStore; it neither
re-encodes images nor converts the original dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tdwm.methods.eff import STATE_DIM


@dataclass(frozen=True)
class EpisodePartition:
    training: tuple[int, ...]
    evaluation: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.training or not self.evaluation:
            raise ValueError("Both training and held-out episodes are required.")
        if min((*self.training, *self.evaluation)) < 0:
            raise ValueError("Episode IDs must be nonnegative.")
        if len(set(self.training)) != len(self.training) or len(
            set(self.evaluation)
        ) != len(self.evaluation):
            raise ValueError("Episode partitions cannot contain duplicates.")
        if set(self.training).intersection(self.evaluation):
            raise ValueError("Training/evaluation episode leakage.")

    @classmethod
    def rp1_cube(cls) -> EpisodePartition:
        return cls(tuple(range(8000)), tuple(range(8000, 10000)))


@dataclass(frozen=True)
class EffReplayBatch:
    state: torch.Tensor
    next_state: torch.Tensor
    bootstrap_state: torch.Tensor
    goal: torch.Tensor
    terminal_after_transition: torch.Tensor
    observed_cost: torch.Tensor
    goal_reached: torch.Tensor
    continuation_valid: torch.Tensor
    vector_valid: torch.Tensor
    efficiency: torch.Tensor
    efficiency_known: torch.Tensor
    offset_chunks: torch.Tensor
    anchor_rows: torch.Tensor
    goal_rows: torch.Tensor
    bootstrap_rows: torch.Tensor
    anchor_episodes: torch.Tensor
    goal_episodes: torch.Tensor


class EffEpisodeReplay:
    """Read-only shared latent replay with explicit step and boundary units.

    ``terminal_at_state[row]`` is true only for a verified environment terminal
    at that state. A time-limit/data boundary must not be invented as terminal.
    Metadata is mandatory; no implicit all-terminal/all-truncation fallback.

    Within-episode offsets are sampled uniformly first, then a legal anchor is
    sampled for that offset, avoiding the strong near-goal bias of clip-local
    sampling. The paper does not publish bin edges/RNG details; this is an
    explicit implementation of its described offset-balanced episode recipe,
    not a claim of bitwise identity to unpublished source code.
    """

    def __init__(
        self,
        store: Any,
        *,
        episodes: tuple[int, ...],
        stride: int,
        terminal_at_state: np.ndarray,
    ) -> None:
        if isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be positive.")
        if not episodes or len(set(episodes)) != len(episodes):
            raise ValueError("Replay needs a nonempty set of unique episode IDs.")
        self.store = store
        self.stride = stride
        self.episodes = np.asarray(episodes, dtype=np.int64)
        ids = np.asarray(store.episode_ids)
        if ids.ndim != 1 or np.any(ids[1:] < ids[:-1]):
            raise ValueError("Source episodes must be contiguous ordered segments.")
        if np.asarray(store.latents).shape != (len(ids), STATE_DIM):
            raise ValueError("Frozen latents must be [global rows, 192].")
        terminal = np.asarray(terminal_at_state)
        if terminal.shape != ids.shape or terminal.dtype != np.bool_:
            raise ValueError("Explicit row-aligned Boolean terminal metadata required.")
        self.terminal = terminal
        self.rows: dict[int, np.ndarray] = {}
        self.costs: dict[int, np.ndarray] = {}
        self.cost_prefix: dict[int, np.ndarray] = {}
        self.terminal_prefix = np.concatenate(
            ([0], np.cumsum(terminal, dtype=np.int64))
        )
        for episode in self.episodes:
            start = int(np.searchsorted(ids, episode, side="left"))
            end = int(np.searchsorted(ids, episode, side="right"))
            if start == end:
                raise ValueError(f"Episode {episode} missing from source store.")
            rows = np.arange(start, end, stride, dtype=np.int64)
            if len(rows) < 2:
                raise ValueError(f"Episode {episode} has no complete state transition.")
            # O(episode length * 192) transient memory, not a copied latent store.
            states = np.asarray(store.latents[rows], dtype=np.float32)
            distances = np.linalg.norm(np.diff(states, axis=0), axis=-1)
            if not np.all(np.isfinite(distances)):
                raise ValueError("Nonfinite frozen states in replay.")
            self.rows[int(episode)] = rows
            self.costs[int(episode)] = distances
            self.cost_prefix[int(episode)] = np.concatenate(
                ([0.0], np.cumsum(distances, dtype=np.float64))
            )

    def _crosses_terminal(self, first_row: int, final_row: int) -> bool:
        # Inclusive of starting state, exclusive of the final (arrival) state.
        return bool(self.terminal_prefix[final_row] - self.terminal_prefix[first_row])

    def sample(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
        backup_primitive_steps: int,
        cross_episode_probability: float,
        epsilon: float,
    ) -> EffReplayBatch:
        if (
            batch_size < 1
            or backup_primitive_steps < 1
            or backup_primitive_steps % self.stride
        ):
            raise ValueError(
                "Positive batch and stride-aligned primitive backup required."
            )
        if not 0 <= cross_episode_probability <= 1 or epsilon <= 0:
            raise ValueError("Invalid cross-episode probability or epsilon.")
        if cross_episode_probability and len(self.episodes) < 2:
            raise ValueError("Cross-episode sampling needs >=2 source episodes.")
        backup = backup_primitive_steps // self.stride
        records = []
        for _ in range(batch_size):
            episode_index = int(rng.integers(len(self.episodes)))
            episode = int(self.episodes[episode_index])
            rows = self.rows[episode]
            cross = bool(rng.random() < cross_episode_probability)
            if cross:
                anchor = int(rng.integers(len(rows) - 1))
                other_index = int(rng.integers(len(self.episodes) - 1))
                other_index += other_index >= episode_index
                goal_episode = int(self.episodes[other_index])
                goal_row = int(rng.choice(self.rows[goal_episode]))
                delta = -1
            else:
                delta = int(rng.integers(1, len(rows)))
                anchor = int(rng.integers(len(rows) - delta))
                goal_episode = episode
                goal_row = int(rows[anchor + delta])
            anchor_row = int(rows[anchor])
            next_row = int(rows[anchor + 1])
            backup_index = min(anchor + backup, len(rows) - 1)
            direct = not cross and anchor + delta <= backup_index
            target_index = anchor + delta if direct else backup_index
            target_row = int(rows[target_index])
            prefix = self.cost_prefix[episode]
            cost = float(prefix[target_index] - prefix[anchor])
            continuation = not self.terminal[target_row] and not self._crosses_terminal(
                anchor_row, target_row
            )
            # A goal on an invalid post-terminal continuation is not a real path.
            direct = direct and not self._crosses_terminal(anchor_row, goal_row)
            path_known = not cross and not self._crosses_terminal(anchor_row, goal_row)
            eta = 0.0
            if path_known:
                full_cost = float(prefix[anchor + delta] - prefix[anchor])
                net = float(
                    np.linalg.norm(
                        self.store.latents[goal_row] - self.store.latents[anchor_row]
                    )
                )
                path_known = full_cost > epsilon and net > epsilon
                if path_known:
                    eta = net / (full_cost + epsilon)
            vector_valid = not self._crosses_terminal(anchor_row, next_row)
            records.append(
                (
                    anchor_row,
                    next_row,
                    target_row,
                    goal_row,
                    episode,
                    goal_episode,
                    cost,
                    direct,
                    continuation,
                    bool(self.terminal[next_row]),
                    vector_valid,
                    eta,
                    path_known,
                    delta,
                )
            )
        arrays = list(zip(*records, strict=True))
        ar, nr, br, gr = (np.asarray(arrays[i], dtype=np.int64) for i in range(4))

        def latents(indices: np.ndarray) -> torch.Tensor:
            # Advanced indexing copies only this minibatch out of the shared mmap.
            return torch.from_numpy(
                np.array(self.store.latents[indices], dtype=np.float32)
            )

        return EffReplayBatch(
            state=latents(ar),
            next_state=latents(nr),
            bootstrap_state=latents(br),
            goal=latents(gr),
            terminal_after_transition=torch.tensor(arrays[9]),
            observed_cost=torch.tensor(arrays[6], dtype=torch.float32),
            goal_reached=torch.tensor(arrays[7]),
            continuation_valid=torch.tensor(arrays[8]),
            vector_valid=torch.tensor(arrays[10]),
            efficiency=torch.tensor(arrays[11], dtype=torch.float32),
            efficiency_known=torch.tensor(arrays[12]),
            offset_chunks=torch.tensor(arrays[13]),
            anchor_rows=torch.from_numpy(ar),
            goal_rows=torch.from_numpy(gr),
            bootstrap_rows=torch.from_numpy(br),
            anchor_episodes=torch.tensor(arrays[4]),
            goal_episodes=torch.tensor(arrays[5]),
        )


def held_out_episode_pairs(
    lengths: np.ndarray,
    *,
    episode_ids: tuple[int, ...],
    goal_offset: int,
    count: int,
    seed: int,
) -> dict[str, list[int]]:
    """Uniform legal start-goal rows in explicitly held-out episodes only.

    Pair selection is independent of model/method/score, and the resulting
    manifest must be saved once and shared by F-only, Eff and EffPlan.
    """
    if count < 1 or goal_offset < 1 or not episode_ids:
        raise ValueError("Invalid evaluation pair request.")
    ids = np.asarray(episode_ids, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if (
        len(set(episode_ids)) != len(ids)
        or np.any(ids < 0)
        or np.any(ids >= len(lengths))
    ):
        raise ValueError("Invalid held-out episode IDs.")
    available = np.maximum(lengths[ids] - goal_offset, 0)
    cumulative = np.cumsum(available)
    if not len(cumulative) or cumulative[-1] < count:
        raise ValueError("Insufficient distinct held-out start-goal pairs.")
    ranks = np.random.default_rng(seed).choice(
        int(cumulative[-1]), count, replace=False
    )
    selected = np.searchsorted(cumulative, ranks, side="right")
    previous = np.concatenate(([0], cumulative[:-1]))
    starts = ranks - previous[selected]
    return {
        "episode_indices": ids[selected].tolist(),
        "start_steps": starts.tolist(),
        "goal_steps": (starts + goal_offset).tolist(),
        "valid_row_ranks": ranks.tolist(),
    }


@dataclass(frozen=True)
class EffPlanReplayBatch:
    real_states: torch.Tensor
    trajectory_valid: torch.Tensor
    rows: torch.Tensor
    anchor_episodes: torch.Tensor
    goal_episodes: torch.Tensor
    state_stride: int


def sample_planner_paths(
    replay: EffEpisodeReplay,
    *,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
    cross_episode_probability: float,
) -> EffPlanReplayBatch:
    """Fixed-time real nodes for P; cross-episode goals NEVER get midpoint labels.

    With H=5 and stride=5, each labelled path contains six real states spanning
    exactly 25 primitive steps. A substituted cross-episode goal changes only
    the endpoint query, and disables the whole path's trajectory-fit loss.
    Its efficiency/dynamics objectives may still be evaluated if explicitly
    enabled by the selected phase-2 protocol.
    """
    if batch_size < 1 or horizon < 2 or not 0 <= cross_episode_probability <= 1:
        raise ValueError("Invalid planner path sampling request.")
    if cross_episode_probability and len(replay.episodes) < 2:
        raise ValueError("Cross-episode planner goals require >=2 episodes.")
    if any(len(replay.rows[int(e)]) <= horizon for e in replay.episodes):
        raise ValueError("Every source episode must contain the full planner horizon.")
    paths, valid, anchors, goals = [], [], [], []
    for _ in range(batch_size):
        index = int(rng.integers(len(replay.episodes)))
        episode = int(replay.episodes[index])
        candidates = replay.rows[episode]
        anchor = int(rng.integers(len(candidates) - horizon))
        rows = candidates[anchor : anchor + horizon + 1].copy()
        legal = not replay._crosses_terminal(int(rows[0]), int(rows[-1]))
        goal_episode = episode
        if rng.random() < cross_episode_probability:
            other = int(rng.integers(len(replay.episodes) - 1))
            other += other >= index
            goal_episode = int(replay.episodes[other])
            rows[-1] = rng.choice(replay.rows[goal_episode])
            legal = False
        paths.append(rows)
        valid.append(legal)
        anchors.append(episode)
        goals.append(goal_episode)
    global_rows = np.stack(paths)
    return EffPlanReplayBatch(
        real_states=torch.from_numpy(
            np.array(replay.store.latents[global_rows], dtype=np.float32)
        ),
        trajectory_valid=torch.tensor(valid, dtype=torch.bool),
        rows=torch.from_numpy(global_rows),
        anchor_episodes=torch.tensor(anchors),
        goal_episodes=torch.tensor(goals),
        state_stride=replay.stride,
    )
