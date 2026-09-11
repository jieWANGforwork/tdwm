"""Episode-disjoint, same-trajectory finite-goal samples for EffAction.

RP1 publishes the 8,000/2,000 episode split and balanced temporal offsets,
but not its sampler implementation. The explicit convention here samples a
chunk offset uniformly, then a valid row uniformly at that offset. It is not
claimed to reproduce an unpublished RP1 sampler bit for bit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from tdwm.methods.actor_free_td_lewm_v1 import project_tasks_to_sphere_v1
from tdwm.training.frozen_latent_store import FrozenLatentStore, validate_episode_ids


@dataclass(frozen=True)
class EffActionSamplingConfig:
    episode_start: int
    episode_stop: int
    min_goal_chunks: int
    max_goal_chunks: int
    grid_phase: int
    efficiency_threshold: float
    efficiency_epsilon: float
    zero_distance_tolerance: float
    direct_max_chunks: int | None
    action_block_steps: int = 5
    goal_source: str = "same_episode_future"
    offset_distribution: str = "uniform_offset_then_uniform_valid_start"
    zero_distance_policy: str = "exclude"

    def __post_init__(self) -> None:
        for name in ("episode_start", "episode_stop", "min_goal_chunks", "max_goal_chunks", "grid_phase", "action_block_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if not 0 <= self.episode_start < self.episode_stop:
            raise ValueError("episode range must be a nonempty half-open interval.")
        if not 1 <= self.min_goal_chunks <= self.max_goal_chunks:
            raise ValueError("goal chunk range must be positive and ordered.")
        if self.action_block_steps != 5 or not 0 <= self.grid_phase < 5:
            raise ValueError("LeWM uses five-step blocks with explicit phase 0..4.")
        if not np.isfinite(self.efficiency_threshold) or not 0 <= self.efficiency_threshold <= 1:
            raise ValueError("efficiency_threshold must be in [0, 1].")
        if not np.isfinite(self.efficiency_epsilon) or self.efficiency_epsilon <= 0:
            raise ValueError("efficiency_epsilon must be finite and positive.")
        if not np.isfinite(self.zero_distance_tolerance) or self.zero_distance_tolerance < 0:
            raise ValueError("zero_distance_tolerance must be finite and nonnegative.")
        if self.direct_max_chunks is not None and (isinstance(self.direct_max_chunks, bool) or not isinstance(self.direct_max_chunks, int) or self.direct_max_chunks < 1):
            raise ValueError("direct_max_chunks must be null or a positive integer.")
        if self.goal_source != "same_episode_future":
            raise ValueError("Cross-episode goals require a separately defined target and planner label.")
        if self.offset_distribution != "uniform_offset_then_uniform_valid_start":
            raise ValueError("Unsupported offset distribution; do not silently change sampling.")
        if self.zero_distance_policy != "exclude":
            raise ValueError("Only explicitly excluding degenerate zero-distance paths is implemented.")


class EffActionEpisodeData:
    """Read-only array view; action rows and latents share raw dataset indices.

    Each sample follows real transitions to its sampled goal. A five-step
    chunk holds five independent five-dimensional normalized actions. The
    next action is unused at the goal and returned as NaN rather than forged.
    """

    def __init__(self, latents: np.ndarray, actions: np.ndarray, episode_ids: np.ndarray, *, config: EffActionSamplingConfig) -> None:
        if latents.ndim != 2 or latents.shape[1] != 192 or latents.dtype != np.float32:
            raise ValueError("latents must be float32 [rows, 192].")
        if actions.shape != (len(latents), 25) or actions.dtype != np.float32:
            raise ValueError("actions must be normalized float32 [rows, 25].")
        ids = validate_episode_ids(episode_ids, total_rows=len(latents))
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        stops = np.r_[starts[1:], len(ids)]
        if config.episode_stop > len(starts):
            raise ValueError("Configured episode split exceeds the dataset.")
        self.latents, self.actions, self.episode_ids = latents, actions, ids
        self.config = config
        self.episode_starts, self.episode_stops = starts, stops
        self.store: FrozenLatentStore | None = None
        self.offsets = np.arange(config.min_goal_chunks, config.max_goal_chunks + 1, dtype=np.int64)
        # Build integer row indices only; never duplicate the frozen cache.
        by_offset: dict[int, list[np.ndarray]] = {int(k): [] for k in self.offsets}
        for episode in range(config.episode_start, config.episode_stop):
            grid = np.arange(starts[episode] + config.grid_phase, stops[episode], 5, dtype=np.int64)
            if len(grid) < 2:
                continue
            z = np.asarray(latents[grid])
            finite_z = np.isfinite(z).all(axis=1)
            finite_a = np.isfinite(actions[grid[:-1]]).all(axis=1)
            bad = ~(finite_z[:-1] & finite_z[1:] & finite_a)
            cumulative_bad = np.r_[0, np.cumsum(bad, dtype=np.int64)]
            for offset in self.offsets:
                if offset >= len(grid):
                    continue
                path_valid = cumulative_bad[offset:] == cumulative_bad[:-offset]
                distance = np.linalg.norm(z[offset:] - z[:-offset], axis=1)
                path_valid &= np.isfinite(distance) & (distance > config.zero_distance_tolerance)
                path_valid &= np.linalg.norm(z[offset:], axis=1) > np.finfo(np.float32).eps
                if np.any(path_valid):
                    by_offset[int(offset)].append(grid[:-offset][path_valid])
        self.rows_by_offset = {
            offset: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
            for offset, chunks in by_offset.items()
        }
        missing = [offset for offset, rows in self.rows_by_offset.items() if not len(rows)]
        if missing:
            raise ValueError(f"No valid nondegenerate paths for configured offsets {missing}; revise the explicit protocol.")

    @classmethod
    def from_store(cls, store: FrozenLatentStore, *, config: EffActionSamplingConfig) -> EffActionEpisodeData:
        if store.frame_skip != 5 or store.action_block_dim != 25:
            raise ValueError("EffAction requires the audited five-step LeWM cache.")
        result = cls(store.latents, store.actions, store.episode_ids, config=config)
        result.store = store
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "sampling": asdict(self.config),
            "split_unit": "whole_episode",
            "episode_count": self.config.episode_stop - self.config.episode_start,
            "eligible_pairs_per_offset": {str(k): len(v) for k, v in self.rows_by_offset.items()},
            "rp1_implementation_match": "published_split_and_balanced_offsets_only; sampler_is_explicit_local_convention",
            "latent_store_manifest_sha256": self.store.manifest_sha256 if self.store else None,
            "terminal": "sampled_goal_reached",
            "direct_successor": "sum_of_real_future_latents_including_goal",
            "path_length": "sum_of_adjacent_chunk_latent_distances",
            "discount": None,
        }

    def sample(self, batch_size: int, *, rng: np.random.Generator, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        offsets = rng.choice(self.offsets, size=batch_size)
        anchors = np.empty(batch_size, dtype=np.int64)
        for offset in np.unique(offsets):
            selected = offsets == offset
            population = self.rows_by_offset[int(offset)]
            anchors[selected] = population[rng.integers(len(population), size=int(selected.sum()))]
        return self.batch_for_rows(anchors, anchors + 5 * offsets, device=device)

    def batch_for_rows(self, anchors: np.ndarray, goals: np.ndarray, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if self.store:
            self.store._assert_immutable()
        anchors, goals = np.asarray(anchors), np.asarray(goals)
        if anchors.ndim != 1 or anchors.dtype.kind not in "iu" or goals.shape != anchors.shape or goals.dtype.kind not in "iu" or not anchors.size:
            raise ValueError("anchor and goal rows must be equally sized nonempty integer vectors.")
        anchors, goals = anchors.astype(np.int64), goals.astype(np.int64)
        if np.any(anchors < 0) or np.any(goals >= len(self.latents)) or np.any(goals <= anchors):
            raise ValueError("Goal must be a valid future dataset row.")
        episodes = self.episode_ids[anchors]
        if np.any(episodes != self.episode_ids[goals]):
            raise ValueError("An EffAction training path cannot cross episodes.")
        cfg = self.config
        if np.any(episodes < cfg.episode_start) or np.any(episodes >= cfg.episode_stop):
            raise ValueError("Requested rows leak outside the configured episode split.")
        if np.any((anchors - self.episode_starts[episodes]) % 5 != cfg.grid_phase) or np.any((goals - anchors) % 5):
            raise ValueError("Rows must follow the configured five-step grid phase.")
        offsets = (goals - anchors) // 5
        if np.any(offsets < cfg.min_goal_chunks) or np.any(offsets > cfg.max_goal_chunks):
            raise ValueError("Goal offsets exceed the explicit sampling range.")
        steps = np.arange(int(offsets.max()) + 1, dtype=np.int64)
        rows = np.minimum(anchors[:, None] + 5 * steps[None, :], goals[:, None])
        path = np.asarray(self.latents[rows], dtype=np.float32)
        future_mask = (steps[None, 1:] <= offsets[:, None])
        action_rows = rows[:, :-1]
        valid_actions = np.isfinite(self.actions[action_rows]).all(axis=-1)
        valid_actions = (valid_actions | ~future_mask).all(axis=-1)
        valid = np.isfinite(path).all(axis=(1, 2)) & valid_actions
        displacement = np.linalg.norm(path[:, -1] - path[:, 0], axis=-1)
        length = np.linalg.norm(np.diff(path, axis=1), axis=-1).sum(axis=-1, dtype=np.float64).astype(np.float32)
        efficiency = displacement / (length + cfg.efficiency_epsilon)
        valid &= np.isfinite(efficiency) & (displacement > cfg.zero_distance_tolerance)
        valid &= np.linalg.norm(path[:, -1], axis=-1) > np.finfo(np.float32).eps
        direct = valid & (efficiency >= cfg.efficiency_threshold)
        if cfg.direct_max_chunks is not None:
            direct &= offsets <= cfg.direct_max_chunks
        total = np.where(future_mask[..., None], path[:, 1:], 0).sum(axis=1, dtype=np.float64).astype(np.float32)
        terminal = offsets == 1
        next_actions = np.full((len(anchors), 25), np.nan, dtype=np.float32)
        next_actions[~terminal] = self.actions[anchors[~terminal] + 5]
        values = {
            "state": path[:, 0], "raw_action": np.asarray(self.actions[anchors]),
            "next_state": path[:, 1], "next_raw_action": next_actions,
            "goal": path[:, -1], "direct_successor_target": total,
            "direct_branch": direct, "valid_mask": valid, "goal_terminal": terminal,
            "net_distance": displacement, "path_length": length, "efficiency": efficiency,
            "anchor_row": anchors, "goal_row": goals, "next_row": anchors + 5,
            "episode_id": episodes, "anchor_step": anchors - self.episode_starts[episodes],
            "goal_chunks": offsets,
        }
        result = {key: torch.as_tensor(np.array(value, copy=True), device=device) for key, value in values.items()}
        # Existing project normalization: goal itself lies on radius sqrt(192),
        # not the goal-current direction and not a randomly mixed task.
        safe_goal = result["goal"].clone()
        safe_goal[~result["valid_mask"]] = 0
        safe_goal[~result["valid_mask"], 0] = 1
        result["task"] = project_tasks_to_sphere_v1(safe_goal)
        return result


EFF_ACTION_LOSS_BATCH_KEYS = (
    "state", "raw_action", "next_state", "next_raw_action", "goal", "task",
    "direct_successor_target", "direct_branch", "valid_mask", "goal_terminal",
)
