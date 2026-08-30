"""Transition-level frozen-cache data access for Actor-Free TD-LeWM V0.

The existing :class:`FrozenLatentClipDataset` intentionally preserves the
baseline clip index.  That is useful for clip-wise objectives, but it makes a
``batch_size`` denote clips and requires every training step to flatten a
whole clip.  TD-JEPA instead samples individual transitions.  This module
builds a compact index over the legal TD positions of an already selected set
of clips and reads only the requested mmap rows in ``__getitem__``.

No image, LeWM encoder, or complete latent clip is touched here.  The selected
clips are first expanded into legal records and then deduplicated by global
row.  A replay sample is therefore uniform over unique transitions rather
than implicitly weighting rows by overlapping-clip multiplicity.  Each
dataset instance deduplicates its own split only; train/validation intersection
policy remains the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from tdwm.training.frozen_latent_store import (
    CUBE_ACTION_BLOCK_DIM,
    FrozenLatentClipDataset,
    FrozenLatentStore,
)

V0_STATE_DIM = 192
V0_ACTION_DIM = CUBE_ACTION_BLOCK_DIM
V0_FIRST_CURRENT_INDEX = 3


def _integer_vector(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in ("i", "u"):
        raise TypeError(f"{label} must contain integers.")
    if array.dtype.kind == "u" and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{label} exceeds int64.")
    return np.asarray(array, dtype=np.int64)


def _integer_matrix(value: Any, *, label: str, columns: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != int(columns):
        raise ValueError(f"{label} must have shape [rows, {int(columns)}].")
    if array.dtype.kind not in ("i", "u"):
        raise TypeError(f"{label} must contain integers.")
    if array.dtype.kind == "u" and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{label} exceeds int64.")
    return np.asarray(array, dtype=np.int64)


def _canonical_item_indices(
    indices: Sequence[int],
    *,
    length: int,
) -> np.ndarray:
    canonical = _integer_vector(indices, label="indices")
    if canonical.size == 0:
        return canonical
    canonical = np.array(canonical, copy=True)
    canonical[canonical < 0] += int(length)
    if np.any(canonical < 0) or np.any(canonical >= int(length)):
        raise IndexError("A V0 transition index is out of range.")
    return canonical


@dataclass(frozen=True)
class ReachableFutureLatentsV0:
    """One selected, episode-reachable future latent per transition."""

    latents: torch.Tensor
    global_rows: torch.Tensor
    offset_steps: torch.Tensor


class FrozenActorFreeTDV0TransitionDataset:
    """Expand selected frozen clips into individual V0 TD transitions.

    Args:
        clip_dataset: Validated row-backed clip dataset.
        clip_indices: Indices in ``clip_dataset`` belonging to one split.
        first_current_index: First TD position in each clip.  V0 fixes this to
            the three-frame LeWM history boundary.
        index_chunk_clips: Maximum number of clips whose action completeness is
            inspected in one temporary NumPy allocation.

    Every sample contains one ``state, action, next_state, next_action`` tuple,
    never a time axis.  Overlapping clips produce only one replay record per
    global row.  When several clips cover the same row, its matched-goal range
    uses the farthest reachable future; an equal-bound tie uses the smallest
    source clip index.  Terminal action blocks retain all recorded normalized
    values and replace only the cache's missing (NaN) slots with zero.  The
    explicit ``terminal`` flag masks that bootstrap in the TD target.  It is
    also true when a finite next action has no same-episode next-next state.

    ``duplicate_record_count`` reports discarded multiplicity
    (expanded-minus-unique), while ``duplicate_global_row_count`` reports how
    many unique rows had multiplicity greater than one.
    """

    def __init__(
        self,
        clip_dataset: FrozenLatentClipDataset,
        clip_indices: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        first_current_index: int = V0_FIRST_CURRENT_INDEX,
        index_chunk_clips: int = 8192,
    ) -> None:
        if not isinstance(clip_dataset, FrozenLatentClipDataset):
            raise TypeError("clip_dataset must be a validated FrozenLatentClipDataset.")
        store = clip_dataset.store
        if store.embed_dim != V0_STATE_DIM:
            raise ValueError("V0 requires 192-dimensional frozen LeWM states.")
        if store.action_block_dim != V0_ACTION_DIM:
            raise ValueError("V0 requires 25-dimensional normalized action blocks.")
        first_current = int(first_current_index)
        if first_current != V0_FIRST_CURRENT_INDEX:
            raise ValueError("V0 fixes first_current_index to 3.")
        if store.history_frames != first_current:
            raise ValueError("Frozen-store history_frames differs from V0.")
        if clip_dataset.frameskip != store.frame_skip:
            raise ValueError("Frozen clip and store frame skips differ.")
        if clip_dataset.num_steps < first_current + 2:
            raise ValueError("Each V0 clip must leave a current and next state.")
        if isinstance(index_chunk_clips, bool) or int(index_chunk_clips) <= 0:
            raise ValueError("index_chunk_clips must be a positive integer.")

        selected = _integer_vector(clip_indices, label="clip_indices")
        if selected.size == 0:
            raise ValueError("clip_indices cannot be empty.")
        if np.any(selected < 0) or np.any(selected >= len(clip_dataset)):
            raise IndexError("clip_indices contains an out-of-range clip index.")

        store._assert_immutable()
        clip_table = _integer_matrix(
            clip_dataset.dataset.clip_indices,
            label="dataset.clip_indices",
            columns=2,
        )
        if clip_table.shape[0] != len(clip_dataset):
            raise ValueError("Dataset clip_indices length is inconsistent.")
        selected_clips = clip_table[selected]
        clip_episodes = selected_clips[:, 0]
        local_starts = selected_clips[:, 1]
        if np.any(clip_episodes < 0) or np.any(
            clip_episodes >= clip_dataset.lengths.size
        ):
            raise ValueError("A selected clip references an invalid episode.")
        selected_episode_lengths = clip_dataset.lengths[clip_episodes]
        span = int(clip_dataset.dataset.span)
        if np.any(local_starts < 0) or np.any(
            local_starts > selected_episode_lengths - span
        ):
            raise ValueError("A selected clip crosses its episode boundary.")
        clip_starts = clip_dataset.offsets[clip_episodes] + local_starts

        dense_final_rows = clip_starts + span - 1
        if np.any(dense_final_rows >= store.total_rows):
            raise IndexError("A selected V0 clip exceeds the frozen store.")
        actual_start_episodes = np.asarray(
            store.episode_ids[clip_starts], dtype=np.int64
        )
        actual_final_episodes = np.asarray(
            store.episode_ids[dense_final_rows], dtype=np.int64
        )
        if not np.array_equal(clip_episodes, actual_start_episodes) or not (
            np.array_equal(clip_episodes, actual_final_episodes)
        ):
            raise ValueError("A selected V0 clip crosses an episode boundary.")

        td_positions = np.arange(
            first_current,
            clip_dataset.num_steps - 1,
            dtype=np.int64,
        )
        frame_skip = int(store.frame_skip)
        final_position = int(clip_dataset.num_steps - 1)
        record_rows: list[np.ndarray] = []
        record_future_ends: list[np.ndarray] = []
        record_terminals: list[np.ndarray] = []
        record_clip_indices: list[np.ndarray] = []
        record_clip_positions: list[np.ndarray] = []

        chunk_size = int(index_chunk_clips)
        for chunk_start in range(0, selected.size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, selected.size)
            starts = clip_starts[chunk_start:chunk_end]
            rows = starts[:, None] + frame_skip * td_positions[None, :]
            next_rows = rows + frame_skip
            episodes = clip_episodes[chunk_start:chunk_end, None]
            if np.any(np.asarray(store.episode_ids[rows]) != episodes) or np.any(
                np.asarray(store.episode_ids[next_rows]) != episodes
            ):
                raise ValueError("A V0 TD pair crosses an episode boundary.")

            current_actions = np.asarray(store.actions[rows])
            next_actions = np.asarray(store.actions[next_rows])
            current_finite = np.isfinite(current_actions).all(axis=-1)
            next_finite = np.isfinite(next_actions).all(axis=-1)
            next_next_rows = next_rows + frame_skip
            next_next_same_episode = np.zeros(next_next_rows.shape, dtype=np.bool_)
            next_next_in_bounds = next_next_rows < store.total_rows
            if np.any(next_next_in_bounds):
                next_next_same_episode[next_next_in_bounds] = (
                    np.asarray(store.episode_ids[next_next_rows[next_next_in_bounds]])
                    == np.broadcast_to(episodes, rows.shape)[next_next_in_bounds]
                )

            # Once the current block is incomplete, that position and all
            # later positions are post-terminal and are not legal TD records.
            legal = np.logical_and.accumulate(current_finite, axis=1)
            # A finite recorded next action is insufficient for continuation
            # when its resulting next-next state lies outside this episode.
            terminal = (~next_finite) | (~next_next_same_episode)
            terminal_goal_positions = np.where(
                terminal,
                td_positions[None, :] + 1,
                final_position,
            )
            future_end_positions = np.minimum.accumulate(
                terminal_goal_positions[:, ::-1], axis=1
            )[:, ::-1]
            future_end_rows = starts[:, None] + (frame_skip * future_end_positions)

            selected_matrix = np.broadcast_to(
                selected[chunk_start:chunk_end, None], rows.shape
            )
            position_matrix = np.broadcast_to(td_positions[None, :], rows.shape)
            record_rows.append(np.asarray(rows[legal], dtype=np.int64))
            record_future_ends.append(
                np.asarray(future_end_rows[legal], dtype=np.int64)
            )
            record_terminals.append(np.asarray(terminal[legal], dtype=np.bool_))
            record_clip_indices.append(
                np.asarray(selected_matrix[legal], dtype=np.int64)
            )
            record_clip_positions.append(
                np.asarray(position_matrix[legal], dtype=np.int64)
            )

        expanded_rows = np.concatenate(record_rows)
        if expanded_rows.size == 0:
            raise ValueError("The selected clips contain no legal V0 transitions.")
        expanded_future_ends = np.concatenate(record_future_ends)
        expanded_terminals = np.concatenate(record_terminals)
        expanded_clip_indices = np.concatenate(record_clip_indices)
        expanded_clip_positions = np.concatenate(record_clip_positions)
        # np.lexsort uses the final key as primary.  Ordering each global-row
        # group by farthest future first, then smallest clip/position, makes
        # its first entry the fully deterministic canonical record.
        order = np.lexsort(
            (
                expanded_clip_positions,
                expanded_clip_indices,
                -expanded_future_ends,
                expanded_rows,
            )
        )
        sorted_rows = expanded_rows[order]
        sorted_future_ends = expanded_future_ends[order]
        sorted_terminals = expanded_terminals[order]
        sorted_clip_indices = expanded_clip_indices[order]
        sorted_clip_positions = expanded_clip_positions[order]
        global_rows, first_indices, multiplicities = np.unique(
            sorted_rows,
            return_index=True,
            return_counts=True,
        )
        terminal_values = sorted_terminals.astype(np.int8, copy=False)
        terminal_minima = np.minimum.reduceat(terminal_values, first_indices)
        terminal_maxima = np.maximum.reduceat(terminal_values, first_indices)
        if np.any(terminal_minima != terminal_maxima):
            raise RuntimeError(
                "Overlapping clips disagree on a transition terminal flag."
            )

        # Lexicographic ordering already placed the canonical entry first in
        # every group, so no Python loop over million-scale unique rows occurs.
        future_end_rows = sorted_future_ends[first_indices]
        terminals = sorted_terminals[first_indices]
        source_clip_indices = sorted_clip_indices[first_indices]
        clip_positions = sorted_clip_positions[first_indices]
        if np.any(future_end_rows < global_rows + frame_skip):
            raise RuntimeError("A V0 future-goal range is empty.")
        if np.any((future_end_rows - global_rows) % frame_skip != 0):
            raise RuntimeError("A V0 future-goal range is not frame aligned.")

        self.store = store
        self.frame_skip = frame_skip
        self.first_current_index = first_current
        self.global_rows = global_rows
        self.future_end_rows = future_end_rows
        self.terminals = terminals
        self.source_clip_indices = source_clip_indices
        self.clip_positions = clip_positions
        self.record_multiplicities = np.asarray(multiplicities, dtype=np.int64)
        self.expanded_record_count = int(expanded_rows.size)
        self.unique_record_count = int(global_rows.size)
        self.duplicate_record_count = int(expanded_rows.size - global_rows.size)
        self.duplicate_global_row_count = int((multiplicities > 1).sum())
        self.max_multiplicity = int(multiplicities.max())
        self.population_diagnostics = {
            "expanded_record_count": self.expanded_record_count,
            "unique_record_count": self.unique_record_count,
            "duplicate_record_count": self.duplicate_record_count,
            "duplicate_global_row_count": self.duplicate_global_row_count,
            "max_multiplicity": self.max_multiplicity,
        }

    def __len__(self) -> int:
        return int(self.global_rows.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(
        self,
        indices: Sequence[int],
    ) -> list[dict[str, torch.Tensor]]:
        if len(indices) == 0:
            return []
        positions = _canonical_item_indices(indices, length=len(self))
        self.store._assert_immutable()
        rows = self.global_rows[positions]
        next_rows = rows + self.frame_skip
        future_ends = self.future_end_rows[positions]
        terminals = self.terminals[positions]
        episode_ids = np.asarray(self.store.episode_ids[rows], dtype=np.int64)

        states = torch.from_numpy(
            np.array(self.store.latents[rows], dtype=np.float32, copy=True)
        )
        next_states = torch.from_numpy(
            np.array(self.store.latents[next_rows], dtype=np.float32, copy=True)
        )
        actions_array = np.array(self.store.actions[rows], dtype=np.float32, copy=True)
        next_actions_array = np.array(
            self.store.actions[next_rows], dtype=np.float32, copy=True
        )
        if not np.isfinite(actions_array).all():
            raise RuntimeError("A pre-indexed V0 current action became non-finite.")
        observed_terminals = ~np.isfinite(next_actions_array).all(axis=-1)
        next_next_rows = next_rows + self.frame_skip
        next_next_same_episode = np.zeros(next_next_rows.shape, dtype=np.bool_)
        next_next_in_bounds = next_next_rows < self.store.total_rows
        if np.any(next_next_in_bounds):
            next_next_same_episode[next_next_in_bounds] = (
                np.asarray(self.store.episode_ids[next_next_rows[next_next_in_bounds]])
                == episode_ids[next_next_in_bounds]
            )
        observed_terminals |= ~next_next_same_episode
        if not np.array_equal(observed_terminals, terminals):
            raise RuntimeError("A pre-indexed V0 terminal marker changed.")
        np.nan_to_num(
            next_actions_array,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        actions = torch.from_numpy(actions_array)
        next_actions = torch.from_numpy(next_actions_array)
        future_counts = (future_ends - rows) // self.frame_skip

        return [
            {
                "state": states[position],
                "action": actions[position],
                "next_state": next_states[position],
                "next_action": next_actions[position],
                "terminal": torch.tensor(bool(terminals[position])),
                "global_row": torch.tensor(int(rows[position]), dtype=torch.int64),
                "episode_id": torch.tensor(
                    int(episode_ids[position]), dtype=torch.int64
                ),
                "goal_future_start_row": torch.tensor(
                    int(next_rows[position]), dtype=torch.int64
                ),
                "goal_future_end_row": torch.tensor(
                    int(future_ends[position]), dtype=torch.int64
                ),
                "goal_future_count": torch.tensor(
                    int(future_counts[position]), dtype=torch.int64
                ),
                "source_clip_index": torch.tensor(
                    int(self.source_clip_indices[positions[position]]),
                    dtype=torch.int64,
                ),
                "clip_position": torch.tensor(
                    int(self.clip_positions[positions[position]]),
                    dtype=torch.int64,
                ),
            }
            for position in range(positions.size)
        ]


def sample_reachable_future_latents_v0(
    store: FrozenLatentStore,
    global_rows: Sequence[int] | np.ndarray | torch.Tensor,
    future_end_rows: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    generator: torch.Generator | None,
    device: str | torch.device = "cpu",
) -> ReachableFutureLatentsV0:
    """Uniformly sample one reachable future latent from per-item bounds.

    ``future_end_rows`` is inclusive.  The first eligible row is always the
    dataset next-state row, ``global_row + frame_skip``.  With a caller-owned
    CPU ``generator``, selection is uniform and keeps goal randomness
    independent from predictor dropout and CUDA RNG state.  Formal validation
    passes its own generator and resets it at every validation epoch, so it
    follows this same uniform distribution over one fixed sampled population.
    ``generator=None`` remains an explicit endpoint-selection mode for audits
    and diagnostics; it is not the formal validation sampler.
    """

    if not isinstance(store, FrozenLatentStore):
        raise TypeError("store must be a validated FrozenLatentStore.")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")
    if generator is not None and torch.device(generator.device).type != "cpu":
        raise ValueError("The V0 future-goal generator must be a CPU generator.")
    rows = _integer_vector(global_rows, label="global_rows")
    ends = _integer_vector(future_end_rows, label="future_end_rows")
    if rows.size == 0 or ends.shape != rows.shape:
        raise ValueError("Future-goal bounds must be non-empty aligned vectors.")
    frame_skip = int(store.frame_skip)
    differences = ends - rows
    if np.any(rows < 0) or np.any(ends >= store.total_rows):
        raise IndexError("A future-goal bound is outside the frozen store.")
    if np.any(differences < frame_skip) or np.any(differences % frame_skip != 0):
        raise ValueError("Future-goal bounds must be positive and frame aligned.")
    store._assert_immutable()
    if np.any(store.episode_ids[ends] != store.episode_ids[rows]):
        raise ValueError("A V0 future-goal bound crosses an episode boundary.")
    maximum_offsets = differences // frame_skip
    if generator is None:
        offset_steps = np.array(maximum_offsets, dtype=np.int64, copy=True)
    else:
        uniform = torch.rand(rows.size, generator=generator, device="cpu").numpy()
        offset_steps = np.floor(uniform * maximum_offsets).astype(np.int64) + 1
    sampled_rows = rows + frame_skip * offset_steps
    if np.any(store.episode_ids[sampled_rows] != store.episode_ids[rows]):
        raise ValueError("A sampled V0 future goal crosses an episode boundary.")

    latents = torch.from_numpy(
        np.array(store.latents[sampled_rows], dtype=np.float32, copy=True)
    )
    target_device = torch.device(device)
    return ReachableFutureLatentsV0(
        latents=latents.to(device=target_device, non_blocking=True),
        global_rows=torch.from_numpy(np.array(sampled_rows, copy=True)).to(
            device=target_device, non_blocking=True
        ),
        offset_steps=torch.from_numpy(np.array(offset_steps, copy=True)).to(
            device=target_device, non_blocking=True
        ),
    )


__all__ = [
    "FrozenActorFreeTDV0TransitionDataset",
    "ReachableFutureLatentsV0",
    "V0_ACTION_DIM",
    "V0_FIRST_CURRENT_INDEX",
    "V0_STATE_DIM",
    "sample_reachable_future_latents_v0",
]
