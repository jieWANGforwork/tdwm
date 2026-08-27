from __future__ import annotations

import io
import queue
import threading
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class StrideBatchPlan:
    """Rows needed to assemble a batch of strided sequence clips."""

    global_starts: tuple[int, ...]
    unique_frame_rows: tuple[int, ...]
    frame_gathers: tuple[tuple[int, ...], ...]
    legacy_row_requests: int

    @property
    def image_row_requests(self) -> int:
        return len(self.unique_frame_rows)


@dataclass(frozen=True)
class PrefetchedStrideBlock:
    """Decoded rows shared by several consecutive LeWM mini-batches."""

    global_starts: tuple[int, ...]
    frame_gathers: tuple[tuple[int, ...], ...]
    row_data: dict[str, Any]
    decoded_images: dict[str, Any]
    dense_actions: Any


@dataclass
class CachedEpisode:
    """One episode held in the bounded streaming JPEG cache."""

    episode: int
    starts: np.ndarray
    columns: dict[str, Any]
    byte_size: int
    cursor: int = 0

    @property
    def remaining(self) -> int:
        return int(self.starts.size) - self.cursor


def build_stride_batch_plan(
    *,
    clip_indices: Sequence[tuple[int, int]],
    offsets: Sequence[int],
    indices: Sequence[int],
    frameskip: int,
    num_steps: int,
) -> StrideBatchPlan:
    """Map clip indices to only the observation rows consumed by LeWM."""

    if frameskip <= 0:
        raise ValueError("frameskip must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    global_starts: list[int] = []
    sample_rows: list[tuple[int, ...]] = []
    clip_count = len(clip_indices)
    for raw_idx in indices:
        idx = int(raw_idx)
        if idx < 0:
            idx += clip_count
        if idx < 0 or idx >= clip_count:
            raise IndexError(f"Clip index {raw_idx} is out of range.")
        episode, start = clip_indices[idx]
        global_start = int(offsets[episode]) + int(start)
        global_starts.append(global_start)
        sample_rows.append(
            tuple(global_start + step * frameskip for step in range(num_steps))
        )

    unique_frame_rows = tuple(sorted({row for rows in sample_rows for row in rows}))
    row_positions = {row: position for position, row in enumerate(unique_frame_rows)}
    frame_gathers = tuple(
        tuple(row_positions[row] for row in rows) for rows in sample_rows
    )
    return StrideBatchPlan(
        global_starts=tuple(global_starts),
        unique_frame_rows=unique_frame_rows,
        frame_gathers=frame_gathers,
        legacy_row_requests=len(indices) * frameskip * num_steps,
    )


def _decode_images(blobs: Sequence[Any]):
    import torch

    if not blobs:
        return torch.empty(0, dtype=torch.uint8)

    try:
        from torchvision.io import ImageReadMode, decode_jpeg

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            encoded = [
                torch.frombuffer(
                    blob
                    if isinstance(blob, (bytes, bytearray))
                    else bytes(blob),
                    dtype=torch.uint8,
                )
                for blob in blobs
            ]
        return torch.stack(decode_jpeg(encoded, mode=ImageReadMode.RGB))
    except (AttributeError, ImportError, RuntimeError, TypeError):
        from PIL import Image

        decoded = []
        for blob in blobs:
            with Image.open(io.BytesIO(bytes(blob))) as image:
                array = np.array(image.convert("RGB"), copy=True)
            decoded.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(decoded)


def _numeric_tensor(values: Any):
    import torch

    array = np.asarray(values)
    if array.dtype == object or array.dtype.kind in ("S", "U"):
        raise TypeError("The stride-aware Cube loader only accepts numeric columns.")
    tensor = torch.tensor(array)
    if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def _payload_nbytes(values: Any) -> int:
    """Estimate resident payload bytes without decoding JPEG objects."""

    array = np.asarray(values)
    if array.dtype != object:
        return int(array.nbytes)
    total = int(array.nbytes)
    for value in array.reshape(-1):
        try:
            total += memoryview(value).nbytes
        except TypeError:
            total += len(bytes(value))
    return total


class StrideAwareLanceDataset:
    """Avoid fetching intermediate Lance image rows discarded by ``frameskip``.

    The adapter uses the public dataset row/column accessors and preserves the
    released LeWM sample structure. Dense actions are still read across the
    complete temporal span; observation and image columns use strided rows.
    """

    def __init__(self, dataset: Any, decoded_frame_store: Any | None = None) -> None:
        required = (
            "clip_indices",
            "offsets",
            "frameskip",
            "num_steps",
            "span",
            "column_names",
            "get_col_data",
            "get_row_data",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "The Lance dataset is missing required public attributes: "
                + ", ".join(missing)
            )
        columns = list(dataset.column_names)
        if "pixels" not in columns or "action" not in columns:
            raise ValueError(
                "The stride-aware LeWM loader requires pixels and action columns."
            )
        image_columns = {
            name for name in columns if name == "pixels" or name.startswith("pixels_")
        }
        if decoded_frame_store is not None:
            if not callable(getattr(decoded_frame_store, "take", None)):
                raise TypeError("decoded_frame_store must provide take(global_rows).")
            if image_columns != {"pixels"}:
                raise ValueError(
                    "decoded_frame_store currently requires exactly one pixels column."
                )
            payload_bytes = getattr(
                decoded_frame_store, "episode_jpeg_payload_bytes", None
            )
            if not isinstance(payload_bytes, (tuple, list)):
                raise TypeError(
                    "decoded_frame_store.episode_jpeg_payload_bytes must be a "
                    "tuple or list."
                )
            episode_count = len(getattr(dataset, "lengths", ()))
            if len(payload_bytes) != episode_count:
                raise ValueError(
                    "decoded_frame_store episode count differs from the Lance "
                    f"dataset: expected {episode_count}, found {len(payload_bytes)}."
                )
            episode_jpeg_payload_bytes = tuple(int(value) for value in payload_bytes)
            if any(value < 0 for value in episode_jpeg_payload_bytes):
                raise ValueError(
                    "decoded_frame_store episode JPEG payload sizes must be "
                    "non-negative."
                )
        else:
            episode_jpeg_payload_bytes = None
        self.dataset = dataset
        self.decoded_frame_store = decoded_frame_store
        self._episode_jpeg_payload_bytes = episode_jpeg_payload_bytes

    def _take_decoded_frames(self, global_rows: Sequence[int]):
        """Gather raw uint8 NCHW frames from the optional decoded store."""

        import torch

        if self.decoded_frame_store is None:
            raise RuntimeError("No decoded_frame_store is configured.")
        rows = [int(row) for row in global_rows]
        frames = self.decoded_frame_store.take(rows)
        if not isinstance(frames, torch.Tensor):
            raise TypeError("decoded_frame_store.take() must return a torch.Tensor.")
        if frames.dtype != torch.uint8:
            raise TypeError("decoded_frame_store.take() must return torch.uint8 frames.")
        if frames.ndim != 4:
            raise ValueError(
                "decoded_frame_store.take() must return NCHW frames."
            )
        if frames.shape[0] != len(rows):
            raise ValueError(
                "decoded_frame_store.take() returned the wrong number of frames."
            )
        return frames

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def prefetch(self, indices: Sequence[int]) -> PrefetchedStrideBlock:
        """Read and decode the strided rows needed by a local clip block once."""
        if not indices:
            raise ValueError("Cannot prefetch an empty clip block.")

        plan = build_stride_batch_plan(
            clip_indices=self.dataset.clip_indices,
            offsets=self.dataset.offsets,
            indices=indices,
            frameskip=int(self.dataset.frameskip),
            num_steps=int(self.dataset.num_steps),
        )
        frame_rows = list(plan.unique_frame_rows)
        image_columns = {
            name
            for name in self.dataset.column_names
            if name == "pixels" or name.startswith("pixels_")
        }
        if self.decoded_frame_store is None:
            row_data = self.dataset.get_row_data(frame_rows)
            decoded_images = {
                name: _decode_images(np.asarray(row_data[name], dtype=object).tolist())
                for name in image_columns
            }
        else:
            # Do not ask Lance for any JPEG-bearing rows when raw frames are
            # already resident. Numeric columns remain on the dataset's public
            # column API; actions are kept dense for each clip's full span.
            row_data = {
                name: np.asarray(self.dataset.get_col_data(name))[frame_rows]
                for name in self.dataset.column_names
                if name not in image_columns and name != "action"
            }
            decoded_images = {
                "pixels": self._take_decoded_frames(frame_rows),
            }
        return PrefetchedStrideBlock(
            global_starts=plan.global_starts,
            frame_gathers=plan.frame_gathers,
            row_data=row_data,
            decoded_images=decoded_images,
            dense_actions=self.dataset.get_col_data("action"),
        )

    def materialize_prefetched(
        self,
        prefetched: PrefetchedStrideBlock,
        positions: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Assemble selected clips from an already-decoded local block."""

        image_columns = set(prefetched.decoded_images)
        block_size = len(prefetched.global_starts)

        results: list[dict[str, Any]] = []
        for raw_position in positions:
            position = int(raw_position)
            if position < 0 or position >= block_size:
                raise IndexError(f"Block position {raw_position} is out of range.")
            global_start = prefetched.global_starts[position]
            gather = prefetched.frame_gathers[position]
            gather_indices = list(gather)
            steps: dict[str, Any] = {}
            for column in self.dataset.column_names:
                if column in image_columns:
                    steps[column] = prefetched.decoded_images[column][gather_indices]
                elif column == "action":
                    action_end = global_start + int(self.dataset.span)
                    steps[column] = _numeric_tensor(
                        prefetched.dense_actions[global_start:action_end]
                    )
                else:
                    steps[column] = _numeric_tensor(
                        np.asarray(prefetched.row_data[column])[gather_indices]
                    )

            if self.dataset.transform:
                steps = self.dataset.transform(steps)
            steps["action"] = steps["action"].reshape(
                int(self.dataset.num_steps), -1
            )
            results.append(steps)
        return results

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        prefetched = self.prefetch(indices)
        return self.materialize_prefetched(prefetched, range(len(indices)))


class BlockPrefetchBatchDataset(IterableDataset):
    """Yield collated LeWM batches from sequentially fetched, decoded blocks.

    The train split is sorted by backing clip index before a block is fetched,
    so Lance receives one local row request and one JPEG decode pass per block.
    Mini-batches are shuffled only after that data is resident in the worker's
    memory.  This keeps the model inputs and sample coverage unchanged while
    removing the batch-by-batch remote-storage round trips.
    """

    def __init__(
        self,
        dataset: StrideAwareLanceDataset,
        source_indices: Sequence[int],
        *,
        batch_size: int,
        block_size: int,
        drop_last: bool,
        seed: int,
        shuffle_batches_within_block: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if block_size < batch_size:
            raise ValueError("block_size must be at least batch_size.")
        if block_size % batch_size:
            raise ValueError("block_size must be divisible by batch_size.")
        self._dataset = dataset
        self._source_indices = tuple(sorted(int(index) for index in source_indices))
        self._batch_size = batch_size
        self._block_size = block_size
        self._drop_last = drop_last
        self._seed = int(seed)
        self._shuffle_batches_within_block = shuffle_batches_within_block

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._source_indices) // self._batch_size
        return (len(self._source_indices) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        import torch
        from torch.utils.data import default_collate, get_worker_info

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        if worker is None:
            schedule_generator = torch.Generator().manual_seed(self._seed)
        else:
            # DataLoader derives worker seeds from its checkpointed generator.
            # Removing the worker offset makes every worker build the same
            # global block permutation before taking its disjoint share.
            schedule_generator = torch.Generator().manual_seed(
                int(worker.seed) - worker_id
            )

        block_starts = list(range(0, len(self._source_indices), self._block_size))
        if len(block_starts) > 1:
            order = torch.randperm(
                len(block_starts), generator=schedule_generator
            ).tolist()
            block_starts = [block_starts[position] for position in order]

        for block_start in block_starts[worker_id::worker_count]:
            source_block = self._source_indices[
                block_start : block_start + self._block_size
            ]
            prefetched = self._dataset.prefetch(source_block)
            batches = [
                list(range(offset, min(offset + self._batch_size, len(source_block))))
                for offset in range(0, len(source_block), self._batch_size)
            ]
            if self._drop_last and batches and len(batches[-1]) != self._batch_size:
                batches.pop()
            if self._shuffle_batches_within_block and len(batches) > 1:
                order = torch.randperm(
                    len(batches), generator=schedule_generator
                ).tolist()
                batches = [batches[position] for position in order]
            for positions in batches:
                yield default_collate(
                    self._dataset.materialize_prefetched(prefetched, positions)
                )


class EpisodeStreamingBatchDataset(IterableDataset):
    """Mix episodes in each batch while reading Lance rows sequentially.

    A single background thread reads contiguous episode blocks into a bounded
    compressed-JPEG cache. The consumer draws at most one clip from each active
    episode, so SIGReg sees diverse trajectories without random remote reads.
    This dataset must run with ``DataLoader(num_workers=0)``; otherwise every
    worker would duplicate the cache and its I/O stream.
    """

    _END = object()

    def __init__(
        self,
        dataset: StrideAwareLanceDataset,
        source_indices: Sequence[int],
        *,
        batch_size: int,
        active_episodes: int,
        read_episodes: int,
        cache_bytes: int,
        prefetch_blocks: int,
        seed: int,
        drop_last: bool,
        min_unique_episodes: int,
    ) -> None:
        if not isinstance(dataset, StrideAwareLanceDataset):
            raise TypeError("Episode streaming requires StrideAwareLanceDataset.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if active_episodes < batch_size:
            raise ValueError("active_episodes must be at least batch_size.")
        if read_episodes <= 0:
            raise ValueError("read_episodes must be positive.")
        if cache_bytes <= 0:
            raise ValueError("cache_bytes must be positive.")
        if prefetch_blocks <= 0:
            raise ValueError("prefetch_blocks must be positive.")
        if not 1 <= min_unique_episodes <= batch_size:
            raise ValueError("min_unique_episodes must lie in [1, batch_size].")
        if not drop_last:
            raise ValueError("Episode streaming currently requires drop_last=True.")

        self._dataset = dataset
        self._batch_size = int(batch_size)
        self._active_episodes = int(active_episodes)
        self._read_episodes = int(read_episodes)
        self._cache_bytes = int(cache_bytes)
        self._prefetch_blocks = int(prefetch_blocks)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._min_unique_episodes = int(min_unique_episodes)
        self._epoch = 0

        episode_starts: list[list[int]] = [
            [] for _ in range(len(self._dataset.lengths))
        ]
        clip_indices = self._dataset.clip_indices
        for raw_index in source_indices:
            episode, start = clip_indices[int(raw_index)]
            episode_starts[int(episode)].append(int(start))
        self._episode_starts = tuple(
            np.asarray(starts, dtype=np.int32) for starts in episode_starts
        )
        self._source_size = sum(int(starts.size) for starts in self._episode_starts)
        self._episode_ids = tuple(
            episode
            for episode, starts in enumerate(self._episode_starts)
            if starts.size
        )
        if len(self._episode_ids) < self._min_unique_episodes:
            raise ValueError(
                "The training split has fewer episodes than min_unique_episodes."
            )

    def __len__(self) -> int:
        if self._drop_last:
            return self._source_size // self._batch_size
        return (self._source_size + self._batch_size - 1) // self._batch_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self._epoch = int(epoch)

    def _read_episode_block(self, episode_ids: Sequence[int]) -> list[CachedEpisode]:
        source = self._dataset.dataset
        first_episode = int(episode_ids[0])
        last_episode = int(episode_ids[-1])
        first_row = int(source.offsets[first_episode])
        last_row = int(source.offsets[last_episode]) + int(
            source.lengths[last_episode]
        )
        if self._dataset.decoded_frame_store is None:
            rows = list(range(first_row, last_row))
            row_data = {
                name: np.asarray(values, dtype=object)
                if np.asarray(values).dtype == object
                else np.asarray(values)
                for name, values in source.get_row_data(rows).items()
            }
        else:
            image_columns = {
                name
                for name in source.column_names
                if name == "pixels" or name.startswith("pixels_")
            }
            # Keep only compact numeric episode metadata in the bounded cache.
            # In particular, never copy the decoded frame store into an episode
            # because that would change both the 6-GiB budget and sampling flow.
            row_data = {
                name: np.asarray(source.get_col_data(name))[first_row:last_row]
                for name in source.column_names
                if name not in image_columns
            }

        episodes: list[CachedEpisode] = []
        for episode in episode_ids:
            episode = int(episode)
            begin = int(source.offsets[episode]) - first_row
            end = begin + int(source.lengths[episode])
            # Copies detach each episode from the temporary multi-episode block,
            # so evicting an episode really releases its share of the cache.
            columns = {
                name: values[begin:end].copy() for name, values in row_data.items()
            }
            starts = self._episode_starts[episode].copy()
            byte_size = int(starts.nbytes) + sum(
                _payload_nbytes(values) for values in columns.values()
            )
            if self._dataset.decoded_frame_store is not None:
                # Preserve the original compressed-JPEG cache accounting even
                # though the JPEG objects themselves are no longer resident.
                # This keeps admission/refill and therefore sampling identical.
                byte_size += int(
                    self._dataset._episode_jpeg_payload_bytes[episode]
                ) + int(source.lengths[episode]) * np.dtype(object).itemsize
            episodes.append(
                CachedEpisode(
                    episode=episode,
                    starts=starts,
                    columns=columns,
                    byte_size=byte_size,
                )
            )
        return episodes

    def _producer(
        self,
        output: queue.Queue,
        stop: threading.Event,
        budget: threading.Condition,
        resident_bytes: list[int],
        episode_blocks: Sequence[Sequence[int]],
    ) -> None:
        try:
            for block_ids in episode_blocks:
                if stop.is_set():
                    return
                block = self._read_episode_block(block_ids)
                block_bytes = sum(episode.byte_size for episode in block)
                if block_bytes > self._cache_bytes:
                    raise MemoryError(
                        "one episode read block exceeds episode_cache_bytes"
                    )
                with budget:
                    while resident_bytes[0] + block_bytes > self._cache_bytes:
                        if stop.is_set():
                            return
                        budget.wait(timeout=0.2)
                    if stop.is_set():
                        return
                    resident_bytes[0] += block_bytes
                while not stop.is_set():
                    try:
                        output.put(block, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            while not stop.is_set():
                try:
                    output.put(self._END, timeout=0.2)
                    return
                except queue.Full:
                    continue
        except BaseException as error:
            while not stop.is_set():
                try:
                    output.put(error, timeout=0.2)
                    return
                except queue.Full:
                    continue

    def _materialize_batch(
        self, selections: Sequence[tuple[CachedEpisode, int]]
    ) -> dict[str, Any]:
        import torch

        source = self._dataset.dataset
        frameskip = int(source.frameskip)
        num_steps = int(source.num_steps)
        span = int(source.span)
        image_columns = {
            name
            for name in source.column_names
            if name == "pixels" or name.startswith("pixels_")
        }
        frame_positions = [
            tuple(start + step * frameskip for step in range(num_steps))
            for _, start in selections
        ]
        batch: dict[str, Any] = {}
        for column in source.column_names:
            if column in image_columns:
                if self._dataset.decoded_frame_store is None:
                    blobs = [
                        episode.columns[column][position]
                        for (episode, _), positions in zip(
                            selections, frame_positions
                        )
                        for position in positions
                    ]
                    decoded = _decode_images(blobs)
                else:
                    global_rows = [
                        int(source.offsets[episode.episode]) + int(position)
                        for (episode, _), positions in zip(
                            selections, frame_positions
                        )
                        for position in positions
                    ]
                    decoded = self._dataset._take_decoded_frames(global_rows)
                batch[column] = decoded.reshape(
                    len(selections), num_steps, *decoded.shape[1:]
                )
            elif column == "action":
                values = np.stack(
                    [
                        np.asarray(episode.columns[column][start : start + span])
                        for episode, start in selections
                    ]
                )
                batch[column] = _numeric_tensor(values)
            else:
                values = np.stack(
                    [
                        np.asarray(episode.columns[column])[list(positions)]
                        for (episode, _), positions in zip(
                            selections, frame_positions
                        )
                    ]
                )
                batch[column] = _numeric_tensor(values)

        if source.transform:
            batch = source.transform(batch)
        batch["action"] = batch["action"].reshape(
            len(selections), num_steps, -1
        )
        batch["_tdwm_episode_id"] = torch.tensor(
            [episode.episode for episode, _ in selections], dtype=torch.int32
        )
        return batch

    def __iter__(self):
        from torch.utils.data import get_worker_info

        if get_worker_info() is not None:
            raise RuntimeError(
                "EpisodeStreamingBatchDataset requires DataLoader(num_workers=0)."
            )

        rng = np.random.default_rng(np.random.SeedSequence([self._seed, self._epoch]))
        episode_blocks = tuple(
            self._episode_ids[offset : offset + self._read_episodes]
            for offset in range(0, len(self._episode_ids), self._read_episodes)
        )
        if self._epoch and len(episode_blocks) > 1:
            rotation_rng = np.random.default_rng(
                np.random.SeedSequence([self._seed, self._epoch, 0xE9150DE])
            )
            rotation = int(rotation_rng.integers(1, len(episode_blocks)))
            episode_blocks = episode_blocks[rotation:] + episode_blocks[:rotation]
        blocks: queue.Queue = queue.Queue(maxsize=self._prefetch_blocks)
        stop = threading.Event()
        budget = threading.Condition()
        resident_bytes = [0]
        producer = threading.Thread(
            target=self._producer,
            args=(blocks, stop, budget, resident_bytes, episode_blocks),
            name="tdwm-episode-prefetch",
            daemon=True,
        )
        producer.start()

        active: dict[int, CachedEpisode] = {}
        pending: deque[CachedEpisode] = deque()
        cached_bytes = 0
        source_finished = False

        def receive_block() -> None:
            nonlocal source_finished
            item = blocks.get()
            if item is self._END:
                source_finished = True
            elif isinstance(item, BaseException):
                raise item
            else:
                pending.extend(item)

        def refill() -> None:
            nonlocal cached_bytes
            while len(active) < self._active_episodes:
                if not pending:
                    if source_finished:
                        return
                    receive_block()
                    if source_finished and not pending:
                        return
                episode = pending[0]
                if (
                    active
                    and cached_bytes + episode.byte_size > self._cache_bytes
                ):
                    return
                pending.popleft()
                rng.shuffle(episode.starts)
                active[episode.episode] = episode
                cached_bytes += episode.byte_size

        emitted = 0
        try:
            refill()
            if len(active) < self._min_unique_episodes:
                raise MemoryError(
                    "episode cache budget cannot hold min_unique_episodes; "
                    f"loaded {len(active)} episodes using {cached_bytes} bytes"
                )

            while emitted < len(self):
                refill()
                eligible = [
                    episode for episode in active.values() if episode.remaining > 0
                ]
                tail_fallback = False
                if len(eligible) < self._min_unique_episodes:
                    remaining = sum(episode.remaining for episode in eligible)
                    if source_finished and not pending and remaining < self._batch_size:
                        break
                    if source_finished and not pending:
                        # The final resident episodes may not cover a complete
                        # batch. Keep all full batches instead of discarding a
                        # sizeable tail; only this tail may repeat episodes.
                        tail_fallback = True
                    else:
                        raise RuntimeError(
                            "episode streaming lost batch diversity before "
                            f"exhausting the epoch: {len(eligible)} active "
                            f"episodes, {remaining} remaining clips"
                        )

                shuffled = [eligible[index] for index in rng.permutation(len(eligible))]
                shuffled.sort(key=lambda episode: episode.remaining, reverse=True)
                selected_episodes = (
                    shuffled if tail_fallback else shuffled[: self._batch_size]
                )
                selections = []
                for episode in selected_episodes:
                    start = int(episode.starts[episode.cursor])
                    episode.cursor += 1
                    selections.append((episode, start))

                while len(selections) < self._batch_size:
                    candidates = [
                        episode
                        for episode in active.values()
                        if episode.remaining > 0
                    ]
                    if not candidates:
                        raise RuntimeError(
                            "episode streaming could not fill a tail batch."
                        )
                    shuffled_candidates = [
                        candidates[index]
                        for index in rng.permutation(len(candidates))
                    ]
                    shuffled_candidates.sort(
                        key=lambda episode: episode.remaining, reverse=True
                    )
                    episode = shuffled_candidates[0]
                    start = int(episode.starts[episode.cursor])
                    episode.cursor += 1
                    selections.append((episode, start))

                unique_episodes = len({episode.episode for episode, _ in selections})
                if not tail_fallback and unique_episodes < self._min_unique_episodes:
                    raise RuntimeError(
                        "episode streaming emitted an insufficiently diverse batch."
                    )

                exhausted = [
                    episode_id
                    for episode_id, episode in active.items()
                    if episode.remaining == 0
                ]
                for episode_id in exhausted:
                    cached_bytes -= active[episode_id].byte_size
                    with budget:
                        resident_bytes[0] -= active[episode_id].byte_size
                        budget.notify_all()
                    del active[episode_id]

                batch = self._materialize_batch(selections)
                with budget:
                    batch["_tdwm_cache_bytes"] = resident_bytes[0]
                yield batch
                emitted += 1
        finally:
            stop.set()
            with budget:
                budget.notify_all()
            producer.join(timeout=2.0)


class PairedEpisodeStreamingBatchDataset(IterableDataset):
    """Pair independently sampled short and long Cube batches per update.

    Each child already emits a fully collated batch. Keeping the streams
    independent lets the LeWM loss retain a large, trajectory-diverse short
    batch while a smaller long-sequence batch supplies the goal-tail target.
    """

    def __init__(
        self,
        world: EpisodeStreamingBatchDataset,
        tail: EpisodeStreamingBatchDataset,
    ) -> None:
        if not isinstance(world, EpisodeStreamingBatchDataset) or not isinstance(
            tail, EpisodeStreamingBatchDataset
        ):
            raise TypeError("Paired streams must be EpisodeStreamingBatchDataset.")
        self.world = world
        self.tail = tail

    def __len__(self) -> int:
        return min(len(self.world), len(self.tail))

    def set_epoch(self, epoch: int) -> None:
        self.world.set_epoch(epoch)
        self.tail.set_epoch(epoch)

    def __iter__(self):
        world_iterator = iter(self.world)
        tail_iterator = iter(self.tail)
        try:
            for _ in range(len(self)):
                yield {
                    "world": next(world_iterator),
                    "tail": next(tail_iterator),
                }
        finally:
            for iterator in (world_iterator, tail_iterator):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
