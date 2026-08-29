"""Immutable row-aligned cache for a frozen LeWM image encoder.

The store is deliberately indexed by *global dataset row*, not by clip.  A
frame shared by many training clips is therefore encoded exactly once.  The
three arrays are published together with a provenance manifest only after
every global row has been written and audited:

``latents.npy``
    float32 ``[total_rows, embed_dim]`` frozen LeWM embeddings.
``action_blocks.npy``
    float32 ``[total_rows, 25]`` normalized five-slot Cube action blocks.
    Slots beyond an episode boundary are NaN, and source terminal NaNs are
    preserved.
``episode_ids.npy``
    int64 ``[total_rows]`` episode identity for strict clip-boundary checks.

The reader validates exact file SHA-256 digests before memory mapping them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

FORMAT_NAME = "tdwm-frozen-lewm-latent-store"
SCHEMA_VERSION = 1
ROW_MAPPING = "array row i corresponds exactly to source dataset global row i"
ACTION_BLOCK_SLOTS = 5
CUBE_ACTION_DIM = 5
CUBE_ACTION_BLOCK_DIM = ACTION_BLOCK_SLOTS * CUBE_ACTION_DIM
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def file_sha256(path: str | Path) -> str:
    """Hash one regular file while rejecting an in-flight mutation."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"File changed while hashing: {resolved}")
    return digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").numpy()
    return np.asarray(value)


def _integer_vector(value: Any, *, label: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if array.dtype.kind not in ("i", "u"):
        raise TypeError(f"{label} must contain integers.")
    if array.dtype.kind == "u" and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{label} exceeds int64.")
    return np.asarray(array, dtype=np.int64)


def validate_episode_ids(episode_ids: Any, *, total_rows: int) -> np.ndarray:
    """Return canonical row-aligned episode IDs after structural validation."""

    episodes = _integer_vector(episode_ids, label="episode_ids")
    if episodes.shape != (int(total_rows),):
        raise ValueError("episode_ids must contain exactly one ID per global row.")
    if episodes.size == 0:
        raise ValueError("A frozen latent store cannot be empty.")
    if np.any(episodes < 0):
        raise ValueError("episode_ids cannot be negative.")
    if np.any(episodes[1:] < episodes[:-1]):
        raise ValueError("episode_ids must form nondecreasing contiguous segments.")
    unique = np.unique(episodes)
    if not np.array_equal(unique, np.arange(unique.size, dtype=np.int64)):
        raise ValueError("episode_ids must be dense and start at zero.")
    return episodes


def normalize_actions(
    raw_actions: Any,
    *,
    mean: Any,
    scale: Any,
    expected_action_dim: int = CUBE_ACTION_DIM,
) -> np.ndarray:
    """Apply the released column normalization while preserving source NaNs."""

    actions = _as_numpy(raw_actions)
    if actions.ndim != 2 or actions.shape[1] != int(expected_action_dim):
        raise ValueError(
            "raw_actions must have shape [total_rows, expected_action_dim]."
        )
    actions = np.asarray(actions, dtype=np.float32)
    action_mean = np.asarray(mean, dtype=np.float32)
    action_scale = np.asarray(scale, dtype=np.float32)
    if action_mean.shape != (expected_action_dim,) or action_scale.shape != (
        expected_action_dim,
    ):
        raise ValueError("Action normalization mean/scale have the wrong shape.")
    if not np.all(np.isfinite(action_mean)) or not np.all(np.isfinite(action_scale)):
        raise ValueError("Action normalization statistics must be finite.")
    if np.any(action_scale <= 0):
        raise ValueError("Action normalization scale must be strictly positive.")
    if np.any(np.isinf(actions)):
        raise ValueError("Raw actions cannot contain infinities.")
    normalized = (actions - action_mean) / action_scale
    if np.any(np.isinf(normalized)):
        raise ValueError("Normalized actions cannot contain infinities.")
    return np.asarray(normalized, dtype=np.float32)


def action_blocks_for_rows(
    global_rows: Any,
    *,
    normalized_actions: Any,
    episode_ids: Any,
    block_slots: int = ACTION_BLOCK_SLOTS,
    validate_episode_layout: bool = True,
    validate_action_array: bool = True,
) -> np.ndarray:
    """Gather row-started action blocks without crossing episode boundaries."""

    rows = _integer_vector(global_rows, label="global_rows")
    actions = _as_numpy(normalized_actions)
    if actions.ndim != 2:
        raise ValueError("normalized_actions must be a two-dimensional array.")
    if actions.dtype.kind != "f":
        raise TypeError("normalized_actions must be floating point.")
    actions = np.asarray(actions, dtype=np.float32)
    if validate_episode_layout:
        episodes = validate_episode_ids(episode_ids, total_rows=actions.shape[0])
    else:
        episodes = _integer_vector(episode_ids, label="episode_ids")
        if episodes.shape != (actions.shape[0],):
            raise ValueError("episode_ids must contain one ID per action row.")
    if block_slots <= 0:
        raise ValueError("block_slots must be positive.")
    if rows.size and (np.any(rows < 0) or np.any(rows >= actions.shape[0])):
        raise IndexError("global_rows contains an out-of-range dataset row.")
    if validate_action_array and np.any(np.isinf(actions)):
        raise ValueError("normalized_actions cannot contain infinities.")

    action_dim = int(actions.shape[1])
    blocks = np.full(
        (rows.size, int(block_slots), action_dim),
        np.nan,
        dtype=np.float32,
    )
    anchor_episodes = episodes[rows]
    for slot in range(int(block_slots)):
        source_rows = rows + slot
        in_bounds = source_rows < actions.shape[0]
        valid = np.zeros(rows.size, dtype=np.bool_)
        if np.any(in_bounds):
            valid[in_bounds] = (
                episodes[source_rows[in_bounds]] == anchor_episodes[in_bounds]
            )
        blocks[valid, slot] = actions[source_rows[valid]]
    return blocks.reshape(rows.size, int(block_slots) * action_dim)


@dataclass(frozen=True)
class FrozenLatentStoreSpec:
    """Protocol identity sealed into one frozen-cache artifact."""

    total_rows: int
    embed_dim: int
    frame_skip: int
    history_frames: int
    action_dim: int
    pretrained_checkpoint_sha256: str
    dataset_source_sha256: str
    column_normalization_sha256: str
    git_revision: str

    def __post_init__(self) -> None:
        if self.total_rows <= 0 or self.embed_dim <= 0:
            raise ValueError("total_rows and embed_dim must be positive.")
        if self.frame_skip != ACTION_BLOCK_SLOTS:
            raise ValueError("The shared Cube cache requires frame_skip=5.")
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive.")
        if self.action_dim != CUBE_ACTION_DIM:
            raise ValueError("The shared Cube cache requires action_dim=5.")
        _validate_sha256(
            self.pretrained_checkpoint_sha256,
            "pretrained_checkpoint_sha256",
        )
        _validate_sha256(self.dataset_source_sha256, "dataset_source_sha256")
        _validate_sha256(
            self.column_normalization_sha256,
            "column_normalization_sha256",
        )
        if (
            not isinstance(self.git_revision, str)
            or _GIT_REVISION_RE.fullmatch(self.git_revision) is None
        ):
            raise ValueError("git_revision must be a full lowercase Git revision.")

    @property
    def action_block_dim(self) -> int:
        return self.frame_skip * self.action_dim


@dataclass(frozen=True)
class EncodedRowBatch:
    """One already-encoded, globally indexed row batch."""

    global_rows: Any
    latents: Any


@dataclass(frozen=True)
class FrozenClipBatch:
    """Cache result with tensors matching the online clip interface."""

    latents: torch.Tensor
    actions: torch.Tensor
    metadata: dict[str, Any]


def iter_encoded_global_rows_once(
    *,
    total_rows: int,
    batch_size: int,
    encode_rows: Callable[[np.ndarray], Any],
) -> Iterator[EncodedRowBatch]:
    """Invoke an extractor once per disjoint sequential global-row batch.

    This is the only traversal used by the CLI.  It never enumerates sequence
    clips, so overlapping clips cannot cause repeated image encoding.
    """

    if total_rows <= 0 or batch_size <= 0:
        raise ValueError("total_rows and batch_size must be positive.")
    for start in range(0, int(total_rows), int(batch_size)):
        stop = min(start + int(batch_size), int(total_rows))
        rows = np.arange(start, stop, dtype=np.int64)
        latents = _as_numpy(encode_rows(rows))
        if latents.ndim != 2 or latents.shape[0] != rows.size:
            raise ValueError(
                "encode_rows must return [len(global_rows), embed_dim] latents."
            )
        yield EncodedRowBatch(global_rows=rows, latents=latents)


class FrozenLatentStoreBuilder:
    """Write all arrays in a private staging directory, then publish atomically."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        spec: FrozenLatentStoreSpec,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite frozen latent store: {self.output_dir}"
            )
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}.staging-",
                dir=self.output_dir.parent,
            )
        )
        self.spec = spec
        self.source_metadata = dict(source_metadata or {})
        self._written = np.zeros(spec.total_rows, dtype=np.bool_)
        self._closed = False
        self._published = False
        self._latents = np.lib.format.open_memmap(
            self.staging_dir / "latents.npy",
            mode="w+",
            dtype=np.float32,
            shape=(spec.total_rows, spec.embed_dim),
        )
        self._actions = np.lib.format.open_memmap(
            self.staging_dir / "action_blocks.npy",
            mode="w+",
            dtype=np.float32,
            shape=(spec.total_rows, spec.action_block_dim),
        )
        self._episode_ids = np.lib.format.open_memmap(
            self.staging_dir / "episode_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(spec.total_rows,),
        )

    def __enter__(self) -> "FrozenLatentStoreBuilder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None or not self._published:
            self.abort()

    def add_rows(
        self,
        *,
        global_rows: Any,
        latents: Any,
        action_blocks: Any,
        episode_ids: Any,
    ) -> None:
        """Write a unique encoded row batch into the private artifact."""

        if self._closed:
            raise RuntimeError("The frozen latent store builder is closed.")
        rows = _integer_vector(global_rows, label="global_rows")
        if rows.size == 0:
            raise ValueError("Cannot add an empty encoded row batch.")
        if np.unique(rows).size != rows.size:
            raise ValueError("An encoded row batch contains duplicate global rows.")
        if np.any(rows < 0) or np.any(rows >= self.spec.total_rows):
            raise IndexError("An encoded row is outside the frozen store.")
        if np.any(self._written[rows]):
            repeated = rows[self._written[rows]][:8].tolist()
            raise ValueError(f"Global rows were encoded more than once: {repeated}")

        latent_array = np.asarray(_as_numpy(latents), dtype=np.float32)
        action_array = np.asarray(_as_numpy(action_blocks), dtype=np.float32)
        episode_array = _integer_vector(episode_ids, label="episode_ids batch")
        if latent_array.shape != (rows.size, self.spec.embed_dim):
            raise ValueError("Encoded latent batch has the wrong shape.")
        if action_array.shape != (rows.size, self.spec.action_block_dim):
            raise ValueError("Encoded action-block batch has the wrong shape.")
        if episode_array.shape != (rows.size,):
            raise ValueError("Encoded episode-ID batch has the wrong shape.")
        if not np.all(np.isfinite(latent_array)):
            raise ValueError("Frozen latents must all be finite.")
        if np.any(np.isinf(action_array)):
            raise ValueError("Frozen action blocks cannot contain infinities.")
        if np.any(episode_array < 0):
            raise ValueError("Episode IDs cannot be negative.")

        self._latents[rows] = latent_array
        self._actions[rows] = action_array
        self._episode_ids[rows] = episode_array
        self._written[rows] = True

    @staticmethod
    def _flush_and_close(array: np.memmap) -> None:
        array.flush()
        mapped = getattr(array, "_mmap", None)
        if mapped is not None:
            mapped.close()

    @staticmethod
    def _file_entry(
        path: Path, *, shape: tuple[int, ...], dtype: np.dtype
    ) -> dict[str, Any]:
        return {
            "path": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "dtype": np.dtype(dtype).name,
            "shape": [int(value) for value in shape],
        }

    def finalize(self) -> dict[str, Any]:
        """Audit complete row coverage and atomically publish the directory."""

        if self._closed:
            raise RuntimeError("The frozen latent store builder is closed.")
        if not np.all(self._written):
            missing = np.flatnonzero(~self._written)[:8].tolist()
            raise RuntimeError(f"Frozen latent store is missing rows: {missing}")

        # Validate the complete episode vector and boundary padding before close.
        episodes = validate_episode_ids(
            np.asarray(self._episode_ids), total_rows=self.spec.total_rows
        )
        reshaped_actions = self._actions.reshape(
            self.spec.total_rows, self.spec.frame_skip, self.spec.action_dim
        )
        all_rows = np.arange(self.spec.total_rows, dtype=np.int64)
        for slot in range(self.spec.frame_skip):
            source = all_rows + slot
            valid = source < self.spec.total_rows
            within_episode = np.zeros(self.spec.total_rows, dtype=np.bool_)
            within_episode[valid] = episodes[source[valid]] == episodes[valid]
            if np.any(~np.isnan(reshaped_actions[~within_episode, slot])):
                raise ValueError(
                    "Action blocks must use NaN padding across episode boundaries."
                )

        latent_shape = tuple(self._latents.shape)
        action_shape = tuple(self._actions.shape)
        episode_shape = tuple(self._episode_ids.shape)
        latent_dtype = self._latents.dtype
        action_dtype = self._actions.dtype
        episode_dtype = self._episode_ids.dtype
        self._flush_and_close(self._latents)
        self._flush_and_close(self._actions)
        self._flush_and_close(self._episode_ids)
        self._closed = True

        latent_path = self.staging_dir / "latents.npy"
        action_path = self.staging_dir / "action_blocks.npy"
        episode_path = self.staging_dir / "episode_ids.npy"
        files = {
            "latents": self._file_entry(
                latent_path, shape=latent_shape, dtype=latent_dtype
            ),
            "action_blocks": self._file_entry(
                action_path, shape=action_shape, dtype=action_dtype
            ),
            "episode_ids": self._file_entry(
                episode_path, shape=episode_shape, dtype=episode_dtype
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_NAME,
            "row_mapping": ROW_MAPPING,
            "total_rows": self.spec.total_rows,
            "embed_dim": self.spec.embed_dim,
            "frame_skip": self.spec.frame_skip,
            "history_frames": self.spec.history_frames,
            "action_dim": self.spec.action_dim,
            "action_block_slots": self.spec.frame_skip,
            "action_block_dim": self.spec.action_block_dim,
            "pretrained_checkpoint_sha256": (self.spec.pretrained_checkpoint_sha256),
            "dataset_source_sha256": self.spec.dataset_source_sha256,
            "column_normalization_sha256": (self.spec.column_normalization_sha256),
            "git_revision": self.spec.git_revision,
            "terminal_nan_policy": "preserve_source_and_pad_episode_tail_with_nan",
            "coverage": {
                "key": "global_dataset_row",
                "expected_rows": self.spec.total_rows,
                "written_rows": int(self._written.sum()),
                "each_global_row_encoded_exactly_once": True,
                "clip_enumeration_used_for_encoding": False,
            },
            "validation": {
                "latents_all_finite": True,
                "action_blocks_have_no_infinities": True,
                "episode_boundary_padding_is_nan": True,
            },
            "files": files,
            "source_metadata": self.source_metadata,
        }
        manifest_path = self.staging_dir / "manifest.json"
        temporary_manifest = self.staging_dir / "manifest.json.tmp"
        with temporary_manifest.open("w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_manifest.replace(manifest_path)
        directory_fd = os.open(self.staging_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(self.staging_dir, self.output_dir)
        parent_fd = os.open(self.output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self._published = True
        return manifest

    def abort(self) -> None:
        """Remove only this builder's unpublished private staging directory."""

        if self._published:
            return
        if not self._closed:
            for array in (self._latents, self._actions, self._episode_ids):
                try:
                    self._flush_and_close(array)
                except (OSError, ValueError):
                    pass
            self._closed = True
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)


def build_frozen_latent_store(
    output_dir: str | Path,
    *,
    spec: FrozenLatentStoreSpec,
    encoded_batches: Iterable[EncodedRowBatch | tuple[Any, Any]],
    normalized_actions: Any,
    episode_ids: Any,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete store from non-overlapping already-encoded row batches."""

    actions = np.asarray(_as_numpy(normalized_actions), dtype=np.float32)
    if actions.shape != (spec.total_rows, spec.action_dim):
        raise ValueError("normalized_actions has the wrong global shape.")
    if np.any(np.isinf(actions)):
        raise ValueError("normalized_actions cannot contain infinities.")
    episodes = validate_episode_ids(episode_ids, total_rows=spec.total_rows)
    with FrozenLatentStoreBuilder(
        output_dir, spec=spec, source_metadata=source_metadata
    ) as builder:
        for raw_batch in encoded_batches:
            if isinstance(raw_batch, EncodedRowBatch):
                rows, latents = raw_batch.global_rows, raw_batch.latents
            else:
                try:
                    rows, latents = raw_batch
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "Each encoded batch must be EncodedRowBatch or (rows, latents)."
                    ) from error
            canonical_rows = _integer_vector(rows, label="global_rows")
            blocks = action_blocks_for_rows(
                canonical_rows,
                normalized_actions=actions,
                episode_ids=episodes,
                block_slots=spec.frame_skip,
                validate_episode_layout=False,
                validate_action_array=False,
            )
            builder.add_rows(
                global_rows=canonical_rows,
                latents=latents,
                action_blocks=blocks,
                episode_ids=episodes[canonical_rows],
            )
        return builder.finalize()


class FrozenLatentStore:
    """Strict, exact-SHA-validated memory-mapped frozen LeWM cache."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_dataset_source_sha256: str | None = None,
        expected_column_normalization_sha256: str | None = None,
        expected_frame_skip: int | None = None,
        expected_history_frames: int | None = None,
        expected_embed_dim: int | None = None,
        expected_action_dim: int | None = None,
    ) -> None:
        root = Path(artifact_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("Frozen latent manifest must contain a JSON object.")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported frozen latent store schema.")
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError("The frozen latent store format is incompatible.")
        if manifest.get("row_mapping") != ROW_MAPPING:
            raise ValueError("The frozen latent store row mapping is incompatible.")

        expected_checkpoint_sha256 = _validate_sha256(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
        bindings = {
            "pretrained_checkpoint_sha256": expected_checkpoint_sha256,
            "dataset_source_sha256": expected_dataset_source_sha256,
            "column_normalization_sha256": (expected_column_normalization_sha256),
            "frame_skip": expected_frame_skip,
            "history_frames": expected_history_frames,
            "embed_dim": expected_embed_dim,
            "action_dim": expected_action_dim,
        }
        for field, expected in bindings.items():
            if expected is not None and manifest.get(field) != expected:
                raise ValueError(
                    f"Frozen latent store {field} differs from the active protocol."
                )
        for digest_field in (
            "pretrained_checkpoint_sha256",
            "dataset_source_sha256",
            "column_normalization_sha256",
        ):
            _validate_sha256(manifest.get(digest_field), digest_field)
        revision = manifest.get("git_revision")
        if (
            not isinstance(revision, str)
            or _GIT_REVISION_RE.fullmatch(revision) is None
        ):
            raise ValueError("Frozen latent store git_revision is invalid.")

        total_rows = manifest.get("total_rows")
        embed_dim = manifest.get("embed_dim")
        frame_skip = manifest.get("frame_skip")
        history_frames = manifest.get("history_frames")
        action_dim = manifest.get("action_dim")
        action_block_dim = manifest.get("action_block_dim")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                total_rows,
                embed_dim,
                frame_skip,
                history_frames,
                action_dim,
                action_block_dim,
            )
        ):
            raise ValueError("Frozen latent dimensions must be positive integers.")
        if frame_skip != ACTION_BLOCK_SLOTS or action_dim != CUBE_ACTION_DIM:
            raise ValueError("The frozen latent store is not a Cube 5x5 cache.")
        if action_block_dim != frame_skip * action_dim:
            raise ValueError("Frozen action_block_dim is inconsistent.")
        if manifest.get("action_block_slots") != frame_skip:
            raise ValueError("Frozen action_block_slots is inconsistent.")
        if manifest.get("terminal_nan_policy") != (
            "preserve_source_and_pad_episode_tail_with_nan"
        ):
            raise ValueError("Frozen action terminal-NaN policy is incompatible.")
        coverage = manifest.get("coverage", {})
        if coverage.get("each_global_row_encoded_exactly_once") is not True:
            raise ValueError("The cache does not prove exactly-once row encoding.")
        if coverage.get("clip_enumeration_used_for_encoding") is not False:
            raise ValueError("The cache was not built by a global-row traversal.")
        if coverage.get("written_rows") != total_rows:
            raise ValueError("Frozen cache row coverage is incomplete.")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Frozen latent manifest is missing files.")
        identities: dict[str, tuple[int, int, int, int]] = {}

        def load_array(
            name: str, *, dtype: np.dtype, shape: tuple[int, ...]
        ) -> np.memmap:
            entry = files.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"files.{name} must describe one array.")
            path = (root / entry["path"]).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"files.{name} escapes the artifact directory."
                ) from error
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = _validate_sha256(
                entry.get("sha256"), f"files.{name}.sha256"
            )
            if file_sha256(path) != expected_hash:
                raise ValueError(f"files.{name} failed exact SHA-256 validation.")
            if entry.get("size_bytes") != path.stat().st_size:
                raise ValueError(f"files.{name}.size_bytes is incorrect.")
            if entry.get("dtype") != np.dtype(dtype).name:
                raise ValueError(f"files.{name}.dtype is incompatible.")
            if entry.get("shape") != list(shape):
                raise ValueError(f"files.{name}.shape is incompatible.")
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if not isinstance(loaded, np.memmap):
                raise TypeError(f"files.{name} was not memory mapped.")
            if loaded.dtype != np.dtype(dtype) or loaded.shape != shape:
                raise ValueError(f"files.{name} array metadata is incompatible.")
            identities[name] = _stat_identity(path)
            return loaded

        latents = load_array("latents", dtype=np.float32, shape=(total_rows, embed_dim))
        actions = load_array(
            "action_blocks",
            dtype=np.float32,
            shape=(total_rows, action_block_dim),
        )
        episodes = load_array("episode_ids", dtype=np.int64, shape=(total_rows,))
        validate_episode_ids(episodes, total_rows=total_rows)
        if not np.all(np.isfinite(latents)):
            raise ValueError("Frozen latent file contains non-finite values.")
        if np.any(np.isinf(actions)):
            raise ValueError("Frozen action-block file contains infinities.")
        reshaped_actions = actions.reshape(total_rows, frame_skip, action_dim)
        all_rows = np.arange(total_rows, dtype=np.int64)
        for slot in range(frame_skip):
            source = all_rows + slot
            valid = source < total_rows
            within_episode = np.zeros(total_rows, dtype=np.bool_)
            within_episode[valid] = episodes[source[valid]] == episodes[valid]
            if np.any(~np.isnan(reshaped_actions[~within_episode, slot])):
                raise ValueError("Frozen action blocks cross an episode boundary.")

        self.root = root
        self.manifest_path = manifest_path
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = manifest
        self.latents = latents
        self.actions = actions
        self.episode_ids = episodes
        self.total_rows = int(total_rows)
        self.embed_dim = int(embed_dim)
        self.frame_skip = int(frame_skip)
        self.history_frames = int(history_frames)
        self.action_dim = int(action_dim)
        self.action_block_dim = int(action_block_dim)
        self._paths = {
            name: (root / files[name]["path"]).resolve() for name in identities
        }
        self._identities = identities
        self._manifest_identity = _stat_identity(manifest_path)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only mmap metadata for spawn-based DataLoader workers.

        NumPy's default memmap pickle embeds the complete array payload.  The
        Lance runtime forces multiprocessing ``spawn`` on Linux, so accepting
        that default would privately copy the multi-GiB cache into every
        worker.  The parent has already performed exact-SHA validation; a
        worker reopens the same immutable files after checking the manifest
        hash and inode identities handed off by the parent.
        """

        self._assert_immutable()
        state = dict(self.__dict__)
        state.pop("latents")
        state.pop("actions")
        state.pop("episode_ids")
        state["_spawn_handoff_version"] = 1
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_spawn_handoff_version", None) != 1:
            raise ValueError("Unsupported frozen-store worker handoff.")
        self.__dict__.update(state)
        if _stat_identity(self.manifest_path) != self._manifest_identity:
            raise RuntimeError(
                "Frozen latent manifest changed before worker mmap reopen."
            )
        manifest_bytes = self.manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != self.manifest_sha256:
            raise RuntimeError(
                "Frozen latent manifest hash changed before worker mmap reopen."
            )
        if json.loads(manifest_bytes) != self.manifest:
            raise RuntimeError("Frozen latent worker received different metadata.")

        arrays: dict[str, np.memmap] = {}
        expected = {
            "latents": (np.float32, (self.total_rows, self.embed_dim)),
            "action_blocks": (
                np.float32,
                (self.total_rows, self.action_block_dim),
            ),
            "episode_ids": (np.int64, (self.total_rows,)),
        }
        for name, (dtype, shape) in expected.items():
            path = self._paths[name]
            if _stat_identity(path) != self._identities[name]:
                raise RuntimeError(
                    f"Frozen latent store changed before worker mmap reopen: {name}"
                )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                not isinstance(array, np.memmap)
                or array.dtype != np.dtype(dtype)
                or array.shape != shape
            ):
                raise ValueError(
                    f"Frozen latent worker mmap metadata differs for {name}."
                )
            arrays[name] = array
        self.latents = arrays["latents"]
        self.actions = arrays["action_blocks"]
        self.episode_ids = arrays["episode_ids"]
        self._assert_immutable()

    def _assert_immutable(self) -> None:
        if _stat_identity(self.manifest_path) != self._manifest_identity:
            raise RuntimeError("Frozen latent manifest changed after validation.")
        for name, expected in self._identities.items():
            if _stat_identity(self._paths[name]) != expected:
                raise RuntimeError(
                    f"Frozen latent store file changed after validation: {name}"
                )

    def gather_clips(
        self,
        global_starts: Sequence[int] | np.ndarray | torch.Tensor,
        num_steps: int,
        frame_skip: int,
        *,
        device: str | torch.device = "cpu",
    ) -> FrozenClipBatch:
        """Gather strict, episode-contained clips from the row-aligned mmap.

        The requested dense action span is ``num_steps * frame_skip`` rows.
        It must fit completely inside the anchor episode; no clip can silently
        borrow action rows or observations from the next episode.
        """

        self._assert_immutable()
        starts = _integer_vector(global_starts, label="global_starts")
        if starts.size == 0:
            raise ValueError("global_starts cannot be empty.")
        if num_steps <= 0 or frame_skip <= 0:
            raise ValueError("num_steps and frame_skip must be positive.")
        if int(frame_skip) != self.frame_skip:
            raise ValueError("Requested frame_skip differs from the frozen cache.")
        span = int(num_steps) * int(frame_skip)
        if np.any(starts < 0) or np.any(starts > self.total_rows - span):
            raise IndexError("A requested clip exceeds the global row bounds.")
        final_dense_rows = starts + span - 1
        anchor_episodes = np.asarray(self.episode_ids[starts], dtype=np.int64)
        final_episodes = np.asarray(self.episode_ids[final_dense_rows], dtype=np.int64)
        if not np.array_equal(anchor_episodes, final_episodes):
            bad = starts[anchor_episodes != final_episodes][:8].tolist()
            raise ValueError(f"Requested clips cross an episode boundary: {bad}")

        offsets = np.arange(int(num_steps), dtype=np.int64) * int(frame_skip)
        rows = starts[:, None] + offsets[None, :]
        frame_episodes = np.asarray(self.episode_ids[rows], dtype=np.int64)
        if np.any(frame_episodes != anchor_episodes[:, None]):
            raise ValueError("A sampled latent row crosses an episode boundary.")
        latent_copy = np.array(self.latents[rows], dtype=np.float32, copy=True)
        action_copy = np.array(self.actions[rows], dtype=np.float32, copy=True)
        action_finite = np.isfinite(action_copy).all(axis=-1)
        target_device = torch.device(device)
        latent_tensor = torch.from_numpy(latent_copy).to(
            device=target_device, non_blocking=True
        )
        action_tensor = torch.from_numpy(action_copy).to(
            device=target_device, non_blocking=True
        )
        metadata: dict[str, Any] = {
            "global_starts": torch.from_numpy(np.array(starts, copy=True)),
            "global_rows": torch.from_numpy(np.array(rows, copy=True)),
            "episode_ids": torch.from_numpy(np.array(anchor_episodes, copy=True)),
            "frame_episode_ids": torch.from_numpy(np.array(frame_episodes, copy=True)),
            "action_finite": torch.from_numpy(action_finite),
            "num_steps": int(num_steps),
            "frame_skip": int(frame_skip),
            "manifest_sha256": self.manifest_sha256,
        }
        if target_device.type != "cpu":
            for key in (
                "global_starts",
                "global_rows",
                "episode_ids",
                "frame_episode_ids",
                "action_finite",
            ):
                metadata[key] = metadata[key].to(
                    device=target_device, non_blocking=True
                )
        return FrozenClipBatch(
            latents=latent_tensor,
            actions=action_tensor,
            metadata=metadata,
        )


class FrozenLatentClipDataset:
    """Expose the original sequence-clip index over a frozen row store.

    The wrapper deliberately preserves the source dataset's ``clip_indices``
    and length.  A seeded ``random_split`` therefore produces the same train
    and validation clip sets as the online image dataset, while ``__getitems__``
    gathers a whole loader batch from the compact memory maps in one operation.
    No pixel/JPEG field is returned.
    """

    def __init__(self, dataset: Any, store: FrozenLatentStore) -> None:
        required = (
            "clip_indices",
            "offsets",
            "lengths",
            "frameskip",
            "num_steps",
            "span",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "The source dataset is missing frozen-clip metadata: "
                + ", ".join(missing)
            )
        if not isinstance(store, FrozenLatentStore):
            raise TypeError("store must be a validated FrozenLatentStore.")
        lengths = _integer_vector(dataset.lengths, label="dataset.lengths")
        offsets = _integer_vector(dataset.offsets, label="dataset.offsets")
        if lengths.size == 0 or lengths.shape != offsets.shape:
            raise ValueError("Dataset episode lengths/offsets are malformed.")
        if np.any(lengths <= 0):
            raise ValueError("Dataset episode lengths must be positive.")
        expected_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(lengths[:-1], dtype=np.int64),
            )
        )
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("Dataset episodes are not contiguous global rows.")
        if int(lengths.sum()) != store.total_rows:
            raise ValueError("Dataset row count differs from the frozen latent store.")
        if int(dataset.frameskip) != store.frame_skip:
            raise ValueError("Dataset frame skip differs from the frozen latent store.")
        if int(dataset.span) != int(dataset.num_steps) * int(dataset.frameskip):
            raise ValueError("Dataset clip span is inconsistent.")

        self.dataset = dataset
        self.store = store
        self.offsets = offsets
        self.lengths = lengths
        self.num_steps = int(dataset.num_steps)
        self.frameskip = int(dataset.frameskip)

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def _clip_start(self, raw_index: int) -> tuple[int, int]:
        index = int(raw_index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Clip index {raw_index} is out of range.")
        episode, local_start = self.dataset.clip_indices[index]
        episode = int(episode)
        local_start = int(local_start)
        if episode < 0 or episode >= self.lengths.size:
            raise ValueError("Dataset clip references an invalid episode.")
        if local_start < 0 or local_start + int(self.dataset.span) > int(
            self.lengths[episode]
        ):
            raise ValueError("Dataset clip crosses its episode boundary.")
        return episode, int(self.offsets[episode]) + local_start

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        identities = [self._clip_start(index) for index in indices]
        expected_episodes = np.asarray(
            [episode for episode, _ in identities], dtype=np.int64
        )
        starts = np.asarray([start for _, start in identities], dtype=np.int64)
        clips = self.store.gather_clips(
            starts,
            num_steps=self.num_steps,
            frame_skip=self.frameskip,
            device="cpu",
        )
        actual_episodes = clips.metadata["episode_ids"].numpy()
        if not np.array_equal(expected_episodes, actual_episodes):
            raise RuntimeError("Frozen clip episode identity is misaligned.")
        return [
            {
                "_tdwm_frozen_latents": clips.latents[position],
                "action": clips.actions[position],
                "_tdwm_global_start": clips.metadata["global_starts"][position],
                "_tdwm_episode_id": clips.metadata["episode_ids"][position],
            }
            for position in range(len(identities))
        ]


__all__ = [
    "ACTION_BLOCK_SLOTS",
    "CUBE_ACTION_BLOCK_DIM",
    "CUBE_ACTION_DIM",
    "EncodedRowBatch",
    "FORMAT_NAME",
    "FrozenClipBatch",
    "FrozenLatentClipDataset",
    "FrozenLatentStore",
    "FrozenLatentStoreBuilder",
    "FrozenLatentStoreSpec",
    "ROW_MAPPING",
    "SCHEMA_VERSION",
    "action_blocks_for_rows",
    "build_frozen_latent_store",
    "file_sha256",
    "iter_encoded_global_rows_once",
    "normalize_actions",
    "validate_episode_ids",
]
