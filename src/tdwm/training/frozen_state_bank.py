"""Strict training-only state bank derived from a frozen latent store.

The bank is an immutable, row-aligned subset of
:class:`~tdwm.training.frozen_latent_store.FrozenLatentStore`.  Its rows are
the sorted union of every TD *current* state in the exact training clip split:

``episode_offsets[episode] + clip_start + frame_skip * current_step``

where ``current_step`` ranges from ``history_frames`` through
``num_steps - 2`` (inclusive).  The manifest seals the source latent-store
manifest, split arrays, Lance clip metadata, index semantics, and every output
array.  This keeps an offline neighbor graph from being paired with a
different checkpoint or split by command-line assertion alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tdwm.training.frozen_latent_store import FrozenLatentStore, file_sha256

FORMAT_NAME = "tdwm-frozen-training-state-bank"
SCHEMA_VERSION = 1
ROW_MAPPING = "arrays are aligned to sorted unique source dataset global rows"
ANCHOR_SOURCE = "union_of_training_clip_td_current_rows"
GLOBAL_ROW_FORMULA = "episode_offsets[episode] + clip_start + frame_skip * current_step"
BANK_FILES = {
    "global_rows": "global_rows.npy",
    "episode_ids": "episode_ids.npy",
    "latents": "latents.npy",
    "actions": "actions.npy",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def array_sha256(values: Any) -> str:
    """Match the canonical split-array digest used by the training code."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _integer_array(
    values: Any,
    *,
    label: str,
    ndim: int,
    trailing_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions.")
    if trailing_size is not None and array.shape[-1] != trailing_size:
        raise ValueError(f"{label} must end in dimension {trailing_size}.")
    if array.dtype.kind not in ("i", "u"):
        raise TypeError(f"{label} must contain integers.")
    if array.dtype.kind == "u" and array.size:
        if int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"{label} exceeds int64.")
    return np.asarray(array, dtype=np.int64)


def load_split_indices(
    split_indices_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and seal one persisted ``split_indices.npz`` artifact."""

    path = Path(split_indices_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as split:
        if "train_indices" not in split.files:
            raise ValueError("split_indices.npz is missing array: train_indices")
        validation_keys = {
            key for key in ("validation_indices", "val_indices") if key in split.files
        }
        if len(validation_keys) != 1:
            raise ValueError(
                "split_indices.npz must contain exactly one of validation_indices "
                "or val_indices."
            )
        validation_key = validation_keys.pop()
        train = _integer_array(split["train_indices"], label="train_indices", ndim=1)
        validation = _integer_array(
            split[validation_key], label="validation_indices", ndim=1
        )
    if train.size == 0:
        raise ValueError("The training split cannot be empty.")
    if np.unique(train).size != train.size:
        raise ValueError("train_indices must be unique.")
    if np.unique(validation).size != validation.size:
        raise ValueError("validation_indices must be unique.")
    if np.intersect1d(train, validation, assume_unique=True).size:
        raise ValueError("Training and validation split indices overlap.")
    metadata = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "train_samples": int(train.size),
        "validation_samples": int(validation.size),
        "validation_array_key": validation_key,
        "train_indices_sha256": array_sha256(train),
        "validation_indices_sha256": array_sha256(validation),
        "array_hash_algorithm": "dtype_text_then_shape_text_then_c_order_bytes",
    }
    return train, validation, metadata


def _canonical_clip_metadata(
    *,
    clip_indices: Any,
    episode_offsets: Any,
    total_rows: int,
    num_steps: int,
    frame_skip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clips = _integer_array(
        clip_indices,
        label="clip_indices",
        ndim=2,
        trailing_size=2,
    )
    offsets = _integer_array(
        episode_offsets,
        label="episode_offsets",
        ndim=1,
    )
    if clips.shape[0] == 0:
        raise ValueError("clip_indices cannot be empty.")
    if offsets.size == 0 or offsets[0] != 0:
        raise ValueError("episode_offsets must be non-empty and start at zero.")
    if np.any(offsets < 0) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("episode_offsets must be strictly increasing.")
    if offsets[-1] >= total_rows:
        raise ValueError("episode_offsets contains an out-of-range row.")
    if num_steps <= 1:
        raise ValueError("num_steps must leave a current/next TD pair.")
    if frame_skip <= 0:
        raise ValueError("frame_skip must be positive.")

    episodes = clips[:, 0]
    starts = clips[:, 1]
    if np.any(episodes < 0) or np.any(episodes >= offsets.size):
        raise IndexError("clip_indices contains an out-of-range episode ID.")
    if np.any(starts < 0):
        raise ValueError("clip starts cannot be negative.")
    episode_ends = np.concatenate(
        [offsets[1:], np.asarray([total_rows], dtype=np.int64)]
    )
    episode_lengths = episode_ends - offsets
    final_observation = starts + (num_steps - 1) * frame_skip
    if np.any(final_observation >= episode_lengths[episodes]):
        raise ValueError("A Lance clip crosses its episode boundary.")
    return clips, offsets, episode_lengths


def derive_training_anchor_rows(
    *,
    train_indices: Any,
    clip_indices: Any,
    episode_offsets: Any,
    total_rows: int,
    num_steps: int,
    history_frames: int,
    frame_skip: int,
    source_episode_ids: Any | None = None,
    clip_chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact sorted union of training TD-current global rows.

    The returned tuple is ``(global_rows, clips, offsets, episode_lengths)``.
    A boolean global-row mask bounds temporary memory even when many
    overlapping training clips map to the same rows.
    """

    if history_frames <= 0 or num_steps <= history_frames + 1:
        raise ValueError(
            "num_steps must leave at least one TD current/next pair after history."
        )
    if clip_chunk_size <= 0:
        raise ValueError("clip_chunk_size must be positive.")
    clips, offsets, episode_lengths = _canonical_clip_metadata(
        clip_indices=clip_indices,
        episode_offsets=episode_offsets,
        total_rows=int(total_rows),
        num_steps=int(num_steps),
        frame_skip=int(frame_skip),
    )
    train = _integer_array(train_indices, label="train_indices", ndim=1)
    if train.size == 0:
        raise ValueError("train_indices cannot be empty.")
    if np.unique(train).size != train.size:
        raise ValueError("train_indices must be unique.")
    if np.any(train < 0) or np.any(train >= clips.shape[0]):
        raise IndexError("train_indices contains an out-of-range clip index.")

    store_episodes = None
    if source_episode_ids is not None:
        store_episodes = _integer_array(
            source_episode_ids,
            label="source_episode_ids",
            ndim=1,
        )
        if store_episodes.shape != (int(total_rows),):
            raise ValueError("source_episode_ids must align to every global row.")
        expected = np.repeat(np.arange(offsets.size, dtype=np.int64), episode_lengths)
        if not np.array_equal(store_episodes, expected):
            raise ValueError(
                "Lance episode offsets differ from frozen latent-store episode IDs."
            )

    current_steps = np.arange(int(history_frames), int(num_steps) - 1, dtype=np.int64)
    current_offsets = current_steps * int(frame_skip)
    selected = np.zeros(int(total_rows), dtype=np.bool_)
    for chunk_start in range(0, train.size, int(clip_chunk_size)):
        selected_clips = clips[train[chunk_start : chunk_start + int(clip_chunk_size)]]
        clip_episodes = selected_clips[:, 0]
        global_starts = offsets[clip_episodes] + selected_clips[:, 1]
        candidates = global_starts[:, None] + current_offsets[None, :]
        if np.any(candidates < 0) or np.any(candidates >= int(total_rows)):
            raise IndexError("A derived TD-current row is outside the latent store.")
        if store_episodes is not None:
            candidate_episodes = np.asarray(store_episodes[candidates])
            if np.any(candidate_episodes != clip_episodes[:, None]):
                raise ValueError(
                    "A derived TD-current row crosses a frozen-store episode."
                )
        selected[candidates.reshape(-1)] = True
    global_rows = np.flatnonzero(selected).astype(np.int64, copy=False)
    if global_rows.size == 0:
        raise RuntimeError("The training split produced no TD-current rows.")
    return global_rows, clips, offsets, episode_lengths


def _flush_and_close(array: np.memmap) -> None:
    array.flush()
    mapped = getattr(array, "_mmap", None)
    if mapped is not None:
        mapped.close()


def _file_entry(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "dtype": np.dtype(array.dtype).name,
        "shape": [int(value) for value in array.shape],
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(dict(manifest), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_frozen_training_state_bank(
    output_dir: str | Path,
    *,
    latent_store: FrozenLatentStore,
    split_indices_path: str | Path,
    expected_training_split_sha256: str,
    clip_indices: Sequence[tuple[int, int]] | np.ndarray,
    episode_offsets: Sequence[int] | np.ndarray,
    num_steps: int,
    history_frames: int,
    frame_skip: int,
    source_metadata: Mapping[str, Any] | None = None,
    copy_chunk_rows: int = 65_536,
) -> dict[str, Any]:
    """Build and atomically publish one provenance-locked training bank."""

    if not isinstance(latent_store, FrozenLatentStore):
        raise TypeError("latent_store must be a validated FrozenLatentStore.")
    expected_split = _validate_sha256(
        expected_training_split_sha256,
        label="expected_training_split_sha256",
    )
    if copy_chunk_rows <= 0:
        raise ValueError("copy_chunk_rows must be positive.")
    if int(frame_skip) != latent_store.frame_skip:
        raise ValueError("frame_skip differs from the frozen latent store.")
    if int(history_frames) != latent_store.history_frames:
        raise ValueError("history_frames differs from the frozen latent store.")
    latent_store._assert_immutable()

    train, validation, split_metadata = load_split_indices(split_indices_path)
    if split_metadata["train_indices_sha256"] != expected_split:
        raise ValueError(
            "The persisted training split SHA-256 differs from the active run."
        )
    global_rows, clips, offsets, episode_lengths = derive_training_anchor_rows(
        train_indices=train,
        clip_indices=clip_indices,
        episode_offsets=episode_offsets,
        total_rows=latent_store.total_rows,
        num_steps=int(num_steps),
        history_frames=int(history_frames),
        frame_skip=int(frame_skip),
        source_episode_ids=latent_store.episode_ids,
    )
    split_union = np.concatenate([train, validation])
    if (
        split_union.size != clips.shape[0]
        or np.any(split_union < 0)
        or np.any(split_union >= clips.shape[0])
        or np.unique(split_union).size != clips.shape[0]
    ):
        raise ValueError(
            "The persisted train/validation split must partition every Lance clip."
        )

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen state bank: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    arrays: dict[str, np.ndarray] = {}
    try:
        global_path = staging / BANK_FILES["global_rows"]
        np.save(global_path, global_rows, allow_pickle=False)
        arrays["global_rows"] = np.load(global_path, mmap_mode="r", allow_pickle=False)

        episode_path = staging / BANK_FILES["episode_ids"]
        bank_episodes = np.asarray(
            latent_store.episode_ids[global_rows], dtype=np.int64
        )
        np.save(episode_path, bank_episodes, allow_pickle=False)
        arrays["episode_ids"] = np.load(episode_path, mmap_mode="r", allow_pickle=False)

        latent_output = np.lib.format.open_memmap(
            staging / BANK_FILES["latents"],
            mode="w+",
            dtype=np.float32,
            shape=(global_rows.size, latent_store.embed_dim),
        )
        action_output = np.lib.format.open_memmap(
            staging / BANK_FILES["actions"],
            mode="w+",
            dtype=np.float32,
            shape=(global_rows.size, latent_store.action_block_dim),
        )
        for start in range(0, global_rows.size, int(copy_chunk_rows)):
            end = min(global_rows.size, start + int(copy_chunk_rows))
            rows = global_rows[start:end]
            latent_chunk = np.asarray(latent_store.latents[rows], dtype=np.float32)
            action_chunk = np.asarray(latent_store.actions[rows], dtype=np.float32)
            if not np.isfinite(latent_chunk).all():
                raise ValueError("A selected frozen latent is non-finite.")
            if not np.isfinite(action_chunk).all():
                bad_rows = rows[~np.isfinite(action_chunk).all(axis=1)][:8]
                raise ValueError(
                    "Training TD-current action anchors must be finite; bad global "
                    f"rows: {bad_rows.tolist()}"
                )
            latent_output[start:end] = latent_chunk
            action_output[start:end] = action_chunk
        _flush_and_close(latent_output)
        _flush_and_close(action_output)
        arrays["latents"] = np.load(
            staging / BANK_FILES["latents"], mmap_mode="r", allow_pickle=False
        )
        arrays["actions"] = np.load(
            staging / BANK_FILES["actions"], mmap_mode="r", allow_pickle=False
        )
        files = {
            name: _file_entry(staging / BANK_FILES[name], array)
            for name, array in arrays.items()
        }

        latent_manifest = latent_store.manifest
        checkpoint_sha = _validate_sha256(
            latent_manifest.get("pretrained_checkpoint_sha256"),
            label="latent_store.pretrained_checkpoint_sha256",
        )
        dataset_sha = _validate_sha256(
            latent_manifest.get("dataset_source_sha256"),
            label="latent_store.dataset_source_sha256",
        )
        normalization_sha = _validate_sha256(
            latent_manifest.get("column_normalization_sha256"),
            label="latent_store.column_normalization_sha256",
        )
        current_steps = list(range(int(history_frames), int(num_steps) - 1))
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_NAME,
            "row_mapping": ROW_MAPPING,
            "pretrained_world_model_sha256": checkpoint_sha,
            "dataset_source_sha256": dataset_sha,
            "column_normalization_sha256": normalization_sha,
            "training_split_sha256": expected_split,
            "latent_store_manifest_sha256": latent_store.manifest_sha256,
            "tdwm_git_revision": latent_manifest.get("git_revision"),
            "row_count": int(global_rows.size),
            "latent_dim": int(latent_store.embed_dim),
            "action_block_dim": int(latent_store.action_block_dim),
            "source_latent_store": {
                "path": str(latent_store.root),
                "manifest_path": str(latent_store.manifest_path),
                "manifest_sha256": latent_store.manifest_sha256,
                "total_rows": int(latent_store.total_rows),
                "pretrained_checkpoint_sha256": checkpoint_sha,
                "dataset_source_sha256": dataset_sha,
                "column_normalization_sha256": normalization_sha,
                "git_revision": latent_manifest.get("git_revision"),
                "embed_dim": int(latent_store.embed_dim),
                "frame_skip": int(latent_store.frame_skip),
                "history_frames": int(latent_store.history_frames),
                "action_dim": int(latent_store.action_dim),
                "action_block_dim": int(latent_store.action_block_dim),
                "files": latent_manifest.get("files"),
            },
            "training_split": split_metadata,
            "clip_metadata": {
                "clip_count": int(clips.shape[0]),
                "episode_count": int(offsets.size),
                "clip_indices_dtype": str(clips.dtype),
                "clip_indices_shape": list(clips.shape),
                "clip_indices_sha256": array_sha256(clips),
                "episode_offsets_dtype": str(offsets.dtype),
                "episode_offsets_shape": list(offsets.shape),
                "episode_offsets_sha256": array_sha256(offsets),
                "episode_lengths_dtype": str(episode_lengths.dtype),
                "episode_lengths_shape": list(episode_lengths.shape),
                "episode_lengths_sha256": array_sha256(episode_lengths),
            },
            "index_semantics": {
                "split": "training_only",
                "split_unit": "sequence_clip_index",
                "anchor_source": ANCHOR_SOURCE,
                "history_frames": int(history_frames),
                "num_steps": int(num_steps),
                "frame_skip": int(frame_skip),
                "first_current_step": int(history_frames),
                "last_current_step_inclusive": int(num_steps) - 2,
                "current_steps": current_steps,
                "global_row_formula": GLOBAL_ROW_FORMULA,
                "deduplication": "sorted_unique_global_rows",
                "retrieval_key": "frozen_current_state_latent",
                "action_anchor": "same_global_row_five_slot_normalized_block",
                "terminal_action_policy": "reject_nonfinite_action_anchor",
            },
            "validation": {
                "global_rows_strictly_increasing": True,
                "source_episode_ids_match_lance_offsets": True,
                "latents_all_finite": True,
                "action_anchors_all_finite": True,
            },
            "source_metadata": dict(source_metadata or {}),
            "files": files,
        }
        _write_manifest(staging / "manifest.json", manifest)
        latent_store._assert_immutable()
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@dataclass(frozen=True)
class FrozenTrainingStateBank:
    """Exact-SHA-validated, memory-mapped training state bank."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    global_rows: np.ndarray
    episode_ids: np.ndarray
    latents: np.ndarray
    actions: np.ndarray
    files: dict[str, dict[str, Any]]
    _paths: dict[str, Path] = dataclass_field(repr=False)
    _identities: dict[str, tuple[int, int, int, int]] = dataclass_field(repr=False)
    _manifest_identity: tuple[int, int, int, int] = dataclass_field(repr=False)

    @property
    def size(self) -> int:
        return int(self.global_rows.shape[0])

    def assert_immutable(self) -> None:
        """Reject mutation after the artifact's exact-hash validation."""

        if _stat_identity(self.manifest_path) != self._manifest_identity:
            raise RuntimeError("Frozen state-bank manifest changed after validation.")
        for name, expected in self._identities.items():
            if _stat_identity(self._paths[name]) != expected:
                raise RuntimeError(
                    f"Frozen state-bank file changed after validation: {name}"
                )


def _manifest_file(
    *,
    root: Path,
    files: Mapping[str, Any],
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> tuple[np.memmap, dict[str, Any], Path]:
    entry = files.get(name)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"files.{name} must describe one array.")
    path = (root / entry["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"files.{name} escapes the state bank.") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = _validate_sha256(entry.get("sha256"), label=f"files.{name}.sha256")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"files.{name} failed exact SHA-256 validation.")
    if entry.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"files.{name}.size_bytes is incorrect.")
    if entry.get("dtype") != np.dtype(dtype).name:
        raise ValueError(f"files.{name}.dtype is incompatible.")
    if entry.get("shape") != list(shape):
        raise ValueError(f"files.{name}.shape is incompatible.")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise TypeError(f"files.{name} was not memory mapped.")
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise ValueError(f"files.{name} array metadata is incompatible.")
    return array, dict(entry), path


def load_frozen_training_state_bank(
    bank_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_training_split_sha256: str | None = None,
    expected_dataset_source_sha256: str | None = None,
    expected_column_normalization_sha256: str | None = None,
) -> FrozenTrainingStateBank:
    """Load a bank only after validating its complete provenance manifest."""

    root = Path(bank_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Frozen state-bank manifest must be a JSON object.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported frozen state-bank schema.")
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError("The frozen state-bank format is incompatible.")
    if manifest.get("row_mapping") != ROW_MAPPING:
        raise ValueError("The frozen state-bank row mapping is incompatible.")

    digests = {
        "pretrained_world_model_sha256": expected_checkpoint_sha256,
        "training_split_sha256": expected_training_split_sha256,
        "dataset_source_sha256": expected_dataset_source_sha256,
        "column_normalization_sha256": expected_column_normalization_sha256,
    }
    for field, expected in digests.items():
        actual = _validate_sha256(manifest.get(field), label=field)
        if expected is not None:
            locked = _validate_sha256(expected, label=f"expected_{field}")
            if actual != locked:
                raise ValueError(
                    f"Frozen state bank {field} differs from the active protocol."
                )
    latent_manifest_sha = _validate_sha256(
        manifest.get("latent_store_manifest_sha256"),
        label="latent_store_manifest_sha256",
    )
    source_store = manifest.get("source_latent_store")
    if not isinstance(source_store, dict):
        raise ValueError("Frozen state bank is missing source_latent_store.")
    if source_store.get("manifest_sha256") != latent_manifest_sha:
        raise ValueError("The latent-store manifest SHA binding is inconsistent.")
    for field in (
        "pretrained_checkpoint_sha256",
        "dataset_source_sha256",
        "column_normalization_sha256",
    ):
        if source_store.get(field) != manifest.get(
            "pretrained_world_model_sha256"
            if field == "pretrained_checkpoint_sha256"
            else field
        ):
            raise ValueError(f"source_latent_store.{field} is inconsistent.")
    revision = manifest.get("tdwm_git_revision")
    if (
        not isinstance(revision, str)
        or _GIT_REVISION_RE.fullmatch(revision) is None
        or source_store.get("git_revision") != revision
    ):
        raise ValueError("The source latent-store Git revision is inconsistent.")

    split = manifest.get("training_split")
    if not isinstance(split, dict):
        raise ValueError("Frozen state bank is missing training_split metadata.")
    _validate_sha256(split.get("file_sha256"), label="training_split.file_sha256")
    split_train_sha = _validate_sha256(
        split.get("train_indices_sha256"),
        label="training_split.train_indices_sha256",
    )
    if split_train_sha != manifest["training_split_sha256"]:
        raise ValueError("The training split SHA binding is inconsistent.")
    _validate_sha256(
        split.get("validation_indices_sha256"),
        label="training_split.validation_indices_sha256",
    )
    train_samples = split.get("train_samples")
    validation_samples = split.get("validation_samples")
    if (
        isinstance(train_samples, bool)
        or not isinstance(train_samples, int)
        or train_samples <= 0
        or isinstance(validation_samples, bool)
        or not isinstance(validation_samples, int)
        or validation_samples < 0
    ):
        raise ValueError("The sealed split sample counts are invalid.")

    semantics = manifest.get("index_semantics")
    expected_semantics = {
        "split": "training_only",
        "split_unit": "sequence_clip_index",
        "anchor_source": ANCHOR_SOURCE,
        "global_row_formula": GLOBAL_ROW_FORMULA,
        "deduplication": "sorted_unique_global_rows",
        "retrieval_key": "frozen_current_state_latent",
        "terminal_action_policy": "reject_nonfinite_action_anchor",
    }
    if not isinstance(semantics, dict):
        raise ValueError("Frozen state bank is missing index_semantics.")
    for field, expected in expected_semantics.items():
        if semantics.get(field) != expected:
            raise ValueError(f"State-bank index semantics differ at {field}.")
    history = semantics.get("history_frames")
    num_steps = semantics.get("num_steps")
    frame_skip = semantics.get("frame_skip")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (history, num_steps, frame_skip)
    ):
        raise ValueError("State-bank sequence dimensions are invalid.")
    if num_steps <= history + 1:
        raise ValueError("State-bank sequence dimensions leave no TD pair.")
    expected_steps = list(range(history, num_steps - 1))
    if semantics.get("first_current_step") != history:
        raise ValueError("State-bank first_current_step is inconsistent.")
    if semantics.get("last_current_step_inclusive") != num_steps - 2:
        raise ValueError("State-bank last_current_step is inconsistent.")
    if semantics.get("current_steps") != expected_steps:
        raise ValueError("State-bank current_steps are inconsistent.")

    clip_metadata = manifest.get("clip_metadata")
    if not isinstance(clip_metadata, dict):
        raise ValueError("Frozen state bank is missing clip_metadata.")
    clip_count = clip_metadata.get("clip_count")
    episode_count = clip_metadata.get("episode_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (clip_count, episode_count)
    ):
        raise ValueError("State-bank clip metadata dimensions are invalid.")
    if train_samples + validation_samples != clip_count:
        raise ValueError("The sealed split does not partition every Lance clip.")
    expected_clip_metadata = {
        "clip_indices_dtype": "int64",
        "clip_indices_shape": [clip_count, 2],
        "episode_offsets_dtype": "int64",
        "episode_offsets_shape": [episode_count],
        "episode_lengths_dtype": "int64",
        "episode_lengths_shape": [episode_count],
    }
    for field_name, expected in expected_clip_metadata.items():
        if clip_metadata.get(field_name) != expected:
            raise ValueError(f"State-bank clip metadata differs at {field_name}.")
    for field_name in (
        "clip_indices_sha256",
        "episode_offsets_sha256",
        "episode_lengths_sha256",
    ):
        _validate_sha256(
            clip_metadata.get(field_name), label=f"clip_metadata.{field_name}"
        )

    row_count = manifest.get("row_count")
    latent_dim = manifest.get("latent_dim")
    action_dim = manifest.get("action_block_dim")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (row_count, latent_dim, action_dim)
    ):
        raise ValueError("Frozen state-bank dimensions must be positive integers.")
    source_rows = source_store.get("total_rows")
    if (
        isinstance(source_rows, bool)
        or not isinstance(source_rows, int)
        or source_rows <= 0
    ):
        raise ValueError("source_latent_store.total_rows is invalid.")
    source_bindings = {
        "embed_dim": latent_dim,
        "frame_skip": frame_skip,
        "history_frames": history,
        "action_block_dim": action_dim,
    }
    for field_name, expected in source_bindings.items():
        if source_store.get(field_name) != expected:
            raise ValueError(f"source_latent_store.{field_name} is inconsistent.")
    source_action_dim = source_store.get("action_dim")
    if (
        isinstance(source_action_dim, bool)
        or not isinstance(source_action_dim, int)
        or source_action_dim <= 0
        or source_action_dim * frame_skip != action_dim
    ):
        raise ValueError("source_latent_store.action_dim is inconsistent.")
    source_files = source_store.get("files")
    if not isinstance(source_files, dict) or set(source_files) != {
        "latents",
        "action_blocks",
        "episode_ids",
    }:
        raise ValueError("source_latent_store.files is incomplete.")
    expected_source_files = {
        "latents": ("float32", [source_rows, latent_dim]),
        "action_blocks": ("float32", [source_rows, action_dim]),
        "episode_ids": ("int64", [source_rows]),
    }
    for name, (dtype_name, shape) in expected_source_files.items():
        entry = source_files.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"source_latent_store.files.{name} is malformed.")
        _validate_sha256(
            entry.get("sha256"),
            label=f"source_latent_store.files.{name}.sha256",
        )
        if (
            not isinstance(entry.get("path"), str)
            or entry.get("dtype") != dtype_name
            or entry.get("shape") != shape
            or isinstance(entry.get("size_bytes"), bool)
            or not isinstance(entry.get("size_bytes"), int)
            or entry.get("size_bytes") <= 0
        ):
            raise ValueError(f"source_latent_store.files.{name} is inconsistent.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(BANK_FILES):
        raise ValueError("Frozen state-bank manifest must seal exactly four arrays.")
    global_rows, global_entry, global_path = _manifest_file(
        root=root,
        files=files,
        name="global_rows",
        dtype=np.int64,
        shape=(row_count,),
    )
    episode_ids, episode_entry, episode_path = _manifest_file(
        root=root,
        files=files,
        name="episode_ids",
        dtype=np.int64,
        shape=(row_count,),
    )
    latents, latent_entry, latent_path = _manifest_file(
        root=root,
        files=files,
        name="latents",
        dtype=np.float32,
        shape=(row_count, latent_dim),
    )
    actions, action_entry, action_path = _manifest_file(
        root=root,
        files=files,
        name="actions",
        dtype=np.float32,
        shape=(row_count, action_dim),
    )
    if row_count > 1 and np.any(global_rows[1:] <= global_rows[:-1]):
        raise ValueError("global_rows must be strictly increasing and unique.")
    if np.any(global_rows < 0) or np.any(global_rows >= source_rows):
        raise ValueError("global_rows contains a source-store row out of range.")
    if np.any(episode_ids < 0):
        raise ValueError("episode_ids cannot be negative.")
    if row_count > 1 and np.any(episode_ids[1:] < episode_ids[:-1]):
        raise ValueError("episode_ids must be nondecreasing with global_rows.")
    if not np.isfinite(latents).all():
        raise ValueError("Frozen state-bank latents must all be finite.")
    if not np.isfinite(actions).all():
        raise ValueError("Frozen state-bank action anchors must all be finite.")
    return FrozenTrainingStateBank(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest=manifest,
        global_rows=global_rows,
        episode_ids=episode_ids,
        latents=latents,
        actions=actions,
        files={
            "global_rows": global_entry,
            "episode_ids": episode_entry,
            "latents": latent_entry,
            "actions": action_entry,
        },
        _paths={
            "global_rows": global_path,
            "episode_ids": episode_path,
            "latents": latent_path,
            "actions": action_path,
        },
        _identities={
            "global_rows": _stat_identity(global_path),
            "episode_ids": _stat_identity(episode_path),
            "latents": _stat_identity(latent_path),
            "actions": _stat_identity(action_path),
        },
        _manifest_identity=_stat_identity(manifest_path),
    )


__all__ = [
    "ANCHOR_SOURCE",
    "BANK_FILES",
    "FORMAT_NAME",
    "FrozenTrainingStateBank",
    "GLOBAL_ROW_FORMULA",
    "ROW_MAPPING",
    "SCHEMA_VERSION",
    "array_sha256",
    "build_frozen_training_state_bank",
    "derive_training_anchor_rows",
    "load_frozen_training_state_bank",
    "load_split_indices",
]
