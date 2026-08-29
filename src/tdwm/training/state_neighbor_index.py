"""Memory-mapped retrieval for frozen-LeWM state-neighbor action prefixes.

The expensive nearest-neighbor search is deliberately an offline step.  A
training batch carries the backing dataset row for each clip; this module maps
those rows to a precomputed set of neighbors and gathers only their recorded
action blocks.  Counterfactual actions therefore never need a TD target and
the main training loop never scans the full latent bank.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

INDEX_SCHEMA_VERSION = 1
INDEX_METHOD = "frozen_lewm_state_neighbor_action_prefix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(
    root: Path,
    files: dict[str, Any],
    name: str,
    *,
    mmap_mode: str = "r",
) -> np.ndarray:
    entry = files.get(name)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"neighbor-index files.{name} must name an array.")
    path = (root / entry["path"]).resolve()
    if root not in path.parents:
        raise ValueError(f"neighbor-index files.{name} escapes the index directory.")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"neighbor-index files.{name}.sha256 is invalid.")
    if _sha256(path) != expected_hash:
        raise ValueError(f"neighbor-index files.{name} failed SHA-256 validation.")
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


@dataclass(frozen=True)
class NeighborActionBatch:
    """Recorded neighbor prefixes and their frozen-state distances."""

    actions: torch.Tensor
    distances: torch.Tensor
    neighbor_rows: torch.Tensor


class StateNeighborActionIndex:
    """Validated, memory-mapped lookup table keyed by global dataset row."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_latent_store_manifest_sha256: str,
        expected_action_block_dim: int,
        expected_k: int | None = None,
    ) -> None:
        root = Path(index_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported state-neighbor index schema.")
        if manifest.get("method") != INDEX_METHOD:
            raise ValueError("The state-neighbor index method is incompatible.")
        if manifest.get("pretrained_world_model_sha256") != expected_checkpoint_sha256:
            raise ValueError(
                "The state-neighbor index was built with a different frozen LeWM."
            )
        source_bank = manifest.get("source_bank")
        if not isinstance(source_bank, dict) or (
            source_bank.get("latent_store_manifest_sha256")
            != expected_latent_store_manifest_sha256
        ):
            raise ValueError(
                "The state-neighbor index was built from a different frozen "
                "latent store."
            )
        if manifest.get("metric") != "squared_l2":
            raise ValueError("The first G1 protocol requires squared-L2 neighbors.")
        if manifest.get("retrieval_key") != "frozen_current_state_latent":
            raise ValueError("The G1 retrieval key must be the frozen current latent.")
        if manifest.get("split") != "training_only":
            raise ValueError(
                "The G1 neighbor bank must contain only the training split."
            )
        if manifest.get("exclude_exact_transition") is not True:
            raise ValueError("The G1 index must exclude the anchor transition.")
        if manifest.get("exclude_same_episode") is not True:
            raise ValueError("The G1 index must exclude same-episode neighbors.")

        action_dim = int(manifest.get("action_block_dim", 0))
        if action_dim != int(expected_action_block_dim):
            raise ValueError("The neighbor action-block dimension is incompatible.")
        k = int(manifest.get("neighbors_per_anchor", 0))
        if k <= 0 or (expected_k is not None and k != int(expected_k)):
            raise ValueError("The neighbor count is incompatible with the protocol.")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("The neighbor-index manifest is missing files.")
        global_rows = _load_array(root, files, "global_rows")
        episode_ids = _load_array(root, files, "episode_ids")
        actions = _load_array(root, files, "actions")
        neighbor_indices = _load_array(root, files, "neighbor_indices")
        neighbor_distances = _load_array(root, files, "neighbor_distances")

        size = int(global_rows.shape[0]) if global_rows.ndim == 1 else -1
        if size <= 0:
            raise ValueError("global_rows must be a non-empty vector.")
        if global_rows.dtype != np.int64:
            raise TypeError("global_rows must use int64.")
        if episode_ids.shape != (size,) or not np.issubdtype(
            episode_ids.dtype, np.integer
        ):
            raise ValueError("episode_ids must be one integer per anchor.")
        if actions.shape != (size, action_dim) or not np.issubdtype(
            actions.dtype, np.floating
        ):
            raise ValueError(
                "actions must contain one floating action block per anchor."
            )
        if neighbor_indices.shape != (size, k) or not np.issubdtype(
            neighbor_indices.dtype, np.integer
        ):
            raise ValueError("neighbor_indices has the wrong shape or dtype.")
        if neighbor_distances.shape != (size, k) or not np.issubdtype(
            neighbor_distances.dtype, np.floating
        ):
            raise ValueError("neighbor_distances has the wrong shape or dtype.")
        if size > 1 and np.any(global_rows[1:] <= global_rows[:-1]):
            raise ValueError("global_rows must be strictly increasing and unique.")

        self.root = root
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.global_rows = global_rows
        self.episode_ids = episode_ids
        self.actions = actions
        self.neighbor_indices = neighbor_indices
        self.neighbor_distances = neighbor_distances
        self.action_block_dim = action_dim
        self.k = k

    def lookup(
        self,
        global_rows: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> NeighborActionBatch:
        """Gather precomputed neighbor action blocks for arbitrary row shape."""

        if not isinstance(global_rows, torch.Tensor) or global_rows.numel() == 0:
            raise ValueError("global_rows must be a non-empty tensor.")
        if global_rows.is_floating_point() or global_rows.is_complex():
            raise TypeError("global_rows must contain integers.")
        query_shape = tuple(global_rows.shape)
        query = global_rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        query_array = query.numpy()
        positions = np.searchsorted(self.global_rows, query_array)
        in_range = positions < self.global_rows.shape[0]
        exact = np.zeros_like(in_range, dtype=np.bool_)
        exact[in_range] = self.global_rows[positions[in_range]] == query_array[in_range]
        if not np.all(exact):
            missing = query_array[~exact][:8].tolist()
            raise KeyError(f"Rows are absent from the G1 neighbor index: {missing}")

        neighbor_indices = np.asarray(self.neighbor_indices[positions], dtype=np.int64)
        if np.any(neighbor_indices < 0) or np.any(
            neighbor_indices >= self.global_rows.shape[0]
        ):
            raise ValueError("The G1 index contains out-of-range neighbor indices.")
        anchor_episodes = np.asarray(self.episode_ids[positions])[:, None]
        neighbor_episodes = np.asarray(self.episode_ids[neighbor_indices])
        if np.any(anchor_episodes == neighbor_episodes):
            raise ValueError("The G1 index contains a forbidden same-episode neighbor.")

        neighbor_actions = np.array(self.actions[neighbor_indices], copy=True)
        distances = np.array(self.neighbor_distances[positions], copy=True)
        neighbor_rows = np.array(self.global_rows[neighbor_indices], copy=True)
        leading = query_shape + (self.k,)
        return NeighborActionBatch(
            actions=torch.from_numpy(neighbor_actions)
            .reshape(leading + (self.action_block_dim,))
            .to(device=device, dtype=dtype, non_blocking=True),
            distances=torch.from_numpy(distances)
            .reshape(leading)
            .to(device=device, dtype=torch.float32, non_blocking=True),
            neighbor_rows=torch.from_numpy(neighbor_rows).reshape(leading),
        )


__all__ = [
    "INDEX_METHOD",
    "INDEX_SCHEMA_VERSION",
    "NeighborActionBatch",
    "StateNeighborActionIndex",
]
