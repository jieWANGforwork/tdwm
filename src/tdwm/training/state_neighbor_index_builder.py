"""Offline construction of frozen-state neighbor action-prefix graphs.

The input bank is produced separately and contains a strict provenance
manifest plus four row-aligned ``.npy`` files: ``global_rows.npy``,
``episode_ids.npy``, ``latents.npy``, and ``actions.npy``.  This module
performs nearest-neighbor search once, filters out the anchor and every
transition from its episode, and writes the compact graph consumed by
:mod:`tdwm.training.state_neighbor_index`.

No class in this module is used by the training step.  Formal construction can
use FAISS HNSW; the exact NumPy backend exists for small audits and tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tdwm.training.frozen_state_bank import (
    BANK_FILES,
    FrozenTrainingStateBank,
    load_frozen_training_state_bank,
)
from tdwm.training.state_neighbor_index import INDEX_METHOD, INDEX_SCHEMA_VERSION


class NeighborSearchBackend(Protocol):
    """Offline squared-L2 search interface used by the graph builder."""

    name: str
    version: str

    def build(self, vectors: np.ndarray) -> None: ...

    def search(
        self, queries: np.ndarray, search_depth: int
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def metadata(self) -> dict[str, Any]: ...


FrozenStateBank = FrozenTrainingStateBank


class ExactNumpySearchBackend:
    """Deterministic exact backend intended only for small tests and audits."""

    name = "exact_numpy"
    version = np.__version__

    def __init__(self, *, max_pairwise_entries: int = 20_000_000) -> None:
        if max_pairwise_entries <= 0:
            raise ValueError("max_pairwise_entries must be positive.")
        self.max_pairwise_entries = int(max_pairwise_entries)
        self._vectors: np.ndarray | None = None
        self._squared_norms: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        matrix = _as_float32_matrix(vectors, name="vectors")
        self._vectors = np.ascontiguousarray(matrix)
        self._squared_norms = np.einsum(
            "nd,nd->n", self._vectors, self._vectors
        )

    def search(
        self, queries: np.ndarray, search_depth: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or self._squared_norms is None:
            raise RuntimeError("The exact backend must be built before search.")
        query = _as_float32_matrix(queries, name="queries")
        if query.shape[1] != self._vectors.shape[1]:
            raise ValueError("Query and bank latent dimensions differ.")
        if search_depth <= 0 or search_depth > self._vectors.shape[0]:
            raise ValueError("search_depth must lie in [1, bank_size].")
        pairwise_entries = int(query.shape[0] * self._vectors.shape[0])
        if pairwise_entries > self.max_pairwise_entries:
            raise RuntimeError(
                "The exact NumPy backend is restricted to small offline audits; "
                "use FAISS HNSW for the formal bank."
            )
        query_norms = np.einsum("nd,nd->n", query, query)[:, None]
        distances = (
            query_norms
            + self._squared_norms[None, :]
            - 2.0 * query @ self._vectors.T
        )
        np.maximum(distances, 0.0, out=distances)
        indices = np.argsort(distances, axis=1, kind="stable")[:, :search_depth]
        selected = np.take_along_axis(distances, indices, axis=1)
        return selected.astype(np.float32, copy=False), indices.astype(
            np.int64, copy=False
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "formal_backend": False,
            "max_pairwise_entries": self.max_pairwise_entries,
        }


class FaissHNSWSearchBackend:
    """Deterministic, single-threaded FAISS HNSW construction backend."""

    name = "faiss_hnsw"

    def __init__(
        self,
        *,
        seed: int,
        threads: int = 1,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 256,
        add_batch_size: int = 65_536,
    ) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "FAISS is required for the formal HNSW neighbor-index build."
            ) from error
        if threads != 1:
            raise ValueError(
                "Deterministic FAISS HNSW construction requires threads=1."
            )
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if min(hnsw_m, ef_construction, ef_search, add_batch_size) <= 0:
            raise ValueError("FAISS HNSW parameters must be positive.")
        self.faiss = faiss
        self.seed = int(seed)
        self.threads = int(threads)
        self.hnsw_m = int(hnsw_m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.add_batch_size = int(add_batch_size)
        self.version = str(getattr(faiss, "__version__", "unknown"))
        self._index = None
        self._insertion_order: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        matrix = _as_float32_matrix(vectors, name="vectors")
        self.faiss.omp_set_num_threads(self.threads)
        index = self.faiss.IndexHNSWFlat(
            int(matrix.shape[1]), self.hnsw_m, self.faiss.METRIC_L2
        )
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        order = np.random.default_rng(self.seed).permutation(matrix.shape[0])
        order = np.asarray(order, dtype=np.int64)
        for start in range(0, order.size, self.add_batch_size):
            selected = order[start : start + self.add_batch_size]
            index.add(np.ascontiguousarray(matrix[selected], dtype=np.float32))
        self._index = index
        self._insertion_order = order

    def search(
        self, queries: np.ndarray, search_depth: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._insertion_order is None:
            raise RuntimeError("The FAISS backend must be built before search.")
        query = _as_float32_matrix(queries, name="queries")
        if search_depth <= 0 or search_depth > self._insertion_order.size:
            raise ValueError("search_depth must lie in [1, bank_size].")
        self.faiss.omp_set_num_threads(self.threads)
        self._index.hnsw.efSearch = max(self.ef_search, int(search_depth))
        distances, internal_indices = self._index.search(
            np.ascontiguousarray(query), int(search_depth)
        )
        valid = internal_indices >= 0
        original_indices = np.full(internal_indices.shape, -1, dtype=np.int64)
        original_indices[valid] = self._insertion_order[internal_indices[valid]]
        return np.asarray(distances, dtype=np.float32), original_indices

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "formal_backend": True,
            "seed": self.seed,
            "threads": self.threads,
            "hnsw_m": self.hnsw_m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "add_batch_size": self.add_batch_size,
            "insertion_order": "numpy_seeded_permutation",
        }


def _as_float32_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError(f"{name} must be a non-empty matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _validate_git_revision(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git_revision must be a full lowercase Git object id.")
    return value


def _array_file_metadata(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "dtype": str(array.dtype),
        "shape": [int(value) for value in array.shape],
        "size_bytes": int(path.stat().st_size),
    }


def load_frozen_state_bank(
    bank_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_training_split_sha256: str | None = None,
) -> FrozenStateBank:
    """Load only a manifest-sealed training bank (legacy raw arrays fail)."""

    return load_frozen_training_state_bank(
        bank_dir,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_training_split_sha256=expected_training_split_sha256,
    )


def _validate_build_budget(
    bank: FrozenStateBank,
    *,
    neighbors: int,
    search_depth: int,
    query_chunk_size: int,
) -> None:
    if neighbors <= 0:
        raise ValueError("neighbors must be positive.")
    if search_depth <= neighbors or search_depth > bank.size:
        raise ValueError(
            "search_depth must be greater than neighbors and no larger than bank size."
        )
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    _, episode_counts = np.unique(bank.episode_ids, return_counts=True)
    if episode_counts.size < 2 or np.any(bank.size - episode_counts < neighbors):
        raise ValueError(
            "Every anchor episode must have at least neighbors transitions outside it."
        )


def _filtered_neighbors(
    bank: FrozenStateBank,
    *,
    anchor_index: int,
    candidate_indices: np.ndarray,
    neighbors: int,
    search_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    seen: set[int] = set()
    eligible: list[int] = []
    anchor_episode = int(bank.episode_ids[anchor_index])
    for raw_index in np.asarray(candidate_indices).reshape(-1):
        index = int(raw_index)
        if index < 0 or index >= bank.size or index in seen:
            continue
        seen.add(index)
        if index == anchor_index or int(bank.episode_ids[index]) == anchor_episode:
            continue
        eligible.append(index)
    if len(eligible) < neighbors:
        raise RuntimeError(
            f"Search depth {search_depth} yielded only {len(eligible)} eligible "
            f"neighbors for global row {int(bank.global_rows[anchor_index])} "
            f"after excluding self and episode {anchor_episode}; increase "
            "search_depth and rebuild."
        )

    eligible_array = np.asarray(eligible, dtype=np.int64)
    anchor = np.asarray(bank.latents[anchor_index], dtype=np.float32)
    candidates = np.asarray(bank.latents[eligible_array], dtype=np.float32)
    differences = candidates - anchor[None, :]
    exact_distances = np.einsum("nd,nd->n", differences, differences)
    order = np.lexsort((eligible_array, exact_distances))[:neighbors]
    selected_indices = eligible_array[order]
    selected_distances = np.asarray(exact_distances[order], dtype=np.float32)
    if not np.isfinite(selected_distances).all() or np.any(selected_distances < 0):
        raise RuntimeError("Filtered squared-L2 neighbor distances are invalid.")
    return selected_indices, selected_distances


def _save_array(path: Path, array: np.ndarray) -> np.ndarray:
    np.save(path, np.asarray(array), allow_pickle=False)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def build_state_neighbor_index(
    *,
    bank_dir: str | Path,
    output_dir: str | Path,
    pretrained_world_model_sha256: str,
    training_split_sha256: str,
    git_revision: str,
    neighbors: int,
    search_depth: int,
    query_chunk_size: int,
    backend: NeighborSearchBackend,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    """Build one immutable training-only state-neighbor graph artifact."""

    checkpoint_hash = _validate_sha256(
        pretrained_world_model_sha256,
        name="pretrained_world_model_sha256",
    )
    split_hash = _validate_sha256(
        training_split_sha256,
        name="training_split_sha256",
    )
    revision = _validate_git_revision(git_revision)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if threads != 1:
        raise ValueError("Deterministic neighbor construction requires threads=1.")
    bank = load_frozen_state_bank(
        bank_dir,
        expected_checkpoint_sha256=checkpoint_hash,
        expected_training_split_sha256=split_hash,
    )
    bank.assert_immutable()
    _validate_build_budget(
        bank,
        neighbors=neighbors,
        search_depth=search_depth,
        query_chunk_size=query_chunk_size,
    )

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite neighbor index: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    vectors = np.asarray(bank.latents, dtype=np.float32)
    backend.build(vectors)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent))
    )
    try:
        output_arrays: dict[str, np.ndarray] = {
            "global_rows": _save_array(staging / "global_rows.npy", bank.global_rows),
            "episode_ids": _save_array(
                staging / "episode_ids.npy", bank.episode_ids
            ),
            "actions": _save_array(staging / "actions.npy", bank.actions),
        }
        neighbor_dtype = np.int32 if bank.size <= np.iinfo(np.int32).max else np.int64
        neighbor_indices = np.lib.format.open_memmap(
            staging / "neighbor_indices.npy",
            mode="w+",
            dtype=neighbor_dtype,
            shape=(bank.size, int(neighbors)),
        )
        neighbor_distances = np.lib.format.open_memmap(
            staging / "neighbor_distances.npy",
            mode="w+",
            dtype=np.float32,
            shape=(bank.size, int(neighbors)),
        )
        for start in range(0, bank.size, int(query_chunk_size)):
            end = min(bank.size, start + int(query_chunk_size))
            _, candidate_indices = backend.search(vectors[start:end], search_depth)
            expected_shape = (end - start, int(search_depth))
            if candidate_indices.shape != expected_shape:
                raise RuntimeError(
                    "Neighbor backend returned indices with shape "
                    f"{candidate_indices.shape}, expected {expected_shape}."
                )
            for local_index in range(end - start):
                anchor_index = start + local_index
                selected_indices, selected_distances = _filtered_neighbors(
                    bank,
                    anchor_index=anchor_index,
                    candidate_indices=candidate_indices[local_index],
                    neighbors=neighbors,
                    search_depth=search_depth,
                )
                neighbor_indices[anchor_index] = selected_indices
                neighbor_distances[anchor_index] = selected_distances
        neighbor_indices.flush()
        neighbor_distances.flush()
        output_arrays["neighbor_indices"] = neighbor_indices
        output_arrays["neighbor_distances"] = neighbor_distances

        files = {
            name: _array_file_metadata(staging / f"{name}.npy", array)
            for name, array in output_arrays.items()
        }
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "method": INDEX_METHOD,
            "pretrained_world_model_sha256": checkpoint_hash,
            "training_split_sha256": split_hash,
            "tdwm_git_revision": revision,
            "metric": "squared_l2",
            "retrieval_key": "frozen_current_state_latent",
            "split": "training_only",
            "exclude_exact_transition": True,
            "exclude_same_episode": True,
            "row_count": bank.size,
            "latent_dim": int(bank.latents.shape[1]),
            "action_block_dim": int(bank.actions.shape[1]),
            "neighbors_per_anchor": int(neighbors),
            "construction": {
                "offline_only": True,
                "query_chunk_size": int(query_chunk_size),
                "search_depth": int(search_depth),
                "seed": int(seed),
                "threads": int(threads),
                "backend": backend.metadata(),
            },
            "source_bank": {
                "path": str(bank.root),
                "manifest_path": str(bank.manifest_path),
                "manifest_sha256": bank.manifest_sha256,
                "pretrained_world_model_sha256": bank.manifest[
                    "pretrained_world_model_sha256"
                ],
                "training_split_sha256": bank.manifest[
                    "training_split_sha256"
                ],
                "latent_store_manifest_sha256": bank.manifest[
                    "latent_store_manifest_sha256"
                ],
                "files": bank.files,
            },
            "libraries": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                backend.name: backend.version,
            },
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        temporary_manifest = staging / "manifest.json.tmp"
        with temporary_manifest.open("w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_manifest.replace(manifest_path)
        bank.assert_immutable()
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BANK_FILES",
    "ExactNumpySearchBackend",
    "FaissHNSWSearchBackend",
    "FrozenStateBank",
    "NeighborSearchBackend",
    "build_state_neighbor_index",
    "load_frozen_state_bank",
]
