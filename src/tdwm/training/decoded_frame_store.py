"""Read-only access to a row-aligned decoded image store.

The on-disk format deliberately stays simple: one C-contiguous NCHW uint8
binary file and one JSON manifest.  The first axis is the source dataset row
axis, so frame ``i`` in the binary file is frame ``i`` in the Lance table.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mmap
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FORMAT_NAME = "tdwm-decoded-frame-store"
SCHEMA_VERSION = 1
ROW_MAPPING = "binary frame i is decoded from Lance pixels row i"
CANONICAL_EPISODE_INDEX_DTYPE = "<i8"


class DecodedFrameStore:
    """Lazy, read-only memory map for decoded NCHW uint8 frames."""

    def __init__(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Decoded frame manifest not found: {path}")

        manifest_bytes = path.read_bytes()
        self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("Decoded frame manifest must contain a JSON object.")

        self._manifest_path = path
        self._metadata = self._validate_manifest(path, manifest)
        data_file = self._metadata["data_file"]
        data_path = Path(data_file).expanduser()
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        self._data_path = data_path.resolve()
        if not self._data_path.is_file():
            raise FileNotFoundError(
                f"Decoded frame binary not found: {self._data_path}"
            )

        expected_bytes = int(self._metadata["size_bytes"])
        data_stat = self._data_path.stat()
        actual_bytes = data_stat.st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                "Decoded frame binary size mismatch: "
                f"found {actual_bytes} bytes, expected {expected_bytes}."
            )
        self._memmap: np.memmap | None = None
        self._sha256_verified = False
        self._page_cache_warmed = False
        self._data_stat = self._stat_identity(data_stat)

    @classmethod
    def from_manifest(cls, path: str | Path) -> "DecodedFrameStore":
        """Open manifest metadata without mapping the binary frame file yet."""

        return cls(path)

    @staticmethod
    def _validate_manifest(
        manifest_path: Path,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported decoded frame manifest schema_version: "
                f"{manifest.get('schema_version')!r}."
            )
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(
                "Decoded frame manifest format must be "
                f"{FORMAT_NAME!r}, found {manifest.get('format')!r}."
            )
        if manifest.get("layout") != "NCHW":
            raise ValueError("Decoded frame manifest layout must be 'NCHW'.")
        if manifest.get("dtype") != "uint8":
            raise ValueError("Decoded frame manifest dtype must be 'uint8'.")

        raw_shape = manifest.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in raw_shape
            )
        ):
            raise ValueError(
                "Decoded frame manifest shape must be four positive integers."
            )
        shape = tuple(raw_shape)
        row_count = manifest.get("row_count")
        if row_count != shape[0]:
            raise ValueError(
                "Decoded frame manifest row_count must equal shape[0]: "
                f"found {row_count!r} and {shape[0]}."
            )

        expected_bytes = int(np.prod(shape, dtype=np.int64))
        if manifest.get("size_bytes") != expected_bytes:
            raise ValueError(
                "Decoded frame manifest size_bytes does not match its uint8 shape: "
                f"found {manifest.get('size_bytes')!r}, expected {expected_bytes}."
            )
        data_file = manifest.get("data_file")
        if not isinstance(data_file, str) or not data_file:
            raise ValueError("Decoded frame manifest data_file must be a path string.")
        sha256 = manifest.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                "Decoded frame manifest sha256 must be a lowercase SHA-256 digest."
            )

        pixel_verification = manifest.get("source_pixel_verification")
        if not isinstance(pixel_verification, dict):
            raise ValueError(
                "Decoded frame manifest source_pixel_verification must be a "
                "JSON object."
            )
        method = pixel_verification.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(
                "Decoded frame manifest source_pixel_verification.method must "
                "be a non-empty string."
            )
        if pixel_verification.get("row_count") != row_count:
            raise ValueError(
                "Decoded frame manifest source_pixel_verification.row_count "
                "must equal row_count."
            )
        for digest_key in ("decoded_sha256", "data_sha256"):
            digest = pixel_verification.get(digest_key)
            DecodedFrameStore._validate_sha256(
                digest,
                f"source_pixel_verification.{digest_key}",
            )
            if digest != sha256:
                raise ValueError(
                    "Decoded frame manifest source_pixel_verification."
                    f"{digest_key} must equal the top-level sha256."
                )
        if pixel_verification.get("matches_data_sha256") is not True:
            raise ValueError(
                "Decoded frame manifest source_pixel_verification."
                "matches_data_sha256 must be true."
            )
        verification_decoder = pixel_verification.get("decoder")
        if not isinstance(verification_decoder, dict):
            raise ValueError(
                "Decoded frame manifest source_pixel_verification.decoder must "
                "be a JSON object."
            )
        if verification_decoder.get("api") != "torchvision.io.decode_jpeg":
            raise ValueError(
                "Decoded frame source pixel verification decoder.api must be "
                "'torchvision.io.decode_jpeg'."
            )
        if verification_decoder.get("mode") != "RGB":
            raise ValueError(
                "Decoded frame source pixel verification decoder.mode must be "
                "'RGB'."
            )
        completed_at_utc = pixel_verification.get("completed_at_utc")
        if not isinstance(completed_at_utc, str) or not completed_at_utc.strip():
            raise ValueError(
                "Decoded frame manifest source_pixel_verification."
                "completed_at_utc must be a non-empty string."
            )

        source = manifest.get("source")
        if not isinstance(source, dict):
            raise ValueError("Decoded frame manifest source must be a JSON object.")
        if source.get("row_count") != row_count:
            raise ValueError(
                "Decoded frame manifest source.row_count must equal row_count."
            )
        if source.get("format") != "lance":
            raise ValueError("Decoded frame manifest source.format must be 'lance'.")
        if source.get("pixel_column") != "pixels":
            raise ValueError(
                "Decoded frame manifest source.pixel_column must be 'pixels'."
            )
        if source.get("row_mapping") != ROW_MAPPING:
            raise ValueError(
                "Decoded frame manifest source.row_mapping does not match the "
                "row-aligned store protocol."
            )
        episode_count = source.get("episode_count")
        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
            or episode_count <= 0
            or episode_count > row_count
        ):
            raise ValueError(
                "Decoded frame manifest source.episode_count must be a positive "
                "integer no greater than row_count."
            )
        for prefix in ("episode_lengths", "episode_offsets"):
            if source.get(f"{prefix}_dtype") != CANONICAL_EPISODE_INDEX_DTYPE:
                raise ValueError(
                    f"Decoded frame manifest source.{prefix}_dtype must be "
                    f"{CANONICAL_EPISODE_INDEX_DTYPE!r}."
                )
            DecodedFrameStore._validate_sha256(
                source.get(f"{prefix}_sha256"),
                f"source.{prefix}_sha256",
            )
        payload_bytes = source.get("episode_jpeg_payload_bytes")
        if not isinstance(payload_bytes, list) or len(payload_bytes) != episode_count:
            raise ValueError(
                "Decoded frame manifest source.episode_jpeg_payload_bytes must "
                "contain one value per episode."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in payload_bytes
        ):
            raise ValueError(
                "Decoded frame manifest source.episode_jpeg_payload_bytes must "
                "contain positive integers."
            )
        jpeg_payload_bytes = source.get("jpeg_payload_bytes")
        if jpeg_payload_bytes != sum(payload_bytes):
            raise ValueError(
                "Decoded frame manifest source.jpeg_payload_bytes must equal the "
                "sum of episode_jpeg_payload_bytes."
            )
        for version_key in ("torch_version", "torchvision_version"):
            if not isinstance(manifest.get(version_key), str) or not manifest[version_key]:
                raise ValueError(
                    f"Decoded frame manifest {version_key} must be a non-empty string."
                )

        validated = dict(manifest)
        validated["shape"] = list(shape)
        return validated

    @staticmethod
    def _validate_sha256(value: Any, label: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"Decoded frame manifest {label} must be a lowercase SHA-256 digest."
            )

    @staticmethod
    def _stat_identity(stat_result: Any) -> dict[str, int]:
        return {
            "dev": int(stat_result.st_dev),
            "inode": int(stat_result.st_ino),
            "size": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
        }

    def _assert_data_identity(self) -> None:
        current = self._stat_identity(self._data_path.stat())
        if current != self._data_stat:
            raise RuntimeError(
                "Decoded frame binary identity changed after the manifest was "
                f"opened: expected {self._data_stat}, found {current}."
            )

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def data_path(self) -> Path:
        return self._data_path

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def data_stat(self) -> dict[str, int]:
        return dict(self._data_stat)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a defensive copy of the complete manifest."""

        return copy.deepcopy(self._metadata)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Complete store shape ``(rows, channels, height, width)``."""

        return tuple(self._metadata["shape"])

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.shape[1:]

    @property
    def row_count(self) -> int:
        return int(self._metadata["row_count"])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._metadata["dtype"])

    @property
    def sha256(self) -> str:
        return str(self._metadata["sha256"])

    @property
    def source(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata["source"])

    @property
    def episode_count(self) -> int:
        return int(self._metadata["source"]["episode_count"])

    @property
    def episode_lengths_sha256(self) -> str:
        return str(self._metadata["source"]["episode_lengths_sha256"])

    @property
    def episode_offsets_sha256(self) -> str:
        return str(self._metadata["source"]["episode_offsets_sha256"])

    @property
    def episode_jpeg_payload_bytes(self) -> tuple[int, ...]:
        return tuple(self._metadata["source"]["episode_jpeg_payload_bytes"])

    @property
    def jpeg_payload_bytes(self) -> int:
        return int(self._metadata["source"]["jpeg_payload_bytes"])

    @property
    def sha256_verified(self) -> bool:
        """Whether this process has hashed and matched the complete binary."""

        return self._sha256_verified

    @property
    def page_cache_warmed(self) -> bool:
        """Whether this process completed a sequential touch of every page."""

        return self._page_cache_warmed

    def _array(self) -> np.memmap:
        self._assert_data_identity()
        if self._memmap is None:
            self._memmap = np.memmap(
                self._data_path,
                dtype=self.dtype,
                mode="r",
                shape=self.shape,
                order="C",
            )
        return self._memmap

    def take(self, rows: Sequence[int] | np.ndarray | Any):
        """Copy selected rows into a CPU ``torch.uint8`` NCHW tensor.

        Duplicate rows and caller order are preserved.  Negative indices are
        rejected so a source row ID can never silently refer to another frame.
        """

        import torch

        if isinstance(rows, torch.Tensor):
            if rows.device.type != "cpu":
                rows = rows.detach().cpu()
            indices = rows.detach().numpy()
        else:
            indices = np.asarray(rows)
        if indices.ndim != 1:
            raise ValueError("Decoded frame row indices must be one-dimensional.")
        if indices.size == 0:
            return torch.empty((0, *self.frame_shape), dtype=torch.uint8)
        if indices.dtype.kind not in ("i", "u"):
            raise TypeError("Decoded frame row indices must be integers.")
        indices = indices.astype(np.int64, copy=False)
        if indices.size and (indices.min() < 0 or indices.max() >= self.row_count):
            raise IndexError(
                f"Decoded frame row is outside [0, {self.row_count})."
            )
        # np.take always creates a writable, C-contiguous result.  The tensor
        # therefore cannot mutate the read-only memmap and is safe for PyTorch.
        selected = np.take(self._array(), indices, axis=0)
        return torch.from_numpy(selected)

    def preload(self, *, verify_sha256: bool = False) -> "DecodedFrameStore":
        """Bring all pages into RAM, optionally verifying bytes while reading.

        ``verify_sha256=True`` is the recommended one-time startup path.  It
        reads every byte sequentially, both warming the page cache and checking
        the manifest digest.  The faster default touches one byte per OS page.
        """

        if verify_sha256:
            self._sha256_verified = False
        self._page_cache_warmed = False
        array = self._array()
        flat = array.reshape(-1)
        if flat.size == 0:
            return self

        mapped = getattr(array, "_mmap", None)
        if mapped is not None and hasattr(mapped, "madvise"):
            try:
                mapped.madvise(mmap.MADV_WILLNEED)
            except (AttributeError, OSError, ValueError):
                pass

        chunk_bytes = 1024**3
        try:
            if verify_sha256:
                digest = hashlib.sha256()
                for start in range(0, flat.size, chunk_bytes):
                    stop = min(start + chunk_bytes, flat.size)
                    digest.update(memoryview(flat[start:stop]))
                actual_sha256 = digest.hexdigest()
                self._page_cache_warmed = True
                if actual_sha256 != self.sha256:
                    raise ValueError(
                        "Decoded frame binary SHA-256 mismatch: "
                        f"found {actual_sha256}, expected {self.sha256}."
                    )
                self._sha256_verified = True
            else:
                page_bytes = getattr(mmap, "PAGESIZE", 4096)
                accumulator = 0
                for start in range(0, flat.size, chunk_bytes):
                    stop = min(start + chunk_bytes, flat.size)
                    touched = flat[start:stop:page_bytes]
                    accumulator ^= int(touched.sum(dtype=np.uint64))
                # Include the final byte when the file is not a page multiple.
                accumulator ^= int(flat[-1])
                self._preload_accumulator = accumulator
                self._page_cache_warmed = True
        finally:
            if mapped is not None and hasattr(mapped, "madvise"):
                try:
                    mapped.madvise(mmap.MADV_RANDOM)
                except (AttributeError, OSError, ValueError):
                    pass
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Serialize paths and identity seals, never the mapped frame array."""

        return {
            "manifest_path": str(self._manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "data_sha256": self.sha256,
            "data_stat": self.data_stat,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected_keys = {
            "manifest_path",
            "manifest_sha256",
            "data_sha256",
            "data_stat",
        }
        if set(state) != expected_keys:
            raise ValueError("Decoded frame pickle identity state is malformed.")
        restored = type(self).from_manifest(state["manifest_path"])
        if restored.manifest_sha256 != state["manifest_sha256"]:
            raise RuntimeError("Decoded frame manifest changed during spawn.")
        if restored.sha256 != state["data_sha256"]:
            raise RuntimeError("Decoded frame data SHA-256 changed during spawn.")
        if restored.data_stat != state["data_stat"]:
            raise RuntimeError("Decoded frame binary changed during spawn.")
        self.__dict__.update(restored.__dict__)
