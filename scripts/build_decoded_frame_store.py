#!/usr/bin/env python3
"""Build a row-aligned uint8 decoded-frame store from a JPEG Lance table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from tdwm.training.decoded_frame_store import (
    CANONICAL_EPISODE_INDEX_DTYPE,
    FORMAT_NAME,
    ROW_MAPPING,
    SCHEMA_VERSION,
)

DEFAULT_ROW_COUNT = 2_010_000
DEFAULT_FRAME_SHAPE = (3, 224, 224)


@dataclass
class LanceSourceScan:
    episode_lengths: np.ndarray
    episode_offsets: np.ndarray
    episode_jpeg_payload_bytes: list[int]

    @property
    def row_count(self) -> int:
        return sum(int(value) for value in self.episode_lengths)

    @property
    def episode_count(self) -> int:
        return int(self.episode_lengths.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode every Lance pixels row exactly once into one row-aligned "
            "NCHW uint8 binary file plus a JSON manifest."
        )
    )
    parser.add_argument("source", type=Path, help="Source .lance directory.")
    parser.add_argument("manifest", type=Path, help="Destination JSON manifest.")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Parallel torchvision JPEG decode workers (default: 12).",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Bounded in-flight decode batches (default: 2 * workers).",
    )
    parser.add_argument("--expected-row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--expected-height", type=int, default=224)
    parser.add_argument("--expected-width", type=int, default=224)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help=(
            "Scan source JPEG payloads and atomically enrich an existing "
            "manifest without rebuilding its binary file."
        ),
    )
    parser.add_argument(
        "--audit-batch-rows",
        type=int,
        default=4096,
        help="Lance scan batch size for --audit-existing (default: 4096).",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_int64_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype=CANONICAL_EPISODE_INDEX_DTYPE)
    if canonical.ndim != 1:
        raise ValueError("Canonical episode index arrays must be one-dimensional.")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _source_metadata(
    source_path: Path,
    scan: LanceSourceScan,
) -> dict[str, Any]:
    payload_bytes = [int(value) for value in scan.episode_jpeg_payload_bytes]
    if len(payload_bytes) != scan.episode_count or any(
        value <= 0 for value in payload_bytes
    ):
        raise RuntimeError(
            "The Lance scan must record positive JPEG payload bytes for every "
            "episode before source metadata is written."
        )
    result: dict[str, Any] = {
        "path": str(source_path),
        "format": "lance",
        "row_count": scan.row_count,
        "pixel_column": "pixels",
        "row_mapping": ROW_MAPPING,
        "episode_count": scan.episode_count,
        "episode_lengths_dtype": CANONICAL_EPISODE_INDEX_DTYPE,
        "episode_lengths_sha256": _canonical_int64_sha256(
            scan.episode_lengths
        ),
        "episode_offsets_dtype": CANONICAL_EPISODE_INDEX_DTYPE,
        "episode_offsets_sha256": _canonical_int64_sha256(scan.episode_offsets),
        "episode_jpeg_payload_bytes": payload_bytes,
        "jpeg_payload_bytes": sum(payload_bytes),
    }
    source_manifest = Path(f"{source_path}.manifest.json")
    if source_manifest.is_file():
        result["manifest_path"] = str(source_manifest.resolve())
        result["manifest_sha256"] = _sha256_file(source_manifest)
    return result


def _validate_existing_source_binding(
    existing_source: Any,
    *,
    source_path: Path,
    decoded_manifest_path: Path,
    row_count: int,
) -> str:
    """Prove an old manifest was already bound to this exact Lance source."""

    if not isinstance(existing_source, dict):
        raise ValueError("Existing manifest source must be a JSON object.")
    expected_fields = {
        "format": "lance",
        "pixel_column": "pixels",
        "row_mapping": ROW_MAPPING,
        "row_count": row_count,
    }
    for field, expected in expected_fields.items():
        if existing_source.get(field) != expected:
            raise ValueError(
                f"Existing manifest source.{field} is "
                f"{existing_source.get(field)!r}, expected {expected!r}."
            )

    recorded_source = existing_source.get("path")
    if not isinstance(recorded_source, str) or not recorded_source:
        raise ValueError("Existing manifest source.path must be a path string.")
    recorded_source_path = Path(recorded_source).expanduser()
    if not recorded_source_path.is_absolute():
        recorded_source_path = decoded_manifest_path.parent / recorded_source_path
    if recorded_source_path.resolve() != source_path:
        raise ValueError(
            "Existing manifest source.path does not identify the requested "
            "Lance directory."
        )

    expected_source_manifest = Path(f"{source_path}.manifest.json").resolve()
    if not expected_source_manifest.is_file():
        raise FileNotFoundError(
            f"Current Lance conversion manifest not found: {expected_source_manifest}"
        )
    recorded_manifest = existing_source.get("manifest_path")
    if not isinstance(recorded_manifest, str) or not recorded_manifest:
        raise ValueError(
            "Existing manifest source.manifest_path must be a path string."
        )
    recorded_manifest_path = Path(recorded_manifest).expanduser()
    if not recorded_manifest_path.is_absolute():
        recorded_manifest_path = decoded_manifest_path.parent / recorded_manifest_path
    recorded_manifest_path = recorded_manifest_path.resolve()
    if not recorded_manifest_path.is_file():
        raise FileNotFoundError(
            f"Recorded Lance conversion manifest not found: {recorded_manifest_path}"
        )
    if recorded_manifest_path != expected_source_manifest:
        raise ValueError(
            "Existing manifest source.manifest_path does not identify the current "
            "Lance conversion manifest."
        )

    recorded_sha256 = existing_source.get("manifest_sha256")
    if (
        not isinstance(recorded_sha256, str)
        or len(recorded_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recorded_sha256
        )
    ):
        raise ValueError(
            "Existing manifest source.manifest_sha256 must be a lowercase "
            "SHA-256 digest."
        )
    actual_sha256 = _sha256_file(expected_source_manifest)
    if actual_sha256 != recorded_sha256:
        raise ValueError(
            "Existing manifest source.manifest_sha256 does not match the current "
            "Lance conversion manifest."
        )
    return actual_sha256


def _load_lance_pixels(
    source_path: Path,
    batch_rows: int,
) -> tuple[LanceSourceScan, Iterator[tuple[int, list[Any]]]]:
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        str(source_path),
        format="lance",
        transform=None,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels"],
    )
    required = ("lengths", "offsets", "get_row_data")
    if any(not hasattr(dataset, name) for name in required):
        raise TypeError(
            "The loaded Lance dataset must expose lengths, offsets, and "
            "get_row_data."
        )
    raw_lengths = np.asarray(dataset.lengths)
    raw_offsets = np.asarray(dataset.offsets)
    if raw_lengths.dtype.kind not in ("i", "u"):
        raise ValueError("The Lance dataset episode lengths must be integers.")
    if raw_offsets.dtype.kind not in ("i", "u"):
        raise ValueError("The Lance dataset episode offsets must be integers.")
    int64_max = np.iinfo(np.int64).max
    if raw_lengths.dtype.kind == "u" and np.any(raw_lengths > int64_max):
        raise ValueError("The Lance dataset episode lengths exceed int64.")
    if raw_offsets.dtype.kind == "u" and np.any(raw_offsets > int64_max):
        raise ValueError("The Lance dataset episode offsets exceed int64.")
    lengths = np.asarray(
        raw_lengths,
        dtype=CANONICAL_EPISODE_INDEX_DTYPE,
    )
    offsets = np.asarray(
        raw_offsets,
        dtype=CANONICAL_EPISODE_INDEX_DTYPE,
    )
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("The Lance dataset has invalid episode lengths.")
    python_row_count = sum(int(value) for value in lengths)
    if python_row_count > int64_max:
        raise ValueError("The Lance dataset row count exceeds int64.")
    if offsets.ndim != 1 or offsets.size != lengths.size:
        raise ValueError(
            "The Lance dataset offsets must have one entry per episode."
        )
    expected_offsets = np.empty_like(lengths)
    expected_offsets[0] = 0
    if lengths.size > 1:
        expected_offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError(
            "The Lance dataset offsets must start at zero and exactly equal "
            "the cumulative preceding episode lengths."
        )
    row_count = python_row_count
    scan = LanceSourceScan(
        episode_lengths=lengths,
        episode_offsets=offsets,
        episode_jpeg_payload_bytes=[0] * int(lengths.size),
    )

    def batches() -> Iterator[tuple[int, list[Any]]]:
        episode = 0
        for start in range(0, row_count, batch_rows):
            stop = min(start + batch_rows, row_count)
            rows = list(range(start, stop))
            row_data = dataset.get_row_data(rows)
            if "pixels" not in row_data:
                raise KeyError("The Lance row data does not contain 'pixels'.")
            blobs = np.asarray(row_data["pixels"], dtype=object).reshape(-1).tolist()
            if len(blobs) != stop - start:
                raise RuntimeError(
                    f"Lance returned {len(blobs)} pixels for rows [{start}, {stop})."
                )
            for local_row, blob in enumerate(blobs):
                global_row = start + local_row
                while (
                    episode + 1 < scan.episode_count
                    and global_row >= int(offsets[episode + 1])
                ):
                    episode += 1
                episode_end = int(offsets[episode] + lengths[episode])
                if global_row < int(offsets[episode]) or global_row >= episode_end:
                    raise RuntimeError(
                        f"Lance row {global_row} is outside audited episode {episode}."
                    )
                try:
                    blob_bytes = memoryview(blob).nbytes
                except TypeError as error:
                    raise TypeError(
                        f"Lance pixels row {global_row} is not buffer-compatible."
                    ) from error
                scan.episode_jpeg_payload_bytes[episode] += int(blob_bytes)
            yield start, blobs

    return scan, batches()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_jpeg_rgb(blobs: Sequence[Any]):
    import torch
    from torchvision.io import ImageReadMode, decode_jpeg

    if not blobs:
        return torch.empty((0, *DEFAULT_FRAME_SHAPE), dtype=torch.uint8)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        encoded = [
            torch.frombuffer(
                blob if isinstance(blob, (bytes, bytearray)) else bytes(blob),
                dtype=torch.uint8,
            )
            for blob in blobs
        ]
    decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB)
    if isinstance(decoded, torch.Tensor):
        if len(blobs) != 1:
            raise RuntimeError(
                "torchvision.decode_jpeg returned one tensor for multiple inputs."
            )
        decoded = [decoded]
    return torch.stack(list(decoded), dim=0)


def _ordered_decode_batches(
    pixel_batches: Iterator[tuple[int, list[Any]]],
    *,
    workers: int,
    max_pending: int,
) -> Iterator[tuple[int, int, Any]]:
    """Decode concurrently while yielding completed tensors in source-row order."""

    if workers == 1:
        for start, blobs in pixel_batches:
            yield start, len(blobs), _decode_jpeg_rgb(blobs)
        return

    pending: deque[tuple[int, int, Future[Any]]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="decoded-frame-jpeg",
    ) as executor:
        for start, blobs in pixel_batches:
            future = executor.submit(_decode_jpeg_rgb, blobs)
            pending.append((start, len(blobs), future))
            if len(pending) >= max_pending:
                first_start, blob_count, first = pending.popleft()
                yield first_start, blob_count, first.result()
        while pending:
            first_start, blob_count, first = pending.popleft()
            yield first_start, blob_count, first.result()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = Path(f"{path}.partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    _fsync_directory(path.parent)


def build_decoded_frame_store(
    source_path: str | Path,
    manifest_path: str | Path,
    *,
    batch_rows: int = 256,
    workers: int = 12,
    max_pending: int | None = None,
    expected_row_count: int | None = DEFAULT_ROW_COUNT,
    expected_frame_shape: tuple[int, int, int] | None = DEFAULT_FRAME_SHAPE,
    report_every: int = 10_000,
) -> dict[str, Any]:
    """Decode the Lance pixel column in source-row order and publish atomically."""

    import torch
    import torchvision

    source = Path(source_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not source.is_dir() or source.suffix.lower() != ".lance":
        raise ValueError(f"Source must be an existing .lance directory: {source}")
    if manifest.suffix.lower() != ".json":
        raise ValueError("Decoded frame manifest path must end in '.json'.")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if max_pending is None:
        max_pending = 2 * workers
    if max_pending < workers:
        raise ValueError("max_pending must be at least workers.")
    if report_every <= 0:
        raise ValueError("report_every must be positive.")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    data_path = manifest.with_suffix(".bin")
    data_partial = Path(f"{data_path}.partial")
    manifest_partial = Path(f"{manifest}.partial")
    collisions = [
        path
        for path in (data_path, data_partial, manifest, manifest_partial)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite decoded frame output: "
            + ", ".join(str(path) for path in collisions)
        )

    source_scan, pixel_batches = _load_lance_pixels(source, batch_rows)
    row_count = source_scan.row_count
    if expected_row_count is not None and row_count != expected_row_count:
        raise ValueError(
            f"Lance row count is {row_count}, expected {expected_row_count}."
        )

    digest = hashlib.sha256()
    written_rows = 0
    written_bytes = 0
    frame_shape: tuple[int, int, int] | None = None
    started = time.monotonic()
    next_report = report_every
    with data_partial.open("xb") as stream:
        decoded_batches = _ordered_decode_batches(
            pixel_batches,
            workers=workers,
            max_pending=max_pending,
        )
        for start, blob_count, decoded in decoded_batches:
            if start != written_rows:
                raise RuntimeError(
                    f"Non-contiguous Lance rows: got {start}, expected {written_rows}."
                )
            if decoded.dtype != torch.uint8 or decoded.ndim != 4:
                raise RuntimeError(
                    "torchvision.decode_jpeg must return NCHW torch.uint8 frames."
                )
            batch_frame_shape = tuple(int(value) for value in decoded.shape[1:])
            if frame_shape is None:
                frame_shape = batch_frame_shape
                if (
                    expected_frame_shape is not None
                    and frame_shape != expected_frame_shape
                ):
                    raise ValueError(
                        f"Decoded frame shape is {frame_shape}, "
                        f"expected {expected_frame_shape}."
                    )
            elif batch_frame_shape != frame_shape:
                raise ValueError(
                    f"Decoded frame shape changed from {frame_shape} "
                    f"to {batch_frame_shape} at row {start}."
                )

            array = decoded.contiguous().cpu().numpy()
            payload = memoryview(array).cast("B")
            digest.update(payload)
            bytes_this_batch = stream.write(payload)
            if bytes_this_batch != len(payload):
                raise OSError(
                    f"Short write: wrote {bytes_this_batch} of {len(payload)} bytes."
                )
            rows_this_batch = int(decoded.shape[0])
            if rows_this_batch != blob_count:
                raise RuntimeError(
                    f"Decoded {rows_this_batch} frames from {blob_count} JPEG rows."
                )
            written_rows += rows_this_batch
            written_bytes += bytes_this_batch

            if written_rows >= next_report or written_rows == row_count:
                elapsed = time.monotonic() - started
                rate = written_rows / elapsed if elapsed else 0.0
                print(
                    f"Decoded {written_rows:,}/{row_count:,} rows "
                    f"({rate:,.1f} rows/s, {written_bytes / 1024**3:.2f} GiB)",
                    flush=True,
                )
                while next_report <= written_rows:
                    next_report += report_every
        stream.flush()
        os.fsync(stream.fileno())

    if written_rows != row_count or frame_shape is None:
        raise RuntimeError(
            f"Decoded {written_rows} rows, expected exactly {row_count}."
        )
    shape = (row_count, *frame_shape)
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if written_bytes != expected_bytes or data_partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            "Decoded binary size does not match its NCHW uint8 shape: "
            f"wrote {written_bytes}, expected {expected_bytes}."
        )

    elapsed = time.monotonic() - started
    data_sha256 = digest.hexdigest()
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "row_count": row_count,
        "shape": list(shape),
        "dtype": "uint8",
        "layout": "NCHW",
        "data_file": data_path.name,
        "size_bytes": written_bytes,
        "sha256": data_sha256,
        "source": _source_metadata(source, source_scan),
        "decoder": {
            "api": "torchvision.io.decode_jpeg",
            "mode": "RGB",
        },
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "source_pixel_verification": {
            "method": "decoded_during_build",
            "row_count": row_count,
            "decoded_sha256": data_sha256,
            "data_sha256": data_sha256,
            "matches_data_sha256": True,
            "decoder": {
                "api": "torchvision.io.decode_jpeg",
                "mode": "RGB",
            },
            "completed_at_utc": completed_at_utc,
        },
        "created_at_utc": completed_at_utc,
        "build_elapsed_seconds": elapsed,
        "batch_rows": batch_rows,
        "workers": workers,
        "max_pending": max_pending,
    }

    # Publish data first and the manifest last.  A visible manifest therefore
    # never points at a missing final binary file.
    data_partial.replace(data_path)
    _fsync_directory(data_path.parent)
    _atomic_json(manifest, result)
    return result


def audit_existing_decoded_frame_store(
    source_path: str | Path,
    manifest_path: str | Path,
    *,
    batch_rows: int = 4096,
    expected_row_count: int | None = DEFAULT_ROW_COUNT,
    report_every: int = 100_000,
) -> dict[str, Any]:
    """Add source episode seals to an existing store without rewriting frames."""

    source = Path(source_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not source.is_dir() or source.suffix.lower() != ".lance":
        raise ValueError(f"Source must be an existing .lance directory: {source}")
    if not manifest.is_file() or manifest.suffix.lower() != ".json":
        raise ValueError(f"Manifest must be an existing JSON file: {manifest}")
    if batch_rows <= 0 or report_every <= 0:
        raise ValueError("batch_rows and report_every must be positive.")
    partial = Path(f"{manifest}.partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to overwrite partial manifest: {partial}")

    with manifest.open(encoding="utf-8") as stream:
        existing = json.load(stream)
    if not isinstance(existing, dict):
        raise ValueError("Decoded frame manifest must contain a JSON object.")
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Existing manifest has an unsupported schema_version.")
    if existing.get("format") != FORMAT_NAME:
        raise ValueError("Existing manifest has an unsupported format.")
    if existing.get("dtype") != "uint8" or existing.get("layout") != "NCHW":
        raise ValueError("Existing manifest must describe NCHW uint8 frames.")
    shape = existing.get("shape")
    row_count = existing.get("row_count")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or row_count != shape[0]
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in shape
        )
    ):
        raise ValueError("Existing manifest has an invalid row_count or shape.")
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if existing.get("size_bytes") != expected_bytes:
        raise ValueError("Existing manifest size_bytes does not match its shape.")
    data_file = existing.get("data_file")
    if not isinstance(data_file, str) or not data_file:
        raise ValueError("Existing manifest has an invalid data_file.")
    data_path = Path(data_file).expanduser()
    if not data_path.is_absolute():
        data_path = manifest.parent / data_path
    if not data_path.is_file() or data_path.stat().st_size != expected_bytes:
        raise ValueError("Existing decoded binary is missing or has the wrong size.")
    data_sha256 = existing.get("sha256")
    if (
        not isinstance(data_sha256, str)
        or len(data_sha256) != 64
        or any(character not in "0123456789abcdef" for character in data_sha256)
    ):
        raise ValueError("Existing manifest has an invalid data SHA-256.")

    bound_source_manifest_sha256 = _validate_existing_source_binding(
        existing.get("source"),
        source_path=source,
        decoded_manifest_path=manifest,
        row_count=row_count,
    )

    source_scan, pixel_batches = _load_lance_pixels(source, batch_rows)
    if expected_row_count is not None and source_scan.row_count != expected_row_count:
        raise ValueError(
            f"Lance row count is {source_scan.row_count}, "
            f"expected {expected_row_count}."
        )
    if source_scan.row_count != row_count:
        raise ValueError(
            f"Lance row count is {source_scan.row_count}, "
            f"but the decoded manifest records {row_count}."
        )

    started = time.monotonic()
    scanned_rows = 0
    next_report = report_every
    for start, blobs in pixel_batches:
        if start != scanned_rows:
            raise RuntimeError(
                f"Non-contiguous Lance rows: got {start}, expected {scanned_rows}."
            )
        scanned_rows += len(blobs)
        if scanned_rows >= next_report or scanned_rows == source_scan.row_count:
            elapsed = time.monotonic() - started
            rate = scanned_rows / elapsed if elapsed else 0.0
            print(
                f"Audited {scanned_rows:,}/{source_scan.row_count:,} JPEG rows "
                f"({rate:,.1f} rows/s)",
                flush=True,
            )
            while next_report <= scanned_rows:
                next_report += report_every
    if scanned_rows != source_scan.row_count:
        raise RuntimeError(
            f"Audited {scanned_rows} rows, expected {source_scan.row_count}."
        )

    audited_source = _source_metadata(source, source_scan)
    if audited_source.get("manifest_sha256") != bound_source_manifest_sha256:
        raise RuntimeError(
            "The Lance conversion manifest changed while its JPEG rows were "
            "being audited."
        )
    result = dict(existing)
    # A payload-size-only audit cannot re-establish decoded pixel equality.
    # Always invalidate any earlier proof; the external full decode verifier
    # must write a fresh source_pixel_verification before Store can open it.
    result.pop("source_pixel_verification", None)
    result["source"] = audited_source
    result["source_audit"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "batch_rows": batch_rows,
        "decoded_binary_rebuilt": False,
    }
    _atomic_json(manifest, result)
    return result


def main() -> None:
    args = parse_args()
    if args.audit_existing:
        result = audit_existing_decoded_frame_store(
            args.source,
            args.manifest,
            batch_rows=args.audit_batch_rows,
            expected_row_count=args.expected_row_count,
            report_every=args.report_every,
        )
    else:
        result = build_decoded_frame_store(
            args.source,
            args.manifest,
            batch_rows=args.batch_rows,
            workers=args.workers,
            max_pending=args.max_pending,
            expected_row_count=args.expected_row_count,
            expected_frame_shape=(3, args.expected_height, args.expected_width),
            report_every=args.report_every,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
