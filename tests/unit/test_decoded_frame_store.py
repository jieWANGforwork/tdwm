import hashlib
import json
import pickle
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tdwm.training.decoded_frame_store import (
    CANONICAL_EPISODE_INDEX_DTYPE,
    FORMAT_NAME,
    ROW_MAPPING,
    SCHEMA_VERSION,
    DecodedFrameStore,
)

try:
    import torch
    import torchvision
except ImportError:
    torch = None
    torchvision = None


class DecodedFrameStoreTest(unittest.TestCase):
    def _write_store(
        self,
        root: Path,
        *,
        shape: tuple[int, int, int, int] = (5, 3, 2, 4),
    ) -> tuple[Path, np.ndarray]:
        frames = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        if shape[0] == 1:
            episode_lengths = np.asarray([1], dtype=CANONICAL_EPISODE_INDEX_DTYPE)
            episode_offsets = np.asarray([0], dtype=CANONICAL_EPISODE_INDEX_DTYPE)
        else:
            episode_lengths = np.asarray(
                [shape[0] - 1, 1],
                dtype=CANONICAL_EPISODE_INDEX_DTYPE,
            )
            episode_offsets = np.asarray(
                [0, shape[0] - 1],
                dtype=CANONICAL_EPISODE_INDEX_DTYPE,
            )
        payload_bytes = [11 + index for index in range(episode_lengths.size)]
        data_path = root / "frames.bin"
        data_path.write_bytes(frames.tobytes())
        data_sha256 = hashlib.sha256(frames.tobytes()).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_NAME,
            "row_count": shape[0],
            "shape": list(shape),
            "dtype": "uint8",
            "layout": "NCHW",
            "data_file": data_path.name,
            "size_bytes": frames.nbytes,
            "sha256": data_sha256,
            "source": {
                "path": "/fixture/source.lance",
                "format": "lance",
                "row_count": shape[0],
                "pixel_column": "pixels",
                "row_mapping": ROW_MAPPING,
                "episode_count": int(episode_lengths.size),
                "episode_lengths_dtype": CANONICAL_EPISODE_INDEX_DTYPE,
                "episode_lengths_sha256": hashlib.sha256(
                    episode_lengths.tobytes()
                ).hexdigest(),
                "episode_offsets_dtype": CANONICAL_EPISODE_INDEX_DTYPE,
                "episode_offsets_sha256": hashlib.sha256(
                    episode_offsets.tobytes()
                ).hexdigest(),
                "episode_jpeg_payload_bytes": payload_bytes,
                "jpeg_payload_bytes": sum(payload_bytes),
            },
            "decoder": {
                "api": "torchvision.io.decode_jpeg",
                "mode": "RGB",
            },
            "torch_version": "fixture-torch",
            "torchvision_version": "fixture-torchvision",
            "source_pixel_verification": {
                "method": "fixture_verification",
                "row_count": shape[0],
                "decoded_sha256": data_sha256,
                "data_sha256": data_sha256,
                "matches_data_sha256": True,
                "decoder": {
                    "api": "torchvision.io.decode_jpeg",
                    "mode": "RGB",
                },
                "completed_at_utc": "2026-08-27T00:00:00+00:00",
            },
        }
        manifest_path = root / "frames.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, frames

    def test_manifest_open_is_lazy_and_resolves_relative_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, frames = self._write_store(Path(temporary))

            store = DecodedFrameStore.from_manifest(manifest_path)

            self.assertIsNone(store._memmap)
            self.assertEqual(store.manifest_path, manifest_path.resolve())
            self.assertEqual(
                store.data_path,
                (manifest_path.parent / "frames.bin").resolve(),
            )
            self.assertEqual(store.shape, frames.shape)
            self.assertEqual(store.frame_shape, frames.shape[1:])
            self.assertEqual(store.row_count, frames.shape[0])
            self.assertEqual(store.dtype, np.dtype(np.uint8))
            self.assertEqual(store.source["row_count"], frames.shape[0])
            self.assertEqual(
                store.manifest_sha256,
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(store.episode_count, 2)
            self.assertEqual(store.episode_jpeg_payload_bytes, (11, 12))
            self.assertEqual(store.jpeg_payload_bytes, 23)

    def test_preload_sequentially_opens_mapping_and_returns_self(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path)

            result = store.preload()

            self.assertIs(result, store)
            self.assertIsInstance(store._memmap, np.memmap)
            self.assertFalse(store.sha256_verified)
            self.assertTrue(store.page_cache_warmed)

    def test_preload_can_verify_sha256_while_warming_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path)

            result = store.preload(verify_sha256=True)

            self.assertIs(result, store)
            self.assertTrue(store.sha256_verified)
            self.assertTrue(store.page_cache_warmed)

    def test_preload_rejects_same_size_sha256_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._write_store(root)
            with (root / "frames.bin").open("r+b") as stream:
                stream.write(b"\xff")
            store = DecodedFrameStore.from_manifest(manifest_path)

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                store.preload(verify_sha256=True)
            self.assertFalse(store.sha256_verified)
            self.assertTrue(store.page_cache_warmed)

    def test_failed_reverification_clears_previous_verified_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path)
            store.preload(verify_sha256=True)
            store._metadata["sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                store.preload(verify_sha256=True)

            self.assertFalse(store.sha256_verified)
            self.assertTrue(store.page_cache_warmed)

    def test_spawn_pickle_contains_only_path_and_reopens_lazily(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path).preload()

            state = store.__getstate__()
            self.assertEqual(state["manifest_path"], str(manifest_path.resolve()))
            self.assertEqual(state["manifest_sha256"], store.manifest_sha256)
            self.assertEqual(state["data_sha256"], store.sha256)
            self.assertEqual(state["data_stat"], store.data_stat)
            self.assertNotIn("_memmap", state)
            restored = pickle.loads(pickle.dumps(store))

            self.assertEqual(restored.manifest_path, store.manifest_path)
            self.assertIsNone(restored._memmap)

    def test_spawn_pickle_rejects_same_size_binary_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, frames = self._write_store(root)
            serialized = pickle.dumps(
                DecodedFrameStore.from_manifest(manifest_path).preload()
            )
            replacement = root / "replacement.bin"
            replacement.write_bytes(np.flip(frames, axis=0).tobytes())
            replacement.replace(root / "frames.bin")

            with self.assertRaisesRegex(RuntimeError, "binary changed"):
                pickle.loads(serialized)

    def test_spawn_pickle_rejects_manifest_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            serialized = pickle.dumps(DecodedFrameStore.from_manifest(manifest_path))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                pickle.loads(serialized)

    def test_metadata_accessors_return_defensive_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path)

            metadata = store.metadata
            source = store.source
            metadata["source"]["row_count"] = -1
            source["row_count"] = -1

            self.assertEqual(store.source["row_count"], store.row_count)

    def test_rejects_incomplete_episode_source_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["source"]["episode_offsets_sha256"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "episode_offsets_sha256"):
                DecodedFrameStore.from_manifest(manifest_path)

    def test_requires_complete_source_pixel_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["source_pixel_verification"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_pixel_verification"):
                DecodedFrameStore.from_manifest(manifest_path)

    def test_source_pixel_verification_digests_must_match_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_pixel_verification"]["decoded_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must equal the top-level"):
                DecodedFrameStore.from_manifest(manifest_path)

    def test_rejects_manifest_row_count_that_does_not_match_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_store(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["row_count"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row_count"):
                DecodedFrameStore.from_manifest(manifest_path)

    def test_rejects_binary_size_that_does_not_match_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._write_store(root)
            with (root / "frames.bin").open("ab") as stream:
                stream.write(b"extra")

            with self.assertRaisesRegex(ValueError, "size mismatch"):
                DecodedFrameStore.from_manifest(manifest_path)

    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_take_preserves_order_and_duplicates_as_uint8_nchw(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, frames = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path)

            selected = store.take(torch.tensor([3, 1, 3], dtype=torch.int64))

            self.assertEqual(selected.dtype, torch.uint8)
            self.assertEqual(tuple(selected.shape), (3, *frames.shape[1:]))
            torch.testing.assert_close(
                selected,
                torch.from_numpy(frames[[3, 1, 3]]),
            )

    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_take_rejects_invalid_rows_and_supports_empty_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, frames = self._write_store(Path(temporary))
            store = DecodedFrameStore.from_manifest(manifest_path)

            empty = store.take([])
            self.assertEqual(tuple(empty.shape), (0, *frames.shape[1:]))
            with self.assertRaises(IndexError):
                store.take([-1])
            with self.assertRaises(IndexError):
                store.take([frames.shape[0]])
            with self.assertRaises(TypeError):
                store.take([1.0])


@unittest.skipUnless(
    torch is not None and torchvision is not None,
    "PyTorch and Torchvision are required",
)
class DecodedFrameStoreBuilderTest(unittest.TestCase):
    class FakeLanceDataset:
        def __init__(self, *, offsets=(0, 2)):
            self.lengths = np.asarray([2, 1], dtype=np.int64)
            self.offsets = np.asarray(offsets, dtype=np.int64)
            self.blobs = [b"a", b"bc", b"def"]

        def get_row_data(self, rows):
            return {"pixels": [self.blobs[int(row)] for row in rows]}

    @staticmethod
    def _fake_stable_worldmodel(dataset):
        return types.SimpleNamespace(
            data=types.SimpleNamespace(
                load_dataset=lambda *args, **kwargs: dataset,
            )
        )

    def test_lance_scan_seals_canonical_episode_layout_and_jpeg_payloads(self):
        import sys

        from scripts import build_decoded_frame_store as builder

        dataset = self.FakeLanceDataset()
        fake_swm = self._fake_stable_worldmodel(dataset)
        with patch.dict(sys.modules, {"stable_worldmodel": fake_swm}):
            scan, batches = builder._load_lance_pixels(
                Path("/fixture/source.lance"),
                batch_rows=2,
            )
            self.assertEqual(list(batches), [(0, [b"a", b"bc"]), (2, [b"def"])])

        self.assertEqual(scan.episode_jpeg_payload_bytes, [3, 3])
        source = builder._source_metadata(Path("/fixture/source.lance"), scan)
        self.assertEqual(source["episode_count"], 2)
        self.assertEqual(source["episode_jpeg_payload_bytes"], [3, 3])
        self.assertEqual(source["jpeg_payload_bytes"], 6)
        self.assertEqual(source["row_mapping"], ROW_MAPPING)
        self.assertEqual(source["episode_lengths_dtype"], "<i8")
        self.assertEqual(source["episode_offsets_dtype"], "<i8")
        self.assertEqual(
            source["episode_lengths_sha256"],
            hashlib.sha256(np.asarray([2, 1], dtype="<i8").tobytes()).hexdigest(),
        )
        self.assertEqual(
            source["episode_offsets_sha256"],
            hashlib.sha256(np.asarray([0, 2], dtype="<i8").tobytes()).hexdigest(),
        )

    def test_lance_scan_rejects_offsets_not_equal_to_cumulative_lengths(self):
        import sys

        from scripts import build_decoded_frame_store as builder

        dataset = self.FakeLanceDataset(offsets=(1, 3))
        fake_swm = self._fake_stable_worldmodel(dataset)
        with (
            patch.dict(sys.modules, {"stable_worldmodel": fake_swm}),
            self.assertRaisesRegex(ValueError, "offsets must start at zero"),
        ):
            builder._load_lance_pixels(
                Path("/fixture/source.lance"),
                batch_rows=2,
            )

    def test_small_parallel_build_is_row_ordered_and_atomic(self):
        from scripts import build_decoded_frame_store as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.lance"
            source.mkdir()
            manifest_path = root / "decoded.json"

            def decode(blobs):
                # Make the first submitted future finish last.  The builder must
                # still emit rows in their original Lance order.
                if blobs[0] == 0:
                    time.sleep(0.05)
                return torch.stack(
                    [
                        torch.full((3, 2, 2), value, dtype=torch.uint8)
                        for value in blobs
                    ]
                )

            pixel_batches = iter([(0, [0, 1]), (2, [2, 3]), (4, [4])])
            source_scan = builder.LanceSourceScan(
                episode_lengths=np.asarray([3, 2], dtype="<i8"),
                episode_offsets=np.asarray([0, 3], dtype="<i8"),
                episode_jpeg_payload_bytes=[17, 13],
            )
            with (
                patch.object(
                    builder,
                    "_load_lance_pixels",
                    return_value=(source_scan, pixel_batches),
                ),
                patch.object(builder, "_decode_jpeg_rgb", side_effect=decode),
                patch.object(
                    builder,
                    "_fsync_directory",
                    wraps=builder._fsync_directory,
                ) as fsync_directory,
            ):
                result = builder.build_decoded_frame_store(
                    source,
                    manifest_path,
                    batch_rows=2,
                    workers=2,
                    max_pending=2,
                    expected_row_count=5,
                    expected_frame_shape=(3, 2, 2),
                    report_every=5,
                )

            expected = np.stack(
                [np.full((3, 2, 2), value, dtype=np.uint8) for value in range(5)]
            )
            data_path = manifest_path.with_suffix(".bin")
            self.assertEqual(data_path.read_bytes(), expected.tobytes())
            self.assertEqual(result["shape"], [5, 3, 2, 2])
            self.assertEqual(
                result["source"]["episode_jpeg_payload_bytes"],
                [17, 13],
            )
            self.assertEqual(
                result["sha256"], hashlib.sha256(expected.tobytes()).hexdigest()
            )
            self.assertEqual(
                result["source_pixel_verification"],
                {
                    "method": "decoded_during_build",
                    "row_count": 5,
                    "decoded_sha256": result["sha256"],
                    "data_sha256": result["sha256"],
                    "matches_data_sha256": True,
                    "decoder": {
                        "api": "torchvision.io.decode_jpeg",
                        "mode": "RGB",
                    },
                    "completed_at_utc": result["created_at_utc"],
                },
            )
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(data_path.is_file())
            self.assertFalse(Path(f"{manifest_path}.partial").exists())
            self.assertFalse(Path(f"{data_path}.partial").exists())
            self.assertGreaterEqual(fsync_directory.call_count, 2)

    def test_audit_existing_enriches_manifest_without_rebuilding_binary(self):
        from scripts import build_decoded_frame_store as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.lance"
            source.mkdir()
            source_manifest = Path(f"{source}.manifest.json")
            source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
            source_manifest_sha256 = hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest()
            manifest_path, frames = DecodedFrameStoreTest()._write_store(root)
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing["source"] = {
                "path": str(source.resolve()),
                "format": "lance",
                "row_count": 5,
                "pixel_column": "pixels",
                "row_mapping": ROW_MAPPING,
                "manifest_path": str(source_manifest.resolve()),
                "manifest_sha256": source_manifest_sha256,
            }
            manifest_path.write_text(json.dumps(existing), encoding="utf-8")
            original_binary = (root / "frames.bin").read_bytes()

            source_scan = builder.LanceSourceScan(
                episode_lengths=np.asarray([3, 2], dtype="<i8"),
                episode_offsets=np.asarray([0, 3], dtype="<i8"),
                episode_jpeg_payload_bytes=[0, 0],
            )

            def audited_batches():
                source_scan.episode_jpeg_payload_bytes[:] = [31, 19]
                yield 0, [b"x"] * 5

            with patch.object(
                builder,
                "_load_lance_pixels",
                return_value=(source_scan, audited_batches()),
            ):
                result = builder.audit_existing_decoded_frame_store(
                    source,
                    manifest_path,
                    batch_rows=5,
                    expected_row_count=5,
                    report_every=5,
                )

            self.assertEqual((root / "frames.bin").read_bytes(), original_binary)
            self.assertEqual(
                result["source"]["episode_jpeg_payload_bytes"],
                [31, 19],
            )
            self.assertFalse(result["source_audit"]["decoded_binary_rebuilt"])
            self.assertNotIn("source_pixel_verification", result)
            with self.assertRaisesRegex(ValueError, "source_pixel_verification"):
                DecodedFrameStore.from_manifest(manifest_path)

    def test_audit_existing_allows_missing_pixel_verification_without_forging_it(self):
        from scripts import build_decoded_frame_store as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.lance"
            source.mkdir()
            source_manifest = Path(f"{source}.manifest.json")
            source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
            source_manifest_sha256 = hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest()
            manifest_path, _ = DecodedFrameStoreTest()._write_store(root)
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            del existing["source_pixel_verification"]
            existing["source"] = {
                "path": str(source.resolve()),
                "format": "lance",
                "row_count": 5,
                "pixel_column": "pixels",
                "row_mapping": ROW_MAPPING,
                "manifest_path": str(source_manifest.resolve()),
                "manifest_sha256": source_manifest_sha256,
            }
            manifest_path.write_text(json.dumps(existing), encoding="utf-8")
            source_scan = builder.LanceSourceScan(
                episode_lengths=np.asarray([3, 2], dtype="<i8"),
                episode_offsets=np.asarray([0, 3], dtype="<i8"),
                episode_jpeg_payload_bytes=[29, 23],
            )
            with patch.object(
                builder,
                "_load_lance_pixels",
                return_value=(source_scan, iter([(0, [b"x"] * 5)])),
            ):
                result = builder.audit_existing_decoded_frame_store(
                    source,
                    manifest_path,
                    batch_rows=5,
                    expected_row_count=5,
                    report_every=5,
                )

            self.assertNotIn("source_pixel_verification", result)
            with self.assertRaisesRegex(ValueError, "source_pixel_verification"):
                DecodedFrameStore.from_manifest(manifest_path)

    def test_audit_existing_rejects_wrong_recorded_source_manifest_sha(self):
        from scripts import build_decoded_frame_store as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.lance"
            source.mkdir()
            source_manifest = Path(f"{source}.manifest.json")
            source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
            manifest_path, _ = DecodedFrameStoreTest()._write_store(root)
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing["source"] = {
                "path": str(source.resolve()),
                "format": "lance",
                "row_count": 5,
                "pixel_column": "pixels",
                "row_mapping": ROW_MAPPING,
                "manifest_path": str(source_manifest.resolve()),
                "manifest_sha256": "0" * 64,
            }
            manifest_path.write_text(json.dumps(existing), encoding="utf-8")

            with (
                patch.object(builder, "_load_lance_pixels") as load_pixels,
                self.assertRaisesRegex(ValueError, "does not match the current"),
            ):
                builder.audit_existing_decoded_frame_store(
                    source,
                    manifest_path,
                    expected_row_count=5,
                )
            load_pixels.assert_not_called()

    def test_audit_existing_rejects_unrelated_source_manifest_path(self):
        from scripts import build_decoded_frame_store as builder

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.lance"
            source.mkdir()
            source_manifest = Path(f"{source}.manifest.json")
            source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
            unrelated = root / "unrelated.manifest.json"
            unrelated.write_text('{"source":"other"}\n', encoding="utf-8")
            manifest_path, _ = DecodedFrameStoreTest()._write_store(root)
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing["source"] = {
                "path": str(source.resolve()),
                "format": "lance",
                "row_count": 5,
                "pixel_column": "pixels",
                "row_mapping": ROW_MAPPING,
                "manifest_path": str(unrelated.resolve()),
                "manifest_sha256": hashlib.sha256(
                    unrelated.read_bytes()
                ).hexdigest(),
            }
            manifest_path.write_text(json.dumps(existing), encoding="utf-8")

            with (
                patch.object(builder, "_load_lance_pixels") as load_pixels,
                self.assertRaisesRegex(ValueError, "does not identify the current"),
            ):
                builder.audit_existing_decoded_frame_store(
                    source,
                    manifest_path,
                    expected_row_count=5,
                )
            load_pixels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
