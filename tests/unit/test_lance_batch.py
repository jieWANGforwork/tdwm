import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np

from tdwm.training.lance_batch import (
    BlockPrefetchBatchDataset,
    EpisodeStreamingBatchDataset,
    PairedEpisodeStreamingBatchDataset,
    StrideAwareLanceDataset,
    build_stride_batch_plan,
)

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None
    Image = None


class FakeDecodedFrameStore:
    def __init__(self, frames, episode_jpeg_payload_bytes=(0,)):
        self.frames = frames
        self.episode_jpeg_payload_bytes = episode_jpeg_payload_bytes
        self.requests = []

    def take(self, global_rows):
        rows = [int(row) for row in global_rows]
        self.requests.append(rows)
        return self.frames[torch.tensor(rows, dtype=torch.long)]


def decoded_test_frames(blobs):
    frames = []
    for blob in blobs:
        with Image.open(BytesIO(blob)) as image:
            array = np.array(image.convert("RGB"), copy=True)
        frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(frames)


class StrideBatchPlanTest(unittest.TestCase):
    def test_only_requests_frames_consumed_by_the_stride(self):
        plan = build_stride_batch_plan(
            clip_indices=[(0, 0), (0, 5)],
            offsets=[0],
            indices=[0, 1],
            frameskip=5,
            num_steps=4,
        )

        self.assertEqual(plan.global_starts, (0, 5))
        self.assertEqual(plan.unique_frame_rows, (0, 5, 10, 15, 20))
        self.assertEqual(plan.frame_gathers, ((0, 1, 2, 3), (1, 2, 3, 4)))
        self.assertEqual(plan.legacy_row_requests, 40)
        self.assertEqual(plan.image_row_requests, 5)

    def test_episode_offsets_are_applied_before_stride(self):
        plan = build_stride_batch_plan(
            clip_indices=[(0, 2), (1, 3)],
            offsets=[0, 100],
            indices=[1, 0],
            frameskip=2,
            num_steps=3,
        )

        self.assertEqual(plan.global_starts, (103, 2))
        self.assertEqual(plan.unique_frame_rows, (2, 4, 6, 103, 105, 107))
        self.assertEqual(plan.frame_gathers, ((3, 4, 5), (0, 1, 2)))

    def test_negative_indices_match_dataset_indexing(self):
        plan = build_stride_batch_plan(
            clip_indices=[(0, 0), (0, 1)],
            offsets=[10],
            indices=[-1],
            frameskip=1,
            num_steps=2,
        )

        self.assertEqual(plan.global_starts, (11,))
        self.assertEqual(plan.unique_frame_rows, (11, 12))

    def test_invalid_parameters_fail_before_access(self):
        with self.assertRaisesRegex(ValueError, "frameskip"):
            build_stride_batch_plan(
                clip_indices=[],
                offsets=[],
                indices=[],
                frameskip=0,
                num_steps=4,
            )
        with self.assertRaisesRegex(IndexError, "out of range"):
            build_stride_batch_plan(
                clip_indices=[(0, 0)],
                offsets=[0],
                indices=[1],
                frameskip=1,
                num_steps=1,
            )


@unittest.skipUnless(torch is not None and Image is not None, "PyTorch is required")
class StrideAwareLanceDatasetTest(unittest.TestCase):
    class FakeLanceDataset:
        def __init__(self):
            self.offsets = np.array([0])
            self.lengths = np.array([10])
            self.frameskip = 2
            self.num_steps = 3
            self.span = 6
            self.clip_indices = [(0, start) for start in range(5)]
            self.column_names = ["pixels", "action", "observation"]
            self.action = np.arange(20, dtype=np.float32).reshape(10, 2)
            self.observation = np.arange(30, dtype=np.float32).reshape(10, 3)
            self.pixels = []
            for value in range(10):
                array = np.full((8, 8, 3), value * 20, dtype=np.uint8)
                buffer = BytesIO()
                Image.fromarray(array).save(buffer, format="JPEG", quality=100)
                self.pixels.append(buffer.getvalue())
            self.requested_rows = None
            self.row_requests = 0
            self.col_requests = []
            self.transform_calls = 0
            self.transform = self._transform

        def __len__(self):
            return len(self.clip_indices)

        def get_row_data(self, rows):
            self.requested_rows = list(rows)
            self.row_requests += 1
            return {
                "pixels": np.asarray([self.pixels[row] for row in rows], dtype=object),
                "action": self.action[rows],
                "observation": self.observation[rows],
            }

        def get_col_data(self, column):
            self.col_requests.append(column)
            if column == "action":
                return self.action
            if column == "observation":
                return self.observation
            raise KeyError(column)

        def get_dim(self, column):
            return self.get_col_data(column).shape[1]

        def _transform(self, sample):
            self.transform_calls += 1
            sample["observation"] = sample["observation"] + 1
            return sample

    def test_batch_preserves_dense_actions_and_strided_observations(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)

        samples = dataset.__getitems__([0, 2])

        self.assertEqual(source.requested_rows, [0, 2, 4, 6])
        self.assertEqual(source.transform_calls, 2)
        self.assertEqual(tuple(samples[0]["pixels"].shape), (3, 3, 8, 8))
        self.assertEqual(samples[0]["_tdwm_global_start"].item(), 0)
        self.assertEqual(samples[1]["_tdwm_global_start"].item(), 2)
        self.assertEqual(samples[0]["_tdwm_global_start"].dtype, torch.int64)
        torch.testing.assert_close(
            samples[0]["action"],
            torch.tensor(source.action[0:6]).reshape(3, 4),
        )
        torch.testing.assert_close(
            samples[1]["observation"],
            torch.tensor(source.observation[[2, 4, 6]]) + 1,
        )

    def test_prefetched_block_decodes_once_and_materializes_selected_clips(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)

        prefetched = dataset.prefetch([0, 1, 2, 3])
        first_batch = dataset.materialize_prefetched(prefetched, [0, 1])
        second_batch = dataset.materialize_prefetched(prefetched, [2, 3])

        self.assertEqual(source.row_requests, 1)
        self.assertEqual(source.transform_calls, 4)
        self.assertEqual(tuple(first_batch[0]["pixels"].shape), (3, 3, 8, 8))
        torch.testing.assert_close(
            second_batch[1]["action"],
            torch.tensor(source.action[3:9]).reshape(3, 4),
        )

    def test_decoded_store_matches_jpeg_path_without_reading_images(self):
        baseline_source = self.FakeLanceDataset()
        baseline = StrideAwareLanceDataset(baseline_source).__getitems__([0, 2])

        source = self.FakeLanceDataset()
        store = FakeDecodedFrameStore(decoded_test_frames(source.pixels))
        dataset = StrideAwareLanceDataset(source, decoded_frame_store=store)
        with patch(
            "tdwm.training.lance_batch._decode_images",
            side_effect=AssertionError("decoded store must bypass JPEG decoding"),
        ):
            accelerated = dataset.__getitems__([0, 2])

        self.assertEqual(source.row_requests, 0)
        self.assertNotIn("pixels", source.col_requests)
        self.assertEqual(store.requests, [[0, 2, 4, 6]])
        for expected, actual in zip(baseline, accelerated):
            torch.testing.assert_close(actual["pixels"], expected["pixels"])
            torch.testing.assert_close(actual["action"], expected["action"])
            torch.testing.assert_close(actual["observation"], expected["observation"])

    def test_decoded_store_episode_count_must_match_lance_dataset(self):
        source = self.FakeLanceDataset()
        store = FakeDecodedFrameStore(
            decoded_test_frames(source.pixels), episode_jpeg_payload_bytes=[]
        )

        with self.assertRaisesRegex(ValueError, "episode count differs"):
            StrideAwareLanceDataset(source, decoded_frame_store=store)

    def test_block_prefetch_yields_every_source_clip_once(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)
        batches = BlockPrefetchBatchDataset(
            dataset,
            [4, 0, 3, 1, 2],
            batch_size=2,
            block_size=4,
            drop_last=False,
            seed=7,
            shuffle_batches_within_block=True,
        )

        emitted = list(batches)

        self.assertEqual(len(emitted), 3)
        self.assertEqual(source.row_requests, 2)
        starts = []
        for batch in emitted:
            starts.extend(batch["pixels"][:, 0, 0, 0, 0].tolist())
        self.assertEqual(sorted(starts), [0, 20, 40, 60, 80])

    def test_block_prefetch_partitions_blocks_across_workers(self):
        source = self.FakeLanceDataset()
        dataset = StrideAwareLanceDataset(source)
        batches = BlockPrefetchBatchDataset(
            dataset,
            [4, 0, 3, 1, 2],
            batch_size=2,
            block_size=4,
            drop_last=False,
            seed=7,
            shuffle_batches_within_block=True,
        )

        class Worker:
            def __init__(self, worker_id):
                self.id = worker_id
                self.num_workers = 2
                self.seed = 11 + worker_id

        def collate_in_test(samples):
            return {
                name: torch.stack([sample[name] for sample in samples])
                for name in samples[0]
            }

        with (
            patch("torch.utils.data.get_worker_info", return_value=Worker(0)),
            patch("torch.utils.data.default_collate", side_effect=collate_in_test),
        ):
            first_worker = list(batches)
        with (
            patch("torch.utils.data.get_worker_info", return_value=Worker(1)),
            patch("torch.utils.data.default_collate", side_effect=collate_in_test),
        ):
            second_worker = list(batches)
        emitted = first_worker + second_worker

        self.assertEqual(len(emitted), 3)
        starts = []
        for batch in emitted:
            starts.extend(batch["pixels"][:, 0, 0, 0, 0].tolist())
        self.assertEqual(sorted(starts), [0, 20, 40, 60, 80])


@unittest.skipUnless(torch is not None and Image is not None, "PyTorch is required")
class EpisodeStreamingBatchDatasetTest(unittest.TestCase):
    class FakeEpisodeDataset:
        def __init__(self, episodes=4, steps=8):
            self.lengths = np.full(episodes, steps, dtype=np.int64)
            self.offsets = np.arange(episodes, dtype=np.int64) * steps
            self.frameskip = 2
            self.num_steps = 2
            self.span = 4
            self.clip_indices = [
                (episode, start)
                for episode in range(episodes)
                for start in range(steps - self.span)
            ]
            self.column_names = ["pixels", "action", "observation"]
            total = episodes * steps
            self.action = np.arange(total * 2, dtype=np.float32).reshape(total, 2)
            self.observation = np.arange(total * 3, dtype=np.float32).reshape(total, 3)
            self.pixels = []
            for value in range(total):
                array = np.full((8, 8, 3), (value * 5) % 256, dtype=np.uint8)
                buffer = BytesIO()
                Image.fromarray(array).save(buffer, format="JPEG", quality=100)
                self.pixels.append(buffer.getvalue())
            self.row_requests = []
            self.col_requests = []
            self.transform = self._transform

        def __len__(self):
            return len(self.clip_indices)

        def get_row_data(self, rows):
            rows = list(rows)
            self.row_requests.append(rows)
            return {
                "pixels": np.asarray([self.pixels[row] for row in rows], dtype=object),
                "action": self.action[rows],
                "observation": self.observation[rows],
            }

        def get_col_data(self, column):
            self.col_requests.append(column)
            if column == "action":
                return self.action
            if column == "observation":
                return self.observation
            raise KeyError(column)

        @staticmethod
        def _transform(batch):
            if batch["action"].shape[-1] != 2:
                raise AssertionError("Action normalization must precede flattening.")
            batch["action"] = batch["action"] / 2
            return batch

    def _stream(
        self,
        *,
        cache_bytes=1024 * 1024,
        use_store=False,
        episodes=4,
        steps=8,
        read_episodes=2,
    ):
        source = self.FakeEpisodeDataset(episodes=episodes, steps=steps)
        store = None
        if use_store:
            payload_bytes = tuple(
                sum(
                    len(source.pixels[row])
                    for row in range(
                        int(source.offsets[episode]),
                        int(source.offsets[episode] + source.lengths[episode]),
                    )
                )
                for episode in range(len(source.lengths))
            )
            store = FakeDecodedFrameStore(
                decoded_test_frames(source.pixels),
                episode_jpeg_payload_bytes=payload_bytes,
            )
        source.decoded_frame_store = store
        dataset = EpisodeStreamingBatchDataset(
            StrideAwareLanceDataset(source, decoded_frame_store=store),
            list(range(len(source))),
            batch_size=4,
            active_episodes=4,
            read_episodes=read_episodes,
            cache_bytes=cache_bytes,
            prefetch_blocks=1,
            seed=7,
            drop_last=True,
            min_unique_episodes=4,
        )
        return source, dataset

    def test_reads_contiguous_episode_blocks_and_mixes_every_batch(self):
        source, dataset = self._stream()

        batches = list(dataset)

        self.assertEqual(len(batches), 4)
        self.assertEqual(source.row_requests, [list(range(16)), list(range(16, 32))])
        for batch in batches:
            self.assertEqual(
                torch.unique(batch["_tdwm_episode_id"]).numel(), 4
            )
            self.assertLessEqual(batch["_tdwm_cache_bytes"], 1024 * 1024)
            self.assertEqual(tuple(batch["pixels"].shape), (4, 2, 3, 8, 8))
            self.assertEqual(tuple(batch["action"].shape), (4, 2, 4))
            self.assertEqual(batch["_tdwm_global_start"].dtype, torch.int64)
            self.assertEqual(tuple(batch["_tdwm_global_start"].shape), (4,))
            # The fake action's first normalized scalar equals its source row,
            # so this also proves metadata follows shuffled clip selection.
            torch.testing.assert_close(
                batch["_tdwm_global_start"],
                batch["action"][:, 0, 0].to(dtype=torch.int64),
            )

    def test_epoch_schedule_is_reproducible(self):
        _, first = self._stream()
        _, second = self._stream()
        first.set_epoch(3)
        second.set_epoch(3)

        first_starts = [batch["action"][:, 0, 0].tolist() for batch in first]
        second_starts = [batch["action"][:, 0, 0].tolist() for batch in second]

        self.assertEqual(first_starts, second_starts)

    def test_decoded_store_matches_stream_without_caching_jpegs(self):
        _, baseline = self._stream()
        source, accelerated = self._stream(use_store=True)

        expected_batches = list(baseline)
        with patch(
            "tdwm.training.lance_batch._decode_images",
            side_effect=AssertionError("decoded store must bypass JPEG decoding"),
        ):
            actual_batches = list(accelerated)

        self.assertEqual(source.row_requests, [])
        self.assertNotIn("pixels", source.col_requests)
        self.assertTrue(source.decoded_frame_store.requests)
        self.assertEqual(len(actual_batches), len(expected_batches))
        for expected, actual in zip(expected_batches, actual_batches):
            torch.testing.assert_close(actual["pixels"], expected["pixels"])
            torch.testing.assert_close(actual["action"], expected["action"])
            torch.testing.assert_close(
                actual["_tdwm_episode_id"], expected["_tdwm_episode_id"]
            )
            self.assertEqual(
                actual["_tdwm_cache_bytes"], expected["_tdwm_cache_bytes"]
            )

    def test_decoded_store_preserves_order_and_accounting_at_cache_limit(self):
        _, probe = self._stream(read_episodes=4)
        cache_bytes = sum(
            episode.byte_size for episode in probe._read_episode_block((0, 1, 2, 3))
        )
        _, baseline = self._stream(cache_bytes=cache_bytes, read_episodes=4)
        source, accelerated = self._stream(
            cache_bytes=cache_bytes,
            read_episodes=4,
            use_store=True,
        )

        expected_batches = list(baseline)
        actual_batches = list(accelerated)

        self.assertEqual(
            max(batch["_tdwm_cache_bytes"] for batch in expected_batches),
            cache_bytes,
        )
        self.assertEqual(len(actual_batches), len(expected_batches))
        for expected, actual in zip(expected_batches, actual_batches):
            torch.testing.assert_close(actual["pixels"], expected["pixels"])
            torch.testing.assert_close(actual["action"], expected["action"])
            torch.testing.assert_close(
                actual["_tdwm_episode_id"], expected["_tdwm_episode_id"]
            )
            self.assertEqual(
                actual["_tdwm_cache_bytes"], expected["_tdwm_cache_bytes"]
            )
        self.assertEqual(source.row_requests, [])

    def test_later_epoch_rotates_the_sequential_episode_stream(self):
        first_source, first = self._stream()
        later_source, later = self._stream()
        later.set_epoch(3)

        list(first)
        list(later)

        self.assertEqual(first_source.row_requests[0], list(range(16)))
        self.assertNotEqual(later_source.row_requests[0], list(range(16)))

    def test_fails_before_training_when_cache_cannot_hold_diverse_pool(self):
        _, dataset = self._stream(cache_bytes=1)

        with self.assertRaisesRegex(MemoryError, "episode_cache_bytes"):
            list(dataset)

    def test_tail_fallback_preserves_full_batches_after_sources_end(self):
        source = self.FakeEpisodeDataset(episodes=6, steps=14)
        source_indices = [0, 10]
        source_indices.extend(range(20, 24))
        source_indices.extend(range(30, 34))
        source_indices.extend(range(40, 50))
        source_indices.extend(range(50, 60))
        dataset = EpisodeStreamingBatchDataset(
            StrideAwareLanceDataset(source),
            source_indices,
            batch_size=4,
            active_episodes=4,
            read_episodes=2,
            cache_bytes=1024 * 1024,
            prefetch_blocks=1,
            seed=7,
            drop_last=True,
            min_unique_episodes=4,
        )

        batches = list(dataset)

        self.assertEqual(len(batches), len(source_indices) // 4)
        unique_counts = [
            torch.unique(batch["_tdwm_episode_id"]).numel() for batch in batches
        ]
        self.assertEqual(unique_counts[:4], [4, 4, 4, 4])
        self.assertEqual(unique_counts[-3:], [2, 2, 2])


class PairedEpisodeStreamingBatchDatasetTest(unittest.TestCase):
    class FakeStream(EpisodeStreamingBatchDataset):
        def __init__(self, name, size):
            self.name = name
            self.size = size
            self.epoch = 0

        def __len__(self):
            return self.size

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            for index in range(self.size):
                yield {"sample": f"{self.name}-{self.epoch}-{index}"}

    def test_pairs_independent_views_and_uses_the_shorter_epoch(self):
        world = self.FakeStream("world", 3)
        tail = self.FakeStream("tail", 5)
        paired = PairedEpisodeStreamingBatchDataset(world, tail)
        paired.set_epoch(4)

        batches = list(paired)

        self.assertEqual(len(paired), 3)
        self.assertEqual(
            batches[2],
            {
                "world": {"sample": "world-4-2"},
                "tail": {"sample": "tail-4-2"},
            },
        )


if __name__ == "__main__":
    unittest.main()
