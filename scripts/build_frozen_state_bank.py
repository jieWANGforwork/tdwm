#!/usr/bin/env python3
"""Build the strict training-only state bank used by offline G1 indexing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from tdwm.training.frozen_latent_store import FrozenLatentStore, file_sha256
from tdwm.training.frozen_state_bank import build_frozen_training_state_bank


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _load_lance_clip_metadata(
    dataset_path: Path,
    *,
    num_steps: int,
    frame_skip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load clip/episode indices without reading or decoding image columns."""

    import stable_worldmodel as swm

    version = importlib.metadata.version("stable-worldmodel")
    if version != "0.1.1":
        raise RuntimeError(f"Expected stable-worldmodel 0.1.1, found {version}.")
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format="lance",
        transform=None,
        num_steps=int(num_steps),
        frameskip=int(frame_skip),
        # Action is a small numeric column.  Pixels are deliberately absent,
        # and no column is eagerly cached for this metadata-only traversal.
        keys_to_load=["action"],
        keys_to_cache=[],
    )
    required = ("clip_indices", "offsets", "lengths")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise TypeError(
            "The Lance dataset is missing clip metadata: " + ", ".join(missing)
        )
    clips = np.asarray(dataset.clip_indices)
    offsets = np.asarray(dataset.offsets)
    lengths = np.asarray(dataset.lengths)
    return clips, offsets, lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive sorted, unique TD-current training rows from a strict "
            "FrozenLatentStore and the exact current-run split_indices.npz."
        )
    )
    parser.add_argument("--latent-store", required=True, type=Path)
    parser.add_argument("--split-indices", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-source-sha256", required=True)
    parser.add_argument("--column-normalization-sha256", required=True)
    parser.add_argument("--training-split-sha256", required=True)
    parser.add_argument("--num-steps", type=int, default=19)
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--copy-chunk-rows", type=int, default=65_536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_dir() or dataset_path.suffix.lower() != ".lance":
        raise ValueError("--dataset must be the audited Cube .lance directory.")
    dataset_manifest_path = Path(f"{dataset_path}.manifest.json").resolve()
    dataset_manifest = _load_json_object(
        dataset_manifest_path, label="Lance conversion manifest"
    )
    if dataset_manifest.get("destination", {}).get("format") != "lance":
        raise ValueError("The dataset conversion manifest is not for Lance.")
    source_sha = dataset_manifest.get("source", {}).get("sha256")
    if source_sha != args.dataset_source_sha256:
        raise ValueError("The Lance source SHA-256 differs from the CLI lock.")

    store = FrozenLatentStore(
        args.latent_store,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_dataset_source_sha256=args.dataset_source_sha256,
        expected_column_normalization_sha256=(args.column_normalization_sha256),
        expected_frame_skip=args.frame_skip,
        expected_history_frames=args.history_frames,
        expected_action_dim=args.action_dim,
    )
    clips, offsets, lengths = _load_lance_clip_metadata(
        dataset_path,
        num_steps=args.num_steps,
        frame_skip=args.frame_skip,
    )
    if lengths.ndim != 1 or lengths.dtype.kind not in ("i", "u"):
        raise ValueError("The Lance episode lengths are malformed.")
    if int(np.asarray(lengths, dtype=np.int64).sum()) != store.total_rows:
        raise ValueError("The Lance row count differs from the frozen latent store.")

    manifest = build_frozen_training_state_bank(
        args.output_dir,
        latent_store=store,
        split_indices_path=args.split_indices,
        expected_training_split_sha256=args.training_split_sha256,
        clip_indices=clips,
        episode_offsets=offsets,
        num_steps=args.num_steps,
        history_frames=args.history_frames,
        frame_skip=args.frame_skip,
        copy_chunk_rows=args.copy_chunk_rows,
        source_metadata={
            "lance_dataset_path": str(dataset_path),
            "lance_manifest_path": str(dataset_manifest_path),
            "lance_manifest_sha256": file_sha256(dataset_manifest_path),
            "metadata_columns_loaded": ["action"],
            "image_columns_loaded": [],
            "stable_worldmodel_version": "0.1.1",
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
