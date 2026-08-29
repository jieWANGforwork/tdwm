#!/usr/bin/env python3
"""Build an offline frozen-state neighbor graph from a precomputed bank."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tdwm.training.state_neighbor_index_builder import (
    ExactNumpySearchBackend,
    FaissHNSWSearchBackend,
    build_state_neighbor_index,
)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a precomputed frozen-state bank into the immutable kNN "
            "graph used by G1 training."
        )
    )
    parser.add_argument(
        "--bank-dir",
        required=True,
        type=Path,
        help=(
            "Strict frozen-state-bank artifact containing manifest.json plus "
            "global_rows.npy, episode_ids.npy, latents.npy, and actions.npy."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-split-sha256", required=True)
    parser.add_argument(
        "--git-revision",
        default=None,
        help="Full source revision; defaults to the current Git HEAD.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        required=True,
        help="Method-configured neighbor count; no experiment default is assumed.",
    )
    parser.add_argument("--search-depth", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--backend",
        choices=("faiss_hnsw", "exact_numpy"),
        default="faiss_hnsw",
        help="Use exact_numpy only for small audits/tests.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--add-batch-size", type=int, default=65_536)
    parser.add_argument("--exact-max-pairwise-entries", type=int, default=20_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = args.git_revision or _git_revision()
    if revision is None:
        raise SystemExit("Pass --git-revision when Git HEAD cannot be resolved.")
    if args.backend == "faiss_hnsw":
        backend = FaissHNSWSearchBackend(
            seed=args.seed,
            threads=args.threads,
            hnsw_m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            add_batch_size=args.add_batch_size,
        )
    else:
        backend = ExactNumpySearchBackend(
            max_pairwise_entries=args.exact_max_pairwise_entries
        )
    manifest = build_state_neighbor_index(
        bank_dir=args.bank_dir,
        output_dir=args.output_dir,
        pretrained_world_model_sha256=args.checkpoint_sha256,
        training_split_sha256=args.training_split_sha256,
        git_revision=revision,
        neighbors=args.neighbors,
        search_depth=args.search_depth,
        query_chunk_size=args.query_chunk_size,
        backend=backend,
        seed=args.seed,
        threads=args.threads,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
