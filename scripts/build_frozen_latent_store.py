#!/usr/bin/env python3
"""Encode every Cube global frame once into a shared frozen-LeWM cache."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from tdwm.training.decoded_frame_store import DecodedFrameStore
from tdwm.training.frozen_latent_store import (
    FrozenLatentStoreSpec,
    build_frozen_latent_store,
    file_sha256,
    iter_encoded_global_rows_once,
    normalize_actions,
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


def _resolve_local_pretrained_export(
    checkpoint_path: str | Path,
) -> tuple[str, Path, Path]:
    requested = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir = requested if requested.is_dir() else requested.parent
    weights = sorted(checkpoint_dir.glob("*.pt"))
    if len(weights) != 1:
        raise FileNotFoundError(
            "A stable-worldmodel export must contain exactly one .pt weight file."
        )
    if not requested.is_dir() and requested != weights[0]:
        raise ValueError("The requested checkpoint is not the export weight file.")
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "Expected the public <cache_dir>/checkpoints/<run_name> export layout."
        )
    return checkpoint_dir.name, weights[0], checkpoint_dir.parent.parent


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _canonical_episode_layout(
    dataset: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("lengths", "offsets", "get_col_data")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise TypeError(
            "The Lance dataset is missing required public attributes: "
            + ", ".join(missing)
        )
    lengths = np.asarray(dataset.lengths)
    offsets = np.asarray(dataset.offsets)
    if lengths.ndim != 1 or offsets.ndim != 1 or lengths.size != offsets.size:
        raise ValueError("Lance episode lengths/offsets are malformed.")
    if lengths.dtype.kind not in ("i", "u") or offsets.dtype.kind not in ("i", "u"):
        raise TypeError("Lance episode lengths/offsets must be integers.")
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("Lance episodes must have positive lengths.")
    expected_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("Lance episode offsets are not contiguous global rows.")
    episode_ids = np.repeat(
        np.arange(lengths.size, dtype=np.int64), lengths
    )
    return lengths, offsets, episode_ids


def _validate_action_terminal_rows(
    raw_actions: Any,
    *,
    lengths: np.ndarray,
    offsets: np.ndarray,
    normalization_samples: Any,
) -> dict[str, Any]:
    """Audit Cube's finite transitions and one all-NaN terminal per episode."""

    actions = np.asarray(raw_actions)
    if actions.ndim != 2:
        raise ValueError("The Lance action column must be two-dimensional.")
    all_finite = np.isfinite(actions).all(axis=1)
    all_nan = np.isnan(actions).all(axis=1)
    if not np.all(all_finite | all_nan):
        raise ValueError(
            "Each action row must be either entirely finite or entirely NaN."
        )
    expected_terminal_rows = np.asarray(offsets + lengths - 1, dtype=np.int64)
    actual_terminal_rows = np.flatnonzero(all_nan)
    if not np.array_equal(actual_terminal_rows, expected_terminal_rows):
        raise ValueError(
            "All-NaN action rows must be exactly the final row of every episode."
        )
    finite_rows = int(all_finite.sum())
    if (
        isinstance(normalization_samples, bool)
        or not isinstance(normalization_samples, int)
        or normalization_samples != finite_rows
    ):
        raise ValueError(
            "Action normalization sample count must equal the number of fully "
            "finite action rows."
        )
    return {
        "row_policy": "each row all-finite or all-NaN",
        "finite_rows": finite_rows,
        "terminal_nan_rows": int(all_nan.sum()),
        "episode_count": int(lengths.size),
        "terminal_nan_rows_match_episode_ends": True,
        "normalization_samples_match_finite_rows": True,
    }


def _validate_decoded_source_binding(
    *,
    decoded: DecodedFrameStore,
    decoded_manifest: Path,
    dataset_path: Path,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    lengths: np.ndarray,
    offsets: np.ndarray,
) -> None:
    source = decoded.source
    recorded_dataset = Path(source.get("path", "")).expanduser()
    if not recorded_dataset.is_absolute():
        recorded_dataset = decoded_manifest.parent / recorded_dataset
    if recorded_dataset.resolve() != dataset_path:
        raise ValueError("Decoded frames are bound to a different Lance dataset.")
    recorded_manifest = Path(source.get("manifest_path", "")).expanduser()
    if not recorded_manifest.is_absolute():
        recorded_manifest = decoded_manifest.parent / recorded_manifest
    if recorded_manifest.resolve() != dataset_manifest_path:
        raise ValueError("Decoded frames are bound to a different Lance manifest.")
    if source.get("manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("Decoded-frame Lance manifest SHA-256 does not match.")
    if decoded.row_count != int(lengths.sum()):
        raise ValueError("Decoded-frame row count differs from Lance.")

    # Reproduce the decoded-store canonical episode layout seals.
    canonical_lengths = np.asarray(lengths, dtype="<i8")
    canonical_offsets = np.asarray(offsets, dtype="<i8")
    import hashlib

    if hashlib.sha256(canonical_lengths.tobytes(order="C")).hexdigest() != (
        decoded.episode_lengths_sha256
    ):
        raise ValueError("Decoded-frame episode lengths differ from Lance.")
    if hashlib.sha256(canonical_offsets.tobytes(order="C")).hexdigest() != (
        decoded.episode_offsets_sha256
    ):
        raise ValueError("Decoded-frame episode offsets differ from Lance.")


def _preprocess_frames(
    frames: Any,
    *,
    device: Any,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    import torch
    import torch.nn.functional as functional

    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("Decoded frames must be uint8 NCHW RGB tensors.")
    pixels = frames.to(device=device, non_blocking=True).to(torch.float32).div_(255.0)
    channel_mean = torch.tensor(mean, device=device).reshape(1, 3, 1, 1)
    channel_std = torch.tensor(std, device=device).reshape(1, 3, 1, 1)
    pixels = (pixels - channel_mean) / channel_std
    if tuple(pixels.shape[-2:]) != (image_size, image_size):
        pixels = functional.interpolate(
            pixels,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one immutable global-row latent/action store shared by the "
            "frozen C, D, F, G1, G2, and G3 experiments. The extractor "
            "traverses global frames, never overlapping sequence clips."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--decoded-manifest", required=True, type=Path)
    parser.add_argument("--column-normalization", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--dataset-source-sha256")
    parser.add_argument("--column-normalization-sha256")
    parser.add_argument("--git-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--report-every-rows", type=int, default=25_000)
    parser.add_argument(
        "--verify-decoded-sha256",
        action="store_true",
        help=(
            "Hash the complete decoded binary before extraction. The latent "
            "builder always binds its recorded SHA; this optional audit adds a "
            "full 282-GiB read before encoding."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.report_every_rows <= 0:
        raise SystemExit("--batch-size and --report-every-rows must be positive.")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive.")

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_dir() or dataset_path.suffix.lower() != ".lance":
        raise ValueError("--dataset must be the audited Cube .lance directory.")
    dataset_manifest_path = Path(f"{dataset_path}.manifest.json").resolve()
    dataset_manifest = _load_json_object(
        dataset_manifest_path, "Lance conversion manifest"
    )
    if dataset_manifest.get("destination", {}).get("format") != "lance":
        raise ValueError("The dataset conversion manifest is not for Lance.")
    dataset_source_sha256 = dataset_manifest.get("source", {}).get("sha256")
    if not isinstance(dataset_source_sha256, str):
        raise ValueError("Lance manifest source.sha256 is missing.")
    if (
        args.dataset_source_sha256 is not None
        and dataset_source_sha256 != args.dataset_source_sha256
    ):
        raise ValueError("The dataset source SHA-256 differs from the CLI lock.")
    dataset_manifest_sha256 = file_sha256(dataset_manifest_path)

    decoded_manifest = args.decoded_manifest.expanduser().resolve()
    decoded = DecodedFrameStore.from_manifest(decoded_manifest)
    if args.verify_decoded_sha256:
        decoded.preload(verify_sha256=True)

    column_path = args.column_normalization.expanduser().resolve()
    column_sha256 = file_sha256(column_path)
    if (
        args.column_normalization_sha256 is not None
        and column_sha256 != args.column_normalization_sha256
    ):
        raise ValueError("Column-normalization SHA-256 differs from the CLI lock.")
    column_stats = _load_json_object(column_path, "column normalization")
    action_stats = column_stats.get("action")
    if not isinstance(action_stats, dict):
        raise ValueError("Column normalization is missing action statistics.")
    observation_stats = column_stats.get("observation")
    if not isinstance(observation_stats, dict):
        raise ValueError("Column normalization is missing observation statistics.")

    checkpoint_name, checkpoint_file, checkpoint_cache = (
        _resolve_local_pretrained_export(args.checkpoint)
    )
    checkpoint_sha256 = file_sha256(checkpoint_file)
    if (
        args.checkpoint_sha256 is not None
        and checkpoint_sha256 != args.checkpoint_sha256
    ):
        raise ValueError("Pretrained checkpoint SHA-256 differs from the CLI lock.")
    revision = args.git_revision or _git_revision()
    if revision is None:
        raise ValueError("Pass --git-revision when Git HEAD cannot be resolved.")

    import stable_worldmodel as swm
    import torch

    stable_version = importlib.metadata.version("stable-worldmodel")
    if stable_version != "0.1.1":
        raise RuntimeError(
            f"Expected stable-worldmodel 0.1.1, found {stable_version}."
        )
    dataset = swm.data.load_dataset(
        str(dataset_path),
        format="lance",
        transform=None,
        num_steps=1,
        frameskip=1,
        keys_to_load=["action"],
        keys_to_cache=["action"],
    )
    lengths, offsets, episode_ids = _canonical_episode_layout(dataset)
    _validate_decoded_source_binding(
        decoded=decoded,
        decoded_manifest=decoded_manifest,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        lengths=lengths,
        offsets=offsets,
    )
    total_rows = int(episode_ids.size)
    raw_actions = np.asarray(dataset.get_col_data("action"))
    if raw_actions.shape != (total_rows, args.action_dim):
        raise ValueError("The Lance action column has the wrong shape.")
    action_row_audit = _validate_action_terminal_rows(
        raw_actions,
        lengths=lengths,
        offsets=offsets,
        normalization_samples=action_stats.get("samples"),
    )
    if observation_stats.get("samples") != total_rows:
        raise ValueError(
            "Observation normalization must be fit on every dataset row."
        )
    normalized_actions = normalize_actions(
        raw_actions,
        mean=action_stats.get("mean"),
        scale=action_stats.get("scale"),
        expected_action_dim=args.action_dim,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction was requested but CUDA is unavailable.")
    if args.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 extraction is supported only on CUDA.")
    model = swm.wm.load_pretrained(checkpoint_name, cache_dir=str(checkpoint_cache))
    model.to(device).eval().requires_grad_(False)
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    reported_until = 0
    preprocessing_parity_audit: dict[str, Any] = {"status": "pending"}

    def encode_rows(rows: np.ndarray) -> np.ndarray:
        nonlocal reported_until
        frames = decoded.take(rows)
        pixels = _preprocess_frames(
            frames,
            device=device,
            image_size=args.image_size,
            mean=image_mean,
            std=image_std,
        )
        if preprocessing_parity_audit["status"] == "pending":
            from tdwm.training.gt_lewm_support import preprocess_image_batch

            formal_pixels = preprocess_image_batch(
                frames.unsqueeze(1).to(device=device, non_blocking=True),
                mean=torch.tensor(image_mean, device=device).reshape(
                    1, 1, 3, 1, 1
                ),
                std=torch.tensor(image_std, device=device).reshape(
                    1, 1, 3, 1, 1
                ),
                size=args.image_size,
            )[:, 0]
            max_abs_difference = float(
                (pixels - formal_pixels).abs().max().detach().cpu()
            )
            if not torch.equal(pixels, formal_pixels):
                raise RuntimeError(
                    "Global-row preprocessing differs from the formal trainer: "
                    f"max_abs_difference={max_abs_difference}"
                )
            preprocessing_parity_audit.update(
                {
                    "status": "passed",
                    "global_rows": rows[: min(8, rows.size)].tolist(),
                    "rows_compared": int(rows.size),
                    "comparison": "torch.equal",
                    "max_abs_difference": max_abs_difference,
                    "formal_reference": (
                        "tdwm.training.gt_lewm_support.preprocess_image_batch"
                    ),
                    "extra_encoder_calls": 0,
                }
            )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.precision == "bfloat16"
            else nullcontext()
        )
        # stable-worldmodel 0.1.1 LeWM.encode computes emb only from pixels.
        # Action only creates the independent act_emb output and proprio is not
        # read, so omitting both avoids needless work without changing emb.
        with torch.inference_mode(), autocast:
            output = model.encode({"pixels": pixels.unsqueeze(1)})
        embeddings = output.get("emb")
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Frozen LeWM encode() did not return tensor 'emb'.")
        if embeddings.shape != (rows.size, 1, args.embed_dim):
            raise ValueError(
                "Frozen LeWM row encoding has an unexpected shape: "
                f"{tuple(embeddings.shape)}"
            )
        completed = int(rows[-1]) + 1
        if (
            completed - reported_until >= args.report_every_rows
            or completed == total_rows
        ):
            print(f"encoded_global_rows={completed}/{total_rows}", flush=True)
            reported_until = completed
        return embeddings[:, 0].float().cpu().numpy()

    spec = FrozenLatentStoreSpec(
        total_rows=total_rows,
        embed_dim=args.embed_dim,
        frame_skip=args.frame_skip,
        history_frames=args.history_frames,
        action_dim=args.action_dim,
        pretrained_checkpoint_sha256=checkpoint_sha256,
        dataset_source_sha256=dataset_source_sha256,
        column_normalization_sha256=column_sha256,
        git_revision=revision,
    )
    batches = iter_encoded_global_rows_once(
        total_rows=total_rows,
        batch_size=args.batch_size,
        encode_rows=encode_rows,
    )
    manifest = build_frozen_latent_store(
        args.output_dir,
        spec=spec,
        encoded_batches=batches,
        normalized_actions=normalized_actions,
        episode_ids=episode_ids,
        source_metadata={
            "checkpoint_path": str(checkpoint_file),
            "dataset_path": str(dataset_path),
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "decoded_manifest_path": str(decoded_manifest),
            "decoded_manifest_sha256": decoded.manifest_sha256,
            "decoded_binary_sha256": decoded.sha256,
            "decoded_binary_sha256_verified_this_run": decoded.sha256_verified,
            "column_normalization_path": str(column_path),
            "episode_count": int(lengths.size),
            "action_row_audit": action_row_audit,
            "stable_worldmodel_version": stable_version,
            "extraction_precision": args.precision,
            "image_preprocessing": {
                "input_dtype": "uint8",
                "scale": "divide_by_255",
                "mean": list(image_mean),
                "std": list(image_std),
                "size": args.image_size,
                "resize_mode": "bilinear_antialias",
            },
            "online_cache_parity_audit": {
                "status": "passed_by_construction_for_every_global_row",
                "latent_encoder_input_keys": ["pixels"],
                "stable_worldmodel_0_1_1_semantics": (
                    "emb depends only on pixels; action only creates act_emb; "
                    "proprio is unused"
                ),
                "cache_value": "float32_copy_of_model_encode_emb_at_time_zero",
                "extra_encoder_call_for_parity": False,
                "online_shape": ["batch", 1, args.embed_dim],
                "cache_row_shape": ["batch", args.embed_dim],
                "formal_preprocessing_smoke": preprocessing_parity_audit,
            },
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
