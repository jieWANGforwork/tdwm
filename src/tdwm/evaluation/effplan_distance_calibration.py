"""Training-only calibration of the absolute five-primitive-step distance gate."""

from pathlib import Path

import numpy as np

from tdwm.training.eff_data import EpisodePartition
from tdwm.training.eff_run import write_json_atomic
from tdwm.training.frozen_latent_store import FrozenLatentStore, file_sha256


def distance_quantile(latents, episode_ids, *, episodes, quantile=0.95, lag=5):
    if not 0 < quantile < 1 or type(lag) is not int or lag < 1:
        raise ValueError("Require a quantile in (0,1) and a positive primitive-step lag.")
    ids = np.asarray(episode_ids)
    if ids.ndim != 1 or np.any(ids[1:] < ids[:-1]):
        raise ValueError("Episode IDs must be contiguous and sorted.")
    if latents.shape != (len(ids), 192) or not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Require row-aligned 192D latents and unique selected episodes.")
    distances = []
    for episode in episodes:
        first = np.searchsorted(ids, episode, side="left")
        last = np.searchsorted(ids, episode, side="right")
        if last-first <= lag:
            raise ValueError(f"Episode {episode} has no valid {lag}-step pair.")
        states = np.asarray(latents[first:last], dtype=np.float64)
        d = np.linalg.norm(states[lag:]-states[:-lag], axis=-1)
        if not np.isfinite(d).all():
            raise ValueError("Nonfinite training distance.")
        distances.append(d)
    values = np.concatenate(distances)
    limit = float(np.quantile(values, quantile, method="linear"))
    if limit <= 0:
        raise ValueError("Degenerate training distances cannot calibrate a positive gate.")
    return dict(local_distance_limit=limit, quantile=quantile, quantile_method="linear",
                primitive_lag=lag, sampling="all_valid_starts_within_each_training_episode",
                pair_count=len(values), episode_count=len(episodes),
                distance_min=float(values.min()), distance_max=float(values.max()),
                distance_mean=float(values.mean()), norm="euclidean_float64")


def calibrate(*, store_path, config_path, output_path):
    import yaml
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Do not overwrite a distance calibration.")
    source = yaml.safe_load(Path(config_path).read_text())["source"]
    store = FrozenLatentStore(
        store_path, expected_checkpoint_sha256=source["lewm_checkpoint_sha256"],
        expected_dataset_source_sha256=source["dataset_source_sha256"],
        expected_column_normalization_sha256=source["column_normalization_sha256"],
        expected_frame_skip=5, expected_embed_dim=192,
    )
    result = distance_quantile(store.latents, store.episode_ids,
                               episodes=EpisodePartition.rp1_cube().training)
    result.update(format="effplan-training-distance-p95-v1", training_episodes=[0, 7999],
                  held_out_episodes_excluded=[8000, 9999],
                  store=str(Path(store_path).resolve()),
                  store_manifest_sha256=file_sha256(Path(store_path)/"manifest.json"),
                  config_sha256=file_sha256(config_path),
                  checkpoint_sha256=source["lewm_checkpoint_sha256"])
    write_json_atomic(output, result)
    return result
