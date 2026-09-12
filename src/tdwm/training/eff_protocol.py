"""Explicit Eff/EffPlan protocol loading and read-only source artifact binding."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from tdwm.evaluation.actor_free_td_lewm_v1_c3 import (
    load_actor_free_td_lewm_v1_c3_evaluation_protocol,
)
from tdwm.evaluation.frozen_actor_free_td_common import _resolve_frozen_dataset_source
from tdwm.training.eff_data import EffEpisodeReplay, EpisodePartition
from tdwm.training.eff_run import EffRunSettings, run_eff_training, write_json_atomic
from tdwm.training.frozen_latent_store import FrozenLatentStore


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unresolved(value, prefix="") -> list[str]:
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        return [
            item
            for key, v in value.items()
            for item in _unresolved(v, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        return [
            item
            for i, v in enumerate(value)
            for item in _unresolved(v, f"{prefix}[{i}]")
        ]
    return []


def load_eff_protocol(path: str | Path, *, stage: str | None = None) -> dict:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected an Eff/EffPlan schema-version-1 configuration.")
    if config.get("method_family") != "effplan":
        raise ValueError("This configuration does not describe Eff/EffPlan.")
    if config["runtime"]["stable_worldmodel_version"] != "0.1.1":
        raise ValueError("Eff/EffPlan requires stable-worldmodel==0.1.1.")
    source = config["source"]
    for key in (
        "lewm_checkpoint_sha256",
        "dataset_source_sha256",
        "column_normalization_sha256",
        "frozen_store_manifest_sha256",
    ):
        digest = source[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"Invalid source binding {key}.")
    if config["data"]["episode_split"] != "rp1_cube_8000_2000":
        raise ValueError("Eff must not fall back to the C-family random clip split.")
    if config["data"]["state_stride"] != 5:
        raise ValueError("This Eff protocol uses the frozen five-step latent stride.")
    if stage is not None:
        if stage not in config or not isinstance(config[stage], dict):
            raise ValueError(f"Missing protocol stage {stage}.")
        pending = _unresolved(config[stage], stage)
        if (
            stage == "eff_training"
            and config["data"].get("efficiency_groups")
            != "time_offset_within_minibatch"
        ):
            pending.append("data.efficiency_groups")
        if config[stage].get("status") != "locked" or pending:
            raise ValueError(
                f"{stage} is not approved for formal execution; unresolved fields: {pending}. "
                "Do not silently fill these choices from unrelated baselines."
            )
    return config


def baseline_reference(config: dict, config_path: str | Path) -> dict:
    path = (
        Path(config_path).resolve().parent / config["source"]["baseline_reference"]
    ).resolve()
    return load_actor_free_td_lewm_v1_c3_evaluation_protocol(path)


def _save_array_atomic(path: Path, array: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".eff-array-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_eff_terminal_metadata(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Audit all numeric terminal rows via the public SWM dataset reader.

    This release is bound to the existing Cube dataset with zero true
    terminals and one final truncation per episode. If that evidence changes,
    fail and request a new indexing audit instead of guessing a shift.
    """
    config = load_eff_protocol(config_path)
    reference = baseline_reference(config, config_path)
    if importlib.metadata.version("stable-worldmodel") != "0.1.1":
        raise RuntimeError("Wrong stable-worldmodel version.")
    dataset_path = Path(dataset_path).expanduser().resolve()
    provenance = _resolve_frozen_dataset_source(dataset_path, reference["dataset"])
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        str(dataset_path),
        format=provenance["format"],
        keys_to_load=["terminated", "truncated"],
    )
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    if lengths.shape != (10000,) or not np.all(lengths == 201):
        raise ValueError(
            "Cube episode layout differs from the verified 10000 x 201 data."
        )
    flags = {}
    for key in ("terminated", "truncated"):
        raw = np.asarray(dataset.get_col_data(key))
        if raw.shape not in {(2010000,), (2010000, 1)}:
            raise ValueError(f"Unexpected {key} column shape.")
        raw = raw.reshape(-1)
        if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
            raise ValueError(f"{key} must contain only finite binary flags.")
        flags[key] = raw.astype(bool)
    expected_truncation = np.zeros(2010000, dtype=bool)
    expected_truncation[np.cumsum(lengths) - 1] = True
    if flags["terminated"].any() or not np.array_equal(
        flags["truncated"], expected_truncation
    ):
        raise ValueError(
            "Terminal evidence changed; re-audit its state indexing before training."
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            "Metadata output must be new; existing artifacts are preserved."
        )
    terminal_path, truncated_path = (
        output / "terminal_at_state.npy",
        output / "truncated_at_state.npy",
    )
    _save_array_atomic(terminal_path, flags["terminated"])
    _save_array_atomic(truncated_path, flags["truncated"])
    manifest = {
        "format": "tdwm-eff-terminal-metadata-v1",
        "dataset_source_sha256": config["source"]["dataset_source_sha256"],
        "dataset": provenance,
        "episode_lengths": lengths.tolist(),
        "total_rows": 2010000,
        "true_terminal_count": 0,
        "truncation_count": 10000,
        "terminal_semantics": "verified_zero_true_terminals_never_promote_truncation",
        "files": {
            "terminal": {
                "path": terminal_path.name,
                "sha256": sha256_file(terminal_path),
            },
            "truncated": {
                "path": truncated_path.name,
                "sha256": sha256_file(truncated_path),
            },
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def load_eff_replays(
    *,
    config: dict,
    latent_store: str | Path,
    terminal_metadata: str | Path,
) -> tuple[EffEpisodeReplay, EffEpisodeReplay, dict]:
    source = config["source"]
    latent_store = Path(latent_store).expanduser().resolve()
    if (
        sha256_file(latent_store / "manifest.json")
        != source["frozen_store_manifest_sha256"]
    ):
        raise ValueError(
            "The frozen store manifest differs from the declared artifact."
        )
    store = FrozenLatentStore(
        latent_store,
        expected_checkpoint_sha256=source["lewm_checkpoint_sha256"],
        expected_dataset_source_sha256=source["dataset_source_sha256"],
        expected_column_normalization_sha256=source["column_normalization_sha256"],
        expected_frame_skip=5,
        expected_history_frames=3,
        expected_embed_dim=192,
        expected_action_dim=5,
    )
    terminal_metadata = Path(terminal_metadata).expanduser().resolve()
    manifest_path = terminal_metadata / "manifest.json"
    terminal_manifest = json.loads(manifest_path.read_text())
    if terminal_manifest.get("format") != "tdwm-eff-terminal-metadata-v1":
        raise ValueError("Unrecognized terminal metadata artifact.")
    if (
        terminal_manifest.get("terminal_semantics")
        != "verified_zero_true_terminals_never_promote_truncation"
    ):
        raise ValueError("Terminal semantics are not approved for this Cube dataset.")
    if terminal_manifest["dataset_source_sha256"] != source["dataset_source_sha256"]:
        raise ValueError(
            "Terminal metadata and frozen latents refer to different data."
        )
    terminal_entry = terminal_manifest["files"]["terminal"]
    terminal_path = (terminal_metadata / terminal_entry["path"]).resolve()
    terminal_path.relative_to(terminal_metadata)
    if sha256_file(terminal_path) != terminal_entry["sha256"]:
        raise ValueError("Terminal array failed checksum validation.")
    terminal = np.load(terminal_path, mmap_mode="r", allow_pickle=False)
    if terminal.dtype != np.bool_ or terminal.shape != (2010000,) or terminal.any():
        raise ValueError("Terminal rows differ from the verified source evidence.")
    split = EpisodePartition.rp1_cube()
    expected_ids = np.repeat(
        np.arange(10000), np.asarray(terminal_manifest["episode_lengths"])
    )
    if not np.array_equal(store.episode_ids, expected_ids):
        raise ValueError("Terminal rows and frozen episode mapping differ.")
    identity = {
        "lewm_checkpoint_sha256": source["lewm_checkpoint_sha256"],
        "dataset_source_sha256": source["dataset_source_sha256"],
        "column_normalization_sha256": source["column_normalization_sha256"],
        "frozen_store_manifest_sha256": store.manifest_sha256,
        "terminal_metadata_sha256": sha256_file(manifest_path),
        "sampling": "uniform_episode_then_uniform_positive_time_offset_then_legal_anchor",
        "efficiency_groups": config["data"]["efficiency_groups"],
        "terminal_semantics": terminal_manifest["terminal_semantics"],
    }
    train = EffEpisodeReplay(
        store, episodes=split.training, stride=5, terminal_at_state=terminal
    )
    validation = EffEpisodeReplay(
        store, episodes=split.evaluation, stride=5, terminal_at_state=terminal
    )
    return train, validation, identity


def train_eff_from_protocol(
    *,
    config_path: str | Path,
    latent_store: str | Path,
    terminal_metadata: str | Path,
    output_dir: str | Path,
    device: str,
    resume: str | Path | None = None,
) -> dict:
    config = load_eff_protocol(config_path, stage="eff_training")
    if importlib.metadata.version("stable-worldmodel") != "0.1.1":
        raise RuntimeError("Wrong stable-worldmodel version.")
    settings = EffRunSettings(**config["eff_training"]["settings"])
    if (
        settings.seed != 3072
        or settings.epochs != 10
        or settings.total_updates != 127960
    ):
        raise ValueError(
            "Formal Eff must retain the declared V1-C training budget/seed."
        )
    train, validation, identity = load_eff_replays(
        config=config,
        latent_store=latent_store,
        terminal_metadata=terminal_metadata,
    )
    return run_eff_training(
        replay=train,
        validation_replay=validation,
        settings=settings,
        source_identity=identity,
        output_dir=output_dir,
        device=device,
        resume=resume,
    )
