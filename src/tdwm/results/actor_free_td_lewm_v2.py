"""Fail-closed acceptance and result archiving for Actor-Free TD-LeWM V2.

The V2 study contains six independently trained methods (C, D, F, G1, G2,
G3) and three planner score modes per method.  This module deliberately does
not infer missing values: every accepted training and every one of the eighteen
formal O50 evaluations must carry its own machine-readable evidence.

The evaluator currently relies on the success threshold implemented by
``stable-worldmodel==0.1.1``.  The protocol records 0.04 metres, but that value
is not passed through the public ``World`` constructor.  Archives therefore
lock the dependency version and disclose that limitation instead of claiming
that the threshold was independently injected at runtime.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import torch

SCHEMA_VERSION = 1
METHOD_FAMILY = "actor_free_td_lewm_v2"
IMPLEMENTATION_VERSION = "v2"
OBJECTIVE_VERSION = 0
DEPLOYMENT_CHECKPOINT_VERSION = 1
SOURCE_V1_COMMIT = "3c4e62ef2ab72387536433f27ef11bce75477e7e"
TRAINING_SEED = 3072
TRAINING_EPOCHS = 10
TRAINING_STEPS = 127_960
OPTIMIZER_STEPS_PER_EPOCH = 12_796
WORLD_MODEL_PARAMETERS = 18_034_628
PREDICTOR_PARAMETERS = 379_072
EPISODES = 50
GOAL_OFFSET = 50
PLANNING_SEED = 42
STABLE_WORLDMODEL_VERSION = "0.1.1"
SELECTION_SHA256 = "e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7"
LANCE_MANIFEST_SHA256 = (
    "9de531030c6bca21a7b3215d7abea3aaf277e68a1e4cec03c8c6e22ad0d20dcd"
)
DATASET_SOURCE_SHA256 = (
    "3cf6477768f1a2979acefa3aeb6c27c45422b8b6fbce8527419943d3e679a245"
)
SPLIT_FILE_SHA256 = "4594afb3603b4258431ff9076c82acbe3ddcaccb277940b825a99017ce83d830"
G1_NEIGHBOR_MANIFEST_SHA256 = (
    "3b2d785790d86c4c45bc10f1cf706f9fc186a02071fb4f8b586eca75a2af76f2"
)

VARIANT_ORDER = ("c", "d", "f", "g1", "g2", "g3")
SCORE_MODES = ("f_only", "g_only", "f_plus_g")
FORMAL_HORIZON_BY_SCORE_MODE = {"f_only": 5, "g_only": 1, "f_plus_g": 5}
SOURCE_V1_SHA256 = {
    "c": "88bd65c48a6c701852f50552ec8f9109d6ae8ac57c467de207aa2c652c0f59a3",
    "d": "3115fffeb83ba6ae7e0c272913fe7a1ba16d42953b2185f6a3f7b168899d819a",
    "f": "b4de1b511075d763194ad1e332d127cbe390553738162f3a402ef8847bb74fd0",
    "g1": "c224d18fcd8390247f115239c4b2db013479a062438cca92003674c739f3e24b",
    "g2": "1c290f91772b42fdf6824d92832c6fff4e2d8ca3ea08089ff1a41016ea1c2ebe",
    "g3": "b279a85b1dd0816bd5fb9724da490810d470755880639297aa13699c86c2d8fb",
}
TRAINING_PROTOCOL_SHA256 = {
    "c": "45662cd2388178de6c95ce8b6915d83918b0ec2e6ab60e34af1b5db223184763",
    "d": "52b438faef4065db14ee0bec9b03a052b6bb83ef8d155b8b682da7b5c621e8a1",
    "f": "0b235946ac6963994aff102b2db0761c3988ed65ad4918c6801f6f37c74bec7e",
    "g1": "52791f459ef651d3f3699e771b95b2f82e8f49ba0c49418354e5367f2f67e5cf",
    "g2": "639e84a5a5fceb06e25424e4ffc0f0c82c9ff48cef091746982351b13d71d3fa",
    "g3": "5d42bee43781ec2150f246071eb786f5dd299bd30c1e308a53b46ff6bafa380f",
}
EVALUATION_PROTOCOL_SHA256 = {
    "c": "ffb1bfdd34bd6453ba910a8ea8119bcd1628279a5b39bb2e439c43384ec5fddd",
    "d": "6b30626b6802f9d3ffac439dbcdaddce8b82c3b9d3461ea0a4ef8ac319c5392e",
    "f": "e6d1741e24b9c7770ff0c7b871b7a8481bdbe0448ab4bb7eb8cb27722b206a62",
    "g1": "9cf193e0b1ce30c8290919bad880ae74f0a4fba9fa8ee0c2d8c9e11ec47186b7",
    "g2": "6f43ce983139a30a06d4650747cf792e5e7c62c3f5a49b802e2dcd17b4f0f0a6",
    "g3": "543d3edce2b3c1d80508d4a86e33bbc02b94e725723c6005a1a0b9101946fc40",
}
CONFIGURED_PROTOCOL_SHA256 = {
    "c": {
        "f_only": "947b323006bc4826a16c3866466514daab0b0a69fc0424f8f52c01d16e116789",
        "g_only": "21b506c6fba473a31d6507c12d2e0ad22f129005c19c098146e5802aa7191ece",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["c"],
    },
    "d": {
        "f_only": "b5965ee6e9e5c3a5b188c97e0ba5b92644d6bc5ae3f52bc50a10aa2c4bd3564f",
        "g_only": "c51dca7286b4b252527e697c730c2a681e5f92532607c22005f1f09073fbcb0d",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["d"],
    },
    "f": {
        "f_only": "cb2c051e630e17780a54243df1c02e4fb8be2e17a11e16404345ea5395d1fb01",
        "g_only": "4e5f246167d602de478532d0201fdfbee5c78c781cc7277c3575297fee8950ec",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["f"],
    },
    "g1": {
        "f_only": "968a4fb434eda3238033fcd3eed17552bd372a566779d0e98618f3224436c535",
        "g_only": "1561da99c8d4e301368beb13fe87744396ba0095f8b47752f47b0dc41f3b5cb4",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["g1"],
    },
    "g2": {
        "f_only": "2b360d4a8c252aeacc7ca8457260e67d1a321ecf81a9d4bbc36727ff92d1d554",
        "g_only": "e7c4664ff1b3927b50ab1dc03866acdb537973fce725ad36ad00635efe80264a",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["g2"],
    },
    "g3": {
        "f_only": "5ad4399724deb497bcfaf8d2a33430a15648fa81dd8e842aa3bec6d9cb9163c2",
        "g_only": "2ee6d6e90082e52aca3db40b6078dc99f2c50b2e7a20071b68c406a1c874460d",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["g3"],
    },
}

FORMAL_CEM = {
    "solver": "CEM",
    "candidates": 300,
    "iterations": 30,
    "elites": 30,
    "initial_variance": 1.0,
    "action_block": 5,
    "frame_skip": 5,
    "receding_horizon": 1,
    "episode_budget": 100,
    "planning_seed": PLANNING_SEED,
    "solver_batch_size": 1,
    "history_len": 1,
    "warm_start": True,
    "initial_distribution": "cem_gaussian_no_actor",
}

DISPLAY_NAMES = {
    "c": "V2-C Coupled Hybrid Goal-Projected TD",
    "d": "V2-D Coupled Hybrid Goal-Value Weighted TD",
    "f": "V2-F Coupled Hybrid Same-Future Advantage",
    "g1": "V2-G1 Coupled Hybrid Neighbor Action Advantage",
    "g2": "V2-G2 Coupled Hybrid Prefix-Mean Advantage",
    "g3": "V2-G3 Coupled Hybrid Prefix-Marginal Advantage",
}
METHOD_SPECS = {
    "c": {
        "loss": "Coupled real-state and predicted-state feature TD plus goal-projected TD",
        "special": "Goal-derived tasks project both the detached target and online prediction",
    },
    "d": {
        "loss": "Coupled real-state and predicted-state feature TD with detached goal-value weights",
        "special": "Goal-subset softmax weights are normalized to mean one",
    },
    "f": {
        "loss": "Coupled real-state and predicted-state feature TD with same-future/different-goal weights",
        "special": "The matching goal is contrasted with all goal-derived tasks in the batch",
    },
    "g1": {
        "loss": "Coupled real-state and predicted-state feature TD with neighbor-action advantage weights",
        "special": "Other-episode frozen-latent KNN actions are comparison-only candidates",
    },
    "g2": {
        "loss": "Coupled real-state and predicted-state feature TD with prefix-mean advantage weights",
        "special": "The full action score is contrasted with zero-suffix action prefixes",
    },
    "g3": {
        "loss": "Coupled real-state and predicted-state feature TD with prefix-marginal advantage weights",
        "special": "Mean adjacent prefix-score gains provide detached weights",
    },
}


class V2ResultValidationError(ValueError):
    """Raised when any formal V2 training or evaluation evidence is invalid."""


@dataclass(frozen=True)
class ValidatedV2Study:
    bundle_root: Path
    acceptance: Mapping[str, Any]
    acceptance_sha256: str
    training: Mapping[str, Mapping[str, Any]]
    evaluations: Mapping[str, Mapping[str, Mapping[str, Any]]]
    selection: Mapping[str, Any]
    selection_sha256: str


class _Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> bool:
        if not condition:
            self.errors.append(message)
            return False
        return True

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, tensor metadata and bytes in a state dictionary."""

    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise V2ResultValidationError(
                "state dictionaries must map names to tensors"
            )
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise V2ResultValidationError(f"{context}: missing required file {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V2ResultValidationError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise V2ResultValidationError(f"{context}: JSON root must be an object")
    return value, raw


def _read_json_audit(path: Path, *, audit: _Audit, context: str) -> dict[str, Any]:
    try:
        return _read_json(path, context=context)[0]
    except V2ResultValidationError as error:
        audit.errors.append(str(error))
        return {}


def _load_checkpoint_audit(
    path: Path, *, audit: _Audit, context: str
) -> dict[str, Any]:
    import torch

    if not audit.require(path.is_file(), f"{context}: missing {path}"):
        return {}
    if not audit.require(path.stat().st_size > 0, f"{context}: checkpoint is empty"):
        return {}
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    except Exception as error:  # pragma: no cover - backend-specific detail
        audit.errors.append(f"{context}: torch.load failed: {error}")
        return {}
    if not isinstance(value, dict):
        audit.errors.append(f"{context}: checkpoint root must be a mapping")
        return {}
    return value


def _mapping_audit(value: Any, *, audit: _Audit, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        audit.errors.append(f"{context}: expected a mapping")
        return {}
    return dict(value)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V2ResultValidationError(f"{context}: expected an object")
    return value


def _exact_audit(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    audit: _Audit,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        audit.require(
            value.get(key) == expected_value,
            f"{context}.{key}: expected {expected_value!r}, got {value.get(key)!r}",
        )


def _exact(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise V2ResultValidationError(
                f"{context}.{key}: expected {expected_value!r}, got {value.get(key)!r}"
            )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise V2ResultValidationError(f"{context}: must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise V2ResultValidationError(f"{context}: must be numeric") from error
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise V2ResultValidationError(f"{context}: must be finite and positive")
    return number


def _same_path(value: Any, expected: Any) -> bool:
    if not isinstance(value, (str, os.PathLike)) or not isinstance(
        expected, (str, os.PathLike)
    ):
        return False
    return Path(value).expanduser().resolve() == Path(expected).expanduser().resolve()


def _substate(
    state: Mapping[str, Any], prefix: str, *, audit: _Audit, context: str
) -> dict[str, torch.Tensor]:
    import torch

    selected = {
        str(key)[len(prefix) :]: tensor
        for key, tensor in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not audit.require(bool(selected), f"{context}: missing {prefix!r} substate"):
        return {}
    if not audit.require(
        all(isinstance(tensor, torch.Tensor) for tensor in selected.values()),
        f"{context}: {prefix!r} substate contains non-tensors",
    ):
        return {}
    return selected


def _tensor_mapping(
    value: Any, *, audit: _Audit, context: str
) -> dict[str, torch.Tensor]:
    import torch

    mapping = _mapping_audit(value, audit=audit, context=context)
    if not audit.require(
        all(
            isinstance(key, str) and isinstance(tensor, torch.Tensor)
            for key, tensor in mapping.items()
        ),
        f"{context}: state must map string names to tensors",
    ):
        return {}
    return dict(mapping)


def _finite_state(
    state: Mapping[str, torch.Tensor], *, audit: _Audit, context: str
) -> dict[str, int]:
    import torch

    stats = {"tensor_count": 0, "floating_tensor_count": 0, "tensor_numel": 0}
    for key, tensor in state.items():
        stats["tensor_count"] += 1
        stats["tensor_numel"] += int(tensor.numel())
        if tensor.is_floating_point() or tensor.is_complex():
            stats["floating_tensor_count"] += 1
            flat = tensor.detach()
            if flat.layout != torch.strided:
                flat = flat.to_dense()
            flat = flat.reshape(-1)
            for start in range(0, flat.numel(), 1_000_000):
                if not bool(torch.isfinite(flat[start : start + 1_000_000]).all()):
                    audit.errors.append(f"{context}.{key}: contains NaN or infinity")
                    break
    return stats


def _state_shapes_match(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    audit: _Audit,
    context: str,
) -> None:
    if not audit.require(set(left) == set(right), f"{context}: state keys differ"):
        return
    for key in left:
        audit.require(
            left[key].shape == right[key].shape and left[key].dtype == right[key].dtype,
            f"{context}.{key}: tensor shape or dtype differs",
        )


def _metric_number(row: Mapping[str, str], names: Sequence[str]) -> float | None:
    for name in names:
        raw = row.get(name)
        if raw not in (None, ""):
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
    return None


def _audit_metrics(run_dir: Path, *, audit: _Audit, variant: str) -> dict[str, Any]:
    paths = sorted(run_dir.glob("metrics/version_*/metrics.csv"))
    audit.require(bool(paths), f"{variant}.metrics: no metrics/version_*/metrics.csv")
    by_epoch: dict[int, dict[str, list[float]]] = {
        epoch: {
            "train_loss": [],
            "train_base_hybrid_td": [],
            "validation_loss": [],
            "validation_base_hybrid_td": [],
        }
        for epoch in range(TRAINING_EPOCHS)
    }
    steps: list[int] = []
    files: list[dict[str, Any]] = []
    aliases = {
        "train_loss": ("train/loss_epoch", "train/loss"),
        "train_base_hybrid_td": (
            "train/base_hybrid_td_loss_epoch",
            "train/base_hybrid_td_loss",
        ),
        "validation_loss": ("validation/loss", "validation/loss_epoch"),
        "validation_base_hybrid_td": (
            "validation/base_hybrid_td_loss",
            "validation/base_hybrid_td_loss_epoch",
        ),
    }
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except Exception as error:
            audit.errors.append(f"{variant}.metrics: cannot read {path}: {error}")
            continue
        files.append(
            {"path": str(path), "sha256": _file_sha256(path), "rows": len(rows)}
        )
        for row in rows:
            if row.get("step") not in (None, ""):
                try:
                    steps.append(int(float(row["step"])))
                except (TypeError, ValueError):
                    audit.errors.append(f"{variant}.metrics: non-integral step")
            if row.get("epoch") in (None, ""):
                continue
            try:
                epoch = int(float(row["epoch"]))
            except (TypeError, ValueError):
                audit.errors.append(f"{variant}.metrics: non-integral epoch")
                continue
            if epoch not in by_epoch:
                continue
            for name, candidates in aliases.items():
                number = _metric_number(row, candidates)
                if number is not None:
                    by_epoch[epoch][name].append(number)
    audit.require(
        bool(steps) and max(steps) == TRAINING_STEPS - 1,
        f"{variant}.metrics: final zero-based step must be {TRAINING_STEPS - 1}",
    )
    curve: list[dict[str, Any]] = []
    for epoch, values_by_name in by_epoch.items():
        item: dict[str, Any] = {"epoch": epoch + 1}
        for name, values in values_by_name.items():
            audit.require(
                bool(values), f"{variant}.metrics: epoch {epoch + 1} missing {name}"
            )
            if values:
                audit.require(
                    all(
                        math.isclose(value, values[-1], rel_tol=1e-8, abs_tol=1e-10)
                        for value in values
                    ),
                    f"{variant}.metrics: epoch {epoch + 1} has conflicting {name}",
                )
                item[name] = values[-1]
        curve.append(item)
    return {
        "files": files,
        "epochs": curve,
        "final_step": max(steps) if steps else None,
    }


def _execution_evidence_path(
    output_root: Path,
    run_dir: Path,
    variant: str,
    evidence_root: Path | None,
) -> Path:
    candidates = [
        run_dir / "execution_evidence.json",
        output_root / variant / "execution_evidence.json",
    ]
    if evidence_root is not None:
        candidates.extend(
            (
                evidence_root / variant / "execution_evidence.json",
                evidence_root / f"{variant}.json",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _audit_execution_evidence(
    path: Path,
    *,
    audit: _Audit,
    variant: str,
    method: str,
    training_revision: str | None,
    source_checkpoint_path: Path,
    deployment_checkpoint_path: Path,
    lightning_last_path: Path,
    lightning_last_payload: Mapping[str, Any],
    dataset: Mapping[str, Any],
    neighbor: Any,
) -> dict[str, Any]:
    evidence = _read_json_audit(
        path, audit=audit, context=f"{variant}.execution_evidence"
    )
    _exact_audit(
        evidence,
        {
            "schema_version": 1,
            "source": "v2_formal_training_launcher",
            "method": method,
            "variant": variant,
        },
        audit=audit,
        context=f"{variant}.execution_evidence",
    )
    audit.require(
        isinstance(evidence.get("hostname"), str) and bool(evidence.get("hostname")),
        f"{variant}.execution_evidence.hostname must be non-empty",
    )
    gpu = _mapping_audit(
        evidence.get("gpu"), audit=audit, context=f"{variant}.execution_evidence.gpu"
    )
    audit.require(
        isinstance(gpu.get("index"), int)
        and not isinstance(gpu.get("index"), bool)
        and gpu["index"] >= 0,
        f"{variant}.execution_evidence.gpu.index must be non-negative",
    )
    for key in ("name", "uuid"):
        audit.require(
            isinstance(gpu.get(key), str) and bool(gpu.get(key)),
            f"{variant}.execution_evidence.gpu.{key} must be non-empty",
        )
    process = _mapping_audit(
        evidence.get("process"),
        audit=audit,
        context=f"{variant}.execution_evidence.process",
    )
    argv = process.get("argv")
    audit.require(
        isinstance(argv, list)
        and len(argv) >= 2
        and all(isinstance(argument, str) and argument for argument in argv),
        f"{variant}.execution_evidence.process.argv must be a non-empty string array",
    )
    if isinstance(argv, list) and all(isinstance(argument, str) for argument in argv):
        audit.require(
            any(
                f"train_actor_free_td_lewm_v2_{variant}.py" in argument
                for argument in argv
            ),
            f"{variant}.execution_evidence.process.argv identifies the wrong method",
        )
        audit.require(
            process.get("argv_sha256") == canonical_sha256(argv),
            f"{variant}.execution_evidence.process.argv_sha256 disagrees",
        )
    _exact_audit(
        process,
        {
            "git_revision": training_revision,
            "git_clean": True,
            "return_code": 0,
        },
        audit=audit,
        context=f"{variant}.execution_evidence.process",
    )
    audit.require(
        _is_git_revision(process.get("git_revision")),
        f"{variant}.execution_evidence.process.git_revision must be a full lowercase SHA",
    )
    audit.require(
        isinstance(process.get("pid"), int) and process["pid"] > 0,
        f"{variant}.execution_evidence.process.pid must be positive",
    )
    for key in ("cwd", "started_at_utc", "ended_at_utc"):
        audit.require(
            isinstance(process.get(key), str) and bool(process.get(key)),
            f"{variant}.execution_evidence.process.{key} must be non-empty",
        )
    timestamps: dict[str, datetime] = {}
    for key in ("started_at_utc", "ended_at_utc"):
        value = process.get(key)
        if isinstance(value, str):
            try:
                timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                audit.errors.append(
                    f"{variant}.execution_evidence.process.{key} is not ISO-8601"
                )
            else:
                audit.require(
                    timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
                    f"{variant}.execution_evidence.process.{key} must be UTC",
                )
                timestamps[key] = timestamp
    if set(timestamps) == {"started_at_utc", "ended_at_utc"}:
        audit.require(
            timestamps["ended_at_utc"] >= timestamps["started_at_utc"],
            f"{variant}.execution_evidence process end precedes start",
        )
    log = _mapping_audit(
        evidence.get("log"), audit=audit, context=f"{variant}.execution_evidence.log"
    )
    for key in ("path", "sha256"):
        audit.require(
            isinstance(log.get(key), str) and bool(log.get(key)),
            f"{variant}.execution_evidence.log.{key} must be non-empty",
        )
    audit.require(
        _is_sha256(log.get("sha256")),
        f"{variant}.execution_evidence.log.sha256 must be SHA-256",
    )
    audit.require(
        isinstance(log.get("size_bytes"), int) and log["size_bytes"] >= 0,
        f"{variant}.execution_evidence.log.size_bytes must be non-negative",
    )
    log_path = Path(str(log.get("path", ""))).expanduser().resolve()
    if audit.require(
        log_path.is_file(), f"{variant}.execution_evidence.log is missing"
    ):
        audit.require(
            log_path.stat().st_size == log.get("size_bytes"),
            f"{variant}.execution_evidence.log.size_bytes disagrees with the file",
        )
        audit.require(
            _file_sha256(log_path) == log.get("sha256"),
            f"{variant}.execution_evidence.log.sha256 disagrees with the file",
        )

    inputs = _mapping_audit(
        evidence.get("inputs"),
        audit=audit,
        context=f"{variant}.execution_evidence.inputs",
    )
    dataset_input = _mapping_audit(
        inputs.get("dataset"),
        audit=audit,
        context=f"{variant}.execution_evidence.inputs.dataset",
    )
    audit.require(
        _same_path(dataset_input.get("path"), dataset.get("path")),
        f"{variant}.execution_evidence dataset path differs from training manifest",
    )
    audit.require(
        _same_path(
            dataset_input.get("manifest_path"), dataset.get("conversion_manifest_path")
        ),
        f"{variant}.execution_evidence dataset manifest path differs",
    )
    audit.require(
        dataset_input.get("manifest_sha256") == LANCE_MANIFEST_SHA256,
        f"{variant}.execution_evidence Lance manifest SHA differs",
    )
    dataset_manifest_path = (
        Path(str(dataset_input.get("manifest_path", ""))).expanduser().resolve()
    )
    if audit.require(
        dataset_manifest_path.is_file(),
        f"{variant}.execution_evidence Lance manifest is missing",
    ):
        audit.require(
            _file_sha256(dataset_manifest_path) == LANCE_MANIFEST_SHA256,
            f"{variant}.execution_evidence Lance manifest file SHA differs",
        )

    checkpoint_input = _mapping_audit(
        inputs.get("initial_v1_checkpoint"),
        audit=audit,
        context=f"{variant}.execution_evidence.inputs.initial_v1_checkpoint",
    )
    audit.require(
        _same_path(checkpoint_input.get("path"), source_checkpoint_path),
        f"{variant}.execution_evidence source V1 checkpoint path differs",
    )
    audit.require(
        checkpoint_input.get("sha256") == SOURCE_V1_SHA256[variant],
        f"{variant}.execution_evidence source V1 checkpoint SHA differs",
    )

    split_input = _mapping_audit(
        inputs.get("split_indices"),
        audit=audit,
        context=f"{variant}.execution_evidence.inputs.split_indices",
    )
    audit.require(
        split_input.get("sha256") == SPLIT_FILE_SHA256,
        f"{variant}.execution_evidence split file SHA differs",
    )
    split_path = Path(str(split_input.get("path", ""))).expanduser().resolve()
    if audit.require(
        split_path.is_file(), f"{variant}.execution_evidence split file is missing"
    ):
        audit.require(
            _file_sha256(split_path) == SPLIT_FILE_SHA256,
            f"{variant}.execution_evidence split file SHA disagrees with the file",
        )

    neighbor_input = inputs.get("neighbor_index")
    if variant == "g1":
        neighbor_mapping = _mapping_audit(
            neighbor_input,
            audit=audit,
            context="g1.execution_evidence.inputs.neighbor_index",
        )
        recorded_neighbor = _mapping_audit(
            neighbor, audit=audit, context="g1.training_manifest.neighbor_index"
        )
        audit.require(
            _same_path(neighbor_mapping.get("path"), recorded_neighbor.get("path")),
            "g1.execution_evidence neighbor index path differs",
        )
        audit.require(
            neighbor_mapping.get("manifest_sha256") == G1_NEIGHBOR_MANIFEST_SHA256,
            "g1.execution_evidence neighbor manifest SHA differs",
        )
        neighbor_manifest_path = (
            Path(str(neighbor_mapping.get("path", ""))).expanduser().resolve()
            / "manifest.json"
        )
        if audit.require(
            neighbor_manifest_path.is_file(),
            "g1.execution_evidence neighbor manifest is missing",
        ):
            audit.require(
                _file_sha256(neighbor_manifest_path) == G1_NEIGHBOR_MANIFEST_SHA256,
                "g1.execution_evidence neighbor manifest file SHA differs",
            )
    else:
        audit.require(
            neighbor_input is None,
            f"{variant}.execution_evidence neighbor_index must be null",
        )

    outputs = _mapping_audit(
        evidence.get("outputs"),
        audit=audit,
        context=f"{variant}.execution_evidence.outputs",
    )
    deployment_output = _mapping_audit(
        outputs.get("deployment_checkpoint"),
        audit=audit,
        context=f"{variant}.execution_evidence.outputs.deployment_checkpoint",
    )
    audit.require(
        _same_path(deployment_output.get("path"), deployment_checkpoint_path),
        f"{variant}.execution_evidence deployment checkpoint path differs",
    )
    _exact_audit(
        deployment_output,
        {
            "epoch": TRAINING_EPOCHS,
            "global_step": TRAINING_STEPS,
            "size_bytes": (
                deployment_checkpoint_path.stat().st_size
                if deployment_checkpoint_path.is_file()
                else None
            ),
        },
        audit=audit,
        context=f"{variant}.execution_evidence.outputs.deployment_checkpoint",
    )
    deployment_sha = (
        _file_sha256(deployment_checkpoint_path)
        if deployment_checkpoint_path.is_file()
        else None
    )
    audit.require(
        deployment_output.get("sha256") == deployment_sha
        and _is_sha256(deployment_sha),
        f"{variant}.execution_evidence deployment checkpoint SHA differs",
    )

    lightning_output = _mapping_audit(
        outputs.get("lightning_last"),
        audit=audit,
        context=f"{variant}.execution_evidence.outputs.lightning_last",
    )
    audit.require(
        _same_path(lightning_output.get("path"), lightning_last_path),
        f"{variant}.execution_evidence Lightning last path differs",
    )
    _exact_audit(
        lightning_output,
        {
            "size_bytes": (
                lightning_last_path.stat().st_size
                if lightning_last_path.is_file()
                else None
            )
        },
        audit=audit,
        context=f"{variant}.execution_evidence.outputs.lightning_last",
    )
    lightning_sha = (
        _file_sha256(lightning_last_path) if lightning_last_path.is_file() else None
    )
    audit.require(
        lightning_output.get("sha256") == lightning_sha and _is_sha256(lightning_sha),
        f"{variant}.execution_evidence Lightning last SHA differs",
    )
    actual_resume_identity = _mapping_audit(
        lightning_last_payload.get("v2_resume_identity"),
        audit=audit,
        context=f"{variant}.last.v2_resume_identity",
    )
    _exact_audit(
        actual_resume_identity,
        {"v2_start_revision": training_revision},
        audit=audit,
        context=f"{variant}.last.v2_resume_identity",
    )
    audit.require(
        lightning_output.get("resume_identity") == actual_resume_identity,
        f"{variant}.execution_evidence Lightning resume identity differs from last.ckpt",
    )
    audit.require(
        _is_git_revision(actual_resume_identity.get("v2_start_revision")),
        f"{variant}.execution_evidence checkpoint resume revision is invalid",
    )

    disk = _mapping_audit(
        evidence.get("disk"),
        audit=audit,
        context=f"{variant}.execution_evidence.disk",
    )
    for key in ("free_bytes_before", "free_bytes_after"):
        audit.require(
            isinstance(disk.get(key), int)
            and not isinstance(disk.get(key), bool)
            and disk[key] >= 0,
            f"{variant}.execution_evidence.disk.{key} must be non-negative",
        )
    return evidence


def _default_world_model_inspector(
    config: Mapping[str, Any],
    online_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> int:
    """Instantiate the complete model and strictly load both V2 world states."""

    import hydra
    from omegaconf import OmegaConf

    model = hydra.utils.instantiate(OmegaConf.create(dict(config)))
    count = sum(parameter.numel() for parameter in model.parameters())
    model.load_state_dict(dict(online_state), strict=True)
    model.load_state_dict(dict(target_state), strict=True)
    del model
    gc.collect()
    return int(count)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def audit_training(
    *,
    output_root: str | Path,
    evidence_root: str | Path | None = None,
    world_model_inspector: Callable[
        [Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
        int,
    ] = _default_world_model_inspector,
) -> dict[str, Any]:
    """Audit the six completed formal V2 trainings without inventing evidence."""

    output_root = Path(output_root).expanduser().resolve()
    resolved_evidence_root = (
        Path(evidence_root).expanduser().resolve()
        if evidence_root is not None
        else None
    )
    audit = _Audit()
    audit.require(output_root.is_dir(), f"training root is missing: {output_root}")
    acceptance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "seed": TRAINING_SEED,
        "expected_epoch": TRAINING_EPOCHS,
        "expected_global_step": TRAINING_STEPS,
        "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
        "predictor_parameter_count": PREDICTOR_PARAMETERS,
        "stable_worldmodel_version": STABLE_WORLDMODEL_VERSION,
        "variants": {},
    }
    world_config_hashes: dict[str, str] = {}
    split_fingerprints: dict[str, str] = {}
    dataset_fingerprints: dict[str, str] = {}
    training_revisions: dict[str, str] = {}

    for variant in VARIANT_ORDER:
        method = f"{METHOD_FAMILY}_{variant}"
        run_dir = output_root / variant / f"seed_{TRAINING_SEED}"
        result_path = run_dir / "training_result.json"
        manifest_path = run_dir / "training_manifest.json"
        deployment_path = (
            run_dir
            / "checkpoints"
            / method
            / variant
            / f"epoch_{TRAINING_EPOCHS:02d}.pt"
        )
        last_path = run_dir / "checkpoints" / "lightning" / "last.ckpt"
        result = _read_json_audit(
            result_path, audit=audit, context=f"{variant}.training_result"
        )
        manifest = _read_json_audit(
            manifest_path, audit=audit, context=f"{variant}.training_manifest"
        )
        _exact_audit(
            result,
            {
                "method": method,
                "method_family": METHOD_FAMILY,
                "variant": variant,
                "implementation_version": IMPLEMENTATION_VERSION,
                "seed": TRAINING_SEED,
                "final_epoch": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
                "source_v1_checkpoint_sha256": SOURCE_V1_SHA256[variant],
            },
            audit=audit,
            context=f"{variant}.training_result",
        )
        audit.require(
            _same_path(result.get("run_dir"), run_dir),
            f"{variant}.training_result.run_dir differs from the formal path",
        )
        audit.require(
            _same_path(result.get("deployment_checkpoint"), deployment_path),
            f"{variant}.training_result.deployment_checkpoint is wrong",
        )
        audit.require(
            _same_path(result.get("last_checkpoint"), last_path),
            f"{variant}.training_result.last_checkpoint is wrong",
        )
        _exact_audit(
            manifest,
            {
                "method": method,
                "method_family": METHOD_FAMILY,
                "variant": variant,
                "implementation_version": IMPLEMENTATION_VERSION,
                "objective_version": OBJECTIVE_VERSION,
                "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
                "seed": TRAINING_SEED,
            },
            audit=audit,
            context=f"{variant}.training_manifest",
        )
        protocol = _mapping_audit(
            manifest.get("protocol"), audit=audit, context=f"{variant}.protocol"
        )
        protocol_sha = canonical_sha256(protocol)
        audit.require(
            protocol_sha == TRAINING_PROTOCOL_SHA256[variant],
            f"{variant}.protocol differs from the locked resolved V2 protocol",
        )
        audit.require(
            manifest.get("protocol_sha256") == protocol_sha,
            f"{variant}.training_manifest.protocol_sha256 disagrees",
        )
        _exact_audit(
            protocol,
            {
                "method": method,
                "method_family": METHOD_FAMILY,
                "variant": variant,
                "implementation_version": IMPLEMENTATION_VERSION,
                "stage": "coupled_hybrid_finetuning",
                "initialization": "corresponding_v1_deployment_finetune",
                "seeds": [TRAINING_SEED],
            },
            audit=audit,
            context=f"{variant}.protocol",
        )
        runtime = _mapping_audit(
            manifest.get("runtime"), audit=audit, context=f"{variant}.runtime"
        )
        _exact_audit(
            runtime,
            {"stable_worldmodel": STABLE_WORLDMODEL_VERSION},
            audit=audit,
            context=f"{variant}.runtime",
        )
        runtime_revision = runtime.get("tdwm_git_revision")
        audit.require(
            _is_git_revision(runtime_revision),
            f"{variant}.runtime.tdwm_git_revision must be a full lowercase SHA",
        )
        if isinstance(runtime_revision, str):
            training_revisions[variant] = runtime_revision
        audit.require(
            isinstance(runtime.get("cuda_device"), str)
            and bool(runtime.get("cuda_device")),
            f"{variant}.runtime.cuda_device must be recorded",
        )
        audit.require(
            isinstance(result.get("peak_cuda_memory_bytes"), int)
            and result["peak_cuda_memory_bytes"] > 0,
            f"{variant}.training_result.peak_cuda_memory_bytes must be positive",
        )
        training = _mapping_audit(
            manifest.get("training"), audit=audit, context=f"{variant}.training"
        )
        _exact_audit(
            training,
            {
                "formal_optimizer_steps": TRAINING_STEPS,
                "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
                "configured_optimizer_steps": TRAINING_STEPS,
                "optimizer_initialized_fresh": True,
                "validation_skipped": False,
            },
            audit=audit,
            context=f"{variant}.training",
        )
        audit.require(
            isinstance(training.get("validation_batches"), int)
            and training["validation_batches"] > 0,
            f"{variant}.training.validation_batches must be positive",
        )
        model = _mapping_audit(
            manifest.get("model"), audit=audit, context=f"{variant}.model"
        )
        _exact_audit(
            model,
            {
                "world_model_parameters": WORLD_MODEL_PARAMETERS,
                "predictor_parameters": PREDICTOR_PARAMETERS,
                "online_world_model_trainable": True,
                "online_action_encoder_trainable": True,
                "target_world_model_trainable": False,
                "target_predictor_trainable": False,
            },
            audit=audit,
            context=f"{variant}.model",
        )
        source_v1 = _mapping_audit(
            manifest.get("source_v1"), audit=audit, context=f"{variant}.source_v1"
        )
        _exact_audit(
            source_v1,
            {
                "method": f"actor_free_td_lewm_v1_{variant}",
                "method_family": "actor_free_td_lewm_v1",
                "variant": variant,
                "implementation_version": "v1",
                "objective_version": 0,
                "deployment_checkpoint_version": 1,
                "source_seed": TRAINING_SEED,
                "source_epoch": TRAINING_EPOCHS,
                "source_global_step": TRAINING_STEPS,
                "source_code_revision": SOURCE_V1_COMMIT,
                "checkpoint_sha256": SOURCE_V1_SHA256[variant],
                "optimizer_state": "reset",
                "optimizer_state_loaded": False,
            },
            audit=audit,
            context=f"{variant}.source_v1",
        )
        dataset = _mapping_audit(
            manifest.get("dataset"), audit=audit, context=f"{variant}.dataset"
        )
        _exact_audit(
            dataset,
            {"format": "lance"},
            audit=audit,
            context=f"{variant}.dataset",
        )
        conversion_path = (
            Path(str(dataset.get("conversion_manifest_path", "")))
            .expanduser()
            .resolve()
        )
        if audit.require(
            conversion_path.is_file(),
            f"{variant}.dataset conversion manifest is missing",
        ):
            audit.require(
                _file_sha256(conversion_path) == LANCE_MANIFEST_SHA256,
                f"{variant}.dataset conversion manifest SHA differs",
            )
        conversion = _mapping_audit(
            dataset.get("conversion_manifest"),
            audit=audit,
            context=f"{variant}.dataset.conversion_manifest",
        )
        conversion_source = _mapping_audit(
            conversion.get("source"),
            audit=audit,
            context=f"{variant}.dataset.conversion_manifest.source",
        )
        audit.require(
            conversion_source.get("sha256") == DATASET_SOURCE_SHA256,
            f"{variant}.dataset source SHA differs",
        )
        split = _mapping_audit(
            dataset.get("split"), audit=audit, context=f"{variant}.dataset.split"
        )
        _exact_audit(
            split,
            {
                "file_sha256": SPLIT_FILE_SHA256,
                "train_indices_sha256": (
                    "a1665554b6f5dc1c4aa37768cd7008fdc96f6a55ec5e8e12d9a93afa99880561"
                ),
                "validation_indices_sha256": (
                    "e5aed8baa556f3f868ed471c511488df2117332837303ba958df278b34a61a6c"
                ),
            },
            audit=audit,
            context=f"{variant}.dataset.split",
        )
        split_fingerprints[variant] = canonical_sha256(
            {
                "train_indices_sha256": split.get("train_indices_sha256"),
                "validation_indices_sha256": split.get("validation_indices_sha256"),
            }
        )
        dataset_fingerprints[variant] = canonical_sha256(
            {key: value for key, value in dataset.items() if key != "path"}
        )
        neighbor = manifest.get("neighbor_index")
        if variant == "g1":
            neighbor_mapping = _mapping_audit(
                neighbor, audit=audit, context="g1.neighbor_index"
            )
            audit.require(
                neighbor_mapping.get("manifest_sha256")
                == G1_NEIGHBOR_MANIFEST_SHA256
                == protocol.get("source_artifacts", {}).get(
                    "g1_neighbor_index_manifest_sha256"
                ),
                "g1.neighbor_index manifest SHA differs from the locked protocol",
            )
        else:
            audit.require(
                neighbor is None, f"{variant}: only G1 may bind a neighbor index"
            )

        metric_audit = _audit_metrics(run_dir, audit=audit, variant=variant)
        deployment = _load_checkpoint_audit(
            deployment_path, audit=audit, context=f"{variant}.epoch10"
        )
        expected_schema = {
            "method",
            "method_family",
            "variant",
            "implementation_version",
            "objective_version",
            "deployment_checkpoint_version",
            "epoch",
            "global_step",
            "world_model_state_dict",
            "target_world_model_state_dict",
            "world_model_config",
            "predictor_state_dict",
            "target_predictor_state_dict",
            "predictor_config",
            "source_v1_provenance",
        }
        audit.require(
            set(deployment) == expected_schema,
            f"{variant}.epoch10: deployment checkpoint schema differs",
        )
        _exact_audit(
            deployment,
            {
                "method": method,
                "method_family": METHOD_FAMILY,
                "variant": variant,
                "implementation_version": IMPLEMENTATION_VERSION,
                "objective_version": OBJECTIVE_VERSION,
                "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
                "epoch": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
            },
            audit=audit,
            context=f"{variant}.epoch10",
        )
        world_state = _tensor_mapping(
            deployment.get("world_model_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.world_model_state_dict",
        )
        target_world_state = _tensor_mapping(
            deployment.get("target_world_model_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.target_world_model_state_dict",
        )
        predictor_state = _tensor_mapping(
            deployment.get("predictor_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.predictor_state_dict",
        )
        target_predictor_state = _tensor_mapping(
            deployment.get("target_predictor_state_dict"),
            audit=audit,
            context=f"{variant}.epoch10.target_predictor_state_dict",
        )
        _state_shapes_match(
            world_state,
            target_world_state,
            audit=audit,
            context=f"{variant}.online_target_world",
        )
        _state_shapes_match(
            predictor_state,
            target_predictor_state,
            audit=audit,
            context=f"{variant}.online_target_predictor",
        )
        for key in set(predictor_state) | set(target_predictor_state):
            audit.require(
                "action_encoder" not in key
                and "actor" not in key
                and "successor" not in key,
                f"{variant}.predictor has forbidden duplicated module key {key!r}",
            )
        audit.require(
            sum(tensor.numel() for tensor in predictor_state.values())
            == PREDICTOR_PARAMETERS,
            f"{variant}.predictor parameter count differs from {PREDICTOR_PARAMETERS}",
        )
        world_config = _mapping_audit(
            deployment.get("world_model_config"),
            audit=audit,
            context=f"{variant}.epoch10.world_model_config",
        )
        world_config_hash = canonical_sha256(world_config)
        world_config_hashes[variant] = world_config_hash
        action_state = _substate(
            world_state,
            "action_encoder.",
            audit=audit,
            context=f"{variant}.online_world",
        )
        target_action_state = _substate(
            target_world_state,
            "action_encoder.",
            audit=audit,
            context=f"{variant}.target_world",
        )
        try:
            instantiated_parameter_count = world_model_inspector(
                world_config, world_state, target_world_state
            )
        except Exception as error:
            audit.errors.append(
                f"{variant}.world_model strict inspection failed: {error}"
            )
            instantiated_parameter_count = None
        audit.require(
            instantiated_parameter_count == WORLD_MODEL_PARAMETERS,
            f"{variant}.world_model instantiated parameter count differs",
        )
        source_path_value = source_v1.get("checkpoint_path")
        source_path = (
            Path(source_path_value).expanduser().resolve()
            if isinstance(source_path_value, str)
            else Path("/__missing_v1_source__")
        )
        audit.require(
            source_path.is_file(), f"{variant}.source_v1 checkpoint is missing"
        )
        source_checkpoint_sha = (
            _file_sha256(source_path) if source_path.is_file() else None
        )
        audit.require(
            source_checkpoint_sha == SOURCE_V1_SHA256[variant],
            f"{variant}.source_v1 checkpoint SHA differs",
        )
        source_payload = _load_checkpoint_audit(
            source_path, audit=audit, context=f"{variant}.source_v1_checkpoint"
        )
        source_world_config = _mapping_audit(
            source_payload.get("world_model_config"),
            audit=audit,
            context=f"{variant}.source_v1_checkpoint.world_model_config",
        )
        source_world_config_hash = canonical_sha256(source_world_config)
        audit.require(
            source_world_config_hash == world_config_hash,
            f"{variant}: V2 full world_model_config differs from its V1 source",
        )
        provenance = _mapping_audit(
            deployment.get("source_v1_provenance"),
            audit=audit,
            context=f"{variant}.epoch10.source_v1_provenance",
        )
        _exact_audit(
            provenance,
            {
                "checkpoint_sha256": SOURCE_V1_SHA256[variant],
                "source_epoch": TRAINING_EPOCHS,
                "source_global_step": TRAINING_STEPS,
                "optimizer_state_loaded": False,
                "target_world_initialization": "copy_of_v1_online_world_model",
            },
            audit=audit,
            context=f"{variant}.epoch10.source_v1_provenance",
        )

        last = _load_checkpoint_audit(last_path, audit=audit, context=f"{variant}.last")
        _exact_audit(
            last,
            {"epoch": TRAINING_EPOCHS - 1, "global_step": TRAINING_STEPS},
            audit=audit,
            context=f"{variant}.last",
        )
        resume_identity = _mapping_audit(
            last.get("v2_resume_identity"),
            audit=audit,
            context=f"{variant}.last.v2_resume_identity",
        )
        _exact_audit(
            resume_identity,
            {"v2_start_revision": runtime_revision},
            audit=audit,
            context=f"{variant}.last.v2_resume_identity",
        )
        required_rng = {
            "v2_data_generator_state",
            "v2_goal_generator_state",
            "v2_task_generator_state",
            "v2_validation_goal_generator_state",
            "v2_validation_task_generator_state",
            "v2_validation_goal_epoch_state",
            "v2_validation_task_epoch_state",
        }
        audit.require(
            required_rng <= set(last), f"{variant}.last resume RNG state is incomplete"
        )
        audit.require(
            isinstance(last.get("optimizer_states"), list)
            and bool(last.get("optimizer_states")),
            f"{variant}.last optimizer state is missing",
        )
        lightning_state = _tensor_mapping(
            last.get("state_dict"), audit=audit, context=f"{variant}.last.state_dict"
        )
        last_world = _substate(
            lightning_state, "model.", audit=audit, context=f"{variant}.last"
        )
        last_target_world = _substate(
            lightning_state, "target_model.", audit=audit, context=f"{variant}.last"
        )
        last_predictor = _substate(
            lightning_state, "predictor.", audit=audit, context=f"{variant}.last"
        )
        last_target_predictor = _substate(
            lightning_state,
            "target_predictor.",
            audit=audit,
            context=f"{variant}.last",
        )
        if all(
            (world_state, target_world_state, predictor_state, target_predictor_state)
        ):
            audit.require(
                state_dict_sha256(last_world) == state_dict_sha256(world_state),
                f"{variant}: Lightning/export online world states differ",
            )
            audit.require(
                state_dict_sha256(last_target_world)
                == state_dict_sha256(target_world_state),
                f"{variant}: Lightning/export target world states differ",
            )
            audit.require(
                state_dict_sha256(last_predictor) == state_dict_sha256(predictor_state),
                f"{variant}: Lightning/export online predictor states differ",
            )
            audit.require(
                state_dict_sha256(last_target_predictor)
                == state_dict_sha256(target_predictor_state),
                f"{variant}: Lightning/export target predictor states differ",
            )
        finite_stats = {
            "online_world": _finite_state(
                world_state, audit=audit, context=f"{variant}.online_world"
            ),
            "target_world": _finite_state(
                target_world_state, audit=audit, context=f"{variant}.target_world"
            ),
            "online_predictor": _finite_state(
                predictor_state, audit=audit, context=f"{variant}.online_predictor"
            ),
            "target_predictor": _finite_state(
                target_predictor_state,
                audit=audit,
                context=f"{variant}.target_predictor",
            ),
        }
        evidence_path = _execution_evidence_path(
            output_root, run_dir, variant, resolved_evidence_root
        )
        execution = _audit_execution_evidence(
            evidence_path,
            audit=audit,
            variant=variant,
            method=method,
            training_revision=(
                runtime_revision if isinstance(runtime_revision, str) else None
            ),
            source_checkpoint_path=source_path,
            deployment_checkpoint_path=deployment_path,
            lightning_last_path=last_path,
            lightning_last_payload=last,
            dataset=dataset,
            neighbor=neighbor,
        )
        checkpoint_sha = (
            _file_sha256(deployment_path) if deployment_path.is_file() else None
        )
        acceptance["variants"][variant] = {
            "method": method,
            "run_dir": str(run_dir),
            "training_result_path": str(result_path),
            "training_result_sha256": _file_sha256(result_path)
            if result_path.is_file()
            else None,
            "training_manifest_path": str(manifest_path),
            "training_manifest_sha256": _file_sha256(manifest_path)
            if manifest_path.is_file()
            else None,
            "protocol_sha256": protocol_sha,
            "checkpoint_path": str(deployment_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": deployment.get("epoch"),
            "checkpoint_global_step": deployment.get("global_step"),
            "training_revision": runtime_revision,
            "checkpoint_bytes": deployment_path.stat().st_size
            if deployment_path.is_file()
            else None,
            "world_model_config_sha256": world_config_hash,
            "source_v1_world_model_config_sha256": source_world_config_hash,
            "online_world_state_sha256": state_dict_sha256(world_state)
            if world_state
            else None,
            "target_world_state_sha256": state_dict_sha256(target_world_state)
            if target_world_state
            else None,
            "online_predictor_state_sha256": state_dict_sha256(predictor_state)
            if predictor_state
            else None,
            "target_predictor_state_sha256": (
                state_dict_sha256(target_predictor_state)
                if target_predictor_state
                else None
            ),
            "online_action_encoder_state_sha256": (
                state_dict_sha256(action_state) if action_state else None
            ),
            "target_action_encoder_state_sha256": (
                state_dict_sha256(target_action_state) if target_action_state else None
            ),
            "world_model_parameter_count": instantiated_parameter_count,
            "predictor_parameter_count": sum(
                tensor.numel() for tensor in predictor_state.values()
            ),
            "finite_state_stats": finite_stats,
            "metrics": metric_audit,
            "source_v1": {
                "checkpoint_path": str(source_path),
                "checkpoint_sha256": source_checkpoint_sha,
                "epoch": source_payload.get("epoch"),
                "global_step": source_payload.get("global_step"),
            },
            "execution_evidence_path": str(evidence_path),
            "execution_evidence_sha256": (
                _file_sha256(evidence_path) if evidence_path.is_file() else None
            ),
            "execution_evidence": execution,
            "last_checkpoint_path": str(last_path),
            "last_checkpoint_sha256": (
                _file_sha256(last_path) if last_path.is_file() else None
            ),
        }
        del deployment, last, source_payload
        gc.collect()

    audit.require(
        len(world_config_hashes) == len(VARIANT_ORDER)
        and len(set(world_config_hashes.values())) == 1,
        "cross-run: six methods do not share one complete world_model_config",
    )
    audit.require(
        len(split_fingerprints) == len(VARIANT_ORDER)
        and len(set(split_fingerprints.values())) == 1,
        "cross-run: six methods do not share one training split",
    )
    audit.require(
        len(dataset_fingerprints) == len(VARIANT_ORDER)
        and len(set(dataset_fingerprints.values())) == 1,
        "cross-run: six methods do not share one training dataset fingerprint",
    )
    audit.require(
        len(training_revisions) == len(VARIANT_ORDER)
        and len(set(training_revisions.values())) == 1,
        "cross-run: six methods do not share one final clean training revision",
    )
    acceptance["common_world_model_config_sha256"] = (
        next(iter(world_config_hashes.values()))
        if len(set(world_config_hashes.values())) == 1
        else None
    )
    acceptance["training_revision"] = (
        next(iter(training_revisions.values()))
        if len(set(training_revisions.values())) == 1
        else None
    )
    acceptance["warnings"] = audit.warnings
    acceptance["errors"] = audit.errors
    acceptance["status"] = (
        "FAIL" if audit.errors else "PASS_WITH_WARNINGS" if audit.warnings else "PASS"
    )
    return acceptance


def write_training_acceptance(acceptance: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically publish one acceptance record, including FAIL evidence."""

    destination = Path(path).expanduser().resolve()
    _atomic_write_json(destination, acceptance)
    return destination


def _validate_training_acceptance(
    bundle_root: Path,
    *,
    acceptance_path: Path | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, Mapping[str, Any]]]:
    path = acceptance_path or bundle_root / "training_acceptance.json"
    acceptance, raw = _read_json(path, context="training_acceptance")
    _exact(
        acceptance,
        {
            "schema_version": SCHEMA_VERSION,
            "seed": TRAINING_SEED,
            "expected_epoch": TRAINING_EPOCHS,
            "expected_global_step": TRAINING_STEPS,
            "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
            "predictor_parameter_count": PREDICTOR_PARAMETERS,
            "stable_worldmodel_version": STABLE_WORLDMODEL_VERSION,
        },
        context="training_acceptance",
    )
    training_revision = acceptance.get("training_revision")
    if not _is_git_revision(training_revision):
        raise V2ResultValidationError(
            "training_acceptance.training_revision must be a full lowercase SHA"
        )
    status = acceptance.get("status")
    if status not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise V2ResultValidationError(
            "training_acceptance.status must be PASS or PASS_WITH_WARNINGS"
        )
    errors = acceptance.get("errors")
    if errors not in (None, []):
        raise V2ResultValidationError("accepted training cannot retain errors")
    warnings = acceptance.get("warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) or not item for item in warnings
    ):
        raise V2ResultValidationError("training_acceptance.warnings is invalid")
    if status == "PASS" and warnings:
        raise V2ResultValidationError("PASS cannot hide warnings")
    if status == "PASS_WITH_WARNINGS" and not warnings:
        raise V2ResultValidationError("PASS_WITH_WARNINGS must disclose warnings")
    common_config_hash = acceptance.get("common_world_model_config_sha256")
    if not _is_sha256(common_config_hash):
        raise V2ResultValidationError(
            "training_acceptance.common_world_model_config_sha256 is invalid"
        )
    variants = _mapping(
        acceptance.get("variants"), context="training_acceptance.variants"
    )
    if set(variants) != set(VARIANT_ORDER):
        raise V2ResultValidationError(
            "training_acceptance must contain exactly c,d,f,g1,g2,g3"
        )
    training: dict[str, Mapping[str, Any]] = {}
    for variant in VARIANT_ORDER:
        item = _mapping(variants[variant], context=f"training_acceptance.{variant}")
        method = f"{METHOD_FAMILY}_{variant}"
        _exact(
            item,
            {
                "method": method,
                "checkpoint_epoch": TRAINING_EPOCHS,
                "checkpoint_global_step": TRAINING_STEPS,
                "training_revision": training_revision,
                "world_model_config_sha256": common_config_hash,
                "source_v1_world_model_config_sha256": common_config_hash,
                "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
                "predictor_parameter_count": PREDICTOR_PARAMETERS,
                "protocol_sha256": TRAINING_PROTOCOL_SHA256[variant],
            },
            context=f"training_acceptance.{variant}",
        )
        for key in (
            "checkpoint_sha256",
            "online_world_state_sha256",
            "target_world_state_sha256",
            "online_predictor_state_sha256",
            "target_predictor_state_sha256",
            "online_action_encoder_state_sha256",
            "target_action_encoder_state_sha256",
            "training_result_sha256",
            "training_manifest_sha256",
            "execution_evidence_sha256",
            "last_checkpoint_sha256",
        ):
            if not _is_sha256(item.get(key)):
                raise V2ResultValidationError(
                    f"training_acceptance.{variant}.{key} must be SHA-256"
                )
        checkpoint_path = (
            Path(str(item.get("checkpoint_path", ""))).expanduser().resolve()
        )
        if not checkpoint_path.is_file():
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: checkpoint is missing"
            )
        if _file_sha256(checkpoint_path) != item["checkpoint_sha256"]:
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: checkpoint SHA changed after acceptance"
            )
        for path_key, sha_key in (
            ("training_result_path", "training_result_sha256"),
            ("training_manifest_path", "training_manifest_sha256"),
            ("execution_evidence_path", "execution_evidence_sha256"),
            ("last_checkpoint_path", "last_checkpoint_sha256"),
        ):
            source = Path(str(item.get(path_key, ""))).expanduser().resolve()
            if not source.is_file() or _file_sha256(source) != item[sha_key]:
                raise V2ResultValidationError(
                    f"training_acceptance.{variant}: {path_key} is missing or changed"
                )
        training_manifest, _ = _read_json(
            Path(str(item["training_manifest_path"])),
            context=f"training_acceptance.{variant}.training_manifest",
        )
        manifest_runtime = _mapping(
            training_manifest.get("runtime"),
            context=f"training_acceptance.{variant}.runtime",
        )
        if manifest_runtime.get("tdwm_git_revision") != training_revision:
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: manifest revision differs"
            )
        execution, _ = _read_json(
            Path(str(item["execution_evidence_path"])),
            context=f"training_acceptance.{variant}.execution_evidence",
        )
        process = _mapping(
            execution.get("process"),
            context=f"training_acceptance.{variant}.execution_evidence.process",
        )
        if process.get("git_revision") != training_revision:
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: evidence revision differs"
            )
        outputs = _mapping(
            execution.get("outputs"),
            context=f"training_acceptance.{variant}.execution_evidence.outputs",
        )
        lightning_output = _mapping(
            outputs.get("lightning_last"),
            context=(
                f"training_acceptance.{variant}.execution_evidence.lightning_last"
            ),
        )
        resume_identity = _mapping(
            lightning_output.get("resume_identity"),
            context=f"training_acceptance.{variant}.resume_identity",
        )
        if resume_identity.get("v2_start_revision") != training_revision:
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: resume revision differs"
            )
        source_v1 = _mapping(
            item.get("source_v1"), context=f"training_acceptance.{variant}.source_v1"
        )
        _exact(
            source_v1,
            {
                "checkpoint_sha256": SOURCE_V1_SHA256[variant],
                "epoch": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
            },
            context=f"training_acceptance.{variant}.source_v1",
        )
        metrics = _mapping(
            item.get("metrics"), context=f"training_acceptance.{variant}.metrics"
        )
        epochs = metrics.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != TRAINING_EPOCHS:
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: metrics must cover ten epochs"
            )
        if metrics.get("final_step") != TRAINING_STEPS - 1:
            raise V2ResultValidationError(
                f"training_acceptance.{variant}: metrics final step is wrong"
            )
        training[variant] = item
    return acceptance, raw, training


def _validate_selection(
    selection: Mapping[str, Any], *, raw: bytes, context: str
) -> None:
    if hashlib.sha256(raw).hexdigest() != SELECTION_SHA256:
        raise V2ResultValidationError(
            f"{context}: episode_selection.json is not the locked seed-42 O50 set"
        )
    keys = ("episode_indices", "start_steps", "goal_steps", "valid_row_ranks")
    for key in keys:
        values = selection.get(key)
        if (
            not isinstance(values, list)
            or len(values) != EPISODES
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            )
        ):
            raise V2ResultValidationError(f"{context}.{key} must contain 50 integers")
    for episode, start, goal in zip(
        selection["episode_indices"],
        selection["start_steps"],
        selection["goal_steps"],
    ):
        if not 0 <= episode < 10_000 or not 0 <= start < goal < 201:
            raise V2ResultValidationError(f"{context}: invalid episode/start/goal")
        if goal - start != GOAL_OFFSET:
            raise V2ResultValidationError(f"{context}: pair is not O50")


def _validate_action_normalization(value: Mapping[str, Any], *, context: str) -> None:
    for key in ("mean", "scale", "variance"):
        entries = value.get(key)
        if not isinstance(entries, list) or len(entries) != 5:
            raise V2ResultValidationError(f"{context}.{key} must contain five values")
        numbers = [_finite(entry, context=f"{context}.{key}") for entry in entries]
        if key == "scale" and any(number <= 0 for number in numbers):
            raise V2ResultValidationError(f"{context}.scale must be positive")
        if key == "variance" and any(number < 0 for number in numbers):
            raise V2ResultValidationError(f"{context}.variance must be non-negative")
    if not isinstance(value.get("samples"), int) or value["samples"] <= 0:
        raise V2ResultValidationError(f"{context}.samples must be positive")


def _without_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_paths(item)
            for key, item in value.items()
            if key != "path" and not str(key).endswith("_path")
        }
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return deepcopy(value)


def _successes(metrics: Mapping[str, Any], *, context: str) -> tuple[bool, ...]:
    canonical = metrics.get("episode_successes")
    legacy = metrics.get("success")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise V2ResultValidationError(
            f"{context}: episode_successes and legacy success disagree"
        )
    values = canonical if canonical is not None else legacy
    if (
        not isinstance(values, list)
        or len(values) != EPISODES
        or any(not isinstance(value, bool) for value in values)
    ):
        raise V2ResultValidationError(
            f"{context}: metrics.episode_successes must contain 50 booleans"
        )
    return tuple(values)


def _validate_evaluation(
    run_root: Path,
    *,
    variant: str,
    score_mode: str,
    training: Mapping[str, Any],
    training_revision: str,
) -> dict[str, Any]:
    context = f"{variant}/{score_mode}"
    result, result_raw = _read_json(
        run_root / "results.json", context=f"{context}.result"
    )
    manifest, manifest_raw = _read_json(
        run_root / "protocol_manifest.json", context=f"{context}.manifest"
    )
    selection, selection_raw = _read_json(
        run_root / "episode_selection.json", context=f"{context}.selection"
    )
    action, action_raw = _read_json(
        run_root / "action_normalization.json", context=f"{context}.action"
    )
    _validate_selection(selection, raw=selection_raw, context=f"{context}.selection")
    _validate_action_normalization(action, context=f"{context}.action")
    if manifest.get("selection") != selection:
        raise V2ResultValidationError(f"{context}: manifest selection differs")
    normalization = _mapping(
        manifest.get("normalization"), context=f"{context}.normalization"
    )
    if normalization.get("action") != action:
        raise V2ResultValidationError(f"{context}: action normalization differs")

    formal = _mapping(
        manifest.get("formal_protocol"), context=f"{context}.formal_protocol"
    )
    formal_sha = canonical_sha256(formal)
    if formal_sha != EVALUATION_PROTOCOL_SHA256[variant]:
        raise V2ResultValidationError(f"{context}: formal protocol hash is wrong")
    method = f"{METHOD_FAMILY}_{variant}"
    _exact(
        formal,
        {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "environment": "cube",
            "stage": "planner_evaluation",
        },
        context=f"{context}.formal_protocol",
    )
    configured = _mapping(manifest.get("protocol"), context=f"{context}.protocol")
    configured_sha = canonical_sha256(configured)
    if configured_sha != CONFIGURED_PROTOCOL_SHA256[variant][score_mode]:
        raise V2ResultValidationError(f"{context}: configured protocol hash is wrong")
    expected_configured = deepcopy(dict(formal))
    expected_configured["inference_objective"]["score_mode"] = score_mode
    expected_configured["planning"]["horizon"] = FORMAL_HORIZON_BY_SCORE_MODE[
        score_mode
    ]
    if configured != expected_configured:
        raise V2ResultValidationError(
            f"{context}: configured protocol changed beyond score mode and horizon"
        )
    if manifest.get("score_mode") != score_mode:
        raise V2ResultValidationError(f"{context}: manifest score mode is wrong")
    inference = _mapping(
        configured.get("inference_objective"), context=f"{context}.inference_objective"
    )
    _exact(
        inference,
        {
            "score_mode": score_mode,
            "f_score": "lewm_rollout_goal_distance",
            "f_score_reducer": "final_predicted_latent_summed_mse",
            "g_score": "negative_goal_projection_of_v2_online_predictor",
            "f_plus_g_split": "first_h_minus_one_blocks_with_f_last_block_with_g",
            "f_plus_g_combination": "prefix_final_f_cost_minus_gamma_power_tail_g_score",
            "g_only_horizon": 1,
            "goal_enters_predictor": True,
            "learned_actor": False,
            "deployed_world_model": "online_v2_world_model",
            "deployed_predictor": "online_v2_predictor",
            "target_modules_used_at_evaluation": False,
            "deployed_modules_frozen": True,
            "training_only_auxiliary_used_at_evaluation": False,
        },
        context=f"{context}.inference_objective",
    )
    expected_horizon = FORMAL_HORIZON_BY_SCORE_MODE[score_mode]
    planning = _mapping(configured.get("planning"), context=f"{context}.planning")
    _exact(
        planning,
        {**FORMAL_CEM, "horizon": expected_horizon},
        context=f"{context}.planning",
    )
    evaluation = _mapping(configured.get("evaluation"), context=f"{context}.evaluation")
    _exact(
        evaluation,
        {"episodes": EPISODES, "goal_offset": GOAL_OFFSET},
        context=f"{context}.evaluation",
    )
    world = _mapping(configured.get("world"), context=f"{context}.world")
    _exact(
        world,
        {"success_threshold_meters": 0.04, "terminate_at_goal": True},
        context=f"{context}.world",
    )
    checkpoint = _mapping(manifest.get("checkpoint"), context=f"{context}.checkpoint")
    _exact(
        checkpoint,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "epoch": TRAINING_EPOCHS,
            "global_step": TRAINING_STEPS,
            "formal_completion_required": True,
        },
        context=f"{context}.checkpoint",
    )
    if not _same_path(checkpoint.get("path"), Path(str(training["checkpoint_path"]))):
        raise V2ResultValidationError(
            f"{context}: evaluation checkpoint path differs from training acceptance"
        )
    if checkpoint.get("sha256") != training["checkpoint_sha256"]:
        raise V2ResultValidationError(
            f"{context}: evaluation checkpoint SHA differs from training acceptance"
        )
    predictor_config = _mapping(
        checkpoint.get("predictor_config"), context=f"{context}.predictor_config"
    )
    _exact(
        predictor_config,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "deployment_checkpoint_version": DEPLOYMENT_CHECKPOINT_VERSION,
            "architecture": "td_jepa_forward_map_v1",
            "state_dim": 192,
            "raw_action_dim": 25,
            "action_dim": 192,
            "action_embedding_dim": 192,
            "task_dim": 192,
            "output_dim": 192,
            "num_parallel": 1,
            "action_processing": "online_shared_lewm_action_encoder",
            "shared_lewm_action_encoder": True,
            "action_encoder_trainable": True,
            "action_encoder_source": "world_model.action_encoder",
            "actor": "none",
            "reward": "none",
        },
        context=f"{context}.predictor_config",
    )
    for key in ("task_sampling", "joint_objective", "source_v1", "source_artifacts"):
        if predictor_config.get(key) != formal.get(key):
            raise V2ResultValidationError(
                f"{context}: predictor_config.{key} differs from formal protocol"
            )
    provenance = _mapping(
        checkpoint.get("source_v1_provenance"),
        context=f"{context}.source_v1_provenance",
    )
    _exact(
        provenance,
        {
            "checkpoint_sha256": SOURCE_V1_SHA256[variant],
            "source_epoch": TRAINING_EPOCHS,
            "source_global_step": TRAINING_STEPS,
            "optimizer_state_loaded": False,
            "target_world_initialization": "copy_of_v1_online_world_model",
        },
        context=f"{context}.source_v1_provenance",
    )
    dataset = _mapping(manifest.get("dataset"), context=f"{context}.dataset")
    _exact(
        dataset,
        {
            "format": "lance",
            "episodes": 10_000,
            "transitions": 2_010_000,
            "source_sha256": DATASET_SOURCE_SHA256,
            "conversion_manifest_sha256": LANCE_MANIFEST_SHA256,
        },
        context=f"{context}.dataset",
    )
    runtime = _mapping(manifest.get("runtime"), context=f"{context}.runtime")
    _exact(
        runtime,
        {
            "stable_worldmodel": STABLE_WORLDMODEL_VERSION,
            "tdwm_git_revision": training_revision,
            "device": "cuda",
        },
        context=f"{context}.runtime",
    )
    for key in ("torch", "python", "platform", "cuda_device"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise V2ResultValidationError(f"{context}.runtime.{key} is missing")
    _exact(
        result,
        {
            "method": method,
            "method_family": METHOD_FAMILY,
            "variant": variant,
            "implementation_version": IMPLEMENTATION_VERSION,
            "score_mode": score_mode,
            "planning_horizon": expected_horizon,
            "smoke": False,
            "pilot": False,
            "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
            "predictor_parameter_count": PREDICTOR_PARAMETERS,
        },
        context=f"{context}.result",
    )
    metrics = _mapping(result.get("metrics"), context=f"{context}.metrics")
    successes = _successes(metrics, context=context)
    success_count = sum(successes)
    percent = _finite(metrics.get("success_rate"), context=f"{context}.success_rate")
    if not math.isclose(percent, 100.0 * success_count / EPISODES, abs_tol=1e-12):
        raise V2ResultValidationError(
            f"{context}: success_rate disagrees with outcomes"
        )
    elapsed = _finite(
        result.get("elapsed_seconds"),
        context=f"{context}.elapsed_seconds",
        positive=True,
    )
    return {
        "variant": variant,
        "method": method,
        "score_mode": score_mode,
        "planning_horizon": expected_horizon,
        "successes": successes,
        "success_count": success_count,
        "success_rate": success_count / EPISODES,
        "elapsed_seconds": elapsed,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "formal_protocol_sha256": formal_sha,
        "configured_protocol_sha256": configured_sha,
        "dataset_fingerprint": canonical_sha256(_without_paths(dataset)),
        "runtime_fingerprint": canonical_sha256(
            {
                key: runtime.get(key)
                for key in (
                    "stable_worldmodel",
                    "torch",
                    "python",
                    "platform",
                    "device",
                    "compatibility_adapter",
                )
            }
        ),
        "normalization_fingerprint": canonical_sha256(action),
        "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "source_files_sha256": {
            "results.json": hashlib.sha256(result_raw).hexdigest(),
            "protocol_manifest.json": hashlib.sha256(manifest_raw).hexdigest(),
            "episode_selection.json": hashlib.sha256(selection_raw).hexdigest(),
            "action_normalization.json": hashlib.sha256(action_raw).hexdigest(),
        },
        "manifest": manifest,
        "result": result,
    }


def validate_bundle(
    bundle_root: str | Path,
    *,
    evaluation_root: str | Path | None = None,
    acceptance_path: str | Path | None = None,
) -> ValidatedV2Study:
    """Validate one complete formal V2 training plus 6 x 3 O50 bundle.

    The launcher layout is ``<bundle>/evaluations/<variant>/<score_mode>``;
    ``<bundle>/o50`` remains an unambiguous legacy alias. Explicit paths are
    supported for export bundles. Unknown or missing directories are rejected.
    """

    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise V2ResultValidationError(f"bundle is not a directory: {root}")
    resolved_acceptance_path = (
        Path(acceptance_path).expanduser().resolve()
        if acceptance_path is not None
        else root / "training_acceptance.json"
    )
    acceptance, acceptance_raw, training = _validate_training_acceptance(
        root, acceptance_path=resolved_acceptance_path
    )
    if evaluation_root is not None:
        resolved_evaluation_root = Path(evaluation_root).expanduser().resolve()
    else:
        candidates = [root / "evaluations", root / "o50"]
        existing = [candidate for candidate in candidates if candidate.is_dir()]
        if len(existing) != 1:
            raise V2ResultValidationError(
                "bundle must contain exactly one evaluations or o50 directory; "
                "pass evaluation_root for an intentional alternate layout"
            )
        resolved_evaluation_root = existing[0]
    if not resolved_evaluation_root.is_dir():
        raise V2ResultValidationError(
            f"missing formal evaluation root: {resolved_evaluation_root}"
        )
    variant_directories = {
        item.name for item in resolved_evaluation_root.iterdir() if item.is_dir()
    }
    if variant_directories != set(VARIANT_ORDER):
        raise V2ResultValidationError(
            "evaluation root must contain exactly c,d,f,g1,g2,g3"
        )

    evaluations: dict[str, dict[str, Mapping[str, Any]]] = {}
    common_selection: Mapping[str, Any] | None = None
    selection_fingerprints: set[str] = set()
    normalization_fingerprints: set[str] = set()
    dataset_fingerprints: set[str] = set()
    runtime_fingerprints: set[str] = set()
    training_revision = str(acceptance["training_revision"])
    for variant in VARIANT_ORDER:
        variant_root = resolved_evaluation_root / variant
        mode_directories = {
            item.name for item in variant_root.iterdir() if item.is_dir()
        }
        if mode_directories != set(SCORE_MODES):
            raise V2ResultValidationError(
                f"evaluation/{variant} must contain exactly f_only,g_only,f_plus_g"
            )
        evaluations[variant] = {}
        method_checkpoint_hashes: set[str] = set()
        method_checkpoint_paths: set[str] = set()
        for score_mode in SCORE_MODES:
            run = _validate_evaluation(
                variant_root / score_mode,
                variant=variant,
                score_mode=score_mode,
                training=training[variant],
                training_revision=training_revision,
            )
            selection = _mapping(
                run["manifest"].get("selection"),
                context=f"{variant}/{score_mode}.selection",
            )
            if common_selection is None:
                common_selection = deepcopy(dict(selection))
            elif selection != common_selection:
                raise V2ResultValidationError(
                    f"{variant}/{score_mode}: all 18 runs must use identical pairs"
                )
            selection_fingerprints.add(str(run["selection_sha256"]))
            normalization_fingerprints.add(str(run["normalization_fingerprint"]))
            dataset_fingerprints.add(str(run["dataset_fingerprint"]))
            runtime_fingerprints.add(str(run["runtime_fingerprint"]))
            method_checkpoint_hashes.add(str(run["checkpoint_sha256"]))
            method_checkpoint_paths.add(
                str(Path(str(run["checkpoint_path"])).expanduser().resolve())
            )
            evaluations[variant][score_mode] = run
        if len(method_checkpoint_hashes) != 1 or len(method_checkpoint_paths) != 1:
            raise V2ResultValidationError(
                f"{variant}: three score modes did not use one identical checkpoint"
            )
    if common_selection is None:  # pragma: no cover - guarded by fixed loops
        raise V2ResultValidationError("formal evaluation grid is empty")
    if selection_fingerprints != {SELECTION_SHA256}:
        raise V2ResultValidationError(
            "the 18 runs do not share the locked O50 selection"
        )
    if len(normalization_fingerprints) != 1:
        raise V2ResultValidationError(
            "the 18 runs do not share one action-normalization artifact"
        )
    if len(dataset_fingerprints) != 1:
        raise V2ResultValidationError(
            "the 18 runs do not share one audited evaluation dataset"
        )
    if len(runtime_fingerprints) != 1:
        raise V2ResultValidationError(
            "the 18 runs do not share one software/runtime contract"
        )
    if sum(len(items) for items in evaluations.values()) != 18:
        raise V2ResultValidationError("formal evaluation grid is not exactly 6 x 3")
    return ValidatedV2Study(
        bundle_root=root,
        acceptance=acceptance,
        acceptance_sha256=hashlib.sha256(acceptance_raw).hexdigest(),
        training=training,
        evaluations=evaluations,
        selection=common_selection,
        selection_sha256=SELECTION_SHA256,
    )


def _pair_hash(episode: int, start: int, goal: int) -> str:
    return canonical_sha256(
        {"episode_index": episode, "start_step": start, "goal_step": goal}
    )


def _ranking(study: ValidatedV2Study) -> list[dict[str, Any]]:
    rows = [
        {
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "success_count": study.evaluations[variant]["f_plus_g"]["success_count"],
            "success_rate": study.evaluations[variant]["f_plus_g"]["success_rate"],
        }
        for variant in VARIANT_ORDER
    ]
    rows.sort(
        key=lambda item: (
            -int(item["success_count"]),
            VARIANT_ORDER.index(item["variant"]),
        )
    )
    previous: int | None = None
    rank = 0
    for position, item in enumerate(rows, 1):
        count = int(item["success_count"])
        if count != previous:
            rank = position
            previous = count
        item["rank"] = rank
    return rows


def build_summary(study: ValidatedV2Study) -> dict[str, Any]:
    """Build a deterministic, provenance-rich summary of the formal study."""

    methods: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        evaluations = study.evaluations[variant]
        curves = []
        for raw in training["metrics"]["epochs"]:
            item = dict(raw)
            item["train_method_objective"] = item["train_loss"]
            item["validation_common_base_td"] = item["validation_base_hybrid_td"]
            curves.append(item)
        methods[variant] = {
            "method": training["method"],
            "display_name": DISPLAY_NAMES[variant],
            "network": (
                "Jointly fine-tuned online LeWM (including its single shared 25D-to-192D "
                "action encoder) plus one 379,072-parameter TD-JEPA predictor; EMA targets"
            ),
            "training_loss": METHOD_SPECS[variant]["loss"],
            "special_mechanism": METHOD_SPECS[variant]["special"],
            "inference": {
                "f_only": "Online LeWM final predicted-latent goal cost, horizon 5",
                "g_only": "Negative goal projection of the online V2 predictor, horizon 1",
                "f_plus_g": "Online LeWM F prefix and final online V2 G tail, horizon 5",
            },
            "training": {
                "seed": TRAINING_SEED,
                "epochs": TRAINING_EPOCHS,
                "global_step": TRAINING_STEPS,
                "checkpoint_sha256": training["checkpoint_sha256"],
                "protocol_canonical_sha256": training["protocol_sha256"],
                "source_v1": deepcopy(dict(training["source_v1"])),
                "world_model_config_canonical_sha256": training[
                    "world_model_config_sha256"
                ],
                "world_model_parameter_count": training["world_model_parameter_count"],
                "predictor_parameter_count": training["predictor_parameter_count"],
                "state_sha256": {
                    key: training[key]
                    for key in (
                        "online_world_state_sha256",
                        "target_world_state_sha256",
                        "online_predictor_state_sha256",
                        "target_predictor_state_sha256",
                        "online_action_encoder_state_sha256",
                        "target_action_encoder_state_sha256",
                    )
                },
                "source_files_sha256": {
                    "training_result.json": training["training_result_sha256"],
                    "training_manifest.json": training["training_manifest_sha256"],
                    "execution_evidence.json": training["execution_evidence_sha256"],
                    "last.ckpt": training["last_checkpoint_sha256"],
                },
                "loss_curve": curves,
                "train_loss_semantics": "method_specific_coupled_hybrid_objective",
                "validation_loss_semantics": "common_base_hybrid_td",
            },
            "evaluations": {
                score_mode: {
                    "planning_horizon": evaluations[score_mode]["planning_horizon"],
                    "success_count": evaluations[score_mode]["success_count"],
                    "success_rate": evaluations[score_mode]["success_rate"],
                    "elapsed_seconds": evaluations[score_mode]["elapsed_seconds"],
                    "checkpoint_sha256": evaluations[score_mode]["checkpoint_sha256"],
                    "formal_protocol_canonical_sha256": evaluations[score_mode][
                        "formal_protocol_sha256"
                    ],
                    "configured_protocol_canonical_sha256": evaluations[score_mode][
                        "configured_protocol_sha256"
                    ],
                    "source_files_sha256": deepcopy(
                        dict(evaluations[score_mode]["source_files_sha256"])
                    ),
                }
                for score_mode in SCORE_MODES
            },
            "f_plus_g_minus_f_only_percentage_points": 100.0
            * (
                evaluations["f_plus_g"]["success_rate"]
                - evaluations["f_only"]["success_rate"]
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "study": {
            "id": "actor_free_td_lewm_v2_cube_seed3072_o50_6x3",
            "method_family": METHOD_FAMILY,
            "training_revision": study.acceptance["training_revision"],
            "environment": "cube",
            "training_seed": TRAINING_SEED,
            "planning_seed": PLANNING_SEED,
            "goal_offset": GOAL_OFFSET,
            "episodes_per_evaluation": EPISODES,
            "training_count": 6,
            "evaluation_count": 18,
            "score_modes": list(SCORE_MODES),
            "formal_horizon_by_score_mode": dict(FORMAL_HORIZON_BY_SCORE_MODE),
            "ranking_metric": "f_plus_g_success_count_only",
            "single_training_seed": True,
            "single_planning_selection": True,
        },
        "architecture": {
            "online_lewm_trainable": True,
            "target_lewm": "ema_full_world_model",
            "single_shared_action_encoder": True,
            "action_processing": (
                "raw25_to_world_model.action_encoder_to_embedding192_for_data_and_CEM"
            ),
            "state_dim": 192,
            "raw_action_dim": 25,
            "action_embedding_dim": 192,
            "task_dim": 192,
            "output_dim": 192,
            "world_model_parameter_count": WORLD_MODEL_PARAMETERS,
            "predictor_parameter_count": PREDICTOR_PARAMETERS,
            "actor": "none",
            "reward": "none",
        },
        "training_acceptance": {
            "status": study.acceptance["status"],
            "warnings": list(study.acceptance.get("warnings", [])),
            "sha256": study.acceptance_sha256,
            "common_world_model_config_canonical_sha256": study.acceptance[
                "common_world_model_config_sha256"
            ],
        },
        "selection": {
            "episode_selection_json_sha256": study.selection_sha256,
            "episode_count": EPISODES,
        },
        "success_threshold_contract": {
            "configured_meters": 0.04,
            "explicit_constructor_argument_supported": False,
            "runtime_authority": "stable-worldmodel==0.1.1 World implementation",
            "disclosure": (
                "The public World constructor used by this evaluator does not accept an "
                "explicit success-threshold argument. The protocol records 0.04 m, while "
                "acceptance locks stable-worldmodel 0.1.1 instead of claiming injection."
            ),
        },
        "ranking_by_f_plus_g": _ranking(study),
        "methods": methods,
        "validation": {
            "complete_6x3_bundle": True,
            "formal_o50_only": True,
            "smoke_or_pilot_runs": 0,
            "common_selection_across_18_runs": True,
            "same_checkpoint_within_each_method": True,
            "mode_specific_horizon_5_1_5": True,
            "full_world_model_config_hash_bound": True,
            "online_and_target_state_subhashes_bound": True,
            "world_model_and_predictor_parameter_counts_bound": True,
            "success_rates_recomputed_from_50_boolean_outcomes": True,
        },
    }


def build_paired_outcomes_csv(study: ValidatedV2Study) -> bytes:
    """Return one row per locked pair and one success column per O50 cell."""

    stream = io.StringIO(newline="")
    success_columns = [
        f"success_{variant}__{score_mode}"
        for variant in VARIANT_ORDER
        for score_mode in SCORE_MODES
    ]
    fields = [
        "selection_position",
        "selection_sha256",
        "episode_index",
        "start_step",
        "goal_step",
        "valid_row_rank",
        "pair_hash",
        *success_columns,
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for index in range(EPISODES):
        episode = int(study.selection["episode_indices"][index])
        start = int(study.selection["start_steps"][index])
        goal = int(study.selection["goal_steps"][index])
        row: dict[str, Any] = {
            "selection_position": index,
            "selection_sha256": study.selection_sha256,
            "episode_index": episode,
            "start_step": start,
            "goal_step": goal,
            "valid_row_rank": study.selection["valid_row_ranks"][index],
            "pair_hash": _pair_hash(episode, start, goal),
        }
        for variant in VARIANT_ORDER:
            for score_mode in SCORE_MODES:
                row[f"success_{variant}__{score_mode}"] = str(
                    study.evaluations[variant][score_mode]["successes"][index]
                ).lower()
        writer.writerow(row)
    return stream.getvalue().encode()


def build_training_curves_csv(study: ValidatedV2Study) -> bytes:
    """Return exactly 60 epoch rows with explicit metric semantics."""

    stream = io.StringIO(newline="")
    fields = [
        "variant",
        "method",
        "display_name",
        "epoch",
        "train_method_objective",
        "train_common_base_hybrid_td",
        "validation_method_objective",
        "validation_common_base_td",
        "train_metric_semantics",
        "validation_metric_semantics",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        for item in training["metrics"]["epochs"]:
            writer.writerow(
                {
                    "variant": variant,
                    "method": training["method"],
                    "display_name": DISPLAY_NAMES[variant],
                    "epoch": item["epoch"],
                    "train_method_objective": f"{item['train_loss']:.12g}",
                    "train_common_base_hybrid_td": (
                        f"{item['train_base_hybrid_td']:.12g}"
                    ),
                    "validation_method_objective": (f"{item['validation_loss']:.12g}"),
                    "validation_common_base_td": (
                        f"{item['validation_base_hybrid_td']:.12g}"
                    ),
                    "train_metric_semantics": "method_specific_coupled_hybrid_objective",
                    "validation_metric_semantics": "common_base_hybrid_td",
                }
            )
    return stream.getvalue().encode()


def _percent(run: Mapping[str, Any]) -> str:
    return (
        f"{100.0 * float(run['success_rate']):.0f}% ({run['success_count']}/{EPISODES})"
    )


def build_markdown_report(study: ValidatedV2Study) -> bytes:
    """Build a concise human-readable report without fabricating missing results."""

    ranking = _ranking(study)
    rank_by_variant = {item["variant"]: item["rank"] for item in ranking}
    lines = [
        "# Results TD — Actor-Free TD-LeWM V2 Cube O50",
        "",
        "本报告只在 6 个训练全部通过验收、18 个正式 O50 单元全部存在且协议一致后生成。",
        "排名预先固定使用 **F+G**，不是在三种评分模式中事后挑最好结果。",
        "",
        "| Rank | Method | F-only | G-only | F+G | Δ F+G − F-only |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANT_ORDER:
        runs = study.evaluations[variant]
        delta = 100.0 * (
            runs["f_plus_g"]["success_rate"] - runs["f_only"]["success_rate"]
        )
        lines.append(
            f"| {rank_by_variant[variant]} | {DISPLAY_NAMES[variant]} | "
            f"{_percent(runs['f_only'])} | {_percent(runs['g_only'])} | "
            f"{_percent(runs['f_plus_g'])} | {delta:+.0f} pp |"
        )
    lines.extend(
        [
            "",
            "## 方法与推理协议",
            "",
            "六种方法都联合微调 online LeWM 与一个 TD-JEPA predictor；25 维原始动作只经过 "
            "`world_model.action_encoder` 这一份共享编码器得到 192 维动作表示。没有 Actor，也没有 reward loss。",
            "",
            "| Method | Training loss | Special mechanism | Inference |",
            "| --- | --- | --- | --- |",
        ]
    )
    for variant in VARIANT_ORDER:
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {METHOD_SPECS[variant]['loss']} | "
            f"{METHOD_SPECS[variant]['special']} | F-only H=5; G-only H=1; F+G H=5 |"
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "",
            f"- 18 个单元共享 selection SHA-256：`{study.selection_sha256}`。",
            "- CEM 固定为 300 candidates、30 iterations、30 elites，episode budget 100；F+G 只在最后一个 action block 使用 G tail。",
            f"- 完整 world-model config canonical SHA-256：`{study.acceptance['common_world_model_config_sha256']}`。",
            "- 每个方法三种评分模式绑定同一 epoch-10 checkpoint；每格成功率由 50 个布尔 outcome 重算。",
            "- 协议写明 0.04 m，但当前 evaluator 调用的 `stable-worldmodel==0.1.1` `World` 公共构造器没有显式 threshold 参数；归档锁版本并披露此限制，没有声称运行时注入了该数值。",
            "- 只有一个 training seed 和一组 planning selection；这是结构消融，不是多随机种子总体结论。",
        ]
    )
    for warning in study.acceptance.get("warnings", []):
        lines.append(f"- Training acceptance warning: {warning}")
    return ("\n".join(lines) + "\n").encode()


def build_archive_readme(study: ValidatedV2Study) -> bytes:
    return f"""# Actor-Free TD-LeWM V2 Cube O50 archive

This directory is generated only from a validated six-training, eighteen-cell
formal bundle. It contains no checkpoints, dataset, video, or console log.

- `summary.json`: protocol, checkpoint, full world-model config, parameter and
  online/target state hashes plus all 18 formal scores.
- `paired_outcomes.csv`: the same 50 locked start-goal pairs and 18 booleans.
- `training_loss_curves.csv`: exactly 6 x 10 epoch rows. Train is each method's
  objective; validation common base TD is explicitly distinguished.
- `checksums.sha256`: byte hashes for every generated artifact and the report.

Locked selection SHA-256: `{study.selection_sha256}`.
F-only/F+G use horizon 5; G-only uses horizon 1. Ranking is F+G only.

Success-threshold disclosure: the protocol records 0.04 m, but the evaluator's
public `World` constructor does not receive that value explicitly. The archive
locks `stable-worldmodel=={STABLE_WORLDMODEL_VERSION}` and does not claim runtime
threshold injection.
""".encode()


def _archive_payloads(
    study: ValidatedV2Study, *, artifact_dir: Path, report_path: Path
) -> tuple[dict[str, bytes], bytes]:
    report = build_markdown_report(study)
    payloads = {
        "README.md": build_archive_readme(study),
        "summary.json": (
            json.dumps(build_summary(study), indent=2, sort_keys=True) + "\n"
        ).encode(),
        "paired_outcomes.csv": build_paired_outcomes_csv(study),
        "training_loss_curves.csv": build_training_curves_csv(study),
    }
    checksum_lines = [
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}"
        for name in sorted(payloads)
    ]
    report_relative = os.path.relpath(report_path, artifact_dir)
    checksum_lines.append(f"{hashlib.sha256(report).hexdigest()}  {report_relative}")
    payloads["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode()
    return payloads, report


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_archive(
    study: ValidatedV2Study,
    *,
    artifact_dir: str | Path,
    report_path: str | Path,
    check: bool = False,
) -> list[Path]:
    """Atomically write, or byte-check, the deterministic V2 result archive."""

    destination = Path(artifact_dir).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    payloads, report_payload = _archive_payloads(
        study, artifact_dir=destination, report_path=report
    )
    expected = {report: report_payload}
    expected.update({destination / name: payload for name, payload in payloads.items()})
    if check:
        for path, payload in expected.items():
            if not path.is_file():
                raise V2ResultValidationError(f"archive check: missing {path}")
            if path.read_bytes() != payload:
                raise V2ResultValidationError(f"archive check: byte drift in {path}")
        return list(expected)
    for path, payload in expected.items():
        _atomic_write_bytes(path, payload)
    return list(expected)


__all__ = [
    "CONFIGURED_PROTOCOL_SHA256",
    "EVALUATION_PROTOCOL_SHA256",
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "G1_NEIGHBOR_MANIFEST_SHA256",
    "LANCE_MANIFEST_SHA256",
    "METHOD_FAMILY",
    "PREDICTOR_PARAMETERS",
    "SCORE_MODES",
    "SELECTION_SHA256",
    "SPLIT_FILE_SHA256",
    "TRAINING_PROTOCOL_SHA256",
    "V2ResultValidationError",
    "ValidatedV2Study",
    "VARIANT_ORDER",
    "WORLD_MODEL_PARAMETERS",
    "audit_training",
    "build_paired_outcomes_csv",
    "build_summary",
    "build_training_curves_csv",
    "canonical_sha256",
    "state_dict_sha256",
    "validate_bundle",
    "write_archive",
    "write_training_acceptance",
]
