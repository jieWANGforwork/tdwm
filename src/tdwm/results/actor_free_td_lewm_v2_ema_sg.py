"""Fail-closed acceptance and archiving for Actor-Free TD-LeWM V2-EMA-SG.

This family intentionally reuses the server-tested V2 result engine while
loading it in an isolated module namespace.  The isolated namespace has its
own immutable identity, protocol hashes, evidence contract and required metric
set, so accepting an old V2 run as V2-EMA-SG (or the reverse) fails closed.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


def _load_isolated_v2_engine() -> ModuleType:
    """Load the shared validator implementation without mutating old V2."""

    engine_name = f"{__name__}._isolated_v2_engine"
    source = Path(__file__).with_name("actor_free_td_lewm_v2.py")
    spec = importlib.util.spec_from_file_location(engine_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the V2 result engine from {source}")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[engine_name] = engine
    spec.loader.exec_module(engine)
    return engine


_engine = _load_isolated_v2_engine()

SCHEMA_VERSION = 1
METHOD_FAMILY = "actor_free_td_lewm_v2_ema_sg"
IMPLEMENTATION_VERSION = "v2_ema_sg"
OBJECTIVE_VERSION = 0
DEPLOYMENT_CHECKPOINT_VERSION = 1
TRAINING_STAGE = "coupled_hybrid_ema_target_finetuning"
TRAINING_INITIALIZATION = "corresponding_v1_deployment_finetune"
INITIALIZATION_CONTRACT = {
    "required_checkpoint_family": "actor_free_td_lewm_v1",
    "required_checkpoint_epoch": 10,
    "v2_checkpoint_as_initialization": "prohibited",
    "optimizer_state": "fresh",
}
TRAINING_EVIDENCE_SOURCE = "v2_ema_sg_formal_training_launcher"
TRAIN_SCRIPT_TEMPLATE = "train_actor_free_td_lewm_v2_ema_sg_{variant}.py"
LOCAL_PREDICTION = "ema_target_lewm_one_step_mse"
LOCAL_PREDICTION_TARGET = "ema_world_model_next_latent"
LOCAL_PREDICTION_TARGET_GRADIENT = "stop_gradient"
EVALUATION_G_SCORE = "negative_goal_projection_of_v2_ema_sg_online_predictor"
DEPLOYED_WORLD_MODEL = "online_v2_ema_sg_world_model"
DEPLOYED_PREDICTOR = "online_v2_ema_sg_predictor"
STUDY_ID = "actor_free_td_lewm_v2_ema_sg_cube_seed3072_o50_6x3"

VARIANT_ORDER = _engine.VARIANT_ORDER
SCORE_MODES = _engine.SCORE_MODES
FORMAL_HORIZON_BY_SCORE_MODE = _engine.FORMAL_HORIZON_BY_SCORE_MODE
WORLD_MODEL_PARAMETERS = _engine.WORLD_MODEL_PARAMETERS
PREDICTOR_PARAMETERS = _engine.PREDICTOR_PARAMETERS
SELECTION_SHA256 = _engine.SELECTION_SHA256
LANCE_MANIFEST_SHA256 = _engine.LANCE_MANIFEST_SHA256
SPLIT_FILE_SHA256 = _engine.SPLIT_FILE_SHA256
G1_NEIGHBOR_MANIFEST_SHA256 = _engine.G1_NEIGHBOR_MANIFEST_SHA256
SOURCE_V1_SHA256 = _engine.SOURCE_V1_SHA256

TRAINING_PROTOCOL_SHA256 = {
    "c": "daf416b4368c31bd9cce5edcceb32e6ff69d5818ab0abc6a14301c4b4bca313e",
    "d": "47bb83f7e0de3c67977cf4fa25559c3f75dae5b242788460532e038147b8aa48",
    "f": "b4de699a41884fa5fefda12c092d2e69be462a5eca91810759ee416711e41216",
    "g1": "aa19bda3e8d67c4a049df07dd1254c821a822d79a5a8f0c93a7c4e2d721cffd0",
    "g2": "5fd2948f397f6b25f2d3593a782d28a7cc95ea443eb33f39f9b7b26087faab99",
    "g3": "12e3da5cf5bfd3162abf48324977364d25ea9693c64cb4fbb258b8fd552032bf",
}
EVALUATION_PROTOCOL_SHA256 = {
    "c": "76714cf5014850e11cffb2523e2367950e819741a3d4094d6e05047590275f75",
    "d": "9a248d1176f1db2e53c95d689da6d63745dc4549a7e3e7d3e670a82449f5fbec",
    "f": "78ff50033b08151cd1d9fe3c16d45036af6861866c1cbd6be768613633beec4f",
    "g1": "7295c4e2bc0f3f3dcde86d8a4102785201950e181e465675dd7af7a7f2b649ba",
    "g2": "81499050555430ca1e6f8bf2305ef481fb0d9944d906eff345182ab8b99999bd",
    "g3": "857abd43516f20681d2fa14e6c026966dff03481a4dcdab0dd1c72a32ab5f772",
}
CONFIGURED_PROTOCOL_SHA256 = {
    "c": {
        "f_only": "139b5a16ae8aadb8a86c23f3c7c6a8a7a689dad22c63d34b7cfd7be405fb7068",
        "g_only": "f8ec1e3b2d510b86c0db9d06169631a977d48c5a3fd8ed8bfdeb6803fd3fe1c0",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["c"],
    },
    "d": {
        "f_only": "227a5d137a404df2ede7ffdfc44639b7993f9e071042ba60520e5ea99bfcb634",
        "g_only": "e2e73c02283a174e6e0e72fa707bb72ecd7397c4e188eb3df9fea69d8a76713b",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["d"],
    },
    "f": {
        "f_only": "3ab6850d924bed6dd8a3a1ef67ee11228f2127276ae4da1749f2cde7c63f7fc2",
        "g_only": "e93d8f44077e6f6804432f91d1ae600c62dfb034ee565c60fa6b0272b482f808",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["f"],
    },
    "g1": {
        "f_only": "6c162f445e337bb098e4bbe18cbdb140c545acde6533da0d235808adedfa09dc",
        "g_only": "84243ea152f8c0dc54219d363950806f68e76ae4c945e2ac81cdd041c23aa905",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["g1"],
    },
    "g2": {
        "f_only": "8c40fe371706fe6407c526f332ce89c063016a8b7596adc9ba34b58cd9955569",
        "g_only": "69d52049f25e997bf3bdcbc7eb9e409e557edc90804533805f6ee513a4ed83b9",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["g2"],
    },
    "g3": {
        "f_only": "6d91d636407cb0cf99d0c3c53e2c07ba03420d65f543922bbc77c2cb2590965a",
        "g_only": "1612858748068da77a2408b4344fd94967b6690816017d7d8fffa362f386c63f",
        "f_plus_g": EVALUATION_PROTOCOL_SHA256["g3"],
    },
}

METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "train_loss": ("train/loss_epoch", "train/loss"),
    "train_prediction_loss": (
        "train/prediction_loss_epoch",
        "train/prediction_loss",
    ),
    "train_prediction_online_reference_mse": (
        "train/prediction_online_reference_mse_epoch",
        "train/prediction_online_reference_mse",
    ),
    "train_online_ema_latent_drift": (
        "train/online_ema_latent_drift_epoch",
        "train/online_ema_latent_drift",
    ),
    "train_base_hybrid_td": (
        "train/base_hybrid_td_loss_epoch",
        "train/base_hybrid_td_loss",
    ),
    "train_method_hybrid_td": (
        "train/method_hybrid_td_loss_epoch",
        "train/method_hybrid_td_loss",
    ),
    "validation_loss": ("validation/loss", "validation/loss_epoch"),
    "validation_prediction_loss": (
        "validation/prediction_loss",
        "validation/prediction_loss_epoch",
    ),
    "validation_prediction_online_reference_mse": (
        "validation/prediction_online_reference_mse",
        "validation/prediction_online_reference_mse_epoch",
    ),
    "validation_online_ema_latent_drift": (
        "validation/online_ema_latent_drift",
        "validation/online_ema_latent_drift_epoch",
    ),
    "validation_base_hybrid_td": (
        "validation/base_hybrid_td_loss",
        "validation/base_hybrid_td_loss_epoch",
    ),
    "validation_method_hybrid_td": (
        "validation/method_hybrid_td_loss",
        "validation/method_hybrid_td_loss_epoch",
    ),
}

DISPLAY_NAMES = {
    variant: name.replace("V2-", "V2-EMA-SG-")
    for variant, name in _engine.DISPLAY_NAMES.items()
}
METHOD_SPECS = deepcopy(_engine.METHOD_SPECS)

# Every function in the isolated engine resolves globals from this private
# namespace.  These assignments specialize it without touching old V2.
for _name, _value in {
    "METHOD_FAMILY": METHOD_FAMILY,
    "IMPLEMENTATION_VERSION": IMPLEMENTATION_VERSION,
    "OBJECTIVE_VERSION": OBJECTIVE_VERSION,
    "DEPLOYMENT_CHECKPOINT_VERSION": DEPLOYMENT_CHECKPOINT_VERSION,
    "TRAINING_STAGE": TRAINING_STAGE,
    "TRAINING_INITIALIZATION": TRAINING_INITIALIZATION,
    "INITIALIZATION_CONTRACT": INITIALIZATION_CONTRACT,
    "TRAINING_EVIDENCE_SOURCE": TRAINING_EVIDENCE_SOURCE,
    "TRAIN_SCRIPT_TEMPLATE": TRAIN_SCRIPT_TEMPLATE,
    "LOCAL_PREDICTION": LOCAL_PREDICTION,
    "LOCAL_PREDICTION_TARGET": LOCAL_PREDICTION_TARGET,
    "LOCAL_PREDICTION_TARGET_GRADIENT": LOCAL_PREDICTION_TARGET_GRADIENT,
    "EVALUATION_G_SCORE": EVALUATION_G_SCORE,
    "DEPLOYED_WORLD_MODEL": DEPLOYED_WORLD_MODEL,
    "DEPLOYED_PREDICTOR": DEPLOYED_PREDICTOR,
    "TRAINING_PROTOCOL_SHA256": TRAINING_PROTOCOL_SHA256,
    "EVALUATION_PROTOCOL_SHA256": EVALUATION_PROTOCOL_SHA256,
    "CONFIGURED_PROTOCOL_SHA256": CONFIGURED_PROTOCOL_SHA256,
    "METRIC_ALIASES": METRIC_ALIASES,
    "DISPLAY_NAMES": DISPLAY_NAMES,
    "METHOD_SPECS": METHOD_SPECS,
    "STRICT_RESUME_IDENTITY": True,
    "EXTENDED_IDENTITY_FIELDS": True,
    "STRICT_METRIC_ACCEPTANCE": True,
}.items():
    setattr(_engine, _name, _value)


def build_training_curves_csv(study: Any) -> bytes:
    """Export every required EMA-target diagnostic for all 6 x 10 epochs."""

    metric_columns = {
        "train_total_loss": "train_loss",
        "train_prediction_loss": "train_prediction_loss",
        "train_prediction_online_reference_mse": (
            "train_prediction_online_reference_mse"
        ),
        "train_online_ema_latent_drift": "train_online_ema_latent_drift",
        "train_base_hybrid_td_loss": "train_base_hybrid_td",
        "train_method_hybrid_td_loss": "train_method_hybrid_td",
        "validation_total_loss": "validation_loss",
        "validation_prediction_loss": "validation_prediction_loss",
        "validation_prediction_online_reference_mse": (
            "validation_prediction_online_reference_mse"
        ),
        "validation_online_ema_latent_drift": (
            "validation_online_ema_latent_drift"
        ),
        "validation_base_hybrid_td_loss": "validation_base_hybrid_td",
        "validation_method_hybrid_td_loss": "validation_method_hybrid_td",
    }
    fields = ["variant", "method", "display_name", "epoch", *metric_columns]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for variant in VARIANT_ORDER:
        training = study.training[variant]
        for item in training["metrics"]["epochs"]:
            row: dict[str, Any] = {
                "variant": variant,
                "method": training["method"],
                "display_name": DISPLAY_NAMES[variant],
                "epoch": item["epoch"],
            }
            row.update(
                {
                    output_name: f"{float(item[source_name]):.12g}"
                    for output_name, source_name in metric_columns.items()
                }
            )
            writer.writerow(row)
    return stream.getvalue().encode()


_base_build_summary = _engine.build_summary
_base_build_markdown_report = _engine.build_markdown_report


def build_summary(study: Any) -> dict[str, Any]:
    """Add the single changed objective to the standard provenance summary."""

    summary = _base_build_summary(study)
    summary["study"]["id"] = STUDY_ID
    summary["study"]["implementation_version"] = IMPLEMENTATION_VERSION
    summary["study"]["local_prediction_contract"] = {
        "loss": LOCAL_PREDICTION,
        "target": LOCAL_PREDICTION_TARGET,
        "target_gradient": LOCAL_PREDICTION_TARGET_GRADIENT,
    }
    for method in summary["methods"].values():
        method["network"] = (
            "Jointly fine-tuned online LeWM and TD-JEPA predictor; the one-step "
            "LeWM MSE target is the stop-gradient EMA next latent"
        )
        method["training"]["local_prediction_contract"] = deepcopy(
            summary["study"]["local_prediction_contract"]
        )
    return summary


def build_markdown_report(study: Any) -> bytes:
    """Render the standard report with the EMA-target distinction explicit."""

    report = _base_build_markdown_report(study).decode()
    report = report.replace(
        "# Results TD — Actor-Free TD-LeWM V2 Cube O50",
        "# Results TD — Actor-Free TD-LeWM V2-EMA-SG Cube O50",
        1,
    )
    marker = "本报告只在 6 个训练全部通过验收"
    sentence = (
        "本实验唯一新增的局部预测变量是：LeWM MSE 使用 stop-gradient EMA "
        "next latent 作为 target；online history、action encoder 和 prediction "
        "保持可训练。\n\n"
    )
    report = report.replace(marker, sentence + marker, 1)
    return report.encode()


def build_archive_readme(study: Any) -> bytes:
    """Describe the complete independent EMA-SG archive."""

    return f"""# Actor-Free TD-LeWM V2-EMA-SG Cube O50 archive

This directory is generated only from a validated six-training, eighteen-cell
formal EMA-SG bundle. It contains no checkpoints, dataset, video, or console
log. Every accepted epoch contains train and validation total loss, prediction
loss, online-reference MSE, online/EMA latent drift, base Hybrid TD and
method-specific Hybrid TD.

The LeWM one-step target is
`stop_gradient(EMA_world_model_next_latent)`; online histories, action
embeddings and predictions remain on the trainable online path.

Locked selection SHA-256: `{study.selection_sha256}`.
F-only/F+G use horizon 5; G-only uses horizon 1. Ranking is F+G only.
""".encode()


# Archive builders in the private engine resolve these names at call time.
_engine.build_training_curves_csv = build_training_curves_csv
_engine.build_summary = build_summary
_engine.build_markdown_report = build_markdown_report
_engine.build_archive_readme = build_archive_readme

V2EMASGResultValidationError = _engine.V2ResultValidationError
ValidatedV2EMASGStudy = _engine.ValidatedV2Study
V2ResultValidationError = V2EMASGResultValidationError
ValidatedV2Study = ValidatedV2EMASGStudy
canonical_sha256 = _engine.canonical_sha256
state_dict_sha256 = _engine.state_dict_sha256
audit_training = _engine.audit_training
validate_bundle = _engine.validate_bundle
write_archive = _engine.write_archive
write_training_acceptance = _engine.write_training_acceptance

# Focused tests intentionally exercise the strict private audit primitives.
_Audit = _engine._Audit
_audit_execution_evidence = _engine._audit_execution_evidence
_audit_metrics = _engine._audit_metrics

__all__ = [
    "CONFIGURED_PROTOCOL_SHA256",
    "DEPLOYMENT_CHECKPOINT_VERSION",
    "EVALUATION_PROTOCOL_SHA256",
    "FORMAL_HORIZON_BY_SCORE_MODE",
    "G1_NEIGHBOR_MANIFEST_SHA256",
    "IMPLEMENTATION_VERSION",
    "LANCE_MANIFEST_SHA256",
    "LOCAL_PREDICTION",
    "LOCAL_PREDICTION_TARGET",
    "LOCAL_PREDICTION_TARGET_GRADIENT",
    "METHOD_FAMILY",
    "OBJECTIVE_VERSION",
    "PREDICTOR_PARAMETERS",
    "SCORE_MODES",
    "SELECTION_SHA256",
    "SPLIT_FILE_SHA256",
    "TRAINING_PROTOCOL_SHA256",
    "TRAINING_INITIALIZATION",
    "TRAINING_STAGE",
    "V2EMASGResultValidationError",
    "V2ResultValidationError",
    "ValidatedV2EMASGStudy",
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

build_paired_outcomes_csv = _engine.build_paired_outcomes_csv
