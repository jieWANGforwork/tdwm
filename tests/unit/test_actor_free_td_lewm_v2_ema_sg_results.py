from __future__ import annotations

import copy
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import yaml

from tdwm.results import actor_free_td_lewm_v2 as old_results
from tdwm.results import actor_free_td_lewm_v2_ema_sg as results

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_protocol(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    parent = value.get("extends")
    if parent is None:
        return value
    return _merge(_load_protocol(path.parent / parent), value)


def test_ema_sg_result_contract_is_isolated_from_old_v2() -> None:
    assert old_results.METHOD_FAMILY == "actor_free_td_lewm_v2"
    assert len(old_results.METRIC_ALIASES) == 4
    assert results.METHOD_FAMILY == "actor_free_td_lewm_v2_ema_sg"
    assert results.IMPLEMENTATION_VERSION == "v2_ema_sg"
    assert results._engine.TRAINING_STAGE == (
        "coupled_hybrid_ema_target_finetuning"
    )
    assert results._engine.TRAINING_INITIALIZATION == (
        "corresponding_v1_deployment_finetune"
    )
    assert results._engine.LOCAL_PREDICTION == "ema_target_lewm_one_step_mse"
    assert results._engine.LOCAL_PREDICTION_TARGET == (
        "ema_world_model_next_latent"
    )
    assert results._engine.LOCAL_PREDICTION_TARGET_GRADIENT == "stop_gradient"
    assert results._engine.STRICT_RESUME_IDENTITY is True
    expected_resume = results._engine._expected_resume_identity(
        variant="c",
        method="actor_free_td_lewm_v2_ema_sg_c",
        training_revision="a" * 40,
    )
    assert expected_resume == {
        "schema_version": 1,
        "method": "actor_free_td_lewm_v2_ema_sg_c",
        "method_family": "actor_free_td_lewm_v2_ema_sg",
        "variant": "c",
        "implementation_version": "v2_ema_sg",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "protocol_sha256": results.TRAINING_PROTOCOL_SHA256["c"],
        "source_v1_sha256": results.SOURCE_V1_SHA256["c"],
        "v2_start_revision": "a" * 40,
        "neighbor_index_manifest_sha256": None,
    }


def test_all_ema_sg_protocol_hashes_are_locked_to_checked_in_configs() -> None:
    config_root = REPOSITORY_ROOT / "configs/experiment"
    for variant in results.VARIANT_ORDER:
        training = _load_protocol(
            config_root
            / f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_train.yaml"
        )
        assert (
            results.canonical_sha256(training)
            == results.TRAINING_PROTOCOL_SHA256[variant]
        )
        formal = _load_protocol(
            config_root
            / f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_checkpoint_o50.yaml"
        )
        assert (
            results.canonical_sha256(formal)
            == results.EVALUATION_PROTOCOL_SHA256[variant]
        )
        for score_mode, horizon in results.FORMAL_HORIZON_BY_SCORE_MODE.items():
            configured = copy.deepcopy(formal)
            configured["inference_objective"]["score_mode"] = score_mode
            configured["planning"]["horizon"] = horizon
            assert (
                results.canonical_sha256(configured)
                == results.CONFIGURED_PROTOCOL_SHA256[variant][score_mode]
            )


def _metric_rows() -> tuple[list[str], list[dict[str, str]]]:
    preferred_columns = [aliases[0] for aliases in results.METRIC_ALIASES.values()]
    fieldnames = ["epoch", "step", *preferred_columns]
    rows = []
    for epoch in range(10):
        row = {"epoch": str(epoch), "step": str((epoch + 1) * 12_796 - 1)}
        row.update(
            {
                column: str(0.1 + epoch + metric_index / 100)
                for metric_index, column in enumerate(preferred_columns)
            }
        )
        rows.append(row)
    return fieldnames, rows


def _write_metrics(path: Path, *, omit: str | None = None) -> None:
    fieldnames, rows = _metric_rows()
    if omit is not None:
        fieldnames.remove(omit)
        for row in rows:
            row.pop(omit)
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_ema_sg_metrics_require_all_twelve_train_validation_series(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics/version_0/metrics.csv"
    _write_metrics(metrics_path)
    audit = results._Audit()
    metrics = results._audit_metrics(tmp_path, audit=audit, variant="c")
    assert audit.errors == []
    assert metrics["final_step"] == 127_959
    assert len(metrics["epochs"]) == 10
    assert set(metrics["epochs"][0]) == {"epoch", *results.METRIC_ALIASES}

    missing_path = tmp_path / "missing/metrics/version_0/metrics.csv"
    missing_column = "validation/online_ema_latent_drift"
    _write_metrics(missing_path, omit=missing_column)
    missing_audit = results._Audit()
    results._audit_metrics(
        tmp_path / "missing", audit=missing_audit, variant="c"
    )
    assert any(
        "validation_online_ema_latent_drift" in error
        for error in missing_audit.errors
    )


def test_ema_sg_training_curve_export_contains_every_audited_metric() -> None:
    epochs = []
    for epoch in range(1, 11):
        item = {"epoch": epoch}
        item.update(
            {
                metric_name: float(epoch + index / 100)
                for index, metric_name in enumerate(results.METRIC_ALIASES)
            }
        )
        epochs.append(item)
    training = {
        variant: {
            "method": f"{results.METHOD_FAMILY}_{variant}",
            "metrics": {"epochs": copy.deepcopy(epochs)},
        }
        for variant in results.VARIANT_ORDER
    }
    study = SimpleNamespace(training=training)
    rows = list(
        csv.DictReader(io.StringIO(results.build_training_curves_csv(study).decode()))
    )
    assert len(rows) == 60
    assert {
        "train_total_loss",
        "train_prediction_loss",
        "train_prediction_online_reference_mse",
        "train_online_ema_latent_drift",
        "train_base_hybrid_td_loss",
        "train_method_hybrid_td_loss",
        "validation_total_loss",
        "validation_prediction_loss",
        "validation_prediction_online_reference_mse",
        "validation_online_ema_latent_drift",
        "validation_base_hybrid_td_loss",
        "validation_method_hybrid_td_loss",
    } <= set(rows[0])


def test_scripts_and_public_exports_are_independent() -> None:
    for name in (
        "accept_actor_free_td_lewm_v2_ema_sg_training.py",
        "archive_actor_free_td_lewm_v2_ema_sg_o50.py",
    ):
        assert (REPOSITORY_ROOT / "scripts" / name).is_file()
    assert results._engine.TRAINING_EVIDENCE_SOURCE == (
        "v2_ema_sg_formal_training_launcher"
    )
    assert results._engine.TRAIN_SCRIPT_TEMPLATE == (
        "train_actor_free_td_lewm_v2_ema_sg_{variant}.py"
    )


def test_locked_hash_values_are_lowercase_sha256() -> None:
    hashes = [
        *results.TRAINING_PROTOCOL_SHA256.values(),
        *results.EVALUATION_PROTOCOL_SHA256.values(),
        *(
            digest
            for modes in results.CONFIGURED_PROTOCOL_SHA256.values()
            for digest in modes.values()
        ),
    ]
    assert all(
        len(value) == 64
        and value == value.lower()
        and hashlib.sha256(bytes.fromhex(value)).hexdigest()
        for value in hashes
    )
