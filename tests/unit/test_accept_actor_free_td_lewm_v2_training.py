from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tdwm.results import actor_free_td_lewm_v2 as results

TEST_REVISION = "a" * 40


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _evidence_fixture(
    tmp_path: Path, monkeypatch
) -> tuple[Path, dict, dict, Path, Path, Path, dict]:
    variant = "c"
    method = f"{results.METHOD_FAMILY}_{variant}"
    checkpoint = tmp_path / "source.pt"
    checkpoint_sha = _write(checkpoint, b"source-v1")
    split = tmp_path / "split_indices.npz"
    split_sha = _write(split, b"split")
    dataset = tmp_path / "cube.lance"
    dataset.mkdir()
    dataset_manifest = tmp_path / "cube.lance.manifest.json"
    lance_sha = _write(dataset_manifest, b"lance-manifest")
    log = tmp_path / "c.log"
    log_sha = _write(log, b"complete\n")
    deployment = tmp_path / "epoch_10.pt"
    deployment_sha = _write(deployment, b"deployment")
    lightning_last = tmp_path / "last.ckpt"
    lightning_sha = _write(lightning_last, b"lightning")
    last_payload = {"v2_resume_identity": {"v2_start_revision": TEST_REVISION}}
    monkeypatch.setitem(results.SOURCE_V1_SHA256, variant, checkpoint_sha)
    monkeypatch.setattr(results, "SPLIT_FILE_SHA256", split_sha)
    monkeypatch.setattr(results, "LANCE_MANIFEST_SHA256", lance_sha)
    argv = [
        "/root/autodl-tmp/envs/tdwm/bin/python",
        "scripts/train_actor_free_td_lewm_v2_c.py",
        "--output-dir",
        "/formal/c",
    ]
    evidence = {
        "schema_version": 1,
        "source": "v2_formal_training_launcher",
        "method": method,
        "variant": variant,
        "hostname": "trainer-1",
        "gpu": {"index": 0, "uuid": "GPU-test", "name": "NVIDIA-test"},
        "process": {
            "pid": 1234,
            "argv": argv,
            "argv_sha256": results.canonical_sha256(argv),
            "cwd": "/root/autodl-tmp/tdwm-v2-formal",
            "git_revision": TEST_REVISION,
            "git_clean": True,
            "started_at_utc": "2026-08-31T00:00:00Z",
            "ended_at_utc": "2026-08-31T01:00:00Z",
            "return_code": 0,
        },
        "log": {
            "path": str(log),
            "size_bytes": log.stat().st_size,
            "sha256": log_sha,
        },
        "inputs": {
            "dataset": {
                "path": str(dataset),
                "manifest_path": str(dataset_manifest),
                "manifest_sha256": lance_sha,
            },
            "initial_v1_checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
            },
            "split_indices": {"path": str(split), "sha256": split_sha},
            "neighbor_index": None,
        },
        "outputs": {
            "deployment_checkpoint": {
                "path": str(deployment),
                "size_bytes": deployment.stat().st_size,
                "sha256": deployment_sha,
                "epoch": 10,
                "global_step": 127_960,
            },
            "lightning_last": {
                "path": str(lightning_last),
                "size_bytes": lightning_last.stat().st_size,
                "sha256": lightning_sha,
                "resume_identity": last_payload["v2_resume_identity"],
            },
        },
        "disk": {"free_bytes_before": 10_000, "free_bytes_after": 9_000},
    }
    evidence_path = tmp_path / "execution_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    dataset_record = {
        "path": str(dataset),
        "conversion_manifest_path": str(dataset_manifest),
    }
    return (
        evidence_path,
        evidence,
        dataset_record,
        checkpoint,
        deployment,
        lightning_last,
        last_payload,
    )


def test_launcher_evidence_schema_is_accepted(tmp_path: Path, monkeypatch) -> None:
    (
        evidence_path,
        evidence,
        dataset,
        checkpoint,
        deployment,
        lightning_last,
        last_payload,
    ) = _evidence_fixture(tmp_path, monkeypatch)
    audit = results._Audit()
    loaded = results._audit_execution_evidence(
        evidence_path,
        audit=audit,
        variant="c",
        method=f"{results.METHOD_FAMILY}_c",
        training_revision=TEST_REVISION,
        source_checkpoint_path=checkpoint,
        deployment_checkpoint_path=deployment,
        lightning_last_path=lightning_last,
        lightning_last_payload=last_payload,
        dataset=dataset,
        neighbor=None,
    )
    assert loaded == evidence
    assert audit.errors == []
    assert audit.warnings == []


def test_launcher_evidence_rejects_dirty_or_nonzero_run(
    tmp_path: Path, monkeypatch
) -> None:
    (
        evidence_path,
        evidence,
        dataset,
        checkpoint,
        deployment,
        lightning_last,
        last_payload,
    ) = _evidence_fixture(tmp_path, monkeypatch)
    evidence["process"]["git_clean"] = False
    evidence["process"]["return_code"] = 7
    evidence_path.write_text(json.dumps(evidence))
    audit = results._Audit()
    results._audit_execution_evidence(
        evidence_path,
        audit=audit,
        variant="c",
        method=f"{results.METHOD_FAMILY}_c",
        training_revision=TEST_REVISION,
        source_checkpoint_path=checkpoint,
        deployment_checkpoint_path=deployment,
        lightning_last_path=lightning_last,
        lightning_last_payload=last_payload,
        dataset=dataset,
        neighbor=None,
    )
    assert any("git_clean" in error for error in audit.errors)
    assert any("return_code" in error for error in audit.errors)


def test_launcher_evidence_rejects_argv_hash_drift(tmp_path: Path, monkeypatch) -> None:
    (
        evidence_path,
        evidence,
        dataset,
        checkpoint,
        deployment,
        lightning_last,
        last_payload,
    ) = _evidence_fixture(tmp_path, monkeypatch)
    evidence["process"]["argv"].append("--changed")
    evidence_path.write_text(json.dumps(evidence))
    audit = results._Audit()
    results._audit_execution_evidence(
        evidence_path,
        audit=audit,
        variant="c",
        method=f"{results.METHOD_FAMILY}_c",
        training_revision=TEST_REVISION,
        source_checkpoint_path=checkpoint,
        deployment_checkpoint_path=deployment,
        lightning_last_path=lightning_last,
        lightning_last_payload=last_payload,
        dataset=dataset,
        neighbor=None,
    )
    assert any("argv_sha256" in error for error in audit.errors)


def test_launcher_evidence_does_not_trust_copied_resume_identity(
    tmp_path: Path, monkeypatch
) -> None:
    (
        evidence_path,
        _,
        dataset,
        checkpoint,
        deployment,
        lightning_last,
        last_payload,
    ) = _evidence_fixture(tmp_path, monkeypatch)
    last_payload["v2_resume_identity"]["v2_start_revision"] = "b" * 40
    audit = results._Audit()
    results._audit_execution_evidence(
        evidence_path,
        audit=audit,
        variant="c",
        method=f"{results.METHOD_FAMILY}_c",
        training_revision=TEST_REVISION,
        source_checkpoint_path=checkpoint,
        deployment_checkpoint_path=deployment,
        lightning_last_path=lightning_last,
        lightning_last_payload=last_payload,
        dataset=dataset,
        neighbor=None,
    )
    assert any("last.v2_resume_identity" in error for error in audit.errors)
    assert any("differs from last.ckpt" in error for error in audit.errors)
