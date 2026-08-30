from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_actor_free_td_lewm_v2_parallel.py"
SPEC = importlib.util.spec_from_file_location("v2_parallel_launcher", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)


def _paths(tmp_path: Path) -> object:
    artifact = tmp_path / "artifacts"
    bundle = artifact / "outputs" / "v2_deadbee"
    dataset = artifact / "data" / "cube.lance"
    return LAUNCHER.Paths(
        repository=ROOT,
        artifact_root=artifact,
        dataset=dataset,
        dataset_manifest=Path(f"{dataset}.manifest.json"),
        split_indices=artifact / "shared" / "split.npz",
        neighbor_index=artifact / "shared" / "neighbors",
        v1_root=artifact / "outputs" / "v1",
        bundle_root=bundle,
        smoke_root=bundle / "smoke",
        formal_root=bundle / "formal",
        evaluation_root=bundle / "evaluations",
    )


def _formal_input_audit() -> dict[str, object]:
    return {
        "dataset": {"manifest": {"sha256": LAUNCHER.DATASET_MANIFEST_SHA256}},
        "split_indices": {"sha256": LAUNCHER.SPLIT_SHA256},
        "neighbor_index": {
            "manifest": {"sha256": LAUNCHER.NEIGHBOR_MANIFEST_SHA256}
        },
        "initial_v1_checkpoints": {
            variant: {"sha256": digest}
            for variant, digest in LAUNCHER.V1_SHA256.items()
        },
    }


def _install_formal_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    job: object,
    *,
    revision: str,
) -> tuple[dict[str, dict[str, object]], Path]:
    method = f"actor_free_td_lewm_v2_{job.variant}"
    protocol = {"protocol": "test", "variant": job.variant}
    protocol_sha256 = LAUNCHER.canonical_json_sha256(protocol)
    neighbor_sha256 = (
        LAUNCHER.NEIGHBOR_MANIFEST_SHA256 if job.variant == "g1" else None
    )
    resume_identity = {
        "schema_version": 1,
        "method": method,
        "method_family": "actor_free_td_lewm_v2",
        "variant": job.variant,
        "implementation_version": "v2",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "protocol_sha256": protocol_sha256,
        "source_v1_sha256": LAUNCHER.V1_SHA256[job.variant],
        "v2_start_revision": revision,
        "neighbor_index_manifest_sha256": neighbor_sha256,
    }
    payloads = {
        "deployment": {
            "method": method,
            "variant": job.variant,
            "epoch": LAUNCHER.FORMAL_EPOCH,
            "global_step": LAUNCHER.FORMAL_GLOBAL_STEP,
            "source_v1_provenance": {
                "checkpoint_sha256": LAUNCHER.V1_SHA256[job.variant]
            },
        },
        "last": {"v2_resume_identity": resume_identity},
    }
    deployment_path = Path(job.expected_checkpoint)
    last_path = Path(job.run_dir) / "checkpoints" / "lightning" / "last.ckpt"
    deployment_path.parent.mkdir(parents=True, exist_ok=True)
    last_path.parent.mkdir(parents=True, exist_ok=True)
    deployment_path.write_bytes(b"deployment checkpoint")
    last_path.write_bytes(b"lightning checkpoint")
    manifest = {
        "method": method,
        "method_family": "actor_free_td_lewm_v2",
        "variant": job.variant,
        "implementation_version": "v2",
        "objective_version": 0,
        "deployment_checkpoint_version": 1,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "source_v1": {
            "checkpoint_sha256": LAUNCHER.V1_SHA256[job.variant]
        },
        "runtime": {"tdwm_git_revision": revision},
        "neighbor_index": (
            {"manifest_sha256": neighbor_sha256}
            if neighbor_sha256 is not None
            else None
        ),
    }
    manifest_path = Path(job.run_dir) / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def load_checkpoint(path: Path) -> dict[str, object]:
        if path == deployment_path:
            return payloads["deployment"]
        if path == last_path:
            return payloads["last"]
        raise AssertionError(path)

    monkeypatch.setattr(LAUNCHER, "_torch_load_checkpoint", load_checkpoint)
    return payloads, manifest_path


def test_training_queue_starts_heavy_variants_and_queues_c_last(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(
        stage="formal", paths=paths, python="/venv/bin/python", formal_resume="never"
    )
    assert [job.variant for job in jobs] == ["g1", "g2", "g3", "d", "f", "c"]
    assert len(jobs) == 6
    for job in jobs:
        assert isinstance(job.argv, tuple)
        assert job.argv[-2:] == ("--resume", "never")
        assert "--smoke" not in job.argv
        assert job.expected_epoch == 10
        assert job.expected_global_step == 127_960
    assert "--neighbor-index" in jobs[0].argv
    assert all("--neighbor-index" not in job.argv for job in jobs[1:])


def test_smoke_max_steps_one_still_expects_two_batches_per_epoch(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    smoke1 = LAUNCHER.build_jobs(
        stage="smoke1", paths=paths, python="python", formal_resume="never"
    )
    smoke2 = LAUNCHER.build_jobs(
        stage="smoke2", paths=paths, python="python", formal_resume="never"
    )
    assert [job.run_dir for job in smoke1] == [job.run_dir for job in smoke2]
    for first, second in zip(smoke1, smoke2):
        assert first.argv[-2:] == ("--resume", "never")
        assert second.argv[-2:] == ("--resume", "required")
        assert first.argv[first.argv.index("--max-steps") + 1] == "1"
        assert second.argv[second.argv.index("--max-steps") + 1] == "1"
        assert first.expected_epoch == 1
        assert first.expected_global_step == 2
        assert second.expected_epoch == 2
        assert second.expected_global_step == 4
        for flag in ("--smoke", "--max-steps", "--skip-validation"):
            assert flag in first.argv and flag in second.argv


def test_formal_required_is_explicit_and_auto_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(
        stage="formal", paths=paths, python="python", formal_resume="required"
    )
    assert all(job.argv[-1] == "required" for job in jobs)
    with pytest.raises(ValueError, match="auto is forbidden"):
        LAUNCHER.build_jobs(
            stage="formal", paths=paths, python="python", formal_resume="auto"
        )


def test_resume_requires_last_checkpoint_and_manifest_for_every_job(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(
        stage="smoke2", paths=paths, python="python", formal_resume="never"
    )
    with pytest.raises(RuntimeError, match="Required resume evidence"):
        LAUNCHER.validate_stage_outputs("smoke2", jobs, "never")
    for job in jobs:
        run_dir = Path(job.run_dir)
        checkpoint = run_dir / "checkpoints" / "lightning" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (run_dir / "training_manifest.json").write_text("{}")
    LAUNCHER.validate_stage_outputs("smoke2", jobs, "never")


def test_never_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(
        stage="formal", paths=paths, python="python", formal_resume="never"
    )
    first_run = Path(jobs[0].run_dir)
    first_run.mkdir(parents=True)
    (first_run / "partial.log").write_text("old run")
    with pytest.raises(RuntimeError, match="resume never"):
        LAUNCHER.validate_stage_outputs("formal", jobs, "never")


def test_eval_expands_all_eighteen_argv_arrays_and_output_cells(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    jobs = LAUNCHER.build_jobs(
        stage="eval", paths=paths, python="python", formal_resume="never"
    )
    assert len(jobs) == 18
    assert [job.variant for job in jobs[:6]] == list(LAUNCHER.VARIANTS)
    assert [job.score_mode for job in jobs[:6]] == ["f_plus_g"] * 6
    assert len({job.output_base for job in jobs}) == 18
    assert {(job.variant, job.score_mode) for job in jobs} == {
        (variant, mode)
        for variant in LAUNCHER.VARIANTS
        for mode in LAUNCHER.SCORE_MODES
    }
    for job in jobs:
        assert "--video" not in job.argv
        assert "--smoke" not in job.argv
        assert "--pilot" not in job.argv
        mode_index = job.argv.index("--score-mode") + 1
        assert job.argv[mode_index] == job.score_mode
        assert Path(job.output_base) == paths.evaluation_root / job.variant / str(
            job.score_mode
        )


def test_eval_verification_requires_formal_identity_and_fifty_episodes(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    job = LAUNCHER.build_jobs(
        stage="eval", paths=paths, python="python", formal_resume="never"
    )[0]
    output = Path(job.run_dir)
    output.mkdir(parents=True)
    result = {
        "method": f"actor_free_td_lewm_v2_{job.variant}",
        "method_family": "actor_free_td_lewm_v2",
        "variant": job.variant,
        "implementation_version": "v2",
        "score_mode": job.score_mode,
        "smoke": False,
        "pilot": False,
        "metrics": {"episode_successes": [False] * 50},
    }
    (output / "results.json").write_text(json.dumps(result))
    for name in (
        "protocol_manifest.json",
        "episode_selection.json",
        "action_normalization.json",
    ):
        (output / name).write_text("{}")
    verified = LAUNCHER.verify_job_output(job)
    assert set(verified) == {
        "results",
        "protocol_manifest",
        "episode_selection",
        "action_normalization",
    }
    result["metrics"]["episode_successes"] = [False] * 49
    (output / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="50 formal episodes"):
        LAUNCHER.verify_job_output(job)


def test_canonical_argv_hash_uses_compact_sorted_json() -> None:
    argv = ["python", "train.py", "--seed", "3072"]
    expected = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                argv, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        )
        .hexdigest()
    )
    assert LAUNCHER.canonical_json_sha256(argv) == expected


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    LAUNCHER.atomic_write_json(destination, {"value": 1})
    LAUNCHER.atomic_write_json(destination, {"value": 2, "complete": True})
    assert json.loads(destination.read_text()) == {"value": 2, "complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_protocol_sha_closure_includes_all_inherited_v2_configs() -> None:
    closure = LAUNCHER._protocol_file_closure(
        ROOT
        / "configs"
        / "experiment"
        / "actor_free_td_lewm_v2_g1_cube_checkpoint_o50.yaml"
    )
    assert {path.name for path in closure} == {
        "actor_free_td_lewm_v2_common_cube_train.yaml",
        "actor_free_td_lewm_v2_g1_cube_train.yaml",
        "actor_free_td_lewm_v2_g1_cube_checkpoint_o50.yaml",
    }


def test_child_wrapper_persists_real_pid_and_exit_code(tmp_path: Path) -> None:
    exit_path = tmp_path / "exit.txt"
    pid_path = tmp_path / "pid.txt"
    code = LAUNCHER._child_main(
        [
            str(exit_path),
            str(pid_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(3)",
        ]
    )
    assert code == 3
    assert exit_path.read_text() == "3\n"
    assert int(pid_path.read_text()) > 0


def test_git_audit_rejects_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(["a" * 40, "?? unexpected.txt"])
    monkeypatch.setattr(LAUNCHER, "_run_text", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        LAUNCHER.audited_git_state(ROOT)


def test_python_runtime_requires_pinned_stable_worldmodel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    python.write_text("")
    response = subprocess.CompletedProcess(
        [],
        0,
        json.dumps(
            {
                "executable": str(python),
                "python": "3.12.0",
                "stable_worldmodel": "0.2.0",
                "torch": "2.7.0",
                "lightning": "2.5.0",
            }
        ),
        "",
    )
    monkeypatch.setattr(LAUNCHER.subprocess, "run", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="stable-worldmodel==0.1.1"):
        LAUNCHER.audit_python_runtime(python)


def test_gpu_query_preserves_uuid_memory_and_compute_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter(
        [
            subprocess.CompletedProcess(
                [], 0, "0, GPU-one, NVIDIA Test, 32768, 30000\n", ""
            ),
            subprocess.CompletedProcess([], 0, "GPU-one, 1234\n", ""),
        ]
    )
    monkeypatch.setattr(
        LAUNCHER.subprocess, "run", lambda *args, **kwargs: next(replies)
    )
    assert LAUNCHER.query_gpus() == [
        LAUNCHER.GPUInfo(0, "GPU-one", "NVIDIA Test", 32768, 30000, (1234,))
    ]


def test_dry_run_prints_audited_plan_without_creating_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _paths(tmp_path)
    paths.bundle_root.parent.mkdir(parents=True)
    monkeypatch.setattr(
        LAUNCHER,
        "audited_git_state",
        lambda repository: {
            "revision": "f" * 40,
            "short_revision": "fffffff",
            "clean": True,
        },
    )
    monkeypatch.setattr(LAUNCHER, "_paths", lambda *args, **kwargs: paths)
    monkeypatch.setattr(LAUNCHER, "audit_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        LAUNCHER,
        "audit_python_runtime",
        lambda python: {"stable_worldmodel": "0.1.1"},
    )
    monkeypatch.setattr(LAUNCHER, "validate_stage_outputs", lambda *args: None)
    monkeypatch.setattr(LAUNCHER, "query_gpus", _available_gpus)
    code = LAUNCHER.main(
        [
            "--stage",
            "formal",
            "--artifact-root",
            str(paths.artifact_root),
            "--python",
            sys.executable,
            "--minimum-initial-disk-gib",
            "0",
            "--dry-run",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert code == 0
    assert plan["dry_run"] is True
    assert len(plan["jobs"]) == 6
    assert not paths.bundle_root.exists()


def test_formal_evidence_matches_acceptance_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    job = LAUNCHER.build_jobs(
        stage="formal", paths=paths, python="python", formal_resume="never"
    )[0]
    revision = "b" * 40
    _install_formal_checkpoints(monkeypatch, job, revision=revision)
    log = tmp_path / "job.log"
    log.write_text("completed\n")
    evidence = LAUNCHER._formal_execution_evidence(
        job=job,
        gpu=LAUNCHER.GPUInfo(2, "GPU-uuid", "GPU Name", 32768, 30000),
        pid=321,
        started_at="2026-08-31T00:00:00+00:00",
        ended_at="2026-08-31T01:00:00+00:00",
        return_code=0,
        log_path=log,
        paths=paths,
        git={"revision": revision, "clean": True},
        input_audit=_formal_input_audit(),
        free_before=55 * 1024**3,
        free_after=50 * 1024**3,
    )
    assert evidence["schema_version"] == 1
    assert evidence["source"] == "v2_formal_training_launcher"
    assert evidence["process"]["argv"] == list(job.argv)
    assert evidence["process"]["argv_sha256"] == LAUNCHER.canonical_json_sha256(
        list(job.argv)
    )
    assert evidence["process"]["git_revision"] == "b" * 40
    assert evidence["process"]["git_clean"] is True
    assert evidence["process"]["return_code"] == 0
    assert evidence["inputs"]["neighbor_index"]["manifest_sha256"] == (
        LAUNCHER.NEIGHBOR_MANIFEST_SHA256
    )
    assert evidence["log"]["size_bytes"] == log.stat().st_size
    deployment = evidence["outputs"]["deployment_checkpoint"]
    last = evidence["outputs"]["lightning_last"]
    assert deployment["epoch"] == 10
    assert deployment["global_step"] == 127_960
    assert deployment["resume_identity"] == last["resume_identity"]
    assert deployment["resume_identity_source"].endswith(
        "last.ckpt['v2_resume_identity']"
    )
    assert last["resume_identity"]["v2_start_revision"] == revision
    assert deployment["sha256"] == LAUNCHER.file_sha256(
        Path(job.expected_checkpoint)
    )


def test_non_g1_formal_evidence_uses_null_neighbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    job = next(
        job
        for job in LAUNCHER.build_jobs(
            stage="formal", paths=paths, python="python", formal_resume="never"
        )
        if job.variant == "d"
    )
    revision = "c" * 40
    _install_formal_checkpoints(monkeypatch, job, revision=revision)
    log = tmp_path / "job.log"
    log.write_text("")
    evidence = LAUNCHER._formal_execution_evidence(
        job=job,
        gpu=LAUNCHER.GPUInfo(0, "uuid", "name", 1, 1),
        pid=1,
        started_at="start",
        ended_at="end",
        return_code=0,
        log_path=log,
        paths=paths,
        git={"revision": revision, "clean": True},
        input_audit=_formal_input_audit(),
        free_before=2,
        free_after=1,
    )
    assert evidence["inputs"]["neighbor_index"] is None
    assert (
        evidence["outputs"]["lightning_last"]["resume_identity"]
        ["neighbor_index_manifest_sha256"]
        is None
    )


@pytest.mark.parametrize(
    ("checkpoint", "field", "replacement", "message"),
    (
        ("deployment", "method", "actor_free_td_lewm_v2_d", "method differs"),
        ("deployment", "variant", "d", "variant differs"),
        ("deployment", "epoch", 9, "epoch differs"),
        ("deployment", "global_step", 127_959, "global_step differs"),
        ("last", "method", "actor_free_td_lewm_v2_d", "resume identity differs"),
        ("last", "variant", "d", "resume identity differs"),
        ("last", "protocol_sha256", "1" * 64, "resume identity differs"),
        ("last", "source_v1_sha256", "2" * 64, "resume identity differs"),
        ("last", "v2_start_revision", "3" * 40, "resume identity differs"),
        (
            "last",
            "neighbor_index_manifest_sha256",
            "4" * 64,
            "resume identity differs",
        ),
    ),
)
def test_formal_output_evidence_rejects_tampered_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    paths = _paths(tmp_path)
    job = LAUNCHER.build_jobs(
        stage="formal", paths=paths, python="python", formal_resume="never"
    )[0]
    revision = "b" * 40
    payloads, _ = _install_formal_checkpoints(
        monkeypatch, job, revision=revision
    )
    if checkpoint == "last":
        payloads[checkpoint]["v2_resume_identity"][field] = replacement
    else:
        payloads[checkpoint][field] = replacement

    with pytest.raises(ValueError, match=message):
        LAUNCHER._formal_output_evidence(
            job=job,
            git={"revision": revision, "clean": True},
            input_audit=_formal_input_audit(),
        )


def test_formal_output_evidence_rejects_tampered_protocol_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    job = LAUNCHER.build_jobs(
        stage="formal", paths=paths, python="python", formal_resume="never"
    )[0]
    revision = "b" * 40
    _, manifest_path = _install_formal_checkpoints(
        monkeypatch, job, revision=revision
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"]["protocol"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="training_manifest.protocol_sha256"):
        LAUNCHER._formal_output_evidence(
            job=job,
            git={"revision": revision, "clean": True},
            input_audit=_formal_input_audit(),
        )


class _ImmediateProcess:
    next_pid = 1000
    return_codes: list[int] = []
    launched: list[list[str]] = []

    def __init__(self, command: list[str], **kwargs: object) -> None:
        del kwargs
        self.pid = self.next_pid
        type(self).next_pid += 1
        self.command = command
        type(self).launched.append(command)
        self.returncode = type(self).return_codes.pop(0)
        exit_path = Path(command[3])
        child_pid_path = Path(command[4])
        LAUNCHER.atomic_write_text(exit_path, f"{self.returncode}\n")
        LAUNCHER.atomic_write_text(child_pid_path, f"{self.pid + 10_000}\n")

    def poll(self) -> int:
        return self.returncode


def _available_gpus() -> list[object]:
    return [
        LAUNCHER.GPUInfo(index, f"uuid-{index}", f"gpu-{index}", 32768, 32768)
        for index in range(5)
    ]


def test_scheduler_fills_five_cards_then_dynamically_runs_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.bundle_root.parent.mkdir(parents=True)
    jobs = LAUNCHER.build_jobs(
        stage="smoke1", paths=paths, python="python", formal_resume="never"
    )
    _ImmediateProcess.return_codes = [0] * 6
    _ImmediateProcess.launched = []
    monkeypatch.setattr(LAUNCHER.subprocess, "Popen", _ImmediateProcess)
    monkeypatch.setattr(LAUNCHER, "verify_job_output", lambda job: {"ok": job.job_id})
    monkeypatch.setattr(LAUNCHER.time, "sleep", lambda _: None)
    code = LAUNCHER.run_launcher(
        stage="smoke1",
        paths=paths,
        jobs=jobs,
        git={"revision": "d" * 40, "clean": True},
        input_audit={},
        gpu_indices=[0, 1, 2, 3, 4],
        minimum_gpu_free_mib=1,
        minimum_dispatch_disk_bytes=1,
        poll_seconds=0,
        launcher_dir=tmp_path / "launcher",
        gpu_provider=_available_gpus,
    )
    assert code == 0
    launched_scripts = [
        next(item for item in command if "scripts/train_" in item)
        for command in _ImmediateProcess.launched
    ]
    assert [Path(script).stem.rsplit("_", 1)[-1] for script in launched_scripts] == [
        "g1",
        "g2",
        "g3",
        "d",
        "f",
        "c",
    ]


def test_scheduler_is_fail_closed_and_never_launches_c_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.bundle_root.parent.mkdir(parents=True)
    jobs = LAUNCHER.build_jobs(
        stage="smoke1", paths=paths, python="python", formal_resume="never"
    )
    _ImmediateProcess.return_codes = [7, 0, 0, 0, 0]
    _ImmediateProcess.launched = []
    monkeypatch.setattr(LAUNCHER.subprocess, "Popen", _ImmediateProcess)
    monkeypatch.setattr(LAUNCHER, "verify_job_output", lambda job: {"ok": job.job_id})
    monkeypatch.setattr(LAUNCHER.time, "sleep", lambda _: None)
    code = LAUNCHER.run_launcher(
        stage="smoke1",
        paths=paths,
        jobs=jobs,
        git={"revision": "e" * 40, "clean": True},
        input_audit={},
        gpu_indices=[0, 1, 2, 3, 4],
        minimum_gpu_free_mib=1,
        minimum_dispatch_disk_bytes=1,
        poll_seconds=0,
        launcher_dir=tmp_path / "launcher",
        gpu_provider=_available_gpus,
    )
    assert code == 1
    assert len(_ImmediateProcess.launched) == 5
    state = json.loads((tmp_path / "launcher" / "state.json").read_text())
    assert state["fail_closed"] is True
    assert state["jobs"]["c"]["state"] == "BLOCKED"
