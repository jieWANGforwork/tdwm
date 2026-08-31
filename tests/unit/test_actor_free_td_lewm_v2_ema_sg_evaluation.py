from __future__ import annotations

import hashlib
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_cli import (
    run_actor_free_td_lewm_v2_ema_sg_evaluation,
)
from tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_common import (
    TRAINING_COMMIT,
    VERSION_DISPLAY_NAME,
    VERSION_KEY,
    _record_v2_ema_identity,
    actor_free_td_v2_ema_sg_output_directory_name,
    configure_actor_free_td_v2_ema_sg_evaluation_mode,
    evaluate_actor_free_td_v2_ema_sg,
    validate_actor_free_td_v2_ema_sg_evaluation_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("c", "d", "f", "g1", "g2", "g3")


def _module(variant: str):
    return importlib.import_module(
        f"tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_{variant}"
    )


def _config(variant: str) -> Path:
    return (
        ROOT
        / "configs"
        / "experiment"
        / f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_checkpoint_o50.yaml"
    )


def _mean_config(variant: str) -> Path:
    return (
        ROOT
        / "configs"
        / "experiment"
        / (
            f"actor_free_td_lewm_v2_ema_sg_{variant}_cube_checkpoint_o50_"
            "g_only_f_rollout_mean.yaml"
        )
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_ema_sg_o50_protocol_has_a_separate_strict_identity(
    variant: str,
) -> None:
    module = _module(variant)
    protocol = getattr(
        module,
        f"load_actor_free_td_lewm_v2_ema_sg_{variant}_evaluation_protocol",
    )(_config(variant))

    assert protocol["method"] == f"actor_free_td_lewm_v2_ema_sg_{variant}"
    assert protocol["method_family"] == "actor_free_td_lewm_v2_ema_sg"
    assert protocol["implementation_version"] == "v2_ema_sg"
    assert protocol["stage"] == "planner_evaluation"
    assert protocol["initialization"] == "corresponding_v1_deployment_finetune"
    assert protocol["joint_objective"]["local_prediction"] == (
        "ema_target_lewm_one_step_mse"
    )
    assert protocol["joint_objective"]["local_prediction_target"] == (
        "ema_world_model_next_latent"
    )
    assert protocol["joint_objective"]["local_prediction_target_gradient"] == (
        "stop_gradient"
    )
    assert protocol["inference_objective"]["deployed_world_model"] == (
        "online_v2_ema_sg_world_model"
    )
    assert protocol["inference_objective"]["deployed_predictor"] == (
        "online_v2_ema_sg_predictor"
    )
    assert module.METHOD_SPEC.display_name == (
        f"Actor-Free TD-LeWM V2 EMA {variant.upper()}"
    )
    assert (
        ROOT / "scripts" / f"evaluate_actor_free_td_lewm_v2_ema_sg_{variant}.py"
    ).is_file()

    for score_mode, horizon in (("f_only", 5), ("g_only", 1), ("f_plus_g", 5)):
        configured = configure_actor_free_td_v2_ema_sg_evaluation_mode(
            protocol,
            smoke=False,
            pilot=False,
            score_mode=score_mode,
        )
        assert configured["planning"]["horizon"] == horizon


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_v2_ema_rollout_mean_overlay_is_loadable_and_explicit(
    variant: str,
) -> None:
    module = _module(variant)
    protocol = getattr(
        module,
        f"load_actor_free_td_lewm_v2_ema_sg_{variant}_evaluation_protocol",
    )(_mean_config(variant))
    inference = protocol["inference_objective"]

    assert protocol["method"] == f"actor_free_td_lewm_v2_ema_sg_{variant}"
    assert protocol["planning"]["horizon"] == 5
    assert inference["score_mode"] == "g_only_f_rollout_mean"
    assert inference["f_transition_used"] is True
    assert inference["f_goal_distance_used"] is False
    assert inference["g_aggregation"] == "mean_over_5_blocks"
    assert inference["state_source_for_q1"] == "current_online_encoder_state"
    assert inference["state_source_for_q2_to_q5"] == (
        "online_lewm_rollout_predicted_states"
    )
    assert protocol["provenance"]["version_key"] == VERSION_KEY
    assert protocol["provenance"]["version_display_name"] == VERSION_DISPLAY_NAME
    assert protocol["provenance"]["training_commit"] == TRAINING_COMMIT


def test_mean_overlay_restores_ema_g_identity_when_overridden_to_first() -> None:
    module = _module("c")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        _mean_config("c")
    )

    configured = configure_actor_free_td_v2_ema_sg_evaluation_mode(
        protocol,
        smoke=False,
        pilot=False,
        score_mode="f_plus_g_first",
        g_first_weight=0.5,
    )

    assert configured["inference_objective"]["g_score"] == (
        "negative_goal_projection_of_v2_ema_sg_online_predictor"
    )


def test_ema_sg_protocol_rejects_original_v2_target_contract() -> None:
    module = _module("c")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        _config("c")
    )
    original = deepcopy(protocol)
    original["joint_objective"]["local_prediction"] = "original_lewm_one_step_mse"
    with pytest.raises(ValueError, match="local_prediction"):
        validate_actor_free_td_v2_ema_sg_evaluation_protocol(
            original,
            spec=module.METHOD_SPEC,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("stage", "coupled_hybrid_ema_target_finetuning"),
        ("initialization", "resume_v2"),
        ("initialization_contract", {}),
    ),
)
def test_ema_sg_protocol_rejects_identity_or_initialization_drift(
    field: str,
    replacement,
) -> None:
    module = _module("c")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol(
        _config("c")
    )
    protocol[field] = replacement

    with pytest.raises(ValueError, match=field):
        validate_actor_free_td_v2_ema_sg_evaluation_protocol(
            protocol,
            spec=module.METHOD_SPEC,
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    ("mode_args", "expected"),
    (
        ((), {"smoke": False, "pilot": False, "checkpoint_epoch": None}),
        (("--smoke",), {"smoke": True, "pilot": False, "checkpoint_epoch": None}),
        (("--pilot",), {"smoke": False, "pilot": True, "checkpoint_epoch": None}),
        (
            ("--checkpoint-epoch", "3"),
            {"smoke": False, "pilot": False, "checkpoint_epoch": 3},
        ),
    ),
)
def test_all_ema_sg_cli_variants_forward_each_evaluation_mode(
    monkeypatch,
    variant: str,
    mode_args: tuple[str, ...],
    expected: dict[str, object],
) -> None:
    captured: dict[str, object] = {}

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    fake_module = SimpleNamespace(
        **{
            f"load_actor_free_td_lewm_v2_ema_sg_{variant}_evaluation_protocol": (
                lambda _path: {}
            ),
            f"evaluate_actor_free_td_lewm_v2_ema_sg_{variant}": evaluate,
        }
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            "protocol.yaml",
            "--dataset",
            "cube.h5",
            "--checkpoint-path",
            "epoch.pt",
            "--output-dir",
            "results",
            *mode_args,
        ],
    )

    run_actor_free_td_lewm_v2_ema_sg_evaluation(variant)

    for key, value in expected.items():
        assert captured[key] == value


@pytest.mark.parametrize(
    ("mode_args", "expected_mode", "expected_weight"),
    (
        (
            ("--score-mode", "f_plus_g_first", "--g-first-weight", "0.25"),
            "f_plus_g_first",
            0.25,
        ),
        (
            ("--score-mode", "g_only_f_rollout_mean"),
            "g_only_f_rollout_mean",
            None,
        ),
    ),
)
def test_ema_cli_forwards_new_score_options_and_training_manifest(
    monkeypatch: pytest.MonkeyPatch,
    mode_args: tuple[str, ...],
    expected_mode: str,
    expected_weight: float | None,
) -> None:
    captured: dict[str, object] = {}

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    fake_module = SimpleNamespace(
        load_actor_free_td_lewm_v2_ema_sg_c_evaluation_protocol=lambda _path: {},
        evaluate_actor_free_td_lewm_v2_ema_sg_c=evaluate,
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            "protocol.yaml",
            "--dataset",
            "cube.h5",
            "--checkpoint-path",
            "epoch.pt",
            "--training-manifest",
            "run/training_manifest.json",
            "--output-dir",
            "results",
            *mode_args,
        ],
    )

    run_actor_free_td_lewm_v2_ema_sg_evaluation("c")

    assert captured["score_mode"] == expected_mode
    assert captured["g_first_weight"] == expected_weight
    assert captured["training_manifest_path"] == "run/training_manifest.json"


def test_ema_sg_output_names_do_not_collide_with_v2() -> None:
    module = _module("g3")
    protocol = module.load_actor_free_td_lewm_v2_ema_sg_g3_evaluation_protocol(
        _config("g3")
    )
    assert (
        actor_free_td_v2_ema_sg_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="f_plus_g",
        )
        == "actor_free_td_lewm_v2_ema_sg_g3_cube_o50_f_plus_g_formal"
    )
    assert (
        actor_free_td_v2_ema_sg_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="f_plus_g_first",
            g_first_weight=0.25,
        )
        == "actor_free_td_lewm_v2_ema_sg_g3_cube_o50_"
        "f_plus_g_first_alpha_0p25_formal"
    )
    assert (
        actor_free_td_v2_ema_sg_output_directory_name(
            protocol,
            smoke=False,
            pilot=False,
            score_mode="g_only_f_rollout_mean",
        )
        == "actor_free_td_lewm_v2_ema_sg_g3_cube_o50_"
        "g_only_f_rollout_mean_formal"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_fixture(
    tmp_path: Path,
    *,
    score_mode: str,
    epoch: int,
) -> tuple[dict[str, object], Path, Path, Path]:
    run_dir = tmp_path / "formal" / "c" / "seed_3072"
    checkpoint = (
        run_dir
        / "checkpoints"
        / "actor_free_td_lewm_v2_ema_sg_c"
        / "c"
        / f"epoch_{epoch:02d}.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-epoch-{epoch}".encode())
    shared_protocol: dict[str, object] = {
        "method": "actor_free_td_lewm_v2_ema_sg_c",
        "method_family": "actor_free_td_lewm_v2_ema_sg",
        "variant": "c",
        "implementation_version": "v2_ema_sg",
        "initialization": "corresponding_v1_deployment_finetune",
        "initialization_contract": {
            "required_checkpoint_family": "actor_free_td_lewm_v1",
            "required_checkpoint_epoch": 10,
            "v2_checkpoint_as_initialization": "prohibited",
            "optimizer_state": "fresh",
        },
        "predictor": {"architecture": "td_jepa_forward_map_v1"},
        "task_sampling": {"sampling": "per_transition_bernoulli"},
        "joint_objective": {"local_prediction": "ema_target_lewm_one_step_mse"},
        "source_v1": {"source_epoch": 10},
        "source_artifacts": {"source": "v1"},
    }
    training_protocol = {
        **shared_protocol,
        "stage": "coupled_hybrid_ema_target_finetuning",
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(
            training_protocol,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    training_manifest = run_dir / "training_manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "method": "actor_free_td_lewm_v2_ema_sg_c",
                "method_family": "actor_free_td_lewm_v2_ema_sg",
                "variant": "c",
                "implementation_version": "v2_ema_sg",
                "objective_version": 0,
                "deployment_checkpoint_version": 1,
                "stage": "coupled_hybrid_ema_target_finetuning",
                "initialization": "corresponding_v1_deployment_finetune",
                "seed": 3072,
                "protocol": training_protocol,
                "protocol_sha256": protocol_sha256,
                "runtime": {"tdwm_git_revision": TRAINING_COMMIT},
            }
        )
    )
    output = tmp_path / "evaluation"
    output.mkdir()
    manifest = {
        "score_mode": score_mode,
        "protocol": {
            **shared_protocol,
            "stage": "planner_evaluation",
            "inference_objective": {"score_mode": score_mode},
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "method": "actor_free_td_lewm_v2_ema_sg_c",
            "method_family": "actor_free_td_lewm_v2_ema_sg",
            "variant": "c",
            "implementation_version": "v2_ema_sg",
            "epoch": epoch,
        },
        "runtime": {"tdwm_git_revision": "a" * 40},
    }
    result: dict[str, object] = {
        "method": "actor_free_td_lewm_v2_ema_sg_c",
        "variant": "c",
        "score_mode": score_mode,
    }
    (output / "protocol_manifest.json").write_text(json.dumps(manifest))
    (output / "results.json").write_text(json.dumps(result))
    return result, output, checkpoint, training_manifest


@pytest.mark.parametrize(
    ("score_mode", "epoch"),
    (("f_plus_g_first", 3), ("g_only_f_rollout_mean", 10)),
)
def test_new_modes_record_exact_v2_ema_artifact_identity(
    tmp_path: Path,
    score_mode: str,
    epoch: int,
) -> None:
    result, output, checkpoint, training_manifest = _identity_fixture(
        tmp_path,
        score_mode=score_mode,
        epoch=epoch,
    )

    recorded = _record_v2_ema_identity(
        result,
        output_dir=output,
        training_manifest_path=training_manifest,
    )
    manifest = json.loads((output / "protocol_manifest.json").read_text())
    persisted = json.loads((output / "results.json").read_text())

    for payload in (recorded, manifest, persisted):
        assert payload["version_key"] == "v2_ema"
        assert payload["version_display_name"] == "V2 EMA"
        assert payload["training_commit"] == TRAINING_COMMIT
        assert payload["evaluation_commit"] == "a" * 40
        assert payload["method"] == "actor_free_td_lewm_v2_ema_sg_c"
        assert payload["epoch"] == epoch
        assert payload["checkpoint_epoch"] == epoch
        assert payload["checkpoint_sha256"] == _sha256(checkpoint)
        assert payload["training_manifest_path"] == str(training_manifest.resolve())
        assert payload["training_manifest_sha256"] == _sha256(training_manifest)


def test_new_mode_identity_rejects_checkpoint_file_hash_drift(tmp_path: Path) -> None:
    result, output, checkpoint, training_manifest = _identity_fixture(
        tmp_path,
        score_mode="f_plus_g_first",
        epoch=3,
    )
    checkpoint.write_bytes(b"mutated-after-runtime-manifest")

    with pytest.raises(ValueError, match="checkpoint file SHA-256"):
        _record_v2_ema_identity(
            result,
            output_dir=output,
            training_manifest_path=training_manifest,
        )


def test_new_mode_identity_rejects_wrong_training_or_evaluation_commit(
    tmp_path: Path,
) -> None:
    result, output, _checkpoint, training_manifest = _identity_fixture(
        tmp_path,
        score_mode="g_only_f_rollout_mean",
        epoch=10,
    )
    training = json.loads(training_manifest.read_text())
    training["runtime"]["tdwm_git_revision"] = "b" * 40
    training_manifest.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="wrong training commit"):
        _record_v2_ema_identity(
            result,
            output_dir=output,
            training_manifest_path=training_manifest,
        )

    training["runtime"]["tdwm_git_revision"] = TRAINING_COMMIT
    training_manifest.write_text(json.dumps(training))
    manifest_path = output / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"]["tdwm_git_revision"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluation commit"):
        _record_v2_ema_identity(
            result,
            output_dir=output,
            training_manifest_path=training_manifest,
        )


def test_new_mode_identity_rejects_manifest_from_another_run(tmp_path: Path) -> None:
    result, output, _checkpoint, training_manifest = _identity_fixture(
        tmp_path,
        score_mode="f_plus_g_first",
        epoch=3,
    )
    foreign = tmp_path / "other_run" / "training_manifest.json"
    foreign.parent.mkdir()
    foreign.write_bytes(training_manifest.read_bytes())

    with pytest.raises(ValueError, match="checkpoint run's exact"):
        _record_v2_ema_identity(
            result,
            output_dir=output,
            training_manifest_path=foreign,
        )


def test_new_mode_rejects_wrong_training_commit_before_o50(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, _output, checkpoint, training_manifest = _identity_fixture(
        tmp_path,
        score_mode="f_plus_g_first",
        epoch=3,
    )
    training = json.loads(training_manifest.read_text())
    training["runtime"]["tdwm_git_revision"] = "b" * 40
    training_manifest.write_text(json.dumps(training))
    common = importlib.import_module(
        "tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_common"
    )
    monkeypatch.setattr(
        common,
        "evaluate_actor_free_td_v2",
        lambda **_kwargs: pytest.fail("O50 evaluator ran before manifest preflight"),
    )
    module = _module("c")

    with pytest.raises(ValueError, match="wrong training commit"):
        evaluate_actor_free_td_v2_ema_sg(
            spec=module.METHOD_SPEC,
            checkpoint_loader=object(),
            policy_factory=object(),
            protocol_path=_config("c"),
            checkpoint_path=checkpoint,
            training_manifest_path=training_manifest,
            output_dir=tmp_path / "must_not_be_created",
            score_mode="f_plus_g_first",
            g_first_weight=1.0,
        )

    assert not (tmp_path / "must_not_be_created").exists()


@pytest.mark.parametrize("score_mode", ("f_only", "g_only", "f_plus_g"))
def test_legacy_modes_do_not_run_new_identity_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
    score_mode: str,
) -> None:
    expected = {"score_mode": score_mode, "legacy": True}
    common = importlib.import_module(
        "tdwm.evaluation.actor_free_td_lewm_v2_ema_sg_common"
    )
    monkeypatch.setattr(common, "evaluate_actor_free_td_v2", lambda **_kwargs: expected)
    monkeypatch.setattr(
        common,
        "_record_v2_ema_identity",
        lambda *_args, **_kwargs: pytest.fail("legacy mode was postprocessed"),
    )

    assert (
        evaluate_actor_free_td_v2_ema_sg(
            spec=object(),
            checkpoint_loader=object(),
            policy_factory=object(),
            output_dir="unused",
            score_mode=score_mode,
        )
        is expected
    )
