from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdwm.training.eff_data import EffEpisodeReplay
from tdwm.training.eff_run import (
    EffRunSettings,
    run_eff_training,
    scheduled_learning_rate,
)


def settings():
    return EffRunSettings(
        seed=3072,
        epochs=2,
        updates_per_epoch=3,
        batch_size=4,
        g_hidden_dim=8,
        v_hidden_dim=8,
        learning_rate=1e-3,
        weight_decay=0.001,
        gamma_g=0.95,
        beta=5,
        critic_coefficient=1,
        ema_rate=0.005,
        gradient_clip=1,
        include_goal_boundary=True,
        backup_primitive_steps=10,
        cross_episode_probability=0.3,
        epsilon=1e-6,
        warmup_fraction=0.1,
        validation_batches=2,
        validation_seed=50,
        checkpoint_every_updates=2,
    )


def replays():
    store = SimpleNamespace(
        latents=np.random.default_rng(7).standard_normal((84, 192)).astype(np.float32),
        episode_ids=np.repeat(np.arange(4), 21),
    )
    return tuple(
        EffEpisodeReplay(
            store, episodes=episodes, stride=5, terminal_at_state=np.zeros(84, bool)
        )
        for episodes in ((0, 1), (2, 3))
    )


def run(output, **kwargs):
    train, validation = replays()
    return run_eff_training(
        replay=train,
        validation_replay=validation,
        settings=settings(),
        source_identity={"cache": "fixture", "terminal_mapping": "verified-fixture"},
        output_dir=output,
        device="cpu",
        **kwargs,
    )


def test_full_loop_runs_all_epochs_and_persists_both_validation_losses(tmp_path):
    result = run(tmp_path / "complete")
    assert result["status"] == "complete" and result["completed_updates"] == 6
    assert set(result["checkpoints"]) == {"1", "2"}
    with open(result["metrics_path"]) as stream:
        records = [json.loads(line) for line in stream]
    assert sum(x["event"] == "training" for x in records) == 6
    values = [x for x in records if x["event"] == "validation"]
    assert len(values) == 2 and "unweighted_total_loss" in values[-1]
    assert result["identity"]["training_episodes"] == [0, 1]


def test_graceful_stop_and_full_resume_match_uninterrupted_parameters(tmp_path):
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 2

    interrupted = run(tmp_path / "resumed", should_stop=stop)
    assert interrupted["status"] == "stopped" and interrupted["completed_updates"] == 2
    resumed = run(
        tmp_path / "resumed", resume=interrupted["last_recoverable_checkpoint"]
    )
    uninterrupted = run(tmp_path / "direct")
    a = torch.load(resumed["last_recoverable_checkpoint"], weights_only=False)
    b = torch.load(uninterrupted["last_recoverable_checkpoint"], weights_only=False)
    assert a["global_step"] == b["global_step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)


def test_existing_results_are_not_overwritten_or_rewound(tmp_path):
    output = tmp_path / "run"
    result = run(output)
    before = (output / "last.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run(output)
    with pytest.raises(ValueError, match="last checkpoint"):
        run(output, resume=result["checkpoints"]["1"]["path"])
    assert (output / "last.pt").read_bytes() == before


def test_resume_finished_run_preserves_checkpoint_inventory(tmp_path):
    result = run(tmp_path / "run")
    restored = run(tmp_path / "run", resume=result["last_recoverable_checkpoint"])
    assert restored["status"] == "complete"
    assert restored["checkpoints"] == result["checkpoints"]


def test_episode_split_leakage_rejected_before_training(tmp_path):
    train, _ = replays()
    with pytest.raises(ValueError, match="leakage"):
        run_eff_training(
            replay=train,
            validation_replay=train,
            settings=settings(),
            source_identity={"cache": "fixture"},
            output_dir=tmp_path,
            device="cpu",
        )


def test_lr_schedule_is_update_indexed_and_ends_at_zero():
    cfg = replace(settings(), epochs=10, updates_per_epoch=10)
    lr = [scheduled_learning_rate(cfg, i) for i in range(cfg.total_updates)]
    assert lr[0] == pytest.approx(cfg.learning_rate / 10)
    assert lr[9] == cfg.learning_rate and lr[-1] == 0
    assert lr[10] >= lr[11] >= lr[12]
