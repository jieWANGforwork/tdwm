"""Exercise the staged CLI runner on an explicitly synthetic immutable cache."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from tdwm.training import eff_action as runner
from tdwm.training.frozen_latent_store import file_sha256


class _Encoder(nn.Module):
    input_dim, emb_dim = 25, 192

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(25, 192)

    def forward(self, action):
        return self.layer(action)


@pytest.fixture
def run_inputs(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "configs/experiment/eff_action_cube_train.yaml"
    protocol = yaml.safe_load(source.read_text())
    for key in ("gv", "planner"):
        protocol["sampling"][key]["max_goal_chunks"] = 1
    config = protocol["training"]
    config["precision"] = "float32"
    config["g"]["hidden_dim"] = config["v"]["hidden_dim"] = config["planner"]["hidden_dim"] = 8
    config["planner"]["iterations"] = 2
    protocol["logging"].update(checkpoint_every_steps=2, metrics_every_steps=2)
    protocol_file = tmp_path / "protocol.yaml"
    protocol_file.write_text(yaml.safe_dump(protocol))
    normalization = tmp_path / "column_normalization.json"
    normalization.write_text(json.dumps({"action": {"mean": [0.] * 5, "scale": [1.] * 5, "variance": [1.] * 5, "samples": 50000}}))
    ids = np.repeat(np.arange(10000, dtype=np.int64), 6)
    z = np.ones((len(ids), 192), dtype=np.float32)
    z[:, 0] = np.tile(np.arange(6, dtype=np.float32), 10000)
    actions = np.zeros((len(ids), 25), dtype=np.float32)
    actions[5::6] = np.nan
    store = SimpleNamespace(latents=z, actions=actions, episode_ids=ids,
        frame_skip=5, action_block_dim=25, manifest_sha256="1" * 64,
        manifest={"column_normalization_sha256": file_sha256(normalization), "dataset_source_sha256": "2" * 64,
                  "source_metadata": {"column_normalization_path": str(normalization)}},
        _assert_immutable=lambda: None)
    torch.manual_seed(999)
    model = nn.Module()
    model.action_encoder = _Encoder().requires_grad_(False).eval()
    monkeypatch.setattr(runner, "load_eff_action_store", lambda *args, **kwargs: store)
    monkeypatch.setattr(runner, "load_frozen_eff_action_backbone", lambda *args, **kwargs: (model, {"checkpoint_sha256": "3" * 64}))
    return {"config_path": protocol_file, "latent_store": tmp_path / "synthetic-cache", "pretrained": tmp_path / "synthetic.pt",
            "output_dir": tmp_path / "run", "device": "cpu", "smoke": True}


def test_runner_three_stages_saved_validation_and_deployable_heads(run_inputs):
    result = runner.train_eff_action(**run_inputs)
    assert result["counters"] == {"gv": 3, "stage1": 2, "stage2": 2}
    root = run_inputs["output_dir"]
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 7
    assert all(record["window_updates"] == 1 for record in records)
    for stage, steps in (("gv", 3), ("stage1", 2), ("stage2", 2)):
        assert (root / f"{stage}_validation_{steps:06d}.json").exists()
        assert (root / f"{stage}_complete.pt").exists()
        assert json.loads((root / f"{stage}_validation_pairs.json").read_text())["anchor_goal_global_rows"]
    _, g, v, planner, metadata = runner.load_eff_action_deployment(
        root / "stage2_complete.pt", pretrained=run_inputs["pretrained"], method="EffActionPlan", device="cpu")
    assert planner is not None and metadata["stage_budget_complete"]
    assert all(not parameter.requires_grad for model in (g, v, planner) for parameter in model.parameters())
    with pytest.raises(ValueError, match="trained planner"):
        runner.load_eff_action_deployment(root / "gv_complete.pt", pretrained=run_inputs["pretrained"], method="EffActionPlan", device="cpu")


def test_runner_resumes_gv_to_planner_and_repairs_missing_validation(run_inputs):
    runner.train_eff_action(**run_inputs, stop_after="gv")
    root = run_inputs["output_dir"]
    missing = root / "gv_validation_000003.json"
    missing.unlink()
    with (root / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"stage": "stage1", "step": 1, "metrics": {"loss": 999}}) + "\n")
    result = runner.train_eff_action(**run_inputs, resume=root / "latest.pt")
    assert result["stage_completed"] == "stage2" and missing.exists()
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 7
    assert not any(row["metrics"]["loss"] == 999 for row in records)
    assert len(list(root.glob("metrics_before_resume_*.jsonl"))) == 1


def test_provisional_protocol_cannot_enter_formal_training(run_inputs):
    with pytest.raises(ValueError, match="provisional"):
        runner.train_eff_action(**{**run_inputs, "smoke": False})
    assert not run_inputs["output_dir"].exists()


def test_metrics_recovery_preserves_original_and_drops_partial_tail(tmp_path):
    log = tmp_path / "metrics.jsonl"
    text = ''.join(json.dumps({"stage": "gv", "step": step}) + '\n' for step in (50, 100, 150)) + '{"partial'
    log.write_text(text)
    backup = runner.restore_eff_action_metrics(log, {"gv": 100, "stage1": 0, "stage2": 0})
    assert backup.read_text() == text
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [50, 100]
