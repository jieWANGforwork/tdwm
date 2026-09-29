"""Optional, tiny CPU Cube check using existing data, never a formal score."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import stable_worldmodel as swm
from PIL import Image

from tdwm.evaluation.rollout_recording import RolloutRecorder
from tdwm.training.eff_protocol import baseline_reference, load_eff_protocol


@pytest.mark.integration
def test_existing_cube_data_records_real_post_action_observations(tmp_path):
    dataset_path = os.environ.get("TDWM_RECORDING_CUBE_DATASET")
    selection_path = os.environ.get("TDWM_RECORDING_CUBE_SELECTION")
    if not dataset_path or not selection_path:
        pytest.skip("Set explicit existing Cube dataset/selection paths; no data download.")
    config_path = Path(__file__).resolve().parents[2] / "configs/experiment/effplan_cube_same_episode_v1.yaml"
    reference = baseline_reference(load_eff_protocol(config_path), config_path)
    selected = json.loads(Path(selection_path).read_text())["pairs"]
    pairs = {key: value[:1] for key, value in selected.items()}
    dataset = swm.data.load_dataset(dataset_path, format="lance", keys_to_load=reference["dataset"]["keys_to_load"])
    recorder = RolloutRecorder(tmp_path / "rollouts", pairs=pairs, budget=3,
                              metadata=dict(method="recording_cpu_check", formal=False))
    wc = reference["world"]
    world = swm.World(
        wc["env_name"], num_envs=1, image_shape=(wc["image_size"], wc["image_size"]),
        max_episode_steps=3, env_type=wc["env_type"], ob_type=wc["ob_type"],
        multiview=wc["multiview"], width=wc["image_size"], height=wc["image_size"],
        visualize_info=wc["visualize_info"], terminate_at_goal=wc["terminate_at_goal"],
        extra_wrappers=[recorder.wrap_environment],
    )

    class ZeroCommand:
        def set_env(self, env):
            self.env = env

        def get_action(self, infos):
            return np.zeros(self.env.action_space.shape, dtype=np.float32)

    try:
        world.set_policy(recorder.wrap_policy(ZeroCommand()))
        result = world.evaluate(
            dataset=dataset, episodes_idx=pairs["episode_indices"],
            start_steps=pairs["start_steps"], goal_offset=25, eval_budget=3,
            callables=[
                dict(method="set_state", args={"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}),
                dict(method="set_target_pos", args={
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": "goal_privileged_block_0_pos"},
                    "target_quat": {"value": "goal_privileged_block_0_quat"},
                }),
            ],
        )
        summary = recorder.finish(result["episode_successes"])
    finally:
        world.close()
    assert summary["status"] == "complete"
    for index in range(1):
        folder = recorder.root / f"episode_{index:04d}"
        trajectory = json.loads((folder / "trajectory.json").read_text())
        assert 1 <= len(trajectory["steps"]) <= 3
        assert len(list((folder / "frames").glob("*.png"))) == len(trajectory["steps"]) + 1
        for row in trajectory["steps"]:
            assert row["action"] == [0.0] * 5
            assert "qpos" in row["after_state"] and "qvel" in row["after_state"]
            with Image.open(folder / row["after_frame"]) as image:
                assert image.mode == "RGB" and image.size == (224, 224)
