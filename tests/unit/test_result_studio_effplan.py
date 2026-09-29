"""Synthetic paired evaluation fixtures; no network, model, or private data."""

import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from tdwm.result_studio.comparison import compare_to_baseline
from tdwm.result_studio.data import (
    available_experiments,
    load_manifest,
    manifest_report,
)
from tdwm.result_studio.import_effplan import (
    RUNS,
    build_manifest,
    write_reference_assets,
)
from tdwm.result_studio.models import method_catalog


def make_sources(root):
    for offset in (25, 50, 100):
        for run, (template, _, score) in RUNS.items():
            folder = root / run / template.format(offset=offset)
            folder.mkdir(parents=True)
            episodes = [
                {
                    "index": i,
                    "episode": 8000 + i,
                    "start": 0,
                    "goal": offset,
                    "success": i % 2 == 0,
                }
                for i in range(50)
            ]
            if run == "p_cem_risk":
                episodes[0]["success"] = False
                episodes[1]["success"] = True
            paired = {"episodes": 50, "goal_offset": offset, "planning_seed": 42}
            selection = {
                "goal_offset": offset,
                "episodes": 50,
                "pairs": {
                    "episode_indices": [e["episode"] for e in episodes],
                    "start_steps": [0] * 50,
                    "goal_steps": [offset] * 50,
                },
            }
            result = {
                "method": "EffPlan",
                "protocol": f"O{offset}",
                "score_mode": score,
                "episodes": 50,
                "successes": 25,
                "success_rate": 50.0,
                "formal": True,
                "episode_results": episodes,
                "selection_sha256": f"synthetic-{offset}",
                "paired_protocol": paired,
                "metrics": {
                    "success_rate": 50.0,
                    "episode_successes": [e["success"] for e in episodes],
                },
            }
            protocol = {
                "method": "EffPlan",
                "protocol": f"O{offset}",
                "score_mode": score,
                "status": "complete",
                "selection": selection,
                "selection_sha256": f"synthetic-{offset}",
                "paired_protocol": paired,
                "normalization": {"mean": [0] * 5},
                "checkpoints": {"EffPlan": {"sha256": "synthetic-checkpoint"}},
                "dataset": {
                    "path": "synthetic.lance",
                    "source_sha256": "synthetic-dataset",
                },
            }
            for name, data in (
                ("result", result),
                ("protocol_manifest", protocol),
                ("episode_results", {"episodes": episodes}),
            ):
                (folder / f"{name}.json").write_text(json.dumps(data))
    return root / "p_cem", root / "p_cem_risk"


class EffPlanImportTests(unittest.TestCase):
    def test_all_groups_labels_and_no_fabricated_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_manifest(*make_sources(root))
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            for offset in (25, 50, 100):
                trials = load_manifest(str(path), offset)
                self.assertEqual(len(trials), 50)
                self.assertEqual(
                    len({s.training_method for s in method_catalog(trials).values()}), 1
                )
                self.assertEqual(
                    compare_to_baseline(trials[0], "p_cem_risk", "p_cem").label, "Lost"
                )
                self.assertEqual(
                    compare_to_baseline(trials[1], "p_cem_risk", "p_cem").label, "New"
                )
                for trial in trials:
                    for tracks in trial.methods.values():
                        self.assertNotIn("predicted", tracks)
                        self.assertFalse(tracks["executed"].frames)
                        self.assertIsNone(tracks["executed"].actions)
                        self.assertIsNone(tracks["executed"].states)
            report, notes, _ = manifest_report(str(path), "p_cem")
            self.assertEqual(len(report), 6)
            self.assertEqual(report[1]["New"], 1)
            self.assertEqual(report[1]["Lost"], 1)
            self.assertTrue(any("不能用参考轨迹代替" in note for note in notes))

    def test_rejects_inconsistent_sources(self):
        mutations = [
            ("result", lambda d: d.update(success_rate=92)),
            ("result", lambda d: d.update(episodes=49)),
            ("result", lambda d: d.update(formal=False)),
            ("result", lambda d: d["episode_results"][0].update(success="false")),
            (
                "protocol_manifest",
                lambda d: d["selection"]["pairs"]["episode_indices"].__setitem__(
                    0, 9999
                ),
            ),
            (
                "protocol_manifest",
                lambda d: d["checkpoints"]["EffPlan"].update(sha256="other"),
            ),
            ("protocol_manifest", lambda d: d["dataset"].update(source_sha256="other")),
            ("protocol_manifest", lambda d: d["normalization"].update(mean=[1] * 5)),
            (
                "protocol_manifest",
                lambda d: d["paired_protocol"].update(planning_seed=43),
            ),
            ("episode_results", lambda d: d["episodes"][0].update(start=1)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            roots = make_sources(Path(directory))
            for file, mutate in mutations:
                with self.subTest(file=file, mutate=mutate):
                    path = roots[1] / "total_work_n10_O25" / f"{file}.json"
                    original = path.read_text()
                    data = json.loads(original)
                    mutate(data)
                    path.write_text(json.dumps(data))
                    try:
                        with self.assertRaises(ValueError):
                            build_manifest(*roots)
                    finally:
                        path.write_text(original)

    def test_reference_export_deduplicates_checks_identity_and_preserves_jpeg(self):
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), "blue").save(buffer, "JPEG")
        jpeg = buffer.getvalue()

        class Dataset:
            bad = False

            def take(self, indices, columns):
                self.rows = [
                    {
                        "episode_idx": index // 201 + int(self.bad),
                        "step_idx": index % 201,
                        "pixels": jpeg,
                        "observation": [float(index)] * 28,
                        "action": [float(index)] * 5,
                    }
                    for index in indices
                ]
                return self

            def to_pylist(self):
                return self.rows

        records = [
            {"episode": 2, "offset": 25, "start": 2, "goal": 27},
            {"episode": 2, "offset": 50, "start": 1, "goal": 51},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = Dataset()
            write_reference_assets(dataset, records, root)
            self.assertEqual(len(list(root.rglob("*.jpg"))), 51)
            self.assertEqual(
                (root / records[0]["reference"]["frames"][0]).read_bytes(), jpeg
            )
            states = np.load(root / records[0]["reference"]["states"])
            actions = np.load(root / records[0]["reference"]["actions"])
            self.assertEqual(states.shape, (26, 28))
            self.assertEqual(actions.shape, (25, 5))
            self.assertEqual(states[0, 0], 404)
            self.assertEqual(states[-1, 0], 429)
            dataset.bad = True
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                write_reference_assets(dataset, records, root)

    def test_experiment_switch_and_summary_with_missing_method_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_manifest(*make_sources(root))
            second = copy.deepcopy(first)
            for trial in second["trials"]:
                trial["episode"] += 100
            for spec in second["method_catalog"].values():
                spec["training_method"] = "Another training fixture"
            for name, data in (("first", first), ("second", second)):
                (root / f"{name}.json").write_text(json.dumps(data))
            registry = root / "experiments.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiments": [
                            {"label": "Pair fixture", "manifest": "first.json"},
                            {"label": "Other fixture", "manifest": "second.json"},
                        ],
                    }
                )
            )
            with patch.dict(os.environ, {"RESULT_STUDIO_EXPERIMENTS": str(registry)}):
                self.assertEqual(
                    available_experiments()["Pair fixture"],
                    (root / "first.json").resolve(),
                )
                app = AppTest.from_file(
                    str(
                        Path(__file__).resolve().parents[2] / "scripts/result_studio.py"
                    )
                ).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(app.selectbox(key="baseline_method").value, "p_cem")
                self.assertEqual(
                    app.multiselect(key="search_methods").value,
                    ["P+CEM", "P+CEM＋动作扰动风险"],
                )
                self.assertTrue(
                    any("不能播放方法自身的路径" in item.value for item in app.warning)
                )
                self.assertEqual(len(app.dataframe[0].value), 6)
                app.button(key="jump_49").click().run()
                app.selectbox(key="experiment").set_value("Other fixture").run()
                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        "样本 01" in item.value and "8100" in item.value
                        for item in app.markdown
                    )
                )
                self.assertEqual(
                    app.multiselect(key="training_methods").value,
                    ["Another training fixture"],
                )
                app.selectbox(key="experiment").set_value("Pair fixture").run()
                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        "800" in item.value or "8049" in item.value
                        for item in app.markdown
                    )
                )


if __name__ == "__main__":
    unittest.main()
