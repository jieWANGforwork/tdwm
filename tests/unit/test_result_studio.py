import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from tdwm.result_studio.analysis import (
    distribution_histogram,
    project_states,
)
from tdwm.result_studio.data import (
    REAL_PREVIEW_MANIFEST,
    demo_trials,
    historical_trials,
    load_manifest,
    resolve_asset,
)
from tdwm.result_studio.endpoints import endpoint_images_html, endpoint_paths
from tdwm.result_studio.models import (
    MethodSpec,
    Trajectory,
    method_catalog,
    select_runs,
)
from tdwm.result_studio.player import CARD_WIDTH, PREVIEW_SIZE, player_html


class AnalysisTests(unittest.TestCase):
    def test_histogram_normalization_and_nan(self):
        result = distribution_histogram(
            {"a": np.array([[1, 2], [3, 4], [np.nan, 2]]), "b": np.array([[5, 6]])},
            0,
            4,
        )
        for name in ["a", "b"]:
            self.assertAlmostEqual(result[result["系列"] == name]["比例"].sum(), 1)

    def test_joint_pca_and_constant(self):
        a = np.arange(30).reshape(10, 3)
        result, variance = project_states({"a": a, "b": a.copy()})
        np.testing.assert_allclose(result.iloc[:10, :2], result.iloc[10:, :2])
        self.assertAlmostEqual(variance.sum(), 1)
        result, variance = project_states({"a": np.ones((3, 2))})
        self.assertTrue(np.isfinite(result[["PC1", "PC2"]]).all().all())
        self.assertEqual(variance.sum(), 0)

    def test_all_demo_groups(self):
        for offset in [25, 50, 100]:
            trials = demo_trials(offset)
            self.assertEqual(len(trials), 50)
            self.assertEqual(trials[0].reference.length, offset + 1)
            self.assertTrue(all(t.demo for t in trials))

    def test_real_groups_are_aligned(self):
        # Synthetic, clearly identified file fixtures: no network or private mirror.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "history.json"
            config.write_text(json.dumps({str(o): f"o{o}" for o in (25, 50, 100)}))
            for offset in (25, 50, 100):
                for method in ("f_only", "f_plus_g"):
                    folder = root / f"o{offset}" / method
                    folder.mkdir(parents=True)
                    (folder / "results.json").write_text(
                        json.dumps(
                            {
                                "metrics": {
                                    "episode_successes": [i % 2 == 0 for i in range(50)]
                                }
                            }
                        )
                    )
                    (folder / "episode_selection.json").write_text(
                        json.dumps(
                            {
                                "episode_indices": list(range(50)),
                                "start_steps": [0] * 50,
                                "goal_steps": [offset] * 50,
                            }
                        )
                    )
            with patch.dict(os.environ, {"RESULT_STUDIO_HISTORY_CONFIG": str(config)}):
                self._assert_history_groups()

    def _assert_history_groups(self):
        for offset in [25, 50, 100]:
            trials, errors = historical_trials(offset)
            self.assertEqual(errors, [])
            self.assertEqual(len(trials), 50)
            self.assertTrue(all(t.goal - t.start == offset for t in trials))
            self.assertTrue(all(len(t.methods) == 2 for t in trials))
            self.assertTrue(all(t.reference.states is None for t in trials))

    def test_asset_escape_rejected(self):
        with self.assertRaises(ValueError):
            resolve_asset(Path.cwd(), "../../outside.npy")

    def test_endpoint_images_use_reference_not_method(self):
        trial = demo_trials(25)[0]
        trial.reference.frames = ["start.jpg", "middle.jpg", "target.jpg"]
        for tracks in trial.methods.values():
            tracks["executed"].frames = ["start.jpg", "wrong-final.jpg"]
        self.assertEqual(endpoint_paths(trial), ("start.jpg", "target.jpg"))
        trial.reference.frames = []
        self.assertEqual(endpoint_paths(trial), (None, None))
        trial.reference.frames = ["start.jpg"]
        self.assertEqual(endpoint_paths(trial), ("start.jpg", None))

    def test_endpoint_row_has_two_static_cards(self):
        result = endpoint_images_html(demo_trials(25)[0])
        self.assertEqual(result.count("<figure "), 2)
        self.assertIn("初始状态", result)
        self.assertIn("目标状态", result)
        self.assertIn("原始帧 25", result)
        trial = demo_trials(25)[0]
        trial.demo = False
        self.assertEqual(endpoint_images_html(trial).count("原始图片待接入"), 2)

    def test_player_fixed_size_with_many_methods(self):
        for count in [1, 2, 3, 12]:
            tracks = [Trajectory(f"method {i}", "executed") for i in range(count)]
            result = player_html(tracks)
            self.assertIn(f"flex:0 0 {CARD_WIDTH}px", result)
            self.assertIn(f"height:{PREVIEW_SIZE}px", result)
            self.assertIn("flex-wrap:wrap", result)
            self.assertNotIn("gridTemplateColumns", result)
            self.assertIn(f"method {count - 1}", result)

    @unittest.skipUnless(
        REAL_PREVIEW_MANIFEST.is_file(), "Real preview assets are local-only"
    )
    def test_real_preview_has_all_selected_reference_frames(self):
        from PIL import Image

        for offset in [25, 50, 100]:
            real = load_manifest(str(REAL_PREVIEW_MANIFEST), offset)
            self.assertEqual(len(real), 50)
            for trial in real:
                self.assertFalse(trial.demo)
                self.assertEqual(len(trial.reference.frames), offset + 1)
                self.assertEqual(trial.reference.states.shape, (offset + 1, 28))
                self.assertEqual(trial.reference.actions.shape, (offset, 5))
                for frame in endpoint_paths(trial):
                    with Image.open(frame) as image:
                        self.assertEqual(image.size, (224, 224))
                        self.assertEqual(image.format, "JPEG")
                self.assertTrue(
                    all(
                        type(v["executed"].success) is bool
                        for v in trial.methods.values()
                    )
                )

    def test_selection_accepts_more_than_three_methods(self):
        trial = demo_trials(25)[0]
        trial.methods = {
            f"method_{i}": {"executed": Trajectory(f"method_{i}", "executed")}
            for i in range(12)
        }
        with patch("tdwm.result_studio.data.demo_trials", return_value=[trial]):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[2] / "scripts/result_studio.py")
            ).run(timeout=30)
            app.sidebar.selectbox[0].set_value("交互演示 · 合成数据").run()
            app.multiselect[0].set_value(list(trial.methods)).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.multiselect[0].value), 12)

    def test_training_and_search_are_independent_filters(self):
        trial = demo_trials(25)[0]
        trial.methods = {
            key: {"executed": Trajectory(key, "executed", success)}
            for key, success in [
                ("a_f", True),
                ("a_fg", False),
                ("b_f", False),
                ("b_fg", True),
            ]
        }
        trial.method_specs = {
            "a_f": MethodSpec("Train A", "f_only"),
            "a_fg": MethodSpec("Train A", "f_plus_g"),
            "b_f": MethodSpec("Train B", "f_only"),
            "b_fg": MethodSpec("Train B", "f_plus_g"),
        }
        catalog = method_catalog([trial])
        self.assertEqual(select_runs(catalog, ["Train A"], ["f_plus_g"]), ["a_fg"])
        self.assertEqual(
            select_runs(catalog, ["Train A", "Train B"], ["f_only"]), ["a_f", "b_f"]
        )
        with patch("tdwm.result_studio.data.demo_trials", return_value=[trial]):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[2] / "scripts/result_studio.py")
            ).run(timeout=30)
            app.sidebar.selectbox[0].set_value("交互演示 · 合成数据").run()
            self.assertEqual(
                app.multiselect(key="training_methods").options, ["Train A", "Train B"]
            )
            self.assertEqual(
                app.multiselect(key="search_methods").options, ["f_only", "f_plus_g"]
            )
            app.multiselect(key="training_methods").set_value(["Train B"]).run()
            app.multiselect(key="search_methods").set_value(["f_plus_g"]).run()
            self.assertFalse(app.exception)
            cards = [item.proto.srcdoc for item in app.get("iframe")]
            self.assertIn("Train B", str(cards))
            self.assertNotIn("Train A", str(cards))
            self.assertIn("f_plus_g", str(cards))
            self.assertNotIn("f_only", str(cards))
            app.toggle[0].set_value(True).run()
            self.assertFalse(app.exception)

    def test_manifest_explicit_method_metadata_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            record = {
                "schema_version": 1,
                "method_catalog": {
                    "run": {"training_method": "Training X", "search_method": "f_only"}
                },
                "trials": [
                    {
                        "episode": 1,
                        "offset": 25,
                        "start": 0,
                        "goal": 25,
                        "methods": {"run": {"executed": {"success": True}}},
                    }
                ],
            }
            path.write_text(json.dumps(record))
            self.assertEqual(
                method_catalog(load_manifest(str(path), 25))["run"],
                MethodSpec("Training X", "f_only"),
            )
            del record["method_catalog"]
            path.write_text(json.dumps(record))
            self.assertEqual(
                method_catalog(load_manifest(str(path), 25))["run"],
                MethodSpec("run", "未指定"),
            )

    def test_pages_and_sample_navigation(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[2] / "scripts/result_studio.py")
        ).run(timeout=30)
        self.assertFalse(app.exception)
        app.sidebar.selectbox[0].set_value("交互演示 · 合成数据").run()
        app.button(key="jump_49").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("样本 50" in item.value for item in app.markdown))
        for page in ["状态与表征", "Action 分析", "数据与扩展"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, page)
        app.sidebar.radio[0].set_value("轨迹对比").run()
        app.sidebar.selectbox[0].set_value("历史评测记录").run()
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
