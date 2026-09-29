"""Adapters isolate file layout from the review and analysis pages."""

import json
import os
from pathlib import Path

import numpy as np

from .models import MethodSpec, Trajectory, Trial

REPO_ROOT = Path(__file__).resolve().parents[3]
METHODS = ["f_only", "f_plus_g"]
REAL_PREVIEW_MANIFEST = Path(
    os.environ.get(
        "RESULT_STUDIO_MANIFEST",
        str(REPO_ROOT / "data/result_studio/real_preview_20260930/manifest.json"),
    )
).expanduser()


def demo_trials(offset: int) -> list[Trial]:
    """Deterministic synthetic coordinates, NOT robot images or measured results."""
    trials = []
    for i in range(50):
        rng = np.random.default_rng(8600 + offset * 100 + i)
        t = np.linspace(0, 1, offset + 1)
        phase = rng.uniform(-1, 1)
        xy = np.column_stack(
            (
                0.12 + 0.73 * t,
                0.26
                + 0.48 * t
                + 0.17 * np.sin(2 * np.pi * t + phase) * np.sin(np.pi * t),
            )
        )
        reference = np.column_stack(
            (
                xy,
                0.3 + 0.18 * np.sin(np.pi * t),
                0.5 + 0.1 * np.cos(2 * np.pi * t),
                0.2 + 0.3 * t,
                0.1 * np.sin(4 * np.pi * t),
            )
        )
        projection = np.random.default_rng(12).normal(size=(6, 16))
        ref = Trajectory(
            "目标参考轨迹",
            "reference",
            states=reference,
            actions=np.diff(reference, axis=0)[:, :5] * 10,
            embeddings=reference @ projection,
        )
        methods = {}
        for j, name in enumerate(["方法 A · 演示", "方法 B · 演示"]):
            success = (i * 7 + offset + j * 3) % 11 < (8 if j else 6)
            states = reference.copy()
            states[:, 0] += (
                (0.04 + 0.02 * j) * np.sin(t * (5 + j) * np.pi) * np.sin(np.pi * t)
            )
            states[:, 1] += (0.06 - 0.02 * j) * np.sin(t * 3 * np.pi)
            if not success:
                states[:, :2] += np.outer(t**2, [0.13, -0.17])
            predicted = states.copy()
            predicted[:, 1] += 0.035 * np.sin(np.pi * t)
            methods[name] = {
                "executed": Trajectory(
                    name,
                    "executed",
                    success,
                    states,
                    np.diff(states, axis=0)[:, :5] * 10,
                    states @ projection,
                ),
                "predicted": Trajectory(
                    name,
                    "predicted",
                    None,
                    predicted,
                    np.diff(predicted, axis=0)[:, :5] * 10,
                    predicted @ projection,
                ),
            }
        trials.append(
            Trial(
                i + 1,
                100 + i * 137,
                offset,
                0,
                offset,
                ref,
                methods,
                "确定性合成数据，仅用于界面与函数演示",
                True,
                {name: MethodSpec(name, "演示搜索") for name in methods},
            )
        )
    return trials


def _read_json(path: Path):
    return json.loads(path.read_text())


def available_experiments() -> dict[str, Path]:
    """A private registry keeps unrelated evaluation selections separate."""
    configured = os.environ.get("RESULT_STUDIO_EXPERIMENTS")
    path = Path(
        configured or REPO_ROOT / "data/result_studio/experiments.json"
    ).expanduser()
    if not configured and not path.is_file():
        return {"当前真实数据预览": REAL_PREVIEW_MANIFEST}
    config = _read_json(path)
    if config.get("schema_version") != 1 or not config.get("experiments"):
        raise ValueError("实验目录必须包含 schema_version: 1 和非空 experiments")
    experiments = {}
    for item in config["experiments"]:
        name = item["label"]
        if not isinstance(name, str) or not name.strip() or name in experiments:
            raise ValueError("实验名称必须非空且不能重复")
        experiments[name] = (path.resolve().parent / item["manifest"]).resolve()
    return experiments


def manifest_report(path: str, baseline: str | None):
    """Summarize saved labels only, without loading images or inferring rollouts."""
    data = _read_json(Path(path))
    rows = []
    for offset in sorted({t["offset"] for t in data["trials"]}):
        trials = [t for t in data["trials"] if t["offset"] == offset]
        for name, spec in data.get("method_catalog", {}).items():
            outcomes, changes = [], {"New": 0, "Lost": 0, "可配对": 0}
            for trial in trials:
                methods = trial["methods"]
                success = methods.get(name, {}).get("executed", {}).get("success")
                base = methods.get(baseline, {}).get("executed", {}).get("success")
                if type(success) is bool:
                    outcomes.append(success)
                if type(success) is bool and type(base) is bool:
                    changes["可配对"] += 1
                    changes["New"] += int(success and not base)
                    changes["Lost"] += int(base and not success)
            rows.append(
                {
                    "测试组": f"O{offset}",
                    "训练方法": spec["training_method"],
                    "搜索方法": spec["search_method"],
                    "成功 / 已标注": f"{sum(outcomes)} / {len(outcomes)}",
                    "成功率": f"{sum(outcomes) / len(outcomes):.0%}"
                    if outcomes
                    else "未标注",
                    "New": changes["New"] if changes["可配对"] else None,
                    "Lost": changes["Lost"] if changes["可配对"] else None,
                    "可配对": changes["可配对"],
                }
            )
    return rows, data.get("import_notes", []), data.get("provenance", {})


def historical_trials(offset: int) -> tuple[list[Trial], list[str]]:
    """Read one known v1-c evaluation per offset; never join by row alone."""
    config_path = os.environ.get("RESULT_STUDIO_HISTORY_CONFIG")
    if not config_path:
        return [], [
            "未配置历史记录目录。请使用真实图像数据清单，或设置 RESULT_STUDIO_HISTORY_CONFIG。"
        ]
    try:
        config = Path(config_path).expanduser().resolve()
        base = (config.parent / _read_json(config)[str(offset)]).resolve()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [], [f"历史记录配置不可用：{exc}"]
    records: dict[tuple[int, int, int], Trial] = {}
    notes = []
    for method in METHODS:
        folder = base / method
        try:
            result = _read_json(folder / "results.json")
            selection = _read_json(folder / "episode_selection.json")
            outcomes = result["metrics"]["episode_successes"]
            ids, starts, goals = [
                selection[k] for k in ("episode_indices", "start_steps", "goal_steps")
            ]
            if not (len(outcomes) == len(ids) == len(starts) == len(goals)):
                raise ValueError("测试编号、起终点和成功标签数量不一致")
            for episode, start, goal, success in zip(ids, starts, goals, outcomes):
                if goal - start != offset or type(success) is not bool:
                    raise ValueError("offset 或成功标签类型不匹配")
                identity = (episode, start, goal)
                if identity not in records:
                    records[identity] = Trial(
                        len(records) + 1,
                        episode,
                        offset,
                        start,
                        goal,
                        Trajectory("目标参考轨迹", "reference"),
                        {},
                        str(base),
                    )
                records[identity].methods[method] = {
                    "executed": Trajectory(
                        method,
                        "executed",
                        success,
                        attributes={"results_path": str(folder / "results.json")},
                    )
                }
                records[identity].method_specs[method] = MethodSpec("v1-c", method)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            notes.append(f"{method}: {exc}")
    return list(records.values()), notes


def resolve_asset(root: Path, value: str) -> Path:
    """Manifests may reference only files inside their dataset directory."""
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"数据路径必须位于数据目录内：{value}")
    if not path.is_file():
        raise ValueError(f"缺少文件：{value}")
    return path


def load_manifest(path: str, offset: int) -> list[Trial]:
    """Portable per-trial manifest with ordered frames, videos and array paths."""
    manifest = Path(path).expanduser().resolve()
    root = manifest.parent
    data = _read_json(manifest)
    if data.get("schema_version") != 1:
        raise ValueError("manifest 的 schema_version 必须为 1")

    def track(record, name, kind):
        success = record.get("success")
        if success is not None and type(success) is not bool:
            raise ValueError("success 必须为 true、false 或 null")
        arrays = {}
        for key in ("states", "actions", "embeddings"):
            if record.get(key):
                value = np.load(
                    resolve_asset(root, record[key]), allow_pickle=False, mmap_mode="r"
                )
                if value.ndim != 2 or value.dtype.kind not in "fiu":
                    raise ValueError(f"{key} 必须是 [时间, 维度] 数值 .npy 数组")
                arrays[key] = value
        frames = [str(resolve_asset(root, item)) for item in record.get("frames", [])]
        video = (
            str(resolve_asset(root, record["video"])) if record.get("video") else None
        )
        lengths = [len(arrays[k]) for k in ("states", "embeddings") if k in arrays]
        if frames:
            lengths.append(len(frames))
        if len(set(lengths)) > 1:
            raise ValueError("同一轨迹的 frames、states、embeddings 必须逐帧对齐")
        if (
            lengths
            and "actions" in arrays
            and len(arrays["actions"]) not in (lengths[0], lengths[0] - 1)
        ):
            raise ValueError("action 数量必须为帧数或帧数减一")
        return Trajectory(
            name,
            kind,
            success,
            frames=frames,
            video=video,
            attributes=record.get("attributes", {}),
            **arrays,
        )

    trials, seen = [], set()
    specs = {}
    for run_id, spec in data.get("method_catalog", {}).items():
        training, search = spec.get("training_method"), spec.get("search_method")
        if not isinstance(training, str) or not training.strip():
            raise ValueError("method_catalog 中 training_method 必须是非空字符串")
        if not isinstance(search, str) or not search.strip():
            raise ValueError("method_catalog 中 search_method 必须是非空字符串")
        specs[run_id] = MethodSpec(training, search)
    # Backward compatibility only for the explicitly identified v1-c preview.
    # Do not infer training methods from arbitrary names like f_only.
    if not specs and data.get("provenance", {}).get("selection") == (
        "v1-c f_only / f_plus_g O25/O50/O100, all 50 trials per offset"
    ):
        specs = {name: MethodSpec("v1-c", name) for name in METHODS}
    for item in data["trials"]:
        if int(item["offset"]) != offset:
            continue
        key = (int(item["episode"]), int(item["start"]), int(item["goal"]))
        if key in seen:
            raise ValueError("同一组存在重复的 episode/start/goal")
        seen.add(key)
        if key[2] - key[1] != offset:
            raise ValueError("目标帧减起始帧必须等于 offset")
        methods = {
            name: {
                kind: track(record, name, kind)
                for kind, record in tracks.items()
                if kind in ("executed", "predicted")
            }
            for name, tracks in item["methods"].items()
        }
        trials.append(
            Trial(
                len(trials) + 1,
                *[key[0], offset, key[1], key[2]],
                track(item.get("reference", {}), "目标参考轨迹", "reference"),
                methods,
                str(manifest),
                method_specs={
                    name: specs.get(name, MethodSpec(name, "未指定"))
                    for name in methods
                },
            )
        )
    return trials
