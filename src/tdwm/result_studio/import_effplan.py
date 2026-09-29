"""Import an audited EffPlan/risk pair, without running or reconstructing rollouts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = {
    "p_cem": ("EffPlan_O{offset}", "P+CEM", "state_path_tracking"),
    "p_cem_risk": (
        "total_work_n10_O{offset}",
        "P+CEM＋动作扰动风险",
        "state_path_tracking_action_robustness_v1",
    ),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_run(folder, offset, score_mode):
    """Fail closed on mismatched labels, selections or reported percentages."""
    result = read_json(folder / "result.json")
    protocol = read_json(folder / "protocol_manifest.json")
    episodes = read_json(folder / "episode_results.json")["episodes"]
    if (
        protocol["status"] != "complete"
        or result.get("formal") is not True
        or result["protocol"] != f"O{offset}"
        or protocol["protocol"] != f"O{offset}"
        or result["method"] != "EffPlan"
        or protocol["method"] != "EffPlan"
        or result["score_mode"] != score_mode
        or protocol["score_mode"] != score_mode
    ):
        raise ValueError(f"未完成或不匹配的 EffPlan 评测：{folder}")
    if len(episodes) != 50 or result["episodes"] != 50:
        raise ValueError(f"要求每组完整的 50 条结果：{folder}")
    if episodes != result["episode_results"]:
        raise ValueError(f"逐条结果文件与 result.json 不一致：{folder}")
    identities = []
    for i, item in enumerate(episodes):
        if (
            item["index"] != i
            or any(type(item[k]) is not int for k in ("episode", "start", "goal"))
            or item["start"] < 0
            or item["goal"] > 200
            or item["goal"] - item["start"] != offset
            or type(item["success"]) is not bool
        ):
            raise ValueError(f"测试编号、帧范围或成功标签无效：{folder}, {i}")
        identities.append((item["episode"], item["start"], item["goal"]))
    if len(set(identities)) != len(identities):
        raise ValueError(f"重复的测试 identity：{folder}")
    selection = protocol["selection"]
    pairs = selection["pairs"]
    selected = list(
        zip(
            pairs["episode_indices"],
            pairs["start_steps"],
            pairs["goal_steps"],
            strict=True,
        )
    )
    outcomes = [item["success"] for item in episodes]
    metrics = result["metrics"]
    if (
        identities != selected
        or selection["goal_offset"] != offset
        or selection["episodes"] != 50
        or not protocol["selection_sha256"]
        or result["selection_sha256"] != protocol["selection_sha256"]
        or result["paired_protocol"] != protocol["paired_protocol"]
        or protocol["paired_protocol"]["goal_offset"] != offset
        or protocol["paired_protocol"]["episodes"] != 50
        or result["successes"] != sum(outcomes)
        or not np.isclose(result["success_rate"], sum(outcomes) * 2)
        or not np.isclose(metrics["success_rate"], sum(outcomes) * 2)
        or metrics["episode_successes"] != outcomes
        or any(type(v) is not bool for v in metrics["episode_successes"])
    ):
        raise ValueError(f"结果、测试选择、协议或成功率互相矛盾：{folder}")
    return result, protocol, episodes


def build_manifest(baseline_root: Path, risk_root: Path):
    """Keep the shared training checkpoint separate from the two search rules."""
    manifest = {
        "schema_version": 1,
        "method_catalog": {
            key: {
                "training_method": "EffPlan · total_work n10",
                "search_method": spec[1],
            }
            for key, spec in RUNS.items()
        },
        "provenance": {
            "selection": "EffPlan / action robustness paired evaluation, 50 per offset",
            "image_format": "dataset JPEG payloads, byte-preserving; reference only",
            "runs": {},
        },
        "import_notes": [
            "两组使用相同训练权重，差别在搜索/评分方式；每个 O 组的 50 条测试、起终点和环境预算已配对核验。",
            "只有参考轨迹具有原数据集图片、28 维状态和 5 维动作。两种方法均未保存执行/预测图片、逐步状态、动作或表征；不能用参考轨迹代替。",
            "风险版本额外调用模型，模型计算预算并不相同。风险评分不是成功概率。",
            "风险诊断记录只有每次评分的汇总值，没有 episode/step 标识，不能可靠分配到某一条测试。",
        ],
        "trials": [],
    }
    dataset = None
    checkpoint_hashes = None
    for offset in (25, 50, 100):
        runs = {}
        for (key, (template, _, score)), root in zip(
            RUNS.items(), (baseline_root, risk_root)
        ):
            folder = root / template.format(offset=offset)
            result, protocol, episodes = read_run(folder, offset, score)
            runs[key] = (result, protocol, episodes)
            diagnostics = folder / "action_robustness_diagnostics.json"
            diag = read_json(diagnostics) if diagnostics.is_file() else {}
            manifest["provenance"]["runs"][f"{key}_O{offset}"] = {
                "directory": str(folder),
                "selection_sha256": protocol["selection_sha256"],
                "paired_protocol": protocol["paired_protocol"],
                "checkpoints": protocol["checkpoints"],
                "compute": protocol.get("compute", {}),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "risk_settings": protocol.get("protocol_overrides", {}).get(
                    "action_robustness"
                ),
                "diagnostics": {k: v for k, v in diag.items() if k != "records"},
                "diagnostic_records": len(diag.get("records", [])),
            }
            current_dataset = protocol["dataset"]
            if dataset is not None and current_dataset != dataset:
                raise ValueError("数据集来源不一致，拒绝将不同实验合并")
            dataset = current_dataset
        _, base_protocol, base_rows = runs["p_cem"]
        _, risk_protocol, risk_rows = runs["p_cem_risk"]
        for key in (
            "selection",
            "selection_sha256",
            "paired_protocol",
            "normalization",
        ):
            if base_protocol[key] != risk_protocol[key]:
                raise ValueError(f"O{offset} 配对不一致：{key}")
        hashes = [
            {k: v["sha256"] for k, v in p["checkpoints"].items()}
            for p in (base_protocol, risk_protocol)
        ]
        if hashes[0] != hashes[1] or not all(hashes[0].values()):
            raise ValueError(f"O{offset} 训练权重不一致")
        if checkpoint_hashes is not None and checkpoint_hashes != hashes[0]:
            raise ValueError("不同 O 组使用了不同训练权重，不能合并为同一模型")
        checkpoint_hashes = hashes[0]
        for base, risk in zip(base_rows, risk_rows, strict=True):
            record = {k: base[k] for k in ("episode", "start", "goal")}
            record.update(offset=offset, reference={}, methods={})
            for key, row in (("p_cem", base), ("p_cem_risk", risk)):
                record["methods"][key] = {
                    "executed": {
                        "success": row["success"],
                        "attributes": {
                            "results_path": manifest["provenance"]["runs"][
                                f"{key}_O{offset}"
                            ]["directory"]
                            + "/result.json",
                            "source_index_zero_based": row["index"],
                            "rollout_saved": False,
                        },
                    }
                }
            manifest["trials"].append(record)
    manifest["provenance"]["dataset"] = dataset
    return manifest


def write_reference_assets(dataset, records, root):
    """Read only requested Cube rows; validate identity before exporting each JPEG."""
    by_episode = defaultdict(list)
    for record in records:
        by_episode[record["episode"]].append(record)
    for i, (episode, trials) in enumerate(sorted(by_episode.items())):
        steps = sorted({s for t in trials for s in range(t["start"], t["goal"] + 1)})
        rows = dataset.take(
            [episode * 201 + step for step in steps],
            columns=["episode_idx", "step_idx", "pixels", "observation", "action"],
        ).to_pylist()
        indexed = {}
        for step, row in zip(steps, rows, strict=True):
            if row["episode_idx"] != episode or row["step_idx"] != step:
                raise ValueError(
                    "Dataset row identity mismatch; no inferred row ordering"
                )
            if not row["pixels"].startswith(b"\xff\xd8"):
                raise ValueError("Expected original JPEG pixels")
            frame = root / f"frames/ep{episode:06d}/step{step:03d}.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(row["pixels"])
            indexed[step] = row
        for trial in trials:
            prefix = f"o{trial['offset']}/ep{episode:06d}_s{trial['start']:03d}_g{trial['goal']:03d}"
            (root / prefix).mkdir(parents=True, exist_ok=True)
            selected = [indexed[s] for s in range(trial["start"], trial["goal"] + 1)]
            for name, values in (
                ("states", [r["observation"] for r in selected]),
                ("actions", [r["action"] for r in selected[:-1]]),
            ):
                np.save(
                    root / prefix / f"{name}.npy",
                    np.asarray(values, dtype=np.float32),
                    allow_pickle=False,
                )
            trial["reference"] = {
                "frames": [
                    f"frames/ep{episode:06d}/step{s:03d}.jpg"
                    for s in range(trial["start"], trial["goal"] + 1)
                ],
                "states": prefix + "/states.npy",
                "actions": prefix + "/actions.npy",
            }
        print(f"Reference images: {i + 1}/{len(by_episode)} episodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--risk-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("输出目录已存在，拒绝覆盖")
    manifest = build_manifest(args.baseline_root.resolve(), args.risk_root.resolve())
    # Use the dataset recorded in the audited protocol, not another dataset by index.
    import lance

    dataset = lance.dataset(manifest["provenance"]["dataset"]["path"])
    args.output.mkdir(parents=True)
    write_reference_assets(dataset, manifest["trials"], args.output)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f"Imported 150 paired trials: {args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
