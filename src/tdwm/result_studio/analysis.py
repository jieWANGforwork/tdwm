"""Pure analysis functions. Register a function once to make it available in UI."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def finite_rows(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("输入必须为非空的 [样本, 维度] 数组")
    return values[np.isfinite(values).all(axis=1)]


def dimension_summary(values):
    values = finite_rows(values)
    if not len(values):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "维度": [f"d{i}" for i in range(values.shape[1])],
            "均值": values.mean(0),
            "标准差": values.std(0),
            "最小值": values.min(0),
            "P05": np.quantile(values, 0.05, axis=0),
            "中位数": np.median(values, axis=0),
            "P95": np.quantile(values, 0.95, axis=0),
            "最大值": values.max(0),
        }
    )


def distribution_histogram(groups, dimension=0, bins=30):
    clean = {name: finite_rows(values) for name, values in groups.items()}
    clean = {name: v for name, v in clean.items() if len(v)}
    if not clean:
        return pd.DataFrame()
    if any(dimension >= v.shape[1] or dimension < 0 for v in clean.values()):
        raise ValueError("选择的维度不存在")
    edges = np.histogram_bin_edges(
        np.concatenate([v[:, dimension] for v in clean.values()]), bins=bins
    )
    rows = []
    for name, values in clean.items():
        counts, _ = np.histogram(values[:, dimension], bins=edges)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "系列": name,
                    "值": (edges[i] + edges[i + 1]) / 2,
                    "比例": count / len(values),
                    "数量": int(count),
                }
            )
    return pd.DataFrame(rows)


def project_states(groups, standardize=True, max_points=10000):
    """One joint PCA basis for comparable vectors. No separate per-group fits."""
    arrays, labels = [], []
    budget = max(2, max_points // max(1, len(groups)))
    for name, values in groups.items():
        values = finite_rows(values)
        if len(values) > budget:
            values = values[np.linspace(0, len(values) - 1, budget, dtype=int)]
        if len(values):
            arrays.append(values)
            labels.extend([name] * len(values))
    if not arrays:
        raise ValueError("没有有限数值可用于投影")
    if len({v.shape[1] for v in arrays}) != 1:
        raise ValueError("各组维度必须相同；不同编码器的表征不能直接拼接比较")
    x = np.concatenate(arrays)
    if len(x) < 2 or x.shape[1] < 2:
        raise ValueError("PCA 至少需要两个样本和两个维度")
    centered = x - x.mean(0)
    scale = x.std(0) if standardize else np.ones(x.shape[1])
    centered /= np.where(scale > 1e-12, scale, 1)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    variance = singular**2
    explained = variance[:2] / variance.sum() if variance.sum() else np.zeros(2)
    frame = pd.DataFrame(
        {"PC1": projected[:, 0], "PC2": projected[:, 1], "系列": labels}
    )
    return frame, explained


def action_summary(values):
    """Raw action summary; makes no undocumented normalization assumptions."""
    return dimension_summary(values)


@dataclass(frozen=True)
class AnalysisSpec:
    title: str
    function: Callable
    input_kind: str
    description: str


REGISTRY = {
    "pca": AnalysisSpec(
        "二维 PCA 投影",
        project_states,
        "groups",
        "共享投影坐标；保留原始高维数据，不把二维距离当作真实性能指标。",
    ),
    "histogram": AnalysisSpec(
        "逐维分布",
        distribution_histogram,
        "groups",
        "所有系列使用相同分箱，每组独立归一化。",
    ),
    "summary": AnalysisSpec(
        "描述统计", dimension_summary, "array", "均值、标准差、分位数和取值范围。"
    ),
}
