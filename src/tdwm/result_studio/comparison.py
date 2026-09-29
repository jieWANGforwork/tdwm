"""Paired New/Lost labels from the same trial's recorded evaluation outcomes."""

import html
from dataclasses import dataclass

from .models import Trial


@dataclass(frozen=True)
class BaselineComparison:
    label: str
    state: str


def compare_to_baseline(
    trial: Trial, method: str, baseline: str | None
) -> BaselineComparison:
    """Compare actual evaluation labels, never pixels or predicted trajectories.

    Both records come from one Trial, already aligned by episode/start/goal/offset.
    A missing method, baseline or boolean outcome is not interpreted as failure.
    """
    if baseline is not None and method == baseline and method in trial.methods:
        return BaselineComparison("Baseline", "baseline")
    candidate = trial.methods.get(method, {}).get("executed")
    reference = trial.methods.get(baseline, {}).get("executed")
    success = candidate.success if candidate is not None else None
    base_success = reference.success if reference is not None else None
    if type(success) is not bool or type(base_success) is not bool:
        return BaselineComparison("无法比较", "unavailable")
    if success and not base_success:
        return BaselineComparison("New", "new")
    if base_success and not success:
        return BaselineComparison("Lost", "lost")
    if success:
        return BaselineComparison("保持成功", "same-success")
    return BaselineComparison("保持失败", "same-failure")


COMPARISON_CSS = """
.baseline-comparison{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:0 16px 13px;font-size:12px;color:#75839a}
.baseline-comparison .comparison-tag{display:inline-block;border-radius:5px;padding:4px 9px;font-size:13px;background:#edf1f7;color:#52627a}
.baseline-comparison .comparison-new{background:#e4f4ec;color:#157450}
.baseline-comparison .comparison-lost{background:#ffeaed;color:#b33e4c}
.baseline-comparison .comparison-baseline{background:#e8efff;color:#305bb5}
.baseline-comparison .comparison-context{overflow-wrap:anywhere}
"""


def comparison_html(comparison: BaselineComparison, baseline_label: str) -> str:
    context = "当前对照方法" if comparison.state == "baseline" else "相对 Baseline"
    title = f"{baseline_label}；按同一样本的实际执行评测标签比较，不代表预测准确率"
    return (
        f'<div class="baseline-comparison" title="{html.escape(title, quote=True)}">'
        f'<span class="comparison-tag comparison-{html.escape(comparison.state, quote=True)}">'
        f"{html.escape(comparison.label)}</span>"
        f'<span class="comparison-context">{context}</span></div>'
    )
