"""The fixed start/goal image pair belonging to a trial, not a method rollout."""

import html

import numpy as np

from .models import Trial
from .player import image_uri


def endpoint_paths(trial: Trial) -> tuple[str | None, str | None]:
    # Never substitute the last frame reached by a method for the target image.
    frames = trial.reference.frames
    if not frames:
        return None, None
    return frames[0], frames[-1] if len(frames) > 1 else None


def _demo_state_image(trial: Trial, index: int, title: str) -> str:
    """A static coordinate diagram; explicitly not a generated robot image."""
    points = np.asarray(trial.reference.states)[:, :2]
    if not np.isfinite(points).all():
        return '<div class="endpoint-empty">状态示意不可用</div>'
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    x, y = (points[index] - low) / span
    x, y = 42 + float(x) * 276, 160 - float(y) * 126
    return f'''<svg role="img" aria-label="{title} · 合成状态示意，非原始图片" viewBox="0 0 360 200">
<rect width="360" height="200" fill="#f0f4fa"/>
<path d="M42 26V160H330" fill="none" stroke="#b7c4d6"/>
<path d="M42 118H330M42 76H330M134 26V160M226 26V160" fill="none" stroke="#e0e7f0"/>
<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#2457db"/>
<text x="42" y="187" fill="#718097" font-size="12">d0={points[index, 0]:.3f} · d1={points[index, 1]:.3f}</text>
</svg>'''


def endpoint_images_html(trial: Trial) -> str:
    cards = []
    for title, step, index, path in zip(
        ("初始状态", "目标状态"),
        (trial.start, trial.goal),
        (0, -1),
        endpoint_paths(trial),
    ):
        note = "参考轨迹原图"
        if path:
            try:
                body = f'<img src="{image_uri(path)}" alt="{title} · 原始帧 {step}">'
            except (OSError, ValueError):
                body = '<div class="endpoint-empty">图片读取失败</div>'
                note = "请检查参考轨迹图像文件"
        elif (
            trial.demo
            and trial.reference.states is not None
            and trial.reference.states.shape[1] >= 2
        ):
            body = _demo_state_image(trial, index, title)
            note = "合成状态示意 · 非原始图片"
        else:
            body = '<div class="endpoint-empty">原始图片待接入</div>'
            note = "不使用方法输出代替目标原图"
        cards.append(
            f'<figure class="endpoint-card"><div class="endpoint-title">{title}<span>原始帧 {step}</span></div>{body}<figcaption>{html.escape(note)}</figcaption></figure>'
        )
    return (
        """<style>
.endpoint-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:8px 0 18px}
.endpoint-card{margin:0;background:#fff;border:1px solid #dee5f0;border-radius:10px;overflow:hidden}
.endpoint-title{display:flex;justify-content:space-between;gap:8px;padding:12px 15px;font-size:14px;font-weight:600;color:#16263e}
.endpoint-title span{font-size:12px;font-weight:400;color:#718097}
.endpoint-card img,.endpoint-card svg{display:block;width:100%;height:200px;object-fit:contain;background:#f0f4fa}
.endpoint-card figcaption{padding:9px 15px;color:#718097;font-size:12px}
.endpoint-empty{height:200px;display:flex;align-items:center;justify-content:center;background:#f0f4fa;color:#7b899b;font-size:14px}
@media(max-width:600px){.endpoint-row{gap:8px}.endpoint-title{flex-direction:column;padding:10px}.endpoint-card img,.endpoint-card svg,.endpoint-empty{height:150px}.endpoint-card figcaption{padding:8px 10px}}
</style><div class="endpoint-row" aria-label="初始状态与目标状态图片">"""
        + "".join(cards)
        + "</div>"
    )
