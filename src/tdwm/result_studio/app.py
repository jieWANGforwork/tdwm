import html
import os

import streamlit as st

from .data import REAL_PREVIEW_MANIFEST, demo_trials, historical_trials, load_manifest
from .endpoints import endpoint_images_html
from .player import CARD_WIDTH, PREVIEW_SIZE, player_html

st.set_page_config(
    page_title="World Model · Result Studio", page_icon="◈", layout="wide"
)
st.markdown(
    """<style>
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1500px}header[data-testid="stHeader"]{background:transparent}[data-testid="stSidebar"]{background:#12223a;color:#dfe8f7}[data-testid="stSidebar"] *{color:inherit}[data-testid="stSidebar"] h1{font-size:23px!important;letter-spacing:-.5px}[data-testid="stSidebar"] [data-baseweb="select"] *{color:#182840}[data-testid="stSidebar"] hr{border-color:#2b3d57}h1{font-size:30px!important;letter-spacing:-.8px}h2{font-size:21px!important}h3{font-size:17px!important}.eyebrow{color:#6e7e96;font:600 12px sans-serif;letter-spacing:2px;margin-bottom:7px}.subhead{color:#728198;font-size:14px;margin-top:-9px;margin-bottom:20px}.pill{display:inline-block;background:#e8efff;color:#2d56b5;border-radius:5px;padding:5px 10px;font-size:12px}.notice{border-left:3px solid #d39a29;background:#fff6df;padding:11px 15px;color:#755619;font-size:14px;border-radius:0 7px 7px 0;margin:8px 0 18px}.stat{background:#fff;border:1px solid #e1e7f0;border-radius:10px;padding:15px 18px;margin-bottom:18px}.stat-label{color:#6e7d94;font-size:13px}.stat-number{font-size:26px;font-weight:650;margin:5px 0}.stat-note{font-size:12px;color:#8290a4}.badge{border-radius:4px;padding:4px 8px;font-size:13px;display:inline-block;background:#eef1f6;color:#67768c}.success{background:#e5f4ec;color:#167452}.failure{background:#fcedef;color:#b23f50}.card-note{color:#7c899b;font-size:12px}.section-line{border-top:1px solid #dde4ee;margin:20px 0}[data-testid="stButton"] button{border-radius:7px}div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:10px}button:focus-visible{outline:3px solid #6686e5!important}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<style>
.stats-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.stats-grid .stat{margin-bottom:0}.stats-grid{margin-bottom:18px}[data-testid="stSidebar"] [role="combobox"],[data-testid="stSidebar"] [role="combobox"] *{color:#14243d!important}@media(max-width:700px){.stats-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.stat{padding:12px}.stat-number{font-size:23px}.block-container{padding-top:1.5rem}.stats-grid .stat-note{font-size:12px}}
.st-key-episode_grid [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:5px}.st-key-episode_grid [data-testid="stColumn"]{min-width:0!important;width:calc(10% - 5px)!important;flex:1 1 0!important}.st-key-episode_grid button{padding:5px 0!important;min-height:36px}
</style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("◈  World Model")
    st.caption("RESULT STUDIO / 实验工作台")
    st.divider()
    page = st.radio(
        "工作区",
        ["轨迹对比", "状态与表征", "Action 分析", "数据与扩展"],
        label_visibility="collapsed",
    )
    st.divider()
    mode = st.selectbox(
        "数据来源",
        [
            "真实图像 · 数据预览",
            "历史评测记录",
            "自定义数据清单",
            "交互演示 · 合成数据",
        ],
        key="data_source_real_images",
    )
    st.caption("演示与真实结果独立，绝不混算。")
    manifest = (
        st.text_input("数据清单路径", os.environ.get("RESULT_STUDIO_MANIFEST", ""))
        if mode == "自定义数据清单"
        else ""
    )
    st.divider()
    st.caption("PRIVATE PREVIEW · 支持 SSH 访问")
    st.caption("先选组别，再看全部 50 条。\n目标 / 执行 / 预测分别管理。")

st.markdown('<div class="eyebrow">EXPERIMENT WORKSPACE</div>', unsafe_allow_html=True)
st.title(page)
st.markdown(
    '<div class="subhead">完整轨迹 · 逐条结果 · 可扩展分析</div>',
    unsafe_allow_html=True,
)
offset = (
    st.segmented_control(
        "目标间隔",
        [25, 50, 100],
        default=25,
        format_func=lambda v: f"O{v}",
        key="offset",
    )
    or 25
)
if mode == "真实图像 · 数据预览":
    if REAL_PREVIEW_MANIFEST.is_file():
        try:
            trials, notes = load_manifest(str(REAL_PREVIEW_MANIFEST), offset), []
        except (OSError, ValueError, KeyError, TypeError) as exc:
            trials, notes = [], [str(exc)]
    else:
        trials, notes = [], []
    loaded = sum(bool(t.reference.frames) for t in trials)
    st.info(
        f"真实机器人图片 · O{offset} 参考轨迹已接入 {loaded} / 50 条。初始图、目标图和参考轨迹来自服务器数据集；方法成功标签来自评测记录，方法轨迹尚未接入。"
    )
elif mode.startswith("交互"):
    trials, notes = demo_trials(offset), []
    st.markdown(
        '<div class="notice">界面演示：当前轨迹、分布和成功标签均为合成数据，不是你的实验结论。可从左侧切换到历史评测记录。</div>',
        unsafe_allow_html=True,
    )
elif mode == "历史评测记录":
    trials, notes = historical_trials(offset)
    st.info(
        "真实历史记录：v1-c / f_only 与 f_plus_g。成功标签来自评测文件；本地尚无对应的轨迹图像和状态数组。"
    )
else:
    if not manifest:
        st.info("在左侧填写 manifest.json 路径。格式说明见「数据与扩展」。")
        trials, notes = [], []
    else:
        try:
            trials, notes = load_manifest(manifest, offset), []
        except (OSError, ValueError, KeyError, TypeError) as exc:
            trials, notes = [], [str(exc)]
for note in notes:
    st.warning(note)

names = list(dict.fromkeys(name for trial in trials for name in trial.methods))


def status_badge(success):
    text, cls = (
        ("成功", "success")
        if success is True
        else ("失败", "failure")
        if success is False
        else ("未标注", "")
    )
    return f'<span class="badge {cls}">{text}</span>'


def overview():
    values = [
        ("测试组", f"O{offset}", "目标帧 − 起始帧"),
        ("测试样本", f"{len(trials)} / 50", "包含成功与失败，不默认筛除"),
    ]
    for name in names[:2]:
        labels = [
            t.methods[name]["executed"].success
            for t in trials
            if name in t.methods and "executed" in t.methods[name]
        ]
        known = [label for label in labels if label is not None]
        values.append(
            (
                name,
                f"{sum(known)} / {len(known)}",
                "成功 / 已标注 · "
                + ("演示" if mode.startswith("交互") else "真实记录"),
            )
        )
    cards = "".join(
        f'<div class="stat"><div class="stat-label">{html.escape(label)}</div><div class="stat-number">{number}</div><div class="stat-note">{note}</div></div>'
        for label, number, note in values
    )
    st.markdown(f'<div class="stats-grid">{cards}</div>', unsafe_allow_html=True)


def comparison_page():
    overview()
    if not trials:
        st.info("当前组没有可读取的样本。")
        return
    control, method_col, view_col = st.columns([1.5, 2, 1.3])
    index = control.selectbox(
        "当前测试样本",
        range(len(trials)),
        format_func=lambda i: (
            f"{i + 1:02d} / {len(trials)} · Episode {trials[i].episode:04d}"
        ),
        key=f"episode_{mode}_{offset}",
    )
    chosen = method_col.multiselect(
        "并排比较方法",
        names,
        default=names[:2],
        help="不限选择数量。每个画面固定 224 × 224，方法多时自动换行，不随数量拉伸。",
    )
    kind = view_col.selectbox(
        "方法轨迹类型",
        ["executed", "predicted"],
        format_func=lambda v: "实际执行轨迹" if v == "executed" else "模型预测轨迹",
    )
    trial = trials[index]
    st.markdown(f"### 样本 {trial.index:02d}  ·  Episode {trial.episode:04d}")
    st.html(endpoint_images_html(trial))
    st.caption(
        f"参考路径：原始帧 {trial.start} — {trial.goal}，包含中间过程。预测轨迹不自动等同于执行轨迹。"
    )
    from .models import Trajectory

    tracks = [trial.reference] + [
        trial.methods.get(name, {}).get(kind, Trajectory(name, kind)) for name in chosen
    ]
    try:
        st.iframe(player_html(tracks, trial.demo), height="content")
    except (OSError, ValueError) as exc:
        st.error(f"轨迹读取失败：{exc}")
    st.caption(
        "图像序列和状态坐标共用同步控件。视频文件接入后可直接播放；不同时间长度按相对进度对齐。"
    )
    if any(t.video for t in tracks):
        with st.container(horizontal=True, wrap=True, horizontal_alignment="left"):
            for track in tracks:
                if track.video:
                    with st.container(width=CARD_WIDTH):
                        st.caption(track.name + " · 完整视频（独立时间轴）")
                        st.video(track.video, width=PREVIEW_SIZE)
    if any(t.frames for t in tracks):
        with st.expander("查看任意原始帧 · 不抽样"):
            image_tracks = [t for t in tracks if t.frames]
            image_track = st.selectbox(
                "图像轨迹",
                range(len(image_tracks)),
                format_func=lambda i: image_tracks[i].name,
            )
            selected = image_tracks[image_track]
            frame_index = (
                st.slider("原始帧编号", 0, len(selected.frames) - 1, 0)
                if len(selected.frames) > 1
                else 0
            )
            st.image(
                selected.frames[frame_index],
                caption=f"{selected.name} · frame {frame_index}",
            )
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.subheader(f"全部测试 · {len(trials)} 条")
    st.caption("每个方块对应一条测试。下方逐条列出方法标签，成功与失败全部保留。")
    with st.container(key="episode_grid"):
        for row in range((len(trials) + 9) // 10):
            columns = st.columns(10)
            for j, col in enumerate(columns):
                i = row * 10 + j
                if i < len(trials):

                    def jump(value=i):
                        st.session_state[f"episode_{mode}_{offset}"] = value

                    col.button(
                        f"{i + 1:02d}",
                        key=f"jump_{i}",
                        type="primary" if i == index else "secondary",
                        width="stretch",
                        on_click=jump,
                    )
    gallery = st.toggle("批量浏览完整轨迹 · 每页 5 条", value=False)
    if gallery:
        total_pages = (len(trials) + 4) // 5
        gallery_page = st.selectbox(
            "轨迹页码",
            range(total_pages),
            format_func=lambda p: f"第 {p + 1} / {total_pages} 页",
            key="gallery_page",
        )
        for item in trials[gallery_page * 5 : gallery_page * 5 + 5]:
            st.markdown(f"**{item.index:02d} · Episode {item.episode:04d}**")
            st.html(endpoint_images_html(item))
            item_tracks = [item.reference] + [
                item.methods.get(name, {}).get(kind, Trajectory(name, kind))
                for name in chosen
            ]
            try:
                st.iframe(player_html(item_tracks, item.demo), height="content")
            except (OSError, ValueError) as exc:
                st.warning(f"该条轨迹读取失败：{exc}")
    with st.expander("逐条成功 / 失败标签 · 展开全部", expanded=True):
        for trial in trials:
            cells = st.columns([1.5] + [1] * len(names))
            cells[0].markdown(f"**{trial.index:02d}**　Episode {trial.episode:04d}")
            for col, name in zip(cells[1:], names):
                track = trial.methods.get(name, {}).get("executed")
                col.markdown(
                    html.escape(name)
                    + "　"
                    + status_badge(track.success if track else None),
                    unsafe_allow_html=True,
                )


if page == "轨迹对比":
    comparison_page()
else:
    from .views import render_analysis, render_data

    if page == "数据与扩展":
        render_data(trials)
    else:
        render_analysis(trials, page, names)
