import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analysis import (
    REGISTRY,
    action_summary,
    dimension_summary,
    distribution_histogram,
    finite_rows,
    project_states,
)
from .data import REPO_ROOT
from .models import method_catalog

PALETTE = ["#6A7B93", "#2457DB", "#BA6B25", "#8654C7"]


def plot(fig, key):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="-apple-system, sans-serif", color="#36465e", size=13),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", y=1.15, x=0),
        height=370,
    )
    fig.update_xaxes(gridcolor="#EDF1F7", zerolinecolor="#e3e9f2")
    fig.update_yaxes(gridcolor="#EDF1F7", zerolinecolor="#e3e9f2")
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def collect_groups(trials, field, kind, selected_names):
    groups, coverage = {}, []
    catalog = method_catalog(trials)
    for name in ["目标参考轨迹"] + list(selected_names):
        label = catalog[name].label if name in catalog else name
        arrays, found = [], 0
        for trial in trials:
            track = (
                trial.reference
                if name == "目标参考轨迹"
                else trial.methods.get(name, {}).get(kind)
            )
            values = getattr(track, field, None)
            if values is not None and len(values):
                arrays.append(values)
                found += 1
        coverage.append({"系列": label, "有数据的轨迹": found, "所选轨迹": len(trials)})
        if arrays:
            if len({a.shape[1] for a in arrays}) != 1:
                raise ValueError(f"{name} 的数组维度不一致，不能合并")
            groups[label] = np.concatenate(arrays)
    return groups, pd.DataFrame(coverage)


def draw_distribution(groups, dimension, key):
    hist = distribution_histogram(groups, dimension)
    if hist.empty:
        st.info("所选数据没有有效数值。")
        return
    plot(
        px.line(
            hist,
            x="值",
            y="比例",
            color="系列",
            markers=True,
            color_discrete_sequence=PALETTE,
            labels={"值": f"d{dimension} 的值", "比例": "分箱频率"},
        ),
        key,
    )


def render_analysis(trials, page, names):
    if not trials:
        st.info("当前组没有样本。请先选择有数据的来源。")
        return
    is_action = page == "Action 分析"
    controls = st.columns([1.5, 1.5, 2])
    scope = controls[0].selectbox("分析范围", ["全部测试", "单条测试"])
    kind = controls[1].selectbox(
        "方法数据",
        ["executed", "predicted"],
        format_func=lambda k: "实际执行" if k == "executed" else "模型预测",
    )
    catalog = method_catalog(trials)
    selected_names = controls[2].multiselect(
        "分析组合（训练 · 搜索）",
        names,
        default=names[:2],
        format_func=lambda name: catalog[name].label,
    )
    selected = trials
    if scope == "单条测试":
        i = st.selectbox(
            "测试编号",
            range(len(trials)),
            format_func=lambda i: f"{i + 1:02d} · Episode {trials[i].episode:04d}",
        )
        selected = [trials[i]]
    field = (
        "actions"
        if is_action
        else st.radio(
            "分析对象",
            ["states", "embeddings"],
            horizontal=True,
            format_func=lambda k: "环境状态 s" if k == "states" else "模型表征 z",
        )
    )
    if not is_action:
        st.caption(
            "环境状态是位置等物理量；模型表征是编码器输出的向量。两者分开分析，不从成功标签反推。"
        )
    else:
        st.caption(
            "展示保存的原始 action 数值。不会擅自把它当作 [-1, 1] 归一化数据，也不会把执行 action 与预测 action 混合。"
        )
    try:
        groups, coverage = collect_groups(selected, field, kind, selected_names)
    except ValueError as exc:
        st.error(str(exc))
        return
    if "目标参考轨迹" in groups and len(groups) == 1 and selected_names:
        st.warning(
            "当前只能分析原数据集的参考轨迹；所选方法没有保存对应数组，以下分布不代表方法执行或预测结果。"
        )
    if not groups:
        st.info(
            f"这些真实记录没有 {field} 数组，暂时不能计算分布。成功标签已接入，但不能代替状态或 action。"
        )
        st.dataframe(coverage, hide_index=True, width="stretch")
        st.caption(
            "在「数据与扩展」中查看数组接入格式；也可以切换到「交互演示」试用分析功能。"
        )
        return
    with st.expander("数据覆盖与计算口径"):
        st.dataframe(coverage, hide_index=True, width="stretch")
        st.write(
            "每个有效时间步权重相同（不是每条轨迹等权）。分布和 PCA 排除含 NaN/Inf 的整行；原始数据不修改。"
        )
        st.write(
            "PCA 最多均匀抽样 10,000 个时间步，在所选系列上共同拟合；不代表完整数据覆盖或任务成功率。"
        )
    points = sum(len(finite_rows(v)) for v in groups.values())
    removed = sum(len(v) - len(finite_rows(v)) for v in groups.values())
    stats = st.columns(3)
    stats[0].metric("有效时间步", f"{points:,}")
    stats[1].metric("系列", len(groups))
    stats[2].metric("排除非有限值行", removed)
    min_dim = min(v.shape[1] for v in groups.values())
    if not is_action:
        st.subheader("状态空间分布" if field == "states" else "表征空间分布")
        can_compare = True
        if field == "embeddings":
            can_compare = st.checkbox(
                "确认这些表征来自同一个编码器 / 同一个特征坐标系",
                value=all(t.demo for t in selected),
            )
            st.caption(
                "不同编码器即使维度相同，也不能直接在同一 PCA 坐标里比较。未确认时，只投影单个系列。"
            )
        proj_groups = groups
        if not can_compare:
            one = st.selectbox("投影系列", list(groups))
            proj_groups = {one: groups[one]}
        standardize = st.checkbox("PCA 前按维度标准化", value=True)
        left, right = st.columns([1.4, 1])
        with left:
            try:
                frame, explained = project_states(proj_groups, standardize)
                fig = px.scatter(
                    frame,
                    x="PC1",
                    y="PC2",
                    color="系列",
                    opacity=0.55,
                    color_discrete_sequence=PALETTE,
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(
                    xaxis_title=f"PC1 · {explained[0]:.1%}",
                    yaxis_title=f"PC2 · {explained[1]:.1%}",
                )
                plot(fig, "pca")
                st.caption(
                    f"前两主成分解释方差：{explained.sum():.1%}。点是时间步，不是独立测试样本。"
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.warning(str(exc))
        with right:
            dimension = st.selectbox(
                "查看分布维度", range(min_dim), format_func=lambda d: f"d{d}"
            )
            draw_distribution(groups, dimension, "state_hist")
    else:
        st.subheader("Action 分布")
        dimension = st.selectbox(
            "Action 维度", range(min_dim), format_func=lambda d: f"action[{d}]"
        )
        left, right = st.columns([1.2, 1])
        with left:
            draw_distribution(groups, dimension, "action_hist")
        with right:
            summaries = []
            for name, values in groups.items():
                summary = action_summary(values)
                if not summary.empty:
                    summary["系列"] = name
                    summaries.append(summary)
            if summaries:
                table = pd.concat(summaries)
                plot(
                    px.bar(
                        table,
                        x="维度",
                        y="标准差",
                        color="系列",
                        barmode="group",
                        color_discrete_sequence=PALETTE,
                    ),
                    "action_std",
                )
        st.subheader("单条 Action 时间序列")
        selected_i = st.selectbox(
            "查看哪条测试",
            range(len(selected)),
            format_func=lambda i: (
                f"{selected[i].index:02d} · Episode {selected[i].episode:04d}"
            ),
        )
        single, _ = collect_groups([selected[selected_i]], field, kind, selected_names)
        fig = go.Figure()
        for i, (name, values) in enumerate(single.items()):
            if dimension < values.shape[1]:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(values)),
                        y=values[:, dimension],
                        name=name,
                        mode="lines",
                        line=dict(color=PALETTE[i % len(PALETTE)]),
                    )
                )
        fig.update_layout(
            xaxis_title="各自的实际 action 步序号（未插值）",
            yaxis_title=f"action[{dimension}]",
        )
        plot(fig, "action_time")
    st.subheader("逐维统计")
    summary_name = st.selectbox("统计系列", list(groups))
    st.dataframe(
        dimension_summary(groups[summary_name]), hide_index=True, width="stretch"
    )
    # Extra analyses can be added without rewriting the page navigation.
    extras = {
        key: spec
        for key, spec in REGISTRY.items()
        if key not in ("pca", "histogram", "summary")
    }
    if extras:
        extra = st.selectbox(
            "扩展分析", list(extras), format_func=lambda k: extras[k].title
        )
        spec = extras[extra]
        st.caption(spec.description)
        if st.button("运行扩展分析"):
            try:
                result = spec.function(
                    groups if spec.input_kind == "groups" else groups[summary_name]
                )
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, width="stretch")
                elif isinstance(result, go.Figure):
                    plot(result, "extension")
                else:
                    st.write(result)
            except Exception as exc:
                st.error(f"扩展分析执行失败：{exc}")


def render_data(trials):
    st.subheader("一次接入，统一管理")
    st.write(
        "每条测试由 episode、起始帧、目标帧共同标识。目标参考轨迹、实际执行轨迹、模型预测轨迹是三类独立数据，不互相冒充。"
    )
    rows = []
    for trial in trials:
        for name, tracks in [("目标参考轨迹", {"reference": trial.reference})] + list(
            trial.methods.items()
        ):
            for kind, track in tracks.items():
                rows.append(
                    {
                        "测试": trial.index,
                        "Episode": trial.episode,
                        "系列": name,
                        "轨迹类型": kind,
                        "图像帧数": len(track.frames),
                        "视频": bool(track.video),
                        "状态": None
                        if track.states is None
                        else str(track.states.shape),
                        "表征": None
                        if track.embeddings is None
                        else str(track.embeddings.shape),
                        "Action": None
                        if track.actions is None
                        else str(track.actions.shape),
                        "成功": track.success,
                    }
                )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=310)
    with st.expander("当前记录来源"):
        st.text("\n".join(dict.fromkeys(t.source for t in trials)))
    st.subheader("已准备的分析函数")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "函数": spec.function.__name__,
                    "名称": spec.title,
                    "说明": spec.description,
                }
                for spec in REGISTRY.values()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("新增分析：编写纯函数，再注册到 REGISTRY。页面会自动出现扩展分析入口。")
    st.subheader("统一数据清单")
    st.write(
        "在左侧选择「自定义数据清单」并填写 JSON 路径。数组使用 .npy；图像按播放顺序列出路径；也支持完整视频。所有路径相对于清单目录。"
    )
    st.code(
        (REPO_ROOT / "configs/result_studio/manifest.example.json").read_text(),
        language="json",
    )
    st.info(
        "这份清单只是格式示例，不含实际图像。没有保存预测图像的模型，只能分析其预测表征；页面不会自动把 latent 还原成图片。目标参考路径需要原始数据集中 start 到 goal 的真实序列。"
    )
    st.caption(
        "attributes 为后续属性保留。第一版只显示原始成功／失败标签，不覆盖评测结论。"
    )
