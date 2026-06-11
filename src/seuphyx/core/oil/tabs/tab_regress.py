"""
Tab 4: AI discovery-based charge clustering and fitting.
"""
# built-in
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sympy as sp

# seuphyx
from seuphyx.core.oil.tabs.regression import (
    CHARGE_CENTER_COL,
    CHARGE_CLUSTER_COL,
    CHARGE_DISTANCE_COL,
    CHARGE_HALF_WIDTH_COL,
    CHARGE_UNIT_COL,
    DISCOVERY_QUALITY_COL,
    DiscoveryRegressionConfig,
    PhysicsRegressionConfig,
    SOURCE_COL,
    TIME_COL,
    USE_FOR_FIT_COL,
    VOLTAGE_COL,
    add_charge_estimates,
    discovery_regression,
    physics_guided_regression,
)
import seuphyx


RESULT_VERSION = "q-ai-clustering-symbolic-v8"


@st.cache_data(show_spinner=False)
def _read_reference_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).dropna(subset=[TIME_COL, VOLTAGE_COL])


def _load_reference_data() -> pd.DataFrame:
    root_reference = Path.cwd() / "oil_drop_reference.csv"
    data_dir = Path(seuphyx.__file__).parent / "data"
    reference_file = (
        root_reference if root_reference.exists()
        else data_dir / "oil_drop_reference.csv")
    data_ref = _read_reference_csv(str(reference_file)).copy()
    data_ref[SOURCE_COL] = f"测试数据:{reference_file.name}"
    return data_ref


def _build_analysis_data(include_reference: bool) -> pd.DataFrame:
    data_parts = []
    user_data = st.session_state.data.copy()
    if not user_data.empty:
        user_data[SOURCE_COL] = "实验数据"
        data_parts.append(user_data)
    if include_reference:
        data_parts.append(_load_reference_data())
    if not data_parts:
        return pd.DataFrame(columns=[TIME_COL, VOLTAGE_COL, SOURCE_COL])
    return pd.concat(data_parts, axis=0, ignore_index=True)


def _plot_measurement_to_charge(data: pd.DataFrame,
                                config: DiscoveryRegressionConfig):
    try:
        charged = add_charge_estimates(data, config)
    except Exception:
        return
    sample = charged[[TIME_COL, VOLTAGE_COL, CHARGE_UNIT_COL]].head(8)

    with st.container(border=True):
        st.subheader("从宏观读数得到电荷估计")
        cols = st.columns(4)
        cols[0].metric("计时距离/mm", f"{config.fall_distance_mm:.2f}")
        cols[1].metric("极板间距/mm", f"{config.plate_distance_mm:.2f}")
        cols[2].metric("有效数据点", len(charged))
        cols[3].metric("q 中位数/1e-19C",
                       f"{charged[CHARGE_UNIT_COL].median():.3f}")
        st.dataframe(sample, use_container_width=True, hide_index=True)


def _plot_charge_density(result: dict):
    clusters = result["clusters"]
    q_values = clusters[CHARGE_UNIT_COL].dropna()
    if q_values.empty:
        return

    peaks = result.get("peaks", [])
    if peaks:
        left = min(peak["center"] - 2.2 * peak["half_width"] for peak in peaks)
        right = max(peak["center"] + 2.2 * peak["half_width"] for peak in peaks)
        plot_min = 0.0 if left < 1.0 else max(0.0, left)
        plot_max = max(10.0, right)
    else:
        plot_min, plot_max = np.percentile(q_values, [1, 95])
        plot_max = min(max(10.0, plot_max), 12.0)
    visible_q = q_values[(q_values >= plot_min) & (q_values <= plot_max)]
    hidden_count = int(len(q_values) - len(visible_q))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=visible_q,
            histnorm="probability density",
            xbins=dict(start=plot_min, end=plot_max, size=0.12),
            name="q 分布",
            marker_color="rgba(54, 105, 163, 0.42)",
        ))

    density = result.get("density", {})
    grid = density.get("grid", np.array([]))
    kde_values = density.get("density", np.array([]))
    if len(grid) and len(kde_values):
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=kde_values,
                mode="lines",
                name="KDE 密度",
                line=dict(color="#0f3d5e", width=3),
            ))

    palette = px.colors.qualitative.Dark24
    for idx, peak in enumerate(peaks):
        center = peak["center"]
        half_width = peak["half_width"]
        color = palette[idx % len(palette)]
        fig.add_vrect(
            x0=center - half_width,
            x1=center + half_width,
            fillcolor=color,
            opacity=0.12,
            line_width=0,
        )
        fig.add_vline(
            x=center,
            line_color=color,
            line_width=2,
            line_dash="dash",
            annotation_text=f"峰{peak['cluster']}",
            annotation_position="top",
        )

    fig.update_layout(
        title="q 分布中的最终峰与半峰宽有效区间",
        xaxis_title="电荷估计 q / 1e-19 C",
        yaxis_title="概率密度",
        bargap=0.04,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(range=[plot_min, plot_max], autorange=False)
    st.plotly_chart(fig, key="charge_density_plot_final_peaks",
                    use_container_width=True)
    if hidden_count > 0:
        st.caption(
            f"图中只显示最终峰附近的数据；{hidden_count} 个长尾或半峰宽外点未进入该视窗，但仍保留在明细表中。")


def _plot_spacing_discovery(result: dict):
    spacing = result.get("spacing", {})
    centers = spacing.get("centers_1e19C")
    fitted = spacing.get("equal_spacing_fit_1e19C")
    gaps = spacing.get("gaps_1e19C")
    if centers is None or len(centers) < 2:
        return
    peak_rank = np.arange(1, len(centers) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=peak_rank,
            y=centers,
            mode="markers+lines+text",
            name="发现的 q 峰中心",
            text=[f"{value:.2f}" for value in centers],
            textposition="top center",
            marker=dict(size=11),
            line=dict(width=2),
        ))
    fig.add_trace(
        go.Scatter(
            x=peak_rank,
            y=fitted,
            mode="lines",
            name="等间距后验检验",
            line=dict(width=3, dash="dash"),
        ))

    if gaps is not None:
        for idx, gap in enumerate(gaps, start=1):
            fig.add_annotation(
                x=idx + 0.5,
                y=(centers[idx - 1] + centers[idx]) / 2,
                text=f"间距 {gap:.2f}",
                showarrow=False,
                yshift=-28,
                font=dict(size=12),
            )

    fig.update_layout(
        title="从峰中心后验发现共同电荷间距",
        xaxis_title="发现峰序号（按 q 从小到大排序）",
        yaxis_title="峰中心 q / 1e-19 C",
        xaxis=dict(dtick=1),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="charge_spacing_plot", use_container_width=True)


def _plot_discovered_curves(result: dict):
    clustered = result["clusters"]
    fig = go.Figure()
    palette = px.colors.qualitative.Dark24

    fit_data = clustered[clustered[USE_FOR_FIT_COL]]
    for idx, cluster_id in enumerate(
            sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique())):
        sub = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
        fig.add_trace(
            go.Scatter(
                x=sub[TIME_COL],
                y=sub[VOLTAGE_COL],
                mode="markers",
                name=f"q峰{int(cluster_id)} 高置信点",
                marker=dict(size=8, color=palette[idx % len(palette)]),
            ))

    outliers = clustered[~clustered[USE_FOR_FIT_COL]]
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=outliers[TIME_COL],
                y=outliers[VOLTAGE_COL],
                mode="markers",
                name="半峰宽外数据",
                marker=dict(size=6, color="rgba(120,120,120,0.55)"),
            ))

    for idx, (cluster_id, (t_line, y_line, _)) in enumerate(
            result["data"].items()):
        fig.add_trace(
            go.Scatter(
                x=t_line,
                y=y_line,
                mode="lines",
                name=f"共享公式曲线 q峰{cluster_id}",
                line=dict(width=3, color=palette[(idx + 8) % len(palette)]),
            ))

    fig.update_layout(
        title="AI 发现的 q 峰在 U-t 平面中的多曲线结构",
        xaxis_title="下落时间 t / s",
        yaxis_title="平衡电压 U / V",
        font=dict(family="DejaVu Serif", size=16),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="discovery_curve_plot", use_container_width=True)


def _render_result_summary(result: dict):
    params = result["global_params"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("发现 q 峰数", params["cluster_count"])
    col2.metric("共同间距/1e-19C", f"{params['spacing_1e19C']:.4f}")
    col3.metric("公式 R2", f"{params['formula_r2']:.4f}")
    col4.metric("参与拟合点数", params["fit_points"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("t 幂指数 beta", f"{params['time_power']:.2f}")
    col2.metric("Q_c 幂指数 alpha", f"{params['charge_power']:.2f}")
    col3.metric("公式 RMSE/V", f"{params['formula_rmse']:.3f}")
    method_names = {
        "neural_teacher_distillation": "神经网络蒸馏",
        "two_stage": "两阶段发现",
        "global_candidate_search": "全局候选搜索",
    }
    method_label = method_names.get(params.get("discovery_method"), "")
    col4.metric("公式来源", method_label)

    st.subheader("机器学习—符号回归得到的共享表达式")
    st.caption(
        "这里的 Q_c 是从 q 分布中发现的簇中心，单位为 1e-19 C；公认元电荷没有参与聚类或拟合。"
        "a/coef 表示由数据发现的曲线尺度，b 表示电压零点或系统偏置项；若公式来源为神经网络蒸馏，符号式是对 AI teacher 的可解释压缩。")
    st.latex(rf"U(t,Q_c) = {sp.latex(result['symbolic_expression'])}")


def _render_physics_comparison(analysis_data: pd.DataFrame):
    with st.expander("传统物理拟合对照检验", expanded=False):
        st.caption(
            "该模块保留陈鹏宇已实现的物理约束路径：先用物理公式反推浮点 N，再按整数半宽剔除并迭代修正参数。它作为精度对照，不参与前面的机器学习发现。")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_n = st.number_input("最大整数 N", min_value=2, max_value=12,
                                    value=5)
        with col2:
            peak_width = st.slider("整数半宽", 0.05, 0.80, 0.25, 0.01)
        with col3:
            min_points = st.number_input("每个整数峰最少点数",
                                         min_value=2,
                                         max_value=20,
                                         value=3)

        if st.button("执行传统物理拟合对照", use_container_width=True):
            try:
                result = physics_guided_regression(
                    analysis_data,
                    PhysicsRegressionConfig(
                        max_n=int(max_n),
                        peak_width=float(peak_width),
                        min_points_per_peak=int(min_points),
                    ),
                )
            except Exception as exc:
                st.error(f"传统物理拟合失败: {exc}")
                return

            params = result.get("global_params", {})
            clusters = result.get("clusters", pd.DataFrame())
            fit_points = int(clusters[USE_FOR_FIT_COL].sum()) if (
                not clusters.empty and USE_FOR_FIT_COL in clusters) else 0
            residual_rmse = np.nan
            if (not clusters.empty and "PhysicsResidual(V)" in clusters
                    and USE_FOR_FIT_COL in clusters):
                residuals = clusters.loc[
                    clusters[USE_FOR_FIT_COL], "PhysicsResidual(V)"].dropna()
                if not residuals.empty:
                    residual_rmse = float(np.sqrt(np.mean(residuals**2)))
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("A", f"{params.get('A', np.nan):.3f}")
            col2.metric("b/V", f"{params.get('b', np.nan):.3f}")
            col3.metric("参与点数", fit_points)
            col4.metric("全局 RMSE/V", f"{residual_rmse:.3f}")
            if result.get("peak_summary") is not None:
                st.dataframe(result["peak_summary"], use_container_width=True,
                             hide_index=True)


def _render_ai_workflow(result: dict):
    params = result["global_params"]
    clustered = result["clusters"]
    peaks = result.get("peaks", [])
    spacing = result.get("spacing", {})
    two_stage = result.get("two_stage", {})

    stability_values = [
        peak.get("stability_percent") for peak in peaks
        if np.isfinite(peak.get("stability_percent", np.nan))
    ]
    stability = np.mean(stability_values) if stability_values else np.nan
    beta = params.get("time_power", np.nan)
    alpha = params.get("charge_power", np.nan)
    method = params.get("discovery_method", "")
    method_text = "NN蒸馏" if method == "neural_teacher_distillation" else "符号搜索"

    steps = [
        {
            "step": "U,t",
            "value": f"{len(clustered)}点",
            "detail": "原始复杂读数",
        },
        {
            "step": "q估计",
            "value": f"{clustered[CHARGE_UNIT_COL].median():.2f}",
            "detail": "连续电荷分布",
        },
        {
            "step": "无监督峰",
            "value": f"{params['cluster_count']}峰",
            "detail": "不预设整数倍",
        },
        {
            "step": "半峰宽",
            "value": f"{params['fit_points']}点",
            "detail": "高置信数据",
        },
        {
            "step": "稳定性",
            "value": f"{stability:.0f}%" if np.isfinite(stability) else "-",
            "detail": "重采样复现",
        },
        {
            "step": "符号发现",
            "value": f"{method_text}",
            "detail": f"α={alpha:.2f}, β={beta:.2f}",
        },
    ]

    with st.container(border=True):
        st.subheader("AI 数据处理路径")
        fig = go.Figure()
        x_values = np.arange(len(steps))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.zeros(len(steps)),
                mode="lines+markers+text",
                marker=dict(size=34, color="#25636f"),
                line=dict(width=3, color="#8ab6bd"),
                text=[item["step"] for item in steps],
                textposition="top center",
                customdata=np.array(
                    [[item["value"], item["detail"]] for item in steps]),
                hovertemplate="%{text}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            ))
        for idx, item in enumerate(steps):
            fig.add_annotation(
                x=idx,
                y=-0.28,
                text=f"{item['value']}<br>{item['detail']}",
                showarrow=False,
                align="center",
                font=dict(size=13),
            )
        fig.update_layout(
            height=230,
            margin=dict(l=20, r=20, t=50, b=60),
            xaxis=dict(visible=False, range=[-0.4, len(steps) - 0.6]),
            yaxis=dict(visible=False, range=[-0.55, 0.35]),
            showlegend=False,
        )
        st.plotly_chart(fig, key="ai_discovery_workflow_v1",
                        use_container_width=True)
        neural = result.get("neural_teacher", {})
        if neural:
            metrics = neural.get("teacher_metrics", {})
            st.caption(
                f"AI teacher 使用 {metrics.get('model_count', 0)} 个 MLP 组成集成，在留出测试集上的 RMSE 为 {metrics.get('test_rmse', np.nan):.2f} V；最终公式是对神经网络平滑曲面的符号蒸馏。")
        elif two_stage:
            time_candidates = two_stage.get("time_stage_candidates",
                                            pd.DataFrame())
            coefficient_candidates = two_stage.get(
                "coefficient_stage_candidates", pd.DataFrame())
            time_count = len(
                time_candidates) if isinstance(time_candidates,
                                               pd.DataFrame) else 0
            coefficient_count = len(
                coefficient_candidates) if isinstance(
                    coefficient_candidates, pd.DataFrame) else 0
            st.caption(
                f"符号发现没有指定正确公式；系统比较了 {time_count} 个共同 t 指数候选和 {coefficient_count} 个 Q_c 幂关系候选，并在误差接近时优先选择更简洁的有理幂。")


def _plot_neural_teacher(result: dict):
    neural = result.get("neural_teacher", {})
    if not neural:
        st.info("神经网络 teacher 未启用或训练失败，本次结果回退到传统符号发现。")
        return

    metrics = neural.get("teacher_metrics", {})
    teacher_points = neural.get("teacher_points", pd.DataFrame())
    candidates = neural.get("candidate_models", pd.DataFrame())

    with st.container(border=True):
        st.subheader("神经网络辅助去噪与符号蒸馏")
        st.caption(
            "神经网络不直接给出物理结论，而是先学习噪声数据中的平滑 U=f(t,Q_c) 关系；符号回归再把这个 AI teacher 压缩成可解释公式。")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AI模型", metrics.get("model", "MLP"))
        col2.metric("集成数量", metrics.get("model_count", 0))
        col3.metric("测试RMSE/V", f"{metrics.get('test_rmse', np.nan):.2f}")
        col4.metric("测试R2", f"{metrics.get('test_r2', np.nan):.3f}")

        if not teacher_points.empty:
            fig = go.Figure()
            clustered = result["clusters"]
            fit_data = clustered[clustered[USE_FOR_FIT_COL]]
            palette = px.colors.qualitative.Dark24
            for idx, cluster_id in enumerate(
                    sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique())):
                raw = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
                teacher = teacher_points[
                    teacher_points["cluster"] == int(cluster_id)]
                color = palette[idx % len(palette)]
                fig.add_trace(
                    go.Scatter(
                        x=raw[TIME_COL],
                        y=raw[VOLTAGE_COL],
                        mode="markers",
                        name=f"峰{int(cluster_id)} 原始高置信点",
                        marker=dict(size=5, color=color, opacity=0.35),
                    ))
                fig.add_trace(
                    go.Scatter(
                        x=teacher[TIME_COL],
                        y=teacher["TeacherVoltage(U/V)"],
                        mode="lines",
                        name=f"峰{int(cluster_id)} AI平滑曲线",
                        line=dict(width=3, color=color),
                    ))
            fig.update_layout(
                title="AI teacher 对多条 U-t 曲线的平滑学习",
                xaxis_title="下落时间 t / s",
                yaxis_title="平衡电压 U / V",
                margin=dict(l=60, r=30, t=60, b=60),
            )
            st.plotly_chart(fig, key="neural_teacher_curve_plot_v1",
                            use_container_width=True)

        if not candidates.empty:
            role_mask = (
                candidates["role"] != ""
                if "role" in candidates.columns
                else pd.Series(False, index=candidates.index))
            selected_mask = (
                candidates["selected"]
                if "selected" in candidates.columns
                else pd.Series(False, index=candidates.index))
            keep_indices = list(
                dict.fromkeys(
                    list(candidates.head(5).index) +
                    list(candidates[role_mask | selected_mask].index)))
            display = candidates.loc[keep_indices].copy()
            display.insert(0, "rank", np.arange(1, len(display) + 1))
            st.markdown("**AI teacher 蒸馏出的候选公式**")
            for _, row in display.iterrows():
                with st.container(border=True):
                    role_names = {
                        "lowest_error": "最低误差经验式",
                        "simple_main_law": "简洁主规律式",
                        "pareto_simple": "Pareto简洁式",
                    }
                    role = role_names.get(row.get("role", ""), "")
                    selected = " · 主推" if bool(row.get("selected", False)) else ""
                    role_label = f" · {role}" if role else ""
                    st.markdown(
                        f"候选 {int(row['rank'])}{selected}{role_label} · `{row['family']}` · "
                        f"RMSE `{row['rmse']:.3f} V` · R² `{row['r2']:.4f}`")
                    st.latex(rf"U(t,Q_c) = {row['latex']}")


def _plot_peak_stability(result: dict):
    summary = result.get("peak_summary", pd.DataFrame())
    if summary.empty or "stability(%)" not in summary.columns:
        return
    display = summary.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"峰{int(value)}" for value in display["cluster"]],
            y=display["stability(%)"],
            name="重采样稳定性",
            marker_color="#2a7f78",
            customdata=np.stack(
                [
                    display["Q_center(1e-19C)"].round(3).astype(str),
                    display["points"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{x}<br>稳定性=%{y:.1f}%<br>"
                "Q中心=%{customdata[0]}<br>"
                "半峰宽内点数=%{customdata[1]}<extra></extra>"),
        ))
    fig.update_layout(
        title="q 峰的重采样稳定性",
        xaxis_title="发现的 q 峰",
        yaxis_title="bootstrap 中被再次发现的比例/%",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="peak_stability_plot_v1",
                    use_container_width=True)


def _plot_two_stage_discovery(result: dict):
    two_stage = result.get("two_stage", {})
    if not two_stage:
        return

    time_candidates = two_stage.get("time_stage_candidates", pd.DataFrame())
    coefficient_candidates = two_stage.get("coefficient_stage_candidates",
                                           pd.DataFrame())
    coefficient_df = two_stage.get("cluster_coefficients", pd.DataFrame())
    if not isinstance(time_candidates, pd.DataFrame) or time_candidates.empty:
        return

    st.subheader("两阶段符号发现的中间结果")
    st.caption(
        "第一阶段让每个 q 峰有独立系数，只共同搜索 t 的幂指数；第二阶段再让这些系数作为样本，搜索它们与 Q_c 的幂关系。")

    fig = go.Figure()
    ordered = time_candidates.sort_values("time_power")
    fig.add_trace(
        go.Scatter(
            x=ordered["time_power"],
            y=ordered["rmse"],
            mode="lines+markers",
            name="共同 t 指数候选",
            line=dict(color="#1f5d78", width=2),
            marker=dict(size=7),
            customdata=np.stack(
                [
                    ordered["r2"].round(4).astype(str),
                    ordered["bic"].round(1).astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "β=%{x:.3f}<br>RMSE=%{y:.3f} V<br>"
                "R²=%{customdata[0]}<br>BIC=%{customdata[1]}<extra></extra>"),
        ))
    selected_time = (
        time_candidates[time_candidates["selected"]]
        if "selected" in time_candidates.columns else pd.DataFrame())
    best_time = (
        selected_time.iloc[0] if not selected_time.empty
        else time_candidates.iloc[0])
    fig.add_vline(
        x=float(best_time["time_power"]),
        line_width=2,
        line_dash="dash",
        line_color="#d45a38",
        annotation_text=f"最佳 β={best_time['time_power']:.2f}",
    )
    fig.update_layout(
        title="阶段一：多条 q 峰曲线共同搜索时间幂指数",
        xaxis_title="候选 β",
        yaxis_title="RMSE / V",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="time_power_stage_plot_v1",
                    use_container_width=True)

    if not isinstance(coefficient_candidates,
                      pd.DataFrame) or coefficient_candidates.empty:
        return
    if not isinstance(coefficient_df, pd.DataFrame) or coefficient_df.empty:
        return

    selected_coeff = (
        coefficient_candidates[coefficient_candidates["selected"]]
        if "selected" in coefficient_candidates.columns else pd.DataFrame())
    coeff_best = (
        selected_coeff.iloc[0] if not selected_coeff.empty
        else coefficient_candidates.iloc[0])
    q_min = float(coefficient_df["Q_center(1e-19C)"].min())
    q_max = float(coefficient_df["Q_center(1e-19C)"].max())
    q_line = np.linspace(q_min, q_max, 200)
    y_line = float(coeff_best["amplitude"]) * np.power(
        q_line, float(coeff_best["charge_power"]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coefficient_df["Q_center(1e-19C)"],
            y=coefficient_df["time_coefficient"],
            mode="markers+text",
            text=[f"峰{int(value)}" for value in coefficient_df["cluster"]],
            textposition="top center",
            name="阶段一得到的各峰系数",
            marker=dict(size=13, color="#d45a38"),
        ))
    fig.add_trace(
        go.Scatter(
            x=q_line,
            y=y_line,
            mode="lines",
            name=f"系数 ≈ A·Q_c^{coeff_best['charge_power']:.2f}",
            line=dict(width=3, color="#234f6c"),
        ))
    fig.update_layout(
        title="阶段二：从各峰系数发现 Q_c 幂关系",
        xaxis_title="峰中心 Q_c / 1e-19 C",
        yaxis_title="阶段一曲线系数",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="charge_coefficient_stage_plot_v1",
                    use_container_width=True)


def _plot_symbolic_candidates(result: dict):
    candidates = result.get("candidate_models", pd.DataFrame())
    if candidates.empty:
        return

    display = candidates.head(8).copy()
    display.insert(0, "rank", np.arange(1, len(display) + 1))
    display["label"] = display.apply(
        lambda row: (
            f"候选{int(row['rank'])} | {row['family']} | "
            f"α={row['charge_power']:.2f}, β={row['time_power']:.2f}"),
        axis=1,
    )
    display["metric_label"] = display.apply(
        lambda row: f"RMSE {row['rmse']:.2f} V · R² {row['r2']:.3f}",
        axis=1,
    )

    st.subheader("全局候选式搜索对照")
    st.caption("这些候选式直接在所有高置信点上搜索，作为两阶段发现结果的对照。RMSE 越低越好；BIC 会同时惩罚误差和公式复杂度。")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=display["rmse"],
            y=display["label"],
            mode="markers+text",
            text=display["metric_label"],
            textposition="middle right",
            marker=dict(
                size=14,
                color=display["bic"],
                colorscale="Teal",
                showscale=True,
                colorbar=dict(title="BIC"),
            ),
            customdata=np.stack(
                [
                    display["mae"].round(3).astype(str),
                    display["bic"].round(1).astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{y}<br>RMSE=%{x:.3f} V<br>"
                "MAE=%{customdata[0]} V<br>"
                "BIC=%{customdata[1]}<extra></extra>"),
        ))
    fig.update_layout(
        title="候选符号表达式误差比较",
        xaxis_title="RMSE / V",
        yaxis_title="候选式结构",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=260, r=140, t=60, b=60),
        height=max(420, 62 * len(display)),
    )
    st.plotly_chart(fig, key="symbolic_candidate_plot_readable_v2",
                    use_container_width=True)

    st.markdown("**候选公式（数学排版）**")
    for _, row in display.iterrows():
        with st.container(border=True):
            st.markdown(
                f"候选 {int(row['rank'])} · `{row['family']}` · "
                f"RMSE `{row['rmse']:.3f} V` · R² `{row['r2']:.4f}` · "
                f"BIC `{row['bic']:.1f}`")
            st.latex(rf"U(t,Q_c) = {row['latex']}")

    cols = [
        "rank",
        "family",
        "charge_power",
        "time_power",
        "rmse",
        "mae",
        "r2",
        "bic",
    ]
    st.dataframe(display[cols], use_container_width=True, hide_index=True)


def render_tab_regress():
    st.header("机器学习—符号回归")
    st.success(
        "本页使用前一阶段发现的 Q 簇和半峰宽筛选结果，进行多簇联合符号回归；传统物理拟合作为对照保留。")

    has_user_data = not st.session_state.data.empty
    if not has_user_data:
        st.info("当前还没有录入实验数据。可以勾选参考数据先查看完整发现流程。")

    clustering_result = st.session_state.get("charge_clustering_result")
    available_clusters = []
    if clustering_result and not clustering_result["peak_summary"].empty:
        available_clusters = [
            int(value) for value in
            clustering_result["peak_summary"]["cluster"].dropna().tolist()
        ]

    with st.form("discovery_regression_form", border=True):
        st.subheader("实验参数、AI 聚类参数与符号回归参数")
        col1, col2, col3 = st.columns(3)
        with col1:
            fall_distance_mm = st.number_input("计时距离/mm",
                                               min_value=0.10,
                                               max_value=5.00,
                                               value=1.45,
                                               step=0.05)
        with col2:
            plate_distance_mm = st.number_input("极板间距/mm",
                                                min_value=1.00,
                                                max_value=10.00,
                                                value=5.00,
                                                step=0.10)
        with col3:
            include_reference = st.checkbox("参考数据参与分析", value=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            clustering_method = st.selectbox(
                "机器学习聚类方法",
                ["K-Means", "Gaussian Mixture", "DBSCAN", "KDE 峰发现"],
                index=0,
            )
        with col2:
            requested_clusters = st.slider("预期簇数", 2, 8, 5, 1)
        with col3:
            half_width_abs = st.slider(
                "半峰宽容差 / x10^-19C",
                0.05,
                0.80,
                0.25,
                0.01,
                help="先聚类，再以该半峰宽筛选簇内偏离点；不会在聚类前删除数据。")

        selected_clusters = None
        if available_clusters:
            selected_clusters = st.multiselect(
                "参与符号回归的簇",
                options=available_clusters,
                default=available_clusters[:6],
                help="可手动取消明显不稳定或教学上暂不讨论的 Q 簇。")

        with st.expander("聚类与符号回归高级参数", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                kde_bandwidth = st.slider(
                    "KDE带宽",
                    0.04,
                    0.35,
                    0.08,
                    0.01,
                    help="控制 q 分布密度曲线的平滑程度。数值越小越容易分出细峰，数值越大越容易把相邻峰合并。")
                max_clusters = st.slider(
                    "最多发现峰数",
                    5,
                    10,
                    8,
                    1,
                    help="KDE 找到很多峰时，只保留显著性最高的前几个峰，避免噪声峰进入分析。")
            with col2:
                peak_prominence = st.slider(
                    "峰显著性阈值",
                    0.005,
                    0.100,
                    0.020,
                    0.005,
                    help="峰必须比周围密度高出一定比例才被保留。数值越大，保留的峰越少。")
                min_points = st.number_input("每个峰最少点数",
                                             min_value=2,
                                             max_value=30,
                                             value=4,
                                             help="半峰宽筛选后，每个 q 峰至少要有多少点，才进入共享公式拟合。")
            with col3:
                dbscan_eps = st.slider("DBSCAN eps", 0.05, 0.80, 0.25, 0.01)
                dbscan_min_samples = st.number_input("DBSCAN min_samples",
                                                     min_value=2,
                                                     max_value=20,
                                                     value=4)
                analysis_upper = st.slider("峰发现主体上分位/%", 70.0, 99.0,
                                           85.0, 1.0,
                                           help="只用 q 分布主体区间寻找峰，长尾点保留但不参与找峰。85 表示先在前 85% 的 q 值中找主体峰。")
                time_power_step = st.select_slider(
                    "符号搜索步长",
                    options=[0.10, 0.05, 0.025],
                    value=0.05,
                    help="搜索 t 的幂指数时的步长。步长越小越精细，但计算更慢。",
                )
            st.caption(
                "参数说明：KDE 带宽控制 q 分布平滑程度；峰显著性阈值控制噪声峰过滤；半峰宽容差控制哪些点算高置信点；主体上分位用于防止少数极端长尾点影响峰发现；符号搜索步长控制幂指数搜索精度。")

        submitted = st.form_submit_button("执行机器学习—符号回归",
                                          use_container_width=True)

    method_map = {
        "K-Means": "KMeans",
        "Gaussian Mixture": "GaussianMixture",
        "DBSCAN": "DBSCAN",
        "KDE 峰发现": "KDE",
    }
    config = DiscoveryRegressionConfig(
        fall_distance_mm=float(fall_distance_mm),
        plate_distance_mm=float(plate_distance_mm),
        clustering_method=method_map[clustering_method],
        requested_clusters=int(requested_clusters),
        half_width_1e19c=float(half_width_abs),
        selected_clusters=tuple(selected_clusters)
        if selected_clusters else None,
        dbscan_eps=float(dbscan_eps),
        dbscan_min_samples=int(dbscan_min_samples),
        kde_bandwidth=float(kde_bandwidth),
        peak_prominence=float(peak_prominence),
        max_clusters=int(max_clusters),
        min_points_per_cluster=int(min_points),
        analysis_upper_percentile=float(analysis_upper),
        time_power_step=float(time_power_step),
    )

    analysis_data = _build_analysis_data(include_reference)
    if not analysis_data.empty:
        _plot_measurement_to_charge(analysis_data, config)

    if submitted:
        if analysis_data.empty:
            st.error("没有可用于分析的数据。请录入实验数据或勾选参考数据。")
            return
        try:
            result = discovery_regression(analysis_data, config)
            st.session_state.regression_results = result
            st.session_state.data_discovery_clustered = result["clusters"]
            st.success("机器学习—符号回归完成。")
        except Exception as exc:
            st.error(f"机器学习—符号回归失败: {exc}")
            return

    if "regression_results" not in st.session_state:
        return

    result = st.session_state.regression_results
    if result.get("mode") != "discovery":
        st.warning("当前会话中保存的是旧版物理约束拟合结果，请重新执行 AI 发现式聚类与拟合。")
        return
    if result.get("result_version") != RESULT_VERSION:
        st.session_state.pop("regression_results", None)
        st.session_state.pop("data_discovery_clustered", None)
        st.info("AI 聚类与符号回归逻辑已更新，请重新点击“执行机器学习—符号回归”。")
        return

    _render_result_summary(result)
    _render_ai_workflow(result)
    _plot_charge_density(result)
    _plot_peak_stability(result)
    _plot_neural_teacher(result)
    _plot_discovered_curves(result)

    with st.expander("查看共同间距和传统符号搜索对照", expanded=False):
        _plot_spacing_discovery(result)
        _plot_two_stage_discovery(result)
        _plot_symbolic_candidates(result)

    _render_physics_comparison(analysis_data)

    if not result["peak_summary"].empty:
        st.subheader("发现的 q 峰与拟合质量")
        st.dataframe(result["peak_summary"], use_container_width=True)

    with st.expander("查看候选式数值指标", expanded=False):
        candidate_cols = [
            "family",
            "charge_power",
            "time_power",
            "b",
            "coef1",
            "coef2",
            "rmse",
            "mae",
            "r2",
            "bic",
        ]
        available_candidate_cols = [
            col for col in candidate_cols
            if col in result["candidate_models"].columns
        ]
        st.dataframe(result["candidate_models"][available_candidate_cols],
                     use_container_width=True)

    with st.expander("查看 q 聚类明细", expanded=False):
        detail_cols = [
            SOURCE_COL,
            TIME_COL,
            VOLTAGE_COL,
            CHARGE_UNIT_COL,
            CHARGE_CLUSTER_COL,
            CHARGE_CENTER_COL,
            CHARGE_DISTANCE_COL,
            CHARGE_HALF_WIDTH_COL,
            "SymbolicResidual(V)",
            USE_FOR_FIT_COL,
            DISCOVERY_QUALITY_COL,
        ]
        available_cols = [
            col for col in detail_cols if col in result["clusters"].columns
        ]
        st.dataframe(result["clusters"][available_cols],
                     use_container_width=True)
