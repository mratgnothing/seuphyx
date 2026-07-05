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
    SOURCE_COL,
    TIME_COL,
    USE_FOR_FIT_COL,
    VOLTAGE_COL,
    add_charge_estimates,
    _candidate_expression,
    _predict_symbolic_candidate,
    discovery_regression,
)


RESULT_VERSION = "q-ai-clustering-symbolic-v10"
CHARGE_CLUSTERING_RESULT_VERSION = "q-ai-clustering-v9"


def _config_cache_key(config: DiscoveryRegressionConfig) -> tuple:
    return (
        float(config.fall_distance_mm),
        float(config.plate_distance_mm),
        float(config.oil_density_kg_m3),
        float(config.air_density_kg_m3),
        float(config.viscosity_pa_s),
        float(config.gravity_m_s2),
        float(config.charge_unit_c),
        float(config.kde_bandwidth),
        float(config.peak_prominence),
        str(config.clustering_method),
        config.requested_clusters,
        float(config.dbscan_eps),
        int(config.dbscan_min_samples),
        int(config.max_clusters),
        int(config.min_points_per_cluster),
        None if config.half_width_1e19c is None else float(
            config.half_width_1e19c),
        float(config.half_width_scale),
        config.selected_clusters,
        float(config.analysis_lower_percentile),
        float(config.analysis_upper_percentile),
        int(config.density_grid_size),
        int(config.stability_bootstrap_samples),
        int(config.min_symbolic_points),
        float(config.symbolic_pareto_tolerance_percent),
        str(config.symbolic_model_strategy),
        int(config.neural_teacher_models),
        float(config.time_power_min),
        float(config.time_power_max),
        float(config.time_power_step),
    )


def _config_from_key(key: tuple) -> DiscoveryRegressionConfig:
    return DiscoveryRegressionConfig(
        fall_distance_mm=key[0],
        plate_distance_mm=key[1],
        oil_density_kg_m3=key[2],
        air_density_kg_m3=key[3],
        viscosity_pa_s=key[4],
        gravity_m_s2=key[5],
        charge_unit_c=key[6],
        kde_bandwidth=key[7],
        peak_prominence=key[8],
        clustering_method=key[9],
        requested_clusters=key[10],
        dbscan_eps=key[11],
        dbscan_min_samples=key[12],
        max_clusters=key[13],
        min_points_per_cluster=key[14],
        half_width_1e19c=key[15],
        half_width_scale=key[16],
        selected_clusters=key[17],
        analysis_lower_percentile=key[18],
        analysis_upper_percentile=key[19],
        density_grid_size=key[20],
        stability_bootstrap_samples=key[21],
        min_symbolic_points=key[22],
        symbolic_pareto_tolerance_percent=key[23],
        symbolic_model_strategy=key[24],
        neural_teacher_models=key[25],
        time_power_min=key[26],
        time_power_max=key[27],
        time_power_step=key[28],
    )


@st.cache_data(show_spinner=False)
def _cached_charge_estimates(data: pd.DataFrame,
                             config_key: tuple) -> pd.DataFrame:
    return add_charge_estimates(data, _config_from_key(config_key))


@st.cache_data(show_spinner=False)
def _cached_discovery_regression(data: pd.DataFrame,
                                 config_key: tuple) -> dict:
    return discovery_regression(data, _config_from_key(config_key))


@st.cache_data(show_spinner=False)
def _read_reference_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).dropna(subset=[TIME_COL, VOLTAGE_COL])


def _load_reference_data() -> pd.DataFrame:
    root_reference = Path.cwd() / "oil_drop_reference.csv"
    reference_file = root_reference
    data_ref = _read_reference_csv(str(reference_file)).copy()
    data_ref[SOURCE_COL] = "根目录测试数据"
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
        charged = _cached_charge_estimates(data, _config_cache_key(config))
    except Exception:
        return

    with st.container(border=True):
        st.subheader("输入数据概览")
        st.caption("后续聚类只使用由 t、U 换算出的连续电荷估计 Q；这里先确认数据规模和量纲是否合理。")
        cols = st.columns(4)
        cols[0].metric("计时距离/mm", f"{config.fall_distance_mm:.2f}")
        cols[1].metric("极板间距/mm", f"{config.plate_distance_mm:.2f}")
        cols[2].metric("有效数据点", len(charged))
        cols[3].metric("q 中位数/1e-19C",
                       f"{charged[CHARGE_UNIT_COL].median():.3f}")


def _plot_charge_density(result: dict, key_suffix: str = ""):
    clusters = result["clusters"]
    q_values = clusters[CHARGE_UNIT_COL].dropna()
    if q_values.empty:
        return

    peaks = [
        peak for peak in result.get("peaks", [])
        if np.isfinite(float(peak.get("center", np.nan))) and
        pd.notna(peak.get("points", 0)) and
        int(peak.get("points", 0) or 0) > 0
    ]
    if peaks:
        left = min(peak["center"] - 2.2 * peak["half_width"] for peak in peaks)
        right = max(peak["center"] + 2.2 * peak["half_width"] for peak in peaks)
        plot_min = 0.0 if left < 1.0 else max(0.0, left)
        plot_max = max(10.0, right)
    else:
        plot_min, plot_max = np.percentile(q_values, [0.5, 85])
        plot_min = max(0.0, float(plot_min))
        plot_max = max(10.0, float(plot_max) * 1.12)
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
        mask = (grid >= plot_min) & (grid <= plot_max)
        fig.add_trace(
            go.Scatter(
                x=grid[mask],
                y=kde_values[mask],
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
    st.plotly_chart(fig, key=f"charge_density_plot_final_peaks{key_suffix}",
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


def _best_candidate_from_result(result: dict) -> dict:
    params = result["global_params"]
    coefficients = [float(params["b"]), float(params["coef1"])]
    coef2 = params.get("coef2", np.nan)
    if np.isfinite(coef2):
        coefficients.append(float(coef2))
    return {
        "family": params["family"],
        "charge_power": float(params["charge_power"]),
        "time_power": float(params["time_power"]),
        "coefficients": coefficients,
    }


def _cluster_curve_items(result: dict) -> dict:
    curve_items = {
        int(cluster_id): (t_line, y_line, expr)
        for cluster_id, (t_line, y_line, expr) in result.get("data", {}).items()
    }
    clustered = result["clusters"]
    fit_data = clustered[
        clustered[USE_FOR_FIT_COL] &
        clustered[CHARGE_CLUSTER_COL].notna()]
    if fit_data.empty:
        return curve_items

    candidate = _best_candidate_from_result(result)
    for cluster_id in sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique()):
        cluster_id = int(cluster_id)
        if cluster_id in curve_items:
            continue
        sub = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
        if sub.empty:
            continue
        q_center = float(sub[CHARGE_CENTER_COL].median())
        t_values = sub[TIME_COL].to_numpy(float)
        t_min, t_max = float(t_values.min()), float(t_values.max())
        if np.isclose(t_min, t_max):
            t_min = max(t_min * 0.95, 1e-9)
            t_max = t_max * 1.05
        t_line = np.linspace(t_min, t_max, 180)
        y_line = _predict_symbolic_candidate(
            candidate,
            np.full_like(t_line, q_center, dtype=float),
            t_line,
        )
        expr = _candidate_expression(candidate, fixed_charge=q_center)
        curve_items[cluster_id] = (t_line, y_line, expr)
    return curve_items


def _plot_discovered_curves(result: dict):
    clustered = result["clusters"]
    fig = go.Figure()
    palette = px.colors.qualitative.Dark24

    fit_data = clustered[clustered[USE_FOR_FIT_COL]]
    curve_items = _cluster_curve_items(result)
    for idx, cluster_id in enumerate(
            sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique())):
        sub = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
        if len(sub) > 320:
            sub = sub.sample(n=320, random_state=17)
        fig.add_trace(
            go.Scatter(
                x=sub[TIME_COL],
                y=sub[VOLTAGE_COL],
                mode="markers",
                name=f"峰{int(cluster_id)} 高置信点",
                marker=dict(size=6, color=palette[idx % len(palette)],
                            opacity=0.48),
            ))

    for idx, cluster_id in enumerate(sorted(curve_items)):
        t_line, y_line, _ = curve_items[cluster_id]
        fig.add_trace(
            go.Scatter(
                x=t_line,
                y=y_line,
                mode="lines",
                name=f"峰{cluster_id} 拟合曲线",
                line=dict(width=3, color=palette[idx % len(palette)]),
            ))

    y_values = []
    if not fit_data.empty:
        y_values.extend(fit_data[VOLTAGE_COL].dropna().to_list())
    for t_line, y_line, _ in curve_items.values():
        y_values.extend(np.asarray(y_line)[np.isfinite(y_line)].tolist())
    y_axis = {}
    if y_values:
        low, high = np.percentile(np.asarray(y_values, dtype=float), [1, 99])
        pad = max((high - low) * 0.12, 10)
        y_axis = dict(range=[low - pad, high + pad])

    fig.update_layout(
        title="最终可用结果：各电荷峰对应的 U-t 曲线",
        xaxis_title="下落时间 t / s",
        yaxis_title="平衡电压 U / V",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=60, r=30, t=85, b=60),
        height=520,
        yaxis=y_axis,
    )
    st.plotly_chart(fig, key="discovery_curve_plot_v2",
                    use_container_width=True)
    st.caption(
        "图中每条线已经代入对应的电荷峰中心，因此学生最终看到的是每个峰自己的 U(t) 曲线；散点只抽样显示高置信点以保持页面流畅。")
    if len(curve_items) < 2:
        st.warning("当前结果只有一个电荷峰满足拟合条件。请检查聚类峰数、半峰宽或最少点数设置。")


def _render_result_summary(result: dict):
    params = result["global_params"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("发现 q 峰数", params["cluster_count"])
    col2.metric("共同间距/1e-19C", f"{params['spacing_1e19C']:.4f}")
    col3.metric("整体 R²", f"{params['formula_r2']:.4f}")
    col4.metric("参与拟合点数", params["fit_points"])

    method_names = {
        "neural_teacher_distillation": "神经网络蒸馏",
        "two_stage": "两阶段发现",
        "global_candidate_search": "全局候选搜索",
    }
    method_label = method_names.get(params.get("discovery_method"), "")
    col1, col2, col3 = st.columns(3)
    col1.metric("t 幂指数", f"{params['time_power']:.2f}")
    col2.metric("RMSE/V", f"{params['formula_rmse']:.3f}")
    col3.metric("公式来源", method_label)

    curve_items = _cluster_curve_items(result)
    st.subheader("最终公式：已代入每个电荷峰中心")
    st.caption(
        "这里展示的是学生可直接使用的每条峰曲线。中间共享模型里的 Qc 已经被对应峰中心代入，所以最终曲线只剩下 t。")
    for cluster_id, (_, _, expr) in sorted(curve_items.items()):
        with st.container(border=True):
            st.markdown(f"**峰 {cluster_id}**")
            st.latex(rf"U_{cluster_id}(t) = {sp.latex(expr)}")

    with st.expander("为什么中间模型里会出现 Qc？", expanded=False):
        st.markdown(
            """
            `Q_c` 不是最后要求学生求解的未知量。它表示 AI 从 Q 分布中发现的某个电荷峰中心。

            符号回归先学习一个共享关系 `U(t, Q_c)`，这样多条曲线能共用同一套规律；
            展示最终结果时，系统会把每个峰自己的 `Q_c` 数值代入，得到上面的 `U_i(t)`。
            """)
        st.latex(rf"U(t,Q_c) = {sp.latex(result['symbolic_expression'])}")


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
        st.info("神经网络 teacher 未启用或训练失败，本次结果回退到候选式符号发现。")
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


SYMBOLIC_MODEL_OPTIONS = {
    "自动比较三类模型": "auto",
    "全局候选式搜索": "global_candidate_search",
    "两阶段符号发现": "two_stage",
    "神经网络 teacher 蒸馏": "neural_teacher_distillation",
}


def _render_global_search_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：先准备一组可能的表达式族，例如单幂律、加性幂律、时间修正项等，再在一组候选幂指数上逐个拟合并比较误差。

        **本实验怎么理解**：系统不会直接写死某个幂指数，而是在 `time_power_min` 到 `time_power_max` 的范围内搜索，让数据决定哪一个候选更好。

        **适合**：快速得到一批可解释候选式，并比较 RMSE、R²、BIC。

        **注意**：候选库仍然限定了搜索空间，所以它是“可控的符号搜索”，不是任意公式生成器。
        """)
    with col_plot:
        beta = np.array([-2.2, -1.8, -1.5, -1.2, -0.8])
        rmse = np.array([21, 13, 7.8, 10.5, 18])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=beta, y=rmse, mode="lines+markers",
                                 marker=dict(size=10, color="#2f6f7e"),
                                 line=dict(width=3, color="#2f6f7e"),
                                 name="候选误差"))
        fig.add_vline(x=-1.5, line_dash="dash", line_color="#c23b22",
                      annotation_text="数据选中的低误差区域")
        fig.update_layout(title="全局搜索：比较一批候选幂指数",
                          xaxis_title="时间幂指数 β",
                          yaxis_title="RMSE / V",
                          height=270,
                          margin=dict(l=55, r=20, t=55, b=45))
        st.plotly_chart(fig, key="guide_symbolic_global",
                        use_container_width=True)


def _render_two_stage_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：先把每个 Q 峰看作一条独立曲线，寻找所有曲线共同的时间幂指数；再研究这些曲线系数如何随峰中心变化。

        **本实验怎么理解**：第一阶段回答“所有峰的 U-t 曲线有没有共同形状”；第二阶段回答“峰中心 Q_c 只是在改变曲线系数吗”。

        **适合**：需要向学生解释多条曲线如何共享同一个物理规律。

        **注意**：中间式会出现 `Q_c`，它只是“峰中心”的占位量；最终展示给学生时会代入每个峰中心，得到 `U_i(t)`。
        """)
    with col_plot:
        q = np.array([1.8, 3.4, 5.0, 6.5, 8.2])
        coef = 420 / q
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q, y=coef, mode="markers+text",
                                 text=[f"峰{i}" for i in range(1, 6)],
                                 textposition="top center",
                                 marker=dict(size=12, color="#c45a33"),
                                 name="各峰曲线系数"))
        x = np.linspace(q.min(), q.max(), 160)
        fig.add_trace(go.Scatter(x=x, y=420 / x, mode="lines",
                                 line=dict(width=3, color="#284b63"),
                                 name="系数-Qc 关系"))
        fig.update_layout(title="两阶段：先找共同 t 形状，再找系数-Qc 关系",
                          xaxis_title="峰中心 Q_c / 1e-19 C",
                          yaxis_title="曲线系数",
                          height=270,
                          margin=dict(l=55, r=20, t=55, b=45))
        st.plotly_chart(fig, key="guide_symbolic_two_stage",
                        use_container_width=True)


def _render_teacher_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：先用神经网络集成学习噪声数据中的平滑曲面，再把这个平滑 teacher 的行为压缩成简洁公式。

        **本实验怎么理解**：神经网络负责抗噪和平滑，符号回归负责把规律翻译成学生能读懂的数学表达式。

        **适合**：实验点有噪声、离群点已经筛出，但每条曲线仍有局部波动。

        **注意**：teacher 不是最终答案；最终仍然要经过符号公式蒸馏和误差检验。

        参考：AI Feynman 使用神经网络和物理启发技巧辅助符号回归；Hinton 等人的知识蒸馏思想是把复杂模型行为压缩到更简单模型中。
        """)
    with col_plot:
        t = np.linspace(12, 120, 180)
        noisy = 40 + 2600 * np.power(t, -1.5) + 8 * np.sin(t / 8)
        smooth = 40 + 2600 * np.power(t, -1.5)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[::8], y=noisy[::8], mode="markers",
                                 marker=dict(size=7, color="#9aa5b1"),
                                 name="高可信实验点"))
        fig.add_trace(go.Scatter(x=t, y=smooth, mode="lines",
                                 line=dict(width=3, color="#2f6f7e"),
                                 name="teacher 平滑曲线"))
        fig.update_layout(title="teacher 蒸馏：先平滑，再压缩成公式",
                          xaxis_title="t / s",
                          yaxis_title="U / V",
                          height=270,
                          margin=dict(l=55, r=20, t=55, b=45))
        st.plotly_chart(fig, key="guide_symbolic_teacher",
                        use_container_width=True)


def _render_symbolic_model_guide():
    with st.container(border=True):
        st.subheader("符号回归三类模型")
        st.caption(
            "符号回归不是直接套最终公式，而是在高可信点上比较不同的可解释模型路径。学生看完三类模型后，再执行回归。")
        tabs = st.tabs(["全局候选式搜索", "两阶段符号发现", "神经网络 teacher 蒸馏"])
        with tabs[0]:
            _render_global_search_guide()
        with tabs[1]:
            _render_two_stage_guide()
        with tabs[2]:
            _render_teacher_guide()
        st.info(
            "关于 t 的幂指数：本页默认在一个连续候选范围内搜索，不把 2/3 或 3/2 直接写成答案；如果数据支持，结果会自然靠近物理上合理的幂指数。")


def render_tab_regress():
    st.header("机器学习—符号回归")
    st.caption(
        "本页接在 AI 聚类之后：先确认高可信点，再选择符号回归模型，最后才展示最终拟合公式。")

    has_user_data = not st.session_state.data.empty
    if not has_user_data:
        st.info("当前还没有导入测量数据。本页默认使用根目录 oil_drop_reference.csv 作为测试数据继续符号回归流程。")

    clustering_result = st.session_state.get("charge_clustering_result")
    if (clustering_result and
            clustering_result.get("result_version")
            != CHARGE_CLUSTERING_RESULT_VERSION):
        clustering_result = None
        st.session_state.pop("charge_clustering_result", None)
        st.session_state.pop("data_discovery_clustered", None)
        st.info("AI 聚类逻辑已更新，符号回归将按当前参数重新发现 Q 峰。")
    available_clusters = []
    if clustering_result and not clustering_result["peak_summary"].empty:
        available_clusters = [
            int(value) for value in
            clustering_result["peak_summary"]["cluster"].dropna().tolist()
        ]

    if clustering_result:
        st.subheader("第一步：确认聚类筛出的高可信点")
        st.caption("半峰宽筛选只发生在聚类之后。色带内的点进入符号回归，色带外的长尾点保留在明细中但不参与拟合。")
        _plot_charge_density(clustering_result, key_suffix="_pre_regression")

    _render_symbolic_model_guide()

    with st.form("discovery_regression_form", border=True):
        st.subheader("第二步：选择模型并执行符号回归")
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
            include_reference = st.checkbox(
                "使用根目录测试数据",
                value=not has_user_data,
                help="没有测量数据时用 D:\\GitHub\\seuphyx\\oil_drop_reference.csv 继续回归；已有测量数据时可作为补充数据。")

        col1, col2, col3 = st.columns(3)
        with col1:
            clustering_method = st.selectbox(
                "机器学习聚类方法",
                ["K-Means", "Gaussian Mixture", "DBSCAN", "KDE 峰发现"],
                index=3,
            )
            with st.popover("方法说明"):
                st.markdown(
                    """
                    **K-Means**：把 Q 值分成预设数量的簇，适合峰数大致已知的快速演示。

                    **Gaussian Mixture**：把每个峰看成一个高斯分布，适合峰宽不同或轻微重叠的情况。

                    **DBSCAN**：按密度找簇，不需要预设簇数，但参数敏感，容易把长尾数据拆成很多小簇。

                    **KDE 峰发现**：先画出 Q 的连续密度，再找峰；当前默认使用它，因为它最贴近“从分布中发现电荷峰”的教学逻辑。
                    """)
        with col2:
            requested_clusters = st.slider(
                "预期/最多峰数",
                2,
                8,
                5,
                1,
                help="K-Means/GMM 使用该簇数；KDE 作为最多保留峰数；DBSCAN 不使用。")
        with col3:
            half_width_abs = st.slider(
                "半峰宽容差 / x10^-19C",
                0.05,
                0.80,
                0.25,
                0.01,
                help="先聚类，再以该半峰宽筛选簇内偏离点；不会在聚类前删除数据。")

        col1, col2 = st.columns([1, 2])
        with col1:
            symbolic_model_label = st.selectbox(
                "符号回归模型策略",
                list(SYMBOLIC_MODEL_OPTIONS.keys()),
                index=0,
                help="自动比较会优先采用可用的神经网络 teacher 蒸馏结果；也可以强制查看某一类模型的最终结果。")
        with col2:
            st.caption(
                "建议课堂演示先保持“自动比较三类模型”，再分别切换到三种模型观察结果差异。三种模型的原理见上方标签页。")

        selected_clusters = None
        if available_clusters:
            selected_clusters = st.multiselect(
                "参与符号回归的簇",
                options=available_clusters,
                default=available_clusters,
                key="regression_selected_clusters_v10",
                help="可手动取消明显不稳定或教学上暂不讨论的 Q 簇。")

        with st.expander("聚类与符号回归高级参数", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                kde_bandwidth = st.slider(
                    "KDE带宽",
                    0.04,
                    0.35,
                    0.08,
                    0.01,
                    help="控制 q 分布密度曲线的平滑程度。数值越小越容易分出细峰，数值越大越容易把相邻峰合并。")
            with col2:
                peak_prominence = st.slider(
                    "峰显著性阈值",
                    0.005,
                    0.100,
                    0.020,
                    0.005,
                    help="峰必须比周围密度高出一定比例才被保留。数值越大，保留的峰越少。")
            with col3:
                max_clusters = st.slider(
                    "最多发现峰数",
                    5,
                    10,
                    5,
                    1,
                    help="KDE 找到很多峰时，只保留显著性最高的前几个峰，避免噪声峰进入分析。")
            with col4:
                min_points = st.number_input("每个峰最少点数",
                                             min_value=2,
                                             max_value=30,
                                             value=4,
                                             help="半峰宽筛选后，每个 q 峰至少要有多少点，才进入共享公式拟合。")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                dbscan_eps = st.slider("DBSCAN eps", 0.05, 0.80, 0.25, 0.01)
            with col2:
                dbscan_min_samples = st.number_input("DBSCAN min_samples",
                                                     min_value=2,
                                                     max_value=20,
                                                     value=4)
            with col3:
                analysis_upper = st.slider("峰发现主体上分位/%", 70.0, 99.0,
                                           85.0, 1.0,
                                           help="只用 q 分布主体区间寻找峰，长尾点保留但不参与找峰。85 表示先在前 85% 的 q 值中找主体峰。")
            with col4:
                time_power_step = st.select_slider(
                    "符号搜索步长",
                    options=[0.10, 0.05, 0.025],
                    value=0.10,
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
    symbolic_strategy = SYMBOLIC_MODEL_OPTIONS[symbolic_model_label]
    teacher_models = (
        2 if symbolic_strategy in ("auto", "neural_teacher_distillation")
        else 0)
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
        density_grid_size=900,
        stability_bootstrap_samples=6,
        symbolic_model_strategy=symbolic_strategy,
        neural_teacher_models=teacher_models,
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
            result = _cached_discovery_regression(
                analysis_data,
                _config_cache_key(config),
            )
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

    st.subheader("第三步：查看 AI 处理过程")
    process_view = st.radio(
        "处理过程视图",
        ["高可信点筛选", "三类模型过程", "后验检验"],
        horizontal=True,
        label_visibility="collapsed",
        key="symbolic_process_view",
    )
    if process_view == "高可信点筛选":
        _plot_charge_density(result, key_suffix="_result_process")
        _plot_peak_stability(result)
    elif process_view == "三类模型过程":
        _render_ai_workflow(result)
        model_view = st.radio(
            "模型过程",
            ["全局候选式搜索", "两阶段符号发现", "神经网络 teacher 蒸馏"],
            horizontal=True,
            key="symbolic_model_process_view",
        )
        if model_view == "全局候选式搜索":
            _plot_symbolic_candidates(result)
        elif model_view == "两阶段符号发现":
            _plot_two_stage_discovery(result)
        else:
            _plot_neural_teacher(result)
    else:
        _plot_spacing_discovery(result)

    st.subheader("第四步：最终拟合结果")
    _render_result_summary(result)
    _plot_discovered_curves(result)

    if not result["peak_summary"].empty:
        st.markdown("**发现的 q 峰与拟合质量**")
        display_cols = [
            "cluster",
            "Q_center(1e-19C)",
            "half_width(1e-19C)",
            "points",
            "mae",
            "r2",
        ]
        display_cols = [
            col for col in display_cols if col in result["peak_summary"]
        ]
        st.dataframe(result["peak_summary"][display_cols],
                     use_container_width=True,
                     hide_index=True)

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
        if st.checkbox("显示完整明细表（较慢）", value=False,
                       key="show_regression_detail_table"):
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
