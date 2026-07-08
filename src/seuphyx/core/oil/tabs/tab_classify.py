"""
Tab 2: AI charge clustering teaching flow.
"""
# third-party
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# seuphyx
from seuphyx.core.oil.tabs.regression import (
    CHARGE_CENTER_COL,
    CHARGE_CLUSTER_COL,
    CHARGE_DISTANCE_COL,
    CHARGE_HALF_WIDTH_COL,
    CHARGE_UNIT_COL,
    DISCOVERY_QUALITY_COL,
    SOURCE_COL,
    TIME_COL,
    USE_FOR_FIT_COL,
    VOLTAGE_COL,
    DiscoveryRegressionConfig,
    add_charge_estimates,
    charge_clustering,
)


CLUSTER_METHODS = {
    "K-Means": "KMeans",
    "Gaussian Mixture": "GaussianMixture",
    "DBSCAN": "DBSCAN",
    "KDE 峰发现": "KDE",
}

CHARGE_CLUSTERING_RESULT_VERSION = "q-ai-clustering-v9"
CHARGE_UNIT_LABEL = "10⁻¹⁹ C"


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
def _cached_charge_clustering(data: pd.DataFrame, config_key: tuple) -> dict:
    return charge_clustering(data, _config_from_key(config_key))


def _build_measurement_data(include_reference: bool) -> pd.DataFrame:
    parts = []
    has_user_data = not st.session_state.data.empty
    if include_reference:
        data_ref = st.session_state.data_ref.copy()
        data_ref[SOURCE_COL] = "根目录测试数据"
        parts.append(data_ref)
    if has_user_data:
        data_user = st.session_state.data.copy()
        data_user[SOURCE_COL] = "学生实测数据"
        parts.append(data_user)
    if not parts:
        return pd.DataFrame(columns=[TIME_COL, VOLTAGE_COL, SOURCE_COL])
    return pd.concat(parts, ignore_index=True)


def _build_q_data(include_reference: bool,
                  config: DiscoveryRegressionConfig) -> pd.DataFrame:
    raw = _build_measurement_data(include_reference)
    if raw.empty:
        return raw
    return _cached_charge_estimates(raw, _config_cache_key(config))


def _source_label(frame: pd.DataFrame, default: str) -> str:
    if SOURCE_COL in frame and not frame.empty:
        values = frame[SOURCE_COL].dropna().unique()
        if len(values) == 1:
            return str(values[0])
    return default


def _display_count(value: int) -> str:
    return f"{value:,}"


def _q_display_window(q_values: pd.Series | np.ndarray,
                      upper_percentile: float = 85.0,
                      min_right: float = 10.0) -> tuple[float, float, int]:
    values = np.asarray(q_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, min_right, 0
    low = min(max(0.0, float(np.percentile(values, 0.5))), min_right)
    high = float(np.percentile(values, min(max(upper_percentile, 50.0), 99.0)))
    high = max(min_right, high * 1.12)
    visible = (values >= low) & (values <= high)
    return float(max(0.0, low)), float(high), int(len(values) - visible.sum())


def _plot_raw_measurements(raw: pd.DataFrame):
    if raw.empty:
        st.info("当前没有可用于显示的数据。")
        return

    with st.container(border=True):
        st.subheader("第一步：先看全部原始测量点")
        st.caption(
            "下落时间 t 和平衡电压 U 的实验直接记录")
        cols = st.columns(4)
        cols[0].metric("总点数", _display_count(len(raw)))
        cols[1].metric("t 中位数/s", f"{raw[TIME_COL].median():.2f}")
        cols[2].metric("U 中位数/V", f"{raw[VOLTAGE_COL].median():.0f}")
        cols[3].metric("数据来源", _source_label(raw, "混合数据"))

        fig = go.Figure()
        if SOURCE_COL in raw.columns:
            reference = raw[raw[SOURCE_COL] == "根目录测试数据"]
            student = raw[raw[SOURCE_COL] == "学生实测数据"]
            other = raw[~raw.index.isin(reference.index.union(student.index))]
        else:
            reference = raw
            student = raw.iloc[0:0]
            other = raw.iloc[0:0]

        if not reference.empty:
            fig.add_trace(
                go.Scattergl(
                    x=reference[TIME_COL],
                    y=reference[VOLTAGE_COL],
                    mode="markers",
                    name="内置基础数据",
                    marker=dict(size=4, color="#355f7d", opacity=0.26),
                    hovertemplate="t=%{x:.2f} s<br>U=%{y:.1f} V<extra></extra>",
                ))
        if not other.empty:
            fig.add_trace(
                go.Scattergl(
                    x=other[TIME_COL],
                    y=other[VOLTAGE_COL],
                    mode="markers",
                    name="其他数据",
                    marker=dict(size=5, color="#6b7280", opacity=0.45),
                    hovertemplate="t=%{x:.2f} s<br>U=%{y:.1f} V<extra></extra>",
                ))
        if not student.empty:
            fig.add_trace(
                go.Scatter(
                    x=student[TIME_COL],
                    y=student[VOLTAGE_COL],
                    mode="markers",
                    name="学生实测数据",
                    marker=dict(
                        size=11,
                        color="#e11d48",
                        opacity=0.96,
                        symbol="circle",
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    hovertemplate="学生实测<br>t=%{x:.2f} s<br>U=%{y:.1f} V<extra></extra>",
                ))
        fig.update_layout(
            title="原始 U-t 点云",
            xaxis_title="下落时间 t / s",
            yaxis_title="平衡电压 U / V",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=60, r=30, t=80, b=60),
        )
        st.plotly_chart(fig, key="raw_measurement_scatter_all_points",
                        use_container_width=True)


def _plot_q_estimation_overview(charged: pd.DataFrame):
    if charged.empty:
        st.info("当前没有可用于显示的数据。")
        return

    with st.container(border=True):
        st.subheader("第三步：由物理量估算连续电荷 Q")
        st.caption(
            "系统根据 t、U 和实验常量，为每个油滴计算一个连续的 Q 估计值。")
        plot_min, plot_max, hidden_count = _q_display_window(
            charged[CHARGE_UNIT_COL])
        cols = st.columns(4)
        cols[0].metric("有效点数", _display_count(len(charged)))
        cols[1].metric(f"Q 中位数/{CHARGE_UNIT_LABEL}",
                       f"{charged[CHARGE_UNIT_COL].median():.3f}")
        cols[2].metric(f"主体 Q 窗口/{CHARGE_UNIT_LABEL}",
                       f"{plot_min:.2f}-{plot_max:.2f}")
        cols[3].metric("t 范围/s",
                       f"{charged[TIME_COL].min():.2f}-{charged[TIME_COL].max():.2f}")

        visible = charged[
            (charged[CHARGE_UNIT_COL] >= plot_min) &
            (charged[CHARGE_UNIT_COL] <= plot_max)]
        col_left, col_right = st.columns([1.25, 1])
        with col_left:
            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=visible[CHARGE_UNIT_COL],
                    y=visible[VOLTAGE_COL],
                    mode="markers",
                    name="连续 Q 估计",
                    marker=dict(size=4, color="#4f6f52", opacity=0.45),
                    hovertemplate="Q=%{x:.3f}<br>U=%{y:.1f} V<extra></extra>",
                ))
            fig.update_layout(
                title="连续 Q-U 点云",
                xaxis_title=f"电荷估计 Q / {CHARGE_UNIT_LABEL}",
                yaxis_title="平衡电压 U / V",
                margin=dict(l=55, r=20, t=50, b=50),
                height=390,
            )
            fig.update_xaxes(range=[plot_min, plot_max], autorange=False)
            st.plotly_chart(fig, key="raw_q_density_plot",
                            use_container_width=True)
        with col_right:
            hist = go.Figure()
            hist.add_trace(
                go.Histogram(
                    x=visible[CHARGE_UNIT_COL],
                    name="连续 Q",
                    marker_color="#2d6f73",
                    opacity=0.86,
                    xbins=dict(size=0.12),
                ))
            hist.update_layout(
                title="Q 一维分布",
                xaxis_title=f"电荷估计 Q / {CHARGE_UNIT_LABEL}",
                yaxis_title="点数",
                bargap=0.03,
                margin=dict(l=55, r=20, t=50, b=50),
                height=390,
            )
            hist.update_xaxes(range=[plot_min, plot_max], autorange=False)
            st.plotly_chart(hist, key="raw_q_histogram_plot",
                            use_container_width=True)
        if hidden_count:
            st.caption(
                f"为避免少数极端长尾值压扁主体分布，图中默认显示 Q 主体区间；{hidden_count} 个长尾点未显示在该窗口内。")


def _normal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    return np.exp(-0.5 * ((x - mean) / sigma)**2) / (
        sigma * np.sqrt(2 * np.pi))


def _render_kmeans_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：先指定要找几个簇，然后反复执行两件事：把每个点分给最近的中心，再把中心移动到本簇点的平均位置。

        **本实验应用**：如果 Q 分布里有几个明显的峰，K-Means 会把 Q 轴切成几个靠近不同中心的区域。

        **适用范围**：峰数大致知道、峰之间分得比较开。

        **误差来源**：它偏好近似圆形、方差接近的簇；离群点或长尾可能把中心拉偏。

        参考：scikit-learn 聚类指南说明 K-Means 通过最小化簇内平方和来划分样本。
        """)
    with col_plot:
        x = np.array([1.3, 1.5, 1.7, 3.0, 3.2, 3.45, 5.0, 5.2, 5.45])
        y = np.array([0.10, 0.18, 0.08, 0.14, 0.24, 0.10, 0.15, 0.25, 0.12])
        centers = np.array([1.5, 3.22, 5.22])
        colors = ["#2f6f7e", "#b05a31", "#6153a5"]
        fig = go.Figure()
        for idx, center in enumerate(centers):
            mask = np.abs(x - center) < 0.55
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=10, color=colors[idx]), name=f"簇{idx+1}"))
            fig.add_trace(go.Scatter(
                x=[center], y=[0.0], mode="markers",
                marker=dict(size=16, color=colors[idx], symbol="x"),
                name=f"中心{idx+1}"))
        fig.update_layout(title="K-Means：点靠近哪个中心就归哪一簇",
                          xaxis_title=f"Q / {CHARGE_UNIT_LABEL}",
                          yaxis=dict(visible=False),
                          height=260, margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig, key="guide_kmeans_plot",
                        use_container_width=True)


def _render_gmm_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：把数据看成多个高斯分布混合而成，每个点不是硬性属于某一簇，而是对每个峰都有一个概率。

        **本实验应用**：一个电荷峰可以有中心和宽度；峰之间有重叠时，GMM 会用概率处理模糊边界。

        **适用范围**：峰形接近钟形、峰宽需要被估计、相邻峰有轻微重叠。

        **误差来源**：通常仍要给出成分数；样本太少或离群点太多时，协方差估计会不稳定。

        参考：scikit-learn 文档将 GMM 描述为有限个未知高斯分布的概率混合模型，并用 EM 算法拟合。
        """)
    with col_plot:
        x = np.linspace(0, 7, 260)
        components = [
            (1.4, 0.35, 0.34, "#2f6f7e"),
            (3.25, 0.45, 0.44, "#b05a31"),
            (5.25, 0.55, 0.28, "#6153a5"),
        ]
        fig = go.Figure()
        total = np.zeros_like(x)
        for idx, (mean, sigma, weight, color) in enumerate(components, start=1):
            y = weight * _normal_pdf(x, mean, sigma)
            total += y
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                     line=dict(width=3, color=color),
                                     name=f"高斯成分{idx}"))
        fig.add_trace(go.Scatter(x=x, y=total, mode="lines",
                                 line=dict(width=3, color="#24292f",
                                           dash="dash"),
                                 name="混合密度"))
        fig.update_layout(title="GMM：多个概率峰相加生成整体分布",
                          xaxis_title=f"Q / {CHARGE_UNIT_LABEL}",
                          yaxis_title="概率密度",
                          height=260, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, key="guide_gmm_plot", use_container_width=True)


def _render_dbscan_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：不先指定簇数，而是寻找“足够密”的区域。一个点在 eps 半径内邻居足够多，就能成为核心点，并把密度相连的点扩展成簇。

        **本实验应用**：Q 轴上点堆得很密的位置会成为电荷峰；孤立点会被标为噪声。

        **适用范围**：想自动识别噪声、离群点较多、不想预设簇数。

        **误差来源**：eps 和 min_samples 很关键；不同峰密度差异很大时，单一 eps 可能不够好。

        参考：DBSCAN 原论文提出用密度连接发现带噪声的空间簇；scikit-learn 也强调其按高密度区域分簇。
        """)
    with col_plot:
        x = np.array([1.0, 1.15, 1.25, 1.35, 2.9, 3.05, 3.2, 3.3, 5.7])
        y = np.array([0.10, 0.20, 0.12, 0.17, 0.12, 0.22, 0.17, 0.09, 0.16])
        labels = np.array([1, 1, 1, 1, 2, 2, 2, 2, -1])
        colors = {1: "#2f6f7e", 2: "#b05a31", -1: "#8a8f98"}
        fig = go.Figure()
        for label in [1, 2, -1]:
            mask = labels == label
            name = "噪声点" if label == -1 else f"密度簇{label}"
            fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers",
                                     marker=dict(size=12,
                                                 color=colors[label]),
                                     name=name))
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=0.85, x1=1.55, y0=-0.15, y1=0.55,
                      line=dict(color="#2f6f7e", dash="dot"))
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=2.75, x1=3.45, y0=-0.14, y1=0.56,
                      line=dict(color="#b05a31", dash="dot"))
        fig.update_layout(title="DBSCAN：密度半径内邻居足够多才成簇",
                          xaxis_title=f"Q / {CHARGE_UNIT_LABEL}",
                          yaxis=dict(visible=False),
                          height=260, margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig, key="guide_dbscan_plot",
                        use_container_width=True)


def _render_kde_guide():
    col_text, col_plot = st.columns([1.05, 1])
    with col_text:
        st.markdown("""
        **核心思想**：每个点都贡献一个小的平滑曲线，把所有小曲线叠加成连续密度曲线，再在密度曲线上找峰。

        **本实验应用**：电荷量如果真的趋向若干稳定取值，Q 的密度曲线会自然出现峰；峰中心就是后续回归使用的 Q 中心。

        **适用范围**：教学展示“从分布找峰”的物理意义，特别适合一维 Q 分布。

        **误差来源**：带宽太小会把噪声当成峰，带宽太大会把相邻峰抹平。

        参考：scikit-learn 密度估计指南说明 KDE 用每个点贡献核函数来形成平滑的非参数分布估计。
        """)
    with col_plot:
        q = np.array([1.2, 1.35, 1.55, 3.0, 3.15, 3.35, 5.0, 5.18, 5.42])
        x = np.linspace(0.2, 6.3, 300)
        density = np.zeros_like(x)
        for point in q:
            density += _normal_pdf(x, point, 0.18)
        density /= len(q)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q, y=np.zeros_like(q), mode="markers",
                                 marker=dict(size=8, color="#607d8b"),
                                 name="样本点"))
        fig.add_trace(go.Scatter(x=x, y=density, mode="lines",
                                 line=dict(width=3, color="#2f6f7e"),
                                 name="KDE 密度"))
        for peak in [1.37, 3.17, 5.2]:
            y = np.interp(peak, x, density)
            fig.add_trace(go.Scatter(x=[peak], y=[y], mode="markers",
                                     marker=dict(size=12, color="#c2362b",
                                                 symbol="triangle-up"),
                                     name="密度峰"))
        fig.update_layout(title="KDE：先估计连续密度，再找峰",
                          xaxis_title=f"Q / {CHARGE_UNIT_LABEL}",
                          yaxis_title="密度",
                          height=260, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, key="guide_kde_plot", use_container_width=True)


def _render_clustering_method_guide():
    with st.container(border=True):
        st.subheader("第二步：选择无监督聚类算法")
        st.caption(
            "四种方法都只依据 Q 分布执行聚类。")
        tabs = st.tabs(["K-Means", "Gaussian Mixture", "DBSCAN", "KDE 峰发现"])
        with tabs[0]:
            _render_kmeans_guide()
        with tabs[1]:
            _render_gmm_guide()
        with tabs[2]:
            _render_dbscan_guide()
        with tabs[3]:
            _render_kde_guide()


def _plot_clustered_q_result(result: dict):
    clustered = result["clusters"]
    palette = px.colors.qualitative.Dark24

    with st.container(border=True):
        st.subheader("第四步：AI 聚类结果")
        st.caption(
            "颜色表示聚类后发现的电荷峰；灰色表示离峰中心太远、暂不进入符号回归的点。")

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            scatter = go.Figure()
            for idx, cluster_id in enumerate(
                    sorted(clustered[CHARGE_CLUSTER_COL].dropna().unique())):
                sub = clustered[
                    (clustered[CHARGE_CLUSTER_COL] == cluster_id) &
                    clustered[USE_FOR_FIT_COL]]
                scatter.add_trace(
                    go.Scattergl(
                        x=sub[TIME_COL],
                        y=sub[VOLTAGE_COL],
                        mode="markers",
                        name=f"峰 {int(cluster_id)} 高可信点",
                        marker=dict(size=5,
                                    color=palette[idx % len(palette)],
                                    opacity=0.62),
                        customdata=np.stack(
                            [sub[CHARGE_UNIT_COL], sub[CHARGE_CENTER_COL]],
                            axis=-1) if not sub.empty else None,
                        hovertemplate=(
                            "t=%{x:.2f} s<br>U=%{y:.1f} V"
                            "<br>Q=%{customdata[0]:.3f}"
                            "<br>峰中心=%{customdata[1]:.3f}<extra></extra>"),
                    ))
            outliers = clustered[~clustered[USE_FOR_FIT_COL]]
            if not outliers.empty:
                scatter.add_trace(
                    go.Scattergl(
                        x=outliers[TIME_COL],
                        y=outliers[VOLTAGE_COL],
                        mode="markers",
                        name="低可信/半峰宽外",
                        marker=dict(size=4, color="rgba(120,120,120,0.38)"),
                        hovertemplate="t=%{x:.2f} s<br>U=%{y:.1f} V<extra></extra>",
                    ))
            scatter.update_layout(
                title="U-t 平面中的聚类颜色",
                xaxis_title="下落时间 t / s",
                yaxis_title="平衡电压 U / V",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                margin=dict(l=55, r=20, t=80, b=55),
                height=440,
            )
            st.plotly_chart(scatter, key="clustered_ut_scatter_plot",
                            use_container_width=True)

        with col_right:
            peak_values = [
                float(peak["center"]) for peak in result.get("peaks", [])
                if np.isfinite(float(peak.get("center", np.nan)))
                and pd.notna(peak.get("points", 0))
                and int(peak.get("points", 0) or 0) > 0
            ]
            if peak_values:
                left = min(peak_values) - 0.8
                right = max(peak_values) + 0.8
                plot_min = max(0.0, left)
                plot_max = max(10.0, right)
            else:
                plot_min, plot_max, _ = _q_display_window(
                    clustered[CHARGE_UNIT_COL])
            hist = go.Figure()
            for idx, cluster_id in enumerate(
                    sorted(clustered[CHARGE_CLUSTER_COL].dropna().unique())):
                sub = clustered[
                    (clustered[CHARGE_CLUSTER_COL] == cluster_id) &
                    clustered[USE_FOR_FIT_COL]]
                sub = sub[
                    (sub[CHARGE_UNIT_COL] >= plot_min) &
                    (sub[CHARGE_UNIT_COL] <= plot_max)]
                hist.add_trace(
                    go.Histogram(
                        x=sub[CHARGE_UNIT_COL],
                        name=f"峰 {int(cluster_id)}",
                        marker_color=palette[idx % len(palette)],
                        opacity=0.78,
                        xbins=dict(size=0.10),
                    ))
            outliers = clustered[~clustered[USE_FOR_FIT_COL]]
            outliers = outliers[
                (outliers[CHARGE_UNIT_COL] >= plot_min) &
                (outliers[CHARGE_UNIT_COL] <= plot_max)]
            if not outliers.empty:
                hist.add_trace(
                    go.Histogram(
                        x=outliers[CHARGE_UNIT_COL],
                        name="低可信点",
                        marker_color="rgba(130,130,130,0.55)",
                        opacity=0.45,
                        xbins=dict(size=0.10),
                    ))
            for idx, peak in enumerate(result.get("peaks", [])):
                color = palette[idx % len(palette)]
                center = float(peak["center"])
                half_width = float(peak["half_width"])
                hist.add_vrect(
                    x0=center - half_width,
                    x1=center + half_width,
                    fillcolor=color,
                    opacity=0.08,
                    line_width=0,
                )
                hist.add_vline(
                    x=center,
                    line_width=2,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"峰{int(peak['cluster'])}",
                    annotation_position="top",
                )
            hist.update_layout(
                title="Q 轴上的峰和半峰宽",
                xaxis_title=f"电荷估计 Q / {CHARGE_UNIT_LABEL}",
                yaxis_title="点数",
                barmode="overlay",
                bargap=0.03,
                showlegend=False,
                margin=dict(l=55, r=20, t=80, b=55),
                height=440,
            )
            hist.update_xaxes(range=[plot_min, plot_max], autorange=False)
            st.plotly_chart(hist, key="clustered_q_histogram_plot",
                            use_container_width=True)

        if not result["peak_summary"].empty:
            summary = result["peak_summary"].copy()
            keep = [
                "cluster",
                "Q_center(1e-19C)",
                "half_width(1e-19C)",
                "points",
                "stability(%)",
                "r2",
            ]
            keep = [col for col in keep if col in summary.columns]
            display = summary[keep].rename(
                columns={
                    "Q_center(1e-19C)": f"峰中心 Q/{CHARGE_UNIT_LABEL}",
                    "half_width(1e-19C)": f"半峰宽/{CHARGE_UNIT_LABEL}",
                })
            st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("查看完整聚类明细（数据较大默认不渲染）", expanded=False):
        if st.checkbox("显示完整明细表（数据较大默认不渲染）", value=False,
                       key="show_cluster_detail_table"):
            detail_cols = [
                SOURCE_COL,
                TIME_COL,
                VOLTAGE_COL,
                CHARGE_UNIT_COL,
                CHARGE_CLUSTER_COL,
                CHARGE_CENTER_COL,
                CHARGE_DISTANCE_COL,
                CHARGE_HALF_WIDTH_COL,
                USE_FOR_FIT_COL,
                DISCOVERY_QUALITY_COL,
            ]
            available_cols = [
                col for col in detail_cols if col in clustered.columns
            ]
            detail = clustered[available_cols].rename(
                columns={
                    CHARGE_UNIT_COL: f"电荷估计 Q/{CHARGE_UNIT_LABEL}",
                    CHARGE_CENTER_COL: f"峰中心/{CHARGE_UNIT_LABEL}",
                    CHARGE_DISTANCE_COL: f"距离峰中心/{CHARGE_UNIT_LABEL}",
                    CHARGE_HALF_WIDTH_COL: f"半峰宽/{CHARGE_UNIT_LABEL}",
                })
            st.dataframe(detail,
                         use_container_width=True,
                         hide_index=True)

def render_tab_classify():
    st.header("机器学习—聚类分析")
    st.caption(
        "先看原始 U-t 点云，再把每个油滴换算为连续电荷 Q，最后用无监督算法从 Q 分布里发现自然簇。")

    has_user_data = not st.session_state.data.empty
    if not has_user_data:
        st.info("当前还没有导入测量数据。本页默认使用内置数据作为测试数据继续流程。")

    include_reference = st.checkbox(
        "合并内置基础数据",
        value=True,
        disabled=True,
        key="classify_include_reference_v2",
        help="学生视觉自动测量和手动录入的点如图中红色点所示。",
    )

    if not has_user_data and not include_reference:
        st.caption("页面已就绪。请完成视觉测量或手动录入数据。")
        return

    raw_measurements = _build_measurement_data(include_reference)
    _plot_raw_measurements(raw_measurements)

    _render_clustering_method_guide()

    with st.container(border=True):
        st.subheader("聚类参数")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            method = st.selectbox(
                "聚类方法",
                list(CLUSTER_METHODS.keys()),
                index=3,
                help="上方标签页解释了每种方法的聚类原理和适用情形。",
            )
        with col2:
            cluster_count = st.slider(
                "预期/最多峰数",
                2,
                8,
                5,
                1,
                help="K-Means/GMM 使用该簇数；KDE 作为最多保留峰数；DBSCAN 不使用。")
        with col3:
            half_width = st.slider(
                f"半峰宽容差 / {CHARGE_UNIT_LABEL}",
                0.05,
                0.80,
                0.25,
                0.01,
                help="聚类完成后，偏离簇中心超过该容差的点会被标记为舍去。")
        with col4:
            min_points_per_cluster = st.number_input(
                "每峰最少点数",
                min_value=2,
                max_value=30,
                value=4,
                help="少于该点数的峰不会进入后续拟合。")

        with st.expander("DBSCAN 与实验参数", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                dbscan_eps = st.slider("DBSCAN eps", 0.05, 0.80, 0.25, 0.01)
            with col2:
                dbscan_min_samples = st.number_input("DBSCAN min_samples",
                                                     min_value=2,
                                                     max_value=20,
                                                     value=4)
            with col3:
                fall_distance_mm = st.number_input("计时距离/mm",
                                                   min_value=0.10,
                                                   max_value=5.00,
                                                   value=1.45,
                                                   step=0.05)
            with col4:
                plate_distance_mm = st.number_input("极板间距/mm",
                                                    min_value=1.00,
                                                    max_value=10.00,
                                                    value=5.00,
                                                    step=0.10)

    config = DiscoveryRegressionConfig(
        fall_distance_mm=float(fall_distance_mm),
        plate_distance_mm=float(plate_distance_mm),
        clustering_method=CLUSTER_METHODS[method],
        requested_clusters=int(cluster_count),
        half_width_1e19c=float(half_width),
        dbscan_eps=float(dbscan_eps),
        dbscan_min_samples=int(dbscan_min_samples),
        max_clusters=int(cluster_count),
        min_points_per_cluster=int(min_points_per_cluster),
        density_grid_size=900,
        stability_bootstrap_samples=6,
    )

    raw_data = _build_q_data(include_reference, config)
    if not raw_data.empty:
        _plot_q_estimation_overview(raw_data)

    submitted = st.button(
        "执行 AI 聚类生成结果",
        use_container_width=True,
        type="primary",
    )

    if submitted:
        if raw_data.empty:
            st.error("没有可用于聚类的数据。")
            return
        try:
            result = _cached_charge_clustering(raw_data,
                                               _config_cache_key(config))
        except Exception as exc:
            st.error(f"AI 聚类失败: {exc}")
            return
        st.session_state.charge_clustering_result = result
        st.session_state.data_discovery_clustered = result["clusters"]
        st.session_state.charge_clustering_method = method
        st.success("AI 聚类完成。可在“机器学习—符号回归”页选择参与回归的簇。")

    result = st.session_state.get("charge_clustering_result")
    if not result:
        return
    if result.get("result_version") != CHARGE_CLUSTERING_RESULT_VERSION:
        st.session_state.pop("charge_clustering_result", None)
        st.session_state.pop("data_discovery_clustered", None)
        st.info("AI 聚类逻辑已更新，请重新点击“执行 AI 聚类，生成彩色结果”。")
        return

    used_method = st.session_state.get("charge_clustering_method")
    if used_method:
        st.caption(f"当前展示的是上次执行的聚类结果，算法：{used_method}。修改参数后请重新点击“执行 AI 聚类”。")
    _plot_clustered_q_result(result)
