"""
Tab 4: 物理约束拟合
"""
# built-in
from pathlib import Path

# third-party
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

# seuphyx
from seuphyx.core.oil.tabs.regression import (
    PhysicsRegressionConfig,
    physics_guided_regression,
)
from seuphyx.core.oil.utils import plotly_plot
import seuphyx


TIME_COL = "FallingTime(t/s)"
VOLTAGE_COL = "BalanceVoltage(U/V)"
PREDICTED_COL = "Predicted"


def _plot_physics_clusters(result):
    clustered = result["clusters"]
    fig = go.Figure()

    fit_data = clustered[clustered["UseForFit"]]
    for n_value in sorted(fit_data["PhysicsN"].dropna().unique()):
        sub = fit_data[fit_data["PhysicsN"] == n_value]
        fig.add_trace(
            go.Scatter(
                x=sub[TIME_COL],
                y=sub[VOLTAGE_COL],
                mode="markers",
                name=f"n={int(n_value)} 高置信点",
                marker=dict(size=9),
            ))

    outliers = clustered[~clustered["UseForFit"]]
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=outliers[TIME_COL],
                y=outliers[VOLTAGE_COL],
                mode="markers",
                name="未参与拟合点",
                marker=dict(size=7, color="rgba(120,120,120,0.55)"),
            ))

    for n_value, (t_line, y_line, _) in result["data"].items():
        fig.add_trace(
            go.Scatter(
                x=t_line,
                y=y_line,
                mode="lines",
                name=f"物理拟合 n={n_value}",
                line=dict(width=3),
            ))

    fig.update_layout(
        title="整数 n 物理聚类与受约束拟合",
        xaxis_title="下落时间 (t/s)",
        yaxis_title="平衡电压 (U/V)",
        font=dict(family="DejaVu Serif", size=16),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="physics_regression_plot", use_container_width=True)


def _plot_integer_distribution(result):
    clustered = result["clusters"]
    n_float = clustered["PhysicsNFloat"].dropna()
    if n_float.empty:
        return

    max_n = result["global_params"]["max_n"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=n_float,
            nbinsx=max(20, max_n * 12),
            name="n 估计值分布",
        ))
    for n_value in range(1, max_n + 1):
        fig.add_vline(x=n_value, line_dash="dash", line_color="#333333")

    fig.update_layout(
        title="物理变换后的整数 n 峰值分布",
        xaxis_title="n_float = A / ((U-b) * t^(3/2))",
        yaxis_title="数据点数量",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="integer_n_distribution", use_container_width=True)


def render_tab_regress():
    if 'data_pred' not in st.session_state:
        st.info("请先在“数据分类”页面完成分类，再进行物理约束拟合。")
        return

    with st.container(border=True):
        data_dir = Path(seuphyx.__file__).parent / "data"
        reference_file = data_dir / "oil_drop_reference.csv"
        data_ref = pd.read_csv(reference_file)
        data = st.session_state.data

        data_user = data.copy()
        data_user["Source"] = "实验数据"
        data_ref = data_ref.copy()
        data_ref["Source"] = "参考数据"
        data_combined = pd.concat([data_user, data_ref], axis=0,
                                  ignore_index=True)

        labels = st.session_state.model.predict(
            data_combined[[TIME_COL, VOLTAGE_COL]].values)
        data_combined[PREDICTED_COL] = labels

        y_pred_labels = np.unique(labels)
        grouped_data = {}
        for label in y_pred_labels:
            legend = f"舍弃数据" if label == y_pred_labels[-1] else f"AI类别{label}"
            grouped_data[legend] = data_combined[
                data_combined[PREDICTED_COL] == label][[
                    TIME_COL, VOLTAGE_COL
                ]].values

        plotly_plot(
            title="机器学习初始分类散点图",
            grouped_data=grouped_data,
            key="classification_scatter_plot2",
            showlegend=True,
        )

        st.session_state.data_combined = data_combined

    st.subheader("物理约束聚类与拟合")
    with st.form("physics_regression_form", border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_n = st.number_input("最大 n", min_value=2, max_value=12, value=5)
        with col2:
            peak_width = st.slider("整数峰半宽", 0.05, 0.60, 0.25, 0.01)
        with col3:
            min_points = st.number_input("每条曲线最少点数",
                                         min_value=2,
                                         max_value=20,
                                         value=3)

        with st.expander("高级参数", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_a = st.number_input("初始 A", value=54402.3027)
            with col2:
                initial_b = st.number_input("初始 b", value=-7.5)
            with col3:
                residual_sigma_factor = st.slider("残差筛选强度", 1.0, 6.0,
                                                  3.0, 0.1)

            use_predicted_init = st.checkbox("用 AI 分类初始化 A 和 b",
                                             value=True)
            include_reference = st.checkbox("参考数据参与拟合", value=True)

        submitted = st.form_submit_button("执行物理约束聚类与拟合",
                                          use_container_width=True)

    if submitted:
        config = PhysicsRegressionConfig(
            max_n=int(max_n),
            peak_width=float(peak_width),
            min_points_per_peak=int(min_points),
            initial_a=float(initial_a),
            initial_b=float(initial_b),
            use_predicted_labels_for_init=bool(use_predicted_init),
            residual_sigma_factor=float(residual_sigma_factor),
        )
        data_for_fit = data_combined if include_reference else st.session_state.data_pred
        try:
            result = physics_guided_regression(data_for_fit, config)
            st.session_state.regression_results = result
            st.session_state.data_physics_clustered = result["clusters"]
        except Exception as exc:
            st.error(f"物理约束拟合失败: {exc}")

    if 'regression_results' not in st.session_state:
        return

    result = st.session_state.regression_results
    params = result["global_params"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("全局 A", f"{params['A']:.4f}")
    col2.metric("全局 b", f"{params['b']:.4f}")
    col3.metric("残差阈值/V", f"{params['residual_limit']:.2f}")
    col4.metric("参与拟合点数",
                int(result["clusters"]["UseForFit"].sum()))

    _plot_integer_distribution(result)
    _plot_physics_clusters(result)

    if not result["peak_summary"].empty:
        st.subheader("各整数 n 的拟合结果")
        st.dataframe(result["peak_summary"], use_container_width=True)

    st.subheader("拟合公式")
    for n_value, (_, _, fitted_expr) in result["data"].items():
        st.write(f"**n = {n_value}:** U(t) = {fitted_expr}")

    with st.expander("查看物理聚类明细", expanded=False):
        detail_cols = [
            TIME_COL,
            VOLTAGE_COL,
            PREDICTED_COL,
            "PhysicsNFloat",
            "PhysicsN",
            "PhysicsNDistance",
            "PhysicsResidual(V)",
            "UseForFit",
            "ClusterQuality",
        ]
        available_cols = [
            col for col in detail_cols if col in result["clusters"].columns
        ]
        st.dataframe(result["clusters"][available_cols],
                     use_container_width=True)
