"""
Tab 2: 数据分类
"""
# third-party
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
# seuphyx
from seuphyx.core.oil.utils import plotly_plot
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


@st.cache_resource(show_spinner=False)
def _load_joblib_model(path: str):
    return joblib.load(path)


def _load_demo_data(model):
    data_ref = st.session_state.data_ref.copy()
    labels = model.predict(data_ref.values)
    data_ref = pd.concat(
        [data_ref, pd.DataFrame({"Predicted": labels})],
        axis=1,
    )

    demo_parts = []
    for label in sorted(data_ref["Predicted"].unique()):
        if label == 6:
            continue
        sub = data_ref[data_ref["Predicted"] == label]
        demo_parts.append(
            sub.sample(
                n=min(20, len(sub)),
                random_state=42,
            ))

    demo_data = pd.concat(demo_parts, ignore_index=True)[[
        "FallingTime(t/s)", "BalanceVoltage(U/V)"
    ]]
    st.session_state.data = demo_data
    st.session_state.data_pred = pd.DataFrame()
    st.session_state.data_ref_pred = st.session_state.data_ref_pred_empty

    oil_drop_csv = st.session_state.work_dir / "oil_drop.csv"
    demo_data.to_csv(oil_drop_csv, index=False)
    return len(demo_data)


def render_traditional_classification():
    # 加载预训练模型
    data_dir = st.session_state.data_dir
    model_file = data_dir / "points_svm_pipeline.joblib"
    if not model_file.exists():
        st.error(f"未找到模型文件：{model_file}\n\n"
                 f"请联系任课老师获取支持。")
        st.stop()

    # 初始化
    if 'data_ref_pred' not in st.session_state:
        st.session_state.data_ref_pred = st.session_state.data_ref_pred_empty

    # 选择模型
    model_options = {
        "预训练模型SVM": _load_joblib_model(str(model_file)),
    }
    work_dir = st.session_state.work_dir
    model_files = work_dir.glob("*.joblib")
    for mf in model_files:
        model_options[mf.stem] = _load_joblib_model(str(mf))

    model_name = st.selectbox("请选择分类模型：",
                              options=list(model_options.keys()),
                              index=0)
    model = model_options[model_name]

    with st.container(border=True):
        st.subheader("测试数据")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("加载内置测试数据", use_container_width=True):
                count = _load_demo_data(model)
                st.success(f"已加载 {count} 个内置测试点。")
                st.rerun()
        with col2:
            if st.button("清空当前数据", use_container_width=True):
                st.session_state.data = pd.DataFrame(
                    columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])
                st.session_state.data_pred = pd.DataFrame()
                oil_drop_csv = st.session_state.work_dir / "oil_drop.csv"
                if oil_drop_csv.exists():
                    oil_drop_csv.unlink()
                st.rerun()
        with col3:
            st.caption("内置测试数据来自 `oil_drop_reference.csv`，按类别均衡抽样，用于快速验证分类和物理拟合流程。")

    if st.session_state.data.empty:
        st.info("当前还没有实验数据。请点击上方“加载内置测试数据”，或先在“数据记录”页手动录入。")
        return

    if not st.session_state.data.empty:
        # 预测分类
        xy_coords = st.session_state.data.values
        y_pred = model.predict(xy_coords)
        st.session_state.data_pred = pd.concat(
            [st.session_state.data,
             pd.DataFrame({"Predicted": y_pred})],
            axis=1,
        )

        if 'data_pred' in st.session_state:
            with st.container(border=True):
                # 显示结果
                y_pred_labels = np.unique(
                    st.session_state.data_pred['Predicted'])
                grouped_data = {}
                for label in y_pred_labels:
                    legend = f"类别{label}"
                    grouped_data[legend] = st.session_state.data_pred[
                        st.session_state.data_pred['Predicted'] == label][[
                            'FallingTime(t/s)', 'BalanceVoltage(U/V)'
                        ]].values

                y_pred_labels = np.unique(
                    st.session_state.data_ref_pred['Predicted'])
                grouped_data_ref = {}
                for label in y_pred_labels:
                    legend = f"参考：舍弃数据" if label == y_pred_labels[
                        -1] else f"参考：类别{label}"
                    grouped_data_ref[legend] = st.session_state.data_ref_pred[
                        st.session_state.data_ref_pred['Predicted'] == label][[
                            'FallingTime(t/s)', 'BalanceVoltage(U/V)'
                        ]].values

                # 绘制分类结果
                plotly_plot(
                    title=f"分类数据散点图（模型：{model_name}）",
                    grouped_data={
                        **grouped_data_ref,
                        **grouped_data,
                    },
                    key="classification_scatter_plot",
                    showlegend=True,
                )
                st.session_state.model = model

                if st.button("**显示/隐藏参考数据分类结果**"):
                    data_ref_pred = st.session_state.data_ref_pred_empty
                    if st.session_state.data_ref_pred.empty:
                        data_ref = st.session_state.data_ref
                        data_ref_pred = pd.concat(
                            [
                                data_ref,
                                pd.DataFrame({
                                    "Predicted":
                                    model.predict(data_ref.values)
                                })
                            ],
                            axis=1,
                        )

                    st.session_state.data_ref_pred = data_ref_pred
                    st.rerun()

        st.dataframe(st.session_state.data_pred)


def _build_q_data(include_reference: bool,
                  config: DiscoveryRegressionConfig) -> pd.DataFrame:
    parts = []
    if include_reference:
        data_ref = st.session_state.data_ref.copy()
        data_ref[SOURCE_COL] = "参考数据"
        parts.append(data_ref)
    if not st.session_state.data.empty:
        data_user = st.session_state.data.copy()
        data_user[SOURCE_COL] = "学生实测数据"
        parts.append(data_user)
    if not parts:
        return pd.DataFrame(columns=[TIME_COL, VOLTAGE_COL, SOURCE_COL])
    return add_charge_estimates(pd.concat(parts, ignore_index=True), config)


def _plot_raw_q_distribution(charged: pd.DataFrame):
    if charged.empty:
        st.info("当前没有可用于显示的数据。")
        return

    if SOURCE_COL in charged.columns:
        reference_mask = charged[SOURCE_COL] == "参考数据"
        reference = charged[reference_mask]
        student = charged[~reference_mask]
        if len(reference) > 800:
            reference = reference.sample(n=800, random_state=7)
    else:
        reference = pd.DataFrame()
        student = charged

    fig = go.Figure()
    if not reference.empty:
        fig.add_trace(
            go.Scatter(
                x=reference[CHARGE_UNIT_COL],
                y=reference[VOLTAGE_COL],
                mode="markers",
                name="参考数据背景",
                marker=dict(size=5, color="rgba(80,80,80,0.28)"),
                customdata=reference[[TIME_COL]].to_numpy(),
                hovertemplate=(
                    "参考 Q=%{x:.3f}<br>U=%{y:.2f} V<br>"
                    "t=%{customdata[0]:.3f} s<extra></extra>"),
            ))
    if not student.empty:
        fig.add_trace(
            go.Scatter(
                x=student[CHARGE_UNIT_COL],
                y=student[VOLTAGE_COL],
                mode="markers",
                name="学生实测数据",
                marker=dict(size=10, color="#d62728", line=dict(width=1,
                                                                 color="white")),
                customdata=student[[TIME_COL]].to_numpy(),
                hovertemplate=(
                    "学生 Q=%{x:.3f}<br>U=%{y:.2f} V<br>"
                    "t=%{customdata[0]:.3f} s<extra></extra>"),
            ))
    fig.update_layout(
        title="聚类前：Q-U 原始分布（未 AI 着色）",
        xaxis_title="电荷量 Q / x10^-19 C",
        yaxis_title="平衡电压 U / V",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="raw_q_distribution_plot",
                    use_container_width=True)

    hist = go.Figure()
    if not reference.empty:
        hist.add_trace(
            go.Histogram(
                x=reference[CHARGE_UNIT_COL],
                name="参考数据背景",
                opacity=0.42,
                marker_color="rgba(80,80,80,0.42)",
                xbins=dict(size=0.12),
            ))
    if not student.empty:
        hist.add_trace(
            go.Histogram(
                x=student[CHARGE_UNIT_COL],
                name="学生实测数据",
                opacity=0.82,
                marker_color="#d62728",
                xbins=dict(size=0.12),
            ))
    hist.update_layout(
        title="聚类前：Q 分布柱状图",
        xaxis_title="电荷量 Q / x10^-19 C",
        yaxis_title="计数",
        barmode="overlay",
        bargap=0.04,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(hist, key="raw_q_histogram_plot", use_container_width=True)


def _plot_clustered_q_result(result: dict):
    clustered = result["clusters"]
    palette = px.colors.qualitative.Dark24
    fig = go.Figure()

    reference = clustered[clustered[SOURCE_COL] == "参考数据"]
    student = clustered[clustered[SOURCE_COL] == "学生实测数据"]
    if len(reference) > 800:
        reference = reference.sample(n=800, random_state=7)
    if not reference.empty:
        fig.add_trace(
            go.Scatter(
                x=reference[CHARGE_UNIT_COL],
                y=reference[VOLTAGE_COL],
                mode="markers",
                name="参考数据背景",
                marker=dict(size=5, color="rgba(55,55,55,0.22)"),
                hovertemplate="参考 Q=%{x:.3f}<br>U=%{y:.2f} V<extra></extra>",
            ))

    fit_student = student[student[USE_FOR_FIT_COL]]
    for idx, cluster_id in enumerate(
            sorted(fit_student[CHARGE_CLUSTER_COL].dropna().unique())):
        sub = fit_student[fit_student[CHARGE_CLUSTER_COL] == cluster_id]
        fig.add_trace(
            go.Scatter(
                x=sub[CHARGE_UNIT_COL],
                y=sub[VOLTAGE_COL],
                mode="markers",
                name=f"AI簇 {int(cluster_id)}",
                marker=dict(size=11, color=palette[idx % len(palette)],
                            line=dict(width=1, color="white")),
                hovertemplate="Q=%{x:.3f}<br>U=%{y:.2f} V<extra></extra>",
            ))

    outliers = student[~student[USE_FOR_FIT_COL]]
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=outliers[CHARGE_UNIT_COL],
                y=outliers[VOLTAGE_COL],
                mode="markers",
                name="半宽外/舍去点",
                marker=dict(size=9, color="rgba(120,120,120,0.75)",
                            symbol="x"),
                hovertemplate="Q=%{x:.3f}<br>U=%{y:.2f} V<extra></extra>",
            ))

    for idx, peak in enumerate(result.get("peaks", [])):
        color = palette[idx % len(palette)]
        center = float(peak["center"])
        half_width = float(peak["half_width"])
        fig.add_vrect(
            x0=center - half_width,
            x1=center + half_width,
            fillcolor=color,
            opacity=0.10,
            line_width=0,
        )
        fig.add_vline(
            x=center,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=f"簇{int(peak['cluster'])}",
            annotation_position="top",
        )

    fig.update_layout(
        title="AI 聚类后：Q-U 分布与半峰宽筛选",
        xaxis_title="电荷量 Q / x10^-19 C",
        yaxis_title="平衡电压 U / V",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, key="clustered_q_distribution_plot",
                    use_container_width=True)

    hist = go.Figure()
    for idx, cluster_id in enumerate(
            sorted(clustered[CHARGE_CLUSTER_COL].dropna().unique())):
        sub = clustered[
            (clustered[CHARGE_CLUSTER_COL] == cluster_id) &
            clustered[USE_FOR_FIT_COL]]
        hist.add_trace(
            go.Histogram(
                x=sub[CHARGE_UNIT_COL],
                name=f"AI簇 {int(cluster_id)}",
                marker_color=palette[idx % len(palette)],
                opacity=0.76,
                xbins=dict(size=0.12),
            ))
    hist.update_layout(
        title="AI 聚类后：Q 分布柱状图",
        xaxis_title="电荷量 Q / x10^-19 C",
        yaxis_title="计数",
        barmode="overlay",
        bargap=0.04,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(hist, key="clustered_q_histogram_plot",
                    use_container_width=True)


def render_tab_classify():
    st.header("机器学习—聚类分析")
    st.caption(
        "本页从每个油滴的 U、t 计算连续电荷量 Q，再用无监督学习自动寻找 Q 分布中的簇；半峰宽容差只在聚类完成后用于标记舍去点。")

    if st.session_state.data.empty:
        st.info("当前还没有学生实测数据。可以先到“数据记录”页录入，或在“传统方法对照”中加载内置测试数据。")

    with st.form("ai_charge_clustering_form", border=True):
        st.subheader("AI 聚类参数")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            method = st.selectbox(
                "聚类方法",
                ["K-Means", "Gaussian Mixture", "DBSCAN", "KDE 峰发现"],
                index=0,
            )
        with col2:
            cluster_count = st.slider("预期簇数", 2, 8, 5, 1,
                                      help="K-Means/GMM 使用该簇数；DBSCAN 不使用。")
        with col3:
            half_width = st.slider(
                "半峰宽容差 / x10^-19C",
                0.05,
                0.80,
                0.25,
                0.01,
                help="聚类完成后，偏离簇中心超过该容差的点会被标记为舍去。")
        with col4:
            include_reference = st.checkbox(
                "显示灰色参考背景",
                value=False,
                help="参考数据有数千条，默认关闭以保证页面初次打开更快。")

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

        submitted = st.form_submit_button("执行 AI 聚类",
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
        clustering_method=method_map[method],
        requested_clusters=int(cluster_count),
        half_width_1e19c=float(half_width),
        dbscan_eps=float(dbscan_eps),
        dbscan_min_samples=int(dbscan_min_samples),
        min_points_per_cluster=2,
    )

    if st.session_state.data.empty and not submitted:
        st.caption("页面已就绪。录入实验数据后会显示 Q 分布；需要教学参考背景时再勾选上方选项。")
        return

    raw_data = _build_q_data(include_reference, config)
    if not st.session_state.data.empty:
        _plot_raw_q_distribution(raw_data)

    if submitted:
        if raw_data.empty:
            st.error("没有可用于聚类的数据。")
            return
        try:
            result = charge_clustering(raw_data, config)
        except Exception as exc:
            st.error(f"AI 聚类失败: {exc}")
            return
        st.session_state.charge_clustering_result = result
        st.session_state.data_discovery_clustered = result["clusters"]
        st.success("AI 聚类完成。可在“机器学习—符号回归”页选择参与回归的簇。")

    result = st.session_state.get("charge_clustering_result")
    if not result:
        return

    _plot_clustered_q_result(result)
    if not result["peak_summary"].empty:
        st.subheader("聚类中心与半峰宽筛选结果")
        st.dataframe(result["peak_summary"], use_container_width=True,
                     hide_index=True)

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
    with st.expander("查看聚类明细", expanded=False):
        available_cols = [
            col for col in detail_cols if col in result["clusters"].columns
        ]
        st.dataframe(result["clusters"][available_cols],
                     use_container_width=True,
                     hide_index=True)
