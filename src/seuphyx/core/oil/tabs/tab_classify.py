"""
Tab 2: 数据分类
"""
# third-party
import streamlit as st
import pandas as pd
import numpy as np
import joblib
# seuphyx
from seuphyx.core.oil.utils import plotly_plot


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


def render_tab_classify():
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
        "预训练模型SVM": joblib.load(model_file),
    }
    work_dir = st.session_state.work_dir
    model_files = work_dir.glob("*.joblib")
    for mf in model_files:
        model_options[mf.stem] = joblib.load(mf)

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
