"""
Tab 1: 数据记录
"""
# third-party
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
# seuphyx
from seuphyx.core.oil.tabs.regression import (
    CHARGE_UNIT_COL,
    TIME_COL,
    VOLTAGE_COL,
    DiscoveryRegressionConfig,
    add_charge_estimates,
)


def render_tab_record():
    # user data file
    work_dir = st.session_state.work_dir
    oil_drop_csv = work_dir / "oil_drop.csv"

    if 'data_ref_record' not in st.session_state:
        st.session_state.data_ref_record = st.session_state.data_ref_empty

    # 记录数据表单
    with st.form(key="write_data", clear_on_submit=True):
        st.subheader("请你输入实验数据：")
        st.write("请你根据实验要求，输入下落时间 (t/s) 和 平衡电压 (U/V) 两个数据。")

        x_coords = [0.0, 0.0, 0.0, 0.0, 0.0]
        y_coords = [0.0, 0.0, 0.0, 0.0, 0.0]
        col1, col2, col3, col4, col5 = st.columns(5)
        for idx, col in enumerate((col1, col2, col3, col4, col5)):
            with col:
                st.write(f"**数据点 {idx+1}**")
                x_coords[idx] = st.number_input("下落时间 (t/s)",
                                                min_value=0.0,
                                                max_value=150.0,
                                                value=0.0,
                                                key=f"x_coord_{idx}")
                y_coords[idx] = st.number_input("平衡电压 (U/V)",
                                                min_value=0.0,
                                                max_value=400.0,
                                                value=0.0,
                                                key=f"y_coord_{idx}")

        submitted = st.form_submit_button("数据记录")
        if submitted:
            valid_coords = []
            for x_coord, y_coord in zip(x_coords, y_coords):
                if x_coord <= 0 or y_coord <= 0:
                    continue

                # 检查数据是否已存在（容差为1e-6）
                is_duplicate = False
                for _, row in st.session_state.data.iterrows():
                    if (abs(row['FallingTime(t/s)'] - x_coord) < 1e-6 and
                            abs(row['BalanceVoltage(U/V)'] - y_coord) < 1e-6):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    valid_coords.append((x_coord, y_coord))
                    st.sidebar.write(f"写入数据: ({x_coord}, {y_coord})")

            # 检查 oil_drop.csv 文件是否存在
            if not oil_drop_csv.exists():
                st.sidebar.warning(f"首次保存数据，已创建新文件 {oil_drop_csv} 。")
                with open(oil_drop_csv, "w") as file:
                    file.write("FallingTime(t/s),BalanceVoltage(U/V)\n")

            # 追加写入数据
            with open(oil_drop_csv, "a+") as file:
                for x_coord, y_coord in valid_coords:
                    file.write(f"{x_coord},{y_coord}\n")

            if len(valid_coords) != 0:
                st.session_state.data = pd.concat(
                    [
                        st.session_state.data,
                        pd.DataFrame(
                            valid_coords,
                            columns=[
                                "FallingTime(t/s)", "BalanceVoltage(U/V)"
                            ],
                        )
                    ],
                    ignore_index=True,
                )

            st.rerun()

    with st.container(border=True):
        st.subheader("实验数据散点图")
        st.caption("记录阶段只显示学生实测数据；参考数据不会在本页出现，避免提前暴露判断依据。")
        if st.session_state.data.empty:
            st.info("当前还没有有效实验数据。")
            return

        charged = add_charge_estimates(
            st.session_state.data,
            DiscoveryRegressionConfig(),
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=charged[CHARGE_UNIT_COL],
                y=charged[VOLTAGE_COL],
                mode="markers",
                name="学生实测数据",
                marker=dict(size=10, color="#d62728", opacity=0.88),
                customdata=charged[[TIME_COL]].to_numpy(),
                hovertemplate=(
                    "Q=%{x:.3f} x10^-19 C<br>"
                    "U=%{y:.2f} V<br>"
                    "t=%{customdata[0]:.3f} s<extra></extra>"),
            ))
        fig.update_layout(
            title="学生实测数据：Q-U 散点",
            xaxis_title="电荷量 Q / x10^-19 C",
            yaxis_title="平衡电压 U / V",
            margin=dict(l=60, r=30, t=60, b=60),
            showlegend=True,
        )
        st.plotly_chart(fig, key="student_q_scatter_plot",
                        use_container_width=True)
        st.dataframe(
            charged[[TIME_COL, VOLTAGE_COL, CHARGE_UNIT_COL]],
            use_container_width=True,
            hide_index=True,
        )
