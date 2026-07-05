"""
Tab 1: 数据记录
"""
# third-party
import streamlit as st
# seuphyx
from seuphyx.core.oil.utils import plotly_plot


def render_tab_record():
    if 'data_ref_record' not in st.session_state:
        st.session_state.data_ref_record = st.session_state.data_ref_empty

    # 已保存的数据记录
    if "data" in st.session_state and not st.session_state.data.empty:
        st.subheader("已保存的实验数据")
        df = st.session_state.data.copy()
        df_display = df.reset_index(drop=True)
        df_display.insert(0, "序号", range(1, len(df_display) + 1))
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info('暂无实验数据，请在"视觉测量"页启动测量录入。')

    # 绘图描述
    if "data" in st.session_state and not st.session_state.data.empty:
        with st.container(border=True):
            plotly_plot(
                title="实验数据散点图",
                grouped_data={
                    "参考数据": st.session_state.data_ref_record.values,
                    "实验数据": st.session_state.data.values,
                },
                key="scatter_plot",
                showlegend=True,
            )

            if st.button("**显示/隐藏参考数据**"):
                if st.session_state.data_ref_record.empty:
                    st.session_state.data_ref_record = st.session_state.data_ref
                else:
                    st.session_state.data_ref_record = st.session_state.data_ref_empty
                st.rerun()
