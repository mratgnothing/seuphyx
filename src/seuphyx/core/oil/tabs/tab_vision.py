"""
Tab 5: 视觉测量预留页面
"""

import streamlit as st

from seuphyx.core.oil.vision import (
    VisionMeasurementConfig,
    build_empty_measurement,
)


def render_tab_vision():
    st.header("视觉自动测量")
    st.info("当前页面已预留视频测量接口；油滴检测、点击跟踪和网格标定将在采集数据集后接入。")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            grid_size_mm = st.number_input("网格间距/mm",
                                           min_value=0.001,
                                           value=0.25,
                                           step=0.001,
                                           format="%.3f")
        with col2:
            frame_rate = st.number_input("帧率/fps",
                                         min_value=1.0,
                                         value=30.0,
                                         step=1.0)
        with col3:
            voltage_v = st.number_input("电压/V", value=0.0, step=1.0)
        with col4:
            measurement_distance_mm = st.number_input("测量距离/mm",
                                                      min_value=0.001,
                                                      value=0.25,
                                                      step=0.001,
                                                      format="%.3f")

        video_file = st.file_uploader("上传油滴运动视频",
                                      type=["mp4", "avi", "mov", "mkv"])

        config = VisionMeasurementConfig(
            grid_size_mm=grid_size_mm,
            frame_rate=frame_rate,
            voltage_v=voltage_v,
            measurement_distance_mm=measurement_distance_mm,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("开始跟踪", disabled=True, use_container_width=True)
        with col2:
            st.button("结束记录", disabled=True, use_container_width=True)
        with col3:
            if st.button("生成记录模板", use_container_width=True):
                st.session_state.vision_measurement_template = (
                    build_empty_measurement(config))

    if video_file is not None:
        st.caption(f"已选择视频文件：{video_file.name}")

    if "vision_measurement_template" in st.session_state:
        measurement = st.session_state.vision_measurement_template
        st.subheader("视觉测量记录模板")
        st.json({
            "start_time_s": measurement.start_time_s,
            "end_time_s": measurement.end_time_s,
            "displacement_mm": measurement.displacement_mm,
            "velocity_mm_s": measurement.velocity_mm_s,
            "equivalent_falling_time_s":
            measurement.equivalent_falling_time_s,
            "voltage_v": measurement.voltage_v,
        })
