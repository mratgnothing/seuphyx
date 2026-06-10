"""
Tab 4: 打印报告
"""
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from seuphyx.core.oil.tabs.regression import (
    CHARGE_CENTER_COL,
    CHARGE_CLUSTER_COL,
    CHARGE_UNIT_COL,
    SOURCE_COL,
    TIME_COL,
    USE_FOR_FIT_COL,
    VOLTAGE_COL,
)


PREDICTED_COL = "Predicted"


def _get_dataframe_state(key):
    value = st.session_state.get(key)
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value
    return pd.DataFrame()


def _get_report_readiness():
    data = _get_dataframe_state("data")
    data_pred = _get_dataframe_state("data_pred")
    has_regression = "regression_results" in st.session_state
    clustered = _get_dataframe_state("data_discovery_clustered")
    work_dir = st.session_state.get("work_dir")
    has_existing_report = bool(work_dir and list(work_dir.glob("report_*.pdf")))

    items = [
        ("实验数据", not data.empty, f"{len(data)} 个数据点" if not data.empty else "尚未录入"),
        ("AI发现", has_regression, "已完成 q 峰发现与共享拟合" if has_regression else "尚未完成"),
        ("PDF 报告", has_existing_report, "已有报告" if has_existing_report else "尚未生成"),
    ]
    return {
        "items": items,
        "can_generate": has_regression and (not data.empty or not clustered.empty),
        "data_pred": data_pred,
        "clustered": clustered,
    }


def _draw_line(pdf, y_coordinate, text, font_name="Song", font_size=12):
    if y_coordinate < 72:
        pdf.showPage()
        pdf.setFont(font_name, font_size)
        y_coordinate = 720
    pdf.drawString(72, y_coordinate, text)
    return y_coordinate - 18


def render_tab_report():
    st.header("打印报告")

    readiness = _get_report_readiness()
    with st.container(border=True):
        st.subheader("报告生成检查")
        cols = st.columns(len(readiness["items"]))
        for col, (label, ok, detail) in zip(cols, readiness["items"]):
            col.metric(label, "通过" if ok else "未完成")
            col.caption(detail)

        if not readiness["can_generate"]:
            st.warning("请先完成“AI发现拟合”，再生成实验报告。")

    if st.button("生成实验报告",
                 disabled=not readiness["can_generate"],
                 use_container_width=True):
        # 获取保存路径
        work_dir = st.session_state.work_dir
        student = st.session_state.student
        pdf_file = work_dir / f"report_{student['id']}_{student['name']}.pdf"
        regression_results = st.session_state.regression_results
        clustered = regression_results["clusters"]
        data_pred = readiness["data_pred"]
        if data_pred.empty:
            data_pred = clustered.copy()

        # 注册中文字体
        font_path = st.session_state.data_dir / "chinese.msyh.ttf"
        pdfmetrics.registerFont(TTFont('Song', font_path))

        # 创建 PDF 画布
        pdf = canvas.Canvas(str(pdf_file), pagesize=letter)

        # 设置字体和大小
        pdf.setFont("Song", 24)
        center_x = letter[0] / 2 - pdf.stringWidth("密立根油滴数据处理报告", "Song", 24) / 2
        y_coordinate = 10 * 72
        pdf.drawString(center_x, y_coordinate, f"密立根油滴数据处理报告")

        pdf.setFont("Song", 12)
        y_coordinate -= 36
        pdf.drawString(72, y_coordinate, f"学生姓名: {student['name']}")
        y_coordinate -= 18
        pdf.drawString(72, y_coordinate, f"学生学号: {student['id']}")
        y_coordinate -= 18
        timestr = f"{datetime.now():%Y.%m.%d}"
        pdf.drawString(72, y_coordinate, f"实验日期: {timestr}")
        y_coordinate -= 18

        y_coordinate = _draw_line(pdf, y_coordinate, "报告生成状态:")
        y_coordinate = _draw_line(pdf, y_coordinate,
                                  f"    分析数据点数: {len(clustered)}")
        use_for_fit_count = int(
            regression_results["clusters"][USE_FOR_FIT_COL].sum())
        y_coordinate = _draw_line(pdf, y_coordinate,
                                  f"    半峰宽内参与拟合点数: {use_for_fit_count}")

        params = regression_results.get("global_params", {})
        if "spacing_1e19C" in params:
            y_coordinate = _draw_line(
                pdf, y_coordinate,
                f"    AI发现共同间距: {params['spacing_1e19C']} x 10^-19 C")

        fit_data = clustered[clustered[USE_FOR_FIT_COL]]
        grouped_data = {}
        for label in sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique()):
            grouped_data[f"q峰{int(label)}"] = fit_data[
                fit_data[CHARGE_CLUSTER_COL] == label][[
                    TIME_COL, VOLTAGE_COL
                ]].values

        with st.container(border=True):
            # 获取回归结果
            data = regression_results['data'].items()
            for label, (_, _, fitted_expr) in data:
                st.write(f"**q峰 {label} 拟合公式:**", fitted_expr)
            data = regression_results['data'].items()

            # 将每个类别和公式保存到 PDF
            y_coordinate = _draw_line(pdf, y_coordinate, "拟合结果:")
            if "symbolic_expression" in regression_results:
                y_coordinate = _draw_line(
                    pdf, y_coordinate,
                    f"共享表达式: {regression_results['symbolic_expression']}")
            for label, (_, _, fitted_expr) in data:
                y_coordinate = _draw_line(
                    pdf, y_coordinate,
                    f"q峰 {label} 拟合公式: {fitted_expr}")

            if not regression_results.get("peak_summary", pd.DataFrame()).empty:
                st.subheader("各 q 峰的拟合指标")
                st.dataframe(regression_results["peak_summary"],
                             use_container_width=True)

            st.sidebar.write(f"AI发现式拟合结果已保存到: {pdf_file}")

            st.subheader("**AI发现得到的共享结构公式**")
            fig = go.Figure(
                data=[
                    go.Scatter(x=data[:, 0],
                               y=data[:, 1],
                               mode="markers",
                               name=label,
                               showlegend=True)
                    for label, data in grouped_data.items()
                ] + [
                    go.Scatter(x=t_line,
                               y=y_line,
                               mode="lines",
                               name=f'共享拟合-q峰{label}')
                    for label, (t_line, y_line, _) in
                    regression_results['data'].items()
                ],
                layout=go.Layout(
                    xaxis=dict(title='下落时间 (t/s)'),
                    yaxis=dict(title='平衡电压 (U/V)'),
                    font=dict(family='DejaVu Serif', size=16),
                    margin=dict(l=60, r=30, t=30, b=60),
                    colorway=px.colors.qualitative.D3,
                ),
            )
            st.plotly_chart(fig, key="regression_plot_print")

            # 保存图像到文件
            image_file = work_dir / "regression_plot.png"
            try:
                fig.write_image(str(image_file))
                y_coordinate -= 18
                pdf.drawImage(str(image_file),
                              72,
                              y_coordinate - 300,
                              width=468,
                              height=300)
                y_coordinate -= 318
            except Exception as exc:
                st.warning(f"拟合图像导出失败，报告仍会保留文字和数据表：{exc}")

        with st.container(border=True):
            st.subheader("已保存的数据点：")
            display_cols = [
                SOURCE_COL,
                TIME_COL,
                VOLTAGE_COL,
                CHARGE_UNIT_COL,
                CHARGE_CLUSTER_COL,
                CHARGE_CENTER_COL,
                USE_FOR_FIT_COL,
            ]
            display_cols = [col for col in display_cols if col in data_pred]
            st.dataframe(
                data_pred[display_cols] if display_cols else data_pred,
                on_select="ignore",
                height=35 * len(data_pred) + 38,
            )

            data = data_pred
            pdf.setFont("Song", 12)
            y_coordinate = _draw_line(pdf, y_coordinate, "已保存的数据点:")
            y_coordinate = _draw_line(
                pdf, y_coordinate,
                "    FallingTime(t/s), BalanceVoltage(U/V), q/1e-19C, qPeak")
            for index, row in data.iterrows():
                charge_value = row.get(CHARGE_UNIT_COL, "")
                cluster_value = row.get(CHARGE_CLUSTER_COL, "")
                y_coordinate = _draw_line(
                    pdf, y_coordinate,
                    f" {row[TIME_COL]:>19}, {row[VOLTAGE_COL]:>19}, {charge_value:>12}, {cluster_value:>8}"
                )

            pdf.save()  # 保存 PDF 文件

        st.success("实验报告已生成。")

        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="下载实验报告 PDF",
                data=pdf_bytes,
                file_name=f"report_{student['id']}_{student['name']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
