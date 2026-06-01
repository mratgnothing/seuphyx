"""
Tab 4: 打印报告
"""
from datetime import datetime
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


def render_tab_report():
    if st.button("生成实验报告"):
        # 获取保存路径
        work_dir = st.session_state.work_dir
        student = st.session_state.student
        pdf_file = work_dir / f"report_{student['id']}_{student['name']}.pdf"

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

        y_pred_labels = np.unique(st.session_state.data_pred['Predicted'])
        grouped_data = {}
        for label in y_pred_labels:
            grouped_data[f"类别{label}"] = st.session_state.data_pred[
                st.session_state.data_pred['Predicted'] == label][[
                    'FallingTime(t/s)', 'BalanceVoltage(U/V)'
                ]].values

        with st.container(border=True):
            # 获取回归结果
            if 'regression_results' in st.session_state:
                data = st.session_state.regression_results['data'].items()
                for label, (_, _, fitted_expr) in data:
                    st.write(f"**类别 {label} 拟合公式:**", fitted_expr)
                data = st.session_state.regression_results['data'].items()

                # 将每个类别和公式保存到 PDF
                y_coordinate -= 18
                pdf.drawString(72, y_coordinate, f"拟合结果:")
                for label, (_, _, fitted_expr) in data:
                    # 写入类别和拟合公式
                    y_coordinate -= 18
                    pdf.drawString(72, y_coordinate,
                                   f"类别 {label} 拟合公式: {fitted_expr}")

                    # 如果公式较长，可能需要换行处理（此处可以根据需要调整）
                    if y_coordinate < 64:
                        pdf.showPage()
                        pdf.setFont("Song", 12)
                        y_coordinate = 720

                st.sidebar.write(f"物理拟合结果已保存到: {pdf_file}")

                st.subheader("**物理约束拟合得到的统一结构公式**")
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
                                   name=f'物理拟合-n={label}')
                        for label, (t_line, y_line, _) in
                        st.session_state['regression_results']['data'].items()
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
                work_dir = st.session_state.work_dir  # 获取保存路径
                image_file = work_dir / "regression_plot.png"  # 设置保存图像的文件路径
                fig.write_image(str(image_file))  # 保存图像为 PNG 文件

                y_coordinate -= 18
                pdf.drawImage(str(image_file),
                              72,
                              y_coordinate - 300,
                              width=468,
                              height=300)
                y_coordinate -= 318

        with st.container(border=True):
            st.subheader("已保存的数据点：")
            st.dataframe(
                st.session_state.data_pred,
                on_select="ignore",
                height=35 * len(st.session_state.data_pred) + 38,
            )

            data = st.session_state.data_pred  # 你的 DataFrame 数据
            pdf.setFont("Song", 12)
            y_coordinate -= 18
            pdf.drawString(72, y_coordinate, f"已保存的数据点:")
            y_coordinate -= 18
            pdf.drawString(
                72, y_coordinate,
                f"    FallingTime(t/s), BalanceVoltage(U/V),      PredictedLabel"
            )
            y_coordinate -= 18
            for index, row in data.iterrows():
                pdf.drawString(
                    72, y_coordinate,
                    f" {row['FallingTime(t/s)']:>19}, {row['BalanceVoltage(U/V)']:>19}, {row['Predicted']:>19}"
                )
                y_coordinate -= 18
                if y_coordinate < 100:
                    pdf.showPage()
                    pdf.setFont("Song", 12)
                    y_coordinate = 720

            pdf.save()  # 保存 PDF 文件

        st.markdown("**请你保存本页面内容，作为本次实验的报告提交。**")

        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="下载实验报告 PDF",
                data=pdf_bytes,
                file_name=f"report_{student['id']}_{student['name']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
