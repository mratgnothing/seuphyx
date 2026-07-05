"""
Tab 4: 打印报告
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from seuphyx.core.oil.tabs.regression import (
    CHARGE_CLUSTER_COL,
    TIME_COL,
    USE_FOR_FIT_COL,
    VOLTAGE_COL,
)


PLOT_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def _get_dataframe_state(key):
    value = st.session_state.get(key)
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value
    return pd.DataFrame()


def _get_regression_results():
    result = st.session_state.get("regression_results")
    if isinstance(result, dict) and isinstance(result.get("clusters"),
                                              pd.DataFrame):
        return result
    return None


def _get_report_readiness():
    data = _get_dataframe_state("data")
    regression_results = _get_regression_results()
    clustered = (
        regression_results["clusters"]
        if regression_results is not None else
        _get_dataframe_state("data_discovery_clustered"))
    work_dir = st.session_state.get("work_dir")
    has_existing_report = bool(work_dir and list(work_dir.glob("report_*.pdf")))
    data_count = len(data) if not data.empty else len(clustered)

    items = [
        ("分析数据", data_count > 0,
         f"{data_count} 个数据点" if data_count > 0 else "尚未导入"),
        ("AI聚类与拟合", regression_results is not None,
         "已完成" if regression_results is not None else "尚未完成"),
        ("PDF报告", has_existing_report,
         "已有报告" if has_existing_report else "尚未生成"),
    ]
    return {
        "items": items,
        "can_generate": regression_results is not None and data_count > 0,
        "regression_results": regression_results,
        "clustered": clustered,
        "data_count": data_count,
    }


def _as_finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _format_number(value, digits=3, suffix=""):
    number = _as_finite_float(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}{suffix}"


def _format_int(value):
    number = _as_finite_float(value)
    if not np.isfinite(number):
        return "-"
    return f"{int(round(number))}"


def _curve_items(result):
    items = []
    for raw_label, value in result.get("data", {}).items():
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            continue
        t_line, y_line, expression = value[:3]
        t_line = np.asarray(t_line, dtype=float)
        y_line = np.asarray(y_line, dtype=float)
        if t_line.size == 0 or y_line.size == 0:
            continue
        try:
            sort_label = int(raw_label)
            display_label = str(sort_label)
        except (TypeError, ValueError):
            sort_label = str(raw_label)
            display_label = str(raw_label)
        items.append((sort_label, display_label, t_line, y_line, expression))
    return sorted(items, key=lambda item: item[0])


def _fit_data(result):
    clustered = result.get("clusters", pd.DataFrame())
    if clustered.empty or USE_FOR_FIT_COL not in clustered:
        return pd.DataFrame()
    return clustered[clustered[USE_FOR_FIT_COL]].copy()


def _report_metrics(result):
    params = result.get("global_params", {})
    clustered = result.get("clusters", pd.DataFrame())
    fit_count = int(clustered[USE_FOR_FIT_COL].sum()) if (
        not clustered.empty and USE_FOR_FIT_COL in clustered) else 0
    return {
        "total_points": len(clustered),
        "fit_points": fit_count,
        "cluster_count": params.get("cluster_count",
                                    len(_curve_items(result))),
        "spacing": params.get("spacing_1e19C", np.nan),
        "r2": params.get("formula_r2", np.nan),
        "rmse": params.get("formula_rmse", np.nan),
    }


def _peak_summary_view(result):
    summary = result.get("peak_summary", pd.DataFrame())
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return pd.DataFrame()

    display_cols = [
        "cluster",
        "Q_center(1e-19C)",
        "half_width(1e-19C)",
        "points",
        "mae",
        "r2",
    ]
    display_cols = [col for col in display_cols if col in summary.columns]
    display = summary[display_cols].copy()
    rename_map = {
        "cluster": "q峰",
        "Q_center(1e-19C)": "峰中心 q/1e-19C",
        "half_width(1e-19C)": "半峰宽/1e-19C",
        "points": "高置信点数",
        "mae": "MAE/V",
        "r2": "R2",
    }
    display = display.rename(columns=rename_map)
    for col in ["峰中心 q/1e-19C", "半峰宽/1e-19C", "MAE/V", "R2"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col],
                                         errors="coerce").round(4)
    if "q峰" in display.columns:
        display["q峰"] = display["q峰"].astype("Int64")
    if "高置信点数" in display.columns:
        display["高置信点数"] = display["高置信点数"].astype("Int64")
    return display


def _build_report_figure(result):
    fig = go.Figure()
    fit_data = _fit_data(result)
    curve_items = _curve_items(result)

    cluster_values = []
    if not fit_data.empty and CHARGE_CLUSTER_COL in fit_data:
        cluster_values = sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique())

    for index, cluster_id in enumerate(cluster_values):
        sub = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
        if len(sub) > 260:
            sub = sub.sample(n=260, random_state=23)
        fig.add_trace(
            go.Scatter(
                x=sub[TIME_COL],
                y=sub[VOLTAGE_COL],
                mode="markers",
                name=f"峰 {int(cluster_id)} 高置信点",
                marker=dict(
                    size=6,
                    opacity=0.45,
                    color=PLOT_COLORS[index % len(PLOT_COLORS)],
                ),
            ))

    for index, (_, display_label, t_line, y_line, _) in enumerate(curve_items):
        fig.add_trace(
            go.Scatter(
                x=t_line,
                y=y_line,
                mode="lines",
                name=f"峰 {display_label} 拟合曲线",
                line=dict(width=3,
                          color=PLOT_COLORS[index % len(PLOT_COLORS)]),
            ))

    y_values = []
    if not fit_data.empty:
        y_values.extend(pd.to_numeric(fit_data[VOLTAGE_COL],
                                      errors="coerce").dropna().tolist())
    for _, _, _, y_line, _ in curve_items:
        finite_y = y_line[np.isfinite(y_line)]
        y_values.extend(finite_y.tolist())

    y_axis = {}
    if y_values:
        low, high = np.percentile(np.asarray(y_values, dtype=float), [1, 99])
        pad = max((high - low) * 0.12, 10)
        y_axis = dict(range=[low - pad, high + pad])

    fig.update_layout(
        title="最终结果图：各 q 峰对应的 U-t 拟合曲线",
        xaxis_title="下落时间 t / s",
        yaxis_title="平衡电压 U / V",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=60, r=30, t=85, b=60),
        height=520,
        colorway=px.colors.qualitative.D3,
        yaxis=y_axis,
    )
    return fig


def _hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _load_plot_font(font_path, size):
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default()


def _write_pillow_regression_plot(image_file, result, font_path):
    width, height = 1100, 700
    left, right, top, bottom = 90, 40, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    fit_data = _fit_data(result)
    curve_items = _curve_items(result)
    if fit_data.empty and not curve_items:
        return False

    x_values = []
    y_values = []
    if not fit_data.empty:
        x_values.extend(pd.to_numeric(fit_data[TIME_COL],
                                      errors="coerce").dropna().tolist())
        y_values.extend(pd.to_numeric(fit_data[VOLTAGE_COL],
                                      errors="coerce").dropna().tolist())
    for _, _, t_line, y_line, _ in curve_items:
        finite = np.isfinite(t_line) & np.isfinite(y_line)
        x_values.extend(t_line[finite].tolist())
        y_values.extend(y_line[finite].tolist())
    if not x_values or not y_values:
        return False

    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y_low = float(np.nanmin(y_values))
    y_high = float(np.nanmax(y_values))
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_low == y_high:
        y_low -= 1
        y_high += 1
    x_pad = (x_max - x_min) * 0.04
    y_pad = max((y_high - y_low) * 0.12, 10)
    x_min -= x_pad
    x_max += x_pad
    y_min = float(y_low - y_pad)
    y_max = float(y_high + y_pad)

    def to_px(x_value, y_value):
        x_px = left + (float(x_value) - x_min) / (x_max - x_min) * plot_w
        y_px = bottom + (y_max - float(y_value)) / (y_max - y_min) * plot_h
        return int(round(x_px)), int(round(y_px))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_plot_font(font_path, 28)
    label_font = _load_plot_font(font_path, 20)
    tick_font = _load_plot_font(font_path, 16)

    draw.text((left, 22), "Final U-t fitting curves",
              fill="#111827", font=title_font)
    for ratio in np.linspace(0, 1, 6):
        x = int(left + ratio * plot_w)
        draw.line((x, bottom, x, bottom + plot_h), fill="#e5e7eb", width=1)
        tick = x_min + ratio * (x_max - x_min)
        draw.text((x - 20, bottom + plot_h + 12), f"{tick:.1f}",
                  fill="#4b5563", font=tick_font)
    for ratio in np.linspace(0, 1, 6):
        y = int(bottom + ratio * plot_h)
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb", width=1)
        tick = y_max - ratio * (y_max - y_min)
        draw.text((18, y - 10), f"{tick:.0f}", fill="#4b5563",
                  font=tick_font)

    draw.line((left, bottom, left, bottom + plot_h), fill="#111827", width=2)
    draw.line((left, bottom + plot_h, left + plot_w, bottom + plot_h),
              fill="#111827", width=2)
    draw.text((left + plot_w // 2 - 45, height - 42), "t / s",
              fill="#111827", font=label_font)
    draw.text((12, top - 36), "U / V", fill="#111827", font=label_font)

    cluster_values = []
    if not fit_data.empty and CHARGE_CLUSTER_COL in fit_data:
        cluster_values = sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique())
    for index, cluster_id in enumerate(cluster_values):
        sub = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
        if len(sub) > 220:
            sub = sub.sample(n=220, random_state=31)
        color = _hex_to_rgb(PLOT_COLORS[index % len(PLOT_COLORS)])
        for _, row in sub.iterrows():
            x_value = _as_finite_float(row.get(TIME_COL))
            y_value = _as_finite_float(row.get(VOLTAGE_COL))
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            x_px, y_px = to_px(x_value, y_value)
            draw.ellipse((x_px - 3, y_px - 3, x_px + 3, y_px + 3),
                         fill=color)

    for index, (_, display_label, t_line, y_line, _) in enumerate(curve_items):
        color = _hex_to_rgb(PLOT_COLORS[index % len(PLOT_COLORS)])
        coords = []
        finite = np.isfinite(t_line) & np.isfinite(y_line)
        for x_value, y_value, ok in zip(t_line, y_line, finite):
            if ok:
                coords.append(to_px(x_value, y_value))
            elif len(coords) >= 2:
                draw.line(coords, fill=color, width=4)
                coords = []
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=4)

        legend_x = left + plot_w - 210
        legend_y = top + 10 + index * 28
        draw.line((legend_x, legend_y + 9, legend_x + 34, legend_y + 9),
                  fill=color, width=4)
        draw.text((legend_x + 42, legend_y), f"peak {display_label}",
                  fill="#111827", font=tick_font)

    image.save(image_file)
    return True


def _save_chart_image(fig, image_file, result, font_path):
    try:
        fig.write_image(str(image_file))
        return "plotly"
    except Exception:
        if _write_pillow_regression_plot(image_file, result, font_path):
            return "fallback"
    return "none"


def _ensure_pdf_font(font_path):
    try:
        pdfmetrics.registerFont(TTFont("Song", str(font_path)))
        return "Song"
    except Exception:
        return "Helvetica"


def _wrap_pdf_text(pdf, text, font_name, font_size, max_width):
    wrapped = []
    for raw_line in str(text).replace("\r", "").split("\n"):
        if not raw_line:
            wrapped.append("")
            continue
        current = ""
        for char in raw_line:
            trial = current + char
            if pdf.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                wrapped.append(current)
                current = char
        wrapped.append(current)
    return wrapped


def _ensure_pdf_space(pdf, y_coordinate, font_name, font_size, required=18):
    if y_coordinate - required < 72:
        pdf.showPage()
        pdf.setFont(font_name, font_size)
        return 720
    return y_coordinate


def _draw_pdf_text(pdf, y_coordinate, text, font_name="Song", font_size=12,
                   indent=0, max_width=468, line_gap=6):
    pdf.setFont(font_name, font_size)
    x_coordinate = 72 + indent
    width = max_width - indent
    for line in _wrap_pdf_text(pdf, text, font_name, font_size, width):
        y_coordinate = _ensure_pdf_space(pdf, y_coordinate, font_name,
                                         font_size)
        pdf.drawString(x_coordinate, y_coordinate, line)
        y_coordinate -= font_size + line_gap
    return y_coordinate


def _draw_pdf_section(pdf, y_coordinate, title, font_name):
    y_coordinate = _ensure_pdf_space(pdf, y_coordinate, font_name, 14, 36)
    pdf.setFont(font_name, 14)
    pdf.drawString(72, y_coordinate, title)
    return y_coordinate - 24


def _create_pdf_report(pdf_file, student, regression_results, font_path, fig):
    font_name = _ensure_pdf_font(font_path)
    pdf = canvas.Canvas(str(pdf_file), pagesize=letter)
    metrics = _report_metrics(regression_results)

    pdf.setFont(font_name, 24)
    title = "密立根油滴数据处理报告"
    center_x = letter[0] / 2 - pdf.stringWidth(title, font_name, 24) / 2
    y_coordinate = 720
    pdf.drawString(center_x, y_coordinate, title)

    y_coordinate -= 38
    y_coordinate = _draw_pdf_text(
        pdf, y_coordinate, f"学生姓名: {student.get('name', '-')}",
        font_name)
    y_coordinate = _draw_pdf_text(
        pdf, y_coordinate, f"学生学号: {student.get('id', '-')}",
        font_name)
    y_coordinate = _draw_pdf_text(
        pdf, y_coordinate, f"实验日期: {datetime.now():%Y.%m.%d}",
        font_name)

    y_coordinate -= 10
    y_coordinate = _draw_pdf_section(pdf, y_coordinate, "一、核心结论",
                                     font_name)
    y_coordinate = _draw_pdf_text(
        pdf, y_coordinate,
        f"本次共分析 {metrics['total_points']} 个数据点，其中 "
        f"{metrics['fit_points']} 个高置信点进入符号回归。AI 聚类发现 "
        f"{_format_int(metrics['cluster_count'])} 个 q 峰。",
        font_name)
    y_coordinate = _draw_pdf_text(
        pdf, y_coordinate,
        "共同电荷间距: "
        f"{_format_number(metrics['spacing'], 4)} x 10^-19 C；"
        f"整体 R2: {_format_number(metrics['r2'], 4)}；"
        f"RMSE: {_format_number(metrics['rmse'], 3)} V。",
        font_name)

    y_coordinate -= 8
    y_coordinate = _draw_pdf_section(pdf, y_coordinate, "二、最终可用公式",
                                     font_name)
    y_coordinate = _draw_pdf_text(
        pdf, y_coordinate,
        "下列公式已经代入各 q 峰中心，最终表达式只以 t 为自变量。",
        font_name)
    for _, display_label, _, _, expression in _curve_items(regression_results):
        y_coordinate = _draw_pdf_text(
            pdf, y_coordinate, f"峰 {display_label}: U(t) = {expression}",
            font_name, font_size=10, indent=12, line_gap=5)

    peak_summary = _peak_summary_view(regression_results)
    if not peak_summary.empty:
        y_coordinate -= 8
        y_coordinate = _draw_pdf_section(pdf, y_coordinate, "三、q峰摘要",
                                         font_name)
        for _, row in peak_summary.iterrows():
            y_coordinate = _draw_pdf_text(
                pdf, y_coordinate,
                "峰 {peak}: q中心={center}, 半峰宽={width}, "
                "高置信点={points}, MAE={mae}, R2={r2}".format(
                    peak=row.get("q峰", "-"),
                    center=row.get("峰中心 q/1e-19C", "-"),
                    width=row.get("半峰宽/1e-19C", "-"),
                    points=row.get("高置信点数", "-"),
                    mae=row.get("MAE/V", "-"),
                    r2=row.get("R2", "-"),
                ),
                font_name, font_size=10, indent=12, line_gap=5)

    y_coordinate -= 8
    y_coordinate = _draw_pdf_section(pdf, y_coordinate, "四、结果图表",
                                     font_name)
    image_file = pdf_file.parent / "regression_plot.png"
    chart_status = _save_chart_image(fig, image_file, regression_results,
                                     font_path)
    if chart_status != "none" and image_file.exists():
        y_coordinate = _ensure_pdf_space(pdf, y_coordinate, font_name, 12,
                                         318)
        pdf.drawImage(str(image_file), 72, y_coordinate - 300,
                      width=468, height=300)
        y_coordinate -= 318
    else:
        y_coordinate = _draw_pdf_text(
            pdf, y_coordinate,
            "结果图表导出组件不可用，本报告保留核心结论与数值结果。",
            font_name)

    pdf.save()
    return chart_status


def _render_report_preview(regression_results):
    metrics = _report_metrics(regression_results)
    st.subheader("报告预览")
    st.caption("报告只保留核心结论、q 峰摘要、最终公式和结果图，不输出逐点明细表。")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("分析数据点", metrics["total_points"])
    col2.metric("高置信点", metrics["fit_points"])
    col3.metric("发现 q 峰", _format_int(metrics["cluster_count"]))
    col4.metric("整体 R2", _format_number(metrics["r2"], 4))

    st.markdown(
        "- AI 先从连续电荷估计中发现 q 峰，再用半峰宽筛选高置信点。\n"
        "- 符号回归只在高置信点上寻找共享规律，报告展示已代入各峰中心后的最终公式。\n"
        "- 逐点数据明细保留在原始 CSV 和分析页面中，报告不再重复输出。")

    curve_items = _curve_items(regression_results)
    if curve_items:
        st.markdown("**最终可用公式**")
        for _, display_label, _, _, expression in curve_items:
            with st.container(border=True):
                st.markdown(f"峰 {display_label}")
                st.code(f"U(t) = {expression}", language="text")

    peak_summary = _peak_summary_view(regression_results)
    if not peak_summary.empty:
        st.markdown("**q 峰摘要**")
        st.dataframe(peak_summary, use_container_width=True, hide_index=True)

    fig = _build_report_figure(regression_results)
    st.plotly_chart(fig, key="report_regression_plot_v2",
                    use_container_width=True)
    return fig


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
            st.warning("请先完成“机器学习—符号回归”，再生成实验报告。")

    regression_results = readiness["regression_results"]
    if regression_results is None:
        return

    fig = _render_report_preview(regression_results)

    if st.button("生成实验报告 PDF",
                 disabled=not readiness["can_generate"],
                 use_container_width=True):
        work_dir = Path(st.session_state.work_dir)
        student = st.session_state.student
        pdf_file = work_dir / f"report_{student['id']}_{student['name']}.pdf"
        font_path = Path(st.session_state.data_dir) / "chinese.msyh.ttf"

        with st.spinner("正在生成报告..."):
            chart_status = _create_pdf_report(
                pdf_file, student, regression_results, font_path, fig)

        st.session_state.last_report_pdf = str(pdf_file)
        st.success("实验报告已生成。")
        if chart_status == "fallback":
            st.info("本机 Plotly 图片导出组件不可用，PDF 已自动使用内置简化图。")
        elif chart_status == "none":
            st.warning("图像导出组件不可用，PDF 已保留文字结论和数值摘要。")

    last_report = st.session_state.get("last_report_pdf")
    if last_report and Path(last_report).exists():
        with open(last_report, "rb") as file:
            st.download_button(
                label="下载实验报告 PDF",
                data=file.read(),
                file_name=Path(last_report).name,
                mime="application/pdf",
                use_container_width=True,
            )
