"""
视觉自动测量模块 — Streamlit 启动器

启动独立 OpenCV 原生窗口进行视频测量（流畅 30fps）。
测量结果直接写入主数据文件，全平台共享。
"""
import subprocess, sys, os
from pathlib import Path

import streamlit as st
import pandas as pd

_STANDALONE_SCRIPT = Path(__file__).resolve().parent.parent / "vision_standalone.py"


def render_tab_vision():
    st.subheader("📷 视觉自动测量")

    work_dir = st.session_state.get("work_dir", Path.cwd())
    main_csv = work_dir / "oil_drop.csv"          # 主数据（手动 + 视觉共用）
    raw_csv = work_dir / "oil_drop_raw.csv"       # 原始测量记录（仅视觉）

    # ---- 启动按钮 ----
    col1, col2, col3 = st.columns(3)
    with col1:
        camera_index = st.selectbox("摄像头设备", options=[0, 1, 2, 3], index=1)

    col_a, col_b = st.columns(2)
    with col_a:
        script = str(_STANDALONE_SCRIPT)
        if st.button("▶ 启动测量", type="primary", use_container_width=True):
            cmd = [
                sys.executable, script,
                "--camera", str(camera_index),
                "--output", str(main_csv),
                "--output-raw", str(raw_csv),
            ]
            try:
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                    if sys.platform == "win32" else 0,
                )
                st.session_state["_vision_launched"] = True
                st.success("✅ 测量窗口已启动！")
            except Exception as e:
                st.error(f"启动失败: {e}")

    with col_b:
        if st.button("🔄 刷新数据（导入最新测量结果）", use_container_width=True):
            # 强制从文件重新加载
            if main_csv.exists():
                st.session_state.data = pd.read_csv(main_csv)
                st.toast("数据已刷新")
                st.rerun()

    # ---- 原始测量记录（按油滴编号分组展示）----
    if raw_csv.exists():
        with st.expander("📋 原始测量记录（按油滴分组）", expanded=True):
            try:
                df_raw = pd.read_csv(raw_csv)
                if not df_raw.empty:
                    droplet_ids = sorted(df_raw["DropletID"].unique())
                    st.caption(f"共 {len(droplet_ids)} 颗油滴，{len(df_raw)} 次原始测量")

                    for did in droplet_ids:
                        sub = df_raw[df_raw["DropletID"] == did]
                        times = sub["FallingTime(t/s)"].tolist()
                        # 直接读预存平均值列
                        if "AvgTime(t/s)" in df_raw.columns:
                            avg_t = sub["AvgTime(t/s)"].iloc[0]
                            avg_u = sub["AvgVoltage(U/V)"].iloc[0]
                        else:
                            avg_t = sum(times) / len(times)
                            avg_u = sum(sub["BalanceVoltage(U/V)"].tolist()) / len(times)

                        st.metric(
                            f"油滴 #{int(did)}",
                            f"平均 {avg_t:.3f}s / {avg_u:.1f}V",
                            delta=f"{len(times)} 次测量",
                        )

                        # 每条测量值带删除按钮
                        btn_cols = st.columns(len(sub) * 2)
                        for i, (_, row) in enumerate(sub.iterrows()):
                            meas_no = int(row["MeasurementNo"])
                            t_val = row["FallingTime(t/s)"]
                            with btn_cols[i * 2]:
                                st.caption(f"第{meas_no}次: {t_val:.3f}s")
                            with btn_cols[i * 2 + 1]:
                                if st.button("✕", key=f"del_raw_{int(did)}_{meas_no}",
                                           help=f"删除油滴#{int(did)} 第{meas_no}次测量"):
                                    # 1. 记住旧平均值（用于在 main CSV 中定位）
                                    old_avg = avg_t
                                    old_u = avg_u

                                    # 2. 从 raw CSV 删除这一行
                                    df_raw_new = df_raw.drop(row.name)
                                    df_raw_new.to_csv(raw_csv, index=False)

                                    # 3. 计算该油滴剩余测量值的新平均
                                    remaining = df_raw_new[df_raw_new["DropletID"] == did]
                                    if not remaining.empty:
                                        new_times = remaining["FallingTime(t/s)"].tolist()
                                        new_avg = sum(new_times) / len(new_times)
                                        new_u = remaining["BalanceVoltage(U/V)"].tolist()[0]

                                        # 更新 main CSV 中对应行
                                        if main_csv.exists():
                                            df_main = pd.read_csv(main_csv)
                                            mask = (
                                                (df_main["FallingTime(t/s)"] - old_avg).abs() < 0.001
                                            ) & (
                                                (df_main["BalanceVoltage(U/V)"] - old_u).abs() < 0.1
                                            )
                                            if mask.any():
                                                df_main.loc[mask, "FallingTime(t/s)"] = round(new_avg, 3)
                                                df_main.to_csv(main_csv, index=False)
                                    else:
                                        # 该油滴所有测量都删完了 → 从 main CSV 删除对应行
                                        if main_csv.exists():
                                            df_main = pd.read_csv(main_csv)
                                            mask = (
                                                (df_main["FallingTime(t/s)"] - old_avg).abs() < 0.001
                                            ) & (
                                                (df_main["BalanceVoltage(U/V)"] - old_u).abs() < 0.1
                                            )
                                            df_main = df_main[~mask]
                                            df_main.to_csv(main_csv, index=False)

                                    # 4. 重新加载 st.session_state.data
                                    if main_csv.exists():
                                        st.session_state.data = pd.read_csv(main_csv)
                                    st.toast(f"已删除 油滴#{int(did)} 第{meas_no}次测量")
                                    st.rerun()

                    st.divider()
                else:
                    st.caption("暂无原始记录")
            except Exception:
                st.caption("无法读取原始记录文件")
    else:
        st.info("还没有原始测量记录，启动测量后数据将在此显示")

    st.divider()

    # ---- 当前数据预览 ----
    st.subheader("📊 当前数据")
    if "data" in st.session_state and not st.session_state.data.empty:
        df = st.session_state.data.copy()
        df_display = df.reset_index(drop=True)
        df_display.insert(0, "序号", range(1, len(df_display) + 1))

        # 选要删的行
        to_delete = st.multiselect(
            "选择要删除的数据（按序号）",
            options=df_display["序号"].tolist(),
            help="选中序号后点下方按钮删除",
        )

        col_del, col_clr = st.columns(2)
        with col_del:
            if st.button("🗑️ 删除选中", use_container_width=True):
                if to_delete:
                    idx_to_drop = [i - 1 for i in to_delete]
                    df = df.drop(df.index[idx_to_drop])
                    new_data = df.reset_index(drop=True)
                    st.session_state.data = new_data
                    # 写回主 CSV
                    new_data.to_csv(main_csv, index=False)
                    st.toast(f"已删除 {len(to_delete)} 条数据")
                    st.rerun()
        with col_clr:
            if st.button("🗑️ 清空全部", use_container_width=True):
                st.session_state.data = pd.DataFrame(
                    columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])
                if main_csv.exists():
                    main_csv.unlink()
                if raw_csv.exists():
                    raw_csv.unlink()
                st.toast("已清空")
                st.rerun()

        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("暂无数据，启动测量或手动录入后数据将在此显示")
