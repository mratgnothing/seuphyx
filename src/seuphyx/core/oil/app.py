# built-in
from datetime import datetime
from pathlib import Path
# third-party
import streamlit as st
import plotly.io as pio
import pandas as pd
# seuphyx
from seuphyx.web import StreamlitConfig, login
import seuphyx

# ==== Streamlit 页面配置 ====

st.session_state.sidebar_state = "expanded"
image_dir = Path(seuphyx.__file__).parent / "images"
st.set_page_config(
    page_title="东南大学物理实验中心人工智能辅助数据处理平台(密立根油滴实验)",
    page_icon=image_dir / "seu_logo.svg",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state,
)


@st.cache_data(show_spinner=False)
def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def _load_reference_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# 读取 logo 文件
seu_logo_svg = _read_text_file(str(image_dir / "seu_logo.svg"))
seu_phy_logo_svg = _read_text_file(str(image_dir / "seu_phy_logo.svg"))

css = '''
header.stAppHeader {
    background: rgba(0, 0, 0, 0);
}

div.stMainBlockContainer {
    padding-top: 0.2rem;
}

div.st-key-app_title div.stHeading {
    text-align: center;
}

div.st-key-app_title div.stHeading h1 {
    font-size: 4.5rem;
    letter-spacing: 0px;
    margin: 0;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
}

div.st-key-app_title div.stHeading h1::before{
    content: "";
    display: inline-block;
    width: 4.5rem;
    height: 4.5rem;
    background-image: url("data:image/svg+xml;base64,''' + __import__(
    'base64').b64encode(seu_logo_svg.encode()).decode() + '''");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    flex-shrink: 0;
}

div.st-key-app_title div.stHeading h1::after {
    content: "";
    display: inline-block;
    width: 4.5rem;
    height: 4.5rem;
    background-image: url("data:image/svg+xml;base64,''' + __import__(
        'base64').b64encode(seu_phy_logo_svg.encode()).decode() + '''");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    flex-shrink: 0;
}

/* 隐藏标题后的锚点链接图标 */
div.st-key-app_title div.stHeading h1 a {
    display: none !important;
}

/* 隐藏标题后的 span 元素 */
div.st-key-app_title div.stHeading h1 span {
    display: none !important;
}
'''

st.html(f"<style>{css}</style>")

with st.container(key="app_title"):
    st.title("人工智能(AI)辅助数据处理平台")

# 添加额外的 CSS 来实现左右对齐的 header
header_css = '''
div.st-key-experiment_header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}

div.st-key-experiment_header .stMarkdown {
    margin: 0;
}

div.st-key-experiment_header .stMarkdown p {
    margin: 1000;
    font-size: 2rem;
    font-weight: 500;
}
'''

st.html(f"<style>{header_css}</style>")


def render_workflow_status():
    """Show the experiment workflow readiness at the top of the app."""
    has_data = ("data" in st.session_state
                and not st.session_state.data.empty)
    root_reference = Path.cwd() / "oil_drop_reference.csv"
    has_test_data = root_reference.exists()
    has_analysis_data = has_data or has_test_data
    has_regression = "regression_results" in st.session_state
    has_clustering = "charge_clustering_result" in st.session_state
    work_dir = st.session_state.get("work_dir")
    has_report = bool(work_dir and list(work_dir.glob("report_*.pdf")))

    steps = [
        ("登录", True, "学生信息已登记"),
        ("数据测量", has_analysis_data, "视觉测量或根目录测试数据"),
        ("AI聚类", has_clustering, "从 Q 分布发现自然簇"),
        ("符号回归", has_regression, "归纳共享可解释公式"),
        ("实验报告", has_report, "生成并下载 PDF 报告"),
    ]

    if not has_analysis_data:
        next_step = "下一步：进入“视觉测量”页完成测量，或检查根目录测试数据 oil_drop_reference.csv。"
    elif not has_data:
        next_step = "当前没有实测数据；可以直接使用根目录 oil_drop_reference.csv 进入 AI 聚类，也可以先完成视觉测量。"
    elif not has_clustering:
        next_step = "下一步：进入“AI聚类”页，用无监督学习发现 Q 分布簇。"
    elif not has_regression:
        next_step = "下一步：进入“机器学习—符号回归”页，选择有效簇并拟合共享公式。"
    elif not has_report:
        next_step = "下一步：进入“打印报告”页生成实验报告。"
    else:
        next_step = "本次实验流程已完成。"

    with st.container(border=True):
        st.markdown("#### 实验流程")
        cols = st.columns(len(steps))
        for idx, (label, done, help_text) in enumerate(steps, start=1):
            status = "完成" if done else "待完成"
            with cols[idx - 1]:
                st.metric(f"{idx}. {label}", status)
                st.caption(help_text)
        st.info(next_step)


with st.container(key="experiment_header"):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**实验名称：密立根油滴实验**")
    with col2:
        st.markdown(
            '<p style="text-align: right; font-size: 2rem; font-weight: 700; margin: 0;"><strong>by 东南大学物理实验中心</strong></p>',
            unsafe_allow_html=True)

# 学生学号登陆
if 'login' not in st.session_state:
    login()
else:
    student_name = st.session_state.login['student_name']
    student_id = st.session_state.login['student_id']

    if 'work_dir' not in st.session_state:
        pwd = Path().cwd().resolve()
        suffix = f'oil_drop_{student_id}_{student_name}'
        dirname = pwd / f"{datetime.now():%Y.%m.%d}" / f"{suffix}"
        dirname.mkdir(parents=True, exist_ok=True)
        st.session_state.work_dir = dirname
        st.session_state.student = {'name': student_name, 'id': student_id}

    st.sidebar.success(f"欢迎 {student_name} ({student_id}) 开始本次课程！")
    st.sidebar.success(f"所有数据将保存至 {st.session_state.work_dir} 目录下。")
    st.sidebar.info('''
                    **如有问题请联系：**
                    
                    任课老师：陈乾 教授
                    
                    邮箱地址：qc119@seu.edu.cn
                    
                    算法支持：夏谦 博士
                    ''')

    # 初始化 session_state
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(
            columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])
    if 'data_ref' not in st.session_state:
        st.session_state.data_ref = pd.DataFrame(
            columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])

    # 初始化工作目录和输入文件
    work_dir = st.session_state.work_dir
    oil_drop_csv = work_dir / "oil_drop.csv"
    pio.templates.default = "ggplot2"

    # 初始化相关参数
    data_dir = Path(seuphyx.__file__).parent / "data"
    st.session_state.data_dir = data_dir
    reference_file = Path.cwd() / "oil_drop_reference.csv"
    st.session_state.data_ref = _load_reference_csv(str(reference_file))
    st.session_state.data_ref_empty = pd.DataFrame(
        columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])
    st.session_state.data_ref_pred_empty = pd.DataFrame(
        columns=["FallingTime(t/s)", "BalanceVoltage(U/V)", "Predicted"])

    # Monitor the shared CSV so visual measurement can append data out of
    # process while the Streamlit app stays in sync.
    _csv_mtime_key = "_oil_drop_csv_mtime"
    if oil_drop_csv.exists():
        current_mtime = oil_drop_csv.stat().st_mtime
        last_mtime = st.session_state.get(_csv_mtime_key, 0)
        if current_mtime > last_mtime or len(st.session_state.data.values) == 0:
            st.session_state.data = pd.read_csv(oil_drop_csv)
            st.session_state[_csv_mtime_key] = current_mtime

        st.sidebar.subheader("当前测量数据")
        st.sidebar.metric("数据点数", len(st.session_state.data))
        with st.sidebar.expander("查看最近 20 条数据", expanded=False):
            st.dataframe(st.session_state.data.tail(20),
                         use_container_width=True,
                         hide_index=True)

    render_workflow_status()

    page_options = [
        "1. 视觉测量",
        "2. AI聚类",
        "3. 机器学习—符号回归",
        "4. 打印报告",
    ]
    if st.session_state.get("oil_active_page") not in (None, *page_options):
        st.session_state.oil_active_page = page_options[0]

    page = st.radio(
        "功能页面",
        page_options,
        horizontal=True,
        label_visibility="collapsed",
        key="oil_active_page",
    )

    # Streamlit tabs 会在每次重跑时执行所有页；这里只渲染当前页来降低输入延迟。
    if page == "1. 视觉测量":
        from seuphyx.core.oil.tabs.tab_record import render_tab_record
        from seuphyx.core.oil.tabs.tab_vision import render_tab_vision
        vision_tab, manual_tab = st.tabs(["视觉自动测量", "手动录入"])
        with vision_tab:
            render_tab_vision()
        with manual_tab:
            render_tab_record()
    elif page == "2. AI聚类":
        from seuphyx.core.oil.tabs.tab_classify import render_tab_classify
        render_tab_classify()
    elif page == "3. 机器学习—符号回归":
        from seuphyx.core.oil.tabs.tab_regress import render_tab_regress
        render_tab_regress()
    elif page == "4. 打印报告":
        from seuphyx.core.oil.tabs.tab_report import render_tab_report
        render_tab_report()
