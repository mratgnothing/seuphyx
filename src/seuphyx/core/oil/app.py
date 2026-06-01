# built-in
from datetime import datetime
from pathlib import Path
# third-party
import streamlit as st
import plotly.io as pio
import pandas as pd
# seuphyx
from seuphyx.web import StreamlitConfig, login
from seuphyx.core.oil.tabs.tab_record import render_tab_record
from seuphyx.core.oil.tabs.tab_train import render_tab_train
from seuphyx.core.oil.tabs.tab_classify import render_tab_classify
from seuphyx.core.oil.tabs.tab_regress import render_tab_regress
from seuphyx.core.oil.tabs.tab_vision import render_tab_vision
from seuphyx.core.oil.tabs.tab_report import render_tab_report
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

# 读取 logo 文件
with open(image_dir / "seu_logo.svg", "r", encoding="utf-8") as f:
    seu_logo_svg = f.read()

with open(image_dir / "seu_phy_logo.svg", "r", encoding="utf-8") as f:
    seu_phy_logo_svg = f.read()

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
    reference_file = data_dir / "oil_drop_reference.csv"
    st.session_state.data_ref = pd.read_csv(reference_file)
    st.session_state.data_ref_empty = pd.DataFrame(
        columns=["FallingTime(t/s)", "BalanceVoltage(U/V)"])
    st.session_state.data_ref_pred_empty = pd.DataFrame(
        columns=["FallingTime(t/s)", "BalanceVoltage(U/V)", "Predicted"])

    # 显示已保存的数据点
    if oil_drop_csv.exists():
        if len(st.session_state.data.values) == 0:
            st.session_state.data = pd.read_csv(oil_drop_csv)

        st.sidebar.subheader("已保存的数据点：")
        st.sidebar.dataframe(st.session_state.data)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["1. 数据记录", "2. 训练模型（选做）", "3. 数据分类", "4. 物理拟合", "5. 视觉测量", "6. 打印报告"])

    # 渲染各个 Tab
    with tab1:
        render_tab_record()

    with tab2:
        render_tab_train()

    with tab3:
        render_tab_classify()

    with tab4:
        render_tab_regress()

    with tab5:
        render_tab_vision()

    with tab6:
        render_tab_report()
