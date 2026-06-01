# built-in
from pathlib import Path
import base64
# third-party
# from streamlit_change_language import cst
import streamlit as st
# seuphyx
import seuphyx


# 将图片转换为 base64 编码
def image_to_base64(path):
    with open(path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode()


class StreamlitConfig(object):

    def __init__(self):
        # cst.change(language='cn')
        container = st.container()

        # self.set_background()
        self.set_header(container)
        self.container = container

        image_dir = Path(seuphyx.__file__).parent / "images"
        st.logo(image=image_dir / "seu_logo_word.svg",
                size='large',
                icon_image=image_dir / "seu_logo.svg")

    @staticmethod
    def set_header(container):
        ### 卷动时固定的标题栏

        container.write("""<div class='fixed-header'/>""",
                        unsafe_allow_html=True)
        st.markdown("""
        <style>
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                position: sticky;
                top: 2.875rem;
                background-color: white;
                z-index: 999;
            }
            # .fixed-header {
            #     border-bottom: 1px solid black;
            # }
        </style>
            """,
                    unsafe_allow_html=True)

    @staticmethod
    def set_background():
        # 加载 logo 和背景
        image_dir = Path(seuphyx.__file__).parent / "images"
        seu_logo_base64 = image_to_base64(image_dir / "seu_logo.png")
        seu_bg_base64 = image_to_base64(image_dir / "seu_background.png")

        # ==== 注入 CSS 样式 ====
        st.markdown(f"""
            <style>
                /* ===== 背景 + 白色透明罩 ===== */
                .stApp {{
                      background-image:
                    linear-gradient(to right,
                        transparent calc(45% - 400px),
                        rgba(255,255,255,0.6) 0 calc(55% + 400px),
                        transparent 0),
                    url("data:image/jpg;base64,{seu_bg_base64}");

                    background-size: 100% 100%, cover;
                    background-position: center, center;
                    background-repeat: no-repeat, no-repeat;

                    /* 关键：只保留竖向透罩，不需要水平透罩 */
                    background-blend-mode: normal, normal;
                }}

                /* ===== 右上角 logo ===== */
                .seu-logo {{
                    position: absolute;
                    top: -15px;
                    right: -30px;
                    width: 250px;
                    z-index: 999;
                }}

                /* ===== 标题 ===== */
                h1, .main-title {{
                    text-align: center;
                    font-size: 32px;
                    color: #222222;                 /* ← 改成深灰/黑色 */
                    font-weight: 700;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.7);  /* 浅白阴影 */
                    margin-top: 60px;
                }}

                /* ===== 副标题 ===== */
                .sub-title {{
                    text-align: center;
                    font-size: 20px;
                    color: #333333;                 /* 深灰文字 */
                    margin-bottom: 40px;
                }}

                /* ===== 按钮样式 ===== */
                div.stButton > button:first-child {{
                    background-color: rgba(255, 255, 255, 0.6);
                    color: #004c99;
                    border: 1px solid #004c99;
                    border-radius: 8px;
                    padding: 0.5em 1.5em;
                    font-size: 16px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
                }}

                /* 悬停时变蓝、白字 */
                div.stButton > button:first-child:hover {{
                    background-color: rgba(0, 102, 204, 0.8);
                    color: white;
                    border-color: rgba(0, 102, 204, 0.8);
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                }}

                /* 点击时稍深一点 */
                div.stButton > button:first-child:active {{
                    background-color: rgba(0, 80, 180, 0.9);
                    color: white;
                    box-shadow: inset 2px 2px 4px rgba(0,0,0,0.2);
                }}

            </style>

            <img src="data:image/png;base64,{seu_logo_base64}" class="seu-logo">
            """,
                    unsafe_allow_html=True)


@st.dialog("请输入你的学号与姓名", dismissible=False)
def login():
    st.write(f"请你输入你的学号与姓名以便记录：")
    student_id = st.text_input("学号：")
    student_name = st.text_input("姓名：")

    if st.button("登记信息"):
        is_valid = True
        # if not student_id.isdigit():
        #     st.error("学号应为数字，请重新输入。")
        #     is_valid = False

        if not 8 <= len(student_id) <= 10:
            st.error("学号应为8到10位，请重新输入。")
            is_valid = False

        if not student_name.isalpha():
            st.error("姓名应为中文或英文字符，请重新输入。")
            is_valid = False

        if is_valid:
            st.session_state.login = {
                "student_id": student_id,
                "student_name": student_name
            }
            st.rerun()
