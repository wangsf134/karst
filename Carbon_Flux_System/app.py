import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import base64
from typing import Dict, Any

# 【新增引入】用于调用大模型 API
from openai import OpenAI

# ================= 0. 环境路径初始化 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# 自动处理 Logo 文件：将本地图片转为 Base64 字符串以实现 HTML 注入
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None


# 定位 Logo 文件
logo_filename = "logo.png"
logo_path = os.path.join(current_dir, logo_filename)
bin_str = get_base64_of_bin_file(logo_path)

# 内部业务模块导入
from config import (
    PAGE_TITLE, PAGE_LAYOUT, VEG_MAPPING,
    DEFAULT_CARBON_PRICE, C_TO_CO2_FACTOR
)
from utils.model_handler import load_model, predict_flux
from utils.visualizer import render_result_chart, render_shap_waterfall
from utils.economics import calculate_carbon_assets
from utils.logger import get_logger

# 初始化系统日志
logger = get_logger("APP_MAIN")

# ================= 1. 全局样式与配置 =================
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

# CSS 注入
st.markdown(
    """
    <style>
    /* 界面汉化与字体优化 */
    [data-testid="stFileUploadDropzone"] div > span { font-size: 0px !important; }
    [data-testid="stFileUploadDropzone"] div > span::after {
        content: "请将区域调查数据表格拖拽至此处" !important;
        font-size: 16px !important; font-weight: bold; color: #31333F;
    }
    [data-testid="stFileUploadDropzone"] small { font-size: 0px !important; }
    [data-testid="stFileUploadDropzone"] small::after {
        content: "单文件限制 200MB • 支持 CSV, XLSX" !important;
        font-size: 13px !important; color: #888888; margin-top: 5px !important; display: block;
    }
    [data-testid="stFileUploadDropzone"] button::after { content: "浏览文件" !important; visibility: visible; }
    [data-testid="stFileUploadDropzone"] button { font-size: 0px !important; }

    html, body, [class*="css"] { font-family: "DejaVu Sans", "Source Sans Pro", "Microsoft YaHei", sans-serif; }

    /* Tab 视觉伪装 */
    div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[role="radiogroup"] {
        display: flex; flex-direction: row; gap: 2rem !important;
        border-bottom: 1px solid #f0f2f6; padding-bottom: 0px; margin-bottom: 1.5rem;
    }
    div[role="radiogroup"] label { cursor: pointer; padding: 0.5rem 0.5rem; border-bottom: 3px solid transparent; transition: all 0.2s ease; }
    div[role="radiogroup"] label p { font-size: 1.1rem !important; font-weight: 600 !important; color: #7a7a7a; }
    div[role="radiogroup"] label:hover p { color: #ff4b4b; }
    div[role="radiogroup"] label:has(input:checked) { border-bottom: 3px solid #ff4b4b !important; }
    div[role="radiogroup"] label:has(input:checked) p { color: #ff4b4b !important; }

    .stApp { margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= 动态渲染 Header =================
if bin_str:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
            <h1 style="margin: 0; padding: 0; font-size: 2.8rem; font-weight: 600; line-height: 1.2;">碳绘喀斯特：县域固碳评估与情景模拟沙盘</h1>
            <img src="data:image/png;base64,{bin_str}" style="max-height: 150px; width: auto; max-width: 300px; object-fit: contain; margin-left: 20px;">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<h1 style="margin: 0; padding: 0; font-size: 2.8rem; font-weight: 600; line-height: 1.2; margin-bottom: 25px;">碳绘喀斯特：县域固碳评估与情景模拟沙盘</h1>',
        unsafe_allow_html=True)

st.markdown("---")

# ================= 2. 核心核算引擎加载 =================
calc_engine = load_model()

# ================= 3. 功能模块布局 =================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "单点精细诊断"

# 【修改点】：增加了一个全新的导航选项 "智能生态助理"
selected_tab = st.radio(
    label="导航菜单",
    options=["单点精细诊断", "区域批量测算", "智能生态助理"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab"
)

# --- 模块 1: 单点精细诊断 ---
if selected_tab == "单点精细诊断":
    st.markdown("### 实时地块模拟")
    st.info("说明：通过调节下方环境参数，系统将实时核算特定地貌配置下的年度固碳潜力。")

    col_input_left, col_input_right = st.columns(2, gap="large")

    with col_input_left:
        st.subheader("气象条件 (年度指标)")
        T = st.slider("年均温度 (℃)", -10.0, 40.0, 15.0)
        RH = st.slider("年均相对湿度 (%)", 0.0, 100.0, 70.0)
        R = st.slider("年降水总量 (mm)", 0.0, 3000.0, 1000.0)
        Rg = st.slider("年均太阳辐射 (W/m²)", 0.0, 1000.0, 150.0)

    with col_input_right:
        st.subheader("地表特征与人为干预")
        Slope = st.slider("坡度 (°)", 0.0, 60.0, 15.0)
        Soil_Thickness = st.slider("土壤厚度 (cm)", 10.0, 100.0, 15.0)
        Rock_Outcrop = st.slider("裸岩率 (%)", 0.0, 100.0, 60.0)
        veg_choice = st.selectbox("目标植被类型", list(VEG_MAPPING.keys()))
        Veg_Type = VEG_MAPPING[veg_choice]

        st.markdown("---")
        st.subheader("碳汇资产核算配置")
        area_ha = st.number_input("评估地块面积 (公顷)", min_value=0.1, value=10.0, step=0.1)
        carbon_price = st.slider("模拟市场碳价 (元/吨)", 0.0, 200.0, DEFAULT_CARBON_PRICE)

    features = {
        'T': T, 'RH': RH, 'R': R, 'Rg': Rg,
        'Slope': Slope, 'Soil_Thickness': Soil_Thickness,
        'Rock_Outcrop': Rock_Outcrop, 'Veg_Type': Veg_Type
    }

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("开始模拟评估与资产核算", type="primary", use_container_width=True)

    if run_btn:
        st.markdown("---")
        st.subheader("报告诊断输出")

        potential_val = predict_flux(calc_engine, features)
        assets = calculate_carbon_assets(potential_val, area_ha, carbon_price)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="年度核算固碳潜力", value=f"{potential_val:.4f} gC/m²/yr")
        with m2:
            st.metric(label="年度 CO$_2$e 核证量", value=f"{assets['annual_co2e_tons']} tCO2e/yr")
        with m3:
            st.metric(label="生态资产综合损益", value=f"¥{assets['annual_revenue']:,.2f}")

        if Soil_Thickness < 15 and Veg_Type == 1:
            st.error("严重预警：当前地块土层深度不足以支撑乔木生长。强行规划将面临极高植被退化风险。")
        elif potential_val > 0:
            st.success("核算通过：当前环境配置具备正向固碳效益及资产转化潜力。")
        else:
            st.warning("警示：当前配置呈负向碳平衡状态，请审视植被选型或加强地力修复。")

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart_left, col_chart_right = st.columns(2, gap="medium")
        with col_chart_left:
            render_result_chart(calc_engine, features)
        with col_chart_right:
            render_shap_waterfall(calc_engine, features)

        st.markdown("---")
        st.subheader("系统逆向反演：适生植被规划推荐")
        st.caption("系统已锁定环境参数，通过全路径矩阵扫描输出综合效益最优配置建议。")

        rec_results = []
        for v_name, v_code in VEG_MAPPING.items():
            test_features = features.copy()
            test_features['Veg_Type'] = v_code
            v_potential = predict_flux(calc_engine, test_features)
            v_assets = calculate_carbon_assets(v_potential, area_ha, carbon_price)
            risk_level = "适宜"
            if Soil_Thickness < 15 and v_code == 1:
                risk_level = "生存受限(土层不足)"
            elif v_potential < 0:
                risk_level = "排碳风险"
            rec_results.append({
                "规划方案": v_name,
                "年度核算固碳潜力 (gC/m²/yr)": round(v_potential, 4),
                "预期综合损益 (元)": v_assets['annual_revenue'],
                "生态约束评价": risk_level
            })

        st.dataframe(pd.DataFrame(rec_results).sort_values(by="年度核算固碳潜力 (gC/m²/yr)", ascending=False),
                     use_container_width=True, hide_index=True)

# --- 模块 3: 智能生态助理 (新增) ---
elif selected_tab == "智能生态助理":
    # 标题已修改为 Qwen3.5-Flash
    st.markdown("### 🤖 智能生态助理 (Qwen3.5-Flash)")
    st.info("我是基于大语言模型的生态规划助手。您可以向我提问关于碳汇核算逻辑、喀斯特地貌修复建议或系统使用说明。")

    # 安全地初始化 OpenAI 客户端 (连接到阿里云百炼)
    try:
        api_key = st.secrets["ALIYUN_API_KEY"]  # 读取阿里云的密码
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云的接口大门
        )
    except KeyError:
        st.warning(
            "⚠️ 系统未检测到 API 密钥。请确保已在 `.streamlit/secrets.toml` 或 Streamlit Cloud 的 Secrets 设置中配置了 `ALIYUN_API_KEY`。")
        st.stop()
    except Exception as e:
        st.error(f"初始化大模型客户端失败: {e}")
        st.stop()

    # 初始化会话级聊天历史 (包含强大的系统人设)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": """你是一个专业的喀斯特生态环境与碳汇资产评估AI助手。
            你的任务是协助用户理解环境参数对固碳潜力的影响。请严谨使用生态学和碳汇经济学术语，用简练的中文分段回答。
            核心规则：
            1. 喀斯特地貌中，土层厚度是核心限制因子。低于15cm不建议种乔木，强制种会面临退化。
            2. 高温（>15℃）在缺土环境下会加剧土壤呼吸作用，导致严重碳排放。
            3. 回答要落地、实在，不要过度堆砌套话。"""}
        ]

    # 将历史对话渲染到界面上 (隐藏掉底层的 system 设定)
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 捕获用户的聊天输入
    if prompt := st.chat_input("您可以这样问：为什么土层厚度低于15cm时系统会发出严重预警？"):

        # 新增：历史记录清理逻辑 (滑动窗口)
        # 设定：最多保留最新的 10 条对话记录（相当于 5 轮“问+答”）。
        # 加上 1 条雷打不动的 System 人设，包裹里最多只能装 11 个元素。
        MAX_HISTORY_LENGTH = 10

        # 检查包裹
        if len(st.session_state.messages) > (MAX_HISTORY_LENGTH + 1):
            # 保护系统人设
            system_prompt = st.session_state.messages[0]
            # 截取最新最近的10条对话
            recent_history = st.session_state.messages[-MAX_HISTORY_LENGTH:]
            #把人设和最新的对话重新拼装，替换掉原来的大雪球
            st.session_state.messages = [system_prompt] + recent_history
        # ====================================================================

        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 调用大模型并显示思考过程 (升级为流式输出！)
        # 【注意这里的缩进修复】：assistant 的框必须和 user 的框平级
        with st.chat_message("assistant"):
            try:
                # 核心改动：模型名字换成了通义千问极速版
                stream = client.chat.completions.create(
                    model="qwen3.5-flash",
                    messages=st.session_state.messages,
                    temperature=0.7,
                    stream=True  # 👈 魔法开关：开启打字机流式输出
                )

                # 使用 Streamlit 原生的打字机渲染组件
                full_response = st.write_stream(stream)

                # 3. 把大模型的完整回答存入历史，让它有“记忆”
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_str = str(e)
                # 兼容阿里可能出现的限流报错提示
                if "429" in error_str or "Throttling" in error_str:
                    st.warning("算力通道当前比较拥挤，或者您发送得太快啦。请等待 10 秒后再试一次哦！")
                else:
                    # 其他真正的报错再显示出来
                    st.error(f"调用 API 失败: {error_str}")