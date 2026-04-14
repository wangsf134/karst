import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import base64
from typing import Dict, Any

# 调用大模型 API
from openai import OpenAI

# ================= 0. 环境路径初始化 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 自动处理 Logo 文件
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

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

# ================= 1. 全局配置与状态初始化 =================
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

# 【核心修复】：全局初始化对话记忆，防止 Tab 切换报错
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """你是内嵌在“碳绘喀斯特”系统中的政务生态辅助智囊。
你的对话对象是一线基层政务工作者、驻村干部及生态规划员。请时刻保持务实、严谨、接地气的工作态度。你的目标是帮他们把好生态关，算好经济账。绝对不要表现出你正在与系统的开发者对话。

【工作目标】
将复杂的碳汇核算数据和生态学原理，转化为基层干部“看得懂、用得上、好落地”的决策参考。既要规避瞎指挥造成的财政浪费，又要兼顾水土保持与乡村振兴。

【内在诊断逻辑（绝对机密：请将以下逻辑自然转化为“施政建议”融入分析，严禁向用户提及“系统规则”、“后台设定”等字眼）】：
1. 资金与生态双重防线（乔木红线）：喀斯特地区土层厚度是“生死线”。当土层<15cm时，必须明确制止种植乔木的规划。要向干部讲透利害：浅土种树不仅成活率极低（浪费涉农资金），且树根钻探会破坏脆弱岩层，反而加剧石漠化。应“因地制宜”推荐灌木或特色草本（如牧草、金银花等），既保水土，又促增收。
2. 碳汇倒挂预警（排碳归因）：若气温较高（>15℃）且伴随缺土，系统核算结果出现“负向”（排碳）。需向干部解释：高温缺土会让地里原有的碳被加速“蒸”出来。建议暂缓将该地块申报碳汇交易项目（避免通不过考核），应先侧重地力修复和保墒培肥。
3. 汇报与回复风格：结论先行，直击痛点，给出实操抓手。多用“建议您”、“从实际落地来看”、“考虑到资金效益”等政务工作语言。坚决杜绝堆砌学术名词和“假大空”的套话。"""}
    ]

# CSS 注入
st.markdown(
    """
    <style>
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
    st.markdown('<h1 style="margin: 0; padding: 0; font-size: 2.8rem; font-weight: 600; line-height: 1.2; margin-bottom: 25px;">碳绘喀斯特：县域固碳评估与情景模拟沙盘</h1>', unsafe_allow_html=True)

st.markdown("---")

# ================= 2. 核心核算引擎加载 =================
calc_engine = load_model()

# ================= 3. 功能模块布局 =================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "单点精细诊断"

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

    features = {'T': T, 'RH': RH, 'R': R, 'Rg': Rg, 'Slope': Slope, 'Soil_Thickness': Soil_Thickness,
                'Rock_Outcrop': Rock_Outcrop, 'Veg_Type': Veg_Type}

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("开始模拟评估与资产核算", type="primary", width="stretch")

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
        cl, cr = st.columns(2, gap="medium")
        with cl:
            render_result_chart(calc_engine, features)
        with cr:
            render_shap_waterfall(calc_engine, features)

        st.markdown("---")
        st.subheader("系统逆向反演：适生植被规划推荐")
        rec_results = []
        for v_name, v_code in VEG_MAPPING.items():
            tf = features.copy()
            tf['Veg_Type'] = v_code
            vp = predict_flux(calc_engine, tf)
            va = calculate_carbon_assets(vp, area_ha, carbon_price)
            risk = "适宜"
            if Soil_Thickness < 15 and v_code == 1:
                risk = "生存受限(土层不足)"
            elif vp < 0:
                risk = "排碳风险"
            rec_results.append({"规划方案": v_name, "年度核算固碳潜力 (gC/m²/yr)": round(vp, 4),
                                "预期综合损益 (元)": va['annual_revenue'], "生态约束评价": risk})
        st.dataframe(pd.DataFrame(rec_results).sort_values(by="年度核算固碳潜力 (gC/m²/yr)", ascending=False),
                     width="stretch", hide_index=True)

# --- 模块 2: 区域批量测算 ---
elif selected_tab == "区域批量测算":
    st.markdown("### 大规模区域测算")
    st.info("说明：支持批量上传区域林班调查数据，系统将自动执行合规性校验并输出核算报表。")

    template_df = pd.DataFrame({
        '地块编号': ['Plot_001', 'Plot_002'], '年均温度 (℃)': [15.0, 18.5], '年均相对湿度 (%)': [70.0, 65.0],
        '年降水总量 (mm)': [1000.0, 850.0], '年均太阳辐射 (W/m²)': [150.0, 200.0], '坡度 (°)': [15.0, 25.0],
        '土壤厚度 (cm)': [10.0, 30.0], '裸岩率 (%)': [60.0, 40.0], '面积 (公顷)': [5.5, 12.0], '植被类型': [1, 2]
    })
    st.download_button(label="下载标准核算数据模板 (.csv)", data=template_df.to_csv(index=False).encode('utf-8-sig'),
                       file_name="固碳资产批量核算模板.csv", mime="text/csv", width="stretch")

    uploaded_file = st.file_uploader(label="数据上传区", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            with st.spinner("正在执行系统合规性校验..."):
                df_input.columns = [str(col).strip() for col in df_input.columns]
                required_cols = ['年均温度 (℃)', '年均相对湿度 (%)', '年降水总量 (mm)', '年均太阳辐射 (W/m²)',
                                 '坡度 (°)', '土壤厚度 (cm)', '裸岩率 (%)', '面积 (公顷)', '植被类型']
                df_clean = df_input.dropna(subset=required_cols)
                st.success("数据校验通过。")

            st.dataframe(df_clean.head(), width="stretch")
            c1, c2 = st.columns(2)
            mapping_dict = {'年均温度 (℃)': 'T', '年均相对湿度 (%)': 'RH', '年降水总量 (mm)': 'R',
                            '年均太阳辐射 (W/m²)': 'Rg', '坡度 (°)': 'Slope', '土壤厚度 (cm)': 'Soil_Thickness',
                            '裸岩率 (%)': 'Rock_Outcrop', '植被类型': 'Veg_Type'}

            if c1.button("执行现状资产核算", type="primary", width="stretch"):
                with st.spinner("正在逐行执行安全核算..."):
                    def safe_predict_current(row):
                        features = {
                            'T': row['年均温度 (℃)'],
                            'RH': row['年均相对湿度 (%)'],
                            'R': row['年降水总量 (mm)'],
                            'Rg': row['年均太阳辐射 (W/m²)'],
                            'Slope': row['坡度 (°)'],
                            'Soil_Thickness': row['土壤厚度 (cm)'],
                            'Rock_Outcrop': row['裸岩率 (%)'],
                            'Veg_Type': row['植被类型']
                        }
                        return predict_flux(calc_engine, features) * 365

                    df_clean['年度核算固碳潜力'] = df_clean.apply(safe_predict_current, axis=1).round(4)

                    df_clean['年度综合损益 (元)'] = df_clean.apply(
                        lambda r: calculate_carbon_assets(r['年度核算固碳潜力'], r['面积 (公顷)'], DEFAULT_CARBON_PRICE)['annual_revenue'],
                        axis=1
                    )

                    st.success("现状资产核算完成！")
                    st.dataframe(df_clean, width="stretch")

                    st.download_button(
                        label="导出现状核算报表",
                        data=df_clean.to_csv(index=False).encode('utf-8-sig'),
                        file_name="现状资产核算报表.csv",
                        mime="text/csv",
                        width="stretch"
                    )

            if c2.button("执行最优规划推演", type="secondary", width="stretch"):
                with st.spinner("正在执行全矩阵推演 (含业务红线约束)..."):
                    def simulate_optimal(row):
                        base_features = {
                            'T': row['年均温度 (℃)'],
                            'RH': row['年均相对湿度 (%)'],
                            'R': row['年降水总量 (mm)'],
                            'Rg': row['年均太阳辐射 (W/m²)'],
                            'Slope': row['坡度 (°)'],
                            'Soil_Thickness': row['土壤厚度 (cm)'],
                            'Rock_Outcrop': row['裸岩率 (%)']
                        }

                        best_veg_name = None
                        max_pot = -float('inf')

                        for v_name, v_code in VEG_MAPPING.items():
                            feats = base_features.copy()
                            feats['Veg_Type'] = v_code

                            pot = predict_flux(calc_engine, feats) * 365

                            if base_features['Soil_Thickness'] < 15 and v_code == 1:
                                pot = -99999.0

                            if pot > max_pot:
                                max_pot = pot
                                best_veg_name = v_name

                        return pd.Series([best_veg_name, max_pot])

                    df_clean[['系统推荐方案', '规划后固碳潜力']] = df_clean.apply(simulate_optimal, axis=1)
                    df_clean['规划后固碳潜力'] = df_clean['规划后固碳潜力'].round(4)

                    df_clean['规划后预期损益 (元)'] = df_clean.apply(
                        lambda r: calculate_carbon_assets(r['规划后固碳潜力'], r['面积 (公顷)'], DEFAULT_CARBON_PRICE)['annual_revenue'],
                        axis=1
                    )

                    st.success("最优规划推演完成！浅土层已自动规避乔木。")
                    st.dataframe(df_clean, width="stretch")

                    st.download_button("导出规划建议书", df_clean.to_csv(index=False).encode('utf-8-sig'),
                                       "最优生态规划建议书.csv", "text/csv")
        except Exception as e:
            st.error(f"批量核算异常：{str(e)}")

# --- 模块 3: 智能生态助理 ---
elif selected_tab == "智能生态助理":
    st.markdown("### 🤖 智能生态助理 (Qwen3.5-Flash)")
    st.info("我是基于大语言模型的政务辅助规划助手。您可以向我提问关于碳汇核算逻辑、喀斯特地貌修复建议或系统使用说明。")

    try:
        client = OpenAI(
            api_key=st.secrets["ALIYUN_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception as e:
        st.warning("⚠️ 请确保已在云端 Secrets 中配置了 `ALIYUN_API_KEY`。")
        st.stop()

    current_messages = st.session_state.get("messages", [])
    for msg in current_messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("您可以这样问：为什么土层厚度低于15cm时系统不建议种树？这对老百姓有什么影响？"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("生态数据分析中..."):
                    stream = client.chat.completions.create(
                        model="qwen3.5-flash",
                        messages=st.session_state["messages"],
                        temperature=0.7,
                        stream=True
                    )
                    full_response = st.write_stream(stream)

                st.session_state["messages"].append({"role": "assistant", "content": full_response})

                MAX_HISTORY = 10
                if len(st.session_state["messages"]) > (MAX_HISTORY + 1):
                    st.session_state["messages"] = [st.session_state["messages"][0]] + st.session_state["messages"][-MAX_HISTORY:]

            except Exception as e:
                if "429" in str(e) or "Throttling" in str(e):
                    st.warning("当前通道较忙，请等待 10 秒后再试。")
                else:
                    st.error(f"调用大模型失败: {e}")