import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import base64
from typing import Dict, Any

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

# ================= 动态渲染 Header：放大的 Logo 与标题 =================
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

selected_tab = st.radio(
    label="导航菜单",
    options=["单点精细诊断", "区域批量测算"],
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
        # 【修改点】：最小土层锁定为 5.0cm，默认值设为 15.0cm 以规避模型边界 Bug
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

# --- 模块 2: 区域批量测算 ---
elif selected_tab == "区域批量测算":
    st.markdown("### 大规模区域测算")
    st.info("说明：支持批量上传区域林班调查数据，系统将自动执行合规性校验并输出核算报表。")

    st.markdown("#### 填表规范说明")
    st.warning("""
    **核心注意：** 【植被类型】字段请务必填写对应的**数字代码**：
    * **1** 代表 **乔木 (森林)** | **2** 代表 **灌木** | **3** 代表 **草本/农田**
    """)

    template_df = pd.DataFrame({
        '地块编号': ['Plot_001', 'Plot_002', 'Plot_003'],
        '年均温度 (℃)': [15.0, 18.5, 16.0],
        '年均相对湿度 (%)': [70.0, 65.0, 75.0],
        '年降水总量 (mm)': [1000.0, 850.0, 1200.0],
        '年均太阳辐射 (W/m²)': [150.0, 200.0, 180.0],
        '坡度 (°)': [15.0, 25.0, 10.0],
        '土壤厚度 (cm)': [10.0, 30.0, 8.0],
        '裸岩率 (%)': [60.0, 40.0, 80.0],
        '面积 (公顷)': [5.5, 12.0, 3.2],
        '植被类型': [1, 2, 3]
    })

    st.download_button(
        label="下载标准核算数据模板 (.csv)",
        data=template_df.to_csv(index=False).encode('utf-8-sig'),
        file_name="固碳资产批量核算模板.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("##### 请选择或拖拽区域调查数据表至下方区域")
    uploaded_file = st.file_uploader(label="数据上传区", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_input = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_input = pd.read_csv(uploaded_file, encoding='gbk')
            else:
                df_input = pd.read_excel(uploaded_file)

            with st.spinner("正在执行系统合规性校验..."):
                cleaned_columns = [
                    str(col).strip().replace("W/m2", "W/m²").replace("W/m^2", "W/m²").replace("（", "(").replace("）",
                                                                                                                ")") for
                    col in df_input.columns]
                df_input.columns = cleaned_columns
                df_clean = df_input.dropna(how='all')
                required_cols = ['年均温度 (℃)', '年均相对湿度 (%)', '年降水总量 (mm)', '年均太阳辐射 (W/m²)',
                                 '坡度 (°)', '土壤厚度 (cm)', '裸岩率 (%)', '面积 (公顷)', '植被类型']
                missing_cols = [col for col in required_cols if col not in df_clean.columns]

                if missing_cols:
                    st.error(f"校验失败：缺失核心业务字段：{missing_cols}")
                    st.stop()

                for col in [c for c in required_cols if c != '植被类型']:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean = df_clean.dropna(subset=required_cols)
                st.success("数据校验通过。")

            st.write("**数据预览 (前5行)：**")
            st.dataframe(df_clean.head(), use_container_width=True)

            col_btn1, col_btn2 = st.columns(2)
            mapping_dict = {
                '年均温度 (℃)': 'T', '年均相对湿度 (%)': 'RH', '年降水总量 (mm)': 'R',
                '年均太阳辐射 (W/m²)': 'Rg', '坡度 (°)': 'Slope', '土壤厚度 (cm)': 'Soil_Thickness',
                '裸岩率 (%)': 'Rock_Outcrop', '植被类型': 'Veg_Type'
            }

            if col_btn1.button("执行现状资产核算", type="primary", use_container_width=True):
                with st.spinner("核算中..."):
                    X_matrix = df_clean[list(mapping_dict.keys())].rename(columns=mapping_dict)
                    results = calc_engine.predict(X_matrix) * 365
                    df_clean['年度核算固碳潜力'] = results.round(4)
                    df_clean['年度综合损益 (元)'] = df_clean.apply(
                        lambda r:
                        calculate_carbon_assets(r['年度核算固碳潜力'], r['面积 (公顷)'], DEFAULT_CARBON_PRICE)[
                            'annual_revenue'], axis=1
                    )
                    st.dataframe(df_clean, use_container_width=True)
                    st.download_button("导出核算报表", df_clean.to_csv(index=False).encode('utf-8-sig'),
                                       "现状资产核算报表.csv", "text/csv")

            if col_btn2.button("执行最优规划推演", type="secondary", use_container_width=True):
                with st.spinner("推演中..."):
                    base_features = df_clean[list(mapping_dict.keys())].rename(columns=mapping_dict).drop(
                        columns=['Veg_Type'])
                    results_matrix = pd.DataFrame({
                        '树': calc_engine.predict(base_features.assign(Veg_Type=1)) * 365,
                        '灌': calc_engine.predict(base_features.assign(Veg_Type=2)) * 365,
                        '草': calc_engine.predict(base_features.assign(Veg_Type=3)) * 365
                    })
                    results_matrix.loc[base_features['Soil_Thickness'] < 15, '树'] = -99999.0
                    best_option_idx = results_matrix.idxmax(axis=1)
                    desc_map = {'树': '乔木 (森林)', '灌': '灌木', '草': '草本/农田'}
                    df_clean['系统推荐方案'] = best_option_idx.map(desc_map)
                    df_clean['规划后固碳潜力'] = results_matrix.max(axis=1).round(4)
                    df_clean['规划后预期损益 (元)'] = df_clean.apply(
                        lambda r: calculate_carbon_assets(r['规划后固碳潜力'], r['面积 (公顷)'], DEFAULT_CARBON_PRICE)[
                            'annual_revenue'], axis=1
                    )
                    st.success("最优规划推演完成。")
                    st.dataframe(df_clean, use_container_width=True)
                    st.download_button("导出规划建议书", df_clean.to_csv(index=False).encode('utf-8-sig'),
                                       "最优生态规划建议书.csv", "text/csv")
        except Exception as e:
            st.error(f"核算异常：{str(e)}")