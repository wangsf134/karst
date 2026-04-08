# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

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

# CSS 注入：包含组件汉化、全局字体优化，以及【核心魔法：Tab 视觉伪装】
st.markdown(
    """
    <style>
    /* ====================================================
       --- 1. 深度汉化文件上传组件 (彻底消灭英文) ---
       ==================================================== */
    
    /* 汉化主提示：Drag and drop file here */
    [data-testid="stFileUploadDropzone"] div > span {
        font-size: 0px !important; /* 将原英文缩至 0 */
    }
    [data-testid="stFileUploadDropzone"] div > span::after {
        content: "请将数据表格拖拽至此处" !important; /* 注入中文 */
        font-size: 16px !important;
        font-weight: bold !important;
        color: #31333F !important;
        display: block !important;
    }

    /* 汉化副提示：Limit 200MB... */
    [data-testid="stFileUploadDropzone"] small {
        font-size: 0px !important;
    }
    [data-testid="stFileUploadDropzone"] small::after {
        content: "单文件大小限制 200MB • 支持 CSV, XLSX" !important;
        font-size: 13px !important;
        color: #888888 !important;
        display: block !important;
        margin-top: 4px !important;
    }

    /* 汉化按钮：Browse files */
    [data-testid="stFileUploadDropzone"] button {
        font-size: 0px !important;
    }
    [data-testid="stFileUploadDropzone"] button::after {
        content: "浏览本地文件" !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
    }

    /* ====================================================
       --- 2. 优化全局字体 ---
       ==================================================== */
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", sans-serif;
    }

    /* ====================================================
       --- 3. 核心魔法：将单选按钮伪装成原生 Tab 标签页 ---
       ==================================================== */
    /* 隐藏单选框的丑陋圆圈 */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    /* 调整布局，让选项水平排列并拉开间距，底部加灰线 */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 2rem !important;
        border-bottom: 1px solid #f0f2f6; 
        padding-bottom: 0px;
        margin-bottom: 1.5rem;
    }
    /* 美化文字，模拟原生 Tab 未选中状态 */
    div[role="radiogroup"] label {
        cursor: pointer;
        padding: 0.5rem 0.5rem;
        margin: 0;
        border-bottom: 3px solid transparent; 
        transition: all 0.2s ease;
    }
    div[role="radiogroup"] label p {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #7a7a7a;
        margin: 0;
    }
    /* 鼠标悬停变色 */
    div[role="radiogroup"] label:hover p {
        color: #ff4b4b;
    }
    /* 核心：选中状态加上红线和红色文字 */
    div[role="radiogroup"] label:has(input:checked) {
        border-bottom: 3px solid #ff4b4b !important; 
    }
    div[role="radiogroup"] label:has(input:checked) p {
        color: #ff4b4b !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("碳绘喀斯特：县域固碳评估与情景模拟沙盘")
st.markdown("---")

# ================= 2. 核心核算引擎加载 =================
calc_engine = load_model()
logger.info("Core calculation engine successfully initialized in memory.")

# ================= 3. 功能模块：带状态记忆的伪装 Tab 布局 =================
# 初始化全局状态
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "单点精细诊断"

# 这里的 Radio 按钮在前端已经被 CSS 彻底伪装成了漂亮的 Tab
selected_tab = st.radio(
    label="导航菜单",
    options=["单点精细诊断", "区域批量测算"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab"
)

# ---------------------------------------------------------
# --- 模块 1: 单点精细诊断 (情景模拟) -------------------------
# ---------------------------------------------------------
if selected_tab == "单点精细诊断":
    st.markdown("### 实时地块模拟")
    st.info("说明：通过调节下方环境参数，系统将基于生态环境动力学特征矩阵，实时核算特定地貌配置下的年度固碳潜力。")

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
        Soil_Thickness = st.slider("土壤厚度 (cm)", 0.0, 100.0, 10.0)
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
            st.metric(label="年度 CO2e 核证量", value=f"{assets['annual_co2e_tons']} tCO2e/yr")
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
        st.caption("系统已锁定环境不可变参数，通过全路径矩阵扫描输出综合效益最优配置建议。")

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

# ---------------------------------------------------------
# --- 模块 2: 区域批量测算 (政务大数据处理) -------------------
# ---------------------------------------------------------
elif selected_tab == "区域批量测算":
    st.markdown("### 大规模区域测算")
    st.info("说明：本模块支持批量上传区域林班调查数据，系统将自动执行合规性校验并输出核算报表。")

    st.markdown("#### 填表规范说明")
    st.warning("""
    **核心注意：** 由于底层矩阵运算规范，**【植被类型】**字段请务必填写对应的**数字代码**，请勿填写中文：
    * **输入 1** 代表 **乔木 (森林)**
    * **输入 2** 代表 **灌木**
    * **输入 3** 代表 **草本/农田**
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
        mime="text/csv"
    )

    st.markdown("---")

    st.markdown("##### 请选择或拖拽区域调查数据表至下方区域")
    uploaded_file = st.file_uploader(
        label="数据上传区",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

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
                cleaned_columns = []
                for col in df_input.columns:
                    c = str(col).strip()
                    c = c.replace("W/m2", "W/m²").replace("W/m^2", "W/m²")
                    c = c.replace("（", "(").replace("）", ")")
                    cleaned_columns.append(c)

                df_input.columns = cleaned_columns

                df_clean = df_input.dropna(how='all')
                required_cols = [
                    '年均温度 (℃)', '年均相对湿度 (%)', '年降水总量 (mm)', '年均太阳辐射 (W/m²)',
                    '坡度 (°)', '土壤厚度 (cm)', '裸岩率 (%)', '面积 (公顷)', '植被类型'
                ]
                missing_cols = [col for col in required_cols if col not in df_clean.columns]

                if missing_cols:
                    st.error(f"校验失败：上传数据缺失核心业务字段：{missing_cols}")
                    with st.expander("点击查看系统识别到的表头列表（供排查对照）"):
                        st.write(list(df_clean.columns))
                    st.stop()

                numeric_fields = [c for c in required_cols if c != '植被类型']
                for col in numeric_fields:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                df_clean = df_clean.dropna(subset=required_cols)
                cleansed_count = len(df_input) - len(df_clean)
                if cleansed_count > 0:
                    st.warning(f"审计提示：已自动剔除 {cleansed_count} 条格式异常记录。")
                else:
                    st.success("校验通过：数据格式完全符合业务逻辑。")

            st.write("**核算前数据预览 (前5行)：**")
            st.dataframe(df_clean.head())

            col_btn1, col_btn2 = st.columns(2)
            mapping_dict = {
                '年均温度 (℃)': 'T', '年均相对湿度 (%)': 'RH',
                '年降水总量 (mm)': 'R', '年均太阳辐射 (W/m²)': 'Rg',
                '坡度 (°)': 'Slope', '土壤厚度 (cm)': 'Soil_Thickness',
                '裸岩率 (%)': 'Rock_Outcrop', '植被类型': 'Veg_Type'
            }

            if col_btn1.button("执行现状资产核算", type="primary", use_container_width=True):
                with st.spinner("现状核算中..."):
                    X_matrix = df_clean[list(mapping_dict.keys())].rename(columns=mapping_dict)
                    results = calc_engine.predict(X_matrix) * 365
                    df_clean['年度核算固碳潜力'] = results.round(4)
                    df_clean['年度综合损益 (元)'] = df_clean.apply(
                        lambda r:
                        calculate_carbon_assets(r['年度核算固碳潜力'], r['面积 (公顷)'], DEFAULT_CARBON_PRICE)[
                            'annual_revenue'], axis=1
                    )
                    st.dataframe(df_clean)
                    st.download_button("导出核算报表", df_clean.to_csv(index=False).encode('utf-8-sig'),
                                       "现状资产核算报表.csv", "text/csv")

            if col_btn2.button("执行最优规划推演", type="secondary", use_container_width=True):
                with st.spinner("全场景生态推演中..."):
                    base_features = df_clean[list(mapping_dict.keys())].rename(columns=mapping_dict).drop(
                        columns=['Veg_Type'])
                    X_tree = base_features.copy();
                    X_tree['Veg_Type'] = 1
                    X_shrub = base_features.copy();
                    X_shrub['Veg_Type'] = 2
                    X_grass = base_features.copy();
                    X_grass['Veg_Type'] = 3

                    p_tree = calc_engine.predict(X_tree) * 365
                    p_shrub = calc_engine.predict(X_shrub) * 365
                    p_grass = calc_engine.predict(X_grass) * 365

                    results_matrix = pd.DataFrame({'树': p_tree, '灌': p_shrub, '草': p_grass})
                    results_matrix.loc[base_features['Soil_Thickness'] < 15, '树'] = -99999.0

                    best_option_idx = results_matrix.idxmax(axis=1)
                    best_potential = results_matrix.max(axis=1)
                    desc_map = {'树': '乔木 (森林)', '灌': '灌木', '草': '草本/农田'}

                    df_clean['系统推荐方案'] = best_option_idx.map(desc_map)
                    df_clean['规划后固碳潜力'] = best_potential.round(4)
                    df_clean['规划后预期损益 (元)'] = df_clean.apply(
                        lambda r: calculate_carbon_assets(r['规划后固碳潜力'], r['面积 (公顷)'], DEFAULT_CARBON_PRICE)[
                            'annual_revenue'], axis=1
                    )
                    st.success("推演完成：已识别最优配置方案。")
                    st.dataframe(df_clean)
                    st.download_button("导出规划建议书", df_clean.to_csv(index=False).encode('utf-8-sig'),
                                       "最优生态规划建议书.csv", "text/csv")

        except Exception as e:
            logger.error(f"System Error: {e}")
            error_str = str(e)
            if "codec can't decode" in error_str:
                st.error(
                    "核算异常：文件编码格式不匹配。请确保 CSV 文件使用 UTF-8 或 GBK 编码（建议将 Excel 保存为 .xlsx 格式再上传）。")
            elif "column" in error_str.lower():
                st.error("核算异常：上传的表格缺少必要的表头或列名不正确，请对比模板文件。")
            else:
                st.error(f"系统核算异常：{error_str}")