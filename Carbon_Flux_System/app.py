# app.py
import streamlit as st
from config import PAGE_TITLE, PAGE_ICON, PAGE_LAYOUT, VEG_MAPPING
from utils.model_handler import load_model, predict_flux
from utils.visualizer import render_result_chart, render_shap_waterfall

# ================= 1. 全局配置 =================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
st.title("碳绘喀斯特：县域固碳评估与情景模拟沙盘")
st.markdown("---")

# ================= 2. 模型预加载 =================
rf_model = load_model()

# ================= 3. 核心功能区：标签页布局 =================
# 创建两个标签页，用图标增强视觉辨识度
tab1, tab2 = st.tabs(["🎯 单点精细诊断", "📂 区域批量测算"])

# --- Tab 1: 单点精细诊断 ---
with tab1:
    st.markdown("### 🔍 实时地块模拟")
    st.info("通过调节左侧参数，实时观察特定地理配置下的固碳潜力及驱动因子。")

    # 采用两列布局，左侧放参数，右侧放诊断结果
    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("⚙️ 环境参数调节")
        # 将原侧边栏的滑块移入 Tab 内部，实现空间复用
        with st.expander("1. 气象条件", expanded=True):
            T = st.slider("温度 (℃)", -10.0, 40.0, 15.0)
            RH = st.slider("相对湿度 (%)", 0.0, 100.0, 70.0)
            R = st.slider("降水量 (mm/h)", 0.0, 100.0, 10.0)
            Rg = st.slider("太阳辐射 (W/m²)", 0.0, 1000.0, 150.0)

        with st.expander("2. 地表特征", expanded=True):
            Slope = st.slider("坡度 (°)", 0.0, 60.0, 15.0)
            Soil_Thickness = st.slider("土壤厚度 (cm)", 0.0, 100.0, 10.0)
            Rock_Outcrop = st.slider("裸岩率 (%)", 0.0, 100.0, 60.0)

        with st.expander("3. 人为干预", expanded=True):
            veg_choice = st.selectbox("目标植被类型", list(VEG_MAPPING.keys()))
            Veg_Type = VEG_MAPPING[veg_choice]

        # 封装特征字典
        features = {
            'T': T, 'RH': RH, 'R': R, 'Rg': Rg,
            'Slope': Slope, 'Soil_Thickness': Soil_Thickness,
            'Rock_Outcrop': Rock_Outcrop, 'Veg_Type': Veg_Type
        }

        run_btn = st.button("开始 AI 模拟评估", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 诊断报告输出")
        if run_btn:
            # 1. 执行预测
            prediction = predict_flux(rf_model, features)
            st.metric(label="预估固碳潜力 (gC/m²/yr)", value=f"{prediction:.4f}")

            # 2. 逻辑判定告警
            if Soil_Thickness < 15 and Veg_Type == 1:
                st.error("🚨 严重预警：当前地块属于重度石漠化区（土层厚度极浅）。种植乔木存在极高死亡风险！")
            elif prediction > 0:
                st.success("✅ 评估通过：当前配置具有正向碳汇效益。")
            else:
                st.warning("⚠️ 提示：当前配置下呈排碳状态，请注意后续生态管护。")

            # 3. 详细可视化展开
            with st.container():
                render_result_chart(rf_model, features)
                st.markdown("---")
                render_shap_waterfall(rf_model, features)
        else:
            st.info("💡 请在左侧调整参数并点击“开始模拟”按钮查看结果。")

# --- Tab 2: 区域批量测算 ---
with tab2:
    st.markdown("### 📂 大规模区域测算")
    st.write("此模块支持上传 CSV/Excel 格式的区域普查数据，进行自动批量化固碳评估。")

    # 预留占位
    st.warning("🚧 批量处理模块正在开发中，敬请期待下一版本扩展。")

    # 示意性的上传组件
    uploaded_file = st.file_uploader("上传区域调查表 (支持 .csv, .xlsx)", type=["csv", "xlsx"])