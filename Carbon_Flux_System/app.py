# app.py
import streamlit as st
import pandas as pd
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
    st.info(
        "此模块支持上传包含多个林班数据的表格，进行瞬间批量化固碳评估。为保证系统识别准确率，请务必先下载并使用标准模板。")

    # 1. 动态生成标准模板并提供下载
    # 构造一个包含两行示例数据的模板 DataFrame
    template_df = pd.DataFrame({
        '地块编号': ['Plot_001', 'Plot_002'],
        'T': [15.0, 18.5],
        'RH': [70.0, 65.0],
        'R': [10.0, 5.0],
        'Rg': [150.0, 200.0],
        'Slope': [15.0, 25.0],
        'Soil_Thickness': [10.0, 30.0],
        'Rock_Outcrop': [60.0, 40.0],
        'Veg_Type': [1, 2]  # 业务提示：1-乔木, 2-灌木, 3-草本
    })

    # 将 DataFrame 转换为 utf-8-sig 编码的 CSV，防止中文在 Excel 中打开时乱码
    template_csv = template_df.to_csv(index=False).encode('utf-8-sig')

    st.download_button(
        label="⬇️ 第一步：下载标准数据输入模板 (.csv)",
        data=template_csv,
        file_name="固碳批量测算模板.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # 2. 文件上传组件
    uploaded_file = st.file_uploader("⬆️ 第二步：上传已填写的区域调查表 (支持 .csv, .xlsx)", type=["csv", "xlsx"])

    # 3. 核心处理逻辑
    if uploaded_file is not None:
        try:
            # 根据文件后缀智能判断读取方式
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("📄 **上传数据预览 (前5行)：**")
            st.dataframe(df.head())

            # 定义模型运算绝对需要的特征列
            required_cols = ['T', 'RH', 'R', 'Rg', 'Slope', 'Soil_Thickness', 'Rock_Outcrop', 'Veg_Type']

            # 列表推导式：比对模板，找出用户上传的表格中缺失的列
            missing_cols = [col for col in required_cols if col not in df.columns]

            # 防呆设计：拦截不规范的表格
            if missing_cols:
                st.error(
                    f"❌ 数据格式校验失败！表格中缺少核心模型所需的列：{missing_cols}。请严格按照模板的表头格式上传数据。")
            else:
                # 校验通过，允许触发批量计算
                if st.button("🚀 开始进行批量反演推演", type="primary"):
                    # 使用 st.spinner 提供计算时的等待视觉反馈
                    with st.spinner("AI 底层模型正在对区域矩阵进行全速计算中..."):
                        # 剥离出特征列直接送入模型，Pandas 和 sklearn 天生支持矩阵化批量运算，速度极快
                        input_features = df[required_cols]
                        predictions = rf_model.predict(input_features)

                        # 将预测结果作为一个新列拼接到原表格最后
                        result_df = df.copy()
                        result_df['预估固碳潜力 (gC/m²/yr)'] = predictions.round(4)  # 保留4位小数

                        st.success(f"✅ 批量测算完成！系统在毫秒级时间内共处理了 {len(result_df)} 条地块数据。")
                        st.dataframe(result_df)

                        # 4. 提供结果导出功能
                        result_csv = result_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 第三步：下载带有 AI 预测结果的完整报表",
                            data=result_csv,
                            file_name="区域批量固碳测算_诊断结果.csv",
                            mime="text/csv",
                            type="primary"
                        )

        except Exception as e:
            # 捕获例如文件损坏、数据类型严重错误等未知异常
            st.error(f"读取文件或解析时发生系统错误，请检查表格内容是否包含非法字符。详细错误信息：{e}")