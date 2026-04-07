# utils/visualizer.py
import streamlit as st
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt

# 设置系统绘图引擎支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def render_result_chart(calc_engine, base_features):
    """
    渲染温度敏感性动态推演折线图 (使用 Altair)
    """
    st.markdown("#### 温度敏感性动态推演")
    st.caption("该图表展示在当前地貌与植被配置下，年度固碳潜力随年均环境温度变化的模拟趋势。")

    # 定义温度推演区间
    temps = list(range(-10, 42, 2))
    annual_predictions = []

    for t in temps:
        # 复制当前地块特征并替换温度变量
        temp_features = base_features.copy()
        temp_features['T'] = t

        input_df = pd.DataFrame([temp_features])

        # 获取底层引擎输出的日均值并进行年度尺度上译 (* 365)
        daily_pred = calc_engine.predict(input_df)[0]
        annual_predictions.append(daily_pred * 365)

    chart_data = pd.DataFrame({
        '年均温度 (℃)': temps,
        '年度固碳潜力 (gC/m²/yr)': annual_predictions
    })

    # 构建动态交互图表
    chart = alt.Chart(chart_data).mark_line(
        point=alt.OverlayMarkDef(color="red", size=50)
    ).encode(
        x=alt.X('年均温度 (℃):Q', title='年均环境温度 (℃)'),
        y=alt.Y('年度固碳潜力 (gC/m²/yr):Q', title='年度核算固碳潜力 (gC/m²/yr)'),
        tooltip=['年均温度 (℃)', '年度固碳潜力 (gC/m²/yr)']
    ).properties(
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def render_shap_waterfall(calc_engine, base_features):
    """
    渲染核算因子边际贡献解析瀑布图
    """
    st.markdown("#### 核算因子边际贡献解析")
    st.caption(
        "注：底层核算引擎基于日均尺度运行。本图表展示了各项环境因子对【日均固碳量 (gC/m²/day)】的边际贡献权重，红色表示正向促进，蓝色表示负向抑制。")

    input_df = pd.DataFrame([base_features])

    # 初始化贡献度解释器
    explainer = shap.TreeExplainer(calc_engine)
    shap_values = explainer(input_df)

    # 因子名称与前端业务逻辑完全对齐
    shap_values.feature_names = [
        "年均温度 (℃)", "年均相对湿度 (%)", "年降水总量 (mm)", "年均太阳辐射 (W/m$^2$)",
        "坡度 (°)", "土壤厚度 (cm)", "裸岩率 (%)", "植被类型"
    ]

    # 执行绘图
    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
    plt.clf()