# utils/visualizer.py
import streamlit as st
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def render_result_chart(model, base_features):
    """
    渲染温度敏感性分析折线图 (使用 Altair)
    """
    st.markdown("#### 温度敏感性动态推演")
    st.caption("该图表展示在当前地貌与植被配置下，年度固碳潜力随年均环境温度变化的趋势。")

    temps = list(range(0, 42, 2))
    predictions = []

    for t in temps:
        temp_features = base_features.copy()
        temp_features['T'] = t

        input_df = pd.DataFrame([temp_features])
        pred = model.predict(input_df)[0]
        predictions.append(pred)

    chart_data = pd.DataFrame({
        '年均温度 (℃)': temps,  # 更改了这里的表头
        '固碳潜力 (gC/m²/yr)': predictions
    })

    chart = alt.Chart(chart_data).mark_line(
        point=alt.OverlayMarkDef(color="red", size=50)
    ).encode(
        x=alt.X('年均温度 (℃):Q', title='年均环境温度 (℃)'),  # 更新 x 轴标题
        y=alt.Y('固碳潜力 (gC/m²/yr):Q', title='年度预估固碳潜力 (gC/m²/yr)'),
        tooltip=['年均温度 (℃)', '固碳潜力 (gC/m²/yr)']
    ).properties(
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def render_shap_waterfall(model, base_features):
    """
    渲染 SHAP 特征贡献瀑布图 (带中文年度单位标识)
    """
    st.markdown("#### AI 决策过程解析 (SHAP)")
    st.caption("瀑布图展示了各项参数是如何将年度固碳均值推向最终预测结果的。红色箭头表示正向增加，蓝色表示负向减少。")

    input_df = pd.DataFrame([base_features])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    # 【必须同步更新字典】：与左侧滑块的新名称完全对齐
    shap_values.feature_names = [
        "年均温度 (℃)", "年均相对湿度 (%)", "年降水总量 (mm)", "年均太阳辐射 (W/m$^2$)",
        "坡度 (°)", "土壤厚度 (cm)", "裸岩率 (%)", "植被类型"
    ]

    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
    plt.clf()