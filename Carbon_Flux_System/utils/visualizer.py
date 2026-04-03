# utils/visualizer.py
import streamlit as st
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt


def render_result_chart(model, base_features):
    """
    渲染温度敏感性分析折线图 (使用 Altair)
    """
    st.markdown("#### 📈 温度敏感性动态推演")
    st.caption("该图表展示在当前地貌与植被配置下，固碳潜力随环境温度变化的趋势。")

    temps = list(range(0, 42, 2))
    predictions = []

    for t in temps:
        temp_features = base_features.copy()
        temp_features['T'] = t

        input_df = pd.DataFrame([temp_features])
        pred = model.predict(input_df)[0]
        predictions.append(pred)

    chart_data = pd.DataFrame({
        '温度 (℃)': temps,
        '固碳潜力 (gC/m²/yr)': predictions
    })

    chart = alt.Chart(chart_data).mark_line(
        point=alt.OverlayMarkDef(color="red", size=50)
    ).encode(
        x=alt.X('温度 (℃):Q', title='环境温度 (℃)'),
        y=alt.Y('固碳潜力 (gC/m²/yr):Q', title='预估固碳潜力 (gC/m²/yr)'),
        tooltip=['温度 (℃)', '固碳潜力 (gC/m²/yr)']
    ).properties(
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def render_shap_waterfall(model, base_features):
    """
    渲染 SHAP 特征贡献瀑布图
    """
    st.markdown("#### 🔍 AI 决策过程解析 (SHAP)")
    st.caption(
        "瀑布图展示了各项参数是如何将固碳均值（E[f(x)]）推向最终预测结果（f(x)）的。红色箭头表示正向增加固碳量，蓝色箭头表示负向减少固碳量。")

    # 转换为 DataFrame
    input_df = pd.DataFrame([base_features])

    # 初始化 TreeExplainer (适用于随机森林)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    # 构建画布
    fig = plt.figure(figsize=(10, 5))

    # 绘制瀑布图，show=False 防止 matplotlib 在终端中弹窗卡死程序
    shap.plots.waterfall(shap_values[0], show=False)

    # 在 Streamlit 中渲染 matplotlib 画布
    st.pyplot(fig)

    # 清理画布内存，防止多次点击运行导致内存泄漏
    plt.clf()