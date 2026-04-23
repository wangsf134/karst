import streamlit as st
import pandas as pd
import altair as alt
import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


# ==========================================
# 自动查找并设置中文字体，解决云端部署中文乱码
# ==========================================
def fix_chinese_display():
    """自动查找并设置中文字体，解决云端部署时的中文乱码问题"""
    # 定义可能的字体文件名（放在项目根目录或 fonts 子目录）
    font_files = [
        'SimHei.ttf', 'simhei.ttf', 'SimSun.ttf', 'simsun.ttf',
        'MicrosoftYaHei.ttf', 'msyh.ttf', 'PingFangSC-Regular.ttf',
        'fonts/SimHei.ttf', 'fonts/simhei.ttf'
    ]

    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent

    # 查找存在的字体文件
    font_path = None
    for font_file in font_files:
        full_path = current_dir / font_file
        if full_path.exists():
            font_path = str(full_path.resolve())
            break

    if font_path:
        # 创建字体属性
        font_prop = mpl.font_manager.FontProperties(fname=font_path)

        # 全局设置字体
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 设置全局字体（更彻底）
        mpl.rcParams['font.sans-serif'] = [font_prop.get_name()]
        mpl.rcParams['axes.unicode_minus'] = False

        return True
    else:
        st.warning("未找到中文字体文件，请将 SimHei.ttf 等字体文件放在项目目录中以避免图表乱码。")
        # 退而求其次，尝试使用系统可能存在的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return False


# 在程序最开始调用这个函数进行全局设置
fix_chinese_display()


# ==========================================
# 业务绘图函数
# ==========================================

def render_result_chart(calc_engine, base_features):
    """
    渲染温度敏感性动态推演折线图 (使用 Altair)
    """
    st.markdown("#### 温度敏感性动态推演")
    st.caption("该图表展示在当前地貌与植被配置下，年度固碳潜力随年均环境温度变化的模拟趋势。")

    # 定义温度推演区间并批量生成预测数据
    temps = list(range(-10, 42, 2))
    temp_df_list = [base_features.copy() | {'T': t} for t in temps]
    input_df = pd.DataFrame(temp_df_list)

    # 执行预测并转换尺度
    daily_preds = calc_engine.predict(input_df)
    annual_predictions = [p * 365 for p in daily_preds]

    chart_data = pd.DataFrame({
        'Temp_C': temps,
        'Carbon_Sink': annual_predictions
    })

    # 构建动态交互图表 (Altair 对中文支持较好，只要浏览器有字体即可)
    chart = alt.Chart(chart_data).mark_line(
        point=alt.OverlayMarkDef(color="red", size=50)
    ).encode(
        x=alt.X('Temp_C:Q', title='环境温度 (C)'),
        y=alt.Y('Carbon_Sink:Q', title='固碳潜力 (gC/m2/yr)'),
        tooltip=['Temp_C', 'Carbon_Sink']
    ).properties(height=400).interactive()

    st.altair_chart(chart, use_container_width=True)


def render_shap_waterfall(calc_engine, base_features):
    """
    渲染核算因子边际贡献解析瀑布图 (使用 SHAP + Matplotlib)
    """
    st.markdown("#### 核算因子边际贡献解析")

    input_df = pd.DataFrame([base_features])

    try:
        explainer = shap.TreeExplainer(calc_engine)
        shap_values = explainer(input_df)

        # 映射特征名称，彻底移除了易报错的特殊符号
        feature_display_names = [
            "年均温度 (C)", "相对湿度 (%)", "年降水量 (mm)", "太阳辐射 (W/m2)",
            "坡度 (Deg)", "土壤厚度 (cm)", "裸岩率 (%)", "植被类型"
        ]

        # 封装为 SHAP 专用的 Explanation 对象
        exp = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=input_df.iloc[0].values,
            feature_names=feature_display_names
        )

        plt.figure(figsize=(10, 5))
        # 显式设置背景色为白色，防止云端出现透明底导致黑屏
        plt.gcf().set_facecolor('white')

        # 调用 SHAP 绘图
        shap.plots.waterfall(exp, show=False)

        # 捕获并渲染
        fig = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig)

        # 清理内存，防止云端 OOM (内存溢出) 和下次绘图异常
        plt.clf()
        plt.close(fig)

    except Exception as e:
        st.error(f"图表渲染异常: {str(e)}")