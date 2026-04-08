import streamlit as st
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os


# ==========================================
# 【核心修复】自动识别后缀并加载字体
# ==========================================
def init_chinese_font():
    """
    针对云端和本地环境的通用字体加载方案
    支持自动搜索 .ttf 和 .ttc 后缀
    """
    # 定位项目根目录 (假设此文件在 utils/ 文件夹下，向上走一级即为根目录)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    # 自动搜索可能的 SimHei 文件名（匹配大小写及后缀）
    target_font = None
    possible_names = ["SimHei", "simhei"]
    possible_exts = [".ttf", ".ttc", ".TTF", ".TTC"]

    for name in possible_names:
        for ext in possible_exts:
            test_path = os.path.join(root_dir, f"{name}{ext}")
            if os.path.exists(test_path):
                target_font = test_path
                break
        if target_font: break

    if target_font:
        try:
            # 1. 动态注册字体文件
            fm.fontManager.addfont(target_font)
            # 2. 获取该文件在系统内部真实的字体名称 (如 'SimHei')
            prop = fm.FontProperties(fname=target_font)
            # 3. 设置全局字体
            plt.rcParams['font.family'] = prop.get_name()
            # 4. 解决负号显示为方块的问题
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception as e:
            st.warning(f"字体注册失败: {e}")
            return False
    else:
        # 报错时提示当前目录下到底有哪些文件，方便排查
        files_in_dir = os.listdir(root_dir)
        st.error(f"找不到 SimHei 字体文件！\n预期路径: {root_dir}\n当前根目录文件列表: {files_in_dir}")
        return False


# ==========================================
# 业务绘图函数
# ==========================================

def render_result_chart(calc_engine, base_features):
    """
    渲染温度敏感性动态推演折线图 (使用 Altair)
    """
    init_chinese_font()  # 确保绘图前加载字体

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

    # 核心：绘图前再次强制刷新字体设置，若失败则停止运行
    if not init_chinese_font():
        st.stop()

    input_df = pd.DataFrame([base_features])

    try:
        explainer = shap.TreeExplainer(calc_engine)
        shap_values = explainer(input_df)

        # 映射特征名称
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
        # 显式设置背景色，防止云端黑屏
        plt.gcf().set_facecolor('white')

        # 调用 SHAP 绘图
        shap.plots.waterfall(exp, show=False)

        # 捕获并渲染
        fig = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig)

        # 清理内存，防止云端 OOM (内存溢出)
        plt.clf()
        plt.close(fig)

    except Exception as e:
        st.error(f"图表渲染异常: {str(e)}")