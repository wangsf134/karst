# utils/visualizer.py
"""
Carbon_Flux_System.utils.visualizer
数据可视化与模型可解释性图表渲染模块。
负责生成面向政务决策的温度敏感性推演图、SHAP 边际贡献瀑布图等关键图表。
模块提供了中文环境自适应的字体加载方案，保障跨平台图表渲染的一致性。
"""

import streamlit as st
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os


# ==========================================
# 【核心修复】自动识别后缀并绝对路径加载字体
# ==========================================
def init_chinese_font():
    """
    初始化中文字体环境，确保 Matplotlib 图表在任意部署环境下正确显示中文。

    该函数自动搜索项目根目录下的 SimHei 字体文件（支持 .ttf 和 .ttc 后缀），
    动态注册并设置为全局默认字体，同时配置负号正常显示。

    Returns:
        bool: 字体加载成功返回 True，否则返回 False 并通过 Streamlit 输出错误提示。

    Notes:
        - 针对云端部署（如 Streamlit Cloud）和本地开发环境，均能自适应查找字体路径。
        - 若找不到字体文件，会在 Streamlit 界面中展示当前根目录的文件列表，辅助运维人员排查。
        - 调用 Matplotlib 绘图函数前，建议先执行此函数进行环境检查。
    """
    # 定位项目根目录（假设此文件在 utils/ 文件夹下，向上走一级即为根目录）
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
            # 1. 动态注册字体文件，将字体路径加入 matplotlib 字体管理器
            fm.fontManager.addfont(target_font)
            # 2. 获取该文件在系统内部真实的字体名称（如 'SimHei'）
            prop = fm.FontProperties(fname=target_font)
            # 3. 设置全局字体，确保后续绘图默认使用该中文字体
            plt.rcParams['font.family'] = prop.get_name()
            # 4. 解决负号显示为方块的问题，使负号正常渲染
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception as e:
            st.warning(f"字体注册失败: {e}")
            return False
    else:
        # 报错时提示当前目录下到底有哪些文件，方便排查字体缺失原因
        files_in_dir = os.listdir(root_dir)
        st.error(f"找不到 SimHei 字体文件！\n预期路径: {root_dir}\n当前根目录文件列表: {files_in_dir}")
        return False


# ==========================================
# 业务绘图函数
# ==========================================

def render_result_chart(calc_engine, base_features):
    """
    渲染温度敏感性动态推演折线图（基于 Altair 实现）。

    该图表展示在当前地貌与植被配置下，年度固碳潜力随年均环境温度变化的模拟趋势，
    为决策者提供“温度扰动风险”的直观感知。

    Args:
        calc_engine: 已加载的随机森林模型对象。
        base_features (dict): 包含当前配置的所有特征值（温度项将被替换）。

    Notes:
        - 温度推演区间设置为 -10℃ 至 40℃，步长 2℃，基本覆盖喀斯特地区可能气候情景。
        - 使用 Altair 进行交互式渲染，支持缩放和悬停提示，便于非技术人员探索数据。
    """
    init_chinese_font()  # 确保绘图前加载中文字体

    st.markdown("#### 温度敏感性动态推演")
    st.caption("该图表展示在当前地貌与植被配置下，年度固碳潜力随年均环境温度变化的模拟趋势。")

    # 定义温度推演区间并批量生成预测数据，用于模拟温度对固碳的影响
    temps = list(range(-10, 42, 2))
    temp_df_list = [base_features.copy() | {'T': t} for t in temps]
    input_df = pd.DataFrame(temp_df_list)

    # 执行预测并转换尺度（从日均值到年度值）
    daily_preds = calc_engine.predict(input_df)
    annual_predictions = [p * 365 for p in daily_preds]

    chart_data = pd.DataFrame({
        'Temp_C': temps,
        'Carbon_Sink': annual_predictions
    })

    # 构建动态交互图表（Altair 对中文支持较好，只要浏览器有字体即可）
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
    渲染核算因子边际贡献解析瀑布图（基于 SHAP 与 Matplotlib）。

    该图表利用 SHAP 可解释性方法，展示每个特征对最终固碳预测值的正向/负向贡献量，
    帮助基层工作人员理解模型决策依据，辅助生态规划的科学化沟通。

    Args:
        calc_engine: 已加载的随机森林模型对象（需为树模型，SHAP 支持 TreeExplainer）。
        base_features (dict): 当前地块的环境特征配置。

    Notes:
        - 特征名称已映射为中文全称，便于直接用于汇报和培训材料。
        - 绘图前显式设置背景色为白色，避免在暗色主题或透明背景下图表元素不可见。
        - 使用完毕立即清理 Matplotlib 图形内存，防止云端环境因内存溢出导致服务中断。
        - 若字体加载失败，函数将直接终止，避免生成乱码图表。
    """
    st.markdown("#### 核算因子边际贡献解析")

    # 核心：绘图前再次强制刷新字体设置，若失败则停止运行，确保图表不出现乱码
    if not init_chinese_font():
        st.stop()

    input_df = pd.DataFrame([base_features])

    try:
        # 使用 SHAP 的 TreeExplainer 解析随机森林模型
        explainer = shap.TreeExplainer(calc_engine)
        shap_values = explainer(input_df)

        # 映射特征名称，使用政务沟通中更易理解的术语
        feature_display_names = [
            "年均温度 (C)", "相对湿度 (%)", "年降水量 (mm)", "太阳辐射 (W/m2)",
            "坡度 (Deg)", "土壤厚度 (cm)", "裸岩率 (%)", "植被类型"
        ]

        # 封装为 SHAP 专用的 Explanation 对象，便于瀑布图绘制
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

        # 清理内存，防止云端 OOM（内存溢出）和下次绘图异常
        plt.clf()
        plt.close(fig)

    except Exception as e:
        st.error(f"图表渲染异常: {str(e)}")