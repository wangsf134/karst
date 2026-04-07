# utils/model_handler.py
import os
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any

# ================= 1. 路径自动溯源 (消除环境依赖) =================
# 获取当前文件的绝对路径 -> utils/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上退一级找到项目根目录 -> Carbon_Flux_System/
BASE_DIR = os.path.dirname(CURRENT_DIR)
# 拼接得到引擎配置文件的标准绝对路径
MODEL_PATH = os.path.join(BASE_DIR, "models", "karst_rf_model.pkl")


@st.cache_resource(show_spinner="正在加载核心测算引擎...")
def load_model():
    """
    加载生态环境测算引擎并将其驻留在内存中。
    使用 @st.cache_resource 确保引擎在系统生命周期内仅加载一次，
    大幅减少页面刷新时的 IO 开销。
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"严重错误：缺失核心测算引擎配置文件！\n预期路径为: {MODEL_PATH}\n请确认文件已部署至 models 文件夹。")
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"核心配置文件解析失败，可能是系统环境依赖冲突。\n详情: {e}")
        st.stop()


def predict_flux(calc_engine: Any, features_dict: Dict[str, float]) -> float:
    """
    执行生态碳汇潜力核算逻辑 (包含日-年尺度转换)。

    Args:
        calc_engine: 已加载的生态核算引擎对象。
        features_dict (Dict[str, float]): 包含 8 个地貌与气象特征键值对的字典。

    Returns:
        float: 核算得到的固碳潜力年度数值 (gC/m²/yr)。
    """
    # 将字典转换为 DataFrame (底层矩阵运算所需的数据结构)
    input_df = pd.DataFrame([features_dict])

    # 执行矩阵推算并提取数组中的第一个标量结果 (此时为日均值)
    daily_result = calc_engine.predict(input_df)[0]

    # 时间尺度上译：将日均固碳量转化为年度总量
    annual_result = daily_result * 365

    return float(annual_result)