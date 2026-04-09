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

# ================= 2. 生态阈值与拦截参数 =================
SAFE_FLOOR = 10.0  # 全系统土壤厚度绝对底线 (cm)
TREE_SOIL_MIN = 15.0  # 乔木生存最低土厚 (cm)
OPTIMAL_TEMP = 15.0  # 呼吸作用激增的临界温度 (℃)


@st.cache_resource(show_spinner="正在加载核心测算引擎...")
def load_model():
    """
    加载生态环境测算引擎并将其驻留在内存中。
    使用 @st.cache_resource 确保引擎在系统生命周期内仅加载一次。
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"严重错误：缺失核心测算引擎配置文件！\n预期路径为: {MODEL_PATH}")
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"核心配置文件解析失败，可能是系统环境依赖冲突。\n详情: {e}")
        st.stop()


def predict_flux(calc_engine: Any, features_dict: Dict[str, float]) -> float:
    """
    执行生态碳汇潜力核算逻辑 (包含 10cm 强制拦截及生态约束惩罚)。

    Args:
        calc_engine: 已加载的生态核算引擎对象。
        features_dict (Dict[str, float]): 包含 8 个地貌与气象特征键值对的字典。

    Returns:
        float: 核算得到的固碳潜力年度数值 (gC/m²/yr)。
    """

    # ==========================================
    # 核心拦截 1：物理强制对齐到 10cm 底线
    # ==========================================
    soil = features_dict.get('Soil_Thickness', SAFE_FLOOR)
    if soil < SAFE_FLOOR:
        soil = SAFE_FLOOR
        features_dict['Soil_Thickness'] = SAFE_FLOOR  # 修正字典值，确保模型输入的也是 10cm

    # 将字典转换为 DataFrame
    input_df = pd.DataFrame([features_dict])

    # 执行矩阵推算 (此时为日均值)
    daily_result = calc_engine.predict(input_df)[0]

    # 时间尺度转换：日均 -> 年度
    annual_result = daily_result * 365.0

    # 获取辅助干预特征
    veg = features_dict.get('Veg_Type', 1)
    temp = features_dict.get('T', 15.0)

    # ==========================================
    # 核心拦截 2：XAI 驱动的生态约束惩罚 (针对 10-15cm 乔木)
    # ==========================================
    # 如果用户选择的是乔木 (Veg_Type=1)，且土层厚度不足以支撑其健康生长
    if veg == 1 and soil < TREE_SOIL_MIN:
        # 1. 基础固碳能力衰减 (模拟植被退化状态)
        # 使用平方衰减，让数值在接近 10cm 时下降得更明显
        decay_factor = (soil / TREE_SOIL_MIN) ** 2
        annual_result *= decay_factor

        # 2. 引入呼吸作用反转惩罚 (核心：解决高温异常上扬 Bug)
        # 当植被因缺土处于不健康状态时，高温 ( > 15℃ ) 会显著加速微生物呼吸和残体分解
        if temp > OPTIMAL_TEMP:
            # 惩罚项随温度差呈指数级增长
            resp_penalty = ((temp - OPTIMAL_TEMP) ** 1.2) * 50.0
            # 从结果中扣除排放，模拟碳源效应
            annual_result = annual_result - resp_penalty - 200.0  # 叠加 200 的基础死亡惩罚项

    return float(annual_result)