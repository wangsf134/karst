# utils/model_handler.py
"""
Carbon_Flux_System.utils.model_handler
核心生态测算引擎的加载与预测推演模块。
负责从本地模型文件加载预训练的随机森林模型，并基于喀斯特地貌与气象特征
执行年度固碳潜力核算。模块内置多项生态保护性约束逻辑，确保政务决策的安全边界。
"""

import os
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any

# ================= 1. 路径自动溯源（消除环境依赖） =================
# 获取当前文件的绝对路径，并向上回溯至项目根目录，确保在任何部署环境下都能准确找到模型文件。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上退一级找到项目根目录 -> Carbon_Flux_System/
BASE_DIR = os.path.dirname(CURRENT_DIR)
# 拼接得到引擎配置文件的标准绝对路径
MODEL_PATH = os.path.join(BASE_DIR, "models", "karst_rf_model.pkl")

# ================= 2. 生态阈值与拦截参数 =================
# SAFE_FLOOR：全系统土壤厚度的绝对安全底线（10 cm），低于此值系统将强制修正输入。
SAFE_FLOOR = 10.0  # 全系统土壤厚度绝对底线 (cm)
# TREE_SOIL_MIN：乔木（Veg_Type=1）能够正常生长的最低土壤厚度。
TREE_SOIL_MIN = 15.0  # 乔木生存最低土厚 (cm)
# OPTIMAL_TEMP：呼吸作用激增的临界温度，高于此值将触发排放惩罚。
OPTIMAL_TEMP = 15.0  # 呼吸作用激增的临界温度 (℃)


@st.cache_resource(show_spinner="正在加载核心测算引擎...")
def load_model():
    """
    加载随机森林生态测算模型，并通过 Streamlit 缓存机制驻留在内存中。

    该函数使用 @st.cache_resource 装饰器，确保模型在系统生命周期内仅加载一次，
    显著降低重复加载带来的磁盘 IO 与时间开销。

    Returns:
        已加载的 sklearn 模型对象（通常为 RandomForestRegressor 实例）。

    Raises:
        streamlit.error: 若模型文件不存在或加载失败，将直接中断程序并显示明确错误信息。

    Notes:
        - 模型文件路径基于项目根目录自动推导，与部署环境无关。
        - 若模型文件格式异常或 sklearn 版本不兼容，错误信息将包含详细异常描述，便于排查。
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
    执行喀斯特生态碳汇潜力核算，并应用多项生态保护约束。

    该函数是模型推理的核心入口，输入一组包含 8 个地貌与气象特征的字典，
    输出经过生态约束修正后的年度固碳潜力（gC/m²/yr）。

    业务逻辑包含两个关键拦截层：
    1) 土壤厚度的物理强制底线（10 cm），防止极端浅土误算；
    2) 以 SHAP 分析为依据的生态约束惩罚，针对缺土高温下乔木的错误选择进行负向修正。

    Args:
        calc_engine (Any): 已加载的生态核算引擎对象（随机森林模型）。
        features_dict (Dict[str, float]): 包含 8 个特征键值对的字典，必须包含以下字段：
            - T (年均温度, ℃)
            - RH (年均相对湿度, %)
            - R (年降水量, mm)
            - Rg (年均太阳辐射, W/m²)
            - Slope (坡度, °)
            - Soil_Thickness (土壤厚度, cm)
            - Rock_Outcrop (裸岩率, %)
            - Veg_Type (植被类型编码: 1-乔木, 2-灌木, 3-草本/农田)

    Returns:
        float: 核算并修正后的年度固碳潜力数值（gC/m²/yr）。

    Notes:
        - 默认采用模型输出的日均值乘以 365 天转换为年度值。
        - 当土壤厚度 < 15 cm 且选择乔木（Veg_Type=1）时，触发衰减因子与高温呼吸惩罚，
          模拟植被退化状态与土壤碳释放效应，防止基层因估算偏差作出错误种植决策。
        - 惩罚机制引入了 200 基础死亡惩罚项，用于表征极端贫瘠条件下的植被基础代谢消耗。
    """

    # ==========================================
    # 核心拦截 1：物理强制对齐到 10cm 底线
    # ==========================================
    soil = features_dict.get('Soil_Thickness', SAFE_FLOOR)
    if soil < SAFE_FLOOR:
        # 喀斯特地区土层厚度是固碳能力的决定性变量，低于 10cm 时几乎不具备植被固碳条件。
        # 为确保核算结果不出现脱离实际的虚高值，强制将输入修正为安全底线。
        soil = SAFE_FLOOR
        features_dict['Soil_Thickness'] = SAFE_FLOOR  # 修正字典值，确保模型输入的也是 10cm

    # 将字典转换为 DataFrame（模型要求二维输入）
    input_df = pd.DataFrame([features_dict])

    # 执行矩阵推算（此时为日均值）
    daily_result = calc_engine.predict(input_df)[0]

    # 时间尺度转换：日均 -> 年度
    annual_result = daily_result * 365.0

    # 获取辅助干预特征
    veg = features_dict.get('Veg_Type', 1)
    temp = features_dict.get('T', 15.0)

    # ==========================================
    # 核心拦截 2：XAI 驱动的生态约束惩罚（针对 10-15cm 乔木）
    # ==========================================
    # 若用户选择了乔木（Veg_Type=1）但土壤厚度不足以支撑其健康生长，
    # 系统将基于生态学先验知识强制进行潜力衰减与呼吸排放惩罚，
    # 避免浅土造林的规划建议给基层带来财政损失和生态风险。
    if veg == 1 and soil < TREE_SOIL_MIN:
        # 1. 基础固碳能力衰减（模拟植被退化状态）
        # 使用平方衰减函数，使得固碳潜力在接近 10cm 时加速下降，
        # 反映根系伸展受限、水分养分供给不足等现实约束。
        decay_factor = (soil / TREE_SOIL_MIN) ** 2
        annual_result *= decay_factor

        # 2. 引入呼吸作用反转惩罚（核心：解决高温异常上扬 Bug）
        # 当植被因缺土处于不健康状态时，高温（>15℃）会显著加速土壤微生物呼吸和残体分解，
        # 导致生态系统从碳汇转变为碳源。惩罚项随温度差呈现指数级增长。
        if temp > OPTIMAL_TEMP:
            # 惩罚项随温度差呈指数级增长，模拟土壤碳释放的加速效应。
            resp_penalty = ((temp - OPTIMAL_TEMP) ** 1.2) * 50.0
            # 从结果中扣除排放惩罚，并叠加 200 gC/m²/yr 的基础死亡损耗，
            # 用于模拟极端贫瘠条件下植被代谢的基本消耗（如细根周转、凋落物分解）。
            annual_result = annual_result - resp_penalty - 200.0  # 叠加 200 的基础死亡惩罚项

    return float(annual_result)