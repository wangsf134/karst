# utils/model_handler.py
import os
import joblib
import pandas as pd
import streamlit as st

# 动态获取模型路径 (彻底消灭硬编码)
# os.path.abspath(__file__) 获取当前 model_handler.py 的路径
# os.path.dirname 向上退两级，找到 Carbon_Flux_System 根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "karst_rf_model.pkl")

@st.cache_resource
def load_model():
    """加载并缓存机器学习模型"""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"模型加载失败！\n尝试读取的路径: {MODEL_PATH}\n报错信息: {e}")
        st.stop()

def predict_flux(model, features_dict):
    """
    封装预测逻辑，将前端传来的字典转为模型需要的 DataFrame
    """
    input_data = pd.DataFrame([features_dict])
    prediction = model.predict(input_data)[0]
    return prediction