# inspect_model.py
"""
Carbon_Flux_System.inspect_model
模型审查独立脚本。
用于在命令行环境下快速检查已训练随机森林模型的基本信息，包括模型类型、
特征数量、特征名称以及核心超参数配置，辅助运维与开发人员验证模型兼容性。
"""

import os
import joblib
import pprint

# 1. 动态获取模型路径（假设此脚本运行在项目根目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "karst_rf_model.pkl")


def inspect_model():
    """
    加载并打印模型的基础身份信息、特征要求与超参数配置。

    该函数适合在模型部署前或出现预测异常时快速诊断模型状态，
    不依赖 Streamlit 环境，可在任意 Python 终端中独立运行。
    """
    print(f"🔍 正在读取模型文件: {MODEL_PATH}")

    # 尝试加载模型
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ 模型解冻成功！\n")
    except Exception as e:
        print(f"❌ 加载失败，请检查模型路径是否正确。\n详细报错: {e}")
        return

    # ================= 模块 1：基础身份信息 =================
    print("-" * 40)
    print("▶️ 1. 模型基础身份")
    print("-" * 40)
    print(f"模型算法类型: {type(model).__name__}")

    # ================= 模块 2：特征输入要求 =================
    print("\n" + "-" * 40)
    print("▶️ 2. 模型特征 (Features) 审查")
    print("-" * 40)

    # 检查特征数量（训练时记录的特征维度）
    if hasattr(model, 'n_features_in_'):
        print(f"必须输入的特征总数: {model.n_features_in_} 个")
    else:
        print("特征总数: 未知 (模型可能较老或未保存此属性)")

    # 检查训练时绑定的具体列名（极其关键，决定了前端表头必须与此一致）
    if hasattr(model, 'feature_names_in_'):
        print("模型底层绑定的特征列名 (请核对是否为英文):")
        for i, name in enumerate(model.feature_names_in_):
            print(f"  特征 {i + 1}: '{name}'")
    else:
        print("特征列名: ⚠️ 该模型在训练时没有保存具体的列名。")

    # ================= 模块 3：超参数配置 =================
    print("\n" + "-" * 40)
    print("▶️ 3. 核心超参数 (Hyperparameters) 配置")
    print("-" * 40)

    params = model.get_params()

    # 针对随机森林，挑几个最关键的参数单独展示
    if type(model).__name__ == "RandomForestRegressor":
        print(f"🌳 森林中树的数量 (n_estimators): {params.get('n_estimators')}")
        print(f"📏 树的最大深度 (max_depth): {params.get('max_depth')}")
        print(f"✂️ 最小分裂样本数 (min_samples_split): {params.get('min_samples_split')}")
        print(f"🎲 随机种子 (random_state): {params.get('random_state')}\n")

    print("所有详细参数列表如下 (字典格式):")
    # 使用 pprint 让字典打印得更美观、自带缩进
    pprint.pprint(params)


if __name__ == "__main__":
    inspect_model()