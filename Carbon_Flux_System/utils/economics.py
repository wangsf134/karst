# utils/economics.py
"""
Carbon_Flux_System.utils.economics
环境经济学核算模块。
负责将物理固碳量（gC/m²/yr）转换为碳资产经济指标，包括总固碳吨数、二氧化碳当量（CO₂e）以及模拟碳交易收益。
该模块是固碳数据向政务决策层汇报经济帐的核心计算单元。
"""

from typing import Dict, Any
from config import C_TO_CO2_FACTOR, M2_TO_HA_FACTOR, G_TO_TON_FACTOR


def calculate_carbon_assets(flux_g_m2: float, area_ha: float, price_per_ton: float) -> Dict[str, Any]:
    """
    根据单位面积年化固碳量计算碳汇资产的经济价值。

    该函数实现了“固碳量 → 二氧化碳当量 → 经济收益”的标准核算链路，
    适用于 CCER（国家核证自愿减排量）口径下的生态资产预估。

    Args:
        flux_g_m2 (float): 单位面积年化固碳量，单位为 gC/m²/yr。
        area_ha (float): 评估地块的面积，单位为公顷 (ha)。
        price_per_ton (float): 模拟碳交易价格，单位为元/吨 CO₂e。

    Returns:
        Dict[str, Any]: 包含以下键值的字典：
            - annual_carbon_tons (float): 年总固碳量，单位为吨碳 (tC)，保留4位小数。
            - annual_co2e_tons (float): 年二氧化碳当量，单位为吨 CO₂e，保留4位小数。
            - annual_revenue (float): 年碳汇收益，单位为元，保留2位小数。

    Notes:
        - 面积转换系数 M2_TO_HA_FACTOR (1 ha = 10,000 m²) 和重量转换系数 G_TO_TON_FACTOR (1 t = 1,000,000 g)
          均从 config.py 中集中读取，确保全系统核算口径统一。
        - C_TO_CO2_FACTOR 为碳转二氧化碳当量的分子量比 (44/12)，符合 IPCC 指南。
        - 经济价值计算基于 CO₂e 而非纯碳，以契合国际碳市场主流交易规则。
    """
    # 计算地块年度总固碳量（吨碳）
    total_carbon_tons = (flux_g_m2 * M2_TO_HA_FACTOR * area_ha) / G_TO_TON_FACTOR
    # 转换为二氧化碳当量（吨 CO₂e）
    total_co2e_tons = total_carbon_tons * C_TO_CO2_FACTOR
    # 基于碳价计算年度收益
    annual_revenue = total_co2e_tons * price_per_ton

    return {
        "annual_carbon_tons": round(total_carbon_tons, 4),
        "annual_co2e_tons": round(total_co2e_tons, 4),
        "annual_revenue": round(annual_revenue, 2)
    }