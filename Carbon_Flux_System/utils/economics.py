# utils/economics.py
from typing import Dict, Any
from config import C_TO_CO2_FACTOR, M2_TO_HA_FACTOR, G_TO_TON_FACTOR


def calculate_carbon_assets(flux_g_m2: float, area_ha: float, price_per_ton: float) -> Dict[str, Any]:
    """
    根据物理固碳量计算经济资产指标。

    Args:
        flux_g_m2 (float): 单位面积年化固碳量 (gC/m²/yr)。
        area_ha (float): 地块面积 (公顷)。
        price_per_ton (float): 模拟碳交易价格 (元/吨)。

    Returns:
        Dict[str, Any]: 包含总固碳吨数、CO2当量及年收益的字典。
    """
    total_carbon_tons = (flux_g_m2 * M2_TO_HA_FACTOR * area_ha) / G_TO_TON_FACTOR
    total_co2e_tons = total_carbon_tons * C_TO_CO2_FACTOR
    annual_revenue = total_co2e_tons * price_per_ton

    return {
        "annual_carbon_tons": round(total_carbon_tons, 4),
        "annual_co2e_tons": round(total_co2e_tons, 4),
        "annual_revenue": round(annual_revenue, 2)
    }