# utils/economics.py
import pandas as pd
from config import C_TO_CO2_FACTOR, M2_TO_HA_FACTOR, G_TO_TON_FACTOR


def calculate_carbon_assets(flux_g_m2, area_ha, price_per_ton):
    """
    计算碳资产相关指标
    :param flux_g_m2: 单位面积年化固碳量 (gC/m2/yr)
    :param area_ha: 地块面积 (公顷)
    :param price_per_ton: 当前碳交易价格 (元/吨)
    :return: 包含各项指标的字典
    """
    # 1. 计算年度总固碳量 (纯碳, 吨)
    total_carbon_tons = (flux_g_m2 * M2_TO_HA_FACTOR * area_ha) / G_TO_TON_FACTOR

    # 2. 转换为二氧化碳当量 (CO2e, 吨)
    total_co2e_tons = total_carbon_tons * C_TO_CO2_FACTOR

    # 3. 计算预期年化收益 (人民币)
    annual_revenue = total_co2e_tons * price_per_ton

    return {
        "annual_carbon_tons": round(total_carbon_tons, 4),
        "annual_co2e_tons": round(total_co2e_tons, 4),
        "annual_revenue": round(annual_revenue, 2)
    }