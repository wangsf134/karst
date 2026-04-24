# utils/logger.py
"""
Carbon_Flux_System.utils.logger
系统日志记录模块。
提供统一的日志获取接口，将所有模块的运行日志按固定格式输出至 logs/system.log，
便于运维人员追踪系统状态与异常定位。
"""

import logging
import os


def get_logger(name: str):
    """
    获取或创建一个 logger 实例，并配置统一的文件输出格式。

    该函数确保每个模块调用时不会重复添加 handler，所有日志均以 UTF-8 编码写入
    项目根目录下的 logs/system.log 文件。

    Args:
        name (str): logger 的名称，通常传入当前模块的 __name__。

    Returns:
        logging.Logger: 配置完成的 logger 对象。

    Notes:
        - 日志级别固定为 INFO，适合生产环境的一般性记录。
        - 日志格式包含时间戳、模块名、日志级别和具体消息，便于快速定位问题。
        - 若 logs 目录不存在，会自动创建，避免因路径缺失导致日志写入失败。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # 创建 logs 目录（首次运行时自动创建）
        if not os.path.exists("logs"):
            os.makedirs("logs")

        handler = logging.FileHandler("logs/system.log", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger