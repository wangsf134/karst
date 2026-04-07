import logging
import os


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # 创建 logs 目录
        if not os.path.exists("logs"):
            os.makedirs("logs")

        handler = logging.FileHandler("logs/system.log", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger