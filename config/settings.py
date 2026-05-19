"""全局配置: 模型名、Milvus 地址、API key 等"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent

# LLM 配置 (待填)
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# LLM_MODEL = "qwen-plus"

# Milvus 配置 (待填)
# MILVUS_URI = "./data/milvus_lite.db"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
