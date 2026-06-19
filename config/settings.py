import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

# LLM 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen-plus"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

# 检索配置
RETRIEVAL = {
    "bm25_top_k": 20,
    "dense_top_k": 20,
    "rrf_k": 60,
    "final_top_k": 10,
}

# Reranker 配置
RERANKER = {
    "provider": "none",  # none / cross_encoder / llm
    "model": "BAAI/bge-reranker-v2-m3",
    "top_k": 10,
    "use_fp16": True,
}

# Embedding 配置
EMBEDDING = {
    "provider": "bge_m3",
    "model": "BAAI/bge-m3",
    "use_fp16": True,
    "batch_size": 16,
    "max_length": 512,
}

# 向量库配置（已从 Milvus 迁移到 ChromaDB）
VECTOR_STORE = {
    "provider": "chroma",
    "persist_dir": str(DATA_DIR / "chroma"),
    "collection_name": "xhs_notes",
    "embedding_dim": 1024,
}

# 部署模式
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "local")
HF_TOKEN = os.getenv("HF_TOKEN", "")