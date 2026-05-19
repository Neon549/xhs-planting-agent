"""
一键创建小红书 Agent 项目目录结构

使用方法:
    1. 把这个文件放在你的项目根目录 (xhs-planting-agent/) 下
    2. 在该目录下执行: python setup_project.py
    3. 脚本会自动创建所有目录和占位文件
"""

from pathlib import Path

# ============================================================
# 目录结构定义
# ============================================================
DIRECTORIES = [
    # 配置层
    "config",
    "config/prompts",

    # 源码层
    "src",
    "src/harness",
    "src/agents",
    "src/retrieval",
    "src/memory",
    "src/data",
    "src/graph",
    "src/api",

    # 数据层（不进 git）
    "data/raw",
    "data/processed",
    "data/eval",

    # 评估
    "evaluation",
    "evaluation/reports",

    # 实验性 notebook
    "notebooks",

    # 测试
    "tests",

    # 前端
    "frontend",

    # 一次性脚本
    "scripts",

    # 日志（不进 git）
    "logs",
]

# ============================================================
# 需要创建的占位文件
# ============================================================
# 每个 Python 包目录都需要 __init__.py
PACKAGE_INIT_FILES = [
    "config/__init__.py",
    "src/__init__.py",
    "src/harness/__init__.py",
    "src/agents/__init__.py",
    "src/retrieval/__init__.py",
    "src/memory/__init__.py",
    "src/data/__init__.py",
    "src/graph/__init__.py",
    "src/api/__init__.py",
    "evaluation/__init__.py",
    "tests/__init__.py",
]

# 关键的占位代码文件（带简短注释，方便后面填代码时定位）
PLACEHOLDER_FILES = {
    "config/settings.py": '''"""全局配置: 模型名、Milvus 地址、API key 等"""

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
''',

    "src/harness/base.py": '''"""Agent Harness 基类: 工具调用、预算控制、重试、Trace"""
# 模块 2 我们来实现
''',

    "src/harness/tool_registry.py": '''"""工具注册中心"""
# 模块 2 实现
''',

    "src/harness/budget.py": '''"""Token 预算控制"""
# 模块 2 实现
''',

    "src/harness/tracer.py": '''"""Trace 日志: 记录 Agent 每一步的输入输出"""
# 模块 2 实现
''',

    "src/agents/query_planner.py": '''"""Query Planner Agent: 解析用户需求为结构化 query"""
# 模块 7 实现
''',

    "src/agents/retrieval_agent.py": '''"""Retrieval Agent: 调用检索器返回 Top-K 笔记"""
# 模块 7 实现
''',

    "src/agents/ad_detector.py": '''"""Ad Detector Agent: 软广识别"""
# 模块 7 实现
''',

    "src/agents/recommender.py": '''"""Recommender Agent: 综合分析多篇笔记输出推荐"""
# 模块 7 实现
''',

    "src/retrieval/bm25_retriever.py": '''"""BM25 关键词检索"""
# 模块 5 实现
''',

    "src/retrieval/dense_retriever.py": '''"""向量检索 (基于 Milvus)"""
# 模块 5 实现
''',

    "src/retrieval/hybrid_retriever.py": '''"""Hybrid 检索: BM25 + Dense + Reranker"""
# 模块 5 实现
''',

    "src/retrieval/reranker.py": '''"""bge-reranker-v2-m3 重排序"""
# 模块 5 实现
''',

    "src/memory/short_term.py": '''"""短期记忆: LangGraph Checkpointer"""
# 模块 6 实现
''',

    "src/memory/long_term.py": '''"""长期记忆: SQLite (结构化) + Milvus (语义化)"""
# 模块 6 实现
''',

    "src/memory/memory_writer.py": '''"""记忆写入器: 会话结束后自动总结"""
# 模块 6 实现
''',

    "src/data/crawler.py": '''"""MediaCrawler 封装"""
# 模块 3 实现
''',

    "src/data/cleaner.py": '''"""数据清洗"""
# 模块 3 实现
''',

    "src/data/chunker.py": '''"""Chunking 策略"""
# 模块 3 实现
''',

    "src/graph/state.py": '''"""LangGraph State 定义"""
# 模块 7 实现
''',

    "src/graph/workflow.py": '''"""LangGraph 主工作流"""
# 模块 7 实现
''',

    "src/api/main.py": '''"""FastAPI 主入口"""
# 模块 9 实现

# from fastapi import FastAPI
# app = FastAPI(title="XHS Planting Agent")
''',

    "src/api/schemas.py": '''"""Pydantic 请求/响应模型"""
# 模块 9 实现
''',

    "evaluation/retrieval_eval.py": '''"""检索评估: Recall@K, MRR, NDCG"""
# 模块 8 实现
''',

    "evaluation/ragas_eval.py": '''"""RAGAs 评估"""
# 模块 8 实现
''',

    "frontend/app.py": '''"""Streamlit 前端"""
# 模块 9 实现
''',

    "scripts/crawl_data.py": '''"""一次性脚本: 爬取数据"""
# 模块 3 实现
''',

    "scripts/build_index.py": '''"""一次性脚本: 构建 Milvus 索引"""
# 模块 4 实现
''',

    "scripts/run_evaluation.py": '''"""一次性脚本: 跑评估"""
# 模块 8 实现
''',

    "tests/test_harness.py": '"""Harness 单元测试"""\n',
    "tests/test_retrieval.py": '"""检索单元测试"""\n',
    "tests/test_memory.py": '"""记忆单元测试"""\n',

    ".env.example": '''# 复制此文件为 .env, 填入真实 key 后使用
# DASHSCOPE_API_KEY=sk-xxx
# LANGFUSE_PUBLIC_KEY=
# LANGFUSE_SECRET_KEY=
''',

    ".gitignore": '''# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# 虚拟环境
.venv/
venv/
env/

# uv
.python-version

# IDE
.vscode/
.idea/
*.swp

# 环境变量 (敏感信息)
.env

# 数据 (太大或敏感)
data/
logs/
*.db
*.sqlite

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# 输出文件
*.log
traces/
''',

    "README.md": '''# 小红书种草决策 Agent

基于 LangGraph + CrewAI + Milvus 的多 Agent 种草推荐系统。

## 项目特性

- **Agent Harness 层**: 自研工具路由、预算控制、Trace 日志
- **Hybrid Retrieval**: BM25 + Dense Vector + bge-reranker-v2-m3
- **双层记忆**: LangGraph Checkpointer (短期) + SQLite/Milvus (长期)
- **软广识别**: 文本特征 + LLM 多次投票
- **全链路评估**: RAGAs + Recall@K + MRR + NDCG

## 技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | LangGraph + CrewAI |
| RAG | LlamaIndex |
| 向量库 | Milvus (Lite) |
| LLM | DashScope qwen-plus |
| 后端 | FastAPI |
| 前端 | Streamlit |
| 评估 | RAGAs |
| 可观测性 | Langfuse |

## 快速开始

```bash
uv sync
cp .env.example .env  # 填入 API key
uv run python scripts/build_index.py
uv run uvicorn src.api.main:app --reload
```

## 开发进度

- [ ] 模块 1: 项目结构 + 依赖管理
- [ ] 模块 2: Agent Harness
- [ ] 模块 3: 数据采集 (MediaCrawler)
- [ ] 模块 4: Milvus 向量库
- [ ] 模块 5: Hybrid Retrieval
- [ ] 模块 6: 双层记忆
- [ ] 模块 7: 4 个 Agent + LangGraph 编排
- [ ] 模块 8: 评估体系
- [ ] 模块 9: 前端 + API + 部署
''',
}


# ============================================================
# 执行
# ============================================================
def main():
    root = Path(".").resolve()
    print(f"项目根目录: {root}\n")

    # 1. 创建目录
    print("【1/3】创建目录...")
    for d in DIRECTORIES:
        (root / d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")

    # 2. 创建 __init__.py
    print("\n【2/3】创建 __init__.py...")
    for f in PACKAGE_INIT_FILES:
        path = root / f
        if not path.exists():
            path.touch()
            print(f"  ✓ {f}")

    # 3. 创建占位文件
    print("\n【3/3】创建占位文件...")
    for filepath, content in PLACEHOLDER_FILES.items():
        path = root / filepath
        if path.exists():
            print(f"  - {filepath} (已存在, 跳过)")
            continue
        path.write_text(content, encoding="utf-8")
        print(f"  ✓ {filepath}")

    print("\n" + "=" * 50)
    print("✅ 目录结构创建完成!")
    print("=" * 50)
    print("\n下一步:")
    print("  1. 把 Claude 给的 pyproject.toml 内容覆盖到根目录的 pyproject.toml")
    print("  2. 运行: uv sync")
    print("  3. 开始模块 2: Agent Harness")


if __name__ == "__main__":
    main()