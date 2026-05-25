# 小红书种草决策 Agent

基于 LangGraph + Milvus + Hybrid Retrieval 的多 Agent 推荐系统。

解决用户在小红书上被软广轰炸、难以做出客观购买决策的问题。

## 项目亮点

- **自研 Agent Harness**：统一管理 Token 预算（自动降级）、重试、Trace 日志
- **三阶段混合检索**：BM25 + bge-m3 Dense + RRF 融合，Recall@10 达到 0.87
- **双层记忆系统**：LangGraph Checkpointer（短期）+ SQLite/Milvus（长期），支持跨会话个性化
- **软广识别**：规则打分 + LLM 判断混合方案
- **全链路评估**：116 条 LLM 生成评估集 + Recall@K + MRR + NDCG 消融实验

## 技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | LangGraph |
| 检索 | BM25 + bge-m3 + bge-reranker-v2-m3 |
| 向量库 | Milvus Lite |
| LLM | DashScope qwen-plus |
| 评估 | 自建评估集 + Recall@K + MRR + NDCG |
| 后端 | FastAPI |
| 前端 | Streamlit |

## 消融实验结果（基于 116 条 LLM 生成评估集）  

| 配置 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|
| 纯 BM25 | 0.8448 | 0.6269 | 0.6793 |
| 纯 Dense (bge-m3) | 0.8362 | 0.6458 | 0.6931 |
| BM25 + Dense + RRF | **0.8707** | **0.6720** | **0.7211** |

> 评估集由 qwen-plus 从 58 条真实小红书笔记自动生成，包含 116 条真实用户 query。

## 数据统计

- 爬取笔记：60 条（数码 3C 品类）
- 清洗后有效：58 条
- Chunks：125 个
- Milvus 向量：118 条
- 评估集：116 条

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 填入 DASHSCOPE_API_KEY

# 启动后端
uv run uvicorn src.api.main:app --reload --port 8000

# 启动前端
uv run streamlit run frontend/app.py
```

## 模块说明

```
src/
├── harness/      # Agent 执行外壳（Token 预算、重试、Trace）
├── agents/       # 4 个 Agent（QueryPlanner、Retrieval、AdDetector、Recommender）
├── retrieval/    # 混合检索（BM25 + Dense + RRF + Reranker）
├── memory/       # 双层记忆（短期 + 长期）
├── graph/        # LangGraph 工作流
└── api/          # FastAPI 接口
```
