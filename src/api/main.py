"""FastAPI 主入口"""
# 模块 9 实现

# from fastapi import FastAPI
# app = FastAPI(title="XHS Planting Agent")
"""
FastAPI 主入口
=============
把 LangGraph 工作流封装成 REST API。

运行:
    uv run uvicorn src.api.main:app --reload --port 8000

接口:
    POST /query   → 主查询接口
    GET  /health  → 健康检查
    GET  /stats   → 系统状态
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import QueryRequest, QueryResponse
from src.graph.workflow import build_workflow, run_workflow
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory

app = FastAPI(
    title="小红书种草决策 Agent",
    description="基于 LangGraph + Milvus 的多 Agent 推荐系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局组件（启动时初始化）──────────────────────────────────
MOCK_CHUNKS = [
    {"chunk_id": "3c_001_0", "note_id": "3c_001",
     "text": "标题: iPhone 16 Pro 用了三个月真实感受\nA18 Pro 性能强，但电池差，价格贵。",
     "metadata": {"category": "数码3C", "liked_count": 3241}},
    {"chunk_id": "3c_002_0", "note_id": "3c_002",
     "text": "标题: 安卓旗舰横评 小米15 vs 华为Mate70\n性能小米强，相机华为好。",
     "metadata": {"category": "数码3C", "liked_count": 8923}},
    {"chunk_id": "fitness_001_0", "note_id": "fitness_001",
     "text": "标题: 家用哑铃怎么选 避坑指南\n入门500以内固定套装，进阶可调节哑铃。",
     "metadata": {"category": "健身器材", "liked_count": 12043}},
    {"chunk_id": "fitness_002_0", "note_id": "fitness_002",
     "text": "标题: 瑜伽垫选购 TPE vs 橡胶\nTPE轻便，天然橡胶防滑最好。",
     "metadata": {"category": "健身器材", "liked_count": 6782}},
]

import random

class MockDenseRetriever:
    def search(self, query: str, top_k: int = 20) -> list[dict]:
        shuffled = MOCK_CHUNKS.copy()
        random.shuffle(shuffled)
        return [{**c, "score": round(random.uniform(0.6, 0.95), 4),
                 "rank": i+1, "source": "dense"}
                for i, c in enumerate(shuffled[:top_k])]

bm25 = BM25Retriever()
bm25.build_index(MOCK_CHUNKS)

retriever = HybridRetriever(
    bm25=bm25,
    dense=MockDenseRetriever(),
    reranker=None,
)

ltm = LongTermMemory(db_path="data/memory.db")
stm = ShortTermMemory()
workflow = build_workflow(retriever=retriever, ltm=ltm, stm=stm)

logger.info("API 组件初始化完成")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/stats")
def get_stats():
    return {
        "chunks_indexed": len(MOCK_CHUNKS),
        "sessions_active": len(stm.get_all_sessions()),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """主查询接口：接收用户问题，返回推荐结果。"""
    logger.info(f"收到查询: {request.user_query} | user={request.user_id}")

    try:
        final_state = run_workflow(
            workflow=workflow,
            user_query=request.user_query,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    ad_results = final_state.get("ad_detection_results", {})
    ad_count = sum(1 for v in ad_results.values() if v.get("is_ad"))

    return QueryResponse(
        user_query=request.user_query,
        parsed_query=final_state.get("parsed_query", {}),
        retrieved_count=len(final_state.get("retrieved_chunks", [])),
        ad_count=ad_count,
        clean_count=len(final_state.get("clean_chunks", [])),
        final_recommendation=final_state.get("final_recommendation", ""),
        sources=final_state.get("recommendation_sources", []),
        steps_taken=final_state.get("steps_taken", []),
        user_context=final_state.get("user_context", ""),
    )