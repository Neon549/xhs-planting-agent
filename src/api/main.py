"""
FastAPI 主入口
=============
支持两种部署模式：
    local：本地 bge-m3（开发用）
    hf_api：HF Inference API（HF Spaces 部署用）
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import QueryRequest, QueryResponse
from src.graph.workflow import build_workflow, run_workflow
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import create_reranker
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from config.settings import RERANKER

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


# ── 初始化检索器 ────────────────────────────────────────────────
def _init_retriever() -> HybridRetriever:
    deploy_mode = os.getenv("DEPLOY_MODE", "local")
    logger.info(f"部署模式: {deploy_mode}")

    # 读取真实 chunks
    chunks_file = Path("data/processed/chunks.json")
    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        logger.info(f"加载真实数据: {len(chunks)} 个 chunks")
    else:
        # 兜底用 mock 数据
        logger.warning("找不到 chunks 文件，使用 mock 数据")
        chunks = [
            {
                "chunk_id": "3c_001_0",
                "note_id": "3c_001",
                "text": "标题: iPhone 16 Pro 用了三个月\nA18 Pro 性能强，电池差，价格贵。",
                "metadata": {"category": "数码3C", "liked_count": 3241},
            },
            {
                "chunk_id": "fitness_001_0",
                "note_id": "fitness_001",
                "text": "标题: 家用哑铃怎么选\n入门500以内固定套装，进阶可调节哑铃。",
                "metadata": {"category": "健身器材", "liked_count": 12043},
            },
        ]

    # BM25
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    # Dense
    if deploy_mode == "hf_api":
        from src.retrieval.dense_retriever import HFApiDenseRetriever

        store = MilvusStore(db_path="data/milvus_lite.db")
        dense = HFApiDenseRetriever(store=store)
    elif deploy_mode == "local":
        try:
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.chroma_store import ChromaStore
            from config.settings import VECTOR_STORE

            store = ChromaStore(
                persist_dir=str(Path("data/chroma")),
                collection_name=VECTOR_STORE["collection_name"],
            )
            dense = DenseRetriever(milvus_store=store)
        except Exception as e:
            logger.warning(f"Dense 检索初始化失败，使用 mock: {e}")
            dense = _mock_dense(chunks)
    else:
        dense = _mock_dense(chunks)

    reranker = create_reranker(RERANKER)
    return HybridRetriever(bm25=bm25, dense=dense, reranker=reranker)


def _mock_dense(chunks):
    import random

    class MockDense:
        def search(self, query, top_k=20):
            shuffled = chunks.copy()
            random.shuffle(shuffled)
            return [
                {
                    **c,
                    "score": round(random.uniform(0.6, 0.95), 4),
                    "rank": i + 1,
                    "source": "dense",
                }
                for i, c in enumerate(shuffled[:top_k])
            ]

    return MockDense()


# ── 全局初始化 ──────────────────────────────────────────────────
retriever = _init_retriever()
ltm = LongTermMemory(db_path="data/memory.db")
stm = ShortTermMemory()
workflow = build_workflow(retriever=retriever, ltm=ltm, stm=stm)
logger.info("API 组件初始化完成")


# ── 接口 ────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "mode": os.getenv("DEPLOY_MODE", "local"),
    }


@app.get("/stats")
def get_stats():
    chunks_file = Path("data/processed/chunks.json")
    chunks_count = 0
    if chunks_file.exists():
        chunks_count = len(json.loads(chunks_file.read_text(encoding="utf-8")))
    return {
        "chunks_indexed": chunks_count,
        "sessions_active": len(stm.get_all_sessions()),
        "deploy_mode": os.getenv("DEPLOY_MODE", "local"),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
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
