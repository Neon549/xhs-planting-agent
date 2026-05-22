"""Retrieval Agent: 调用检索器返回 Top-K 笔记"""
# 模块 7 实现
"""
Retrieval Agent
===============
调用 HybridRetriever，召回相关笔记 chunks。

"""

from loguru import logger
from src.graph.state import AgentState
from src.retrieval.hybrid_retriever import HybridRetriever


def retrieval_agent_node(state: AgentState, retriever: HybridRetriever) -> dict:
    """LangGraph 节点：混合检索。"""
    search_keywords = state.get("search_keywords", state["user_query"])
    parsed_query = state.get("parsed_query", {})

    logger.info(f"[RetrievalAgent] 检索: {search_keywords}")

    # 调用混合检索（模块 5）
    chunks = retriever.search(
        query=search_keywords,
        top_k=10,
        use_reranker=False,   # demo 阶段不加载大模型，关闭 reranker
    )

    # 按品类过滤（如果 QueryPlanner 识别了品类）
    category = parsed_query.get("category", "")
    if category and category != "其他":
        filtered = [c for c in chunks if c.get("metadata", {}).get("category") == category]
        if filtered:   # 过滤后还有结果才用，否则用原始结果
            chunks = filtered

    logger.info(f"[RetrievalAgent] 召回 {len(chunks)} 条 chunks")

    return {
        "retrieved_chunks": chunks,
        "steps_taken": ["retrieval_agent"],
    }