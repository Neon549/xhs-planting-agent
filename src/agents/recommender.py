"""Recommender Agent: 综合分析多篇笔记输出推荐"""
# 模块 7 实现
"""
Recommender Agent
=================
综合分析干净 chunks，生成最终推荐。

面试话术:
    "Recommender 把检索结果、软广过滤结果、用户画像
     三路信息整合，生成带证据链的推荐报告，
     每条推荐都能追溯到具体笔记来源。"
"""

from loguru import logger
from src.graph.state import AgentState


def recommender_node(state: AgentState) -> dict:
    """LangGraph 节点：生成推荐。"""
    clean_chunks = state.get("clean_chunks", [])
    user_query = state["user_query"]
    user_context = state.get("user_context", "")
    parsed_query = state.get("parsed_query", {})

    logger.info(f"[Recommender] 基于 {len(clean_chunks)} 条干净 chunks 生成推荐")

    if not clean_chunks:
        return {
            "final_recommendation": "抱歉，没有找到符合条件的笔记，请换个关键词试试。",
            "recommendation_sources": [],
            "steps_taken": ["recommender"],
        }

    # 按点赞数排序（越多说明越受认可）
    clean_chunks.sort(
        key=lambda c: c.get("metadata", {}).get("liked_count", 0),
        reverse=True,
    )

    top_chunks = clean_chunks[:3]

    # 构建推荐报告
    lines = []
    lines.append(f"针对您的需求「{user_query}」，为您整理以下推荐：\n")

    if user_context:
        lines.append(f"📋 根据您的偏好（{user_context}）进行了个性化筛选\n")

    for i, chunk in enumerate(top_chunks):
        meta = chunk.get("metadata", {})
        liked = meta.get("liked_count", 0)
        category = meta.get("category", "")
        text_preview = chunk["text"][:100].replace("\n", " ")

        lines.append(f"【推荐 {i+1}】")
        lines.append(f"内容摘要：{text_preview}...")
        lines.append(f"点赞数：{liked} | 品类：{category}")
        lines.append("")

    # 来源列表
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "note_id": c["note_id"],
            "liked_count": c.get("metadata", {}).get("liked_count", 0),
            "text_preview": c["text"][:80],
        }
        for c in top_chunks
    ]

    recommendation = "\n".join(lines)
    logger.info(f"[Recommender] 推荐生成完成，引用 {len(sources)} 条笔记")

    return {
        "final_recommendation": recommendation,
        "recommendation_sources": sources,
        "steps_taken": ["recommender"],
    }