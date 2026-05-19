"""LangGraph State 定义"""
# 模块 7 实现
"""
LangGraph State 定义
====================
State 是贯穿所有 Agent 的数据容器。
每个 Agent 节点读取 State，处理后返回更新的字段。

设计原则:
    - 所有字段都有默认值（避免 KeyError）
    - 字段按流程顺序排列（便于理解数据流向）
    - 用 Annotated + operator.add 支持列表的追加合并

面试话术:
    "State 用 TypedDict 定义，字段按数据流顺序排列，
     列表字段用 Annotated[list, operator.add] 支持
     多节点追加而不是覆盖，避免并行节点的数据丢失。"
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    贯穿整个 Agent 工作流的状态。

    数据流向:
        user_query
            ↓ QueryPlanner 填写
        parsed_query, search_keywords, user_context
            ↓ RetrievalAgent 填写
        retrieved_chunks
            ↓ AdDetector 填写
        ad_detection_results, clean_chunks
            ↓ Recommender 填写
        final_recommendation
    """

    # ── 输入 ──────────────────────────────────────────────────
    user_id: str                    # 用户 ID（用于查长期记忆）
    session_id: str                 # 会话 ID（用于查短期记忆）
    user_query: str                 # 用户原始输入

    # ── QueryPlanner 输出 ─────────────────────────────────────
    parsed_query: dict              # 结构化解析结果
                                    # {"category": "数码3C", "keywords": [...],
                                    #  "price_range": [0, 2000], "requirements": [...]}
    search_keywords: str            # 优化后的搜索词（传给检索器）
    user_context: str               # 从长期记忆取出的用户画像

    # ── RetrievalAgent 输出 ───────────────────────────────────
    retrieved_chunks: Annotated[list, operator.add]
                                    # 召回的 chunks（用 add 支持追加）

    # ── AdDetector 输出 ──────────────────────────────────────
    ad_detection_results: dict      # {chunk_id: {"is_ad": bool, "confidence": float}}
    clean_chunks: list              # 过滤软广后的干净 chunks

    # ── Recommender 输出 ─────────────────────────────────────
    final_recommendation: str       # 最终推荐文本
    recommendation_sources: list    # 推荐依据的笔记列表（用于展示）

    # ── 元信息 ───────────────────────────────────────────────
    error: str                      # 错误信息（有错误时填写）
    steps_taken: Annotated[list, operator.add]
                                    # 执行步骤记录（调试用）