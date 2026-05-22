"""LangGraph 主工作流"""
# 模块 7 实现
"""
LangGraph 主工作流
==================
把 4 个 Agent 节点串联成完整的工作流。

图结构:
    START
      ↓
    query_planner
      ↓
    retrieval_agent
      ↓
    ad_detector
      ↓
    recommender
      ↓
    END

"""

import functools
from langgraph.graph import StateGraph, START, END
from loguru import logger

from src.graph.state import AgentState
from src.agents.query_planner import query_planner_node
from src.agents.retrieval_agent import retrieval_agent_node
from src.agents.ad_detector import ad_detector_node
from src.agents.recommender import recommender_node
from src.retrieval.hybrid_retriever import HybridRetriever
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


def build_workflow(
    retriever: HybridRetriever,
    ltm: LongTermMemory,
    stm: ShortTermMemory,
) -> object:
    """
    构建并编译 LangGraph 工作流。

    使用 functools.partial 把依赖注入到节点函数，
    让节点函数保持 (state) -> dict 的纯净签名。
    """
    graph = StateGraph(AgentState)

    # 注入依赖
    planner_with_deps = functools.partial(query_planner_node, ltm=ltm)
    retrieval_with_deps = functools.partial(retrieval_agent_node, retriever=retriever)

    # 添加节点
    graph.add_node("query_planner", planner_with_deps)
    graph.add_node("retrieval_agent", retrieval_with_deps)
    graph.add_node("ad_detector", ad_detector_node)
    graph.add_node("recommender", recommender_node)

    # 添加边（执行顺序）
    graph.add_edge(START, "query_planner")
    graph.add_edge("query_planner", "retrieval_agent")
    graph.add_edge("retrieval_agent", "ad_detector")
    graph.add_edge("ad_detector", "recommender")
    graph.add_edge("recommender", END)

    # 编译
    compiled = graph.compile()
    logger.info("LangGraph 工作流编译完成")
    return compiled


def run_workflow(
    workflow,
    user_query: str,
    user_id: str = "anonymous",
    session_id: str = "default",
) -> AgentState:
    """
    运行工作流，返回最终 State。
    """
    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "user_query": user_query,
        "parsed_query": {},
        "search_keywords": "",
        "user_context": "",
        "retrieved_chunks": [],
        "ad_detection_results": {},
        "clean_chunks": [],
        "final_recommendation": "",
        "recommendation_sources": [],
        "error": "",
        "steps_taken": [],
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = workflow.invoke(initial_state, config=config)
    return final_state