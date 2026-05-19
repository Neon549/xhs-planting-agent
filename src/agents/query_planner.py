"""Query Planner Agent: 解析用户需求为结构化 query"""
# 模块 7 实现
"""
Query Planner Agent
===================
解析用户自然语言输入，转成结构化 query。

输入: "油痘肌想买一款平价粉底液，预算200以内"
输出: {
    "category": "美妆",
    "keywords": ["粉底液", "油痘肌", "平价"],
    "price_range": [0, 200],
    "skin_type": "油痘肌",
    "requirements": ["适合油痘肌", "平价"]
}

面试话术:
    "QueryPlanner 把模糊的自然语言转成结构化参数，
     一方面用于 BM25 的关键词提取，
     另一方面用于 Milvus 的标量过滤（按品类/价格筛选）。"
"""

import json
import re
from loguru import logger
from src.graph.state import AgentState
from src.memory.long_term import LongTermMemory


# 品类关键词映射
CATEGORY_MAP = {
    "数码3C": ["手机", "iPhone", "安卓", "笔记本", "电脑", "耳机", "平板", "数码"],
    "健身器材": ["哑铃", "瑜伽", "健身", "跑步机", "器械", "弹力带"],
}

# 不依赖 LLM 的规则提取（降级方案）
def rule_based_parse(query: str) -> dict:
    """规则提取，不需要 LLM，用于 demo 和降级。"""
    # 品类识别
    category = "其他"
    for cat, keywords in CATEGORY_MAP.items():
        if any(kw in query for kw in keywords):
            category = cat
            break

    # 价格提取
    price_range = [0, 99999]
    price_match = re.search(r'(\d+)\s*(?:元|块|以内|左右)', query)
    if price_match:
        price_range = [0, int(price_match.group(1))]

    # 关键词提取（简单分词）
    import jieba
    tokens = [t for t in jieba.cut(query) if len(t) > 1]

    return {
        "category": category,
        "keywords": tokens,
        "price_range": price_range,
        "requirements": [query],   # 直接用原始 query 作为需求描述
    }


def query_planner_node(state: AgentState, ltm: LongTermMemory = None) -> dict:
    """
    LangGraph 节点函数。

    接收 State，返回需要更新的字段（dict）。
    LangGraph 会自动把返回值 merge 进 State。
    """
    user_query = state["user_query"]
    user_id = state.get("user_id", "anonymous")

    logger.info(f"[QueryPlanner] 解析 query: {user_query}")

    # 解析 query
    parsed = rule_based_parse(user_query)

    # 构建搜索关键词（用于 BM25）
    search_keywords = " ".join(parsed["keywords"][:5])

    # 从长期记忆取用户画像
    user_context = ""
    if ltm:
        user_context = ltm.build_user_context(user_id)

    logger.info(f"[QueryPlanner] 品类={parsed['category']} | 关键词={search_keywords}")

    return {
        "parsed_query": parsed,
        "search_keywords": search_keywords,
        "user_context": user_context,
        "steps_taken": ["query_planner"],
    }