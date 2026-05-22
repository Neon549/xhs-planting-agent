#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/18 23:54
@updated: 2026/5/18 23:54
@version: 1.0
@description: 
"""
"""
完整工作流验证 Demo
==================
把所有模块串起来跑一次端到端测试。
不需要真实 LLM 和 Embedding 模型。

"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import build_workflow, run_workflow
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.milvus_store import MilvusStore
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.memory.memory_writer import MemoryWriter


# Mock chunks（同模块 5 demo）
MOCK_CHUNKS = [
    {"chunk_id": "3c_001_0", "note_id": "3c_001",
     "text": "标题: iPhone 16 Pro 用了三个月，说几个没人说的缺点\nA18 Pro 性能强，但电池差，价格贵。",
     "metadata": {"category": "数码3C", "liked_count": 3241}},
    {"chunk_id": "3c_002_0", "note_id": "3c_002",
     "text": "标题: 安卓旗舰横评 小米15 vs 华为Mate70\n性能小米强，相机华为好，续航华为最长。",
     "metadata": {"category": "数码3C", "liked_count": 8923}},
    {"chunk_id": "3c_003_0", "note_id": "3c_003",
     "text": "标题: 轻薄笔记本推荐 预算5000以内\n重量1.3kg，续航12小时，性价比超高！必买！闭眼入！链接在主页🔗",
     "metadata": {"category": "数码3C", "liked_count": 500}},
    {"chunk_id": "fitness_001_0", "note_id": "fitness_001",
     "text": "标题: 家用哑铃怎么选 避坑指南\n入门500以内买固定套装，进阶买可调节哑铃推荐Bowflex。",
     "metadata": {"category": "健身器材", "liked_count": 12043}},
    {"chunk_id": "fitness_002_0", "note_id": "fitness_002",
     "text": "标题: 瑜伽垫选购 TPE vs 橡胶\nTPE轻便适合家用，天然橡胶防滑最好但有气味。",
     "metadata": {"category": "健身器材", "liked_count": 6782}},
]


def make_mock_retriever() -> HybridRetriever:
    """构建一个不需要真实模型的 mock 检索器。"""
    bm25 = BM25Retriever()
    bm25.build_index(MOCK_CHUNKS)

    # Mock DenseRetriever：search 方法直接返回随机排序的 chunks
    class MockDenseRetriever:
        def search(self, query: str, top_k: int = 20) -> list[dict]:
            results = []
            shuffled = MOCK_CHUNKS.copy()
            random.shuffle(shuffled)
            for i, c in enumerate(shuffled[:top_k]):
                results.append({**c, "score": round(random.uniform(0.6, 0.95), 4),
                                 "rank": i+1, "source": "dense"})
            return results

    retriever = HybridRetriever(
        bm25=bm25,
        dense=MockDenseRetriever(),
        reranker=None,
    )
    return retriever


def main():
    print("=" * 60)
    print("  完整 Agent 工作流验证 Demo")
    print("=" * 60)

    # 初始化组件
    retriever = make_mock_retriever()
    ltm = LongTermMemory(db_path="data/memory.db")
    stm = ShortTermMemory()

    # 预设一个用户的长期记忆（模拟已有历史的用户）
    ltm.update_profile("user_001", {"budget": 3000, "prefer": "轻薄"})

    # 构建工作流
    workflow = build_workflow(retriever=retriever, ltm=ltm, stm=stm)

    # 测试两个 query
    test_cases = [
        ("user_001", "推荐一款轻薄笔记本，预算3000"),
        ("user_002", "家用健身器材哑铃怎么选"),
    ]

    for user_id, query in test_cases:
        print(f"\n{'─'*50}")
        print(f"用户: {user_id} | 查询: {query}")
        print(f"{'─'*50}")

        final_state = run_workflow(
            workflow=workflow,
            user_query=query,
            user_id=user_id,
            session_id=f"session_{user_id}",
        )

        print(f"\n执行步骤: {final_state['steps_taken']}")
        print(f"品类识别: {final_state['parsed_query'].get('category')}")
        print(f"搜索词: {final_state['search_keywords']}")
        print(f"召回 chunks: {len(final_state['retrieved_chunks'])} 条")

        ad_results = final_state['ad_detection_results']
        ad_count = sum(1 for v in ad_results.values() if v['is_ad'])
        print(f"软广检测: {ad_count}/{len(ad_results)} 条被标记")

        print(f"干净 chunks: {len(final_state['clean_chunks'])} 条")
        print(f"\n【最终推荐】\n{final_state['final_recommendation']}")

        if final_state.get('user_context'):
            print(f"【个性化上下文】{final_state['user_context']}")

    print("\n" + "=" * 60)
    print("  ✅ 完整工作流验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()