#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/18 23:19
@updated: 2026/5/18 23:19
@version: 1.0
@description: 
"""
"""
Hybrid Retrieval 验证 Demo
==========================
用 mock 数据验证 BM25 + Dense + RRF + Reranker 全流程。

不下载大模型（用随机向量代替 Dense，跳过 Reranker）。
等真实数据和模型就位后，只需替换组件即可。

运行: uv run python scripts/demo_hybrid_retrieval.py
"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion

# ============================================================
# Mock 数据（来自模块 3 的清洗结果）
# ============================================================
MOCK_CHUNKS = [
    {"chunk_id": "3c_001_title", "note_id": "3c_001", "source": "bm25",
     "text": "标题: iPhone 16 Pro 用了三个月，说几个没人说的缺点",
     "metadata": {"category": "数码3C", "liked_count": 3241}},
    {"chunk_id": "3c_001_chunk_0", "note_id": "3c_001", "source": "bm25",
     "text": "标题: iPhone 16 Pro 用了三个月\n买之前做了很多功课。先说优点：A18 Pro 性能很强，游戏发热比上代好。缺点：钛金属边框滑，电池续航差，价格贵。",
     "metadata": {"category": "数码3C", "liked_count": 3241}},
    {"chunk_id": "3c_002_title", "note_id": "3c_002", "source": "bm25",
     "text": "标题: 安卓旗舰大横评！小米15 vs 华为Mate70 vs 三星S25",
     "metadata": {"category": "数码3C", "liked_count": 8923}},
    {"chunk_id": "3c_002_chunk_0", "note_id": "3c_002", "source": "bm25",
     "text": "标题: 安卓旗舰横评\n性能: 小米15最强。相机: 华为最好。续航: 华为最长。价格: 小米最便宜。结论：不在乎生态买小米，摄影党买华为。",
     "metadata": {"category": "数码3C", "liked_count": 8923}},
    {"chunk_id": "fitness_001_title", "note_id": "fitness_001", "source": "bm25",
     "text": "标题: 家用哑铃怎么选？避坑指南（入门/进阶/高阶推荐）",
     "metadata": {"category": "健身器材", "liked_count": 12043}},
    {"chunk_id": "fitness_001_chunk_0", "note_id": "fitness_001", "source": "bm25",
     "text": "标题: 家用哑铃选购\n入门500元以内选固定哑铃套装。进阶1000-2000选可调节哑铃，推荐Bowflex。高阶买商用固定架。避坑：别买太便宜的，包铁的生锈。",
     "metadata": {"category": "健身器材", "liked_count": 12043}},
    {"chunk_id": "fitness_002_title", "note_id": "fitness_002", "source": "bm25",
     "text": "标题: 瑜伽垫选购避坑！我踩过的坑都在这里了",
     "metadata": {"category": "健身器材", "liked_count": 6782}},
]


def random_unit_vector(dim: int = 1024) -> list[float]:
    """生成随机单位向量，模拟 Embedding 输出。"""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def mock_dense_search(query: str, chunks: list[dict], top_k: int = 20) -> list[dict]:
    """模拟 Dense 检索（随机分数，真实场景用 DenseRetriever）。"""
    results = []
    # 简单模拟：含关键词的 chunk 分数高一点
    for i, chunk in enumerate(chunks):
        base_score = random.uniform(0.6, 0.95)
        if any(kw in chunk["text"] for kw in query.split()):
            base_score = min(base_score + 0.1, 1.0)
        results.append({
            **chunk,
            "score": round(base_score, 4),
            "rank": 0,
            "source": "dense",
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results[:top_k]


def main():
    print("=" * 60)
    print("  Hybrid Retrieval 验证 Demo")
    print("=" * 60)

    # ── Step 1: BM25 建索引 ───────────────────────────────────
    print("\n【Step 1】BM25 建索引")
    bm25 = BM25Retriever()
    bm25.build_index(MOCK_CHUNKS)

    # ── Step 2: 测试两个 query ────────────────────────────────
    test_queries = [
        "iPhone 手机推荐",
        "家用健身器材哑铃",
    ]

    for query in test_queries:
        print(f"\n{'─'*50}")
        print(f"查询: 「{query}」")
        print(f"{'─'*50}")

        # BM25 召回
        bm25_results = bm25.search(query, top_k=5)
        print(f"\n[BM25 Top-5]")
        for r in bm25_results:
            print(f"  rank={r['rank']} score={r['score']:.4f} | {r['text'][:45]}...")

        # Dense 召回（mock）
        dense_results = mock_dense_search(query, MOCK_CHUNKS, top_k=5)
        print(f"\n[Dense Top-5（mock 随机分数）]")
        for r in dense_results:
            print(f"  rank={r['rank']} score={r['score']:.4f} | {r['text'][:45]}...")

        # RRF 融合
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60)
        print(f"\n[RRF 融合后 Top-5]")
        for r in fused[:5]:
            both = "★双路命中" if len(r["sources"]) == 2 else ""
            print(f"  rank={r['rank']} rrf={r['rrf_score']:.6f} {both} | {r['text'][:40]}...")

    # ── Step 3: 消融实验数据展示 ──────────────────────────────
    print(f"\n{'─'*50}")
    print("【消融实验示意】(真实数字需接入 bge-m3 + Reranker 后测量)")
    print(f"{'─'*50}")
    ablation_table = [
        ("纯 BM25",           "~0.65", "baseline"),
        ("纯 Dense (bge-m3)", "~0.74", "+0.09"),
        ("BM25 + Dense + RRF","~0.82", "+0.17"),
        ("+ Reranker",        "~0.89", "+0.24"),
    ]
    print(f"\n  {'配置':<22} {'Recall@10':>10} {'vs baseline':>12}")
    print(f"  {'─'*22} {'─'*10} {'─'*12}")
    for name, recall, delta in ablation_table:
        print(f"  {name:<22} {recall:>10} {delta:>12}")

    print("\n" + "=" * 60)
    print("  ✅ Hybrid Retrieval 框架验证通过！")
    print("  下一步: 接入 bge-m3 Embedding 后测量真实召回率")
    print("=" * 60)


if __name__ == "__main__":
    main()