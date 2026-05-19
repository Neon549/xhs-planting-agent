#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/18 23:59
@updated: 2026/5/18 23:59
@version: 1.0
@description: 
"""
"""
评估体系验证 Demo
=================
用 mock 数据跑完整的评估流程，生成对比报告。

运行: uv run python scripts/demo_evaluation.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.retrieval_eval import RetrievalEvaluator, recall_at_k, reciprocal_rank, ndcg_at_k
from evaluation.eval_set_generator import EvalSetGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion

# ── Mock 数据 ─────────────────────────────────────────────────
MOCK_NOTES = [
    {"note_id": "3c_001", "title": "iPhone 16 Pro 用了三个月真实感受",
     "full_text": "标题: iPhone 16 Pro 用了三个月真实感受\n正文: A18 Pro 性能强，电池差，价格贵。",
     "category": "数码3C"},
    {"note_id": "3c_002", "title": "安卓旗舰横评小米华为三星",
     "full_text": "标题: 安卓旗舰横评\n正文: 性能小米强，相机华为好，续航华为最长。",
     "category": "数码3C"},
    {"note_id": "fitness_001", "title": "家用哑铃怎么选避坑指南",
     "full_text": "标题: 家用哑铃怎么选\n正文: 入门买固定套装，进阶买可调节哑铃。",
     "category": "健身器材"},
    {"note_id": "fitness_002", "title": "瑜伽垫选购TPE还是橡胶",
     "full_text": "标题: 瑜伽垫选购\n正文: TPE轻便，天然橡胶防滑最好但有气味。",
     "category": "健身器材"},
    {"note_id": "3c_003", "title": "轻薄笔记本推荐5000以内",
     "full_text": "标题: 轻薄笔记本推荐\n正文: 重量1.3kg，续航12小时，性价比高。",
     "category": "数码3C"},
]

MOCK_CHUNKS = [
    {"chunk_id": f"{n['note_id']}_0", "note_id": n["note_id"],
     "text": n["full_text"], "metadata": {"category": n["category"]}}
    for n in MOCK_NOTES
]

# ── 手动构造评估集（模拟标注数据）────────────────────────────
MOCK_EVAL_SET = [
    {"query": "iPhone推荐", "relevant_note_ids": ["3c_001"]},
    {"query": "安卓手机横评", "relevant_note_ids": ["3c_002"]},
    {"query": "家用哑铃选购", "relevant_note_ids": ["fitness_001"]},
    {"query": "瑜伽垫怎么选", "relevant_note_ids": ["fitness_002"]},
    {"query": "轻薄笔记本", "relevant_note_ids": ["3c_003"]},
    {"query": "数码好物推荐", "relevant_note_ids": ["3c_001", "3c_002", "3c_003"]},
    {"query": "健身器材推荐", "relevant_note_ids": ["fitness_001", "fitness_002"]},
]


def main():
    print("=" * 60)
    print("  评估体系验证 Demo")
    print("=" * 60)

    # ── Step 1：生成评估集 ────────────────────────────────────
    print("\n【Step 1】规则生成评估集")
    generator = EvalSetGenerator(llm_client=None)
    auto_eval_set = generator.generate_rule_based(MOCK_NOTES)
    print(f"  自动生成: {len(auto_eval_set)} 条")
    print(f"  示例: {auto_eval_set[0]}")

    # ── Step 2：指标计算演示 ──────────────────────────────────
    print("\n【Step 2】指标计算演示")
    retrieved = ["3c_001", "3c_003", "fitness_001", "3c_002", "fitness_002"]
    relevant = ["3c_001", "3c_002"]

    print(f"  召回结果: {retrieved}")
    print(f"  相关文档: {relevant}")
    print(f"  Recall@3 = {recall_at_k(retrieved, relevant, k=3):.4f}")
    print(f"  Recall@5 = {recall_at_k(retrieved, relevant, k=5):.4f}")
    print(f"  MRR      = {reciprocal_rank(retrieved, relevant):.4f}")
    print(f"  NDCG@5   = {ndcg_at_k(retrieved, relevant, k=5):.4f}")

    # ── Step 3：消融实验 ──────────────────────────────────────
    print("\n【Step 3】消融实验（BM25 vs Hybrid）")

    # BM25
    bm25 = BM25Retriever()
    bm25.build_index(MOCK_CHUNKS)

    # Mock Dense（随机召回，模拟语义检索）
    import random
    class MockDense:
        def search(self, query, top_k=10):
            shuffled = MOCK_CHUNKS.copy()
            random.shuffle(shuffled)
            return [{**c, "score": random.uniform(0.6, 0.95),
                     "rank": i+1, "source": "dense"}
                    for i, c in enumerate(shuffled[:top_k])]

    # Hybrid（BM25 + MockDense + RRF）
    class HybridWrapper:
        def __init__(self, bm25, dense):
            self.bm25 = bm25
            self.dense = dense

        def search(self, query, top_k=10):
            b = self.bm25.search(query, top_k=top_k)
            d = self.dense.search(query, top_k=top_k)
            fused = reciprocal_rank_fusion([b, d])
            return fused[:top_k]

    hybrid = HybridWrapper(bm25, MockDense())

    # 评估
    evaluator = RetrievalEvaluator()
    evaluator.load_eval_set_from_list(MOCK_EVAL_SET)

    bm25_metrics = evaluator.evaluate(bm25, k=5)
    hybrid_metrics = evaluator.evaluate(hybrid, k=5)

    # ── Step 4：打印对比报告 ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  {'配置':<20} {'Recall@5':>10} {'MRR':>8} {'NDCG@5':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*8}")
    print(f"  {'纯 BM25':<20} {bm25_metrics['recall@5']:>10.4f} {bm25_metrics['mrr']:>8.4f} {bm25_metrics['ndcg@5']:>8.4f}")
    print(f"  {'BM25+Dense+RRF':<20} {hybrid_metrics['recall@5']:>10.4f} {hybrid_metrics['mrr']:>8.4f} {hybrid_metrics['ndcg@5']:>8.4f}")

    delta_recall = hybrid_metrics['recall@5'] - bm25_metrics['recall@5']
    delta_mrr = hybrid_metrics['mrr'] - bm25_metrics['mrr']
    print(f"\n  混合检索提升: Recall@5 {delta_recall:+.4f} | MRR {delta_mrr:+.4f}")

    # 保存报告
    report = {
        "eval_set_size": len(MOCK_EVAL_SET),
        "k": 5,
        "results": {
            "BM25": bm25_metrics,
            "Hybrid(BM25+Dense+RRF)": hybrid_metrics,
        },
        "delta": {
            "recall@5": round(delta_recall, 4),
            "mrr": round(delta_mrr, 4),
        }
    }
    evaluator.save_report(report, "evaluation/reports/ablation_report.json")
    print(f"\n  报告已保存: evaluation/reports/ablation_report.json")

    print("\n" + "=" * 60)
    print("  ✅ 评估体系验证通过！")
    print("  下一步: 接入真实数据后更新报告，把数字写进 README")
    print("=" * 60)


if __name__ == "__main__":
    main()