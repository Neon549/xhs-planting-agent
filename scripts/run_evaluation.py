"""一次性脚本: 跑评估"""
# 模块 8 实现
"""
真实数据评估
运行: uv run python scripts/run_evaluation.py
"""


import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from evaluation.retrieval_eval import RetrievalEvaluator
from loguru import logger


class RealDenseRetriever:
    """真实 bge-m3 向量检索器。"""
    def __init__(self, model, store: MilvusStore):
        self.model = model
        self.store = store

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        output = self.model.encode(
            [query], batch_size=1, max_length=512,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        query_embedding = output["dense_vecs"][0].tolist()
        results = self.store.search(query_embedding=query_embedding, top_k=top_k)
        for i, r in enumerate(results):
            r["rank"] = i + 1
            r["source"] = "dense"
        return results


class HybridWrapper:
    def __init__(self, bm25, dense):
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        b = self.bm25.search(query, top_k=top_k)
        d = self.dense.search(query, top_k=top_k)
        return reciprocal_rank_fusion([b, d])[:top_k]


def main():
    print("=" * 60)
    print("  真实数据完整评估")
    print("=" * 60)

    # 读取真实 chunks
    chunks_file = Path("data/processed/chunks.json")
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    print(f"\n加载 {len(chunks)} 个 chunks")

    # 读取真实评估集
    eval_file = Path("data/eval/llm_eval_set.json")
    eval_set = json.loads(eval_file.read_text(encoding="utf-8"))
    print(f"评估集: {len(eval_set)} 条")

    # 加载模型
    print("\n加载 bge-m3...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # 建 BM25 索引
    print("建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    # 连接 Milvus
    store = MilvusStore(db_path="data/milvus_lite.db")
    dense = RealDenseRetriever(model=model, store=store)
    hybrid = HybridWrapper(bm25=bm25, dense=dense)

    # 评估
    evaluator = RetrievalEvaluator()
    evaluator.load_eval_set_from_list(eval_set)

    print("\n开始评估（需要几分钟）...")
    bm25_metrics = evaluator.evaluate(bm25, k=10)
    dense_metrics = evaluator.evaluate(dense, k=10)
    hybrid_metrics = evaluator.evaluate(hybrid, k=10)

    # 输出结果
    print(f"\n{'─'*55}")
    print(f"  {'配置':<25} {'Recall@10':>10} {'MRR':>8} {'NDCG@10':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8}")
    print(f"  {'纯 BM25':<25} {bm25_metrics['recall@10']:>10.4f} {bm25_metrics['mrr']:>8.4f} {bm25_metrics['ndcg@10']:>8.4f}")
    print(f"  {'纯 Dense (bge-m3)':<25} {dense_metrics['recall@10']:>10.4f} {dense_metrics['mrr']:>8.4f} {dense_metrics['ndcg@10']:>8.4f}")
    print(f"  {'BM25+Dense+RRF':<25} {hybrid_metrics['recall@10']:>10.4f} {hybrid_metrics['mrr']:>8.4f} {hybrid_metrics['ndcg@10']:>8.4f}")

    delta = hybrid_metrics['recall@10'] - bm25_metrics['recall@10']
    print(f"\n  混合检索 vs BM25: Recall@10 {delta:+.4f}")

    # 保存
    report = {
        "data_stats": {"chunks": len(chunks), "eval_set": len(eval_set)},
        "results": {
            "BM25": bm25_metrics,
            "Dense_bge_m3": dense_metrics,
            "Hybrid_BM25_Dense_RRF": hybrid_metrics,
        }
    }
    evaluator.save_report(report, "evaluation/reports/real_eval_report.json")
    print(f"\n  报告已保存: evaluation/reports/real_eval_report.json")

    print("\n" + "=" * 60)
    print("  ✅ 真实评估完成！把这些数字更新到 README")
    print("=" * 60)


if __name__ == "__main__":
    main()