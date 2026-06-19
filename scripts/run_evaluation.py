"""
真实数据完整评估（含 Reranker）
运行: python scripts/run_evaluation.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.chroma_store import ChromaStore
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from evaluation.retrieval_eval import RetrievalEvaluator
from config.settings import VECTOR_STORE, EMBEDDING
from loguru import logger


class RealDenseRetriever:
    def __init__(self, model, store: ChromaStore):
        self.model = model
        self.store = store

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        output = self.model.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
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


class HybridWithRerankerWrapper:
    def __init__(self, bm25, dense, reranker):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        b = self.bm25.search(query, top_k=20)
        d = self.dense.search(query, top_k=20)
        fused = reciprocal_rank_fusion([b, d])[:30]
        return self.reranker.rerank(query, fused, top_k=top_k)


def main():
    print("=" * 60)
    print("  真实数据完整评估（含 Reranker）")
    print("=" * 60)

    # 读取数据
    chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))
    eval_set = json.loads(
        Path("data/eval/llm_eval_set.json").read_text(encoding="utf-8")
    )
    print(f"\nChunks: {len(chunks)} | 评估集: {len(eval_set)}")

    # 加载模型
    print(f"\n加载 {EMBEDDING['model']}...")
    embed_model = BGEM3FlagModel(EMBEDDING["model"], use_fp16=EMBEDDING["use_fp16"])

    print("加载 bge-reranker-v2-m3...")
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    reranker._load_model()

    # 构建检索器
    print("构建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.load_index("data/bm25_index.pkl")

    store = ChromaStore(
        persist_dir=VECTOR_STORE["persist_dir"],
        collection_name=VECTOR_STORE["collection_name"],
    )
    dense = RealDenseRetriever(model=embed_model, store=store)
    hybrid = HybridWrapper(bm25=bm25, dense=dense)
    hybrid_rerank = HybridWithRerankerWrapper(bm25=bm25, dense=dense, reranker=reranker)

    # 评估
    evaluator = RetrievalEvaluator()
    evaluator.load_eval_set_from_list(eval_set)

    print("\n开始评估...")
    print("  评估 BM25...")
    bm25_metrics = evaluator.evaluate(bm25, k=10)
    print("  评估 Dense...")
    dense_metrics = evaluator.evaluate(dense, k=10)
    print("  评估 Hybrid...")
    hybrid_metrics = evaluator.evaluate(hybrid, k=10)
    print("  评估 Hybrid + Reranker...")
    rerank_metrics = evaluator.evaluate(hybrid_rerank, k=10)

    # 输出结果
    print(f"\n{'─'*60}")
    print(f"  {'配置':<28} {'Recall@10':>10} {'MRR':>8} {'NDCG@10':>8}")
    print(f"  {'─'*28} {'─'*10} {'─'*8} {'─'*8}")
    for name, m in [
        ("纯 BM25", bm25_metrics),
        ("纯 Dense (bge-m3)", dense_metrics),
        ("BM25+Dense+RRF", hybrid_metrics),
        ("BM25+Dense+RRF+Reranker", rerank_metrics),
    ]:
        print(f"  {name:<28} {m['recall@10']:>10.4f} {m['mrr']:>8.4f} {m['ndcg@10']:>8.4f}")

    delta = rerank_metrics["recall@10"] - bm25_metrics["recall@10"]
    print(f"\n  Reranker vs BM25: Recall@10 {delta:+.4f}")

    # 保存报告
    report = {
        "data_stats": {"chunks": len(chunks), "eval_set": len(eval_set)},
        "results": {
            "BM25": bm25_metrics,
            "Dense_bge_m3": dense_metrics,
            "Hybrid_BM25_Dense_RRF": hybrid_metrics,
            "Hybrid_BM25_Dense_RRF_Reranker": rerank_metrics,
        },
    }
    Path("evaluation/reports").mkdir(parents=True, exist_ok=True)
    Path("evaluation/reports/full_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  报告已保存: evaluation/reports/full_eval_report.json")
    print("\n" + "=" * 60)
    print("  ✅ 完整评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()