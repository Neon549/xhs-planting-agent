"""Hybrid 检索: BM25 + Dense + Reranker"""

# 模块 5 实现
"""
混合检索器（Hybrid Retriever）
==============================
整合 BM25 + Dense + RRF 融合 + Reranker，实现三阶段检索。

这是整个 RAG 系统最核心的模块，召回率的关键所在。

三阶段流程:
    1. 双路粗召回: BM25(Top-20) + Dense(Top-20) → 候选集 ~30条
    2. RRF 融合:   倒数排名融合 → 综合排名
    3. Reranker:   Cross-Encoder 精排 → 最终 Top-10

RRF（Reciprocal Rank Fusion）公式:
    score(d) = Σ 1/(k + rank_i(d))    k=60（常数，防止排名靠后的文档被过度惩罚）
    两路结果独立排名，融合后综合排名更鲁棒。

"""

from loguru import logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import create_reranker
from src.retrieval.reranker import BaseReranker

reranker = create_reranker()


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    倒数排名融合（RRF）。

    参数:
        result_lists: 多路检索结果列表，每路已按分数排好序
        k: RRF 常数，通常取 60

    返回:
        按 RRF 分数排序的合并列表（去重，保留来源信息）
    """
    # chunk_id → RRF 分数
    rrf_scores: dict[str, float] = {}
    # chunk_id → chunk 数据（保留原始信息）
    chunk_map: dict[str, dict] = {}
    # chunk_id → 来源列表
    sources_map: dict[str, list[str]] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            chunk_id = chunk["chunk_id"]

            # RRF 公式
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score

            # 保存 chunk 数据
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
                sources_map[chunk_id] = []
            sources_map[chunk_id].append(chunk.get("source", "unknown"))

    # 排序
    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )

    results = []
    for rank, chunk_id in enumerate(sorted_ids):
        chunk = dict(chunk_map[chunk_id])
        chunk["rrf_score"] = round(rrf_scores[chunk_id], 6)
        chunk["sources"] = sources_map[chunk_id]  # 记录这个 chunk 被哪几路召回
        chunk["rank"] = rank + 1
        results.append(chunk)

    return results


class HybridRetriever:
    """
    混合检索器，整合三阶段流程。

    用法:
        retriever = HybridRetriever(
            bm25=bm25_retriever,
            dense=dense_retriever,
            reranker=reranker,
        )

        results = retriever.search(
            query="iPhone 16 Pro 值得买吗",
            top_k=10,
            use_reranker=True,
        )
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: BaseReranker | None = None,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k

    def search(
            self,
            query: str,
            top_k: int = 10,
            use_reranker: bool = True,
            trace: "TraceContext | None" = None,
    ) -> list[dict]:
        logger.info(f"混合检索开始: '{query}' | top_k={top_k} | reranker={use_reranker}")

        # 阶段1：双路粗召回
        bm25_results = self.bm25.search(query, top_k=self.bm25_top_k)
        dense_results = self.dense.search(query, top_k=self.dense_top_k)
        if trace:
            trace.record("bm25", {"count": len(bm25_results),
                                  "top3_ids": [r["chunk_id"] for r in bm25_results[:3]]})
            trace.record("dense", {"count": len(dense_results),
                                   "top3_ids": [r["chunk_id"] for r in dense_results[:3]]})

        logger.debug(f"粗召回: BM25={len(bm25_results)} | Dense={len(dense_results)}")

        # 阶段2：RRF融合
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        both_recalled = sum(1 for c in fused if len(c.get("sources", [])) == 2)
        if trace:
            trace.record("rrf", {"fused_count": len(fused), "both_recalled": both_recalled})

        # 阶段3：Reranker精排
        if use_reranker and self.reranker is not None:
            rerank_candidates = fused[:30]
            results = self.reranker.rerank(query, rerank_candidates, top_k=top_k)
            if trace:
                top_score = results[0].get("rerank_score", 0) if results else 0
                trace.record("rerank", {"top_score": top_score, "count": len(results)})
        else:
            results = fused[:top_k]

        if trace:
            trace.save()

        logger.info(f"混合检索完成: '{query}' → {len(results)} 条")
        return results

    def search_ablation(self, query: str, top_k: int = 10) -> dict:
        """
        消融实验：对比三种配置的结果。

        返回:
        {
            "bm25_only": [...],     # 纯 BM25
            "dense_only": [...],    # 纯 Dense
            "hybrid_no_rerank": [...],  # BM25+Dense+RRF
            "hybrid_rerank": [...],     # 完整三阶段
        }

        用于评估报告，量化各组件的贡献。
        """
        logger.info(f"消融实验: '{query}'")

        bm25_results = self.bm25.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        hybrid_no_rerank = fused[:top_k]

        hybrid_rerank = []
        if self.reranker is not None:
            hybrid_rerank = self.reranker.rerank(query, fused[:30], top_k=top_k)

        return {
            "bm25_only": bm25_results[:top_k],
            "dense_only": dense_results[:top_k],
            "hybrid_no_rerank": hybrid_no_rerank,
            "hybrid_rerank": hybrid_rerank,
        }
