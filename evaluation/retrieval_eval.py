"""检索评估: Recall@K, MRR, NDCG"""
# 模块 8 实现
"""
检索评估模块
============
计算 Recall@K、MRR、NDCG@K，对比三种检索配置。

核心概念:
    ground_truth: {query: [relevant_note_id_1, relevant_note_id_2, ...]}
    retrieved:    {query: [retrieved_note_id_1, ..., retrieved_note_id_K]}

面试话术:
    "构建了 200 条标注评估集，对比 BM25、Dense、Hybrid+Reranker
     三种配置在 Recall@10 和 MRR 上的表现，
     定量验证了混合检索相比单路的提升。"
"""

import math
import json
from pathlib import Path
from loguru import logger


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Recall@K：前 K 个结果里，召回了多少相关文档。

    公式: |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Reciprocal Rank：第一个相关结果排在第几位的倒数。

    公式: 1 / rank_of_first_relevant
    没有相关结果则返回 0。
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    NDCG@K：归一化折损累计增益。

    考虑排名位置，相关文档排越前得分越高。
    公式: DCG@K / IDCG@K
    """
    relevant_set = set(relevant_ids)

    def dcg(ids, k):
        score = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            if doc_id in relevant_set:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids, k)
    # IDCG：理想情况下（相关文档全排最前）的 DCG
    ideal_ids = list(relevant_set) + [""] * k
    ideal_dcg = dcg(ideal_ids, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


class RetrievalEvaluator:
    """
    检索评估器。

    用法:
        evaluator = RetrievalEvaluator()

        # 加载评估集
        evaluator.load_eval_set("data/eval/eval_set.json")

        # 评估某个检索配置
        results = evaluator.evaluate(retriever=bm25_retriever, k=10)
        print(results)
        # {"recall@10": 0.65, "mrr": 0.58, "ndcg@10": 0.61}
    """

    def __init__(self):
        self.eval_set: list[dict] = []   # [{query, relevant_note_ids}, ...]

    def load_eval_set(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.eval_set = data
        logger.info(f"评估集加载: {len(self.eval_set)} 条")

    def load_eval_set_from_list(self, data: list[dict]) -> None:
        self.eval_set = data
        logger.info(f"评估集加载: {len(self.eval_set)} 条")

    def evaluate(self, retriever, k: int = 10) -> dict:
        """
        对一个检索器跑完整评估。

        retriever 需要有 search(query, top_k) -> list[dict] 方法。
        """
        recall_scores, mrr_scores, ndcg_scores = [], [], []

        for item in self.eval_set:
            query = item["query"]
            relevant_ids = item["relevant_note_ids"]

            results = retriever.search(query, top_k=k)
            # 从 chunk_id 提取 note_id（一篇笔记可能有多个 chunk）
            retrieved_note_ids = list(dict.fromkeys(
                r["note_id"] for r in results
            ))

            recall_scores.append(recall_at_k(retrieved_note_ids, relevant_ids, k))
            mrr_scores.append(reciprocal_rank(retrieved_note_ids, relevant_ids))
            ndcg_scores.append(ndcg_at_k(retrieved_note_ids, relevant_ids, k))

        metrics = {
            f"recall@{k}": round(sum(recall_scores) / len(recall_scores), 4),
            "mrr": round(sum(mrr_scores) / len(mrr_scores), 4),
            f"ndcg@{k}": round(sum(ndcg_scores) / len(ndcg_scores), 4),
            "num_queries": len(self.eval_set),
        }
        return metrics

    def compare_retrievers(self, retrievers: dict, k: int = 10) -> dict:
        """
        对比多个检索器。

        retrievers: {"BM25": bm25_retriever, "Hybrid": hybrid_retriever}
        返回对比表格数据。
        """
        results = {}
        for name, retriever in retrievers.items():
            logger.info(f"评估: {name}")
            metrics = self.evaluate(retriever, k=k)
            results[name] = metrics
            logger.info(f"{name}: {metrics}")
        return results

    def save_report(self, results: dict, path: str) -> None:
        """保存评估报告为 JSON。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"评估报告已保存: {path}")