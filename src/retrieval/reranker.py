"""bge-reranker-v2-m3 重排序"""
# 模块 5 实现
"""
Reranker 精排模块
================
用 Cross-Encoder 对混合召回的候选集重新打分排序。

模型: bge-reranker-v2-m3（BAAI/bge-reranker-v2-m3）
原理:
    Bi-Encoder（Dense 检索）: query 和 doc 各自编码成向量，点积算相似度
        → 快，可以预计算，但交互信息少
    Cross-Encoder（Reranker）: [query, doc] 拼接后一起编码，输出相关性分数
        → 慢，不能预计算，但精度高（直接建模交互）

为什么要两阶段:
    候选集 2000 条 → Bi-Encoder 粗排 → 候选集 30 条 → Cross-Encoder 精排 → Top-10
    不能对所有 2000 条跑 Cross-Encoder，太慢。

面试话术:
    "检索分两阶段：Bi-Encoder 粗排（速度快，可预计算向量），
     Cross-Encoder 精排（精度高，只对候选集 ~30 条调用）。
     bge-reranker-v2-m3 在 CMTEB 重排序榜上是最优开源模型之一。"
"""

from loguru import logger


class Reranker:
    """
    bge-reranker-v2-m3 精排器。

    用法:
        reranker = Reranker()
        reranker.load_model()

        # 输入: query + 候选 chunks 列表
        results = reranker.rerank(
            query="手机推荐",
            candidates=[{"chunk_id": "...", "text": "...", ...}, ...],
            top_k=10,
        )
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """
        加载 Reranker 模型。

        bge-reranker-v2-m3 约 1.1GB，比 Embedding 模型小。
        同样支持 HF_ENDPOINT 镜像加速。
        """
        from FlagEmbedding import FlagReranker
        logger.info(f"加载 Reranker 模型: {self.model_name} ...")
        self.model = FlagReranker(
            self.model_name,
            use_fp16=True,
        )
        logger.info("Reranker 模型加载完成")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        对候选集精排。

        参数:
            query: 用户查询
            candidates: 混合召回的候选 chunks
            top_k: 精排后返回的数量

        返回:
            精排后的 Top-K chunks，新增 "rerank_score" 字段
        """
        if not candidates:
            return []

        if self.model is None:
            self.load_model()

        # 构造 (query, doc) 对，这是 Cross-Encoder 的输入格式
        pairs = [(query, c["text"]) for c in candidates]

        # 计算相关性分数
        # normalize=True: 把分数归一化到 [0, 1]，便于理解
        scores = self.model.compute_score(pairs, normalize=True)

        # 如果只有一对，compute_score 返回单个 float，统一成列表
        if isinstance(scores, float):
            scores = [scores]

        # 合并分数，排序
        scored = []
        for chunk, score in zip(candidates, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = round(float(score), 4)
            scored.append(chunk_copy)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = scored[:top_k]

        logger.debug(
            f"Reranker: {len(candidates)} 候选 → Top-{top_k} | "
            f"最高分: {top_results[0]['rerank_score'] if top_results else 'N/A'}"
        )
        return top_results