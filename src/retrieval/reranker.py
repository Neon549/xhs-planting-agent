"""
Reranker 模块
=============
可插拔的重排序组件，支持三种模式：
    - none：不重排，原样返回
    - cross_encoder：bge-reranker 本地模型
    - llm：用 LLM 打分（后续扩展）

切换方式：修改 config/settings.py 里的 RERANKER["provider"]
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from loguru import logger


# ── 抽象基类 ──────────────────────────────────────────────────────
class BaseReranker(ABC):
    """所有 Reranker 的抽象基类，定义统一接口。"""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """对候选集精排，返回 Top-K。"""
        ...

    def validate(self, query: str, candidates: list[dict]) -> None:
        """输入校验，子类自动继承。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates 不能为空列表")


# ── 实现 1：不重排 ────────────────────────────────────────────────
class NoneReranker(BaseReranker):
    """空对象模式：不重排，原样返回 Top-K。"""

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        self.validate(query, candidates)
        logger.debug(f"NoneReranker: 跳过重排，返回前 {top_k} 条")
        return candidates[:top_k]


# ── 实现 2：Cross-Encoder 重排 ────────────────────────────────────
class CrossEncoderReranker(BaseReranker):
    """bge-reranker-v2-m3 精排。"""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.model = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from FlagEmbedding import FlagReranker

        logger.info(f"加载 Reranker 模型: {self.model_name}")
        self.model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        self.validate(query, candidates)
        self._load_model()

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)

        if isinstance(scores, float):
            scores = [scores]

        scored = []
        for chunk, score in zip(candidates, scores):
            c = dict(chunk)
            c["rerank_score"] = round(float(score), 4)
            scored.append(c)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(
            f"CrossEncoderReranker: {len(candidates)} → Top-{top_k} | 最高分: {scored[0]['rerank_score']}"
        )
        return scored[:top_k]


# ── 实现 3：LLM 重排（预留，后续扩展）─────────────────────────────
class LLMReranker(BaseReranker):
    """用 LLM 打相关性分数重排（待实现）。"""

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        self.validate(query, candidates)
        logger.warning("LLMReranker 尚未实现，降级到 NoneReranker")
        return candidates[:top_k]


# ── Factory ───────────────────────────────────────────────────────
def create_reranker(config: dict | None = None) -> BaseReranker:
    """
    根据配置创建 Reranker 实例。

    用法:
        from config.settings import RERANKER
        from src.retrieval.reranker import create_reranker

        reranker = create_reranker(RERANKER)

    config 示例:
        {"provider": "cross_encoder", "model": "BAAI/bge-reranker-v2-m3", "top_k": 10}
    """
    if config is None:
        from config.settings import RERANKER

        config = RERANKER

    provider = config.get("provider", "none")
    logger.info(f"创建 Reranker: provider={provider}")

    if provider == "cross_encoder":
        return CrossEncoderReranker(
            model_name=config.get("model", "BAAI/bge-reranker-v2-m3"),
            use_fp16=config.get("use_fp16", True),
        )
    elif provider == "llm":
        return LLMReranker()
    else:
        return NoneReranker()
