"""BM25 关键词检索"""
# 模块 5 实现
"""
BM25 关键词检索器
================
基于词频统计的关键词检索，补充向量检索的精确匹配能力。

BM25 是 TF-IDF 的改进版：
    - TF（词频）：词在文档里出现越多，得分越高
    - IDF（逆文档频率）：词在所有文档里越稀有，得分越高
    - BM25 额外考虑文档长度归一化（长文档不占便宜）

对标: Elasticsearch 默认的打分算法就是 BM25。
实现: 用 rank_bm25 库（纯 Python，无需服务）。

"""

import jieba
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from loguru import logger


def tokenize(text: str) -> list[str]:
    """
    中文分词。

    使用 jieba 分词，过滤单字和停用词。
    jieba 是最主流的中文分词库，面试可以提。
    """
    tokens = jieba.cut(text)
    # 过滤单字（噪声多）和纯空格
    return [t for t in tokens if len(t) > 1 and t.strip()]


class BM25Retriever:
    """
    BM25 检索器。

    用法:
        retriever = BM25Retriever()
        retriever.build_index(chunks)        # 建索引
        results = retriever.search("手机推荐", top_k=20)
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []       # 原始 chunk 列表
        self.tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[dict]) -> None:
        """
        基于 chunks 建立 BM25 索引。

        chunks 里每个元素需要有 "text" 字段。
        """
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 索引建立完成: {len(chunks)} 个 chunks")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 检索。

        返回:
            [{"chunk_id": str, "note_id": str, "text": str,
              "score": float, "metadata": dict, "rank": int}, ...]
        """
        if self.bm25 is None:
            raise RuntimeError("请先调用 build_index() 建立索引")

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning(f"query '{query}' 分词后为空，返回空结果")
            return []

        scores = self.bm25.get_scores(query_tokens)

        # 取 Top-K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break   # BM25 分数为 0 表示完全不相关
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "note_id": chunk["note_id"],
                "text": chunk["text"],
                "score": round(float(scores[idx]), 4),
                "metadata": chunk.get("metadata", {}),
                "rank": rank + 1,
                "source": "bm25",
            })

        logger.debug(f"BM25 检索: '{query}' → {len(results)} 条结果")
        return results

    def save_index(self, path: str) -> None:
        """保存索引到文件（避免每次重建）。"""
        import pickle
        data = {
            "bm25": self.bm25,
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }
        Path(path).write_bytes(pickle.dumps(data))
        logger.info(f"BM25 索引已保存: {path}")

    def load_index(self, path: str) -> None:
        """从文件加载索引。"""
        import pickle
        data = pickle.loads(Path(path).read_bytes())
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        logger.info(f"BM25 索引已加载: {path} ({len(self.chunks)} chunks)")