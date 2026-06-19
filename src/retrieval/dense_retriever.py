"""向量检索 (基于 Milvus)"""

# 模块 5 实现
"""
Dense 向量检索器
===============
用 Embedding 模型把文本转成向量，在 Milvus 里做语义相似度搜索。

Embedding 模型选型: bge-m3（BAAI/bge-m3）
    - 支持中英文双语
    - 1024 维输出
    - 在 MTEB 排行榜上中文效果最好的开源模型之一
    - 支持最长 8192 token（小红书笔记完全够用）
"""

from loguru import logger
from src.retrieval.milvus_store import MilvusStore


class HFApiDenseRetriever:
    """
    用 HF Inference API 做 embedding，不需要本地模型。
    部署到 HF Spaces 时使用。
    """

    def __init__(self, store: MilvusStore, hf_token: str = None):
        self.store = store
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        result = response.json()
        # 返回第一个向量
        if isinstance(result, list) and isinstance(result[0], list):
            return result[0]
        return result

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        embedding = self.embed(query)
        results = self.store.search(query_embedding=embedding, top_k=top_k)
        for i, r in enumerate(results):
            r["rank"] = i + 1
            r["source"] = "dense"
        return results


class DenseRetriever:
    """
    向量检索器。

    用法:
        retriever = DenseRetriever(milvus_store=store)
        retriever.load_model()      # 加载 Embedding 模型

        # 建库时: 文本 → 向量 → 存 Milvus
        embedding = retriever.embed(text)
        store.insert([{"chunk_id":..., "embedding": embedding, ...}])

        # 检索时: query → 向量 → Milvus 搜索
        results = retriever.search("手机推荐", top_k=20)
    """

    def __init__(self, milvus_store: MilvusStore, model_name: str = "BAAI/bge-m3"):
        self.store = milvus_store
        self.model_name = model_name
        self.model = None  # 懒加载，用到时才加载

    def load_model(self) -> None:
        """
        加载 Embedding 模型（bge-m3）。

        bge-m3 约 2.2GB，第一次会自动下载到 ~/.cache/huggingface/
        如果下载慢，可以设置:
            export HF_ENDPOINT=https://hf-mirror.com
        """
        from FlagEmbedding import BGEM3FlagModel

        logger.info(f"加载 Embedding 模型: {self.model_name} ...")
        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=True,  # 半精度，减少显存占用，速度更快
        )
        logger.info("Embedding 模型加载完成")

    def embed(self, text: str) -> list[float]:
        """
        单条文本 → 1024 维向量。

        返回归一化后的向量（适合 COSINE 相似度计算）。
        """
        if self.model is None:
            self.load_model()

        output = self.model.encode(
            [text],
            batch_size=1,
            max_length=512,  # bge-m3 建议 512 以内，超出截断
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        批量文本 → 向量列表（建库时用，比单条快很多）。
        """
        if self.model is None:
            self.load_model()

        logger.info(f"批量 Embedding: {len(texts)} 条文本，batch_size={batch_size}")
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [vec.tolist() for vec in output["dense_vecs"]]

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        语义向量检索。

        返回格式与 BM25Retriever.search() 相同，便于后续融合。
        """
        query_embedding = self.embed(query)
        raw_results = self.store.search(query_embedding=query_embedding, top_k=top_k)

        # 统一格式，加上 rank 和 source 字段
        results = []
        for rank, r in enumerate(raw_results):
            results.append(
                {
                    "chunk_id": r["chunk_id"],
                    "note_id": r["note_id"],
                    "text": r["text"],
                    "score": r["score"],
                    "metadata": r["metadata"],
                    "rank": rank + 1,
                    "source": "dense",
                }
            )

        logger.debug(f"Dense 检索: '{query}' → {len(results)} 条结果")
        return results
