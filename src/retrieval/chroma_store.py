# src/retrieval/chroma_store.py
import chromadb
from pathlib import Path
from loguru import logger


class ChromaStore:
    """
    ChromaDB 向量存储，接口与 MilvusStore 保持一致。
    替换 Milvus Lite，支持本地持久化。
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "xhs_notes",
        embedding_dim: int = 1024,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_dim = embedding_dim
        logger.info(f"ChromaStore 初始化: {persist_dir}/{collection_name}")

    def insert(self, records: list[dict]) -> int:
        """
        批量插入，records 格式：
        [{"chunk_id", "note_id", "text", "embedding", "metadata"}]
        """
        if not records:
            return 0

        seen = {}
        for r in records:
            seen[r["chunk_id"]] = r
        records = list(seen.values())
        logger.info(f"去重后: {len(records)} 条")

        ids = [r["chunk_id"] for r in records]
        embeddings = [r["embedding"] for r in records]
        documents = [r["text"] for r in records]
        metadatas = []
        for r in records:
            # ChromaDB metadata 只支持 str/int/float/bool
            meta = r.get("metadata", {})
            clean_meta = {
                "note_id": r["note_id"],
                "title": str(meta.get("title", "")),
                "author": str(meta.get("author", "")),
                "category": str(meta.get("category", "")),
                "liked_count": int(meta.get("liked_count", 0)),
                "comment_count": int(meta.get("comment_count", 0)),
            }
            metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"ChromaStore 插入 {len(records)} 条")
        return len(records)

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        """
        向量相似度检索，返回格式与 MilvusStore.search() 一致。
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance → similarity score
            score = 1 - dist
            output.append({
                "chunk_id": chunk_id,
                "note_id": meta.get("note_id", ""),
                "text": text,
                "score": round(float(score), 4),
                "metadata": meta,
            })

        return output

    def count(self) -> int:
        return self.collection.count()

    def delete_all(self) -> None:
        """清空集合（重建索引时用）。"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaStore 已清空")