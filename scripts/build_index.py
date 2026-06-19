#!/usr/bin/env python3
"""重建 ChromaDB 向量索引"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from src.retrieval.chroma_store import ChromaStore
from config.settings import VECTOR_STORE, EMBEDDING
from loguru import logger


def main():
    print("=" * 60)
    print("  用真实 bge-m3 重建 ChromaDB 索引")
    print("=" * 60)

    # 读取 chunks
    chunks_file = Path("data/processed/chunks.json")
    if not chunks_file.exists():
        print("❌ 找不到 chunks 文件，请先运行 crawl_data.py")
        return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    print(f"\n读取 {len(chunks)} 个 chunks")

    # 加载 bge-m3
    print(f"\n加载 Embedding 模型: {EMBEDDING['model']} ...")
    model = BGEM3FlagModel(EMBEDDING["model"], use_fp16=EMBEDDING["use_fp16"])
    print("模型加载完成")

    # 批量 Embedding
    print(f"\n开始 Embedding（batch_size={EMBEDDING['batch_size']}）...")
    texts = [c["text"] for c in chunks]
    output = model.encode(
        texts,
        batch_size=EMBEDDING["batch_size"],
        max_length=EMBEDDING["max_length"],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = output["dense_vecs"]
    print(f"Embedding 完成: {len(embeddings)} 条，维度: {len(embeddings[0])}")

    # 初始化 ChromaStore 并清空旧数据
    print("\n初始化 ChromaDB...")
    store = ChromaStore(
        persist_dir=VECTOR_STORE["persist_dir"],
        collection_name=VECTOR_STORE["collection_name"],
        embedding_dim=VECTOR_STORE["embedding_dim"],
    )
    store.delete_all()
    print("旧索引已清空")

    # 插入
    print("\n插入到 ChromaDB...")
    records = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append({
            "chunk_id": chunk["chunk_id"],
            "note_id": chunk["note_id"],
            "text": chunk["text"],
            "embedding": embedding.tolist(),
            "metadata": chunk.get("metadata", {}),
        })

    inserted = store.insert(records)

    print("\n" + "=" * 60)
    print(f"  ✅ ChromaDB 索引构建完成！共 {store.count()} 条向量")
    print(f"  📁 存储路径: {VECTOR_STORE['persist_dir']}")
    print("=" * 60)

if __name__ == "__main__":
    main()