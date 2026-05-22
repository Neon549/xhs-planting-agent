"""一次性脚本: 构建 Milvus 索引"""
# 模块 4 实现
"""
用真实 bge-m3 重建 Milvus 索引
运行: uv run python scripts/build_index.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from src.retrieval.milvus_store import MilvusStore
from loguru import logger


def main():
    print("=" * 60)
    print("  用真实 bge-m3 重建 Milvus 索引")
    print("=" * 60)

    # 读取 chunks
    chunks_file = Path("data/processed/chunks.json")
    if not chunks_file.exists():
        print("❌ 找不到 chunks 文件，请先运行 crawl_data.py")
        return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    print(f"\n读取 {len(chunks)} 个 chunks")

    # 加载 bge-m3
    print("\n加载 bge-m3 模型（首次运行会下载 ~2.2GB）...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("模型加载完成")

    # 批量 embedding
    print("\n开始 Embedding...")
    texts = [c["text"] for c in chunks]
    output = model.encode(
        texts,
        batch_size=16,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = output["dense_vecs"]
    print(f"Embedding 完成: {len(embeddings)} 条，维度: {len(embeddings[0])}")

    # 重建 Milvus
    print("\n重建 Milvus 索引...")
    # 先删掉旧数据库
    import shutil
    db_path = Path("data/milvus_lite.db")
    if db_path.exists():
        shutil.rmtree(db_path)
        print("已删除旧索引")

    store = MilvusStore(db_path="data/milvus_lite.db")
    store.init_collection(drop_if_exists=False)

    # 插入
    milvus_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        milvus_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "note_id": chunk["note_id"],
            "text": chunk["text"],
            "embedding": embedding.tolist(),
            "metadata": chunk["metadata"],
        })

    inserted = store.insert(milvus_chunks)
    print(f"插入 {inserted} 条到 Milvus")

    print("\n" + "=" * 60)
    print(f"  ✅ 真实索引构建完成！共 {store.count()} 条向量")
    print("=" * 60)


if __name__ == "__main__":
    main()