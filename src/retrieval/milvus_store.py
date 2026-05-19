#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/18 23:23
@updated: 2026/5/18 23:23
@version: 1.0
@description: 
"""
"""
Milvus 向量库封装
================
统一管理笔记向量的存储和检索。

使用 Milvus Lite（本地文件存储），开发阶段零配置。
生产环境只需把 uri 改成 Milvus Standalone 地址。
"""

import json
from pathlib import Path
from loguru import logger

from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "xhs_notes"
EMBEDDING_DIM = 1024


class MilvusStore:
    def __init__(self, db_path: str = "data/milvus_lite.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(db_path)
        logger.info(f"Milvus 连接成功: {db_path}")

    def init_collection(self, drop_if_exists: bool = False) -> None:
        if drop_if_exists and self.client.has_collection(COLLECTION_NAME):
            self.client.drop_collection(COLLECTION_NAME)
            logger.info(f"已删除旧 Collection: {COLLECTION_NAME}")

        if self.client.has_collection(COLLECTION_NAME):
            logger.info(f"Collection 已存在: {COLLECTION_NAME}，跳过创建")
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("note_id", DataType.VARCHAR, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("metadata", DataType.VARCHAR, max_length=2048)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection 创建成功: {COLLECTION_NAME}")

    def insert(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        data = []
        for chunk in chunks:
            data.append({
                "chunk_id": chunk["chunk_id"],
                "note_id": chunk["note_id"],
                "text": chunk["text"][:4096],
                "embedding": chunk["embedding"],
                "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
            })
        result = self.client.insert(collection_name=COLLECTION_NAME, data=data)
        inserted = result.get("insert_count", len(data))
        logger.info(f"插入 {inserted} 条 chunks 到 Milvus")
        return inserted

    def search(self, query_embedding: list[float], top_k: int = 10, filter_expr: str | None = None) -> list[dict]:
        kwargs = dict(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["chunk_id", "note_id", "text", "metadata"],
        )
        if filter_expr:
            kwargs["filter"] = filter_expr

        results = self.client.search(**kwargs)
        hits = []
        for hit in results[0]:
            try:
                metadata = json.loads(hit["entity"].get("metadata", "{}"))
            except json.JSONDecodeError:
                metadata = {}
            hits.append({
                "chunk_id": hit["entity"]["chunk_id"],
                "note_id": hit["entity"]["note_id"],
                "text": hit["entity"]["text"],
                "score": round(hit["distance"], 4),
                "metadata": metadata,
            })
        return hits

    def delete_by_note_id(self, note_id: str) -> None:
        self.client.delete(collection_name=COLLECTION_NAME, filter=f'note_id == "{note_id}"')
        logger.info(f"已删除笔记 {note_id} 的所有 chunks")

    def count(self) -> int:
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        return int(stats.get("row_count", 0))

    def get_collection_info(self) -> dict:
        return {
            "name": COLLECTION_NAME,
            "count": self.count(),
            "db_path": self.db_path,
            "embedding_dim": EMBEDDING_DIM,
        }