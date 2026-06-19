#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/6/19 15:57
@updated: 2026/6/19 15:57
@version: 1.0
@description: 
"""
class XHSIngestionPipeline:
    """
    小红书数据入库流水线
    chunks.json → Embedding → ChromaStore + BM25Index
    """
    def run(self, chunks_path="data/processed/chunks.json",
            on_progress=None) -> IngestionResult:
        # 1. Load chunks
        # 2. DenseEncoder 批量 embed（bge-m3）
        # 3. ChromaStore.insert()
        # 4. BM25Indexer.build()
        # 5. 记录 Trace
        # 每次查询自动记录
        {
            "trace_id": "uuid",
            "trace_type": "query",
            "query": "iPhone推荐",
            "stages": {
                "bm25": {"top5_ids": [...], "elapsed_ms": 12},
                "dense": {"top5_ids": [...], "elapsed_ms": 89},
                "rrf": {"fused_count": 28, "elapsed_ms": 3},
                "rerank": {"top_score": 0.94, "elapsed_ms": 210}
            }
        }