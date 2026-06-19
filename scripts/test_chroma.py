#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/6/19 16:17
@updated: 2026/6/19 16:17
@version: 1.0
@description: 
"""
import sys
sys.path.insert(0, '.')
from src.retrieval.chroma_store import ChromaStore
from src.retrieval.dense_retriever import DenseRetriever
from config.settings import VECTOR_STORE
from src.retrieval.trace import TraceContext
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
import pickle

store = ChromaStore(
    persist_dir=VECTOR_STORE['persist_dir'],
    collection_name=VECTOR_STORE['collection_name'],
)
print(f'ChromaDB 已有 {store.count()} 条向量')

retriever = DenseRetriever(milvus_store=store)
retriever.load_model()
results = retriever.search('iPhone推荐', top_k=3)
for r in results:
    print(f"score={r['score']} | {r['text'][:50]}")

bm25 = BM25Retriever()
bm25.load_index('data/bm25_index.pkl')

hybrid = HybridRetriever(bm25=bm25, dense=retriever)

trace = TraceContext(query='iPhone推荐')
results = hybrid.search('iPhone推荐', top_k=5, use_reranker=False, trace=trace)
print(f'\nHybrid结果: {len(results)} 条')
print('Trace已保存到 logs/traces.jsonl')