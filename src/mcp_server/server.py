#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XHS Planting Agent - MCP Server
支持工具：
  - search_xhs_notes: 混合检索小红书笔记
  - get_note_categories: 查询知识库品类统计
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import json
import pickle
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from loguru import logger

from src.retrieval.chroma_store import ChromaStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import create_reranker
from src.retrieval.trace import TraceContext
from src.mcp_server.tools.search_notes import (
    SEARCH_TOOL,
    CATEGORIES_TOOL,
    execute_search,
    execute_categories,
)
from config.settings import VECTOR_STORE, RETRIEVAL, RERANKER


def build_retriever() -> HybridRetriever:
    """初始化混合检索器，模型常驻内存。"""
    logger.info("初始化 ChromaStore...")
    store = ChromaStore(
        persist_dir=VECTOR_STORE["persist_dir"],
        collection_name=VECTOR_STORE["collection_name"],
    )

    logger.info("加载 bge-m3 Embedding 模型...")
    dense = DenseRetriever(milvus_store=store)
    dense.load_model()

    logger.info("加载 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.load_index("data/bm25_index.pkl")

    reranker = create_reranker(RERANKER)

    retriever = HybridRetriever(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        bm25_top_k=RETRIEVAL["bm25_top_k"],
        dense_top_k=RETRIEVAL["dense_top_k"],
        rrf_k=RETRIEVAL["rrf_k"],
    )
    logger.info("混合检索器初始化完成")
    return retriever


# ── 全局 retriever（进程启动时加载一次，常驻内存）──
retriever = build_retriever()

# ── MCP Server ──────────────────────────────────────
server = Server("xhs-rag-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [SEARCH_TOOL, CATEGORIES_TOOL]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    logger.info(f"工具调用: {name} | 参数: {arguments}")

    if name == "search_xhs_notes":
        trace = TraceContext(query=arguments.get("query", ""))
        result = execute_search(arguments, retriever)
        trace.save()
        return result

    elif name == "get_note_categories":
        return execute_categories()

    else:
        return [types.TextContent(type="text", text=f"未知工具: {name}")]


async def run():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    """entry point，供 pyproject.toml scripts 调用。"""
    logger.info("XHS RAG MCP Server 启动...")
    asyncio.run(run())


if __name__ == "__main__":
    main()