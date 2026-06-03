#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/6/3 15:12
@updated: 2026/6/3 15:12
@version: 1.0
@description:
"""

from __future__ import annotations

"""
xhs 种草决策 Agent - MCP Server
================================
把检索能力通过 MCP 协议暴露给 Claude Desktop。

启动方式:
    uv run python src/mcp_server/server.py

Claude Desktop 配置:
    {
      "mcpServers": {
        "xhs-agent": {
          "command": "uv",
          "args": ["run", "python", "src/mcp_server/server.py"],
          "cwd": "D:\\code\\ProjectExample\\xhs-planting-agent"
        }
      }
    }
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from loguru import logger

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.reranker import create_reranker
from src.mcp_server.tools.search_notes import (
    SEARCH_TOOL,
    CATEGORIES_TOOL,
    execute_search,
    execute_categories,
)
from config.settings import RERANKER


# ── 初始化检索器 ────────────────────────────────────────────────
def _init_retriever() -> HybridRetriever:
    """启动时初始化，之后每次请求复用。"""
    logger.info("初始化检索组件...")

    chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))
    logger.info(f"加载 {len(chunks)} 个 chunks")

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    class RealDenseRetriever:
        def __init__(self):
            self._model = None
            self._store = MilvusStore(db_path="data/milvus_lite.db")

        def _load_model(self):
            if self._model is None:
                from FlagEmbedding import BGEM3FlagModel

                logger.info("加载 bge-m3...")
                self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        def search(self, query: str, top_k: int = 20) -> list[dict]:
            self._load_model()
            output = self._model.encode(
                [query],
                batch_size=1,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            results = self._store.search(
                query_embedding=output["dense_vecs"][0].tolist(),
                top_k=top_k,
            )
            for i, r in enumerate(results):
                r["rank"] = i + 1
                r["source"] = "dense"
            return results

    retriever = HybridRetriever(
        bm25=bm25,
        dense=RealDenseRetriever(),
        reranker=create_reranker(RERANKER),
        bm25_top_k=20,
        dense_top_k=20,
    )
    logger.info("检索组件初始化完成")
    return retriever


# ── MCP Server ──────────────────────────────────────────────────
app = Server("xhs-planting-agent")
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = _init_retriever()
    return _retriever


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """声明工具列表，Claude 连接时调用。"""
    return [SEARCH_TOOL, CATEGORIES_TOOL]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """路由工具调用。"""
    logger.info(f"工具调用: {name}")

    if name == "search_xhs_notes":
        return await asyncio.to_thread(execute_search, arguments, get_retriever())
    elif name == "get_note_categories":
        return await asyncio.to_thread(execute_categories)
    else:
        return [types.TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    # 日志走 stderr，不污染 MCP 的 stdout
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("xhs MCP Server 启动中...")

    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
