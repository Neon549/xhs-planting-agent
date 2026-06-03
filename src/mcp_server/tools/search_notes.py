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
检索工具：search_xhs_notes + get_note_categories
"""
import json
from pathlib import Path
from mcp import types
from loguru import logger

# ── 工具定义（LLM 靠这个决定要不要调用）────────────────────────
SEARCH_TOOL = types.Tool(
    name="search_xhs_notes",
    description=(
        "在小红书笔记知识库中搜索相关笔记。"
        "支持数码3C（手机、电脑、耳机等）和健身器材（哑铃、瑜伽垫等）品类。"
        "使用混合检索（关键词+语义），按相关性排序。"
        "适合：产品推荐、选购建议、品牌对比等问题。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索问题，例如：'iPhone 16 值得买吗'",
            },
            "top_k": {
                "type": "integer",
                "description": "返回数量，默认5，最多10",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    },
)

CATEGORIES_TOOL = types.Tool(
    name="get_note_categories",
    description="查询知识库支持的品类和笔记数量统计。",
    inputSchema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)


# ── 工具执行 ────────────────────────────────────────────────────
def execute_search(arguments: dict, retriever) -> list[types.TextContent]:
    """执行混合检索，格式化结果。"""
    query = arguments.get("query", "").strip()
    top_k = min(arguments.get("top_k", 5), 10)

    if not query:
        return [types.TextContent(type="text", text="请提供搜索关键词")]

    try:
        results = retriever.search(query=query, top_k=top_k)

        if not results:
            return [
                types.TextContent(
                    type="text", text=f"没有找到关于「{query}」的相关笔记。"
                )
            ]

        lines = [f"## 关于「{query}」的检索结果\n"]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            text = r.get("text", "")[:150]
            liked = meta.get("liked_count", 0)
            category = meta.get("category", "未知")
            lines.append(f"### {i}. {category} | 点赞 {liked}")
            lines.append(f"{text}...\n")

        lines.append("*数据来源：小红书真实笔记*")
        return [types.TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        logger.error(f"检索失败: {e}")
        return [types.TextContent(type="text", text=f"检索出错：{e}")]


def execute_categories() -> list[types.TextContent]:
    """返回知识库品类统计。"""
    try:
        notes = json.loads(
            Path("data/processed/notes.json").read_text(encoding="utf-8")
        )
        category_count: dict[str, int] = {}
        for note in notes:
            cat = note.get("category", "未知")
            category_count[cat] = category_count.get(cat, 0) + 1

        lines = ["## 知识库品类统计\n"]
        for cat, count in sorted(category_count.items()):
            lines.append(f"- **{cat}**：{count} 篇笔记")
        lines.append(f"\n总计：{len(notes)} 篇笔记")

        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"获取品类失败：{e}")]
