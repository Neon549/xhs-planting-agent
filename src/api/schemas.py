"""Pydantic 请求/响应模型"""
# 模块 9 实现
"""API 请求/响应数据模型"""

from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    user_query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    use_reranker: bool = False


class ChunkResult(BaseModel):
    chunk_id: str
    note_id: str
    text_preview: str
    score: float
    category: str
    liked_count: int
    is_ad: Optional[bool] = None


class QueryResponse(BaseModel):
    user_query: str
    parsed_query: dict
    retrieved_count: int
    ad_count: int
    clean_count: int
    final_recommendation: str
    sources: list[dict]
    steps_taken: list[str]
    user_context: str