#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/6/22 14:04
@updated: 2026/6/22 14:04
@version: 1.0
@description: 
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态接口：图片 → Qwen-VL 识别 → HybridRetriever 检索
"""

import base64
import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger
from openai import OpenAI

router = APIRouter(prefix="/multimodal", tags=["multimodal"])

# DashScope Qwen-VL 客户端
_vl_client = None

def get_vl_client() -> OpenAI:
    global _vl_client
    if _vl_client is None:
        _vl_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _vl_client


def describe_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """调用 Qwen-VL 识别图片，返回文字描述。"""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    client = get_vl_client()
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": (
                            "请识别这张图片里的产品或物品，用简洁的中文描述："
                            "1. 这是什么产品/物品（品牌、型号、类别）"
                            "2. 主要特征"
                            "输出格式：直接描述，不超过50字，适合用来搜索小红书笔记。"
                        ),
                    },
                ],
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


@router.post("/search")
async def multimodal_search(
    file: UploadFile = File(...),
):
    """
    上传图片 → Qwen-VL 识别内容 → 检索相关小红书笔记
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    # 读取图片
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB 限制
        raise HTTPException(status_code=400, detail="图片不能超过 10MB")

    logger.info(f"多模态搜索: {file.filename} ({file.content_type}, {len(image_bytes)//1024}KB)")

    # Qwen-VL 识别
    try:
        description = describe_image(image_bytes, file.content_type)
        logger.info(f"图片识别结果: {description}")
    except Exception as e:
        logger.error(f"Qwen-VL 识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片识别失败: {str(e)}")

    # 用识别结果检索笔记（从 main.py 拿 retriever）
    from src.api.main import retriever
    try:
        results = retriever.search(query=description, top_k=6, use_reranker=False)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        results = []

    return {
        "image_description": description,
        "query_used": description,
        "results": results[:6],
        "total": len(results),
    }