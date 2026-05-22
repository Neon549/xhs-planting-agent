"""Ad Detector Agent: 软广识别"""
# 模块 7 实现
"""
Ad Detector Agent
=================
识别笔记是否为软广。

软广特征（规则层）:
    - 大量感叹号、夸张词（"绝绝子"、"yyds"、"超级好用"）
    - 出现品牌+链接组合
    - 评论全是好评，没有负面声音
    - 标题含"推荐"+"品牌名"

实现: 规则打分 + LLM 判断的混合方案
    规则打分快（毫秒级），先过滤明显软广
    LLM 判断慢但准，只对规则打分在中间区域的笔记调用

"""

import re
from loguru import logger
from src.graph.state import AgentState


# 软广特征词
AD_KEYWORDS = [
    "链接在主页", "🔗", "dm我", "私信我", "点击购买",
    "限时优惠", "买一送一", "折扣码", "优惠券",
    "品牌方送我", "合作款", "恰饭",
]

EXAGGERATION_WORDS = [
    "绝绝子", "yyds", "神仙", "超级好用", "无敌", "完美",
    "必买", "闭眼入", "人手一个", "全网最",
]


def compute_ad_score(text: str) -> float:
    """
    计算软广得分（0-1，越高越可能是软广）。
    """
    score = 0.0
    text_lower = text.lower()

    # 特征 1：广告关键词（权重高）
    ad_hits = sum(1 for kw in AD_KEYWORDS if kw in text)
    score += min(ad_hits * 0.2, 0.6)

    # 特征 2：夸张词（中等权重）
    exag_hits = sum(1 for kw in EXAGGERATION_WORDS if kw in text)
    score += min(exag_hits * 0.1, 0.3)

    # 特征 3：感叹号密度（轻度特征）
    exclamation_ratio = text.count("！") / max(len(text), 1)
    if exclamation_ratio > 0.05:
        score += 0.1

    # 特征 4：emoji 密度（轻度特征）
    emoji_count = len(re.findall(r'[^\x00-\x7F]', text))
    emoji_ratio = emoji_count / max(len(text), 1)
    if emoji_ratio > 0.1:
        score += 0.1

    return min(score, 1.0)


def ad_detector_node(state: AgentState) -> dict:
    """LangGraph 节点：软广识别。"""
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "ad_detection_results": {},
            "clean_chunks": [],
            "steps_taken": ["ad_detector"],
        }

    logger.info(f"[AdDetector] 检测 {len(chunks)} 条 chunks")

    detection_results = {}
    clean_chunks = []
    ad_count = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk.get("text", "")
        score = compute_ad_score(text)
        is_ad = score >= 0.4   # 阈值 0.4

        detection_results[chunk_id] = {
            "is_ad": is_ad,
            "confidence": round(score, 3),
        }

        if not is_ad:
            clean_chunks.append(chunk)
        else:
            ad_count += 1

    logger.info(
        f"[AdDetector] 软广: {ad_count}/{len(chunks)} | "
        f"干净 chunks: {len(clean_chunks)}"
    )

    return {
        "ad_detection_results": detection_results,
        "clean_chunks": clean_chunks,
        "steps_taken": ["ad_detector"],
    }