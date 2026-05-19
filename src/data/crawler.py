"""MediaCrawler 封装"""
# 模块 3 实现
"""
MediaCrawler 封装
================
把 MediaCrawler 爬取的原始数据读取到项目中。

MediaCrawler 是独立项目，不集成到本项目里，
我们只负责读取它输出的 JSON/JSONL 文件。

使用流程:
    1. 在 MediaCrawler 目录下跑爬虫，数据落地到它自己的 data/ 目录
    2. 把爬取的文件复制到本项目的 data/raw/ 目录
    3. 调用 load_raw_data() 读取，传给 cleaner.py 处理
"""

import json
from pathlib import Path
from loguru import logger


KEYWORDS_MAP = {
    # 数码 3C
    "iPhone": "数码3C",
    "手机": "数码3C",
    "安卓": "数码3C",
    "小米": "数码3C",
    "华为": "数码3C",
    "三星": "数码3C",
    "笔记本": "数码3C",
    "电脑": "数码3C",
    "耳机": "数码3C",
    "平板": "数码3C",
    # 健身器材
    "哑铃": "健身器材",
    "瑜伽": "健身器材",
    "健身": "健身器材",
    "跑步机": "健身器材",
    "弹力带": "健身器材",
    "器械": "健身器材",
}


def load_raw_data(raw_dir: str = "data/raw") -> list[dict]:
    """
    读取 data/raw/ 下所有 JSON/JSONL 文件。

    返回原始笔记列表（未清洗），传给 cleaner.py 处理。
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"raw 目录不存在: {raw_dir}")
        return []

    all_notes = []

    # 支持 .json 和 .jsonl 两种格式（MediaCrawler 可以输出两种）
    for file in raw_path.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_notes.extend(data)
            elif isinstance(data, dict):
                notes = data.get("data", data.get("notes", [data]))
                all_notes.extend(notes)
            logger.info(f"读取 {file.name}: {len(all_notes)} 条")
        except Exception as e:
            logger.error(f"读取失败 {file.name}: {e}")

    for file in raw_path.glob("*.jsonl"):
        try:
            lines = file.read_text(encoding="utf-8").strip().split("\n")
            notes = [json.loads(line) for line in lines if line.strip()]
            all_notes.extend(notes)
            logger.info(f"读取 {file.name}: {len(notes)} 条")
        except Exception as e:
            logger.error(f"读取失败 {file.name}: {e}")

    logger.info(f"共读取原始笔记: {len(all_notes)} 条")
    return all_notes


def get_keywords_map() -> dict[str, str]:
    """返回关键词→品类映射表。"""
    return KEYWORDS_MAP