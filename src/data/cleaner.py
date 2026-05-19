"""数据清洗"""
# 模块 3 实现
"""
数据清洗模块
===========
把 MediaCrawler 爬取的原始 JSON 清洗成项目统一格式。
"""

import json
import re
from pathlib import Path
from datetime import datetime
from loguru import logger


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、零宽字符。"""
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()

def parse_count(value) -> int:
    """解析数字，支持 '2.5万'、'1.2w'、'3k' 等格式。"""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return 0
    value = value.strip()
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return int(float(value))
    except ValueError:
        pass
    # 处理中文单位
    if '万' in value or 'w' in value.lower():
        num_str = value.replace('万', '').replace('w', '').replace('W', '').strip()
        try:
            return int(float(num_str) * 10000)
        except ValueError:
            return 0
    if 'k' in value.lower():
        num_str = value.replace('k', '').replace('K', '').strip()
        try:
            return int(float(num_str) * 1000)
        except ValueError:
            return 0
    return 0

def extract_tags(raw_note: dict) -> list[str]:
    """从原始笔记中提取话题标签。"""
    tags = []
    for tag in raw_note.get("tag_list", []):
        if isinstance(tag, dict):
            tags.append(tag.get("name", ""))
        elif isinstance(tag, str):
            tags.append(tag)
    content = raw_note.get("desc", "") or raw_note.get("content", "")
    hashtags = re.findall(r'#(\S+?)(?:\s|$|#)', content)
    tags.extend(hashtags)
    return list(set(filter(None, tags)))


def infer_category(raw_note: dict, keywords_map: dict[str, str]) -> str:
    """根据关键词推断笔记品类。"""
    title = raw_note.get("title", "") or raw_note.get("display_title", "")
    content = raw_note.get("desc", "") or raw_note.get("content", "")
    text = (title + " " + content).lower()
    for keyword, category in keywords_map.items():
        if keyword.lower() in text:
            return category
    return "其他"


def clean_note(raw_note: dict, category: str = "未知") -> dict | None:
    """
    清洗单条笔记。
    返回 None 表示不合格（内容太短、无标题等）。
    """
    note_id = raw_note.get("note_id") or raw_note.get("id", "")
    title = raw_note.get("title") or raw_note.get("display_title", "")
    content = raw_note.get("desc") or raw_note.get("content", "")

    if not note_id:
        return None
    if not title and not content:
        return None
    if len((title + content)) < 20:
        return None

    title = clean_text(title)
    content = clean_text(content)
    tags = extract_tags(raw_note)
    tags_str = " ".join(f"#{t}" for t in tags)

    full_text = f"标题: {title}\n正文: {content}"
    if tags_str:
        full_text += f"\n标签: {tags_str}"

    interact_info = raw_note.get("interact_info", {}) or {}
    liked_count = parse_count(interact_info.get("liked_count", 0) or raw_note.get("liked_count", 0) or 0)
    comment_count = parse_count(interact_info.get("comment_count", 0) or raw_note.get("comment_count", 0) or 0)

    user = raw_note.get("user", {}) or {}
    author = user.get("nickname", "") or raw_note.get("author", "未知")

    comments = []
    for c in raw_note.get("comments", [])[:20]:
        if isinstance(c, dict):
            comment_text = c.get("content", "") or c.get("text", "")
            if comment_text:
                comments.append(clean_text(comment_text))
        elif isinstance(c, str):
            comments.append(clean_text(c))

    return {
        "note_id": note_id,
        "title": title,
        "content": content,
        "full_text": full_text,
        "tags": tags,
        "liked_count": liked_count,
        "comment_count": comment_count,
        "comments": comments,
        "author": author,
        "category": category,
        "crawled_at": datetime.now().isoformat(),
        "is_ad": None,
    }


def clean_raw_file(
    raw_file: Path,
    output_dir: Path,
    category: str,
    keywords_map: dict[str, str] | None = None,
) -> list[dict]:
    """清洗一个 MediaCrawler 输出的 JSON 文件。"""
    logger.info(f"开始清洗: {raw_file.name} | 品类: {category}")

    raw_data = json.loads(raw_file.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict):
        raw_notes = raw_data.get("data", raw_data.get("notes", [raw_data]))
    else:
        raw_notes = raw_data

    cleaned = []
    skipped = 0

    for raw_note in raw_notes:
        if keywords_map:
            category = infer_category(raw_note, keywords_map)
        note = clean_note(raw_note, category=category)
        if note is None:
            skipped += 1
            continue
        cleaned.append(note)

    # 去重
    seen_ids = set()
    deduped = []
    for note in cleaned:
        if note["note_id"] not in seen_ids:
            seen_ids.add(note["note_id"])
            deduped.append(note)

    logger.info(
        f"清洗完成: {raw_file.name} | "
        f"原始: {len(raw_notes)} | 有效: {len(deduped)} | "
        f"过滤: {skipped + (len(cleaned) - len(deduped))}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"cleaned_{raw_file.stem}.json"
    output_file.write_text(
        json.dumps(deduped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"已保存: {output_file}")
    return deduped