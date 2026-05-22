"""Chunking 策略"""
# 模块 3 实现
"""
Chunking 策略
============
把笔记切分成适合向量化的 chunk。

策略对比（面试可以讲）:
    策略 A: 整篇笔记作为一个 chunk → 短笔记用
    策略 B: 固定长度切分 → 不推荐，会截断语义
    策略 C: 语义切分（按段落）← 我们用这个
    策略 D: 父子文档 → 进阶，模块 5 后可加

"""

from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Chunk:
    """一个向量化单元。"""
    chunk_id: str
    note_id: str
    text: str
    chunk_type: str = "full"
    metadata: dict = field(default_factory=dict)


def chunk_note(note: dict, max_chars: int = 300) -> list[Chunk]:
    """对单条笔记进行 chunking，返回 Chunk 列表。"""
    note_id = note["note_id"]
    title = note.get("title", "")
    content = note.get("content", "")
    full_text = note.get("full_text", "")

    base_metadata = {
        "note_id": note_id,
        "title": title,
        "author": note.get("author", ""),
        "category": note.get("category", ""),
        "liked_count": note.get("liked_count", 0),
        "comment_count": note.get("comment_count", 0),
        "tags": note.get("tags", []),
        "is_ad": note.get("is_ad"),
    }

    chunks = []

    # 短笔记：整篇作为一个 chunk
    if len(full_text) <= max_chars:
        chunks.append(Chunk(
            chunk_id=f"{note_id}_chunk_0",
            note_id=note_id,
            text=full_text,
            chunk_type="full",
            metadata=base_metadata,
        ))
        return chunks

    # 长笔记：先加标题 chunk
    if title:
        chunks.append(Chunk(
            chunk_id=f"{note_id}_title",
            note_id=note_id,
            text=f"标题: {title}",
            chunk_type="title",
            metadata=base_metadata,
        ))

    # 按段落切分正文
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    current_chunk_text = f"标题: {title}\n" if title else ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current_chunk_text) + len(para) + 1 <= max_chars:
            current_chunk_text += para + "\n"
        else:
            if current_chunk_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{note_id}_chunk_{chunk_idx}",
                    note_id=note_id,
                    text=current_chunk_text.strip(),
                    chunk_type="paragraph",
                    metadata=base_metadata,
                ))
                chunk_idx += 1
            current_chunk_text = f"标题: {title}\n{para}\n" if title else f"{para}\n"

    if current_chunk_text.strip():
        chunks.append(Chunk(
            chunk_id=f"{note_id}_chunk_{chunk_idx}",
            note_id=note_id,
            text=current_chunk_text.strip(),
            chunk_type="paragraph",
            metadata=base_metadata,
        ))

    return chunks


def chunk_notes(notes: list[dict], max_chars: int = 300) -> list[Chunk]:
    """批量 chunking。"""
    all_chunks = []
    for note in notes:
        all_chunks.extend(chunk_note(note, max_chars=max_chars))

    logger.info(
        f"Chunking 完成: {len(notes)} 条笔记 → {len(all_chunks)} 个 chunks | "
        f"平均 {len(all_chunks)/max(len(notes),1):.1f} chunks/笔记"
    )
    return all_chunks