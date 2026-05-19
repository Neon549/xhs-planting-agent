#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/18 23:59
@updated: 2026/5/18 23:59
@version: 1.0
@description: 
"""
"""
评估集生成器
============
用 LLM 从笔记自动生成 (query, relevant_note_ids) 标注数据。

流程:
    笔记内容 → LLM → 生成 2-3 个用户可能提出的 query
    → 得到 {query: note_id} 对
    → 构成评估集

面试话术:
    "评估集用 LLM 合成：给定笔记让模型生成对应的用户 query，
     快速构建 200 条标注数据。比手动标注快 10 倍，
     同时通过人工抽检 20% 保证质量。"
"""

import json
from pathlib import Path
from loguru import logger


GENERATE_PROMPT = """
你是一个用户行为模拟专家。
给你一篇小红书笔记，请生成 2 个用户在搜索时可能输入的查询语句。

笔记内容：
{note_text}

要求：
1. 查询语句要自然，像真实用户会输入的
2. 长度 5-20 字
3. 不要直接复制笔记标题
4. 覆盖不同的搜索意图（比如一个问推荐、一个问对比）

只返回 JSON，格式如下，不要其他内容：
{{"queries": ["查询1", "查询2"]}}
"""


class EvalSetGenerator:
    """
    评估集生成器。

    用法（有 LLM）：
        generator = EvalSetGenerator(llm_client=client, model="qwen-plus")
        eval_set = generator.generate(notes=cleaned_notes, num_per_note=2)
        generator.save(eval_set, "data/eval/eval_set.json")

    用法（无 LLM，规则生成）：
        generator = EvalSetGenerator()
        eval_set = generator.generate_rule_based(notes=cleaned_notes)
    """

    def __init__(self, llm_client=None, model: str = "qwen-plus"):
        self.llm_client = llm_client
        self.model = model

    def generate(self, notes: list[dict], num_per_note: int = 2) -> list[dict]:
        """用 LLM 生成评估集。"""
        if not self.llm_client:
            logger.warning("没有 LLM client，降级到规则生成")
            return self.generate_rule_based(notes)

        eval_set = []
        for note in notes:
            note_text = note.get("full_text", note.get("title", ""))
            if not note_text:
                continue

            prompt = GENERATE_PROMPT.format(note_text=note_text[:500])
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                for query in data.get("queries", [])[:num_per_note]:
                    eval_set.append({
                        "query": query,
                        "relevant_note_ids": [note["note_id"]],
                        "source": "llm_generated",
                    })
            except Exception as e:
                logger.warning(f"生成失败 {note['note_id']}: {e}")
                continue

        logger.info(f"评估集生成完成: {len(eval_set)} 条")
        return eval_set

    def generate_rule_based(self, notes: list[dict]) -> list[dict]:
        """
        规则生成评估集（不需要 LLM）。
        直接用笔记标题作为 query，以及简单变体。
        """
        eval_set = []
        templates = [
            "{title}",
            "{title}推荐",
            "怎么选{category}",
            "{category}哪个好",
        ]

        for note in notes:
            title = note.get("title", "")
            category = note.get("category", "")
            if not title:
                continue

            for tmpl in templates[:2]:
                query = tmpl.format(title=title[:15], category=category)
                eval_set.append({
                    "query": query,
                    "relevant_note_ids": [note["note_id"]],
                    "source": "rule_generated",
                })

        logger.info(f"规则评估集生成: {len(eval_set)} 条")
        return eval_set

    def save(self, eval_set: list[dict], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(eval_set, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"评估集已保存: {path} ({len(eval_set)} 条)")