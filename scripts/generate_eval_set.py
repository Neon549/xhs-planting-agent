#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/22 11:12
@updated: 2026/5/22 11:12
@version: 1.0
@description: 
"""

"""
用 LLM 生成高质量评估集
运行: uv run python scripts/generate_eval_set.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from evaluation.eval_set_generator import EvalSetGenerator

load_dotenv()


def main():
    print("=" * 60)
    print("  用 qwen-plus 生成真实评估集")
    print("=" * 60)

    # 读取真实笔记
    notes_file = Path("data/processed/notes.json")
    notes = json.loads(notes_file.read_text(encoding="utf-8"))
    print(f"\n读取 {len(notes)} 条笔记")

    # 初始化 DashScope 客户端（OpenAI 兼容接口）
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 生成评估集
    generator = EvalSetGenerator(llm_client=client, model="qwen-plus")
    print("\n开始生成（每条笔记生成 2 个 query，共约 116 条）...")
    eval_set = generator.generate(notes=notes, num_per_note=2)

    # 保存
    generator.save(eval_set, "data/eval/llm_eval_set.json")

    # 展示样本
    print("\n【样本展示】")
    for item in eval_set[:5]:
        print(f"  query: {item['query']}")
        print(f"  相关笔记: {item['relevant_note_ids']}")
        print()

    print(f"\n✅ 生成完成！共 {len(eval_set)} 条评估数据")
    print("下一步: uv run python scripts/run_evaluation.py")


if __name__ == "__main__":
    main()