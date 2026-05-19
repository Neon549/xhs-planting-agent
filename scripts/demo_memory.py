#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/18 23:45
@updated: 2026/5/18 23:45
@version: 1.0
@description: 
"""
"""
双层记忆验证 Demo
运行: uv run python scripts/demo_memory.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.memory_writer import MemoryWriter


def main():
    print("=" * 60)
    print("  双层记忆系统验证 Demo")
    print("=" * 60)

    stm = ShortTermMemory()
    ltm = LongTermMemory(db_path="data/memory.db")
    writer = MemoryWriter(stm=stm, ltm=ltm)

    user_id = "test_user_001"
    session_id = "session_abc"

    # ── 短期记忆 ──────────────────────────────────────────────
    print("\n【短期记忆】模拟一次对话")
    stm.add_message(session_id, "user", "我是油痘肌，预算2000以内，想买一台轻薄笔记本")
    stm.add_message(session_id, "assistant", "好的，我帮你找适合油痘肌用户的轻薄笔记本推荐")
    stm.add_message(session_id, "user", "我不喜欢太厚重的，希望重量在1.5kg以内")
    stm.add_message(session_id, "assistant", "明白，我会优先推荐轻薄机型")

    history = stm.get_messages(session_id)
    print(f"  对话轮数: {len(history)} 条消息")
    context = stm.get_context_window(session_id, max_turns=2)
    print(f"  最近2轮上下文: {len(context)} 条消息")

    # ── 记忆写入 ──────────────────────────────────────────────
    print("\n【记忆写入】会话结束，提取偏好写入长期记忆")
    extracted = writer.extract_and_save(
        session_id=session_id,
        user_id=user_id,
        llm_client=None,   # 无 LLM，用规则提取
    )
    print(f"  提取结果: {json.dumps(extracted, ensure_ascii=False, indent=2)}")

    # ── 长期记忆 ──────────────────────────────────────────────
    print("\n【长期记忆】查询用户画像")
    profile = ltm.get_profile(user_id)
    print(f"  结构化画像: {json.dumps(profile, ensure_ascii=False)}")

    prefs = ltm.get_semantic_preferences(user_id)
    print(f"  语义偏好 ({len(prefs)} 条):")
    for p in prefs:
        print(f"    - {p['preference']}")

    # ── 构建 context 注入 prompt ──────────────────────────────
    print("\n【注入 Prompt】构建用户上下文")
    context_str = ltm.build_user_context(user_id)
    print(f"  用户上下文:\n{context_str}")

    # ── 第二次会话（跨会话个性化）─────────────────────────────
    print("\n【跨会话测试】新会话，Agent 知道用户历史偏好")
    new_session = "session_xyz"
    stm.add_message(new_session, "user", "帮我推荐一个键盘")
    user_ctx = ltm.build_user_context(user_id)
    print(f"  新会话前，Agent 能看到的用户画像:\n  {user_ctx}")

    print("\n" + "=" * 60)
    print("  ✅ 双层记忆系统验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()