"""长期记忆: SQLite (结构化) + Milvus (语义化)"""
# 模块 6 实现
"""
长期记忆
========
跨会话持久化用户偏好，SQLite 存结构化数据，Milvus 存语义偏好。

"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from loguru import logger


class LongTermMemory:
    """
    长期记忆管理器。

    用法:
        ltm = LongTermMemory(db_path="data/memory.db")

        # 更新结构化偏好
        ltm.update_profile(user_id="user_123", attributes={
            "skin_type": "油痘肌",
            "budget_3c": 2000,
            "dislike": "香精,酒精",
        })

        # 存语义偏好
        ltm.add_semantic_preference(
            user_id="user_123",
            preference="喜欢轻薄清爽的产品，不喜欢厚重油腻的质地",
        )

        # 查询
        profile = ltm.get_profile("user_123")
        prefs = ltm.get_semantic_preferences("user_123")
    """

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"LongTermMemory 初始化完成: {db_path}")

    def _init_db(self) -> None:
        """建表（如果不存在）。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id     TEXT PRIMARY KEY,
                    attributes  TEXT NOT NULL DEFAULT '{}',
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_preferences (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    preference  TEXT NOT NULL,
                    category    TEXT DEFAULT 'general',
                    created_at  TEXT NOT NULL
                )
            """)
            conn.commit()

    def update_profile(self, user_id: str, attributes: dict) -> None:
        """
        更新用户结构化画像（合并更新，不覆盖已有字段）。

        attributes 示例:
            {"skin_type": "油痘肌", "budget_3c": 2000, "dislike": "香精"}
        """
        existing = self.get_profile(user_id)
        merged = {**existing, **attributes}   # 新值覆盖旧值
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_profiles (user_id, attributes, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    attributes = excluded.attributes,
                    updated_at = excluded.updated_at
            """, (user_id, json.dumps(merged, ensure_ascii=False), datetime.now().isoformat()))
            conn.commit()
        logger.info(f"[LTM] 更新用户画像: {user_id} | {attributes}")

    def get_profile(self, user_id: str) -> dict:
        """获取用户结构化画像。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT attributes FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return {}

    def add_semantic_preference(
        self, user_id: str, preference: str, category: str = "general"
    ) -> None:
        """
        添加一条语义偏好（自然语言描述）。

        category 可以是: "style"（风格）/ "product"（产品偏好）/ "general"
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO semantic_preferences (user_id, preference, category, created_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, preference, category, datetime.now().isoformat()))
            conn.commit()
        logger.info(f"[LTM] 语义偏好: {user_id} | {preference[:50]}")

    def get_semantic_preferences(self, user_id: str, limit: int = 10) -> list[dict]:
        """获取用户最近的语义偏好列表。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT preference, category, created_at
                FROM semantic_preferences
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()
        return [{"preference": r[0], "category": r[1], "created_at": r[2]} for r in rows]

    def build_user_context(self, user_id: str) -> str:
        """
        把用户画像 + 语义偏好拼成文本，注入 Agent 的 system prompt。

        输出示例:
            用户画像: 肤质=油痘肌, 预算(数码)=2000元
            偏好: 喜欢轻薄清爽的产品; 不喜欢香精味重的
        """
        profile = self.get_profile(user_id)
        prefs = self.get_semantic_preferences(user_id, limit=5)

        lines = []
        if profile:
            profile_str = ", ".join(f"{k}={v}" for k, v in profile.items())
            lines.append(f"用户画像: {profile_str}")
        if prefs:
            pref_str = "; ".join(p["preference"] for p in prefs)
            lines.append(f"偏好: {pref_str}")

        return "\n".join(lines) if lines else ""