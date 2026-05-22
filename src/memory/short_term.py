"""短期记忆: LangGraph Checkpointer"""
# 模块 6 实现
"""
短期记忆
========
管理单次对话内的上下文，基于 LangGraph Checkpointer。

本质：把对话历史（messages）存在内存里，
每轮对话都能看到之前说过的内容。

"""

from langgraph.checkpoint.memory import MemorySaver
from loguru import logger


class ShortTermMemory:
    """
    短期记忆管理器。

    用法:
        stm = ShortTermMemory()

        # 添加一轮对话
        stm.add_message(session_id="user_123", role="user", content="我是油痘肌")
        stm.add_message(session_id="user_123", role="assistant", content="好的，我记住了")

        # 获取对话历史（传给 LLM 作为上下文）
        history = stm.get_messages(session_id="user_123")

        # 获取摘要（用于注入长期记忆）
        summary = stm.get_summary(session_id="user_123")
    """

    def __init__(self):
        # MemorySaver 是 LangGraph 内置的内存 checkpointer
        # 生产环境可换成 SqliteSaver 持久化
        self.checkpointer = MemorySaver()
        # 用字典存各 session 的消息（简化版，够用）
        self._sessions: dict[str, list[dict]] = {}
        logger.info("ShortTermMemory 初始化完成")

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条消息到对话历史。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        logger.debug(f"[STM] {session_id} | {role}: {content[:50]}")

    def get_messages(self, session_id: str) -> list[dict]:
        """获取某个 session 的完整对话历史。"""
        return self._sessions.get(session_id, [])

    def get_context_window(self, session_id: str, max_turns: int = 5) -> list[dict]:
        """
        获取最近 N 轮对话（避免上下文过长）。

        max_turns=5 表示最近 5 轮（10 条消息：5 user + 5 assistant）
        """
        messages = self._sessions.get(session_id, [])
        return messages[-(max_turns * 2):]

    def get_summary(self, session_id: str) -> str:
        """
        把对话历史拼成文本摘要（用于写入长期记忆）。
        """
        messages = self._sessions.get(session_id, [])
        if not messages:
            return ""
        lines = []
        for m in messages:
            prefix = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{prefix}: {m['content']}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        """清除某个 session 的记忆（对话结束时调用）。"""
        self._sessions.pop(session_id, None)
        logger.info(f"[STM] 已清除 session: {session_id}")

    def get_all_sessions(self) -> list[str]:
        return list(self._sessions.keys())