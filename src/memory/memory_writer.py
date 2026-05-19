"""记忆写入器: 会话结束后自动总结"""
# 模块 6 实现
"""
记忆写入器
==========
会话结束后，自动从对话历史中提取关键信息，写入长期记忆。

核心思路：用 LLM 读对话历史，提取结构化信息 + 语义偏好，
然后分别写入 SQLite 和 Milvus。

面试话术:
    "会话结束后触发 MemoryWriter，用 LLM 从对话中
     抽取用户偏好（肤质、预算、风格描述），
     结构化字段入 SQLite，语义描述 embedding 后入 Milvus，
     实现跨会话个性化。"
"""

import json
from loguru import logger
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory


# 提取用户偏好的 Prompt 模板
EXTRACT_PROMPT = """
你是一个用户画像提取助手。
请从下面的对话中提取用户的偏好信息，以 JSON 格式返回。

对话内容：
{conversation}

请提取以下信息（没有提到的字段填 null）：
{{
    "structured": {{
        "skin_type": "肤质（油皮/干皮/混合肌/敏感肌/油痘肌）",
        "budget": "预算金额（数字）",
        "dislike": "不喜欢的成分或特性",
        "category_interest": "感兴趣的品类"
    }},
    "semantic_preferences": [
        "用自然语言描述的偏好1",
        "用自然语言描述的偏好2"
    ]
}}

只返回 JSON，不要其他内容。
"""


class MemoryWriter:
    """
    记忆写入器。

    用法:
        writer = MemoryWriter(stm=short_term, ltm=long_term)

        # 会话结束后调用
        writer.extract_and_save(
            session_id="session_abc",
            user_id="user_123",
            llm_client=client,   # OpenAI 兼容客户端
            model="qwen-plus",
        )
    """

    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory):
        self.stm = stm
        self.ltm = ltm

    def extract_and_save(
        self,
        session_id: str,
        user_id: str,
        llm_client=None,
        model: str = "qwen-plus",
    ) -> dict:
        """
        从对话历史提取偏好，写入长期记忆。

        如果没有 llm_client，用规则提取（降级方案）。
        """
        conversation = self.stm.get_summary(session_id)
        if not conversation:
            logger.info(f"[MemoryWriter] session {session_id} 无对话历史，跳过")
            return {}

        if llm_client:
            extracted = self._extract_with_llm(conversation, llm_client, model)
        else:
            extracted = self._extract_with_rules(conversation)

        # 写入长期记忆
        structured = extracted.get("structured", {})
        # 过滤掉 null 值
        structured = {k: v for k, v in structured.items() if v is not None}
        if structured:
            self.ltm.update_profile(user_id, structured)

        for pref in extracted.get("semantic_preferences", []):
            if pref:
                self.ltm.add_semantic_preference(user_id, pref)

        logger.info(
            f"[MemoryWriter] 记忆写入完成: {user_id} | "
            f"结构化字段: {len(structured)} | "
            f"语义偏好: {len(extracted.get('semantic_preferences', []))}"
        )
        return extracted

    def _extract_with_llm(self, conversation: str, llm_client, model: str) -> dict:
        """用 LLM 提取偏好（高精度）。"""
        prompt = EXTRACT_PROMPT.format(conversation=conversation)
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            # 去掉可能的 markdown 代码块
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM 提取失败，降级到规则提取: {e}")
            return self._extract_with_rules(conversation)

    def _extract_with_rules(self, conversation: str) -> dict:
        """规则提取偏好（降级方案，不需要 LLM）。"""
        structured = {}
        semantic = []

        # 简单关键词规则
        skin_keywords = {
            "油痘肌": "油痘肌", "油皮": "油皮", "干皮": "干皮",
            "混合肌": "混合肌", "敏感肌": "敏感肌",
        }
        for kw, val in skin_keywords.items():
            if kw in conversation:
                structured["skin_type"] = val
                break

        # 预算提取（简单正则）
        import re
        budget_match = re.search(r'预算[^\d]*(\d+)', conversation)
        if budget_match:
            structured["budget"] = int(budget_match.group(1))

        # 语义偏好（按句子拆分，找含偏好词的句子）
        pref_keywords = ["喜欢", "不喜欢", "偏好", "希望", "要求", "讨厌"]
        for line in conversation.split("\n"):
            if any(kw in line for kw in pref_keywords):
                semantic.append(line.strip())

        return {"structured": structured, "semantic_preferences": semantic[:3]}