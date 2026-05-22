"""Token 预算控制"""
# 模块 2 实现
"""
Token 预算控制模块
=================
防止一次用户请求消耗过多 Token。

三级策略:
    1. 用量 < 80%  → PROCEED (正常执行)
    2. 80%-100%    → DOWNGRADE (换小模型)
    3. >= 100%     → STOP (停止调用)

对标: smolagents 的 max_steps 机制（限制步数间接控制 token）
我们更细: 直接按 token 数 + 调用次数双重控制。

"""

from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class BudgetAction(Enum):
    PROCEED = "proceed"
    DOWNGRADE = "downgrade"
    STOP = "stop"


DEFAULT_DOWNGRADE_MAP = {
    "qwen-plus": "qwen-turbo",
    "qwen-max": "qwen-plus",
    "gpt-4o": "gpt-4o-mini",
}


@dataclass
class BudgetState:
    """单个 Session 的预算状态。"""

    max_tokens: int = 50_000
    used_tokens: int = 0
    warning_threshold: float = 0.8
    num_llm_calls: int = 0
    max_llm_calls: int = 20

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def usage_ratio(self) -> float:
        if self.max_tokens == 0:
            return 1.0
        return self.used_tokens / self.max_tokens


class BudgetManager:
    """
    Token 预算管理器。

    用法:
        budget = BudgetManager(max_tokens=50000)
        state = budget.create_state()

        action = budget.check(state)
        if action == BudgetAction.STOP:
            return "预算耗尽"

        model = budget.get_model(state, preferred="qwen-plus")
        # 可能返回 "qwen-turbo"（降级）

        budget.consume(state, prompt_tokens=200, completion_tokens=100)
    """

    def __init__(
        self,
        max_tokens: int = 50_000,
        max_llm_calls: int = 20,
        warning_threshold: float = 0.8,
        downgrade_map: dict[str, str] | None = None,
    ):
        self.max_tokens = max_tokens
        self.max_llm_calls = max_llm_calls
        self.warning_threshold = warning_threshold
        self.downgrade_map = downgrade_map or DEFAULT_DOWNGRADE_MAP

    def create_state(self) -> BudgetState:
        return BudgetState(
            max_tokens=self.max_tokens,
            max_llm_calls=self.max_llm_calls,
            warning_threshold=self.warning_threshold,
        )

    def check(self, state: BudgetState) -> BudgetAction:
        if state.num_llm_calls >= state.max_llm_calls:
            logger.warning(f"预算: 调用次数 {state.num_llm_calls} 达上限, STOP")
            return BudgetAction.STOP

        if state.remaining_tokens <= 0:
            logger.warning(f"预算: Token 耗尽 ({state.used_tokens}/{state.max_tokens}), STOP")
            return BudgetAction.STOP

        if state.usage_ratio >= state.warning_threshold:
            logger.info(f"预算: 使用率 {state.usage_ratio:.1%}, DOWNGRADE")
            return BudgetAction.DOWNGRADE

        return BudgetAction.PROCEED

    def get_model(self, state: BudgetState, preferred: str) -> str:
        action = self.check(state)
        if action == BudgetAction.DOWNGRADE:
            downgraded = self.downgrade_map.get(preferred, preferred)
            if downgraded != preferred:
                logger.info(f"模型降级: {preferred} → {downgraded}")
            return downgraded
        return preferred

    def consume(self, state: BudgetState, prompt_tokens: int, completion_tokens: int) -> None:
        total = prompt_tokens + completion_tokens
        state.used_tokens += total
        state.num_llm_calls += 1
        logger.debug(
            f"预算消耗: +{total} tokens | "
            f"累计: {state.used_tokens}/{state.max_tokens} ({state.usage_ratio:.1%}) | "
            f"调用: {state.num_llm_calls}/{state.max_llm_calls}"
        )

    def get_summary(self, state: BudgetState) -> dict:
        return {
            "max_tokens": state.max_tokens,
            "used_tokens": state.used_tokens,
            "remaining_tokens": state.remaining_tokens,
            "usage_ratio": round(state.usage_ratio, 4),
            "num_llm_calls": state.num_llm_calls,
            "max_llm_calls": state.max_llm_calls,
            "status": self.check(state).value,
        }