"""Agent Harness 基类: 工具调用、预算控制、重试、Trace"""
# 模块 2 我们来实现
"""
Agent Harness 基类
==================
整合 Tracer、BudgetManager、ToolRegistry，为每个 Agent 提供统一的执行外壳。

Agent 只关注: "用什么 prompt + 调什么工具 + 返回什么结果"
Harness 负责: "怎么调 LLM + 怎么重试 + 怎么记日志 + 怎么控预算"

对标:
    - smolagents 的 ToolCallingAgent.run() → 我们的 call_llm()
    - smolagents 的 process_tool_calls() → 我们的 handle_tool_calls()

"""

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.harness.tracer import Tracer, Session, Step
from src.harness.budget import BudgetManager, BudgetState, BudgetAction
from src.harness.tool_registry import ToolRegistry


class AgentHarness(ABC):
    """
    Agent 基类。所有业务 Agent 继承它，实现 run() 即可。

    子类示例:
        class QueryPlannerAgent(AgentHarness):
            def run(self, user_query, session, budget_state):
                messages = [{"role": "user", "content": user_query}]
                response = self.call_llm(messages, session, budget_state)
                return {"parsed_query": response["content"]}
    """

    def __init__(
        self,
        agent_name: str,
        description: str,
        model: str = "qwen-plus",
        llm_client: OpenAI | None = None,
        tracer: Tracer | None = None,
        budget_manager: BudgetManager | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.agent_name = agent_name
        self.description = description
        self.model = model
        self.llm_client = llm_client
        self.tracer = tracer or Tracer()
        self.budget_manager = budget_manager or BudgetManager()
        self.registry = registry or ToolRegistry()
        logger.info(f"Agent 初始化: {agent_name} | model: {model}")

    @abstractmethod
    def run(self, user_query: str, session: Session, budget_state: BudgetState) -> dict:
        """Agent 核心逻辑, 子类必须实现。"""
        ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _call_llm_with_retry(self, messages: list[dict], model: str, tools: list[dict] | None = None) -> Any:
        """带重试的 LLM 调用 (tenacity 指数退避: 2s → 4s → 8s)。"""
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.llm_client.chat.completions.create(**kwargs)

    def call_llm(
        self,
        messages: list[dict],
        session: Session,
        budget_state: BudgetState,
        tools: list[dict] | None = None,
        action_name: str = "llm_call",
    ) -> dict:
        """
        统一 LLM 调用入口。

        自动处理: 预算检查 → 模型选择 → 重试 → Token 记录 → Trace

        返回:
            {"content": str, "tool_calls": list|None, "model_used": str, "token_usage": dict}
        """
        # 1. 预算检查
        action = self.budget_manager.check(budget_state)
        if action == BudgetAction.STOP:
            logger.warning(f"[{self.agent_name}] 预算耗尽, 拒绝调用")
            return {"content": "[预算耗尽]", "tool_calls": None, "model_used": None, "token_usage": {}}

        # 2. 选模型（可能降级）
        actual_model = self.budget_manager.get_model(budget_state, preferred=self.model)

        # 3. Trace
        step = self.tracer.start_step(session, agent_name=self.agent_name, action=action_name)
        step.input_data = {"model": actual_model, "num_messages": len(messages), "has_tools": tools is not None}

        try:
            # 4. 调 LLM
            response = self._call_llm_with_retry(messages, model=actual_model, tools=tools)
            choice = response.choices[0]
            message = choice.message

            usage = response.usage
            token_usage = {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
                "total": usage.total_tokens if usage else 0,
            }

            # 5. 更新预算
            self.budget_manager.consume(budget_state, token_usage["prompt"], token_usage["completion"])

            # 6. 解析工具调用
            tool_calls_data = None
            if message.tool_calls:
                tool_calls_data = [
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in message.tool_calls
                ]

            result = {
                "content": message.content or "",
                "tool_calls": tool_calls_data,
                "model_used": actual_model,
                "token_usage": token_usage,
            }

            step.output_data = {"content_preview": (message.content or "")[:200], "num_tool_calls": len(tool_calls_data) if tool_calls_data else 0}
            step.token_usage = token_usage
            self.tracer.end_step(step)
            return result

        except Exception as e:
            step.error = f"{type(e).__name__}: {e}"
            self.tracer.end_step(step)
            logger.error(f"[{self.agent_name}] LLM 调用失败: {e}")
            raise

    def handle_tool_calls(self, tool_calls: list[dict], session: Session) -> list[dict]:
        """批量执行工具调用, 每个工具记录独立 Trace Step。"""
        import json as _json
        results = []
        for tc in tool_calls:
            name = tc["name"]
            try:
                arguments = _json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            except _json.JSONDecodeError:
                arguments = {}

            step = self.tracer.start_step(session, agent_name=self.agent_name, action=f"tool:{name}")
            step.input_data = {"tool": name, "arguments": arguments}

            result = self.registry.execute(name, arguments)

            step.output_data = result
            if not result["success"]:
                step.error = result.get("error", "unknown")
            self.tracer.end_step(step)

            results.append({"tool_call_id": tc.get("id", ""), "name": name, "result": result})
        return results