#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/5/16 17:57
@updated: 2026/5/16 17:57
@version: 1.0
@description: 
"""
"""
Harness 验证 Demo
==================
用一个 EchoAgent 测试 Tracer + Budget + ToolRegistry 全流程。

不需要 API key, 纯本地 mock 验证。
"""

import json
import os
import sys

# 把项目根目录加到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness import (
    AgentHarness, ToolRegistry, ToolSpec,
    BudgetManager, Tracer, Session, BudgetState,
)


# ============================================================
# 1. 定义一个模拟工具
# ============================================================
def mock_search_notes(query: str, top_k: int = 5) -> list[dict]:
    """模拟搜索小红书笔记。"""
    return [
        {"title": f"【测评】{query}真实体验 #{i+1}", "score": round(0.9 - i * 0.05, 2)}
        for i in range(top_k)
    ]


# ============================================================
# 2. 定义一个最简单的 Agent
# ============================================================
class EchoAgent(AgentHarness):
    """
    测试用 Agent: 不调真实 LLM, 只验证 Harness 组件协作。
    """

    def run(self, user_query: str, session: Session, budget_state: BudgetState) -> dict:
        # Step 1: 模拟 query 解析
        step = self.tracer.start_step(session, agent_name=self.agent_name, action="parse_query")
        step.input_data = {"raw_query": user_query}
        parsed = {"category": "数码3C", "keywords": user_query}
        step.output_data = parsed
        self.tracer.end_step(step)

        # Step 2: 调用工具
        tool_result = self.registry.execute("search_notes", {"query": user_query, "top_k": 3})

        step2 = self.tracer.start_step(session, agent_name=self.agent_name, action="tool:search_notes")
        step2.input_data = {"query": user_query, "top_k": 3}
        step2.output_data = tool_result
        self.tracer.end_step(step2)

        # Step 3: 模拟 token 消耗
        self.budget_manager.consume(budget_state, prompt_tokens=200, completion_tokens=100)

        return {
            "agent": self.agent_name,
            "parsed_query": parsed,
            "search_results": tool_result,
            "budget_summary": self.budget_manager.get_summary(budget_state),
        }


# ============================================================
# 3. 运行
# ============================================================
def main():
    print("=" * 60)
    print("  Agent Harness 验证 Demo")
    print("=" * 60)

    # 初始化共享组件
    tracer = Tracer(log_dir="logs")
    budget_mgr = BudgetManager(max_tokens=10_000, max_llm_calls=5)
    registry = ToolRegistry()

    # 注册工具
    registry.register(ToolSpec(
        name="search_notes",
        description="在小红书笔记库中搜索相关笔记",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 5},
            },
            "required": ["query"],
        },
        func=mock_search_notes,
    ))

    # 创建 Agent
    agent = EchoAgent(
        agent_name="EchoAgent",
        description="测试用 Agent",
        model="qwen-plus",
        tracer=tracer,
        budget_manager=budget_mgr,
        registry=registry,
    )

    # 执行
    session = tracer.start_session(user_query="iPhone 16 Pro 值得买吗")
    budget_state = budget_mgr.create_state()

    result = agent.run("iPhone 16 Pro 值得买吗", session, budget_state)
    log_file = tracer.end_session(session, final_output=result)

    # 输出
    print("\n" + "-" * 60)
    print("【Agent 输出】")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n【Session 摘要】")
    print(json.dumps(tracer.get_session_summary(session), ensure_ascii=False, indent=2))

    print(f"\n【Trace 日志已保存】{log_file}")

    print("\n【已注册工具】")
    print(registry.list_tools())

    print("\n【OpenAI Function Calling 格式】")
    print(json.dumps(registry.to_openai_tools(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("  ✅ Harness 验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()