"""Trace 日志: 记录 Agent 每一步的输入输出"""
# 模块 2 实现
"""
Trace 日志模块
=============
记录 Agent 每一步的输入、输出、耗时、Token 消耗。

核心概念:
    - Session: 一次用户请求的完整生命周期
    - Step: Session 中的一个执行步骤（一次 LLM 调用 / 一次工具执行）

对标: smolagents/monitoring.py

面试话术:
    "Harness 层内置 Trace 模块, 每次请求自动记录全链路日志,
     包括每个 Agent 的输入输出、耗时、Token 消耗, 支持事后分析和调试。"
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class Step:
    """一个 Agent 执行步骤的记录。"""

    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_name: str = ""
    action: str = ""
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """一次用户请求的完整 Trace。"""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_query: str = ""
    steps: list[Step] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    final_output: dict = field(default_factory=dict)


class Tracer:
    """
    Trace 管理器。

    用法:
        tracer = Tracer(log_dir="logs")
        session = tracer.start_session(user_query="推荐一款粉底液")

        step = tracer.start_step(session, agent_name="QueryPlanner", action="parse")
        step.input_data = {"raw_query": "..."}
        step.output_data = {"parsed": {...}}
        step.token_usage = {"prompt": 100, "completion": 50, "total": 150}
        tracer.end_step(step)

        tracer.end_session(session, final_output={"recommendation": "..."})
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Tracer 初始化, 日志目录: {self.log_dir}")

    def start_session(self, user_query: str) -> Session:
        session = Session(user_query=user_query)
        logger.info(f"[Session {session.session_id}] 开始 | query: {user_query}")
        return session

    def start_step(self, session: Session, agent_name: str, action: str) -> Step:
        step = Step(agent_name=agent_name, action=action, start_time=time.time())
        session.steps.append(step)
        logger.debug(f"[Session {session.session_id}] Step {step.step_id} | {agent_name}.{action} 开始")
        return step

    def end_step(self, step: Step) -> None:
        step.end_time = time.time()
        step.duration_ms = round((step.end_time - step.start_time) * 1000, 2)
        total = step.token_usage.get("total", 0)
        logger.debug(
            f"[Step {step.step_id}] {step.agent_name}.{step.action} 完成 | "
            f"耗时 {step.duration_ms}ms | tokens: {total}"
        )

    def end_session(self, session: Session, final_output: dict | None = None) -> Path:
        if final_output:
            session.final_output = final_output

        session.total_tokens = sum(s.token_usage.get("total", 0) for s in session.steps)
        session.total_duration_ms = round(sum(s.duration_ms for s in session.steps), 2)

        log_file = self.log_dir / f"trace_{session.session_id}.json"
        log_file.write_text(
            json.dumps(asdict(session), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[Session {session.session_id}] 结束 | "
            f"共 {len(session.steps)} 步 | "
            f"总 tokens: {session.total_tokens} | "
            f"总耗时: {session.total_duration_ms}ms | "
            f"日志: {log_file}"
        )
        return log_file

    def get_session_summary(self, session: Session) -> dict:
        return {
            "session_id": session.session_id,
            "user_query": session.user_query,
            "num_steps": len(session.steps),
            "total_tokens": session.total_tokens,
            "total_duration_ms": session.total_duration_ms,
            "steps_summary": [
                {
                    "agent": s.agent_name,
                    "action": s.action,
                    "duration_ms": s.duration_ms,
                    "tokens": s.token_usage.get("total", 0),
                    "error": s.error,
                }
                for s in session.steps
            ],
        }