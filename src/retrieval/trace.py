#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/6/19 16:18
@updated: 2026/6/19 16:18
@version: 1.0
@description: 
"""
import uuid
import time
import json
from pathlib import Path
from loguru import logger


class TraceContext:
    """
    轻量级查询追踪，记录每次检索各阶段耗时和中间结果。
    输出到 logs/traces.jsonl，每行一条完整记录。
    """

    def __init__(self, query: str, trace_type: str = "query"):
        self.trace_id = str(uuid.uuid4())[:8]
        self.query = query
        self.trace_type = trace_type
        self.stages = {}
        self._start = time.time()
        self._stage_start = time.time()

    def record(self, stage: str, data: dict) -> None:
        elapsed = round((time.time() - self._stage_start) * 1000, 1)
        self.stages[stage] = {**data, "elapsed_ms": elapsed}
        self._stage_start = time.time()
        logger.debug(f"[Trace {self.trace_id}] {stage}: {elapsed}ms")

    def save(self, path: str = "logs/traces.jsonl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        total_ms = round((time.time() - self._start) * 1000, 1)
        record = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "query": self.query,
            "total_ms": total_ms,
            "stages": self.stages,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[Trace {self.trace_id}] 查询完成，总耗时 {total_ms}ms")