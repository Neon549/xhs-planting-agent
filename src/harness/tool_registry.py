"""工具注册中心"""
# 模块 2 实现
"""
工具注册中心
===========
统一管理所有 Agent 可调用的工具。

每个工具有: name, description, parameters(JSON Schema), func
注册后可以:
    - 按名字调用
    - 自动转换为 OpenAI function calling 格式
    - 错误包装（Agent 只需检查 success 字段）

对标: smolagents/tools.py 的 Tool 基类
区别: 我们用更轻量的「函数 + Schema」方式, 不需要类继承。

"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class ToolSpec:
    """工具规格, 直接对应 OpenAI function calling 格式。"""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    func: Callable | None = None

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    工具注册中心。

    用法:
        registry = ToolRegistry()

        # 手动注册
        registry.register(ToolSpec(name="search", description="搜索笔记", func=my_func))

        # 装饰器注册
        @registry.tool(name="detect_ad", description="检测软广")
        def detect_ad(note_text: str) -> dict: ...

        # 执行
        result = registry.execute("search", {"query": "iPhone"})

        # 转 OpenAI 格式
        tools = registry.to_openai_tools()
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            logger.warning(f"工具 '{spec.name}' 已存在, 将被覆盖")
        self._tools[spec.name] = spec
        logger.info(f"工具注册: {spec.name}")

    def tool(self, name: str, description: str, parameters: dict | None = None):
        """装饰器方式注册工具。"""
        def decorator(func: Callable) -> Callable:
            spec = ToolSpec(name=name, description=description, parameters=parameters or {}, func=func)
            self.register(spec)
            return func
        return decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self, tool_names: list[str] | None = None) -> list[dict]:
        if tool_names is None:
            tool_names = self.list_tools()
        tools = []
        for name in tool_names:
            spec = self._tools.get(name)
            if spec:
                tools.append(spec.to_openai_tool())
        return tools

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        """
        执行工具, 返回统一格式:
            成功: {"success": True, "result": <返回值>}
            失败: {"success": False, "error": <错误信息>}
        """
        spec = self._tools.get(name)
        if not spec:
            return {"success": False, "error": f"工具 '{name}' 未注册。可用: {self.list_tools()}"}
        if not spec.func:
            return {"success": False, "error": f"工具 '{name}' 没有绑定函数"}

        arguments = arguments or {}
        try:
            result = spec.func(**arguments)
            return {"success": True, "result": result}
        except TypeError as e:
            return {"success": False, "error": f"参数错误: {e}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}