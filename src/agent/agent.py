"""AgentRAG — 自驱式 RAG 框架，事件驱动的推理-行动循环。

传统 RAG 是被动的：查询 → 检索 → 生成。
AgentRAG 是主动的：Agent 自主分析问题 → 选择策略 → 执行检索 → 评估结果 → 改写重试 → 综合输出。

核心循环:
  1. 发送消息给 LLM（附带工具定义）
  2. LLM 返回 tool_calls → 执行工具 → 将结果追加到消息历史 → 回到步骤 1
  3. LLM 返回文本 → 流式输出 token → 完成

事件驱动架构: 每个 Agent 动作都通过 AgentEvent 子类向外通知，
调用方可以据此更新 UI、记录日志等。
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from .llm import LLMProvider
from .memory.conversation import ConversationBuffer
from .memory.working import WorkingMemory
from .prompts.system import SYSTEM_PROMPT_TEMPLATE
from .tools.base import ToolResult
from .tools.query_tools import create_query_tools
from .tools.reflection_tools import create_reflection_tools
from .tools.registry import ToolRegistry
from .tools.search_tools import create_search_tools

# ==================== Agent 事件类型 ====================


@dataclass
class ThinkingEvent:
    """Agent 正在思考/规划。"""

    content: str


@dataclass
class ToolCallEvent:
    """Agent 决定调用工具。"""

    tool_name: str
    arguments: dict
    call_id: str = ""


@dataclass
class ToolResultEvent:
    """工具执行完成。"""

    tool_name: str
    result: ToolResult


@dataclass
class TokenEvent:
    """LLM 输出的文本 token（流式）。"""

    token: str


@dataclass
class DoneEvent:
    """Agent 完成本轮推理。"""

    final_answer: str
    sources: list[dict] = field(default_factory=list)
    tool_calls_made: int = 0
    total_tokens: int = 0


AgentEvent = Union[ThinkingEvent, ToolCallEvent, ToolResultEvent, TokenEvent, DoneEvent]


# ==================== Research Agent ====================


class ResearchAgent:
    """AgentRAG 自驱式检索 Agent。

    与传统 RAG 的本质区别：Agent 不是"检索→生成"管道，
    而是自主决策的智能体 — 分析问题、选择策略、评估结果、自纠错。

    使用方式:
        agent = ResearchAgent(rag_system)
        agent.initialize()
        async for event in agent.chat("自注意力机制是如何工作的?"):
            match event:
                case ToolCallEvent(...):
                    print(f"Agent 调用: {event.tool_name}")
                case TokenEvent(token):
                    print(token, end="")
                case DoneEvent(answer, sources, ...):
                    print(f"\n完成 (工具调用: {event.tool_calls_made}次)")
        agent.close()
    """

    def __init__(
        self,
        rag_system,  # RAGSystem 实例
        max_iterations: int = 15,
    ):
        self.rag = rag_system
        self.max_iterations = max_iterations

        # 这些在 initialize() 中设置
        self.llm: LLMProvider | None = None
        self.tool_registry: ToolRegistry | None = None
        self._initialized = False

    # ==================== 初始化 ====================

    def initialize(self) -> None:
        """初始化 LLM Provider、工具注册表，注册所有工具。"""
        # LLM Provider
        self.llm = LLMProvider(
            api_key=self.rag.llm_client.api_key,
            base_url=str(self.rag.llm_client.base_url),
            model=self.rag.llm_model,
        )

        # 工具注册表
        self.tool_registry = ToolRegistry()

        # 注册检索工具（依赖 RAGSystem）
        self.tool_registry.register_many(create_search_tools(self.rag))

        # 注册查询变换工具（依赖 LLM）
        self.tool_registry.register_many(
            create_query_tools(self.rag.llm_client, self.rag.llm_model)
        )

        # 注册反思评估工具（依赖 LLM）
        self.tool_registry.register_many(
            create_reflection_tools(self.rag.llm_client, self.rag.llm_model)
        )

        self._initialized = True

    def close(self) -> None:
        self.rag.close()

    # ==================== 主入口 ====================

    async def chat(
        self,
        query: str,
        conversation_id: str = "default",
        conversation: ConversationBuffer | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """ReAct Agent 主循环。

        Args:
            query: 用户问题
            conversation_id: 对话 ID（用于记忆管理）
            conversation: 可选的 ConversationBuffer 实例

        Yields:
            AgentEvent 子类（ThinkingEvent, ToolCallEvent, ToolResultEvent, TokenEvent, DoneEvent）
        """
        if not self._initialized:
            raise RuntimeError("Agent 未初始化，请先调用 initialize()")

        # 记忆管理
        conv = conversation or ConversationBuffer(
            conversation_id=conversation_id,
            max_turns=20,
            ttl_seconds=3600,
        )

        # 工作记忆（单轮内共享）
        working = WorkingMemory()

        # 构建消息历史
        messages = self._build_initial_messages(query, conv, working)

        tool_calls_count = 0
        total_tokens = 0

        # ReAct 循环
        for iteration in range(self.max_iterations):
            yield ThinkingEvent(f"Iteration {iteration + 1}/{self.max_iterations}")

            try:
                response = await self.llm.chat(
                    messages=messages,
                    tools=self.tool_registry.get_openai_schemas() if self.tool_registry else None,
                    temperature=0.3,
                )
            except Exception as exc:
                traceback.print_exc()
                yield DoneEvent(
                    final_answer=f"LLM 调用失败: {exc}",
                    tool_calls_made=tool_calls_count,
                    total_tokens=total_tokens,
                )
                return

            total_tokens += response.get("usage", {}).get("total_tokens", 0)

            # 检查是否有工具调用
            tool_calls = response.get("tool_calls")
            if tool_calls:
                # 将 assistant 消息加入历史
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.get("content"),
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                # 执行每个工具调用
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    yield ToolCallEvent(
                        tool_name=tool_name,
                        arguments=arguments,
                        call_id=tc["id"],
                    )

                    # 执行工具
                    result = self.tool_registry.execute(tool_name, arguments)
                    tool_calls_count += 1

                    yield ToolResultEvent(tool_name=tool_name, result=result)

                    # 将工具结果加入消息历史
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result.content,
                        }
                    )

                    # 更新工作记忆
                    if result.success:
                        working.set(
                            f"last_{tool_name}",
                            f"结果数: {result.metadata.get('count', 'N/A')}, "
                            f"摘要: {result.content[:200]}",
                        )
                    else:
                        working.set(
                            f"last_{tool_name}_error",
                            result.error or "unknown error",
                        )

                # 继续循环，让 LLM 看到工具结果后决定下一步
                continue

            # 没有工具调用 → 最终回答
            final_content = response.get("content") or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": final_content,
                }
            )

            # 保存对话历史
            conv.add("user", query)
            conv.add("assistant", final_content)

            yield DoneEvent(
                final_answer=final_content,
                sources=self._extract_sources_from_context(messages),
                tool_calls_made=tool_calls_count,
                total_tokens=total_tokens,
            )
            return

        # 超出最大迭代次数
        yield DoneEvent(
            final_answer="推理超出最大步数限制，请尝试简化问题。",
            tool_calls_made=tool_calls_count,
            total_tokens=total_tokens,
        )

    async def chat_stream(
        self,
        query: str,
        conversation_id: str = "default",
        conversation: ConversationBuffer | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """ReAct Agent 流式版本 — 最终答案以 token 流输出。

        工具调用阶段行为与 chat() 相同。最终回答阶段改为流式输出 token。
        """
        if not self._initialized:
            raise RuntimeError("Agent 未初始化，请先调用 initialize()")

        conv = conversation or ConversationBuffer(conversation_id=conversation_id)

        working = WorkingMemory()
        messages = self._build_initial_messages(query, conv, working)

        tool_calls_count = 0
        total_tokens = 0

        for iteration in range(self.max_iterations):
            yield ThinkingEvent(f"Iteration {iteration + 1}/{self.max_iterations}")

            # 最后几次迭代使用流式输出（没有 tool_calls 时会直接流式输出答案）
            try:
                response = await self.llm.chat(
                    messages=messages,
                    tools=self.tool_registry.get_openai_schemas() if self.tool_registry else None,
                    temperature=0.3,
                )
            except Exception as exc:
                traceback.print_exc()
                yield DoneEvent(
                    final_answer=f"LLM 调用失败: {exc}",
                    tool_calls_made=tool_calls_count,
                    total_tokens=total_tokens,
                )
                return

            total_tokens += response.get("usage", {}).get("total_tokens", 0)
            tool_calls = response.get("tool_calls")

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.get("content"),
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    yield ToolCallEvent(
                        tool_name=tool_name,
                        arguments=arguments,
                        call_id=tc["id"],
                    )

                    result = self.tool_registry.execute(tool_name, arguments)
                    tool_calls_count += 1

                    yield ToolResultEvent(tool_name=tool_name, result=result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result.content,
                        }
                    )

                    if result.success:
                        working.set(
                            f"last_{tool_name}",
                            f"结果数: {result.metadata.get('count', 'N/A')}",
                        )

                continue

            # 没有工具调用 → 流式输出最终答案
            # 从非流式响应中拿到内容后模拟流式输出
            final_content = response.get("content") or ""
            for char in final_content:
                yield TokenEvent(token=char)
                await asyncio.sleep(0)  # 让出控制权

            messages.append(
                {
                    "role": "assistant",
                    "content": final_content,
                }
            )

            conv.add("user", query)
            conv.add("assistant", final_content)

            yield DoneEvent(
                final_answer=final_content,
                sources=self._extract_sources_from_context(messages),
                tool_calls_made=tool_calls_count,
                total_tokens=total_tokens,
            )
            return

        yield DoneEvent(
            final_answer="推理超出最大步数限制，请尝试简化问题。",
            tool_calls_made=tool_calls_count,
            total_tokens=total_tokens,
        )

    # ==================== 内部方法 ====================

    def _build_initial_messages(
        self,
        query: str,
        conversation: ConversationBuffer,
        working: WorkingMemory,
    ) -> list[dict]:
        """构建初始消息列表（system prompt + 历史 + 当前问题）。"""
        # 工具描述
        tool_descriptions = (
            "\n".join(
                f"- **{name}**: {tool.description}"
                for name, tool in self.tool_registry._tools.items()
            )
            if self.tool_registry
            else "（无可用工具）"
        )

        # 上下文信息
        context_parts = []
        if conversation and len(conversation) > 0:
            context_parts.append(f"已有 {len(conversation)} 条历史消息")
        if working and len(working) > 0:
            context_parts.append(working.to_prompt_context())

        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            context_info="\n".join(context_parts) if context_parts else "无特殊上下文",
        )

        messages = [{"role": "system", "content": system_content}]

        # 加入历史对话
        if conversation and len(conversation) > 0:
            messages.extend(conversation.get_messages(max_turns=5))

        # 加入当前问题
        messages.append({"role": "user", "content": query})

        return messages

    @staticmethod
    def _extract_sources_from_context(messages: list[dict]) -> list[dict]:
        """从消息历史中提取检索到的来源信息。"""
        sources = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                # 简单提取页码引用
                import re

                pages = re.findall(r"页码(\d+)", content)
                for p in pages:
                    sources.append({"page": int(p), "type": "retrieved_chunk"})
        return sources
