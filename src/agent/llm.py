"""LLM Provider — 封装 OpenAI 兼容客户端，支持 function calling、流式、重试。"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, AsyncIterator

from openai import APIError, AsyncOpenAI, AuthenticationError


class LLMProvider:
    """OpenAI 兼容的 LLM 调用封装。

    特性:
    - 同步和异步接口
    - 自动指数退避重试（最多 3 次）
    - Function calling (tool use) 支持
    - 流式输出
    - 错误分类和友好报错
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen-plus",
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # 同步客户端（用于同步代码路径）
        from openai import OpenAI
        self._sync_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    # ==================== 异步接口 ====================

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ) -> dict:
        """异步 LLM 调用（非流式），返回完整响应。

        Returns:
            {
                "role": "assistant",
                "content": str | None,
                "tool_calls": list[dict] | None,
                "finish_reason": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int},
            }
        """
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                response = await self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message

                return {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ] or None,
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                }
            except AuthenticationError:
                raise
            except Exception:
                if attempt >= self.max_retries - 1:
                    raise
                delay = 2 ** attempt
                await asyncio.sleep(delay)

        raise RuntimeError("LLM 调用失败（超出最大重试次数）")

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> AsyncIterator[dict]:
        """异步 LLM 流式调用，逐 token 产出事件。

        Yields:
            {"type": "token", "content": str}  — 文本 token
            {"type": "tool_call", "name": str, "arguments": str}  — 工具调用（增量）
            {"type": "done", "finish_reason": str}  — 完成信号
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        tool_calls_accumulator: dict[int, dict] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # 文本 token
            if delta.content:
                yield {"type": "token", "content": delta.content}

            # 工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_accumulator:
                        tool_calls_accumulator[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    acc = tool_calls_accumulator[idx]
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

            # 完成
            if chunk.choices[0].finish_reason:
                # 先输出累积的工具调用
                for acc in tool_calls_accumulator.values():
                    if acc["name"]:
                        yield {
                            "type": "tool_call",
                            "id": acc["id"],
                            "name": acc["name"],
                            "arguments": acc["arguments"],
                        }
                yield {
                    "type": "done",
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    # ==================== 同步接口（兼容旧代码）====================

    def chat_sync(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> str:
        """同步 LLM 调用，返回纯文本（兼容旧代码）。"""
        for attempt in range(self.max_retries):
            try:
                response = self._sync_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except AuthenticationError:
                raise
            except APIError:
                if attempt >= self.max_retries - 1:
                    raise
                traceback.print_exc()
                time.sleep(2 ** attempt)
            except Exception:
                if attempt >= self.max_retries - 1:
                    raise
                traceback.print_exc()
                time.sleep(2 ** attempt)

        raise RuntimeError("LLM 调用失败（超出最大重试次数）")
