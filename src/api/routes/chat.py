"""POST /api/v1/chat — SSE 流式 Agent 对话。"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid

from fastapi import APIRouter, HTTPException

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    from fastapi.responses import StreamingResponse

    class EventSourceResponse(StreamingResponse):
        """Small SSE fallback for environments without sse-starlette."""

        def __init__(self, content, media_type="text/event-stream", **kwargs):
            async def encode_events():
                async for item in content:
                    if isinstance(item, dict):
                        event = item.get("event", "message")
                        data = item.get("data", "")
                        yield f"event: {event}\ndata: {data}\n\n"
                    else:
                        yield item

            super().__init__(encode_events(), media_type=media_type, **kwargs)


from src.agent.orchestrator import (
    DoneEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from src.api.dependencies import conversation_manager, get_agent, restore_agent
from src.api.schemas import ChatRequest, ChatResponse, Source

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent(request.knowledge_base_id)
    if agent is None:
        agent = await asyncio.to_thread(restore_agent, request.knowledge_base_id)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未就绪，请先上传 PDF")

    conversation_id = request.conversation_id or str(uuid.uuid4())[:8]
    conv = conversation_manager.get_or_create(conversation_id, request.knowledge_base_id)

    if request.stream:
        return EventSourceResponse(
            _stream_events(agent, request.query, conv, conversation_id),
            media_type="text/event-stream",
        )

    final_answer = ""
    sources = []
    tool_count = 0
    async for event in agent.run(request.query, conversation=conv):
        if isinstance(event, TokenEvent):
            final_answer += event.token
        elif isinstance(event, DoneEvent):
            if event.final_answer:
                final_answer = event.final_answer
            sources = [
                Source(
                    text=source.get("text", ""),
                    page=source.get("page", 0),
                    score=source.get("score", 0.0),
                )
                for source in event.sources
            ]
            tool_count = event.tool_calls_made

    return ChatResponse(
        answer=final_answer,
        sources=sources,
        conversation_id=conversation_id,
        tool_calls_count=tool_count,
    )


async def _stream_events(agent, query: str, conv, conversation_id: str):
    """SSE 事件生成器。"""
    try:
        async for event in agent.run(query, conversation=conv):
            if isinstance(event, ThinkingEvent):
                yield {
                    "event": "thinking",
                    "data": json.dumps({"content": event.content}, ensure_ascii=False),
                }
            elif isinstance(event, ToolCallEvent):
                yield {
                    "event": "tool_call",
                    "data": json.dumps(
                        {"tool": event.tool_name, "arguments": event.arguments},
                        ensure_ascii=False,
                    ),
                }
            elif isinstance(event, ToolResultEvent):
                result = event.result
                yield {
                    "event": "tool_result",
                    "data": json.dumps(
                        {
                            "tool": event.tool_name,
                            "success": result.success,
                            "result_count": result.metadata.get("count", 0),
                            "preview": result.content[:200],
                            "error": result.error,
                        },
                        ensure_ascii=False,
                    ),
                }
            elif isinstance(event, TokenEvent):
                yield {
                    "event": "token",
                    "data": json.dumps({"token": event.token}, ensure_ascii=False),
                }
            elif isinstance(event, DoneEvent):
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "answer": event.final_answer,
                            "conversation_id": conversation_id,
                            "tool_calls_count": event.tool_calls_made,
                            "total_tokens": event.total_tokens,
                            "sources": [
                                {
                                    "text": source.get("text", ""),
                                    "page": source.get("page", 0),
                                    "score": source.get("score", 0.0),
                                }
                                for source in event.sources
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
    except Exception as exc:
        logger.error("SSE stream error: %s", exc)
        traceback.print_exc()
        yield {
            "event": "error",
            "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
        }
