"""POST /api/v1/chat — SSE 流式 Agent 对话。"""

from __future__ import annotations

import json
import logging
import traceback
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import ChatRequest, ChatResponse, Source
from src.api.dependencies import get_agent, conversation_manager
from src.agent.agent import (
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    TokenEvent,
    DoneEvent,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未就绪，请先上传 PDF")

    conversation_id = request.conversation_id or str(uuid.uuid4())[:8]
    conv = conversation_manager.get_or_create(conversation_id)

    if request.stream:
        return EventSourceResponse(
            _stream_events(agent, request.query, conv, conversation_id),
            media_type="text/event-stream",
        )
    else:
        final_answer = ""
        sources = []
        tool_count = 0
        token_count = 0

        async for event in agent.chat(request.query, conversation=conv):
            match event:
                case TokenEvent(token=t):
                    final_answer += t
                case DoneEvent(final_answer=ans, sources=srcs, tool_calls_made=tc, total_tokens=tt):
                    if ans:
                        final_answer = ans
                    sources = [
                        Source(text=s.get("text", ""), page=s.get("page", 0), score=0)
                        for s in srcs
                    ]
                    tool_count = tc
                    token_count = tt

        return ChatResponse(
            answer=final_answer,
            sources=sources,
            conversation_id=conversation_id,
            tool_calls_count=tool_count,
        )


async def _stream_events(agent, query: str, conv, conversation_id: str):
    """SSE 事件生成器。"""
    try:
        async for event in agent.chat(query, conversation=conv):
            match event:
                case ThinkingEvent(content=c):
                    yield {
                        "event": "thinking",
                        "data": json.dumps({"content": c}, ensure_ascii=False),
                    }
                case ToolCallEvent(tool_name=tn, arguments=args):
                    yield {
                        "event": "tool_call",
                        "data": json.dumps(
                            {"tool": tn, "arguments": args}, ensure_ascii=False
                        ),
                    }
                case ToolResultEvent(tool_name=tn, result=r):
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({
                            "tool": tn,
                            "success": r.success,
                            "result_count": r.metadata.get("count", 0),
                            "preview": r.content[:200],
                            "error": r.error,
                        }, ensure_ascii=False),
                    }
                case TokenEvent(token=t):
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": t}, ensure_ascii=False),
                    }
                case DoneEvent(final_answer=ans, sources=srcs, tool_calls_made=tc, total_tokens=tt):
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "answer": ans,
                            "conversation_id": conversation_id,
                            "tool_calls_count": tc,
                            "total_tokens": tt,
                            "sources": [
                                {"text": s.get("text", ""), "page": s.get("page", 0)}
                                for s in srcs
                            ],
                        }, ensure_ascii=False),
                    }
    except Exception as exc:
        logger.error(f"SSE stream error: {exc}")
        traceback.print_exc()
        yield {
            "event": "error",
            "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
        }
