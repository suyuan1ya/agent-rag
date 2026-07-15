"""AgentRAG API 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1, max_length=5000)
    conversation_id: str | None = Field(None, description="对话 ID，不传则创建新对话")
    stream: bool = Field(True, description="是否使用 SSE 流式输出")
    knowledge_base_id: str = Field("default", min_length=1, max_length=64)


class Source(BaseModel):
    text: str = Field(..., description="检索到的文本片段")
    page: int = Field(..., description="来源页码")
    score: float = Field(..., description="相关度分数")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 最终回答")
    sources: list[Source] = Field(default_factory=list)
    conversation_id: str = Field(..., description="对话 ID")
    tool_calls_count: int = Field(0)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: str = Field("hybrid", pattern="^(dense|keyword|hybrid)$")
    top_k: int = Field(5, ge=1, le=20)
    knowledge_base_id: str = Field("default", min_length=1, max_length=64)


class SearchResponse(BaseModel):
    results: list[Source]
    query: str
    strategy: str


class HealthResponse(BaseModel):
    status: str
    version: str = "0.3.0"
    vector_store_connected: bool = False
    model_loaded: bool = False
    indexed_chunks: int = 0
