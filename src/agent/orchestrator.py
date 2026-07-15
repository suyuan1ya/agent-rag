"""Agent Orchestrator — FSM-driven ReAct agent replacing the simple for-loop.

Key improvements over ResearchAgent:
  - Finite State Machine with explicit states and transitions
  - Serializable state for crash recovery
  - Async-native tool execution
  - Built-in evaluation loop with refinement (max 2 cycles)
  - Graceful degradation on errors
  - Event-driven output (same event types for backward compatibility)
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from .fsm import (
    AgentContext,
    AgentState,
    AgentStateMachine,
)
from .llm import LLMProvider
from .memory.conversation import ConversationBuffer
from .memory.working import WorkingMemory
from .prompts.system import SYSTEM_PROMPT_TEMPLATE
from .tools.base import ToolResult
from .tools.query_tools import create_query_tools
from .tools.reflection_tools import create_reflection_tools
from .tools.registry import ToolRegistry
from .tools.search_tools import create_search_tools

# ── Events (same as before for backward compatibility) ───────


@dataclass
class ThinkingEvent:
    content: str


@dataclass
class StateTransitionEvent:
    """NEW: emitted when the FSM transitions between states."""

    from_state: str
    to_state: str
    reason: str


@dataclass
class ToolCallEvent:
    tool_name: str
    arguments: dict
    call_id: str = ""


@dataclass
class ToolResultEvent:
    tool_name: str
    result: ToolResult


@dataclass
class TokenEvent:
    token: str


@dataclass
class DoneEvent:
    final_answer: str
    sources: list[dict] = field(default_factory=list)
    tool_calls_made: int = 0
    total_tokens: int = 0
    degraded: bool = False
    state_trace: list[str] = field(default_factory=list)


AgentEvent = Union[
    ThinkingEvent,
    StateTransitionEvent,
    ToolCallEvent,
    ToolResultEvent,
    TokenEvent,
    DoneEvent,
]


# ── Orchestrator ─────────────────────────────────────────────


class AgentOrchestrator:
    """FSM-driven agent orchestrator.

    Usage:
        orch = AgentOrchestrator(rag_engine, llm_provider)
        orch.initialize()
        async for event in orch.run("违约金条款是什么?"):
            match event:
                case StateTransitionEvent(...):
                    print(f"FSM: {event.from_state} → {event.to_state}")
                case DoneEvent(answer=ans):
                    print(f"Answer: {ans}")
    """

    def __init__(
        self,
        rag_engine,  # RAGEngine (from src.infrastructure.rag)
        max_iterations: int = 15,
        max_refinements: int = 2,
        settings=None,
    ):
        self.rag = rag_engine
        self.max_iterations = max_iterations
        self.max_refinements = max_refinements
        self.settings = settings

        self.llm: LLMProvider | None = None
        self.tool_registry: ToolRegistry | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize LLM provider and register all tools."""
        from src.core.config import get_settings

        settings = self.settings or get_settings()
        self.llm = LLMProvider(
            api_key=settings.dashscope_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )

        self.tool_registry = ToolRegistry()

        # Register 8 core tools (3 search + 3 query + 2 reflection)
        self.tool_registry.register_many(create_search_tools(self.rag))
        self.tool_registry.register_many(create_query_tools(self.llm.client, self.llm.model))
        self.tool_registry.register_many(create_reflection_tools(self.llm.client, self.llm.model))

        self._initialized = True

    def close(self) -> None:
        if hasattr(self.rag, "close"):
            self.rag.close()

    # ── public API ───────────────────────────────────────────

    async def run(
        self,
        query: str,
        conversation: ConversationBuffer | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute the full FSM-driven agent loop for a query."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        # Setup
        conv = conversation or ConversationBuffer(max_turns=20, ttl_seconds=3600)
        working = WorkingMemory()
        messages = self._build_initial_messages(query, conv, working)

        ctx = AgentContext(
            query=query,
            messages=messages,
            max_refinements=self.max_refinements,
        )

        fsm = AgentStateMachine()
        fsm.start(ctx)
        yield StateTransitionEvent("__start__", fsm.current_state.value, "Query received")

        # Execute FSM loop
        iteration = 0
        while not fsm.is_terminal() and iteration < self.max_iterations:
            iteration += 1
            state = fsm.current_state

            yield ThinkingEvent(f"[{state.value}] iteration {iteration}/{self.max_iterations}")

            try:
                await self._execute_state(state, ctx, working)
            except Exception as exc:
                traceback.print_exc()
                ctx.error_message = f"{type(exc).__name__}: {exc}"
                fsm.force_state(AgentState.ERROR, ctx)

            # Evaluate transitions
            prev_state = fsm.current_state
            new_state = fsm.transition(ctx)

            if new_state != prev_state:
                reason = self._transition_reason(prev_state, new_state, ctx)
                yield StateTransitionEvent(prev_state.value, new_state.value, reason)

        # Handle terminal states
        if fsm.current_state == AgentState.ERROR:
            yield DoneEvent(
                final_answer=ctx.error_message or "An error occurred during analysis.",
                tool_calls_made=ctx.tool_calls_made,
                total_tokens=ctx.total_tokens,
                degraded=True,
                state_trace=[s.value for s, _ in fsm.state_history],
            )
        elif iteration >= self.max_iterations:
            yield DoneEvent(
                final_answer="Analysis exceeded maximum iterations. Please simplify your query.",
                tool_calls_made=ctx.tool_calls_made,
                total_tokens=ctx.total_tokens,
                state_trace=[s.value for s, _ in fsm.state_history],
            )
        else:
            # Save conversation
            conv.add("user", query)
            conv.add("assistant", ctx.final_answer)
            yield DoneEvent(
                final_answer=ctx.final_answer,
                sources=ctx.sources,
                tool_calls_made=ctx.tool_calls_made,
                total_tokens=ctx.total_tokens,
                degraded=ctx.degraded,
                state_trace=[s.value for s, _ in fsm.state_history],
            )

    # ── state executors ──────────────────────────────────────

    async def _execute_state(
        self, state: AgentState, ctx: AgentContext, working: WorkingMemory
    ) -> None:
        """Dispatch to the appropriate state handler."""
        handlers = {
            AgentState.INTENT_ANALYSIS: self._do_intent_analysis,
            AgentState.STRATEGY_SELECTION: self._do_strategy_selection,
            AgentState.RETRIEVAL: self._do_retrieval,
            AgentState.EVALUATION: self._do_evaluation,
            AgentState.REFINEMENT: self._do_refinement,
            AgentState.ANSWER_GENERATION: self._do_answer_generation,
        }
        handler = handlers.get(state)
        if handler:
            await handler(ctx, working)

    async def _do_intent_analysis(self, ctx: AgentContext, working: WorkingMemory) -> None:
        """Classify user intent: qa, clause_extraction, risk_assessment, comparison."""
        prompt = (
            f"分析以下用户查询的意图。输出 JSON:\n"
            f'{{"intent": "qa|clause_extraction|risk_assessment|comparison", '
            f'"complexity": "simple|medium|complex", '
            f'"reasoning": "简要说明"}}\n\n'
            f"查询: {ctx.query}"
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.get("content", "{}").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
            ctx.intent = result.get("intent", "qa")
            working.set("intent", ctx.intent)
            working.set("complexity", result.get("complexity", "medium"))
            ctx.total_tokens += response.get("usage", {}).get("total_tokens", 0)
        except Exception:
            ctx.intent = "qa"  # default fallback

    async def _do_strategy_selection(self, ctx: AgentContext, working: WorkingMemory) -> None:
        """Select tool strategy based on intent and complexity."""
        complexity = working.get("complexity", "medium")

        strategies = {
            "qa": {
                "simple": ["hybrid_search"],
                "medium": ["hybrid_search", "evaluate_sufficiency"],
                "complex": [
                    "decompose_question",
                    "hybrid_search",
                    "evaluate_sufficiency",
                    "rewrite_query",
                ],
            },
            "clause_extraction": {
                "simple": ["keyword_search"],
                "medium": ["hybrid_search", "keyword_search"],
                "complex": [
                    "decompose_question",
                    "hybrid_search",
                    "keyword_search",
                    "evaluate_sufficiency",
                ],
            },
            "risk_assessment": {
                "simple": ["hybrid_search"],
                "medium": ["hybrid_search", "evaluate_sufficiency", "verify_citation"],
                "complex": [
                    "decompose_question",
                    "hybrid_search",
                    "evaluate_sufficiency",
                    "verify_citation",
                    "rewrite_query",
                ],
            },
            "comparison": {
                "simple": ["hybrid_search"],
                "medium": ["decompose_question", "hybrid_search"],
                "complex": [
                    "decompose_question",
                    "hybrid_search",
                    "evaluate_sufficiency",
                    "rewrite_query",
                    "hybrid_search",
                ],
            },
        }

        intent_strategies = strategies.get(ctx.intent, strategies["qa"])
        ctx.strategy = intent_strategies.get(complexity, intent_strategies["medium"])
        ctx.current_tool_index = 0
        working.set("strategy", json.dumps(ctx.strategy))

    async def _do_retrieval(self, ctx: AgentContext, working: WorkingMemory) -> None:
        """Execute tools sequentially according to the selected strategy."""
        if ctx.current_tool_index >= len(ctx.strategy):
            return

        tool_name = ctx.strategy[ctx.current_tool_index]

        # Determine arguments based on tool type
        arguments = self._build_tool_arguments(tool_name, ctx, working)

        result = await self.tool_registry.execute_async(tool_name, arguments)
        ctx.tool_calls_made += 1

        if result.success:
            # Store results for evaluation
            if tool_name in ("dense_search", "keyword_search", "hybrid_search"):
                ctx.retrieval_results.extend(result.metadata.get("results", []))
            working.set(f"last_{tool_name}_result", result.content[:500])
        else:
            working.set(f"last_{tool_name}_error", result.error or "unknown")

        ctx.current_tool_index += 1

    async def _do_evaluation(self, ctx: AgentContext, working: WorkingMemory) -> None:
        """Evaluate retrieval sufficiency using the evaluate_sufficiency tool."""
        if not ctx.retrieval_results:
            ctx.evaluation_sufficient = False
            return

        # Build results summary
        summary_parts = []
        summary_parts.append(f"{len(ctx.retrieval_results)} retrieved chunks")

        # Try to use evaluate_sufficiency if available
        try:
            result = await self.tool_registry.execute_async(
                "evaluate_sufficiency",
                {
                    "question": ctx.query,
                    "results_summary": "; ".join(summary_parts),
                },
            )
            ctx.tool_calls_made += 1

            if result.success and result.metadata:
                ctx.evaluation_sufficient = result.metadata.get("sufficient", True)
                ctx.evaluation_score = result.metadata.get("score", 0.5)
            else:
                # Rule-based fallback: sufficient if we have any results
                total_results = len(ctx.retrieval_results)
                ctx.evaluation_sufficient = total_results >= 2
                ctx.evaluation_score = min(total_results / 5.0, 1.0)
        except Exception:
            total_results = len(ctx.retrieval_results)
            ctx.evaluation_sufficient = total_results >= 2
            ctx.evaluation_score = min(total_results / 5.0, 1.0)

        ctx.total_tokens += 200  # estimate for evaluation LLM call

    async def _do_refinement(self, ctx: AgentContext, working: WorkingMemory) -> None:
        """Refine query and prepare for re-retrieval."""
        ctx.refinement_count += 1

        # Use rewrite_query if available and not already tried
        try:
            summary = working.get(
                "last_hybrid_search_result",
                ctx.retrieval_results[-1].get("text", "")[:300] if ctx.retrieval_results else "",
            )
            result = await self.tool_registry.execute_async(
                "rewrite_query",
                {
                    "original_query": ctx.query,
                    "context": summary,
                },
            )
            ctx.tool_calls_made += 1
            if result.success:
                refined = result.content
                working.set("refined_query", refined)
        except Exception:
            working.set("refined_query", ctx.query)

        # Reset tool index to allow re-search
        ctx.current_tool_index = 0
        ctx.retrieval_results = []

    async def _do_answer_generation(self, ctx: AgentContext, working: WorkingMemory) -> None:
        """Generate final answer from retrieved context."""
        # Build context from retrieval results
        context_parts = []
        for r in ctx.retrieval_results:
            context_parts.append(
                f"[第 {r.get('page', 0)} 页，相关度 {r.get('score', 0):.3f}]\n{r.get('text', '')}"
            )

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > 6000:
            context = context[:6000] + "..."

        prompt = (
            "你是一个专业的合同分析助手。请严格基于以下检索到的合同内容回答用户问题。\n"
            "如果检索内容不足以完全回答问题，请明确说明局限性。\n"
            "在回答中引用具体条款和位置信息。\n\n"
            f"检索内容:\n{context}\n\n"
            f"用户问题: {ctx.query}\n\n"
            "请提供专业、准确的分析回答:"
        )

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个专业的合同分析法律助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            ctx.final_answer = response.get("content", "") or "无法生成回答。"
            ctx.total_tokens += response.get("usage", {}).get("total_tokens", 0)
            ctx.sources = self._extract_sources(ctx.retrieval_results)
        except Exception as exc:
            traceback.print_exc()
            ctx.final_answer = f"回答生成失败: {exc}"
            ctx.error_message = str(exc)

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _transition_reason(from_state: AgentState, to_state: AgentState, ctx: AgentContext) -> str:
        """Generate human-readable transition reason."""
        reasons = {
            (AgentState.IDLE, AgentState.INTENT_ANALYSIS): "Starting query analysis",
            (AgentState.INTENT_ANALYSIS, AgentState.STRATEGY_SELECTION): f"Intent: {ctx.intent}",
            (
                AgentState.STRATEGY_SELECTION,
                AgentState.RETRIEVAL,
            ): f"Strategy: {', '.join(ctx.strategy)}",
            (
                AgentState.RETRIEVAL,
                AgentState.EVALUATION,
            ): f"Retrieved {len(ctx.retrieval_results)} result sets",
            (
                AgentState.EVALUATION,
                AgentState.ANSWER_GENERATION,
            ): f"Sufficient (score={ctx.evaluation_score:.2f})",
            (
                AgentState.EVALUATION,
                AgentState.REFINEMENT,
            ): f"Insufficient, refinement {ctx.refinement_count + 1}/{ctx.max_refinements}",
            (AgentState.REFINEMENT, AgentState.RETRIEVAL): "Re-searching with refined query",
            (AgentState.ANSWER_GENERATION, AgentState.DONE): "Answer ready",
        }
        return reasons.get((from_state, to_state), f"{from_state.value} → {to_state.value}")

    def _build_initial_messages(
        self, query: str, conversation: ConversationBuffer, working: WorkingMemory
    ) -> list[dict]:
        """Build initial message list for the LLM conversation."""
        tool_descriptions = (
            "\n".join(
                f"- **{name}**: {tool.description}"
                for name, tool in self.tool_registry._tools.items()
            )
            if self.tool_registry
            else "（无可用工具）"
        )

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
        if conversation and len(conversation) > 0:
            messages.extend(conversation.get_messages(max_turns=5))
        messages.append({"role": "user", "content": query})
        return messages

    def _build_tool_arguments(
        self, tool_name: str, ctx: AgentContext, working: WorkingMemory
    ) -> dict:
        """Build appropriate arguments for a tool based on context."""
        if tool_name in ("dense_search", "keyword_search", "hybrid_search"):
            # Use refined query if available, otherwise original
            query = working.get("refined_query", ctx.query)
            return {"query": query, "top_k": 5}
        elif tool_name == "decompose_question":
            return {"question": ctx.query}
        elif tool_name == "rewrite_query":
            return {"original_query": ctx.query}
        elif tool_name == "generate_hypothetical_answer":
            return {"question": ctx.query}
        elif tool_name == "evaluate_sufficiency":
            return {"question": ctx.query, "results_summary": ""}
        elif tool_name == "verify_citation":
            return {"claim": "", "source_texts": ""}
        return {}

    @staticmethod
    def _extract_sources(retrieval_results: list[dict]) -> list[dict]:
        """Extract source metadata from retrieval results."""
        sources = []
        for r in retrieval_results:
            if "page" in r:
                sources.append(
                    {
                        "page": r["page"],
                        "text": r.get("text", ""),
                        "score": r.get("score", 0.0),
                        "type": "retrieved_chunk",
                    }
                )
        return sources
