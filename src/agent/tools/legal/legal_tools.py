"""Legal-specific Agent tools — clause extraction, comparison, risk assessment."""

from __future__ import annotations

import json
from typing import Any

from ..base import Tool, ToolResult, tool


def create_legal_tools(llm_client: Any, llm_model: str) -> list[Tool]:
    """Create legal-domain tools for contract intelligence.

    Args:
        llm_client: OpenAI-compatible async client
        llm_model: model name for legal reasoning tasks

    Returns:
        [extract_clause, compare_clauses, assess_risk, check_compliance]
    """

    @tool(
        name="extract_clause",
        description="Extract specific clause type from contract documents. "
                    "Searches for clauses like termination, confidentiality, indemnification, "
                    "payment terms, etc. Returns complete clause text with page numbers. "
                    "Use this when asked to find a specific type of clause.",
        parameters={
            "type": "object",
            "properties": {
                "clause_type": {
                    "type": "string",
                    "description": "Type of clause to extract: termination, confidentiality, "
                                   "indemnification, liability, payment, delivery, governing_law, "
                                   "dispute_resolution, force_majeure, ip, non_compete, penalty, "
                                   "warranty, insurance",
                },
                "context_summary": {
                    "type": "string",
                    "description": "Brief summary of retrieved context for the LLM to analyze",
                },
            },
            "required": ["clause_type", "context_summary"],
        },
    )
    async def extract_clause(clause_type: str, context_summary: str) -> ToolResult:
        from ..prompts.legal.legal_prompts import CLAUSE_EXTRACTION_PROMPT

        prompt = CLAUSE_EXTRACTION_PROMPT.format(
            clause_type=clause_type,
            context=context_summary[:6000],
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
            return ToolResult(
                content=json.dumps(result, ensure_ascii=False, indent=2),
                metadata={
                    "clause_type": clause_type,
                    "found": result.get("found", False),
                    "count": len(result.get("clauses", [])),
                },
            )
        except Exception as exc:
            return ToolResult(
                content=json.dumps(
                    {"found": False, "clauses": [], "note": str(exc)},
                    ensure_ascii=False,
                ),
                error=f"Clause extraction failed: {exc}",
            )

    @tool(
        name="compare_clauses",
        description="Compare the same clause type across two contracts. "
                    "Shows differences in rights, obligations, liability, and terms. "
                    "Use this when asked to compare provisions across documents.",
        parameters={
            "type": "object",
            "properties": {
                "clause_type": {
                    "type": "string",
                    "description": "Type of clause being compared",
                },
                "context_a": {
                    "type": "string",
                    "description": "Relevant clauses from Contract A",
                },
                "context_b": {
                    "type": "string",
                    "description": "Relevant clauses from Contract B",
                },
            },
            "required": ["clause_type", "context_a", "context_b"],
        },
    )
    async def compare_clauses(
        clause_type: str, context_a: str, context_b: str
    ) -> ToolResult:
        from ..prompts.legal.legal_prompts import CLAUSE_COMPARISON_PROMPT

        prompt = CLAUSE_COMPARISON_PROMPT.format(
            clause_type=clause_type,
            context_a=context_a[:4000],
            context_b=context_b[:4000],
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            return ToolResult(
                content=content,
                metadata={"clause_type": clause_type},
            )
        except Exception as exc:
            return ToolResult(
                content=f"Comparison failed: {exc}",
                error=str(exc),
            )

    @tool(
        name="assess_risk",
        description="Assess legal risks in contract clauses. Evaluates: "
                    "balance of obligations, unilateral termination rights, "
                    "indemnification asymmetry, jurisdiction favorability, "
                    "IP ownership ambiguity, auto-renewal traps, vague language. "
                    "Returns risk level, findings, and suggested fixes.",
        parameters={
            "type": "object",
            "properties": {
                "context_summary": {
                    "type": "string",
                    "description": "Retrieved contract clauses to analyze for risks",
                },
                "focus_areas": {
                    "type": "string",
                    "description": "Optional: comma-separated risk areas to focus on "
                                   "(e.g., 'termination, liability, ip'). Default: all areas.",
                },
            },
            "required": ["context_summary"],
        },
    )
    async def assess_risk(context_summary: str, focus_areas: str = "") -> ToolResult:
        from ..prompts.legal.legal_prompts import RISK_ASSESSMENT_PROMPT

        focus_note = f"\n请重点分析以下方面: {focus_areas}" if focus_areas else ""
        prompt = RISK_ASSESSMENT_PROMPT.format(
            context=context_summary[:8000] + focus_note,
        )

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
            return ToolResult(
                content=json.dumps(result, ensure_ascii=False, indent=2),
                metadata={
                    "risk_level": result.get("overall_risk_level", "unknown"),
                    "risk_score": result.get("overall_risk_score", 0.0),
                    "finding_count": len(result.get("findings", [])),
                },
            )
        except Exception as exc:
            return ToolResult(
                content=json.dumps(
                    {
                        "overall_risk_level": "unknown",
                        "overall_risk_score": 0.0,
                        "findings": [],
                        "summary": f"Risk assessment failed: {exc}",
                    },
                    ensure_ascii=False,
                ),
                error=f"Risk assessment failed: {exc}",
            )

    return [extract_clause, compare_clauses, assess_risk]
