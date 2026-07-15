"""Domain models for the Legal Contract Intelligence Platform.

Multi-tenant architecture: each Tenant has isolated document sets.
Documents contain ContractClauses, and analysis produces RiskReports.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ── Enums ──────────────────────────────────────────────────

class DocumentStatus(enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    PARTIAL = "partial"  # Some chunks ingested, some failed


class ContractType(enum.Enum):
    NDA = "nda"                         # Non-disclosure agreement
    EMPLOYMENT = "employment"           # Employment contract
    SERVICE = "service"                 # Service agreement
    LEASE = "lease"                     # Lease/rental agreement
    LICENSE = "license"                 # License agreement
    PURCHASE = "purchase"              # Purchase/sales agreement
    LOAN = "loan"                       # Loan agreement
    PARTNERSHIP = "partnership"        # Partnership/joint venture
    M_AND_A = "m_and_a"               # Merger & acquisition
    OTHER = "other"


class ClauseType(enum.Enum):
    # Common clause types across contract types
    TERMINATION = "termination"         # Termination clauses
    CONFIDENTIALITY = "confidentiality" # Confidentiality/NDA
    INDEMNIFICATION = "indemnification" # Indemnification
    LIABILITY = "liability"            # Limitation of liability
    PAYMENT = "payment"                # Payment terms
    DELIVERY = "delivery"              # Delivery/performance
    GOVERNING_LAW = "governing_law"    # Governing law/jurisdiction
    DISPUTE_RESOLUTION = "dispute"     # Dispute resolution/arbitration
    FORCE_MAJEURE = "force_majeure"    # Force majeure
    INTELLECTUAL_PROPERTY = "ip"       # IP rights
    NON_COMPETE = "non_compete"        # Non-compete
    ASSIGNMENT = "assignment"          # Assignment/transfer
    WARRANTY = "warranty"              # Warranty/representation
    INSURANCE = "insurance"            # Insurance
    PENALTY = "penalty"                # Penalty/liquidated damages


class RiskLevel(enum.Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Domain Entities ─────────────────────────────────────────

@dataclass
class Tenant:
    """A tenant (law firm or corporate legal department)."""
    id: str
    name: str
    api_key_hash: str
    rate_limit_tokens_per_min: int = 60
    created_at: datetime = field(default_factory=datetime.now)
    document_count: int = 0
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rate_limit": self.rate_limit_tokens_per_min,
            "created_at": self.created_at.isoformat(),
            "document_count": self.document_count,
            "is_active": self.is_active,
        }


@dataclass
class LegalDocument:
    """A legal contract document ingested into the system."""
    id: str
    tenant_id: str
    filename: str
    contract_type: ContractType
    parties: list[str] = field(default_factory=list)
    effective_date: str | None = None   # ISO date string
    expiration_date: str | None = None
    chunk_count: int = 0
    ingested_at: datetime = field(default_factory=datetime.now)
    status: DocumentStatus = DocumentStatus.PROCESSING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "contract_type": self.contract_type.value,
            "parties": self.parties,
            "effective_date": self.effective_date,
            "expiration_date": self.expiration_date,
            "chunk_count": self.chunk_count,
            "ingested_at": self.ingested_at.isoformat(),
            "status": self.status.value,
        }


@dataclass
class ContractClause:
    """An extracted clause from a legal document."""
    id: str
    document_id: str
    tenant_id: str
    clause_type: ClauseType
    title: str                          # e.g., "第十条 违约责任"
    content: str                        # Full clause text
    page_number: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "clause_type": self.clause_type.value,
            "title": self.title,
            "content": self.content[:500],
            "page_number": self.page_number,
        }


@dataclass
class RiskReport:
    """Risk assessment result for a contract or clause."""
    id: str
    document_id: str
    tenant_id: str
    clause_type: ClauseType | None = None
    risk_level: RiskLevel = RiskLevel.NONE
    risk_score: float = 0.0            # 0.0 = safe, 1.0 = highest risk
    findings: list[RiskFinding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "clause_type": self.clause_type.value if self.clause_type else None,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": self.recommendations,
        }


@dataclass
class RiskFinding:
    """A specific risk identified in a contract clause."""
    category: str                       # e.g., "unilateral_termination"
    description: str
    severity: RiskLevel
    clause_reference: str               # e.g., "第12条第3款"
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity.value,
            "clause_reference": self.clause_reference,
            "suggested_fix": self.suggested_fix,
        }


# ── Query Models ────────────────────────────────────────────

@dataclass
class ContractQuery:
    """A query submitted to the contract intelligence agent."""
    text: str
    tenant_id: str
    intent: str = "qa"                  # qa | clause_extraction | risk_assessment | comparison
    target_document_ids: list[str] = field(default_factory=list)
    target_clause_types: list[ClauseType] = field(default_factory=list)
    max_results: int = 5
    stream: bool = True
