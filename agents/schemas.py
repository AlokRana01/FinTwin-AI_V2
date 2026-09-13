"""
agents/schemas.py
=================
Defines universal, strongly-typed result containers, provenance structures,
and computation classifications for FinTwin AI's multi-agent architecture.

Architectural Rules:
- Distinguishes authoritative numerical calculations from agent interpretations.
- Every numerical output must carry provenance metadata tracing back to its engine.
- Separates computation type from calibrated statistical confidence.
- Containers are deeply immutable and safe for concurrent execution.
"""

from dataclasses import dataclass, field
from typing import Mapping, Tuple, Optional, Any, Sequence, Dict, List
from types import MappingProxyType
from enum import Enum
from datetime import datetime, timezone


class AgentStatus(str, Enum):
    """
    Execution status of an individual agent or tool invocation.
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ComputationType(str, Enum):
    """
    Classifies the mathematical/computational origin of data.
    """
    DETERMINISTIC = "deterministic"  # Exact math/algebra/tax calculation (100% authoritative)
    MODEL_OUTPUT = "model_output"    # ML Model inference (XGBoost / K-Means)
    HEURISTIC = "heuristic"          # Configured domain rules / threshold evaluation
    FALLBACK = "fallback"            # Degradation / rule-based recovery output


@dataclass(frozen=True)
class AuthoritativeProvenance:
    """
    Audit trail recording the exact engine, tool, version, and execution trace of a calculation.
    """
    source_engine: str          # e.g., "HealthScoreEngine", "IndianTaxCalculator", "GoalEngine"
    source_tool: str            # e.g., "calculate_health_score", "calculate_tax"
    calculation_version: str    # e.g., "v1.2.0-deterministic"
    request_id: str             # Unique request/session identifier
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_engine": self.source_engine,
            "source_tool": self.source_tool,
            "calculation_version": self.calculation_version,
            "request_id": self.request_id,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass(frozen=True)
class AuthoritativeData:
    """
    Raw, exact numerical and structured results from deterministic engines or ML models.
    Guaranteed immutable and distinct from qualitative agent commentary.
    """
    provenance: AuthoritativeProvenance
    computation_type: ComputationType
    metrics: Mapping[str, Any]
    arrays: Optional[Mapping[str, Tuple[float, ...]]] = None
    xai_attributions: Optional[Mapping[str, float]] = None
    confidence_score: Optional[float] = None  # Populated ONLY when technically calibrated

    def __post_init__(self):
        # Enforce read-only mapping for metrics
        if isinstance(self.metrics, dict):
            object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        # Enforce read-only mappings for arrays and XAI
        if isinstance(self.arrays, dict):
            converted_arrays = {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in self.arrays.items()}
            object.__setattr__(self, "arrays", MappingProxyType(converted_arrays))
        if isinstance(self.xai_attributions, dict):
            object.__setattr__(self, "xai_attributions", MappingProxyType(dict(self.xai_attributions)))

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "provenance": self.provenance.to_dict(),
            "computation_type": self.computation_type.value,
            "metrics": dict(self.metrics),
        }
        if self.arrays:
            result["arrays"] = {k: list(v) for k, v in self.arrays.items()}
        if self.xai_attributions:
            result["xai_attributions"] = dict(self.xai_attributions)
        if self.confidence_score is not None:
            result["confidence_score"] = self.confidence_score
        return result


@dataclass(frozen=True)
class AgentResult:
    """
    Universal immutable result container returned by all specialist agents.
    Strictly isolates authoritative computation from agent interpretation and warnings.
    """
    agent_name: str
    status: AgentStatus
    intent: str
    authoritative_data: Optional[AuthoritativeData] = None
    insights: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        # Enforce immutable tuples for collections
        if not isinstance(self.insights, tuple):
            object.__setattr__(self, "insights", tuple(self.insights))
        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))
        if not isinstance(self.errors, tuple):
            object.__setattr__(self, "errors", tuple(self.errors))

    @classmethod
    def create_success(
        cls,
        agent_name: str,
        intent: str,
        authoritative_data: AuthoritativeData,
        insights: Sequence[str] = (),
        warnings: Sequence[str] = (),
    ) -> "AgentResult":
        """
        Convenience constructor for a successful agent execution.
        """
        return cls(
            agent_name=agent_name,
            status=AgentStatus.SUCCESS,
            intent=intent,
            authoritative_data=authoritative_data,
            insights=tuple(insights),
            warnings=tuple(warnings),
            errors=(),
        )

    @classmethod
    def create_partial(
        cls,
        agent_name: str,
        intent: str,
        authoritative_data: Optional[AuthoritativeData],
        insights: Sequence[str] = (),
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
    ) -> "AgentResult":
        """
        Convenience constructor for a partially degraded agent execution.
        """
        return cls(
            agent_name=agent_name,
            status=AgentStatus.PARTIAL,
            intent=intent,
            authoritative_data=authoritative_data,
            insights=tuple(insights),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    @classmethod
    def create_failure(
        cls,
        agent_name: str,
        intent: str,
        errors: Sequence[str],
        warnings: Sequence[str] = (),
    ) -> "AgentResult":
        """
        Convenience constructor for a failed agent execution.
        """
        return cls(
            agent_name=agent_name,
            status=AgentStatus.FAILED,
            intent=intent,
            authoritative_data=None,
            insights=(),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Sanitized plain dictionary representation.
        """
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "intent": self.intent,
            "authoritative_data": self.authoritative_data.to_dict() if self.authoritative_data else None,
            "insights": list(self.insights),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "timestamp": self.timestamp,
        }
