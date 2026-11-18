from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypeAlias, TypedDict


class ReportBlock(TypedDict, total=False):
    type: Literal["text", "code", "artifact", "verification", "evidence_check"]
    content: Any
    language: str
    label: str
    expanded: bool


AgentMessage: TypeAlias = dict[str, Any]
ProgressEvent: TypeAlias = dict[str, Any]
ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]
StructuredReport: TypeAlias = list[ReportBlock]


@dataclass(frozen=True)
class AgentRetryPolicy:
    """Controls normal and fallback correction attempts for an agent."""

    primary_attempts: int = 7
    fallback_attempts: int = 5

    @property
    def total_attempts(self) -> int:
        return self.primary_attempts + self.fallback_attempts


@dataclass(frozen=True)
class EngineerResult:
    """Typed output produced by the Engineer agent."""

    event_log: Any
    generated_code: str
    messages: list[AgentMessage]
    accepted_artifact_paths: tuple[str, ...] = ()
    failed_artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalystResult:
    """Typed output produced by the Analyst agent."""

    report: StructuredReport
    generated_code: str
    messages: list[AgentMessage]
    sent_artifact_ids: tuple[str, ...]
    evidence_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Assessment of whether an analyst report answers the user request."""

    answers_question: bool
    explanation: str
    missing_aspects: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "answers_question": self.answers_question,
            "explanation": self.explanation,
            "missing_aspects": list(self.missing_aspects),
        }


@dataclass(frozen=True)
class EvidenceLocation:
    """Exact source location supporting or contradicting one claim."""

    source: Literal["artifact", "context", "log_abstraction"]
    source_id: str
    artifact_id: str | None = None
    path: str | None = None
    row: int | None = None
    column: str | None = None
    value: Any = None
    claim_value: Any = None
    numeric_check: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "row": self.row,
            "column": self.column,
            "value": self.value,
            "claim_value": self.claim_value,
            "numeric_check": self.numeric_check,
        }


@dataclass(frozen=True)
class EvidenceClaim:
    """Evidence assessment for one factual or non-verifiable report claim."""

    id: str
    text: str
    question_part_ids: tuple[str, ...]
    status: Literal["supported", "contradicted", "unsupported", "non_verifiable"]
    explanation: str
    evidence: tuple[EvidenceLocation, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "question_part_ids": list(self.question_part_ids),
            "status": self.status,
            "explanation": self.explanation,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class QuestionPartAssessment:
    """Coverage assessment for one independently answerable request part."""

    id: str
    text: str
    data_type: Literal[
        "time",
        "frequency",
        "structure",
        "conformance",
        "causal",
        "prediction",
        "data_quality",
    ] | None
    status: Literal["answered", "partial", "unanswered"]
    unanswered_cause: Literal[
        "analyst_missed_available_evidence",
        "engineer_did_not_generate_evidence",
    ] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "data_type": self.data_type,
            "status": self.status,
            "unanswered_cause": self.unanswered_cause,
        }


@dataclass(frozen=True)
class EvidenceCheckResult:
    """Grounding and coverage assessment for an Analyst report."""

    claims: tuple[EvidenceClaim, ...]
    question_parts: tuple[QuestionPartAssessment, ...]
    confidence: Literal["high", "medium", "low"]
    confidence_explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claims": [claim.as_dict() for claim in self.claims],
            "question_parts": [part.as_dict() for part in self.question_parts],
            "confidence": self.confidence,
            "confidence_explanation": self.confidence_explanation,
        }


@dataclass(frozen=True)
class PMaxWorkflowResult:
    """Complete result of the PMAx workflow and its optional final checks."""

    engineer: EngineerResult
    analyst: AnalystResult
    intermediate_results: tuple[Any, ...] = ()
    verification: VerificationResult | None = None
    evidence_check: EvidenceCheckResult | None = None


class WorkflowStep(Protocol):
    """Extension point for agents that run between Engineer and Analyst."""

    name: str

    def execute(
        self, state: Any, progress_callback: ProgressCallback | None = None
    ) -> Any:
        ...
