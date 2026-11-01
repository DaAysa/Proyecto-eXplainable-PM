from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypeAlias, TypedDict


class ReportBlock(TypedDict, total=False):
    type: Literal["text", "code", "artifact", "verification"]
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


@dataclass(frozen=True)
class AnalystResult:
    """Typed output produced by the Analyst agent."""

    report: StructuredReport
    generated_code: str
    messages: list[AgentMessage]
    sent_artifact_ids: tuple[str, ...]


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
class PMaxWorkflowResult:
    """Complete result of the Engineer -> Analyst -> optional Verifier workflow."""

    engineer: EngineerResult
    analyst: AnalystResult
    intermediate_results: tuple[Any, ...] = ()
    verification: VerificationResult | None = None


class WorkflowStep(Protocol):
    """Extension point for agents that run between Engineer and Analyst."""

    name: str

    def execute(
        self, state: Any, progress_callback: ProgressCallback | None = None
    ) -> Any:
        ...
