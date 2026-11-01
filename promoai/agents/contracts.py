from __future__ import annotations

from dataclasses import asdict, dataclass
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
class VerifierResult:
    """Independent review; an unavailable review never means a negative verdict."""

    report: dict[str, Any] | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = list(result["evidence"])
        return result


@dataclass(frozen=True)
class PMaxWorkflowResult:
    """Complete result of the Engineer -> Analyst -> Verifier workflow."""

    engineer: EngineerResult
    analyst: AnalystResult
    intermediate_results: tuple[Any, ...] = ()
    verifier: VerifierResult | None = None


class WorkflowStep(Protocol):
    """Extension point for agents that run between Engineer and Analyst."""

    name: str

    def execute(
        self, state: Any, progress_callback: ProgressCallback | None = None
    ) -> Any:
        ...
