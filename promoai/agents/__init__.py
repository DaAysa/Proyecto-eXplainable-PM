"""Extensible PMAx agent architecture."""

from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import (
    AgentRetryPolicy,
    AnalystResult,
    EngineerResult,
    EvidenceCheckResult,
    PMaxWorkflowResult,
    VerificationResult,
)
from promoai.agents.engineer import EngineerAgent
from promoai.agents.state import PMaxSession, ProcessState
from promoai.agents.verifier import VerifierAgent

__all__ = [
    "AnalystAgent",
    "AnalystResult",
    "AgentRetryPolicy",
    "EngineerAgent",
    "EngineerResult",
    "EvidenceCheckerAgent",
    "EvidenceCheckResult",
    "PMaxSession",
    "PMaxWorkflow",
    "PMaxWorkflowResult",
    "ProcessState",
    "VerificationResult",
    "VerifierAgent",
]


def __getattr__(name: str):
    # Keep the public imports while avoiding eager workflow imports. The latter
    # would preload evidence_checker and break clean ``python -m`` execution.
    if name == "PMaxWorkflow":
        from promoai.agents.workflow import PMaxWorkflow

        return PMaxWorkflow
    if name == "EvidenceCheckerAgent":
        from promoai.agents.evidence_checker import EvidenceCheckerAgent

        return EvidenceCheckerAgent
    raise AttributeError(name)
