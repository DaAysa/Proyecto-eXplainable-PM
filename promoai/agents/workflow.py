from __future__ import annotations

from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import (
    AnalystResult,
    EngineerResult,
    EvidenceCheckResult,
    PMaxWorkflowResult,
    ProgressCallback,
    VerificationResult,
    WorkflowStep,
)
from promoai.agents.engineer import EngineerAgent
from promoai.agents.evidence import apply_analyst_evidence, capture_analyst_evidence
from promoai.agents.evidence_checker import EvidenceCheckerAgent
from promoai.agents.persistence import AgentRunRecorder
from promoai.agents.state import ProcessState
from promoai.agents.verifier import VerifierAgent
from promoai.general_utils.llm_connection import LLMConnection


class PMaxWorkflow:
    """Coordinates PMAx agents and owns state transitions between them."""

    def __init__(
        self,
        credentials: LLMConnection,
        engineer: EngineerAgent | None = None,
        analyst: AnalystAgent | None = None,
        verifier: VerifierAgent | None = None,
        recorder: AgentRunRecorder | None = None,
        intermediate_steps: list[WorkflowStep] | None = None,
        evidence_checker: EvidenceCheckerAgent | None = None,
    ):
        self.engineer = engineer or EngineerAgent(credentials)
        self.analyst = analyst or AnalystAgent(credentials)
        self.verifier = verifier or VerifierAgent(credentials)
        self.evidence_checker = evidence_checker or EvidenceCheckerAgent(credentials)
        self.recorder = recorder or AgentRunRecorder()
        self.intermediate_steps = list(intermediate_steps or [])

    def run_engineer(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> EngineerResult:
        result = self.engineer.execute(state, progress_callback)
        if hasattr(state, "apply_engineer_result"):
            state.apply_engineer_result(result)
        else:
            state["event_log"] = result.event_log
            state["log_abstraction"] = state.generate_log_abstraction()
            state["messages_eng"] = result.messages
        self.recorder.record_engineer(state, result)
        return result

    def run_analyst(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalystResult:
        snapshot = capture_analyst_evidence(state)
        result = self.analyst.execute(state, progress_callback)
        if result.evidence_snapshot is None:
            apply_analyst_evidence(state, snapshot)
        if hasattr(state, "apply_analyst_result"):
            state.apply_analyst_result(result)
        else:
            apply_analyst_evidence(state, result.evidence_snapshot)
            state["messages_ana"] = result.messages
            state["final_report"] = result.report
            state["sent_artifacts"].extend(result.sent_artifact_ids)
            state.flush_context()
        self.recorder.record_analyst(state, result)
        return result

    def run_evidence_checker(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> EvidenceCheckResult:
        try:
            result = self.evidence_checker.execute(state, progress_callback)
            if hasattr(state, "apply_evidence_check_result"):
                state.apply_evidence_check_result(result)
            else:
                state["evidence_check"] = result.as_dict()
            self.recorder.record_evidence_checker(state, result)
        except Exception as exc:
            state["evidence_check"] = {"unavailable": True, "explanation": str(exc)}
            raise
        return result

    def run_verifier(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> VerificationResult:
        result = self.verifier.execute(state, progress_callback)
        if hasattr(state, "apply_verification_result"):
            state.apply_verification_result(result)
        else:
            state["response_verification"] = result.as_dict()
        self.recorder.record_verifier(state, result)
        return result

    def execute(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> PMaxWorkflowResult:
        engineer_result = self.run_engineer(state, progress_callback)
        intermediate_results = tuple(
            step.execute(state, progress_callback)
            for step in self.intermediate_steps
        )
        analyst_result = self.run_analyst(state, progress_callback)
        verification_result = None
        state["response_verification"] = None
        state["evidence_check"] = None
        if state.get("response_validation_enabled", False):
            try:
                verification_result = self.run_verifier(state, progress_callback)
            except Exception as exc:
                state["response_verification"] = {"unavailable": True, "explanation": str(exc)}
        evidence_result = None
        if state.get("evidence_check_enabled", False):
            try:
                evidence_result = self.run_evidence_checker(state, progress_callback)
            except Exception:
                # A check failure must preserve the completed Analyst report.
                pass
        return PMaxWorkflowResult(
            engineer=engineer_result,
            analyst=analyst_result,
            intermediate_results=intermediate_results,
            verification=verification_result,
            evidence_check=evidence_result,
        )
