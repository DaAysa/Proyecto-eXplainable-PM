from __future__ import annotations

from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import (
    AnalystResult,
    EngineerResult,
    PMaxWorkflowResult,
    ProgressCallback,
    VerificationResult,
    WorkflowStep,
)
from promoai.agents.engineer import EngineerAgent
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
    ):
        self.engineer = engineer or EngineerAgent(credentials)
        self.analyst = analyst or AnalystAgent(credentials)
        self.verifier = verifier or VerifierAgent(credentials)
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
        result = self.analyst.execute(state, progress_callback)
        if hasattr(state, "apply_analyst_result"):
            state.apply_analyst_result(result)
        else:
            state["messages_ana"] = result.messages
            state["final_report"] = result.report
            state["sent_artifacts"].extend(result.sent_artifact_ids)
            state.flush_context()
        self.recorder.record_analyst(state, result)
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
        if state.get("response_validation_enabled", False):
            verification_result = self.run_verifier(state, progress_callback)
        return PMaxWorkflowResult(
            engineer=engineer_result,
            analyst=analyst_result,
            intermediate_results=intermediate_results,
            verification=verification_result,
        )
