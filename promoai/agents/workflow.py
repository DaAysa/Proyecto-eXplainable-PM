from __future__ import annotations

import logging

from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import (
    AnalystResult,
    EngineerResult,
    PMaxWorkflowResult,
    ProgressCallback,
    VerifierResult,
    WorkflowStep,
)
from promoai.agents.engineer import EngineerAgent
from promoai.agents.persistence import AgentRunRecorder
from promoai.agents.state import ProcessState
from promoai.agents.verifier import UNAVAILABLE_MESSAGE, VerifierAgent
from promoai.general_utils.llm_connection import LLMConnection


class PMaxWorkflow:
    """Coordinates PMAx agents and owns state transitions between them."""

    def __init__(
        self,
        credentials: LLMConnection,
        engineer: EngineerAgent | None = None,
        analyst: AnalystAgent | None = None,
        recorder: AgentRunRecorder | None = None,
        intermediate_steps: list[WorkflowStep] | None = None,
        verifier: VerifierAgent | None = None,
    ):
        self.engineer = engineer or EngineerAgent(credentials)
        self.analyst = analyst or AnalystAgent(credentials)
        self.recorder = recorder or AgentRunRecorder()
        self.intermediate_steps = list(intermediate_steps or [])
        self.verifier = verifier or VerifierAgent(credentials)

    def run_engineer(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> EngineerResult:
        state["verification_result"] = None
        state["verifier_evidence"] = None
        state["analyst_report_request"] = None
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
        state["verification_result"] = None
        state["analyst_report_request"] = None
        result = self.analyst.execute(state, progress_callback)
        if hasattr(state, "apply_analyst_result"):
            state.apply_analyst_result(result)
        else:
            state["messages_ana"] = result.messages
            state["final_report"] = result.report
            state["sent_artifacts"].extend(result.sent_artifact_ids)
            state.flush_context()
        state["analyst_report_request"] = len(state["user_request"])
        self.recorder.record_analyst(state, result)
        return result

    def run_verifier(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> VerifierResult:
        try:
            result = self.verifier.execute(state, progress_callback)
            if not isinstance(result, VerifierResult):
                raise TypeError("Invalid Verifier result.")
            payload = result.to_dict()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Verifier unavailable (%s).", type(exc).__name__
            )
            result = VerifierResult(error=UNAVAILABLE_MESSAGE)
            payload = result.to_dict()
        state["verification_result"] = payload
        record = getattr(self.recorder, "record_verifier", None)
        if record is not None:
            try:
                record(state, result)
            except Exception as exc:
                # Review persistence must never prevent delivery of the Analyst report.
                logging.getLogger(__name__).warning(
                    "Could not persist verification (%s).", type(exc).__name__
                )
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
        verifier_result = self.run_verifier(state, progress_callback)
        return PMaxWorkflowResult(
            engineer=engineer_result,
            analyst=analyst_result,
            intermediate_results=intermediate_results,
            verifier=verifier_result,
        )
