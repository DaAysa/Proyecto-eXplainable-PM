from __future__ import annotations

import copy
import logging
from typing import Callable
from uuid import uuid4

from promoai.agents.contracts import AgentRetryPolicy, EngineerResult, ProgressCallback
from promoai.agents.pm4py_wrapper import LLMClient, PM4PYWrapper
from promoai.agents.prompt_builders import EngineerPromptBuilder
from promoai.agents.state import ProcessState
from promoai.general_utils.llm_connection import (
    LLMConnection,
    generate_result_with_error_handling,
)


ERROR_MESSAGE_CODE_GENERATION_ENG = """
Please update the model to fix the error. Make sure" \
                          f" to save the updated final event log is the variable 'final_event_log'. """


class EngineerAgent:
    """Generates and executes event-log preprocessing code."""

    name = "Engineer"

    def __init__(
        self,
        credentials: LLMConnection,
        prompt_builder: EngineerPromptBuilder | None = None,
        result_generator: Callable = generate_result_with_error_handling,
        retry_policy: AgentRetryPolicy | None = None,
    ):
        self._credentials = credentials
        self._prompt_builder = prompt_builder or EngineerPromptBuilder()
        self._result_generator = result_generator
        self._retry_policy = retry_policy or AgentRetryPolicy()

    def execute(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> EngineerResult:
        # Keep failed/previous artifacts for legacy behavior, but never attest to them.
        request_index = len(state["user_request"])
        state["verification_result"] = None
        state["analyst_report_request"] = None
        state["verifier_evidence"] = None
        accepted_artifacts = {}
        run_id = uuid4().hex
        api = PM4PYWrapper(state, LLMClient(self._credentials))
        messages = self._prompt_builder.build(state, api.get_API_summary())
        llm_args = self._llm_args(state, progress_callback)

        def extract_accepted_attempt(response, args=None):
            nonlocal accepted_artifacts
            before = dict(state["saved_artifacts"])
            extracted = api.code_extraction(response, args)
            # An exception above leaves no accepted snapshot for that attempt.
            changed = {
                path: artifact
                for path, artifact in state["saved_artifacts"].items()
                if path not in before or before[path] is not artifact
            }
            try:
                accepted_artifacts = copy.deepcopy(changed)
            except Exception:
                accepted_artifacts = {}
                logging.getLogger(__name__).warning(
                    "Artifact provenance snapshot unavailable."
                )
            return extracted

        code, event_log, updated_messages = self._result_generator(
            messages,
            extraction_function=extract_accepted_attempt,
            llm_name=self._credentials.llm_name,
            ai_provider=self._credentials.ai_provider,
            api_key=self._credentials.api_key,
            llm_args=llm_args,
            max_iterations=self._retry_policy.primary_attempts,
            additional_iterations=self._retry_policy.fallback_attempts,
            standard_error_message=ERROR_MESSAGE_CODE_GENERATION_ENG,
        )
        state["verifier_evidence"] = {
            "run_id": run_id,
            "request_index": request_index,
            "question": state["user_request"][-1],
            "artifacts": accepted_artifacts,
        }
        return EngineerResult(
            event_log=event_log,
            generated_code=code,
            messages=updated_messages,
        )

    def _llm_args(
        self, state: ProcessState, progress_callback: ProgressCallback | None
    ) -> dict:
        args = dict(self._credentials.args or {})
        args["artifact_session_dir"] = state["artifact_session_dir"]
        args["agent_name"] = self.name
        args["progress_callback"] = progress_callback
        return args
