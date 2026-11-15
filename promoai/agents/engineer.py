from __future__ import annotations

from typing import Callable

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
        api = PM4PYWrapper(state, LLMClient(self._credentials))
        messages = self._prompt_builder.build(state, api.get_API_summary())
        llm_args = self._llm_args(state, progress_callback)
        failed_artifacts: list[str] = []
        accepted_artifacts: list[str] = []

        def tracked_extraction(response: str, args=None):
            before = set(state["saved_artifacts"])
            try:
                extracted = api.code_extraction(response, args)
            except Exception:
                for path in state["saved_artifacts"]:
                    if path not in before and path not in failed_artifacts:
                        failed_artifacts.append(path)
                raise
            for path in state["saved_artifacts"]:
                if (
                    path not in before
                    and path not in failed_artifacts
                    and path not in accepted_artifacts
                ):
                    accepted_artifacts.append(path)
            return extracted

        code, event_log, updated_messages = self._result_generator(
            messages,
            extraction_function=tracked_extraction,
            llm_name=self._credentials.llm_name,
            ai_provider=self._credentials.ai_provider,
            api_key=self._credentials.api_key,
            llm_args=llm_args,
            max_iterations=self._retry_policy.primary_attempts,
            additional_iterations=self._retry_policy.fallback_attempts,
            standard_error_message=ERROR_MESSAGE_CODE_GENERATION_ENG,
        )
        return EngineerResult(
            event_log=event_log,
            generated_code=code,
            messages=updated_messages,
            accepted_artifact_paths=tuple(accepted_artifacts),
            failed_artifact_paths=tuple(failed_artifacts),
        )

    def _llm_args(
        self, state: ProcessState, progress_callback: ProgressCallback | None
    ) -> dict:
        args = dict(self._credentials.args or {})
        args["artifact_session_dir"] = state["artifact_session_dir"]
        args["agent_name"] = self.name
        args["progress_callback"] = progress_callback
        return args
