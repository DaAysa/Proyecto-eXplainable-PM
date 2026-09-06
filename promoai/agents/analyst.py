from __future__ import annotations

from functools import partial
from typing import Callable

from promoai.agents.artifact_catalog import ArtifactCatalog
from promoai.agents.contracts import AgentRetryPolicy, AnalystResult, ProgressCallback
from promoai.agents.prompt_builders import AnalystPromptBuilder
from promoai.agents.state import ProcessState
from promoai.agents.utils import code_extraction_report
from promoai.general_utils.llm_connection import (
    LLMConnection,
    generate_result_with_error_handling,
)


ERROR_MESSAGE_CODE_GENERATION_ANALYST = """
Please update the model to fix the error. Make sure to save the final report in the variable 'final_report' as a list of dictionaries with keys 'type' and 'content', where 'type' can be either 'text' or 'artifact'. For artifacts, the content should be the key referencing the artifact (e.g., 'artifact_0') that you want to include in the report."""


class AnalystAgent:
    """Produces a structured report from Engineer context and artifacts."""

    name = "Analyst"

    def __init__(
        self,
        credentials: LLMConnection,
        prompt_builder: AnalystPromptBuilder | None = None,
        result_generator: Callable = generate_result_with_error_handling,
        retry_policy: AgentRetryPolicy | None = None,
    ):
        self._credentials = credentials
        self._prompt_builder = prompt_builder or AnalystPromptBuilder()
        self._result_generator = result_generator
        self._retry_policy = retry_policy or AgentRetryPolicy()

    def execute(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalystResult:
        catalog = ArtifactCatalog.from_saved_artifacts(state["saved_artifacts"])
        prompt_input = catalog.build_prompt_input(
            state["sent_artifacts"], state["context"]
        )
        messages = self._prompt_builder.build(state, prompt_input)
        parser = partial(
            code_extraction_report,
            valid_artifact_ids=catalog.ids,
            long_dfs=catalog.long_dataframe_ids,
        )

        try:
            report_code, report, updated_messages = self._result_generator(
                messages,
                extraction_function=parser,
                llm_name=self._credentials.llm_name,
                ai_provider=self._credentials.ai_provider,
                api_key=self._credentials.api_key,
                llm_args=self._llm_args(state, progress_callback),
                max_iterations=self._retry_policy.primary_attempts,
                additional_iterations=self._retry_policy.fallback_attempts,
                standard_error_message=ERROR_MESSAGE_CODE_GENERATION_ANALYST,
            )
            resolved_report = catalog.resolve_report_paths(report)
        except Exception as exc:
            raise Exception(f"Error during analyst node execution: {exc}") from exc

        return AnalystResult(
            report=resolved_report,
            generated_code=report_code,
            messages=updated_messages,
            sent_artifact_ids=tuple(catalog.ids),
        )

    def _llm_args(
        self, state: ProcessState, progress_callback: ProgressCallback | None
    ) -> dict:
        args = dict(self._credentials.args or {})
        args["artifact_session_dir"] = state["artifact_session_dir"]
        args["agent_name"] = self.name
        args["progress_callback"] = progress_callback
        return args
