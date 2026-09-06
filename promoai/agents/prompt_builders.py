from __future__ import annotations

from dataclasses import dataclass

from promoai.agents.state import ProcessState


def _latest_request(state: ProcessState) -> str:
    """Read the aggregate through its API, with support for legacy mappings."""
    if hasattr(state, "latest_request"):
        return state.latest_request
    return state["user_request"][-1]


@dataclass(frozen=True)
class AnalystPromptInput:
    artifact_dataframe_section: str
    artifact_visualization_section: str
    context: str


class EngineerPromptBuilder:
    """Builds Engineer conversations without performing I/O or model calls."""

    def build(self, state: ProcessState, api_summary: str) -> list[dict[str, str]]:
        system_message = f"""
    You are a Process Mining Data Engineer.

    Your task is to preprocess the event log data according to the user's request, using the provided API methods and any necessary data manipulation with pandas or numpy.
    {api_summary} \n

    DATA: \n
    - Assume that the log is already loaded and accessible via api.event_log. \n
    - Furthermore, the data has the following structure: {state['log_abstraction']} \n
    - Activities should be stored in the 'concept:name' column, timestamps in 'time:timestamp', case identifier in 'case:concept:name'. \n
    EXPECTED OUTPUT:
    - Always return the final event log as `final_event_log` variable after any preprocessing steps, so it can be passed down to the subsequent agent.

    IMPORTANT:
    - If the previous code you generated is able to answer the user's new request without any modifications, just return the event log without any changes.
    - MANDATORY PREPROCESSING: if the actvities or timestamps or case identifiers are not in the expected columns, you need to preprocess the log to ensure they are. You can use the API methods or pandas for this. Otherwise, you will trigger errors in the subsequent steps. \n
    """
        if not state["messages_eng"]:
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": _latest_request(state)},
            ]
        return state["messages_eng"] + [
            {"role": "user", "content": _latest_request(state)}
        ]


class AnalystPromptBuilder:
    """Builds Analyst conversations from session and artifact context."""

    def build(
        self, state: ProcessState, prompt_input: AnalystPromptInput
    ) -> list[dict[str, str]]:
        messages = state["messages_ana"]
        if not messages:
            messages.append(
                {
                    "role": "system",
                    "content": self._initial_message(state, prompt_input.context),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": self._follow_up_message(
                        _latest_request(state), prompt_input.context
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": self._artifact_message(state, prompt_input),
            }
        )
        return messages

    @staticmethod
    def _follow_up_message(last_request: str, context: str) -> str:
        return f"""
    The Process Engineer has executed the following preprocessing steps on the event log based on the user's request "{context}":
    Please use this information to generate a report, making sure to reference artifacts that are relevant to the current user request and to support your analysis with the provided visualizations and data summaries.

    REMEMBER:
    - YOUR GOAL is to answer the User Request: "{last_request}" by presenting and interpreting the provided artifacts. \n
    - Recall the output format: \n
    ```python
    final_report = [
        {{"type": "text", "content": "### Analysis of Process Distribution\\nAs requested, I have analyzed the activity distribution..."}},
        {{"type": "artifact", "content": "artifact_0"}},
        {{"type": "text", "content": "The chart above (artifact_0) clearly shows that..."}}
    ]
    ```
    \n
    - Make sure to follow the strict rules introduced in the initial message (e.g., referencing artifacts by their exact keys, including the visualization if the user asked for it and it is provided, etc.) when generating your report.
    - Do NOT mention any errors or internal processing steps. Your report should be a polished analysis that directly addresses the user's request using the provided artifacts as evidence to support your conclusions.
    - AVOID adding a dataframe/table to the report unless the user EXPLICITLY asks for it. Instead, summarize the key insights from it.
    You will receive the artifacts in the consequent message.
    """

    @staticmethod
    def _initial_message(state: ProcessState, context: str) -> str:
        return f"""
    You are a process analyst.

    CRITICAL INSTRUCTION:
    The Process Engineer has already executed the analysis. Your SOLE TASK is to compile the final report using the artifacts provided below.
    - DO NOT ask the user if they want to generate a visualization; it has ALREADY been generated.
    - DO NOT say "I can generate..." or "If you want me to...".
    - YOUR GOAL is to answer the User Request: "{_latest_request(state)}" by presenting and interpreting the provided artifacts.

    DATA SUMMARY:
    - Log Abstraction: {state['log_abstraction']}
    - Additional Context: {context}

    TASK:
    Construct the `final_report` variable. Interleave your analysis with the relevant artifact keys (e.g., "artifact_0").
    - If an artifact is a visualization (PNG), use it to support your findings.
    - If an artifact is a table (CSV), summarize the key insights in text if it's relevant. DO NOT include the dataframe in the report unless EXPLICITLY stated by user in the request.
    - For visualizations, you will get an abstraction in form of a description (content of the artifact or its summary). Use it to identify if it is relevant and to support your analysis.

    OUTPUT FORMAT:
    Your output must be Python code that defines a variable `final_report` wrapped in a python code block (```python ... ```).

    Example:
    ```python
    final_report = [
        {{"type": "text", "content": "### Analysis of Process Distribution\\nAs requested, I have analyzed the activity distribution..."}},
        {{"type": "artifact", "content": "artifact_0"}},
        {{"type": "text", "content": "The chart above (artifact_0) clearly shows that..."}}
    ]
    ```

    STRICT RULES:
    1. Reference artifacts by their EXACT keys (e.g., "artifact_0").
    2. Do NOT mention any errors or internal processing steps.
    3. If the user asked for a visualization and it is provided in the ARTIFACTS section, you MUST include it in the report.
    4. You may reuse artifacts from previous iterations by referencing their keys, but you cannot introduce new artifacts that were not provided in the ARTIFACTS section.
    5. Before generating the report, carefully analyze the provided artifacts, respond after making sure you have fully utilized the available information to answer the user's request.
    6. In the text, DO NOT refer to the artifacts as "artifact_0" but rather as "the chart above", "the chart below", etc. based on the type of artifact and its relevance to the analysis.
    7. DO NOT include dataframes in your report unless explicitly asked, instead, summarize the key insights from the dataframe in text form.
"""

    @staticmethod
    def _artifact_message(
        state: ProcessState, prompt_input: AnalystPromptInput
    ) -> str:
        return f"These are the artifacts that have been generated in the current iteration \
        and are relevant to the user's request but have not been included in the previous report: \n \n \
        {prompt_input.artifact_dataframe_section} \n\n  \
        {prompt_input.artifact_visualization_section} \n\n \
        Recall the user request: {_latest_request(state)} \n\n "
