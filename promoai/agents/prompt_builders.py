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
        causal_guidance = (
            """
    SAX4BPM CAUSAL ANALYSIS:
    - SAX4BPM is a causal process-discovery method. It uses the event log's activity ordering and timestamps to infer directional execution dependencies between activities, such as activity A influencing the timing of activity B. These are observationally inferred dependencies, not proof of an interventional causal effect.
    - You do not have access to SAX4BPM's source code and must not import or call the `sax` package directly. The supported interface available to you is `api.discover_causal_dependencies(min_strength=0.3, modality="chain")`.
    - The supported modalities are `"chain"` and `"parent"`. Use `modality="parent"` when the request compares temporal quantities, elapsed durations, waiting times, or whether taking longer in one interval delays another interval. Parent modality is complementary process evidence; it does not replace calculating the requested intervals directly.
    - This method analyzes `api.event_log`, returns no value, and automatically saves two artifacts for the Analyst when available: a causal graph and a table with `cause_activity`, `effect_activity`, and `strength`. Do not recreate or save these SAX4BPM artifacts yourself. A unified SAX4BPM graph can be binary; in that case `strength=1` means only that the edge is present, not that it has maximum causal strength.
    - You MUST call `api.discover_causal_dependencies()` whenever the user asks what activity causes, influences, explains, produces, or contributes to another activity, delay, waiting time, or process behavior. Questions phrased as "why", "what causes", "por qué", "qué causa", "influye", "provoca", or similar causal language also require this call. Do this even when previous generated code can otherwise be reused.
    - Apply any requested filters or column normalization before calling it. If pandas preprocessing creates a new dataframe, assign it to `api.event_log` before the call so SAX4BPM analyzes the prepared log.
    - For an activity-to-activity causal request, the minimum valid code is: `api.discover_causal_dependencies()` followed by `final_event_log = api.event_log`.
    - For a temporal comparison, first construct both requested elapsed-time variables from correctly paired events. Pair repeated business objects with their object identifier (for example `OfferID`) rather than independently taking the first matching event in a case. Save a compact labeled dataframe containing the sample size, interval definitions, exclusions, descriptive statistics, Pearson and Spearman correlations, a regression slope, confidence interval, and p-value when they can be computed. Preserve labels from `describe()` by using `rename_axis("statistic").reset_index()` before saving. Also call `api.discover_causal_dependencies(modality="parent")` as complementary process evidence.
    - Do not replace SAX4BPM with only a DFG, variants, correlations, averages, or descriptive charts for an activity-to-activity causal request; those analyses complement SAX4BPM but do not substitute for it. Conversely, do not use a SAX4BPM activity graph as a substitute for directly analyzing requested derived durations.
    - SAX4BPM is limited here to activity-to-activity execution dependencies. Do not use it to claim that a case attribute, resource, derived duration, or other feature causes a business outcome or KPI, or that one derived duration causes another; compute direct associations for those questions and leave this causal limitation explicit for the analyst. \n
    """
            if state.get("causal_enabled", True)
            else ""
        )
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
    {causal_guidance}
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
                        _latest_request(state),
                        prompt_input.context,
                        state.get("causal_enabled", True),
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
    def _follow_up_message(
        last_request: str, context: str, causal_enabled: bool = True
    ) -> str:
        causal_guidance = (
            """
    SAX4BPM CONTEXT:
    - SAX4BPM is a causal process-discovery method that analyzes activity ordering and timestamps in an event log to infer directional execution dependencies between process activities.
    - For business-activity nodes, a relationship `A -> B` means that A is the inferred cause activity and B is the inferred effect activity. Nodes named AND, OR, XOR, or `or_n` are structural gateways, not business activities.
    - Check the supplied context before interpreting `strength`. If the unified graph is binary, `strength=1` is only an edge-presence indicator: do not rank edges or call 1 a maximum effect. Only for non-binary coefficients does `strength` indicate relative strength; stronger values represent stronger evidence within the discovered model.
    - Chain modality compares activity timings relative to a shared trace anchor. Parent modality compares local activity timings relative to a preceding process anchor. Neither modality automatically constructs two user-defined elapsed-duration variables.
    - SAX4BPM results are observational evidence about activity-to-activity process behavior. They do not prove an interventional causal effect and do not establish that a resource, case attribute, or external factor causes a business outcome.
    - AVOID adding a dataframe/table to the report unless the user EXPLICITLY asks for it. The SAX4BPM causal edge table is an exception: include it together with the causal graph when both are provided.
    - When causal artifacts are provided, call their relationships "SAX4BPM-inferred causal execution dependencies", include both the graph and edge table, and do not present them as interventionally proven causation.
    - Treat the SAX4BPM edge table and its textual summary as the source of truth for which graph edges are present. Identify the relevant cause -> effect relationships and explain their direction. Only report their strengths when the context says the values are non-binary coefficients.
    - Do not infer a relationship that is absent from the SAX4BPM evidence. If no dependencies meet the minimum strength threshold, state that clearly instead of proposing unsupported causes.
    - If the request compares derived durations or asks whether an attribute or resource causes a KPI or business outcome, use the direct statistical artifact as the primary evidence. Parent-modality SAX4BPM output is complementary process evidence and does not establish the duration-to-duration causal claim.
        """
            if causal_enabled
            else ""
        )
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
    {causal_guidance}
    You will receive the artifacts in the consequent message.
    """

    @staticmethod
    def _initial_message(state: ProcessState, context: str) -> str:
        causal_enabled = state.get("causal_enabled", True)
        causal_table_exception = (
            ", except for a SAX4BPM causal edge table, which should be included "
            "with its causal graph"
            if causal_enabled
            else ""
        )
        causal_rules = (
            """
    SAX4BPM CONTEXT:
    - SAX4BPM is a causal process-discovery method that analyzes activity ordering and timestamps in an event log to infer directional execution dependencies between process activities.
    - For business-activity nodes, a relationship `A -> B` means that A is the inferred cause activity and B is the inferred effect activity. Nodes named AND, OR, XOR, or `or_n` are structural gateways, not business activities.
    - Check the supplied context before interpreting `strength`. If the unified graph is binary, `strength=1` is only an edge-presence indicator: do not rank edges or call 1 a maximum effect. Only for non-binary coefficients does `strength` indicate relative strength; stronger values represent stronger evidence within the discovered model.
    - Chain modality compares activity timings relative to a shared trace anchor. Parent modality compares local activity timings relative to a preceding process anchor. Neither modality automatically constructs two user-defined elapsed-duration variables.
    - SAX4BPM results are observational evidence about activity-to-activity process behavior. They do not prove an interventional causal effect and do not establish that a resource, case attribute, or external factor causes a business outcome.

    8. When causal artifacts are provided, include both the graph and edge table, describe their relationships as "SAX4BPM-inferred causal execution dependencies", and do not present them as interventionally proven causation.
    9. Treat the SAX4BPM edge table and its textual summary as the source of truth for which graph edges are present. Identify the relevant cause -> effect relationships and explain their direction. Only report their strengths when the context says the values are non-binary coefficients.
    10. Do not infer relationships absent from the SAX4BPM evidence. If no dependencies meet the minimum strength threshold, state that clearly instead of proposing unsupported causes.
    11. SAX4BPM supports activity-to-activity timing dependencies here. If the user compares derived durations, use the direct paired-duration statistical artifact as the primary evidence and parent-modality SAX4BPM output only as complementary process context. If the user asks whether an attribute, resource, or other feature causes a KPI or business outcome, clearly explain that descriptive associations do not establish that causal claim.
        """
            if causal_enabled
            else ""
        )
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
    - If an artifact is a table (CSV), summarize the key insights in text if it's relevant. DO NOT include the dataframe in the report unless EXPLICITLY stated by user in the request{causal_table_exception}.
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
    7. DO NOT include dataframes in your report unless explicitly asked, instead, summarize the key insights from the dataframe in text form.{causal_table_exception}.
    {causal_rules}
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
