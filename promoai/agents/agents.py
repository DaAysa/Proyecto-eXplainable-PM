"""Backwards-compatible PMAx agent API.

New code should use :class:`PMaxWorkflow`. The functions in this module keep
the original public entry points working for callers that still use nodes.
"""

from __future__ import annotations

import copy
from typing import List

from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import ProgressCallback
from promoai.agents.engineer import EngineerAgent
from promoai.agents.prompt_builders import AnalystPromptBuilder
from promoai.agents.state import ProcessState
from promoai.agents.workflow import PMaxWorkflow
from promoai.general_utils.llm_connection import LLMConnection

__all__ = [
    "AnalystAgent",
    "EngineerAgent",
    "PMaxWorkflow",
    "ProcessState",
    "analyst_node",
    "engineer_node",
    "generate_initial_message_for_analyst",
    "init_state",
    "snapshot_state",
    "update_message_for_analyst",
]


def snapshot_state(
    state: ProcessState, previous_states: List[ProcessState]
) -> List[ProcessState]:
    previous_states.append(copy.deepcopy(state))
    return previous_states


def init_state(
    user_request: str,
    event_log,
    artifact_session_dir: str | None = None,
    source_log_path: str | None = None,
) -> ProcessState:
    return ProcessState(
        user_request=user_request,
        event_log=event_log,
        artifact_session_dir=artifact_session_dir,
        source_log_path=source_log_path,
    )


def engineer_node(
    state: ProcessState,
    LLMCredentials: LLMConnection,
    progress_callback: ProgressCallback | None = None,
):
    """Legacy Engineer entry point backed by the new workflow."""
    result = PMaxWorkflow(LLMCredentials).run_engineer(state, progress_callback)
    return state, result.event_log, result.generated_code


def analyst_node(
    state: ProcessState,
    LLMCredentials: LLMConnection,
    progress_callback: ProgressCallback | None = None,
) -> ProcessState:
    """Legacy Analyst entry point backed by the new workflow."""
    PMaxWorkflow(LLMCredentials).run_analyst(state, progress_callback)
    return state


def update_message_for_analyst(last_request: str, context: str) -> str:
    """Compatibility wrapper for the former prompt helper."""
    return AnalystPromptBuilder._follow_up_message(last_request, context)


def generate_initial_message_for_analyst(
    state: ProcessState, context: str
) -> str:
    """Compatibility wrapper for the former prompt helper."""
    return AnalystPromptBuilder._initial_message(state, context)
