from __future__ import annotations

import pandas as pd
import pm4py

from promoai.agents.contracts import AnalystResult, EngineerResult
from promoai.agents.state import ProcessState
from promoai.general_utils.artifact_store import (
    append_manifest_entry,
    create_managed_path,
    write_json_artifact,
    write_text_artifact,
)


class AgentRunRecorder:
    """Persists agent outputs while keeping persistence out of agent logic."""

    def record_engineer(self, state: ProcessState, result: EngineerResult) -> None:
        request_index = len(state["user_request"])
        self._record_code(
            state,
            "engineer",
            request_index,
            state["user_request"][-1],
            result.generated_code,
        )
        self._record_event_log(
            state, result.event_log, request_index, "engineer_output"
        )

    def record_analyst(self, state: ProcessState, result: AnalystResult) -> None:
        request_index = len(state["user_request"])
        self._record_code(
            state,
            "analyst",
            request_index,
            state["user_request"][-1],
            result.generated_code,
        )
        report_path = write_json_artifact(
            state["artifact_session_dir"],
            "reports",
            f"assistant_report_request_{request_index}",
            result.report,
            prefix="report",
        )
        append_manifest_entry(
            state["artifact_session_dir"],
            category="reports",
            file_path=report_path,
            description=f"Structured analyst report for request {request_index}.",
            artifact_type="report",
            extra={"entries": len(result.report)},
        )

    @staticmethod
    def _record_code(
        state: ProcessState,
        role: str,
        request_index: int,
        user_request: str,
        code: str,
    ) -> str:
        file_path = write_text_artifact(
            state["artifact_session_dir"],
            "code",
            f"{role}_step_{request_index}_{user_request}",
            code,
            suffix=".py",
            prefix=role,
        )
        append_manifest_entry(
            state["artifact_session_dir"],
            category="code",
            file_path=file_path,
            description=f"{role.title()} generated code for request {request_index}.",
            artifact_type="generated_code",
            extra={
                "role": role,
                "request_index": request_index,
                "user_request": user_request,
            },
        )
        return file_path

    @staticmethod
    def _record_event_log(
        state: ProcessState, event_log, request_index: int, label: str
    ) -> str | None:
        try:
            dataframe = (
                event_log
                if isinstance(event_log, pd.DataFrame)
                else pm4py.convert_to_dataframe(event_log)
            )
        except Exception:
            return None

        file_path = create_managed_path(
            state["artifact_session_dir"],
            "event_logs",
            f"request_{request_index}_{label}",
            ".csv",
            prefix="event_log",
        )
        dataframe.to_csv(file_path, index=False)
        append_manifest_entry(
            state["artifact_session_dir"],
            category="event_logs",
            file_path=file_path,
            description=f"Event log snapshot after {label} for request {request_index}.",
            artifact_type="event_log_snapshot",
            extra={"rows": len(dataframe), "columns": list(dataframe.columns)},
        )
        return file_path
