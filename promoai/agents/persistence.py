from __future__ import annotations

import pandas as pd
import pm4py

from promoai.agents.contracts import (
    AnalystResult,
    EvidenceCheckResult,
    EngineerResult,
    VerificationResult,
)
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
        if result.evidence_snapshot is not None:
            evidence_path = write_json_artifact(
                state["artifact_session_dir"],
                "evidence",
                f"analyst_evidence_request_{request_index}",
                result.evidence_snapshot,
                prefix="analyst_evidence",
            )
            append_manifest_entry(
                state["artifact_session_dir"],
                category="evidence",
                file_path=evidence_path,
                description=f"Evidence visible to the Analyst for request {request_index}.",
                artifact_type="analyst_evidence",
                extra={
                    "request_index": request_index,
                    "artifacts": len(result.evidence_snapshot.get("artifacts", [])),
                    "context_entries": len(result.evidence_snapshot.get("context", [])),
                },
            )

    def record_verifier(
        self, state: ProcessState, result: VerificationResult
    ) -> None:
        request_index = len(state["user_request"])
        verification_path = write_json_artifact(
            state["artifact_session_dir"],
            "reports",
            f"response_verification_request_{request_index}",
            result.as_dict(),
            prefix="verification",
        )
        append_manifest_entry(
            state["artifact_session_dir"],
            category="reports",
            file_path=verification_path,
            description=f"Response verification for request {request_index}.",
            artifact_type="response_verification",
            extra={"answers_question": result.answers_question},
        )

    def record_evidence_check(
        self, state: ProcessState, result: EvidenceCheckResult
    ) -> None:
        request_index = state.get("active_request_index", len(state["user_request"]))
        result_path = write_json_artifact(
            state["artifact_session_dir"],
            "reports",
            f"evidence_check_request_{request_index}",
            result.as_dict(),
            prefix="evidence_check",
        )
        append_manifest_entry(
            state["artifact_session_dir"],
            category="reports",
            file_path=result_path,
            description=f"Evidence check for request {request_index}.",
            artifact_type="evidence_check",
            extra={
                "request_index": request_index,
                "confidence": result.confidence,
                "claims": len(result.claims),
            },
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
