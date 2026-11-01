from __future__ import annotations

import json
import re
from typing import Callable

from promoai.agents.contracts import (
    ProgressCallback,
    StructuredReport,
    VerificationResult,
)
from promoai.agents.state import ProcessState
from promoai.general_utils.llm_connection import LLMConnection, query_llm


class VerifierAgent:
    """Checks whether an Analyst report answers the latest user request.

    This agent assesses relevance and completeness only. It deliberately does
    not verify facts, metrics, provenance, writing style, or artifact content.
    """

    name = "Verifier"

    def __init__(
        self,
        credentials: LLMConnection,
        response_generator: Callable = query_llm,
    ):
        self._credentials = credentials
        self._response_generator = response_generator

    def execute(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> VerificationResult:
        messages = self._build_messages(
            state.latest_request
            if hasattr(state, "latest_request")
            else state["user_request"][-1],
            state["final_report"],
        )
        llm_args = dict(self._credentials.args or {})
        llm_args["artifact_session_dir"] = state["artifact_session_dir"]

        self._notify(
            progress_callback,
            "request_started",
            attempt=1,
            total_attempts=1,
            provider=self._credentials.ai_provider,
            model=self._credentials.llm_name,
        )
        try:
            response = self._response_generator(
                messages,
                api_key=self._credentials.api_key,
                llm_name=self._credentials.llm_name,
                ai_provider=self._credentials.ai_provider,
                llm_args=llm_args,
            )
        except Exception as exc:
            self._notify(
                progress_callback,
                "request_failed",
                attempt=1,
                error=str(exc),
            )
            raise

        self._notify(progress_callback, "response_received", attempt=1)
        result = self._parse_response(response)
        self._notify(progress_callback, "attempt_completed", attempt=1)
        return result

    @staticmethod
    def _build_messages(
        user_request: str, report: StructuredReport | str
    ) -> list[dict[str, str]]:
        report_text = VerifierAgent._report_as_text(report)
        return [
            {
                "role": "system",
                "content": (
                    "You are a response verifier. Determine only whether the "
                    "assistant response addresses the user's question or request. "
                    "Evaluate relevance and whether every explicit part of the request "
                    "is answered. Do not fact-check values, validate evidence or "
                    "provenance, critique writing style, or suggest a rewritten answer. "
                    "Return only valid JSON with this schema: "
                    '{"answers_question": true, "explanation": "brief reason", '
                    '"missing_aspects": ["unanswered part"]}. '
                    "Use the same language as the user's request for explanation and "
                    "missing_aspects. If the request is fully answered, return an empty "
                    "missing_aspects list."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{user_request}\n\n"
                    f"ASSISTANT RESPONSE:\n{report_text}"
                ),
            },
        ]

    @staticmethod
    def _report_as_text(report: StructuredReport | str) -> str:
        if isinstance(report, str):
            return report

        parts: list[str] = []
        for block in report:
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("content", "")))
            elif block_type == "artifact":
                parts.append("[The response includes a supporting artifact here.] ")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _parse_response(response: str) -> VerificationResult:
        candidate = response.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                candidate = candidate[start : end + 1]

        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Verifier returned an invalid JSON response.") from exc

        answers_question = payload.get("answers_question")
        explanation = payload.get("explanation")
        missing_aspects = payload.get("missing_aspects", [])
        if not isinstance(answers_question, bool):
            raise ValueError("Verifier field 'answers_question' must be boolean.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Verifier field 'explanation' must be a non-empty string.")
        if not isinstance(missing_aspects, list) or not all(
            isinstance(item, str) and item.strip() for item in missing_aspects
        ):
            raise ValueError("Verifier field 'missing_aspects' must be a list of strings.")

        return VerificationResult(
            answers_question=answers_question,
            explanation=explanation.strip(),
            missing_aspects=tuple(item.strip() for item in missing_aspects),
        )

    def _notify(
        self,
        callback: ProgressCallback | None,
        event: str,
        **details,
    ) -> None:
        if callback is not None:
            callback({"event": event, "agent": self.name, **details})
