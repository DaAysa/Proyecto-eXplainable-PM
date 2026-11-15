from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from promoai.agents.artifact_catalog import ArtifactCatalog
from promoai.agents.contracts import (
    EvidenceCheckResult,
    ProgressCallback,
    StructuredReport,
)
from promoai.agents.state import ProcessState
from promoai.agents.utils import transform_dataframe_for_llms
from promoai.general_utils.llm_connection import LLMConnection, query_llm


DATA_TYPES = {
    "time",
    "frequency",
    "structure",
    "conformance",
    "causal",
    "prediction",
    "data_quality",
}
CLAIM_STATUSES = {"supported", "contradicted", "unsupported", "non_verifiable"}
PART_STATUSES = {"answered", "partial", "unanswered"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def capture_analyst_evidence(state: ProcessState) -> dict[str, Any]:
    """Capture exactly the evidence boundary before the Analyst clears context."""
    request_index = len(state["user_request"])
    catalog = ArtifactCatalog.from_saved_artifacts(state["saved_artifacts"])
    sent_ids = set(state.get("sent_artifacts", []))
    attempt = state.get("engineer_artifact_attempts", {}).get(
        str(request_index), {}
    )
    failed_paths = set(attempt.get("failed", []))

    previous_by_path: dict[str, dict[str, Any]] = {}
    for old_snapshot in state.get("evidence_snapshots", []):
        for artifact in old_snapshot.get("artifacts", []):
            previous_by_path[artifact["path"]] = artifact

    artifacts: list[dict[str, Any]] = []
    for artifact in catalog.artifacts:
        is_current = artifact.id not in sent_ids
        previous = previous_by_path.get(artifact.file_path)
        artifacts.append(
            {
                "id": artifact.id,
                "path": artifact.file_path,
                "description": artifact.description,
                "type": artifact.type,
                "request_index": (
                    request_index
                    if is_current
                    else (previous or {}).get("request_index")
                ),
                "scope": "current" if is_current else "history",
                "eligible": (
                    artifact.file_path not in failed_paths
                    and (previous or {}).get("eligible", True)
                ),
                "content_preview": _json_safe(artifact.evidence_content),
            }
        )

    snapshot = {
        "schema_version": 1,
        "request_index": request_index,
        "user_request": state["user_request"][-1],
        "artifacts": artifacts,
        "context": [str(item) for item in state.get("context", [])],
        "log_abstraction": str(state.get("log_abstraction", "")),
        "legacy_reconstruction": False,
    }
    snapshots = state.setdefault("evidence_snapshots", [])
    snapshots[:] = [
        item for item in snapshots if item.get("request_index") != request_index
    ]
    snapshots.append(snapshot)
    return snapshot


class EvidenceCheckerAgent:
    """Checks whether Analyst claims are grounded in the evidence it received."""

    name = "EvidenceChecker"

    def __init__(
        self,
        credentials: LLMConnection,
        response_generator: Callable = query_llm,
    ):
        self._credentials = credentials
        self._response_generator = response_generator

    def execute(
        self,
        state: ProcessState | dict[str, Any],
        progress_callback: ProgressCallback | None = None,
    ) -> EvidenceCheckResult:
        snapshot = self._latest_snapshot(state)
        request = snapshot["user_request"]
        report_text = self._report_as_text(state["final_report"])
        artifact_session_dir = state.get("artifact_session_dir")

        decomposition = self._call_json(
            self._decomposition_messages(request, report_text),
            progress_callback,
            attempt=1,
            total_attempts=3,
            artifact_session_dir=artifact_session_dir,
        )
        parts, claims, response_language = self._validate_decomposition(decomposition)

        eligible_artifacts = [
            artifact
            for artifact in snapshot.get("artifacts", [])
            if artifact.get("eligible", True)
        ]
        routing = self._call_json(
            self._routing_messages(parts, eligible_artifacts, snapshot),
            progress_callback,
            attempt=2,
            total_attempts=3,
            artifact_session_dir=artifact_session_dir,
        )
        routes = self._validate_routing(routing, parts, eligible_artifacts)

        verification = self._call_json(
            self._verification_messages(
                request,
                report_text,
                parts,
                claims,
                routes,
                eligible_artifacts,
                snapshot,
                response_language,
            ),
            progress_callback,
            attempt=3,
            total_attempts=3,
            artifact_session_dir=artifact_session_dir,
        )
        assessments, part_assessments, summary = self._validate_verification(
            verification, claims, parts
        )
        return self._build_result(
            parts,
            claims,
            assessments,
            part_assessments,
            routes,
            eligible_artifacts,
            summary,
        )

    @staticmethod
    def _latest_snapshot(state: dict[str, Any]) -> dict[str, Any]:
        snapshots = state.get("evidence_snapshots", [])
        request_index = len(state.get("user_request", []))
        for snapshot in reversed(snapshots):
            if snapshot.get("request_index") == request_index:
                return snapshot
        if "saved_artifacts" in state:
            return capture_analyst_evidence(state)  # type: ignore[arg-type]
        raise ValueError("No Analyst evidence snapshot is available for this request.")

    def _call_json(
        self,
        messages: list[dict[str, str]],
        callback: ProgressCallback | None,
        *,
        attempt: int,
        total_attempts: int,
        artifact_session_dir: str | None,
    ) -> dict[str, Any]:
        llm_args = dict(self._credentials.args or {})
        llm_args["temperature"] = 0
        if artifact_session_dir:
            llm_args["artifact_session_dir"] = artifact_session_dir
        self._notify(
            callback,
            "request_started",
            attempt=attempt,
            total_attempts=total_attempts,
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
            self._notify(callback, "request_failed", attempt=attempt, error=str(exc))
            raise
        self._notify(callback, "response_received", attempt=attempt)
        payload = self._parse_json(response)
        self._notify(callback, "attempt_completed", attempt=attempt)
        return payload

    @staticmethod
    def _decomposition_messages(request: str, report: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You split a process-mining question and answer into atomic units. "
                    "Identify the language of the QUESTION, not the answer or evidence, "
                    "as an ISO 639-1 code in response_language. Return only JSON. "
                    "Question parts must have id, text, and data_type, "
                    "where data_type is one of time, frequency, structure, conformance, "
                    "causal, prediction, data_quality. Claims must have id, text, part_ids, "
                    "and verifiability (verifiable or non_verifiable). Split compound factual "
                    "sentences into atomic claims. Opinions and recommendations are non_verifiable. "
                    "Schema: {\"response_language\":\"en\","
                    "\"question_parts\":[{\"id\":\"part_1\",\"text\":\"...\","
                    "\"data_type\":\"frequency\"}],\"claims\":[{\"id\":\"claim_1\","
                    "\"text\":\"...\",\"part_ids\":[\"part_1\"],"
                    "\"verifiability\":\"verifiable\"}]}"
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{request}\n\nANALYST RESPONSE:\n{report}",
            },
        ]

    @staticmethod
    def _routing_messages(
        parts: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
        snapshot: dict[str, Any],
    ) -> list[dict[str, str]]:
        metadata = []
        for artifact in artifacts:
            schema: list[str] = []
            if artifact.get("type") == "dataframe":
                try:
                    schema = list(pd.read_csv(artifact["path"], nrows=0).columns)
                except Exception:
                    pass
            preview = artifact.get("content_preview")
            metadata.append(
                {
                    "id": artifact["id"],
                    "description": artifact.get("description", ""),
                    "type": artifact.get("type"),
                    "scope": artifact.get("scope"),
                    "columns": schema,
                    "preview_keys": list(preview) if isinstance(preview, dict) else [],
                }
            )
        context_metadata = [
            {"index": index, "preview": value[:200]}
            for index, value in enumerate(snapshot.get("context", []))
        ]
        return [
            {
                "role": "system",
                "content": (
                    "Route each question part only to evidence that can answer its data type. "
                    "You receive metadata, not evidence values. Prefer current artifacts; use "
                    "history only if current evidence is insufficient. Return only JSON: "
                    "{\"routes\":[{\"part_id\":\"part_1\","
                    "\"artifact_ids\":[\"artifact_0\"],\"include_context\":true,"
                    "\"include_log_abstraction\":false}]}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question_parts": parts,
                        "artifact_metadata": metadata,
                        "context_metadata": context_metadata,
                        "log_abstraction_available": bool(
                            snapshot.get("log_abstraction")
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @classmethod
    def _verification_messages(
        cls,
        request: str,
        report: str,
        parts: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        routes: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
        snapshot: dict[str, Any],
        response_language: str,
    ) -> list[dict[str, str]]:
        by_id = {artifact["id"]: artifact for artifact in artifacts}
        evidence_by_part: list[dict[str, Any]] = []
        for route in routes:
            selected = []
            for artifact_id in route["artifact_ids"]:
                artifact = by_id[artifact_id]
                selected.append(
                    {
                        "id": artifact_id,
                        "request_index": artifact.get("request_index"),
                        "description": artifact.get("description"),
                        "type": artifact.get("type"),
                        "content": cls._format_preview(artifact.get("content_preview")),
                    }
                )
            entry: dict[str, Any] = {
                "part_id": route["part_id"],
                "artifacts": selected,
            }
            if route["include_context"]:
                entry["context"] = snapshot.get("context", [])
            if route["include_log_abstraction"]:
                entry["log_abstraction"] = snapshot.get("log_abstraction", "")
            evidence_by_part.append(entry)

        schema = (
            "{\"claim_assessments\":[{\"claim_id\":\"claim_1\","
            "\"status\":\"supported|contradicted|unsupported|non_verifiable\","
            "\"explanation\":\"...\",\"citation\":{\"source_type\":"
            "\"artifact|context|log_abstraction\",\"artifact_id\":\"artifact_0\","
            "\"row\":1,\"column\":\"value\",\"value\":0.5121,"
            "\"claimed_value\":\"51.21%\"}}],\"part_assessments\":[{"
            "\"part_id\":\"part_1\",\"status\":\"answered|partial|unanswered\","
            "\"evidence_contains_answer\":true}],\"summary\":\"...\"}"
        )
        return [
            {
                "role": "system",
                "content": (
                    "Check only whether each Analyst claim is present in the supplied evidence. "
                    "Do not judge whether the artifacts were correctly calculated. Give an exact "
                    "citation. For CSV cells, row is the one-based data row excluding the header, "
                    "column is the exact header, and value is the evidence value. Include claimed_value "
                    "for numeric claims. The QUESTION language has been classified as ISO 639-1 "
                    f"code '{response_language}'. Write every human-readable explanation and the "
                    "summary exclusively in that language. The question's language is authoritative; "
                    "ignore the language used by evidence, prior assistant text, or metadata. "
                    f"Return only JSON with this schema: {schema}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": request,
                        "response": report,
                        "question_parts": parts,
                        "claims": claims,
                        "evidence_by_part": evidence_by_part,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _format_preview(value: Any, limit: int = 12000) -> Any:
        if not isinstance(value, str):
            value = json.dumps(_json_safe(value), ensure_ascii=False)
        if len(value) <= limit:
            return value
        return value[:limit] + "\n[preview truncated]"

    @staticmethod
    def _validate_decomposition(
        payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        parts = payload.get("question_parts")
        claims = payload.get("claims")
        response_language = payload.get("response_language")
        if not isinstance(response_language, str) or not re.fullmatch(
            r"[A-Za-z]{2}", response_language.strip()
        ):
            raise ValueError(
                "EvidenceChecker field 'response_language' must be an ISO 639-1 code."
            )
        if not isinstance(parts, list) or not parts:
            raise ValueError("EvidenceChecker must return non-empty question_parts.")
        if not isinstance(claims, list):
            raise ValueError("EvidenceChecker field 'claims' must be a list.")
        part_ids: set[str] = set()
        for part in parts:
            if not isinstance(part, dict) or not all(
                isinstance(part.get(key), str) and part[key].strip()
                for key in ("id", "text", "data_type")
            ):
                raise ValueError("EvidenceChecker returned an invalid question part.")
            if part["data_type"] not in DATA_TYPES or part["id"] in part_ids:
                raise ValueError("EvidenceChecker returned an invalid question part type or id.")
            part_ids.add(part["id"])
        claim_ids: set[str] = set()
        for claim in claims:
            if not isinstance(claim, dict) or not all(
                isinstance(claim.get(key), str) and claim[key].strip()
                for key in ("id", "text", "verifiability")
            ):
                raise ValueError("EvidenceChecker returned an invalid claim.")
            if claim["id"] in claim_ids or claim["verifiability"] not in {
                "verifiable",
                "non_verifiable",
            }:
                raise ValueError("EvidenceChecker returned an invalid claim id or type.")
            mapped = claim.get("part_ids")
            if not isinstance(mapped, list) or any(item not in part_ids for item in mapped):
                raise ValueError("EvidenceChecker claim references an unknown question part.")
            claim_ids.add(claim["id"])
        return parts, claims, response_language.strip().lower()

    @staticmethod
    def _validate_routing(
        payload: dict[str, Any],
        parts: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        routes = payload.get("routes")
        if not isinstance(routes, list):
            raise ValueError("EvidenceChecker field 'routes' must be a list.")
        part_ids = {part["id"] for part in parts}
        artifact_ids = {artifact["id"] for artifact in artifacts}
        by_part: dict[str, dict[str, Any]] = {}
        for route in routes:
            if not isinstance(route, dict) or route.get("part_id") not in part_ids:
                raise ValueError("EvidenceChecker route references an unknown question part.")
            selected = route.get("artifact_ids", [])
            if not isinstance(selected, list) or any(
                artifact_id not in artifact_ids for artifact_id in selected
            ):
                raise ValueError("EvidenceChecker route references an ineligible artifact.")
            for key in ("include_context", "include_log_abstraction"):
                if not isinstance(route.get(key), bool):
                    raise ValueError(f"EvidenceChecker route field '{key}' must be boolean.")
            by_part[route["part_id"]] = {
                "part_id": route["part_id"],
                "artifact_ids": list(dict.fromkeys(selected)),
                "include_context": route["include_context"],
                "include_log_abstraction": route["include_log_abstraction"],
            }
        return [
            by_part.get(
                part["id"],
                {
                    "part_id": part["id"],
                    "artifact_ids": [],
                    "include_context": False,
                    "include_log_abstraction": False,
                },
            )
            for part in parts
        ]

    @staticmethod
    def _validate_verification(
        payload: dict[str, Any],
        claims: list[dict[str, Any]],
        parts: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
        raw_claims = payload.get("claim_assessments")
        raw_parts = payload.get("part_assessments")
        summary = payload.get("summary")
        if not isinstance(raw_claims, list) or not isinstance(raw_parts, list):
            raise ValueError("EvidenceChecker returned invalid assessment lists.")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("EvidenceChecker field 'summary' must be a non-empty string.")
        claim_ids = {claim["id"] for claim in claims}
        part_ids = {part["id"] for part in parts}
        assessments: dict[str, dict[str, Any]] = {}
        for item in raw_claims:
            if not isinstance(item, dict) or item.get("claim_id") not in claim_ids:
                raise ValueError("EvidenceChecker assessment references an unknown claim.")
            if item.get("status") not in CLAIM_STATUSES:
                raise ValueError("EvidenceChecker returned an invalid claim status.")
            if not isinstance(item.get("explanation"), str):
                raise ValueError("EvidenceChecker claim explanation must be a string.")
            citation = item.get("citation")
            if citation is not None and not isinstance(citation, dict):
                raise ValueError("EvidenceChecker citation must be an object or null.")
            assessments[item["claim_id"]] = item
        part_assessments: dict[str, dict[str, Any]] = {}
        for item in raw_parts:
            if not isinstance(item, dict) or item.get("part_id") not in part_ids:
                raise ValueError("EvidenceChecker assessment references an unknown part.")
            if item.get("status") not in PART_STATUSES or not isinstance(
                item.get("evidence_contains_answer"), bool
            ):
                raise ValueError("EvidenceChecker returned an invalid part assessment.")
            part_assessments[item["part_id"]] = item
        return assessments, part_assessments, summary.strip()

    def _build_result(
        self,
        parts: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        assessments: dict[str, dict[str, Any]],
        part_assessments: dict[str, dict[str, Any]],
        routes: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
        summary: str,
    ) -> EvidenceCheckResult:
        artifact_by_id = {artifact["id"]: artifact for artifact in artifacts}
        route_by_part = {route["part_id"]: route for route in routes}
        checked_claims: list[dict[str, Any]] = []
        numeric_failure = False
        for claim in claims:
            assessment = dict(
                assessments.get(
                    claim["id"],
                    {
                        "claim_id": claim["id"],
                        "status": "unsupported",
                        "explanation": "No assessment was returned for this claim.",
                        "citation": None,
                    },
                )
            )
            if claim["verifiability"] == "non_verifiable":
                assessment["status"] = "non_verifiable"
                assessment["citation"] = None
            citation = assessment.get("citation")
            if assessment["status"] in {"supported", "contradicted"}:
                allowed_artifacts = {
                    artifact_id
                    for part_id in claim.get("part_ids", [])
                    for artifact_id in route_by_part[part_id]["artifact_ids"]
                }
                source_type = (
                    citation.get("source_type") if isinstance(citation, dict) else None
                )
                if source_type == "artifact":
                    artifact = artifact_by_id.get(citation.get("artifact_id"))
                    if artifact is None or citation.get("artifact_id") not in allowed_artifacts:
                        assessment["status"] = "unsupported"
                        assessment["explanation"] = (
                            "The assessment does not cite an artifact selected for this question part."
                        )
                        assessment["citation"] = None
                    else:
                        citation["path"] = artifact["path"]
                        citation["request_index"] = artifact.get("request_index")
                        required_location = ["value"]
                        if artifact.get("type") == "dataframe":
                            required_location.extend(["row", "column"])
                        if any(citation.get(field) is None for field in required_location):
                            assessment["status"] = "unsupported"
                            assessment["explanation"] = (
                                "The cited evidence does not include the required value and table location."
                            )
                elif source_type == "context":
                    if not any(
                        route_by_part[part_id]["include_context"]
                        for part_id in claim.get("part_ids", [])
                    ):
                        assessment["status"] = "unsupported"
                        assessment["citation"] = None
                    elif citation.get("value") is None:
                        assessment["status"] = "unsupported"
                        assessment["explanation"] = (
                            "The context citation does not identify the supporting value or excerpt."
                        )
                elif source_type == "log_abstraction":
                    if not any(
                        route_by_part[part_id]["include_log_abstraction"]
                        for part_id in claim.get("part_ids", [])
                    ):
                        assessment["status"] = "unsupported"
                        assessment["citation"] = None
                    elif citation.get("value") is None:
                        assessment["status"] = "unsupported"
                        assessment["explanation"] = (
                            "The log-abstraction citation does not identify the supporting value or excerpt."
                        )
                else:
                    assessment["status"] = "unsupported"
                    assessment["explanation"] = (
                        "A supported or contradicted claim must include a valid evidence citation."
                    )
                    assessment["citation"] = None
            failed = self._validate_numeric_citation(
                claim, assessment, artifact_by_id
            )
            numeric_failure = numeric_failure or failed
            checked_claims.append(
                {
                    **claim,
                    "status": assessment["status"],
                    "explanation": assessment.get("explanation", ""),
                    "citation": assessment.get("citation"),
                }
            )

        checked_parts: list[dict[str, Any]] = []
        for part in parts:
            assessment = part_assessments.get(
                part["id"],
                {
                    "status": "unanswered",
                    "evidence_contains_answer": False,
                },
            )
            item = {**part, "status": assessment["status"]}
            if assessment["status"] == "unanswered":
                route = route_by_part[part["id"]]
                evidence_was_available = assessment["evidence_contains_answer"] and bool(
                    route["artifact_ids"]
                    or route["include_context"]
                    or route["include_log_abstraction"]
                )
                item["probable_cause"] = (
                    "analyst_omitted_available_evidence"
                    if evidence_was_available
                    else "engineer_did_not_generate_evidence"
                )
            checked_parts.append(item)

        verifiable = [
            claim for claim in checked_claims if claim["status"] != "non_verifiable"
        ]
        supported = sum(claim["status"] == "supported" for claim in verifiable)
        unsupported = sum(claim["status"] == "unsupported" for claim in verifiable)
        contradicted = any(
            claim["status"] == "contradicted" for claim in verifiable
        )
        if (
            verifiable
            and supported == len(verifiable)
            and all(part["status"] == "answered" for part in checked_parts)
            and not numeric_failure
        ):
            confidence = "high"
        elif contradicted or numeric_failure or unsupported > supported:
            confidence = "low"
        else:
            confidence = "medium"
        return EvidenceCheckResult(
            claims=tuple(checked_claims),
            question_parts=tuple(checked_parts),
            confidence=confidence,
            summary=summary,
        )

    @classmethod
    def _validate_numeric_citation(
        cls,
        claim: dict[str, Any],
        assessment: dict[str, Any],
        artifact_by_id: dict[str, dict[str, Any]],
    ) -> bool:
        citation = assessment.get("citation")
        if not isinstance(citation, dict) or citation.get("source_type") != "artifact":
            return False
        artifact = artifact_by_id.get(citation.get("artifact_id"))
        if not artifact or artifact.get("type") != "dataframe":
            return False
        row = citation.get("row")
        column = citation.get("column")
        claimed_value = citation.get("claimed_value")
        if row is None or column is None or claimed_value is None:
            return False
        try:
            row_number = int(row)
            dataframe = pd.read_csv(artifact["path"])
            if row_number < 1 or row_number > len(dataframe) or column not in dataframe:
                raise ValueError("CSV citation is outside the table.")
            actual_value = dataframe.iloc[row_number - 1][column]
            citation["value"] = _json_safe(actual_value)
            citation["path"] = artifact["path"]
            equivalent = cls._numbers_equivalent(
                claimed_value, actual_value, claim.get("text", "")
            )
            citation["numeric_validation"] = (
                "confirmed" if equivalent else "contradicted"
            )
            if equivalent:
                return False
            assessment["status"] = "contradicted"
            assessment["explanation"] = (
                f"The cited CSV cell contains {actual_value!s}, which does not match "
                f"the claimed value {claimed_value!s}."
            )
            return True
        except Exception as exc:
            citation["numeric_validation"] = "failed"
            citation["numeric_validation_error"] = str(exc)
            assessment["status"] = "unsupported"
            assessment["explanation"] = (
                "The numeric claim could not be confirmed at the cited CSV location."
            )
            return True

    @classmethod
    def _numbers_equivalent(
        cls, claimed_value: Any, evidence_value: Any, claim_text: str
    ) -> bool:
        claimed = cls._parse_number(claimed_value)
        evidence = cls._parse_number(evidence_value)
        if claimed is None or evidence is None:
            return False
        claim_number, claim_percent, decimals = claimed
        evidence_number, evidence_percent, _ = evidence
        claim_percent = claim_percent or "%" in claim_text
        claim_candidates = [(claim_number, cls._rounding_tolerance(decimals))]
        evidence_candidates = [evidence_number]
        if claim_percent:
            claim_candidates.append(
                (claim_number / 100.0, cls._rounding_tolerance(decimals) / 100.0)
            )
        if evidence_percent:
            evidence_candidates.append(evidence_number / 100.0)
        if (abs(claim_number) > 1 >= abs(evidence_number)) or (
            abs(evidence_number) > 1 >= abs(claim_number)
        ):
            claim_candidates.append(
                (claim_number / 100.0, cls._rounding_tolerance(decimals) / 100.0)
            )
            evidence_candidates.append(evidence_number / 100.0)
        return any(
            math.isclose(left, right, rel_tol=1e-6, abs_tol=max(1e-9, tolerance))
            for left, tolerance in claim_candidates
            for right in evidence_candidates
        )

    @staticmethod
    def _rounding_tolerance(decimals: int) -> float:
        return 0.5 * (10 ** (-decimals))

    @staticmethod
    def _parse_number(value: Any) -> tuple[float, bool, int] | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Real) and not pd.isna(value):
            text = str(value)
        elif isinstance(value, str):
            text = value.strip()
        else:
            return None
        is_percent = "%" in text
        cleaned = re.sub(r"[^0-9,\.\-+eE]", "", text)
        if not cleaned:
            return None
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            cleaned = cleaned.replace(",", ".")
        try:
            number = float(cleaned)
        except ValueError:
            return None
        mantissa = cleaned.lower().split("e", 1)[0]
        decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        return number, is_percent, decimals

    @staticmethod
    def _report_as_text(report: StructuredReport | str) -> str:
        if isinstance(report, str):
            return report
        return "\n\n".join(
            str(block.get("content", ""))
            for block in report
            if block.get("type") == "text"
        ).strip()

    @staticmethod
    def _parse_json(response: str) -> dict[str, Any]:
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
            raise ValueError("EvidenceChecker returned an invalid JSON response.") from exc
        if not isinstance(payload, dict):
            raise ValueError("EvidenceChecker JSON response must be an object.")
        return payload

    def _notify(
        self, callback: ProgressCallback | None, event: str, **details: Any
    ) -> None:
        if callback is not None:
            callback({"event": event, "agent": self.name, **details})


def _resolve_manifest_path(session_dir: Path, entry: dict[str, Any]) -> Path:
    original = Path(entry["file_path"])
    if original.exists():
        return original
    return session_dir / str(entry.get("category", "")) / original.name


def _load_manifest(session_dir: Path) -> list[dict[str, Any]]:
    manifest = session_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Session manifest not found: {manifest}")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def load_saved_session(
    session_dir: str | os.PathLike[str], request_index: int | None = None
) -> dict[str, Any]:
    """Load a saved session into the minimal state required by EvidenceChecker."""
    root = Path(session_dir).resolve()
    entries = _load_manifest(root)
    requests: dict[int, Path] = {}
    reports: dict[int, Path] = {}
    snapshots: dict[int, Path] = {}
    for entry in entries:
        path = _resolve_manifest_path(root, entry)
        name = path.name
        report_match = re.search(r"assistant_report_request_(\d+)", name)
        snapshot_match = re.search(r"analyst_evidence_request_(\d+)", name)
        if entry.get("type") == "user_request":
            # The UI historically numbered request files by chat-message count
            # (1, 3, 5, ...), so session order is the reliable request index.
            requests[len(requests) + 1] = path
        if report_match:
            reports[int(report_match.group(1))] = path
        if snapshot_match:
            snapshots[int(snapshot_match.group(1))] = path
    complete = sorted(set(requests) & set(reports))
    if not complete:
        raise ValueError("The session has no complete request/report pair.")
    selected = complete[-1] if request_index is None else request_index
    if selected not in complete:
        raise ValueError(f"Request {selected} is not complete in this session.")

    if selected in snapshots and snapshots[selected].exists():
        snapshot = json.loads(snapshots[selected].read_text(encoding="utf-8"))
        for artifact in snapshot.get("artifacts", []):
            path = Path(artifact["path"])
            if not path.exists():
                category = "dataframes" if artifact["type"] == "dataframe" else "visualizations"
                artifact["path"] = str(root / category / path.name)
    else:
        snapshot = _reconstruct_legacy_snapshot(root, entries, requests, reports, selected)

    return {
        "user_request": [
            requests[index].read_text(encoding="utf-8")
            for index in complete
            if index <= selected
        ],
        "final_report": json.loads(reports[selected].read_text(encoding="utf-8")),
        "artifact_session_dir": str(root),
        "evidence_snapshots": [snapshot],
        "evidence_check": None,
    }


def _reconstruct_legacy_snapshot(
    root: Path,
    entries: list[dict[str, Any]],
    requests: dict[int, Path],
    reports: dict[int, Path],
    selected: int,
) -> dict[str, Any]:
    report_positions: dict[int, int] = {}
    for position, entry in enumerate(entries):
        match = re.search(r"assistant_report_request_(\d+)", Path(entry["file_path"]).name)
        if match:
            report_positions[int(match.group(1))] = position
    artifacts: list[dict[str, Any]] = []
    previous_report_position = -1
    request_owner = 1
    artifact_index = 0
    for position, entry in enumerate(entries):
        while (
            request_owner in report_positions
            and position > report_positions[request_owner]
        ):
            previous_report_position = report_positions[request_owner]
            request_owner += 1
        if entry.get("type") not in {"dataframe", "visualization"}:
            continue
        if request_owner > selected:
            break
        path = _resolve_manifest_path(root, entry)
        preview = entry.get("data_preview")
        if entry.get("type") == "dataframe" and path.exists():
            try:
                preview = transform_dataframe_for_llms(pd.read_csv(path))
            except Exception:
                preview = preview or "The dataframe could not be previewed."
        artifacts.append(
            {
                "id": f"artifact_{artifact_index}",
                "path": str(path),
                "description": entry.get("description", ""),
                "type": entry["type"],
                "request_index": request_owner,
                "scope": "current" if request_owner == selected else "history",
                "eligible": _legacy_artifact_is_from_successful_attempt(
                    entries,
                    position,
                    request_owner,
                    previous_report_position,
                    report_positions.get(request_owner, len(entries)),
                ),
                "content_preview": preview,
            }
        )
        artifact_index += 1
    return {
        "schema_version": 1,
        "request_index": selected,
        "user_request": requests[selected].read_text(encoding="utf-8"),
        "artifacts": artifacts,
        "context": [],
        "log_abstraction": "",
        "legacy_reconstruction": True,
    }


def _legacy_artifact_is_from_successful_attempt(
    entries: list[dict[str, Any]],
    artifact_position: int,
    request_index: int,
    lower_bound: int,
    upper_bound: int,
) -> bool:
    engineer_code_position = None
    for position in range(lower_bound + 1, upper_bound + 1):
        if position >= len(entries):
            break
        entry = entries[position]
        extra = entry.get("extra", {})
        if (
            entry.get("type") == "generated_code"
            and extra.get("role") == "engineer"
            and extra.get("request_index") == request_index
        ):
            engineer_code_position = position
            break
    if engineer_code_position is None:
        return False
    last_response = max(
        (
            position
            for position in range(lower_bound + 1, engineer_code_position)
            if entries[position].get("type") == "llm_response"
        ),
        default=None,
    )
    if last_response is None:
        return False
    return last_response < artifact_position < engineer_code_position


def evaluate_saved_session(
    session_dir: str | os.PathLike[str],
    credentials: LLMConnection,
    request_index: int | None = None,
    *,
    response_generator: Callable = query_llm,
    progress_callback: ProgressCallback | None = None,
) -> EvidenceCheckResult:
    from promoai.agents.persistence import AgentRunRecorder

    state = load_saved_session(session_dir, request_index)
    result = EvidenceCheckerAgent(
        credentials, response_generator=response_generator
    ).execute(state, progress_callback)
    state["evidence_check"] = result.as_dict()
    AgentRunRecorder().record_evidence_check(state, result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EvidenceChecker on a saved PMAx session."
    )
    parser.add_argument("session_dir")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-index", type=int)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    api_key = os.environ.get("PMAX_API_KEY")
    if not api_key:
        parser.error("Set PMAX_API_KEY before running the saved-session evaluator.")
    credentials = LLMConnection(
        api_key,
        args.model,
        args.provider,
        {"END_POINT": args.endpoint} if args.endpoint else {},
    )
    result = evaluate_saved_session(
        args.session_dir, credentials, request_index=args.request_index
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
