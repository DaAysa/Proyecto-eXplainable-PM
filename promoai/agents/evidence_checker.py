from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from promoai.agents.contracts import (
    EvidenceCheckResult,
    EvidenceClaim,
    EvidenceLocation,
    ProgressCallback,
    QuestionPartAssessment,
    StructuredReport,
)
from promoai.agents.state import ProcessState
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
CLAIM_STATUSES = {"supported", "contradicted", "unsupported"}
PART_STATUSES = {"answered", "partial", "unanswered"}
_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?\s*%?"
    r"|(?<![\w])[-+]?\d+(?:[.,]\d+)?\s*%?"
)
_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


class EvidenceCheckerAgent:
    """Checks whether Analyst claims are grounded in the evidence it received.

    The model selects and interprets evidence. Numeric values located in CSV
    tables are then confirmed deterministically against the complete local file.
    The agent deliberately does not validate how an artifact was calculated.
    """

    name = "EvidenceChecker"

    def __init__(
        self,
        credentials: LLMConnection,
        response_generator: Callable = query_llm,
    ):
        self._credentials = credentials
        self._response_generator = response_generator
        self._call_index = 0

    def execute(
        self,
        state: ProcessState,
        progress_callback: ProgressCallback | None = None,
    ) -> EvidenceCheckResult:
        history = list(state.get("analyst_evidence_history", []))
        if not history:
            raise ValueError("No Analyst evidence snapshot is available for this request.")

        request = (
            state.latest_request
            if hasattr(state, "latest_request")
            else state["user_request"][-1]
        )
        report = state["final_report"]
        sources = self._collect_sources(history)
        plan = self._plan(request, report, sources, state, progress_callback)

        claim_results: dict[str, EvidenceClaim] = {}
        part_results: list[QuestionPartAssessment] = []
        claims_by_id = {item["id"]: item for item in plan["claims"]}

        for part_position, part in enumerate(plan["question_parts"], start=1):
            part_claims = [
                claim
                for claim in plan["claims"]
                if part["id"] in claim["question_part_ids"]
            ]
            selected = [
                sources[source_id]
                for source_id in part["candidate_source_ids"]
                if source_id in sources
            ]
            response = self._check_part(
                request,
                part,
                part_claims,
                selected,
                state,
                progress_callback,
                part_position,
                len(plan["question_parts"]),
            )
            part_status = response["status"]
            cause = None
            if part_status == "unanswered":
                cause = (
                    "analyst_missed_available_evidence"
                    if selected
                    else "engineer_did_not_generate_evidence"
                )
            part_results.append(
                QuestionPartAssessment(
                    id=part["id"],
                    text=part["text"],
                    data_type=part["data_type"],
                    status=part_status,
                    unanswered_cause=cause,
                )
            )
            for checked_claim in response["claims"]:
                definition = claims_by_id[checked_claim["claim_id"]]
                result = EvidenceClaim(
                    id=definition["id"],
                    text=definition["text"],
                    question_part_ids=tuple(definition["question_part_ids"]),
                    status=checked_claim["status"],
                    explanation=checked_claim["explanation"],
                    evidence=tuple(checked_claim["evidence"]),
                )
                existing = claim_results.get(result.id)
                claim_results[result.id] = self._merge_claim(existing, result)

        for definition in plan["claims"]:
            if definition["verifiability"] == "non_verifiable":
                claim_results[definition["id"]] = EvidenceClaim(
                    id=definition["id"],
                    text=definition["text"],
                    question_part_ids=tuple(definition["question_part_ids"]),
                    status="non_verifiable",
                    explanation=(
                        "This is an opinion or recommendation rather than a factual "
                        "claim that can be checked against the supplied evidence."
                    ),
                )
            elif definition["id"] not in claim_results:
                claim_results[definition["id"]] = EvidenceClaim(
                    id=definition["id"],
                    text=definition["text"],
                    question_part_ids=tuple(definition["question_part_ids"]),
                    status="unsupported",
                    explanation="No selected evidence was linked to this claim.",
                )

        ordered_claims = tuple(
            self._verify_numeric_locations(claim_results[item["id"]], sources)
            for item in plan["claims"]
        )
        confidence, explanation = self._confidence(ordered_claims, part_results)
        return EvidenceCheckResult(
            claims=ordered_claims,
            question_parts=tuple(part_results),
            confidence=confidence,
            confidence_explanation=explanation,
        )

    def _plan(
        self,
        request: str,
        report: StructuredReport | str,
        sources: dict[str, dict[str, Any]],
        state: ProcessState,
        callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        source_index = [
            {
                "source_id": source_id,
                "source": source["source"],
                "artifact_id": source.get("artifact_id"),
                "description": source.get("description"),
                "type": source.get("type"),
                "current_request": source.get("current_request", False),
                "short_preview": str(source.get("preview", ""))[:240]
                if source["source"] != "artifact"
                else None,
            }
            for source_id, source in sources.items()
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You plan an evidence-grounding check for a process-mining report. "
                    "Split the request into independently answerable parts and assign "
                    "each exactly one data_type from time, frequency, structure, "
                    "conformance, causal, prediction, data_quality, or null for a "
                    "non-factual request. Split the report into atomic claims and map "
                    "them to request parts. Opinions and recommendations are "
                    "non_verifiable. Select only source IDs capable of answering each "
                    "part, preferring current_request sources but allowing earlier "
                    "sources. Do not judge truth yet. Return only JSON: "
                    '{"question_parts":[{"id":"part_1","text":"...",'
                    '"data_type":"frequency","candidate_source_ids":["artifact_0"]}],'
                    '"claims":[{"id":"claim_1","text":"...",'
                    '"question_part_ids":["part_1"],'
                    '"verifiability":"verifiable"}]}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{request}\n\n"
                    f"ANALYST REPORT:\n{self._report_as_text(report)}\n\n"
                    f"EVIDENCE INDEX:\n{json.dumps(source_index, ensure_ascii=False)}"
                ),
            },
        ]
        response = self._invoke(messages, state, callback, "planning", 1, 1)
        return self._parse_plan(response, set(sources))

    def _check_part(
        self,
        request: str,
        part: dict[str, Any],
        claims: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        state: ProcessState,
        callback: ProgressCallback | None,
        attempt: int,
        total: int,
    ) -> dict[str, Any]:
        source_payload = [self._render_source(source, claims) for source in sources]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence checker. Check only the supplied request part "
                    "and only the supplied evidence excerpts. Do not validate how an "
                    "artifact was calculated. For each factual claim return supported, "
                    "contradicted, or unsupported. Give exact source_id and, for table "
                    "evidence, the 1-based data row (header excluded), exact column, "
                    "observed value, and the corresponding claim_value. Assess whether "
                    "the response part is answered, partial, or unanswered. Return only "
                    "results for claims marked verifiable; non_verifiable claims may "
                    "inform coverage but must be omitted from the claims result. Return only "
                    "JSON: {\"part_id\":\"part_1\",\"status\":\"answered\","
                    "\"claims\":[{\"claim_id\":\"claim_1\","
                    "\"status\":\"supported\",\"explanation\":\"...\","
                    "\"evidence\":[{\"source_id\":\"artifact_0\","
                    "\"row\":1,\"column\":\"metric\",\"value\":51.21,"
                    "\"claim_value\":\"51.21%\"}]}]}. Use the same language "
                    "as the user for explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"FULL REQUEST:\n{request}\n\n"
                    f"REQUEST PART:\n{json.dumps(part, ensure_ascii=False)}\n\n"
                    f"CLAIMS:\n{json.dumps(claims, ensure_ascii=False)}\n\n"
                    f"SELECTED EVIDENCE ONLY:\n{json.dumps(source_payload, ensure_ascii=False, default=str)}"
                ),
            },
        ]
        response = self._invoke(
            messages, state, callback, "part_check", attempt, total
        )
        return self._parse_part_response(
            response,
            part["id"],
            {claim["id"] for claim in claims if claim["verifiability"] == "verifiable"},
            {source["source_id"] for source in sources},
            {source["source_id"]: source for source in sources},
        )

    def _invoke(
        self,
        messages: list[dict[str, str]],
        state: ProcessState,
        callback: ProgressCallback | None,
        phase: str,
        attempt: int,
        total: int,
    ) -> str:
        self._call_index += 1
        args = dict(self._credentials.args or {})
        args["artifact_session_dir"] = state["artifact_session_dir"]
        args["temperature"] = 0
        self._notify(
            callback,
            "request_started",
            attempt=attempt,
            total_attempts=total,
            provider=self._credentials.ai_provider,
            model=self._credentials.llm_name,
            phase=phase,
        )
        try:
            response = self._response_generator(
                messages,
                api_key=self._credentials.api_key,
                llm_name=self._credentials.llm_name,
                ai_provider=self._credentials.ai_provider,
                llm_args=args,
            )
        except Exception as exc:
            self._notify(
                callback,
                "request_failed",
                attempt=attempt,
                error=str(exc),
                phase=phase,
            )
            raise
        self._notify(callback, "response_received", attempt=attempt, phase=phase)
        self._notify(callback, "attempt_completed", attempt=attempt, phase=phase)
        return response

    @staticmethod
    def _parse_plan(response: str, source_ids: set[str]) -> dict[str, Any]:
        payload = _json_payload(response, "EvidenceChecker planning")
        parts = payload.get("question_parts")
        claims = payload.get("claims")
        if not isinstance(parts, list) or not parts:
            raise ValueError("EvidenceChecker planning requires question_parts.")
        if not isinstance(claims, list):
            raise ValueError("EvidenceChecker planning requires a claims list.")

        part_ids: set[str] = set()
        normalized_parts = []
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("Every question part must be an object.")
            part_id = _nonempty(part.get("id"), "question part id")
            if part_id in part_ids:
                raise ValueError(f"Duplicate question part id: {part_id}")
            part_ids.add(part_id)
            data_type = part.get("data_type")
            if data_type not in DATA_TYPES and data_type is not None:
                raise ValueError(f"Invalid question-part data_type: {data_type}")
            candidates = part.get("candidate_source_ids", [])
            if not isinstance(candidates, list) or not all(
                isinstance(item, str) and item in source_ids for item in candidates
            ):
                raise ValueError("Question part contains an unknown evidence source ID.")
            normalized_parts.append(
                {
                    "id": part_id,
                    "text": _nonempty(part.get("text"), "question part text"),
                    "data_type": data_type,
                    "candidate_source_ids": list(dict.fromkeys(candidates)),
                }
            )

        claim_ids: set[str] = set()
        normalized_claims = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("Every claim must be an object.")
            claim_id = _nonempty(claim.get("id"), "claim id")
            if claim_id in claim_ids:
                raise ValueError(f"Duplicate claim id: {claim_id}")
            claim_ids.add(claim_id)
            mapped_parts = claim.get("question_part_ids", [])
            if not isinstance(mapped_parts, list) or not all(
                isinstance(item, str) and item in part_ids for item in mapped_parts
            ):
                raise ValueError("Claim contains an unknown question part ID.")
            verifiability = claim.get("verifiability")
            if verifiability not in {"verifiable", "non_verifiable"}:
                raise ValueError("Claim verifiability must be verifiable or non_verifiable.")
            normalized_claims.append(
                {
                    "id": claim_id,
                    "text": _nonempty(claim.get("text"), "claim text"),
                    "question_part_ids": list(dict.fromkeys(mapped_parts)),
                    "verifiability": verifiability,
                }
            )
        return {"question_parts": normalized_parts, "claims": normalized_claims}

    @staticmethod
    def _parse_part_response(
        response: str,
        expected_part_id: str,
        expected_claim_ids: set[str],
        selected_source_ids: set[str],
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        payload = _json_payload(response, "EvidenceChecker part check")
        if payload.get("part_id") != expected_part_id:
            raise ValueError("EvidenceChecker returned an unexpected question part ID.")
        status = payload.get("status")
        if status not in PART_STATUSES:
            raise ValueError("EvidenceChecker returned an invalid question-part status.")
        raw_claims = payload.get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("EvidenceChecker part claims must be a list.")
        normalized = []
        seen: set[str] = set()
        for claim in raw_claims:
            if not isinstance(claim, dict):
                raise ValueError("EvidenceChecker claim result must be an object.")
            claim_id = claim.get("claim_id")
            if claim_id not in expected_claim_ids or claim_id in seen:
                raise ValueError("EvidenceChecker returned an unknown or duplicate claim ID.")
            seen.add(claim_id)
            claim_status = claim.get("status")
            if claim_status not in CLAIM_STATUSES:
                raise ValueError("EvidenceChecker returned an invalid claim status.")
            raw_evidence = claim.get("evidence", [])
            if not isinstance(raw_evidence, list):
                raise ValueError("Claim evidence must be a list.")
            locations = []
            for location in raw_evidence:
                if not isinstance(location, dict):
                    raise ValueError("Evidence location must be an object.")
                source_id = location.get("source_id")
                if source_id not in selected_source_ids:
                    raise ValueError("EvidenceChecker referenced an unselected source ID.")
                source = sources[source_id]
                row = location.get("row")
                if row is not None and (not isinstance(row, int) or row < 1):
                    raise ValueError("Evidence table rows must be positive integers.")
                column = location.get("column")
                if column is not None and not isinstance(column, str):
                    raise ValueError("Evidence column must be a string or null.")
                locations.append(
                    EvidenceLocation(
                        source=source["source"],
                        source_id=source_id,
                        artifact_id=source.get("artifact_id"),
                        path=source.get("path"),
                        row=row,
                        column=column,
                        value=location.get("value"),
                        claim_value=location.get("claim_value"),
                    )
                )
            normalized.append(
                {
                    "claim_id": claim_id,
                    "status": claim_status,
                    "explanation": _nonempty(
                        claim.get("explanation"), "claim explanation"
                    ),
                    "evidence": locations,
                }
            )
        return {"status": status, "claims": normalized}

    @staticmethod
    def _collect_sources(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        sources: dict[str, dict[str, Any]] = {}
        latest = history[-1]
        for artifact in latest.get("artifacts", []):
            if not artifact.get("evidence_eligible", True):
                continue
            source_id = artifact["id"]
            sources[source_id] = {
                "source_id": source_id,
                "source": "artifact",
                "artifact_id": artifact["id"],
                "path": artifact.get("path"),
                "description": artifact.get("description", ""),
                "type": artifact.get("type", ""),
                "preview": artifact.get("preview", ""),
                "current_request": bool(artifact.get("sent_this_request", False)),
            }
        for snapshot in history:
            request_index = snapshot.get("request_index", 0)
            is_current = snapshot is latest
            for index, entry in enumerate(snapshot.get("context") or []):
                source_id = f"context_{request_index}_{index}"
                sources[source_id] = {
                    "source_id": source_id,
                    "source": "context",
                    "description": f"Analyst context for request {request_index}",
                    "preview": entry,
                    "current_request": is_current,
                }
            abstraction = snapshot.get("log_abstraction")
            if abstraction:
                source_id = f"log_abstraction_{request_index}"
                sources[source_id] = {
                    "source_id": source_id,
                    "source": "log_abstraction",
                    "description": f"Event log abstraction for request {request_index}",
                    "preview": abstraction,
                    "current_request": is_current,
                }
        return sources

    @staticmethod
    def _render_source(
        source: dict[str, Any], claims: list[dict[str, Any]]
    ) -> dict[str, Any]:
        rendered = {
            "source_id": source["source_id"],
            "source": source["source"],
            "description": source.get("description"),
            "current_request": source.get("current_request", False),
        }
        if source["source"] != "artifact":
            rendered["content"] = source.get("preview", "")
            return rendered
        rendered["artifact_id"] = source.get("artifact_id")
        rendered["path"] = source.get("path")
        rendered["preview"] = source.get("preview", "")
        path = source.get("path")
        if source.get("type") == "dataframe" and path and Path(path).exists():
            rendered["targeted_rows"] = _target_csv_rows(
                path, " ".join(claim["text"] for claim in claims)
            )
        return rendered

    def _verify_numeric_locations(
        self, claim: EvidenceClaim, sources: dict[str, dict[str, Any]]
    ) -> EvidenceClaim:
        if claim.status == "non_verifiable":
            return claim
        claim_numbers = _number_tokens(claim.text)
        new_locations: list[EvidenceLocation] = []
        forced_status: str | None = None
        reasons: list[str] = []
        for location in claim.evidence:
            source = sources.get(location.source_id, {})
            is_csv = (
                source.get("source") == "artifact"
                and source.get("type") == "dataframe"
                and str(source.get("path", "")).lower().endswith(".csv")
            )
            expected_token = (
                str(location.claim_value)
                if location.claim_value is not None
                else (claim_numbers[0] if len(claim_numbers) == 1 else None)
            )
            if not is_csv or expected_token is None:
                new_locations.append(location)
                continue
            if location.row is None or not location.column:
                check = {"passed": False, "reason": "missing_table_location"}
                new_locations.append(replace(location, numeric_check=check))
                forced_status = forced_status or "unsupported"
                reasons.append("numeric table evidence has no exact row and column")
                continue
            path = source.get("path")
            try:
                dataframe = pd.read_csv(path)
                if location.column not in dataframe.columns:
                    raise KeyError(f"column {location.column!r} does not exist")
                if location.row < 1 or location.row > len(dataframe):
                    raise IndexError("row is outside the CSV")
                actual = dataframe.iloc[location.row - 1][location.column]
                if hasattr(actual, "item"):
                    actual = actual.item()
                percent_semantics = (
                    "%" in expected_token
                    or any(
                        word in location.column.casefold()
                        for word in ("percent", "rate", "ratio", "proportion")
                    )
                )
                passed, check = _compare_numeric(
                    expected_token, actual, percent_semantics=percent_semantics
                )
                new_locations.append(
                    replace(location, value=actual, numeric_check=check)
                )
                if not passed:
                    forced_status = "contradicted"
                    reasons.append(
                        f"CSV row {location.row}, column {location.column} contains {actual!r}"
                    )
            except Exception as exc:
                check = {"passed": False, "reason": str(exc)}
                new_locations.append(replace(location, numeric_check=check))
                forced_status = forced_status or "unsupported"
                reasons.append(f"the CSV location could not be confirmed: {exc}")

        if forced_status is None:
            return replace(claim, evidence=tuple(new_locations))
        explanation = claim.explanation
        if reasons:
            explanation = f"{explanation} Numeric confirmation: {'; '.join(reasons)}."
        return replace(
            claim,
            status=forced_status,  # type: ignore[arg-type]
            explanation=explanation,
            evidence=tuple(new_locations),
        )

    @staticmethod
    def _merge_claim(
        existing: EvidenceClaim | None, incoming: EvidenceClaim
    ) -> EvidenceClaim:
        if existing is None:
            return incoming
        priority = {"supported": 0, "unsupported": 1, "contradicted": 2}
        status = (
            incoming.status
            if priority.get(incoming.status, 0) > priority.get(existing.status, 0)
            else existing.status
        )
        evidence = tuple(
            {
                (
                    item.source_id,
                    item.row,
                    item.column,
                    str(item.value),
                ): item
                for item in (*existing.evidence, *incoming.evidence)
            }.values()
        )
        return replace(
            existing,
            status=status,  # type: ignore[arg-type]
            explanation=f"{existing.explanation} {incoming.explanation}".strip(),
            evidence=evidence,
        )

    @staticmethod
    def _confidence(
        claims: tuple[EvidenceClaim, ...],
        parts: list[QuestionPartAssessment],
    ) -> tuple[str, str]:
        verifiable = [claim for claim in claims if claim.status != "non_verifiable"]
        if any(claim.status == "contradicted" for claim in verifiable) or any(
            part.status == "unanswered" for part in parts
        ):
            return (
                "low",
                "At least one claim is contradicted or one request part is unanswered.",
            )
        if (
            not verifiable
            or any(claim.status == "unsupported" for claim in verifiable)
            or any(part.status == "partial" for part in parts)
        ):
            return (
                "medium",
                "No contradiction was found, but support or request coverage is incomplete.",
            )
        return (
            "high",
            "Every verifiable claim is supported and every request part is answered.",
        )

    @staticmethod
    def _report_as_text(report: StructuredReport | str) -> str:
        if isinstance(report, str):
            return report
        parts = []
        for block in report:
            if block.get("type") == "text":
                parts.append(str(block.get("content", "")))
            elif block.get("type") == "artifact":
                parts.append("[A supporting artifact is displayed here.]")
        return "\n\n".join(parts).strip()

    def _notify(
        self,
        callback: ProgressCallback | None,
        event: str,
        **details: Any,
    ) -> None:
        if callback is not None:
            callback({"event": event, "agent": self.name, **details})


def _json_payload(response: str, label: str) -> dict[str, Any]:
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
        raise ValueError(f"{label} returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must return a JSON object.")
    return payload


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"EvidenceChecker {label} must be a non-empty string.")
    return value.strip()


def _target_csv_rows(path: str, query: str, limit: int = 20) -> str:
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        return "The CSV is empty."
    if len(dataframe) < 50:
        display = dataframe.copy()
        display.insert(0, "__row__", range(1, len(display) + 1))
        return display.to_markdown(index=False)
    words = {word.casefold() for word in _WORD_RE.findall(query)}
    numbers = _number_tokens(query)
    scored: list[tuple[int, int]] = []
    for index, row in dataframe.iterrows():
        text = " ".join(str(value) for value in row.tolist()).casefold()
        score = sum(1 for word in words if word in text)
        for token in numbers:
            parsed = _parse_number(token)
            if parsed is None:
                continue
            for value in row.tolist():
                try:
                    if math.isclose(float(value), parsed[0], rel_tol=0, abs_tol=parsed[1]):
                        score += 3
                        break
                except (TypeError, ValueError):
                    continue
        if score:
            scored.append((score, index))
    selected = [index for _, index in sorted(scored, reverse=True)[:limit]]
    if not selected:
        return "No targeted rows matched the claim terms; use the artifact preview only."
    display = dataframe.loc[sorted(selected)].copy()
    display.insert(0, "__row__", [index + 1 for index in sorted(selected)])
    return display.to_markdown(index=False)


def _number_tokens(text: str) -> list[str]:
    return [match.group(0).strip() for match in _NUMBER_RE.finditer(text or "")]


def _parse_number(value: Any) -> tuple[float, float] | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    text = text.rstrip("%")
    if not text:
        return None
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        text = text.replace(thousands, "").replace(decimal, ".")
    elif "," in text:
        before, after = text.rsplit(",", 1)
        if len(after) == 3 and before.lstrip("+-") not in {"0", ""}:
            text = before + after
        else:
            text = before + "." + after
    try:
        number = float(text)
    except ValueError:
        return None
    decimal_match = re.search(r"[.,](\d+)", str(value).rstrip("%").strip())
    decimals = len(decimal_match.group(1)) if decimal_match else 0
    tolerance = max(1e-9, 0.5 * (10 ** (-decimals)))
    return number, tolerance


def _compare_numeric(
    claim_value: Any, actual_value: Any, *, percent_semantics: bool
) -> tuple[bool, dict[str, Any]]:
    expected = _parse_number(claim_value)
    actual = _parse_number(actual_value)
    if expected is None or actual is None:
        return False, {
            "passed": False,
            "reason": "claim or evidence value is not numeric",
            "claim_value": claim_value,
            "evidence_value": actual_value,
        }
    expected_variants = [expected]
    actual_variants = [actual]
    if percent_semantics:
        expected_variants.extend(
            [(expected[0] / 100, expected[1] / 100), (expected[0] * 100, expected[1] * 100)]
        )
        actual_variants.extend(
            [(actual[0] / 100, actual[1] / 100), (actual[0] * 100, actual[1] * 100)]
        )
    for expected_number, expected_tolerance in expected_variants:
        for actual_number, actual_tolerance in actual_variants:
            tolerance = max(expected_tolerance, actual_tolerance, 1e-9)
            if math.isclose(
                expected_number, actual_number, rel_tol=0, abs_tol=tolerance
            ):
                return True, {
                    "passed": True,
                    "claim_value": claim_value,
                    "evidence_value": actual_value,
                    "normalized_claim_value": expected_number,
                    "normalized_evidence_value": actual_number,
                    "tolerance": tolerance,
                }
    return False, {
        "passed": False,
        "claim_value": claim_value,
        "evidence_value": actual_value,
        "normalized_claim_value": expected[0],
        "normalized_evidence_value": actual[0],
        "tolerance": expected[1],
    }

