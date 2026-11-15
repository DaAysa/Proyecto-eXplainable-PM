from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from promoai.agents.contracts import EvidenceCheckResult, ProgressCallback
from promoai.agents.evidence_checker import EvidenceCheckerAgent
from promoai.agents.persistence import AgentRunRecorder
from promoai.general_utils.llm_connection import LLMConnection, query_llm


def check_saved_session(
    session_dir: str,
    credentials: LLMConnection,
    request_index: int | None = None,
    *,
    persist: bool = True,
    progress_callback: ProgressCallback | None = None,
    response_generator: Callable = query_llm,
) -> dict[int, EvidenceCheckResult]:
    """Run EvidenceChecker for one or every request in a persisted PMAx session."""

    session_path = _find_session_dir(session_dir)
    states = load_saved_session_states(str(session_path))
    if request_index is not None:
        if request_index not in states:
            raise ValueError(f"Request {request_index} does not exist in this session.")
        states = {request_index: states[request_index]}

    checker = EvidenceCheckerAgent(
        credentials, response_generator=response_generator
    )
    recorder = AgentRunRecorder()
    results: dict[int, EvidenceCheckResult] = {}
    for index, state in states.items():
        result = checker.execute(state, progress_callback)
        payload = result.as_dict()
        state["evidence_check"] = payload
        state.setdefault("evidence_checks", {})[index] = payload
        if persist:
            recorder.record_evidence_check(state, result)
        results[index] = result
    return results


def load_saved_session_states(session_dir: str) -> dict[int, dict[str, Any]]:
    """Reconstruct request-scoped EvidenceChecker states from a session manifest."""

    session_path = _find_session_dir(session_dir)
    entries = _read_manifest(session_path)
    indexed = list(enumerate(entries))
    request_entries = [
        (position, entry)
        for position, entry in indexed
        if entry.get("type") == "user_request"
    ]
    report_entries = [
        (position, entry) for position, entry in indexed if entry.get("type") == "report"
    ]
    if not request_entries or not report_entries:
        raise ValueError("The saved session has no request/report pairs to check.")

    requests: dict[int, dict[str, Any]] = {}
    for ordinal, (position, entry) in enumerate(request_entries, start=1):
        index = _entry_request_index(entry, ordinal)
        path = _resolve_entry_path(session_path, entry)
        requests[index] = {
            "position": position,
            "text": path.read_text(encoding="utf-8").strip(),
        }

    reports: dict[int, dict[str, Any]] = {}
    for ordinal, (position, entry) in enumerate(report_entries, start=1):
        index = _entry_request_index(entry, ordinal)
        path = _resolve_entry_path(session_path, entry)
        reports[index] = {
            "position": position,
            "report": json.loads(path.read_text(encoding="utf-8")),
        }

    exact_snapshots: dict[int, dict[str, Any]] = {}
    for ordinal, (_, entry) in enumerate(
        (
            item
            for item in indexed
            if item[1].get("type") == "analyst_evidence"
        ),
        start=1,
    ):
        index = _entry_request_index(entry, ordinal)
        path = _resolve_entry_path(session_path, entry)
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _rebase_snapshot_paths(session_path, snapshot)
        exact_snapshots[index] = snapshot

    legacy_snapshots = _legacy_snapshots(session_path, entries, requests, reports)
    ordered_indices = sorted(set(requests) & set(reports))
    histories: list[dict[str, Any]] = []
    states: dict[int, dict[str, Any]] = {}
    request_texts: list[str] = []
    for index in ordered_indices:
        snapshot = exact_snapshots.get(index, legacy_snapshots[index])
        histories.append(snapshot)
        request_texts.append(requests[index]["text"])
        states[index] = {
            "user_request": list(request_texts),
            "final_report": reports[index]["report"],
            "artifact_session_dir": str(session_path),
            "analyst_evidence_history": list(histories),
            "active_request_index": index,
            "evidence_check": None,
            "evidence_checks": {},
        }
    return states


def _legacy_snapshots(
    session_path: Path,
    entries: list[dict[str, Any]],
    requests: dict[int, dict[str, Any]],
    reports: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    artifacts = [
        {
            "position": position,
            "entry": entry,
            "id": f"artifact_{artifact_index}",
        }
        for artifact_index, (position, entry) in enumerate(
            (item for item in enumerate(entries) if item[1].get("type") in {"dataframe", "visualization"})
        )
    ]
    eligibility: dict[int, bool] = {}
    first_seen: dict[int, int] = {}
    ordered_indices = sorted(set(requests) & set(reports))
    for order, index in enumerate(ordered_indices):
        start = requests[index]["position"]
        end = reports[index]["position"]
        engineer_code_positions = [
            position
            for position, entry in enumerate(entries)
            if start < position < end
            and entry.get("type") == "generated_code"
            and (entry.get("extra") or {}).get("role") == "engineer"
        ]
        engineer_end = engineer_code_positions[0] if engineer_code_positions else end
        responses = [
            position
            for position, entry in enumerate(entries)
            if start < position < engineer_end
            and entry.get("type") in {"llm_response", "llm_response_text"}
        ]
        successful_response = responses[-1] if responses else start
        current_artifacts = [
            artifact
            for artifact in artifacts
            if start < artifact["position"] < engineer_end
        ]
        for artifact in current_artifacts:
            first_seen[artifact["position"]] = index
            eligibility[artifact["position"]] = (
                artifact["position"] > successful_response
                if responses and engineer_code_positions
                else True
            )

    snapshots: dict[int, dict[str, Any]] = {}
    for index in ordered_indices:
        report_position = reports[index]["position"]
        request_position = requests[index]["position"]
        visible = []
        for artifact in artifacts:
            if artifact["position"] >= report_position:
                continue
            entry = artifact["entry"]
            resolved = _resolve_entry_path(session_path, entry)
            visible.append(
                {
                    "id": artifact["id"],
                    "path": str(resolved),
                    "description": entry.get("description", ""),
                    "type": (
                        "dataframe"
                        if entry.get("type") == "dataframe"
                        else "visualization"
                    ),
                    "preview": entry.get("data_preview", ""),
                    "first_seen_request": first_seen.get(artifact["position"], index),
                    "sent_this_request": request_position < artifact["position"] < report_position,
                    "evidence_eligible": eligibility.get(artifact["position"], True),
                }
            )
        snapshots[index] = {
            "schema_version": 0,
            "request_index": index,
            "user_request": requests[index]["text"],
            "artifacts": visible,
            "context": [],
            "log_abstraction": None,
            "legacy_context_unavailable": True,
        }
    return snapshots


def _find_session_dir(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "manifest.jsonl").is_file():
        return candidate
    manifests = list(candidate.glob("*/manifest.jsonl"))
    if len(manifests) == 1:
        return manifests[0].parent
    manifests = list(candidate.glob("**/manifest.jsonl"))
    if len(manifests) == 1:
        return manifests[0].parent
    raise ValueError(
        "Expected a PMAx session directory, or a parent containing exactly one session."
    )


def _read_manifest(session_path: Path) -> list[dict[str, Any]]:
    entries = []
    for line in (session_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _entry_request_index(entry: dict[str, Any], fallback: int) -> int:
    extra = entry.get("extra") or {}
    if isinstance(extra.get("request_index"), int):
        return extra["request_index"]
    searchable = f"{entry.get('description', '')} {entry.get('file_path', '')}"
    match = re.search(r"request[_ ](\d+)", searchable, re.IGNORECASE)
    return int(match.group(1)) if match else fallback


def _resolve_entry_path(session_path: Path, entry: dict[str, Any]) -> Path:
    recorded = Path(entry.get("file_path", ""))
    if recorded.is_file():
        return recorded
    category = str(entry.get("category", ""))
    relative = session_path / category / recorded.name
    if relative.is_file():
        return relative
    matches = list(session_path.rglob(recorded.name)) if recorded.name else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve saved artifact: {recorded}")


def _rebase_snapshot_paths(session_path: Path, snapshot: dict[str, Any]) -> None:
    for artifact in snapshot.get("artifacts", []):
        recorded = Path(str(artifact.get("path", "")))
        if recorded.is_file():
            continue
        matches = list(session_path.rglob(recorded.name)) if recorded.name else []
        if len(matches) == 1:
            artifact["path"] = str(matches[0])

