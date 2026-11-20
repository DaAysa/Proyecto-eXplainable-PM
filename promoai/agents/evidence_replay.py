"""Evaluate saved sessions without running Engineer or Analyst.

Run with ``python -m promoai.agents.evidence_replay --help``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PureWindowsPath

from promoai.agents.evidence_checker import EvidenceCheckerAgent
from promoai.agents.persistence import AgentRunRecorder
from promoai.general_utils.llm_connection import LLMConnection


def _session_path(session: Path, saved_path: str) -> Path:
    """Resolve relocated sessions using their preserved category and filename."""
    original = Path(saved_path)
    parts = PureWindowsPath(saved_path).parts if "\\" in saved_path else original.parts
    if len(parts) >= 2:
        relocated = session / parts[-2] / parts[-1]
        if relocated.is_file():
            return relocated
    if original.is_file():
        return original
    if len(parts) >= 2:
        return relocated
    return session / saved_path


def _is_engineer_retry(session: Path, entry: dict) -> bool:
    """Identify a failed execution boundary when old request traces survive."""
    try:
        trace = json.loads(_session_path(session, entry["file_path"]).read_text(encoding="utf-8"))
        request = trace["request"]
        payload = request.get("payload", request)
        messages = payload.get("messages", payload.get("input", payload.get("contents", [])))
        system = payload.get("config", {}).get("system_instruction", "") or ""
        system += " ".join(str(message.get("content", "")) for message in messages if message.get("role") == "system")
        last = messages[-1]
        text = str(last.get("content", "")) + " ".join(part.get("text", "") for part in last.get("parts", []))
        return "Process Mining Data Engineer" in system and "Executing your code led to an error!" in text
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return False


def load_evidence_session(session_dir: str | Path, request_index: int) -> dict:
    session = Path(session_dir).resolve()
    entries = [json.loads(line) for line in (session / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshots = []
    for entry in entries:
        if entry["type"] == "analyst_evidence":
            snapshot = json.loads(_session_path(session, entry["file_path"]).read_text(encoding="utf-8"))
            if snapshot["request_index"] <= request_index:
                for artifact in snapshot["artifacts"]:
                    artifact["file_path"] = str(_session_path(session, artifact["file_path"]))
                snapshots.append(snapshot)
    requests = []
    request_texts = {snapshot["request_index"]: snapshot["user_request"] for snapshot in snapshots}
    pending_artifacts = []
    artifact_count = 0
    engineer_completed = False
    report_path = None
    recorded_indices = {snapshot["request_index"] for snapshot in snapshots}
    for entry in entries:
        entry_type = entry["type"]
        if entry_type == "user_request":
            if not engineer_completed:
                for artifact in pending_artifacts:
                    artifact["eligible"] = False
            engineer_completed = False
            requests.append(_session_path(session, entry["file_path"]).read_text(encoding="utf-8"))
            request_texts.setdefault(len(requests), requests[-1])
        elif entry_type == "llm_request" and pending_artifacts and not engineer_completed:
            if _is_engineer_retry(session, entry):
                for artifact in pending_artifacts:
                    artifact["eligible"] = False
        elif entry_type == "generated_code":
            extra = entry.get("extra", {})
            if extra.get("role") == "engineer":
                engineer_completed = True
            if extra.get("request_index") and extra.get("user_request"):
                request_texts.setdefault(extra["request_index"], extra["user_request"])
        elif entry_type in {"dataframe", "visualization"}:
            if len(requests) > request_index:
                continue
            pending_artifacts.append({
                "id": f"artifact_{artifact_count}",
                "file_path": str(_session_path(session, entry["file_path"])),
                "description": entry["description"],
                "preview": str(entry.get("data_preview", entry["description"])) if entry_type == "dataframe"
                else "Visual representation available (PNG). Reference this key to display the chart",
                "eligible": True,
            })
            artifact_count += 1
        elif entry_type == "report":
            match = re.search(r"assistant_report_request_(\d+)\.json$", entry["file_path"])
            if match is None:
                continue
            index = int(match.group(1))
            if index > request_index:
                continue
            if index not in recorded_indices:
                snapshots.append({
                    "request_index": index, "user_request": request_texts.get(index, ""),
                    "artifacts": list(pending_artifacts), "context": [], "log_abstraction": None,
                    "provenance": "inferred_order_no_context",
                })
            pending_artifacts.clear()
            if index == request_index:
                report_path = _session_path(session, entry["file_path"])
    if report_path is None or request_index < 1 or not request_texts.get(request_index):
        raise ValueError(f"No saved Analyst report for request {request_index} in {session}.")
    # Re-running Analyst for the same question supersedes its previous snapshot.
    latest_snapshots = {snapshot["request_index"]: snapshot for snapshot in snapshots}
    return {
        "artifact_session_dir": str(session),
        "user_request": [request_texts.get(index, "") for index in range(1, request_index + 1)],
        "evidence_request_index": request_index,
        "analyst_evidence": list(latest_snapshots.values()),
        "final_report": json.loads(report_path.read_text(encoding="utf-8")),
    }


def check_saved_session(session_dir, request_index, credentials, response_generator=None, progress_callback=None, persist=True):
    state = load_evidence_session(session_dir, request_index)
    kwargs = {"response_generator": response_generator} if response_generator is not None else {}
    result = EvidenceCheckerAgent(credentials, **kwargs).execute(state, progress_callback)
    if persist:
        AgentRunRecorder().record_evidence_checker(state, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run EvidenceChecker on a saved PMAx session.")
    parser.add_argument("session_dir")
    parser.add_argument("--request", type=int, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="PMAX_API_KEY")
    parser.add_argument("--llm-args", default="{}", help="Provider arguments as a JSON object.")
    args = parser.parse_args()
    import os

    credentials = LLMConnection(os.environ[args.api_key_env], args.model, args.provider, json.loads(args.llm_args))
    result = check_saved_session(args.session_dir, args.request, credentials)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
