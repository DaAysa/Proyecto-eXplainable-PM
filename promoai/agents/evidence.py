"""Record exactly what the Analyst received without altering its conversation."""
from __future__ import annotations

from typing import Any


def capture_analyst_evidence(state) -> dict[str, Any]:
    from promoai.agents.artifact_catalog import ArtifactCatalog

    excluded = set(state.get("excluded_evidence_artifacts", []))
    sent = set(state.get("sent_artifacts", []))
    artifacts = []
    catalog = ArtifactCatalog.from_saved_artifacts(state["saved_artifacts"])
    for artifact in catalog.artifacts:
        if artifact.id in sent:
            continue
        artifacts.append({
            "id": artifact.id,
            "file_path": artifact.file_path,
            "description": artifact.description,
            "preview": str(artifact.content),
            "eligible": artifact.file_path not in excluded,
        })
    return {
        "request_index": len(state["user_request"]),
        "user_request": state["user_request"][-1],
        "artifacts": artifacts,
        "context": list(state["context"]),
        "excluded_context_indices": list(state.get("excluded_evidence_context", [])),
        # Follow-up prompts retain the initial system message in their history.
        "log_abstraction": state["log_abstraction"] if not state["messages_ana"] else None,
        "provenance": "recorded",
    }


def apply_analyst_evidence(state, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    history = state.setdefault("analyst_evidence", [])
    history[:] = [item for item in history if item["request_index"] != snapshot["request_index"]]
    history.append(snapshot)


def evidence_sources(state) -> list[dict[str, Any]]:
    """Prefer the current request, then the evidence retained in Analyst history."""
    sources = []
    seen_paths = set()
    request_index = state.get("evidence_request_index", len(state["user_request"]))
    for snapshot in sorted(state.get("analyst_evidence", []), key=lambda x: x["request_index"], reverse=True):
        index = snapshot["request_index"]
        if index > request_index:
            continue
        for artifact in snapshot["artifacts"]:
            path = artifact["file_path"]
            if not artifact.get("eligible", True) or path in seen_paths:
                continue
            seen_paths.add(path)
            sources.append({
                **artifact,
                "source_id": f"q{index}/{artifact['id']}",
                "request_index": index,
                "current_request": index == request_index,
                "kind": "table" if path.lower().endswith(".csv") else "visualization",
            })
        excluded = set(snapshot.get("excluded_context_indices", []))
        texts = [(f"context_{i}", text) for i, text in enumerate(snapshot.get("context", [])) if i not in excluded]
        if snapshot.get("log_abstraction"):
            texts.append(("log_abstraction", snapshot["log_abstraction"]))
        for name, text in texts:
            sources.append({
                "source_id": f"q{index}/{name}", "id": name,
                "request_index": index, "current_request": index == request_index,
                "kind": "context", "description": str(text)[:200], "preview": str(text),
            })
    return sources
