from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from promoai.agents.contracts import StructuredReport
from promoai.agents.prompt_builders import AnalystPromptInput


@dataclass(frozen=True)
class AgentArtifact:
    id: str
    file_path: str
    description: str
    content: Any
    type: str


class ArtifactCatalog:
    """Stable view of the artifacts available to the Analyst for one run."""

    def __init__(self, artifacts: list[AgentArtifact]):
        self._artifacts = artifacts
        self._by_id = {artifact.id: artifact for artifact in artifacts}

    @classmethod
    def from_saved_artifacts(
        cls, saved_artifacts: dict[str, tuple[Any, Any]]
    ) -> "ArtifactCatalog":
        artifacts: list[AgentArtifact] = []
        for index, (file_path, (description, data)) in enumerate(
            saved_artifacts.items()
        ):
            artifact_type = cls._type_from_path(file_path)
            content = (
                "Visual representation available (PNG). Reference this key to display the chart"
                if artifact_type == "visualization"
                else data
            )
            artifacts.append(
                AgentArtifact(
                    id=f"artifact_{index}",
                    file_path=file_path,
                    description=str(description).strip("{}'\" "),
                    content=content,
                    type=artifact_type,
                )
            )
        return cls(artifacts)

    @staticmethod
    def _type_from_path(file_path: str) -> str:
        if file_path.endswith((".png", ".jpg", ".jpeg")):
            return "visualization"
        if file_path.endswith(".csv"):
            return "dataframe"
        raise ValueError(
            f"Unsupported artifact type for artifact {file_path}. "
            "Only image and csv files are supported."
        )

    @property
    def ids(self) -> list[str]:
        return [artifact.id for artifact in self._artifacts]

    @property
    def long_dataframe_ids(self) -> list[str]:
        return [
            artifact.id
            for artifact in self._artifacts
            if artifact.type == "dataframe" and len(str(artifact.content)) > 500
        ]

    def build_prompt_input(
        self, sent_artifact_ids: list[str], context_entries: list[str]
    ) -> AnalystPromptInput:
        dataframes = "\n".join(
            self._format_dataframe(artifact)
            for artifact in self._artifacts
            if artifact.type == "dataframe" and artifact.id not in sent_artifact_ids
        )
        dataframes += "\n \n IMPORTANT: DATAFRAMES SHOULD NOT BE INCLUDED IN THE REPORT IF THE USER DID NOT REQUEST THEM EXPLICITLY. \n \n"

        visualizations = "\n".join(
            self._format_visualization(artifact)
            for artifact in self._artifacts
            if artifact.type == "visualization"
            and artifact.id not in sent_artifact_ids
        )
        context = (
            " ".join(context_entries)
            if context_entries
            else "No additional context available."
        )
        return AnalystPromptInput(dataframes, visualizations, context)

    @staticmethod
    def _format_dataframe(artifact: AgentArtifact) -> str:
        return (
            f"\n =====DATAFRAME {artifact.id}======:\n "
            f"Description: {artifact.description} \n "
            f"Content: {artifact.content} \n "
            f"                                ======END OF {artifact.id} ====== \n\n"
        )

    @staticmethod
    def _format_visualization(artifact: AgentArtifact) -> str:
        return (
            f"\n =====VISUALIZATION {artifact.id}======:\n "
            f"Description: {artifact.description} \n "
            f"Content: {artifact.content} \n \n "
            f"======END OF {artifact.id} ====== \n\n"
        )

    def resolve_report_paths(self, report: StructuredReport) -> StructuredReport:
        resolved_report = copy.deepcopy(report)
        for entry in resolved_report:
            if entry.get("type") != "artifact":
                continue
            artifact_id = entry.get("content")
            artifact = self._by_id.get(artifact_id)
            if artifact is None:
                raise ValueError(f"Key {artifact_id} not found.")
            entry["content"] = artifact.file_path
        return resolved_report
