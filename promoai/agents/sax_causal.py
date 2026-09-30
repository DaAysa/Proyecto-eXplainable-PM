from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

import pandas as pd
import pm4py


CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
TIMESTAMP_COLUMN = "time:timestamp"
EDGE_COLUMNS = ["cause_activity", "effect_activity", "strength"]
SUPPORTED_MODALITIES = {"chain", "parent"}


class SAXCausalInputError(ValueError):
    """Raised when an event log cannot be converted safely for SAX4BPM."""


@dataclass(frozen=True)
class SAXCausalAnalysis:
    """Artifacts derived from a SAX4BPM causal execution dependency model."""

    edges: pd.DataFrame
    graph: Optional[Any]
    node_count: int
    modality: str = "chain"
    algorithm: str = "positive_lingam"
    edge_values_are_binary: bool = False


def validate_min_strength(min_strength: float) -> float:
    """Validate and normalize the causal edge reporting threshold."""
    if isinstance(min_strength, bool) or not isinstance(min_strength, (int, float)):
        raise ValueError("min_strength must be a number between 0 and 1.")

    value = float(min_strength)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("min_strength must be a number between 0 and 1.")
    return value


def validate_causal_modality(modality: str) -> str:
    """Validate and normalize the SAX4BPM temporal anchoring modality."""
    if not isinstance(modality, str):
        raise ValueError("modality must be either 'chain' or 'parent'.")
    value = modality.strip().lower()
    if value not in SUPPORTED_MODALITIES:
        raise ValueError("modality must be either 'chain' or 'parent'.")
    return value


def normalize_event_log(event_log: Any) -> pd.DataFrame:
    """Return a normalized event-log copy suitable for SAX4BPM."""
    dataframe = (
        event_log
        if isinstance(event_log, pd.DataFrame)
        else pm4py.convert_to_dataframe(event_log)
    ).copy()
    missing_columns = [
        column
        for column in (CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise SAXCausalInputError(
            "SAX4BPM causal analysis requires the standard event-log columns: "
            + ", ".join(missing_columns)
            + "."
        )
    if dataframe.empty:
        raise SAXCausalInputError(
            "SAX4BPM causal analysis requires a non-empty event log."
        )

    dataframe[CASE_ID_COLUMN] = dataframe[CASE_ID_COLUMN].astype("string")
    dataframe[ACTIVITY_COLUMN] = dataframe[ACTIVITY_COLUMN].astype("string")
    try:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="raise",
            utc=True,
            format="mixed",
        )
    except (TypeError, ValueError) as exc:
        raise SAXCausalInputError(
            "SAX4BPM causal analysis requires valid event timestamps."
        ) from exc

    mandatory_data = dataframe[[CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN]]
    if mandatory_data.isna().any().any():
        raise SAXCausalInputError(
            "SAX4BPM causal analysis does not accept missing case IDs, activities, "
            "or timestamps."
        )

    return dataframe.sort_values(
        [CASE_ID_COLUMN, TIMESTAMP_COLUMN], kind="stable"
    ).reset_index(drop=True)


def _load_sax_modules():
    from sax.core.causal_process_discovery import causal_discovery
    from sax.core.process_mining import process_mining

    return process_mining, causal_discovery


def analyze_causal_dependencies(
    event_log: Any,
    min_strength: float = 0.3,
    modality: str = "chain",
) -> SAXCausalAnalysis:
    """Discover activity execution dependencies with SAX4BPM."""
    threshold = validate_min_strength(min_strength)
    normalized_modality = validate_causal_modality(modality)
    dataframe = normalize_event_log(event_log)
    sax_process_mining, sax_causal_discovery = _load_sax_modules()
    from sax.core.causal_process_discovery.causal_constants import Algorithm, Modality

    sax_modality = (
        Modality.PARENT
        if normalized_modality == "parent"
        else Modality.CHAIN
    )
    # SAX4BPM 0.0.2 has a defect in ParentAnchorTransformer's PositiveLiNGAM
    # branch: it checks the loop's variant string instead of the requested
    # algorithm and can leave the algorithm variable unbound. Standard LiNGAM
    # exercises the same parent-anchored transformation without that defect.
    sax_algorithm = (
        Algorithm.LINGAM
        if normalized_modality == "parent"
        else Algorithm.POSITIVE_LINGAM
    )

    sax_event_log = sax_process_mining.create_from_dataframe(
        dataframe,
        kloop_unroling=False,
        case_id=CASE_ID_COLUMN,
        activity_key=ACTIVITY_COLUMN,
        timestamp_key=TIMESTAMP_COLUMN,
    )
    model = sax_causal_discovery.discover_causal_dependencies(
        dataObject=sax_event_log,
        algorithm=sax_algorithm,
        modality=sax_modality,
        prior_knowledge=True,
        threshold=threshold,
    )
    columns = list(model.getColumns())
    relationships = sax_causal_discovery.get_model_causal_representation(
        model, p_value_threshold=threshold
    )
    edge_records = []
    for (cause, effect), strength in relationships.items():
        strength_value = float(strength)
        if isfinite(strength_value) and strength_value >= threshold:
            edge_records.append(
                {
                    "cause_activity": cause,
                    "effect_activity": effect,
                    "strength": strength_value,
                }
            )
    edges = pd.DataFrame(edge_records, columns=EDGE_COLUMNS)
    edge_values_are_binary = bool(edge_records) and all(
        record["strength"] in (0.0, 1.0) for record in edge_records
    )
    if not edges.empty:
        edges = edges.sort_values(
            "strength", ascending=False, kind="stable"
        ).reset_index(drop=True)

    graph = None
    if columns:
        graph = sax_causal_discovery.view_causal_dependencies(
            model, p_value_threshold=threshold
        )

    return SAXCausalAnalysis(
        edges=edges,
        graph=graph,
        node_count=len(columns),
        modality=normalized_modality,
        algorithm=sax_algorithm.value,
        edge_values_are_binary=edge_values_are_binary,
    )
