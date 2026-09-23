import importlib.util
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from promoai.agents.artifact_catalog import ArtifactCatalog
from promoai.agents.pm4py_wrapper import PM4PYWrapper
from promoai.agents.prompt_builders import AnalystPromptBuilder, EngineerPromptBuilder
from promoai.agents.sax_causal import (
    EDGE_COLUMNS,
    SAXCausalAnalysis,
    SAXCausalInputError,
    analyze_causal_dependencies,
    normalize_event_log,
    validate_min_strength,
)
from promoai.agents.state import PMaxSession


def _event_log(case_count=3):
    rows = []
    origin = pd.Timestamp("2026-01-01T08:00:00Z")
    for case_index in range(case_count):
        case_start = origin + timedelta(days=case_index)
        activity_offsets = (
            ("A", 0),
            ("B", 3 + case_index),
            ("C", 8 + 2 * case_index),
        )
        for activity, minutes in activity_offsets:
            rows.append(
                {
                    "case:concept:name": case_index,
                    "concept:name": activity,
                    "time:timestamp": case_start + timedelta(minutes=minutes),
                }
            )
    return pd.DataFrame(rows)


class _FakeGraph:
    def render(self, base_path, format="png", cleanup=True):
        output_path = Path(f"{base_path}.{format}")
        output_path.write_bytes(b"fake graph")
        return str(output_path)


class SAXCausalAdapterTests(unittest.TestCase):
    def test_normalizes_required_columns_and_timestamps_without_mutating_input(self):
        event_log = _event_log(1)
        event_log["time:timestamp"] = event_log["time:timestamp"].astype(str)

        normalized = normalize_event_log(event_log)

        self.assertEqual(str(normalized["case:concept:name"].dtype), "string")
        self.assertEqual(str(normalized["concept:name"].dtype), "string")
        self.assertIsNotNone(normalized["time:timestamp"].dt.tz)
        self.assertEqual(str(event_log["case:concept:name"].dtype), "int64")

    def test_rejects_missing_required_columns_and_invalid_timestamps(self):
        with self.assertRaises(SAXCausalInputError):
            normalize_event_log(pd.DataFrame({"concept:name": ["A"]}))

        event_log = _event_log(1)
        event_log["time:timestamp"] = event_log["time:timestamp"].astype(str)
        event_log.loc[0, "time:timestamp"] = "not-a-timestamp"
        with self.assertRaises(SAXCausalInputError):
            normalize_event_log(event_log)

    def test_validates_strength_threshold(self):
        self.assertEqual(validate_min_strength(0), 0.0)
        self.assertEqual(validate_min_strength(1), 1.0)
        for value in (-0.01, 1.01, float("nan"), True, "0.3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_min_strength(value)

    def test_maps_sax_edges_and_orders_them_by_strength(self):
        model = Mock()
        model.getColumns.return_value = ["A", "B", "C"]
        sax_process_mining = SimpleNamespace(
            create_from_dataframe=Mock(return_value="sax-event-log")
        )
        sax_causal_discovery = SimpleNamespace(
            discover_causal_dependencies=Mock(return_value=model),
            get_model_causal_representation=Mock(
                return_value={
                    ("A", "B"): 0.41,
                    ("B", "C"): 0.83,
                    ("C", "A"): 0.1,
                    ("A", "D"): float("nan"),
                }
            ),
            view_causal_dependencies=Mock(return_value="graph"),
        )

        with patch(
            "promoai.agents.sax_causal._load_sax_modules",
            return_value=(sax_process_mining, sax_causal_discovery),
        ):
            result = analyze_causal_dependencies(_event_log(), min_strength=0.4)

        self.assertEqual(list(result.edges.columns), EDGE_COLUMNS)
        self.assertEqual(result.edges["cause_activity"].tolist(), ["B", "A"])
        self.assertEqual(result.edges["effect_activity"].tolist(), ["C", "B"])
        self.assertEqual(result.node_count, 3)
        self.assertEqual(result.graph, "graph")
        create_call = sax_process_mining.create_from_dataframe.call_args
        self.assertEqual(create_call.kwargs["case_id"], "case:concept:name")
        self.assertEqual(create_call.kwargs["activity_key"], "concept:name")
        self.assertEqual(create_call.kwargs["timestamp_key"], "time:timestamp")
        sax_causal_discovery.discover_causal_dependencies.assert_called_once_with(
            dataObject="sax-event-log", prior_knowledge=True
        )
        sax_causal_discovery.get_model_causal_representation.assert_called_once_with(
            model, p_value_threshold=0.4
        )


class SAXCausalWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.event_log = _event_log()
        self.state = PMaxSession("causal question", self.event_log, self.temp_dir.name)
        self.api = PM4PYWrapper(self.state, None)

    def test_saves_edge_table_and_graph_without_changing_event_log(self):
        edges = pd.DataFrame(
            [{"cause_activity": "A", "effect_activity": "B", "strength": 0.8}]
        )
        analysis = SAXCausalAnalysis(edges=edges, graph=_FakeGraph(), node_count=3)

        with patch(
            "promoai.agents.pm4py_wrapper.analyze_causal_dependencies",
            return_value=analysis,
        ) as analyze:
            self.api.discover_causal_dependencies(0.4)

        analyze.assert_called_once_with(self.event_log, 0.4)
        artifact_paths = list(self.state["saved_artifacts"])
        self.assertEqual(len(artifact_paths), 2)
        self.assertTrue(any(path.endswith(".csv") for path in artifact_paths))
        self.assertTrue(any(path.endswith(".png") for path in artifact_paths))
        self.assertTrue(all(Path(path).exists() for path in artifact_paths))
        self.assertIs(self.api.event_log, self.event_log)
        self.assertTrue(
            any("observationally inferred" in item for item in self.state["context"])
        )

    def test_discovery_failure_becomes_empty_evidence_instead_of_failing_workflow(self):
        with patch(
            "promoai.agents.pm4py_wrapper.analyze_causal_dependencies",
            side_effect=RuntimeError("too few traces"),
        ):
            self.api.discover_causal_dependencies()

        artifact_paths = list(self.state["saved_artifacts"])
        self.assertEqual(len(artifact_paths), 1)
        empty_edges = pd.read_csv(artifact_paths[0])
        self.assertEqual(list(empty_edges.columns), EDGE_COLUMNS)
        self.assertTrue(empty_edges.empty)
        self.assertTrue(
            any("could not infer" in item for item in self.state["context"])
        )

    def test_missing_dependency_is_a_configuration_error(self):
        with patch(
            "promoai.agents.pm4py_wrapper.analyze_causal_dependencies",
            side_effect=ModuleNotFoundError("sax"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                self.api.discover_causal_dependencies()

    def test_generated_engineer_code_uses_api_without_importing_sax(self):
        response = """```python
api.discover_causal_dependencies()
final_event_log = api.event_log
```"""

        with patch.object(self.api, "discover_causal_dependencies") as discover:
            code, final_event_log = self.api.code_extraction(response)

        discover.assert_called_once_with()
        self.assertNotIn("import sax", code)
        self.assertIs(final_event_log, self.event_log)

    def test_disabled_causal_analysis_is_hidden_and_cannot_run(self):
        self.state["causal_enabled"] = False

        self.assertNotIn("discover_causal_dependencies", self.api.get_API_summary())
        with patch(
            "promoai.agents.pm4py_wrapper.analyze_causal_dependencies"
        ) as analyze:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                self.api.discover_causal_dependencies()

        analyze.assert_not_called()


class SAXCausalPromptTests(unittest.TestCase):
    def test_engineer_prompt_advertises_supported_causal_scope(self):
        state = {
            "messages_eng": [],
            "user_request": ["Which activity causes B to be delayed?"],
            "log_abstraction": "summary",
        }

        prompt = EngineerPromptBuilder().build(
            state, PM4PYWrapper.get_API_summary()
        )[0]["content"]

        self.assertIn("min_strength: float = 0.3", prompt)
        self.assertIn("api.discover_causal_dependencies()", prompt)
        self.assertIn("You MUST call", prompt)
        self.assertIn("Do this even when previous generated code", prompt)
        self.assertIn("do not substitute for it", prompt)
        self.assertIn("must not import or call the `sax` package directly", prompt)
        self.assertIn("automatically saves two artifacts", prompt)
        self.assertIn("final_event_log = api.event_log", prompt)
        self.assertIn("case attribute", prompt)
        self.assertIn("business outcome or KPI", prompt)

    def test_analyst_prompt_prevents_overstated_causal_claims(self):
        state = {
            "user_request": ["What causes B?"],
            "log_abstraction": "summary",
        }

        prompt = AnalystPromptBuilder._initial_message(state, "causal artifacts")

        self.assertIn("SAX4BPM-inferred causal execution dependencies", prompt)
        self.assertIn("interventionally proven causation", prompt)
        self.assertIn("include both the graph and edge table", prompt)

    def test_disabled_causal_analysis_is_omitted_from_prompts(self):
        state = {
            "messages_eng": [],
            "user_request": ["Analyze the process"],
            "log_abstraction": "summary",
            "causal_enabled": False,
        }

        engineer_prompt = EngineerPromptBuilder().build(state, "API summary")[0][
            "content"
        ]
        analyst_prompt = AnalystPromptBuilder._initial_message(state, "context")

        self.assertNotIn("SAX4BPM", engineer_prompt)
        self.assertNotIn("SAX4BPM", analyst_prompt)

    def test_causal_edge_table_is_allowed_even_when_its_preview_is_long(self):
        catalog = ArtifactCatalog.from_saved_artifacts(
            {
                "causal.csv": (
                    "SAX4BPM causal execution dependencies (minimum strength 0.3)",
                    "x" * 600,
                ),
                "ordinary.csv": ("Ordinary long table", "x" * 600),
            }
        )

        prompt_input = catalog.build_prompt_input([], [])

        self.assertEqual(catalog.long_dataframe_ids, ["artifact_1"])
        self.assertIn(
            "SAX4BPM CAUSAL EDGE TABLE",
            prompt_input.artifact_dataframe_section,
        )


@unittest.skipUnless(
    importlib.util.find_spec("sax") is not None,
    "sax4bpm is not installed in this test environment",
)
class SAXCausalIntegrationTests(unittest.TestCase):
    def test_sax_runs_on_repeated_synthetic_process_log(self):
        result = analyze_causal_dependencies(_event_log(case_count=40))

        self.assertEqual(list(result.edges.columns), EDGE_COLUMNS)
        self.assertGreater(result.node_count, 0)
        self.assertIsNotNone(result.graph)


if __name__ == "__main__":
    unittest.main()
