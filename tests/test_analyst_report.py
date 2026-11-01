import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from promoai.agents.analyst import AnalystAgent
from promoai.agents.state import PMaxSession
from promoai.agents.utils import code_extraction_report
from promoai.general_utils.llm_connection import LLMConnection


def _report_response(blocks):
    return f"```python\nfinal_report = {blocks!r}\n```"


class AnalystReportTests(unittest.TestCase):
    def setUp(self):
        print_patch = patch("promoai.agents.utils.constants.ENABLE_PRINTS", False)
        print_patch.start()
        self.addCleanup(print_patch.stop)

    def test_accepts_descriptive_text_and_structured_artifact_references(self):
        blocks = [
            {"type": "artifact", "content": "artifact_0"},
            {
                "type": "text",
                "content": "The causal dependency graph above shows A influencing B.",
            },
        ]

        _, report = code_extraction_report(
            _report_response(blocks), None, ["artifact_0"], []
        )

        self.assertEqual(report, blocks)

    def test_rejects_internal_reference_in_causal_report_narrative(self):
        text = (
            "The causal dependency graph above (and the corresponding causal "
            "execution dependencies in artifact_0) illustrates the directional "
            "relationships between process steps."
        )

        with self.assertRaisesRegex(ValueError, "internal artifact ID"):
            code_extraction_report(
                _report_response([{"type": "text", "content": text}]),
                None,
                ["artifact_0"],
                [],
            )

    def test_rejects_unknown_and_formatted_internal_ids_in_text(self):
        for text in (
            "See `artifact_12` for details.",
            "The evidence is in ARTIFACT_99.",
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "internal artifact ID"):
                    code_extraction_report(
                        _report_response([{"type": "text", "content": text}]),
                        None,
                        ["artifact_0"],
                        [],
                    )

    def test_unknown_structured_artifact_reference_remains_invalid(self):
        with self.assertRaisesRegex(Exception, "Invalid artifact reference"):
            code_extraction_report(
                _report_response([{"type": "artifact", "content": "artifact_99"}]),
                None,
                ["artifact_0"],
                [],
            )

    def test_analyst_retries_leaked_id_and_resolves_corrected_report(self):
        leaked_response = _report_response(
            [{"type": "text", "content": "See artifact_0 for the dependencies."}]
        )
        corrected_response = _report_response(
            [
                {"type": "artifact", "content": "artifact_0"},
                {"type": "text", "content": "The graph above shows the dependencies."},
            ]
        )
        with tempfile.TemporaryDirectory() as task_dir:
            event_log = pd.DataFrame(
                {
                    "case:concept:name": ["1"],
                    "concept:name": ["A"],
                    "time:timestamp": pd.to_datetime(["2026-01-01"]),
                }
            )
            state = PMaxSession("Show the causal dependencies", event_log, task_dir)
            state["saved_artifacts"] = {"graph.png": ("Causal dependency graph", {})}
            credentials = LLMConnection("unused", "unused", "unused", {})
            with patch(
                "promoai.general_utils.llm_connection.query_llm",
                side_effect=[leaked_response, corrected_response],
            ) as query:
                result = AnalystAgent(credentials).execute(state)

        self.assertEqual(query.call_count, 2)
        self.assertEqual(result.report[0], {"type": "artifact", "content": "graph.png"})
        self.assertEqual(
            result.report[1],
            {"type": "text", "content": "The graph above shows the dependencies."},
        )
        errors = [message for message in result.messages if message.get("type") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("internal artifact ID", errors[0]["content"])
        self.assertIn("descriptive name", errors[0]["content"])


if __name__ == "__main__":
    unittest.main()
