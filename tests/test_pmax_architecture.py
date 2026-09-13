import tempfile
import unittest

import pandas as pd

from promoai.agents.artifact_catalog import ArtifactCatalog
from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import AnalystResult, EngineerResult
from promoai.agents.engineer import EngineerAgent
from promoai.agents.agents import generate_initial_message_for_analyst
from promoai.agents.state import PMaxSession, ProcessState
from promoai.agents.workflow import PMaxWorkflow
from promoai.general_utils.llm_connection import LLMConnection


class _FakeEngineer:
    def __init__(self, result):
        self.result = result

    def execute(self, state, progress_callback=None):
        return self.result


class _FakeAnalyst:
    def __init__(self, result):
        self.result = result

    def execute(self, state, progress_callback=None):
        return self.result


class _RecordingRecorder:
    def __init__(self):
        self.calls = []

    def record_engineer(self, state, result):
        self.calls.append(("engineer", result))

    def record_analyst(self, state, result):
        self.calls.append(("analyst", result))


class _ContextStep:
    name = "Context enricher"

    def execute(self, state, progress_callback=None):
        state.add_context("extra context")
        return "enriched"


class PMaxArchitectureTests(unittest.TestCase):
    def _session(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        event_log = pd.DataFrame(
            {
                "case:concept:name": ["1"],
                "concept:name": ["Start"],
                "time:timestamp": pd.to_datetime(["2026-01-01"]),
            }
        )
        return PMaxSession("analyze", event_log, temp_dir.name)

    def test_process_state_name_remains_compatible(self):
        self.assertIs(ProcessState, PMaxSession)

    def test_legacy_prompt_helper_accepts_mapping_state(self):
        state = {"user_request": ["request"], "log_abstraction": "summary"}
        prompt = generate_initial_message_for_analyst(state, "context")
        self.assertIn('User Request: "request"', prompt)

    def test_workflow_applies_typed_agent_results(self):
        session = self._session()
        processed_log = session["event_log"].assign(processed=True)
        engineer_result = EngineerResult(
            processed_log, "final_event_log = api.event_log", [{"role": "assistant"}]
        )
        analyst_result = AnalystResult(
            [{"type": "text", "content": "Done"}],
            "final_report = []",
            [{"role": "assistant"}],
            ("artifact_0",),
        )
        recorder = _RecordingRecorder()
        workflow = PMaxWorkflow(
            credentials=object(),
            engineer=_FakeEngineer(engineer_result),
            analyst=_FakeAnalyst(analyst_result),
            recorder=recorder,
            intermediate_steps=[_ContextStep()],
        )

        result = workflow.execute(session)

        self.assertIs(result.engineer, engineer_result)
        self.assertIs(result.analyst, analyst_result)
        self.assertEqual(result.intermediate_results, ("enriched",))
        self.assertIs(session["event_log"], processed_log)
        self.assertEqual(session["final_report"], analyst_result.report)
        self.assertEqual(session["sent_artifacts"], ["artifact_0"])
        self.assertEqual([name for name, _ in recorder.calls], ["engineer", "analyst"])

    def test_artifact_catalog_keeps_legacy_ids_and_resolves_paths(self):
        catalog = ArtifactCatalog.from_saved_artifacts(
            {
                "chart.png": ("Cases per variant", {"a": 1}),
                "summary.csv": ("Summary", "preview"),
            }
        )

        self.assertEqual(catalog.ids, ["artifact_0", "artifact_1"])
        resolved = catalog.resolve_report_paths(
            [
                {"type": "text", "content": "Result"},
                {"type": "artifact", "content": "artifact_0"},
            ]
        )
        self.assertEqual(resolved[1]["content"], "chart.png")

    def test_engineer_agent_returns_typed_result(self):
        session = self._session()
        credentials = LLMConnection("key", "model", "provider", {})

        def generate(messages, **kwargs):
            self.assertEqual(kwargs["max_iterations"], 7)
            self.assertEqual(kwargs["additional_iterations"], 5)
            return "final_event_log = api.event_log", session["event_log"], messages

        result = EngineerAgent(credentials, result_generator=generate).execute(session)

        self.assertIsInstance(result, EngineerResult)
        self.assertIs(result.event_log, session["event_log"])
        self.assertEqual(result.generated_code, "final_event_log = api.event_log")
        self.assertEqual(result.messages[-1]["content"], "analyze")

    def test_analyst_agent_returns_typed_result_and_resolves_artifact(self):
        session = self._session()
        session["saved_artifacts"] = {"chart.png": ("Chart", {"cases": 1})}
        credentials = LLMConnection("key", "model", "provider", {})

        def generate(messages, **kwargs):
            self.assertEqual(kwargs["max_iterations"], 7)
            self.assertEqual(kwargs["additional_iterations"], 5)
            return (
                "final_report = []",
                [{"type": "artifact", "content": "artifact_0"}],
                messages,
            )

        result = AnalystAgent(credentials, result_generator=generate).execute(session)

        self.assertIsInstance(result, AnalystResult)
        self.assertEqual(result.report[0]["content"], "chart.png")
        self.assertEqual(result.sent_artifact_ids, ("artifact_0",))


if __name__ == "__main__":
    unittest.main()
