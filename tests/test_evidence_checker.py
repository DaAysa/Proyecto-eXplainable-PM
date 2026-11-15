import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from promoai.agents.contracts import (
    AnalystResult,
    EngineerResult,
    EvidenceCheckResult,
    VerificationResult,
)
from promoai.agents.evidence_checker import (
    EvidenceCheckerAgent,
    capture_analyst_evidence,
    evaluate_saved_session,
    load_saved_session,
)
from promoai.agents.engineer import EngineerAgent
from promoai.agents.state import PMaxSession
from promoai.agents.workflow import PMaxWorkflow
from promoai.general_utils.llm_connection import LLMConnection


def _responses(
    claimed_value="51.21%",
    status="supported",
    response_language="en",
    summary="The response is grounded in the available evidence.",
):
    return [
        json.dumps(
            {
                "response_language": response_language,
                "question_parts": [
                    {
                        "id": "part_1",
                        "text": "Percentage of cancelled cases",
                        "data_type": "frequency",
                    }
                ],
                "claims": [
                    {
                        "id": "claim_1",
                        "text": f"Cancelled cases represent {claimed_value} of all cases.",
                        "part_ids": ["part_1"],
                        "verifiability": "verifiable",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "routes": [
                    {
                        "part_id": "part_1",
                        "artifact_ids": ["artifact_0"],
                        "include_context": False,
                        "include_log_abstraction": False,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "claim_assessments": [
                    {
                        "claim_id": "claim_1",
                        "status": status,
                        "explanation": "The value is present in the outcome table.",
                        "citation": {
                            "source_type": "artifact",
                            "artifact_id": "artifact_0",
                            "row": 1,
                            "column": "value",
                            "value": 0.5121,
                            "claimed_value": claimed_value,
                        },
                    }
                ],
                "part_assessments": [
                    {
                        "part_id": "part_1",
                        "status": "answered",
                        "evidence_contains_answer": True,
                    }
                ],
                "summary": summary,
            }
        ),
    ]


class _SequenceResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.responses.pop(0)


class EvidenceCheckerTests(unittest.TestCase):
    def _session(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        csv_path = root / "outcomes.csv"
        pd.DataFrame(
            {"metric": ["cancelled_share"], "value": [0.5121]}
        ).to_csv(csv_path, index=False)
        event_log = pd.DataFrame(
            {
                "case:concept:name": ["1"],
                "concept:name": ["Start"],
                "time:timestamp": pd.to_datetime(["2026-01-01"]),
            }
        )
        session = PMaxSession("What percentage was cancelled?", event_log, str(root))
        session["saved_artifacts"] = {
            str(csv_path): (
                "Case outcome shares",
                "```markdown\n| metric | value |\n| cancelled_share | 0.5121 |\n```",
            )
        }
        session["context"] = ["The table contains case-level outcome shares."]
        session["final_report"] = [
            {
                "type": "text",
                "content": "Cancelled cases represent 51.21% of all cases.",
            }
        ]
        capture_analyst_evidence(session)
        return session

    def test_checker_uses_temperature_zero_and_confirms_percentage_from_csv(self):
        session = self._session()
        responder = _SequenceResponder(_responses())
        events = []
        checker = EvidenceCheckerAgent(
            LLMConnection("key", "model", "provider", {"top_p": 0.8}),
            response_generator=responder,
        )

        result = checker.execute(session, events.append)

        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.claims[0]["status"], "supported")
        self.assertEqual(
            result.claims[0]["citation"]["numeric_validation"], "confirmed"
        )
        self.assertEqual(len(responder.calls), 3)
        self.assertTrue(
            all(call[1]["llm_args"]["temperature"] == 0 for call in responder.calls)
        )
        self.assertTrue(
            all(call[1]["llm_args"]["top_p"] == 0.8 for call in responder.calls)
        )
        self.assertEqual(
            [event["attempt"] for event in events if event["event"] == "request_started"],
            [1, 2, 3],
        )
        final_system_prompt = responder.calls[2][0][0]["content"]
        self.assertIn("ISO 639-1 code 'en'", final_system_prompt)
        self.assertIn("exclusively in that language", final_system_prompt)

    def test_numeric_code_overrides_an_incorrect_llm_support_decision(self):
        session = self._session()
        responder = _SequenceResponder(_responses("60%", status="supported"))

        result = EvidenceCheckerAgent(
            LLMConnection("key", "model", "provider", {}), responder
        ).execute(session)

        self.assertEqual(result.claims[0]["status"], "contradicted")
        self.assertEqual(result.confidence, "low")

    def test_question_language_is_propagated_to_the_final_llm_stage(self):
        session = self._session()
        session["user_request"][-1] = "¿Qué porcentaje de casos fue cancelado?"
        session["evidence_snapshots"][-1]["user_request"] = session["user_request"][-1]
        responder = _SequenceResponder(
            _responses(
                response_language="es",
                summary="La respuesta está respaldada por la evidencia disponible.",
            )
        )

        result = EvidenceCheckerAgent(
            LLMConnection("key", "model", "provider", {}), responder
        ).execute(session)

        final_system_prompt = responder.calls[2][0][0]["content"]
        self.assertIn("ISO 639-1 code 'es'", final_system_prompt)
        self.assertEqual(
            result.summary,
            "La respuesta está respaldada por la evidencia disponible.",
        )

    def test_numpy_integer_cells_are_compared_and_serialized_as_numbers(self):
        self.assertTrue(
            EvidenceCheckerAgent._numbers_equivalent(
                6173, np.int64(6173), "6,173 distinct cases"
            )
        )

        session = self._session()
        csv_path = next(iter(session["saved_artifacts"]))
        pd.DataFrame({"metric": ["total_cases"], "value": [6173]}).to_csv(
            csv_path, index=False
        )
        responses = _responses("6173", status="supported")
        verification = json.loads(responses[2])
        verification["claim_assessments"][0]["citation"].update(
            {"value": 6173, "claimed_value": 6173}
        )
        responses[2] = json.dumps(verification)

        result = EvidenceCheckerAgent(
            LLMConnection("key", "model", "provider", {}),
            _SequenceResponder(responses),
        ).execute(session)

        citation = result.claims[0]["citation"]
        self.assertEqual(result.claims[0]["status"], "supported")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(citation["value"], 6173)
        self.assertIsInstance(citation["value"], int)

    def test_invalid_json_is_rejected(self):
        session = self._session()
        checker = EvidenceCheckerAgent(
            LLMConnection("key", "model", "provider", {}),
            response_generator=lambda *args, **kwargs: "not json",
        )
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            checker.execute(session)

    def test_snapshot_preserves_context_and_excludes_failed_attempt_artifact(self):
        session = self._session()
        failed_path = str(Path(session["artifact_session_dir"]) / "failed.csv")
        pd.DataFrame({"value": [999]}).to_csv(failed_path, index=False)
        session["saved_artifacts"][failed_path] = ("Failed attempt", "999")
        session["engineer_artifact_attempts"]["1"] = {
            "successful": [],
            "failed": [failed_path],
        }

        snapshot = capture_analyst_evidence(session)

        self.assertEqual(snapshot["context"], session["context"])
        self.assertTrue(snapshot["log_abstraction"].startswith("Event Log Abstraction"))
        by_path = {artifact["path"]: artifact for artifact in snapshot["artifacts"]}
        self.assertFalse(by_path[failed_path]["eligible"])

    def test_snapshot_distinguishes_current_and_historical_artifacts(self):
        session = self._session()
        session["sent_artifacts"] = ["artifact_0"]
        session.add_request("How long do accepted cases take?")
        current_path = str(Path(session["artifact_session_dir"]) / "durations.csv")
        pd.DataFrame({"duration": [12.0]}).to_csv(current_path, index=False)
        session["saved_artifacts"][current_path] = (
            "Accepted case durations",
            "duration = 12",
        )

        snapshot = capture_analyst_evidence(session)

        self.assertEqual(
            [(item["id"], item["scope"]) for item in snapshot["artifacts"]],
            [("artifact_0", "history"), ("artifact_1", "current")],
        )
        self.assertEqual(snapshot["artifacts"][0]["request_index"], 1)
        self.assertEqual(snapshot["artifacts"][1]["request_index"], 2)

    def test_engineer_tracks_artifacts_from_failed_and_successful_attempts(self):
        session = self._session()

        def generate(messages, extraction_function, **kwargs):
            failed_code = """```python
api.save_dataframe(pd.DataFrame({"metric": ["bad", "bad2"], "value": [1, 2]}), "Failed metric")
raise ValueError("retry")
final_event_log = api.event_log
```"""
            successful_code = """```python
api.save_dataframe(pd.DataFrame({"metric": ["ok", "ok2"], "value": [3, 4]}), "Successful metric")
final_event_log = api.event_log
```"""
            with self.assertRaises(Exception):
                extraction_function(failed_code, False)
            code, event_log = extraction_function(successful_code, False)
            return code, event_log, messages

        EngineerAgent(
            LLMConnection("key", "model", "provider", {}),
            result_generator=generate,
        ).execute(session)

        tracking = session["engineer_artifact_attempts"]["1"]
        self.assertEqual(len(tracking["failed"]), 1)
        self.assertEqual(len(tracking["successful"]), 1)
        self.assertIn("failed_metric", tracking["failed"][0])
        self.assertIn("successful_metric", tracking["successful"][0])


class SavedSessionTests(unittest.TestCase):
    @property
    def fixture(self):
        return Path(__file__).parent / "fixtures" / "legacy_pmax_session"

    def test_legacy_loader_rebases_paths_and_recovers_artifact_order(self):
        state = load_saved_session(self.fixture)
        snapshot = state["evidence_snapshots"][0]

        self.assertTrue(snapshot["legacy_reconstruction"])
        self.assertEqual(snapshot["context"], [])
        self.assertEqual(snapshot["artifacts"][0]["id"], "artifact_0")
        self.assertTrue(Path(snapshot["artifacts"][0]["path"]).exists())
        self.assertTrue(snapshot["artifacts"][0]["eligible"])

    def test_saved_session_api_runs_with_simulated_llm_and_persists_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "session"
            import shutil

            shutil.copytree(self.fixture, destination)
            responder = _SequenceResponder(_responses())
            result = evaluate_saved_session(
                destination,
                LLMConnection("key", "model", "provider", {}),
                response_generator=responder,
            )

            self.assertEqual(result.confidence, "high")
            saved = list((destination / "reports").glob("*evidence_check_request_1.json"))
            self.assertEqual(len(saved), 1)
            manifest = (destination / "manifest.jsonl").read_text(encoding="utf-8")
            self.assertIn('"type": "evidence_check"', manifest)

    def test_original_temp_session_is_loadable_when_available(self):
        session = (
            Path(__file__).parents[1]
            / "temp_4_11_26"
            / "11"
            / "26"
            / "artifacts"
            / "pmax_20261104_225038_5464b83b"
        )
        if not session.exists():
            self.skipTest("The archived temp session is not present in this checkout.")
        state = load_saved_session(session)
        self.assertEqual(state["evidence_snapshots"][0]["request_index"], 1)
        self.assertTrue(state["evidence_snapshots"][0]["artifacts"])


class _FakeAgent:
    def __init__(self, name, result, order):
        self.name = name
        self.result = result
        self.order = order

    def execute(self, state, progress_callback=None):
        self.order.append(self.name)
        return self.result


class _NoopRecorder:
    def record_engineer(self, state, result):
        pass

    def record_analyst(self, state, result):
        pass

    def record_analyst_evidence(self, state, snapshot):
        pass

    def record_verifier(self, state, result):
        pass

    def record_evidence_check(self, state, result):
        pass


class WorkflowEvidenceTests(unittest.TestCase):
    def test_workflow_runs_independent_checks_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            event_log = pd.DataFrame(
                {
                    "case:concept:name": ["1"],
                    "concept:name": ["Start"],
                    "time:timestamp": pd.to_datetime(["2026-01-01"]),
                }
            )
            state = PMaxSession(
                "Analyze",
                event_log,
                temporary,
                response_validation_enabled=True,
                evidence_validation_enabled=True,
            )
            order = []
            engineer_result = EngineerResult(event_log, "code", [])
            analyst_result = AnalystResult(
                [{"type": "text", "content": "Answer"}], "code", [], ()
            )
            verification_result = VerificationResult(True, "Complete")
            evidence_result = EvidenceCheckResult((), (), "medium", "No claims")
            workflow = PMaxWorkflow(
                credentials=object(),
                engineer=_FakeAgent("Engineer", engineer_result, order),
                analyst=_FakeAgent("Analyst", analyst_result, order),
                verifier=_FakeAgent("Verifier", verification_result, order),
                evidence_checker=_FakeAgent(
                    "EvidenceChecker", evidence_result, order
                ),
                recorder=_NoopRecorder(),
            )

            result = workflow.execute(state)

            self.assertEqual(
                order, ["Engineer", "Analyst", "Verifier", "EvidenceChecker"]
            )
            self.assertIs(result.evidence_check, evidence_result)

    def test_evidence_check_still_runs_when_verifier_fails(self):
        class FailingVerifier:
            def execute(self, state, progress_callback=None):
                raise RuntimeError("verifier unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            event_log = pd.DataFrame(
                {
                    "case:concept:name": ["1"],
                    "concept:name": ["Start"],
                    "time:timestamp": pd.to_datetime(["2026-01-01"]),
                }
            )
            state = PMaxSession(
                "Analyze",
                event_log,
                temporary,
                response_validation_enabled=True,
                evidence_validation_enabled=True,
            )
            order = []
            evidence_result = EvidenceCheckResult((), (), "medium", "No claims")
            workflow = PMaxWorkflow(
                credentials=object(),
                engineer=_FakeAgent(
                    "Engineer", EngineerResult(event_log, "code", []), order
                ),
                analyst=_FakeAgent(
                    "Analyst",
                    AnalystResult(
                        [{"type": "text", "content": "Answer"}], "code", [], ()
                    ),
                    order,
                ),
                verifier=FailingVerifier(),
                evidence_checker=_FakeAgent(
                    "EvidenceChecker", evidence_result, order
                ),
                recorder=_NoopRecorder(),
            )

            result = workflow.execute(state)

            self.assertEqual(order[-1], "EvidenceChecker")
            self.assertTrue(state["response_verification"]["unavailable"])
            self.assertIs(result.evidence_check, evidence_result)


if __name__ == "__main__":
    unittest.main()
