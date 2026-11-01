import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from promoai.agents.contracts import AnalystResult, EngineerResult, VerifierResult
from promoai.agents.state import PMaxSession
from promoai.agents.verifier import (
    UNAVAILABLE_MESSAGE,
    VerifierAgent,
    build_verifier_payload,
    parse_verification,
)
from promoai.agents.workflow import PMaxWorkflow
from promoai.general_utils.llm_connection import LLMConnection


def verdict(*, addressed=True, status="supported", evidence_ids=None):
    return {
        "answers_question": addressed,
        "requirements": [{"requirement": "Tiempo promedio de A a B en minutos", "addressed": addressed}],
        "claims": [{
            "claim": "El tiempo promedio de A a B es 15 minutos.",
            "status": status,
            "evidence_ids": ["artifact_0"] if evidence_ids is None else evidence_ids,
            "explanation": "Comparación con el resultado disponible.",
        }],
        "suggestions": [] if addressed else ["Incluir el resultado solicitado."],
    }


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.credentials = LLMConnection("secret-key", "configured-model", "configured-provider", {})
        event_log = pd.DataFrame({
            "case:concept:name": ["1", "1"],
            "concept:name": ["RAW_LOG_ONLY_A", "RAW_LOG_ONLY_B"],
            "time:timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        })
        self.state = PMaxSession(
            "¿Cuál es el tiempo promedio de A a B en minutos?", event_log, self.temp_dir.name
        )
        self.state["saved_artifacts"] = {
            "metric.csv": ("Tiempo promedio de A a B; todos los casos; unidad: minutos", "mean_minutes: 15")
        }
        self.state["final_report"] = [
            {"type": "text", "content": "El tiempo promedio de A a B es 15 minutos."},
            {"type": "artifact", "content": "metric.csv"},
        ]

    def test_complete_incomplete_contradicted_and_unverifiable_cases(self):
        cases = [
            (True, "supported", "mean_minutes: 15", ["artifact_0"]),
            (False, "not_verifiable", "mean_minutes: 15", []),
            (True, "contradicted", "mean_minutes: 12", ["artifact_0"]),
            (True, "not_verifiable", None, []),
        ]
        for addressed, status, preview, ids in cases:
            with self.subTest(addressed=addressed, status=status):
                self.state["saved_artifacts"]["metric.csv"] = (
                    "Tiempo promedio de A a B, minutos", preview
                )
                expected = verdict(addressed=addressed, status=status, evidence_ids=ids)
                if addressed:
                    self.state["final_report"] = [
                        {"type": "text", "content": "El tiempo promedio de A a B es 15 minutos."},
                        {"type": "artifact", "content": "metric.csv"},
                    ]
                else:
                    self.state["final_report"] = [
                        {"type": "text", "content": "Para calcularlo se deben restar las marcas temporales."}
                    ]
                    expected["claims"] = []
                query = Mock(return_value=json.dumps(expected))
                original = copy.deepcopy(self.state["final_report"])

                result = VerifierAgent(self.credentials, query=query).execute(self.state)

                self.assertIsNone(result.error)
                self.assertEqual(result.report, expected)
                self.assertEqual(self.state["final_report"], original)
                query.assert_called_once()
                payload = json.loads(query.call_args.args[0][1]["content"])
                if preview is not None and addressed:
                    self.assertEqual(payload["evidence"][0]["preview"], preview)
                else:
                    self.assertEqual(payload["evidence"], [])

    def test_input_has_only_latest_question_report_and_computed_evidence(self):
        self.state["messages_eng"] = [{"content": "SECRET_ENGINEER_HISTORY"}]
        self.state["messages_ana"] = [{"content": "SECRET_ANALYST_HISTORY"}]
        self.state["previous_code"] = "SECRET_ENGINEER_CODE"
        self.state["context"] = ["SECRET_OLD_CONTEXT"]
        query = Mock(return_value=json.dumps(verdict()))

        result = VerifierAgent(self.credentials, query=query).execute(self.state)

        self.assertIsNotNone(result.report)
        messages = query.call_args.args[0]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {"question", "report", "evidence"})
        for secret in ("SECRET_ENGINEER_HISTORY", "SECRET_ANALYST_HISTORY", "SECRET_ENGINEER_CODE",
                       "SECRET_OLD_CONTEXT", "RAW_LOG_ONLY_A", "secret-key", "metric.csv"):
            self.assertNotIn(secret, messages[1]["content"])
        self.assertEqual(payload["report"][1]["content"], "artifact_0")
        self.assertEqual(query.call_args.kwargs["llm_name"], self.credentials.llm_name)
        self.assertEqual(query.call_args.kwargs["ai_provider"], self.credentials.ai_provider)

    def test_only_artifacts_referenced_by_analyst_are_sent(self):
        self.state["saved_artifacts"] = {
            "unused.csv": ("Unused result", "UNUSED_RESULT_99"),
            "metric.csv": ("Mean A to B, minutes", "mean_minutes: 15"),
            "chart.png": ("Mean chart", {"mean_minutes": 15}),
        }
        self.state["final_report"].extend([
            {"type": "artifact", "content": "chart.png"},
            {"type": "artifact", "content": "metric.csv"},
        ])

        payload = build_verifier_payload(self.state)

        self.assertEqual([item["id"] for item in payload["evidence"]], ["artifact_1", "artifact_2"])
        self.assertEqual(payload["report"][1]["content"], "artifact_1")
        self.assertNotIn("UNUSED_RESULT_99", json.dumps(payload))
        self.assertIn("mean_minutes", payload["evidence"][1]["preview"])

    def test_invalid_output_or_provider_failure_does_not_retry_or_change_report(self):
        bad_boolean = verdict()
        bad_boolean["answers_question"] = "false"
        unknown_evidence = verdict(evidence_ids=["invented"])
        no_evidence = verdict(evidence_ids=[])
        inconsistent = verdict(addressed=False)
        inconsistent["answers_question"] = True
        responses = ["not JSON", "```python\nraise RuntimeError('must not execute')\n```",
                     json.dumps(bad_boolean), json.dumps(unknown_evidence),
                     json.dumps(no_evidence), json.dumps(inconsistent), RuntimeError("provider unavailable")]
        for response in responses:
            with self.subTest(response=str(response)):
                query = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
                original = copy.deepcopy(self.state["final_report"])
                result = VerifierAgent(self.credentials, query=query).execute(self.state)
                query.assert_called_once()
                self.assertIsNone(result.report)
                self.assertEqual(result.error, UNAVAILABLE_MESSAGE)
                self.assertEqual(self.state["final_report"], original)

    def test_json_fence_is_parsed_without_running_code(self):
        result = parse_verification("```json\n" + json.dumps(verdict()) + "\n```", {"artifact_0"})
        self.assertTrue(result["answers_question"])

    def test_workflow_orders_agents_and_persists_review_separately(self):
        order = []
        analyst_report = copy.deepcopy(self.state["final_report"])

        def engineer_execute(state, callback):
            order.append("Engineer")
            return EngineerResult(state["event_log"], "engineer code", [])

        def analyst_execute(state, callback):
            order.append("Analyst")
            return AnalystResult(analyst_report, "analyst code", [], ())

        review = VerifierResult(report=verdict(status="not_verifiable", evidence_ids=[]))

        def verifier_execute(state, callback):
            order.append("Verifier")
            self.assertEqual(state["final_report"], analyst_report)
            return review

        workflow = PMaxWorkflow(
            self.credentials,
            engineer=Mock(execute=engineer_execute),
            analyst=Mock(execute=analyst_execute),
            verifier=Mock(execute=verifier_execute),
        )
        result = workflow.execute(self.state)
        self.assertEqual(order, ["Engineer", "Analyst", "Verifier"])
        self.assertIs(result.verifier, review)
        self.assertEqual(self.state["final_report"], analyst_report)
        paths = list((Path(self.temp_dir.name) / "reports").glob("*.json"))
        saved = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        self.assertIn(analyst_report, saved)
        self.assertIn(review.to_dict(), saved)

    def test_workflow_keeps_report_on_verifier_and_recorder_failure(self):
        original = copy.deepcopy(self.state["final_report"])
        workflow = PMaxWorkflow(
            self.credentials,
            verifier=Mock(execute=Mock(side_effect=RuntimeError("service down"))),
            recorder=Mock(record_verifier=Mock(side_effect=OSError("disk unavailable"))),
        )
        result = workflow.run_verifier(self.state)
        self.assertIsNone(result.report)
        self.assertEqual(self.state["verification_result"]["error"], UNAVAILABLE_MESSAGE)
        self.assertEqual(self.state["final_report"], original)


class VerifierUITests(unittest.TestCase):
    def test_review_renders_collapsed_and_preserves_original_answer(self):
        from streamlit.testing.v1 import AppTest

        result = VerifierResult(
            report=verdict(),
            evidence=({"id": "artifact_0", "description": "Mean A to B, minutes",
                       "preview": "mean_minutes: 15", "truncated": False},),
        ).to_dict()
        script = (
            "from pmax import display_chat_message\n"
            f"result = {result!r}\n"
            "display_chat_message('assistant', ["
            "{'type': 'text', 'content': 'Original answer'}, "
            "{'type': 'verification', 'content': result}])\n"
        )
        app = AppTest.from_string(script).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.expander[0].label, "Verificación de la respuesta")
        self.assertFalse(app.expander[0].proto.expanded)
        self.assertIn("Original answer", [element.value for element in app.markdown])
        self.assertTrue(any("**¿La pregunta fue respondida?** Sí" in element.value for element in app.markdown))

    def test_unavailable_review_renders_without_a_negative_verdict(self):
        from streamlit.testing.v1 import AppTest

        script = (
            "from pmax import display_verification\n"
            "display_verification({'report': None, 'error': 'unavailable'})\n"
        )
        app = AppTest.from_string(script).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.info[0].value, UNAVAILABLE_MESSAGE)
        self.assertFalse(app.markdown)


if __name__ == "__main__":
    unittest.main()
