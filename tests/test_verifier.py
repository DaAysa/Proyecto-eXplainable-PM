import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from promoai.agents.contracts import AnalystResult, EngineerResult, VerifierResult
from promoai.agents.engineer import EngineerAgent
from promoai.agents.state import PMaxSession
from promoai.agents.verifier import (
    MAX_EVIDENCE_CHARS,
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
        self.state["verifier_evidence"] = {
            "run_id": "accepted-run", "request_index": 1,
            "question": self.state.latest_request,
            "artifacts": copy.deepcopy(self.state["saved_artifacts"]),
        }
        self.state["final_report"] = [
            {"type": "text", "content": "El tiempo promedio de A a B es 15 minutos."},
            {"type": "artifact", "content": "metric.csv"},
        ]
        self.state["analyst_report_request"] = 1

    def test_complete_incomplete_contradicted_and_unverifiable_cases(self):
        cases = [
            (True, "supported", "mean_minutes: 15", ["artifact_0"]),
            (False, "not_verifiable", "mean_minutes: 15", []),
            (True, "contradicted", "mean_minutes: 12", ["artifact_0"]),
            (True, "not_verifiable", None, []),
        ]
        for addressed, status, preview, ids in cases:
            with self.subTest(addressed=addressed, status=status):
                self.state["verifier_evidence"]["artifacts"] = (
                    {"metric.csv": ("Tiempo promedio de A a B, minutos", preview)}
                    if preview is not None else {}
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
                if preview is not None:
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
        self.assertEqual(set(payload), {"question", "report", "evidence", "evidence_limits"})
        for secret in ("SECRET_ENGINEER_HISTORY", "SECRET_ANALYST_HISTORY", "SECRET_ENGINEER_CODE",
                       "SECRET_OLD_CONTEXT", "RAW_LOG_ONLY_A", "secret-key", "metric.csv"):
            self.assertNotIn(secret, messages[1]["content"])
        self.assertEqual(payload["report"][1]["content"], "artifact_0")
        self.assertEqual(query.call_args.kwargs["llm_name"], self.credentials.llm_name)
        self.assertEqual(query.call_args.kwargs["ai_provider"], self.credentials.ai_provider)

    def test_unconfirmed_artifacts_keep_original_ids_but_are_not_evidence(self):
        self.state["saved_artifacts"] = {
            "old.csv": ("Old request", "mean_minutes: 99"),
            "failed.csv": ("Failed attempt", "mean_minutes: 22"),
            "metric.csv": ("Accepted attempt", "mean_minutes: 15"),
            "chart.png": ("Accepted chart", {"mean_minutes": 15}),
        }
        self.state["verifier_evidence"]["artifacts"]["chart.png"] = (
            "Accepted chart", {"mean_minutes": 15}
        )
        self.state["final_report"].append({"type": "artifact", "content": "failed.csv"})

        payload = build_verifier_payload(self.state)

        self.assertEqual([item["id"] for item in payload["evidence"]], ["artifact_2", "artifact_3"])
        self.assertEqual(payload["report"][1]["content"], "artifact_2")
        self.assertFalse(payload["report"][2]["accepted_run"])
        self.assertNotIn("mean_minutes: 99", json.dumps(payload))
        self.assertNotIn("mean_minutes: 22", json.dumps(payload))
        self.assertIn("mean_minutes", payload["evidence"][1]["preview"])

    def test_raw_event_log_previews_are_excluded(self):
        raw = self.state["event_log"]
        for data in (raw, raw.to_markdown(index=False), raw.to_dict("records"),
                     {column: raw[column].tolist() for column in raw.columns}):
            with self.subTest(kind=type(data).__name__):
                self.state["verifier_evidence"]["artifacts"] = {"metric.csv": ("Raw data", data)}
                payload = build_verifier_payload(self.state)
                self.assertEqual(payload["evidence"], [])
                self.assertNotIn("RAW_LOG_ONLY_A", json.dumps(payload))

    def test_preview_budget_and_truncation_are_explicit(self):
        artifacts = {f"{index}.csv": ("Computed result", "x" * 8000) for index in range(30)}
        self.state["saved_artifacts"] = artifacts
        self.state["verifier_evidence"]["artifacts"] = artifacts
        payload = build_verifier_payload(self.state)
        self.assertLessEqual(sum(len(item["preview"]) for item in payload["evidence"]), MAX_EVIDENCE_CHARS)
        self.assertTrue(all(item["truncated"] for item in payload["evidence"]))
        self.assertGreater(payload["evidence_limits"]["unavailable_or_omitted"], 0)

    def test_computed_numpy_chart_data_and_series_are_available(self):
        self.state["saved_artifacts"] = {
            "chart.png": ("Mean by group, minutes", {"minutes": np.array([15.0, 12.0])}),
            "metric.csv": ("Computed summary, minutes", pd.Series({"mean_minutes": 15.0})),
        }
        self.state["verifier_evidence"]["artifacts"] = copy.deepcopy(self.state["saved_artifacts"])
        payload = build_verifier_payload(self.state)
        self.assertEqual(len(payload["evidence"]), 2)
        self.assertTrue(any("15.0, 12.0" in item["preview"] for item in payload["evidence"]))
        self.assertTrue(any("mean_minutes" in item["preview"] for item in payload["evidence"]))

    def test_new_request_clears_review_and_cannot_reuse_old_evidence(self):
        self.state["verification_result"] = {"report": verdict()}
        self.state.add_request("Ahora comparar solamente 2025")
        self.assertIsNone(self.state["verification_result"])
        self.assertIsNone(self.state["verifier_evidence"])
        # A new accepted Analyst report alone does not confirm old Engineer data.
        self.state["analyst_report_request"] = 2
        payload = build_verifier_payload(self.state)
        self.assertEqual(payload["evidence"], [])
        self.assertEqual(payload["question"], "Ahora comparar solamente 2025")

    def test_stale_analyst_report_is_not_reviewed(self):
        self.state["analyst_report_request"] = None
        query = Mock()
        result = VerifierAgent(self.credentials, query=query).execute(self.state)
        query.assert_not_called()
        self.assertIsNone(result.report)
        self.assertEqual(result.error, UNAVAILABLE_MESSAGE)

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

    def test_only_successful_engineer_attempt_is_captured(self):
        self.state["saved_artifacts"] = {"old.csv": ("Old", "99")}

        def extract(api, response, args=None):
            if response == "failed":
                self.state.update_artifacts("failed.csv", "Failed", "22")
                raise ValueError("execution failed after saving an artifact")
            self.state.update_artifacts("metric.csv", "Accepted", "15")
            return "code", self.state["event_log"]

        def generate(messages, extraction_function, **kwargs):
            with self.assertRaises(ValueError):
                extraction_function("failed")
            code, log = extraction_function("accepted")
            return code, log, messages

        with patch("promoai.agents.engineer.PM4PYWrapper.code_extraction", new=extract):
            EngineerAgent(self.credentials, result_generator=generate).execute(self.state)

        self.assertEqual(set(self.state["saved_artifacts"]), {"old.csv", "failed.csv", "metric.csv"})
        snapshot = self.state["verifier_evidence"]
        self.assertEqual(set(snapshot["artifacts"]), {"metric.csv"})
        self.assertEqual(snapshot["request_index"], 1)
        self.assertEqual(snapshot["question"], self.state.latest_request)

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
            self.assertEqual(state["analyst_report_request"], 1)
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
