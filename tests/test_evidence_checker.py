import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from promoai.agents.contracts import (
    AnalystResult,
    EngineerResult,
    EvidenceCheckResult,
    EvidenceClaim,
    QuestionPartAssessment,
    VerificationResult,
)
from promoai.agents.analyst import AnalystAgent
from promoai.agents.engineer import EngineerAgent
from promoai.agents.evidence_checker import EvidenceCheckerAgent
from promoai.agents.evidence_sessions import (
    check_saved_session,
    load_saved_session_states,
)
from promoai.agents.state import PMaxSession
from promoai.agents.persistence import AgentRunRecorder
from promoai.agents.workflow import PMaxWorkflow
from promoai.general_utils.llm_connection import LLMConnection
from promoai.general_utils.llm_connection import query_llm
from promoai.general_utils.ai_providers import AIProviders


FIXTURE = Path(__file__).parent / "fixtures" / "legacy_frequency_session"


class _ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.responses.pop(0)


def _frequency_plan():
    return json.dumps(
        {
            "question_parts": [
                {
                    "id": "part_1",
                    "text": "Case frequency and percentage for each activity",
                    "data_type": "frequency",
                    "candidate_source_ids": ["artifact_0"],
                }
            ],
            "claims": [
                {
                    "id": "claim_1",
                    "text": "A_Cancelled appears in 4,778 cases, representing 77.40%.",
                    "question_part_ids": ["part_1"],
                    "verifiability": "verifiable",
                }
            ],
        }
    )


def _frequency_check(percent="77.40%"):
    return json.dumps(
        {
            "part_id": "part_1",
            "status": "answered",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "status": "supported",
                    "explanation": "Both values are in the activity table.",
                    "evidence": [
                        {
                            "source_id": "artifact_0",
                            "row": 10,
                            "column": "distinct_cases",
                            "value": 4778,
                            "claim_value": "4,778",
                        },
                        {
                            "source_id": "artifact_0",
                            "row": 10,
                            "column": "percentage_of_total_cases",
                            "value": 77.40158755872348,
                            "claim_value": percent,
                        },
                    ],
                }
            ],
        }
    )


class EvidenceCheckerTests(unittest.TestCase):
    def _credentials(self):
        return LLMConnection("key", "model", "provider", {})

    def test_legacy_session_rebases_paths_and_confirms_numeric_values(self):
        states = load_saved_session_states(str(FIXTURE))
        state = states[1]
        snapshot = state["analyst_evidence_history"][-1]
        self.assertEqual(snapshot["artifacts"][0]["id"], "artifact_0")
        self.assertEqual(snapshot["context"], [])
        self.assertTrue(Path(snapshot["artifacts"][0]["path"]).is_file())
        snapshot["artifacts"].append(
            {
                "id": "artifact_1",
                "path": str(FIXTURE / "dataframes" / "missing.csv"),
                "description": "Unrelated artifact",
                "type": "dataframe",
                "preview": "UNRELATED_PREVIEW_MUST_NOT_REACH_PART_CHECK",
                "sent_this_request": True,
                "evidence_eligible": True,
            }
        )

        llm = _ScriptedLLM(_frequency_plan(), _frequency_check())
        events = []
        result = EvidenceCheckerAgent(
            self._credentials(), response_generator=llm
        ).execute(state, events.append)

        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.claims[0].status, "supported")
        self.assertTrue(
            all(item.numeric_check["passed"] for item in result.claims[0].evidence)
        )
        self.assertTrue(all(call[1]["llm_args"]["temperature"] == 0 for call in llm.calls))
        self.assertNotIn(
            "UNRELATED_PREVIEW_MUST_NOT_REACH_PART_CHECK",
            llm.calls[1][0][-1]["content"],
        )
        self.assertEqual(
            [event["agent"] for event in events if event["event"] == "request_started"],
            ["EvidenceChecker", "EvidenceChecker"],
        )

    def test_numeric_mismatch_overrides_llm_and_lowers_confidence(self):
        state = load_saved_session_states(str(FIXTURE))[1]
        llm = _ScriptedLLM(_frequency_plan(), _frequency_check("80%"))

        result = EvidenceCheckerAgent(
            self._credentials(), response_generator=llm
        ).execute(state)

        self.assertEqual(result.claims[0].status, "contradicted")
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.claims[0].evidence[1].numeric_check["passed"])

    def test_context_is_valid_evidence_without_a_table_location(self):
        state = {
            "user_request": ["What is the conformance fitness?"],
            "final_report": [
                {"type": "text", "content": "The conformance fitness is 0.91."}
            ],
            "artifact_session_dir": str(FIXTURE),
            "analyst_evidence_history": [
                {
                    "request_index": 1,
                    "artifacts": [],
                    "context": ["Conformance Checking - Alignments: Fitness: 0.91"],
                    "log_abstraction": "Total Cases: 20",
                }
            ],
        }
        plan = json.dumps(
            {
                "question_parts": [
                    {
                        "id": "part_1",
                        "text": "Conformance fitness",
                        "data_type": "conformance",
                        "candidate_source_ids": ["context_1_0"],
                    }
                ],
                "claims": [
                    {
                        "id": "claim_1",
                        "text": "The conformance fitness is 0.91.",
                        "question_part_ids": ["part_1"],
                        "verifiability": "verifiable",
                    }
                ],
            }
        )
        checked = json.dumps(
            {
                "part_id": "part_1",
                "status": "answered",
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "status": "supported",
                        "explanation": "The context contains the metric.",
                        "evidence": [
                            {
                                "source_id": "context_1_0",
                                "value": 0.91,
                                "claim_value": "0.91",
                            }
                        ],
                    }
                ],
            }
        )
        result = EvidenceCheckerAgent(
            self._credentials(), response_generator=_ScriptedLLM(plan, checked)
        ).execute(state)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.claims[0].evidence[0].source, "context")

    def test_invalid_planning_json_is_rejected(self):
        state = load_saved_session_states(str(FIXTURE))[1]
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            EvidenceCheckerAgent(
                self._credentials(), response_generator=_ScriptedLLM("not-json")
            ).execute(state)

    def test_recommendations_are_non_verifiable_and_do_not_fail_coverage(self):
        state = {
            "user_request": ["What should the team improve first?"],
            "final_report": [
                {"type": "text", "content": "The team should simplify handoffs."}
            ],
            "artifact_session_dir": str(FIXTURE),
            "analyst_evidence_history": [
                {
                    "request_index": 1,
                    "artifacts": [],
                    "context": [],
                    "log_abstraction": None,
                }
            ],
        }
        plan = json.dumps(
            {
                "question_parts": [
                    {
                        "id": "part_1",
                        "text": "Improvement recommendation",
                        "data_type": None,
                        "candidate_source_ids": [],
                    }
                ],
                "claims": [
                    {
                        "id": "claim_1",
                        "text": "The team should simplify handoffs.",
                        "question_part_ids": ["part_1"],
                        "verifiability": "non_verifiable",
                    }
                ],
            }
        )
        checked = json.dumps(
            {"part_id": "part_1", "status": "answered", "claims": []}
        )
        result = EvidenceCheckerAgent(
            self._credentials(), response_generator=_ScriptedLLM(plan, checked)
        ).execute(state)
        self.assertEqual(result.claims[0].status, "non_verifiable")
        self.assertEqual(result.question_parts[0].status, "answered")
        self.assertEqual(result.confidence, "medium")

    def test_saved_session_api_can_run_without_persisting(self):
        llm = _ScriptedLLM(_frequency_plan(), _frequency_check())
        results = check_saved_session(
            str(FIXTURE),
            self._credentials(),
            request_index=1,
            persist=False,
            response_generator=llm,
        )
        self.assertEqual(results[1].confidence, "high")

    def test_temperature_is_forwarded_to_native_provider_clients(self):
        targets = {
            AIProviders.GOOGLE.value: "generate_response_with_history_google",
            AIProviders.ANTHROPIC.value: "generate_response_with_history_anthropic",
            AIProviders.COHERE.value: "generate_response_with_history_cohere",
        }
        for provider, target in targets.items():
            with self.subTest(provider=provider), patch(
                f"promoai.general_utils.llm_connection.{target}",
                return_value="{}",
            ) as mocked:
                query_llm(
                    [{"role": "user", "content": "check"}],
                    "key",
                    "model",
                    provider,
                    {"temperature": 0, "artifact_session_dir": str(FIXTURE)},
                )
                self.assertEqual(mocked.call_args.kwargs["temperature"], 0)

    def test_engineer_tracks_failed_artifacts_without_removing_them(self):
        with tempfile.TemporaryDirectory() as task_dir:
            event_log = pd.DataFrame(
                {
                    "case:concept:name": ["1", "2"],
                    "concept:name": ["Start", "Start"],
                    "time:timestamp": pd.to_datetime(
                        ["2026-01-01", "2026-01-02"]
                    ),
                }
            )
            state = PMaxSession("analyze", event_log, task_dir)

            def generate(messages, extraction_function, **kwargs):
                failed = """```python
api.save_dataframe(pd.DataFrame({'metric': [123]}), 'Failed attempt metric')
raise ValueError('retry me')
final_event_log = api.event_log
```"""
                with self.assertRaises(Exception):
                    extraction_function(failed, False)
                successful = """```python
api.save_dataframe(pd.DataFrame({'metric': [456]}), 'Accepted metric')
final_event_log = api.event_log
```"""
                code, result = extraction_function(successful, False)
                return code, result, messages

            result = EngineerAgent(
                self._credentials(), result_generator=generate
            ).execute(state)
            state.apply_engineer_result(result)

            self.assertEqual(len(state["saved_artifacts"]), 2)
            self.assertEqual(len(result.failed_artifact_paths), 1)
            self.assertEqual(len(result.accepted_artifact_paths), 1)
            self.assertIn(
                result.failed_artifact_paths[0],
                state["ineligible_evidence_artifacts"],
            )

    def test_analyst_snapshot_preserves_context_and_excludes_failed_evidence(self):
        with tempfile.TemporaryDirectory() as task_dir:
            event_log = pd.DataFrame(
                {
                    "case:concept:name": ["1"],
                    "concept:name": ["Start"],
                    "time:timestamp": pd.to_datetime(["2026-01-01"]),
                }
            )
            state = PMaxSession("analyze", event_log, task_dir)
            failed_path = str(Path(task_dir) / "failed.csv")
            accepted_path = str(Path(task_dir) / "accepted.csv")
            state["saved_artifacts"] = {
                failed_path: ("Failed", "failed preview"),
                accepted_path: ("Accepted", "accepted preview"),
            }
            state["ineligible_evidence_artifacts"] = [failed_path]
            state["artifact_provenance"] = {failed_path: 1, accepted_path: 1}
            state.add_context("Fitness: 0.91")

            def generate(messages, **kwargs):
                return (
                    "final_report = []",
                    [{"type": "text", "content": "Done"}],
                    messages,
                )

            result = AnalystAgent(
                self._credentials(), result_generator=generate
            ).execute(state)
            state.apply_analyst_result(result)
            AgentRunRecorder().record_analyst(state, result)

            self.assertEqual(result.evidence_snapshot["context"], ["Fitness: 0.91"])
            self.assertEqual(state["context"], [])
            self.assertFalse(result.evidence_snapshot["artifacts"][0]["evidence_eligible"])
            sources = EvidenceCheckerAgent._collect_sources(
                state["analyst_evidence_history"]
            )
            self.assertNotIn("artifact_0", sources)
            self.assertIn("artifact_1", sources)
            self.assertEqual(
                len(list((Path(task_dir) / "evidence").glob("*.json"))), 1
            )


class _OrderedAgent:
    def __init__(self, name, order, result=None, error=None):
        self.name = name
        self.order = order
        self.result = result
        self.error = error

    def execute(self, state, progress_callback=None):
        self.order.append(self.name)
        if self.error:
            raise self.error
        return self.result


class _Recorder:
    def record_engineer(self, state, result):
        pass

    def record_analyst(self, state, result):
        pass

    def record_verifier(self, state, result):
        pass

    def record_evidence_check(self, state, result):
        pass


class EvidenceWorkflowTests(unittest.TestCase):
    def test_verifier_and_evidence_toggles_are_independent(self):
        combinations = {
            (False, False): ["Engineer", "Analyst"],
            (True, False): ["Engineer", "Analyst", "Verifier"],
            (False, True): ["Engineer", "Analyst", "EvidenceChecker"],
            (True, True): [
                "Engineer",
                "Analyst",
                "Verifier",
                "EvidenceChecker",
            ],
        }
        for (verifier_enabled, evidence_enabled), expected_order in combinations.items():
            with self.subTest(
                verifier=verifier_enabled, evidence=evidence_enabled
            ), tempfile.TemporaryDirectory() as task_dir:
                event_log = pd.DataFrame(
                    {
                        "case:concept:name": ["1"],
                        "concept:name": ["Start"],
                        "time:timestamp": pd.to_datetime(["2026-01-01"]),
                    }
                )
                state = PMaxSession(
                    "analyze",
                    event_log,
                    task_dir,
                    response_validation_enabled=verifier_enabled,
                    evidence_check_enabled=evidence_enabled,
                )
                order = []
                analyst = AnalystResult(
                    [{"type": "text", "content": "Done"}],
                    "report",
                    [],
                    (),
                    {
                        "request_index": 1,
                        "artifacts": [],
                        "context": [],
                        "log_abstraction": state["log_abstraction"],
                    },
                )
                evidence = EvidenceCheckResult(
                    claims=(),
                    question_parts=(
                        QuestionPartAssessment(
                            "part_1", "Analyze", None, "answered"
                        ),
                    ),
                    confidence="medium",
                    confidence_explanation="No factual claim.",
                )
                workflow = PMaxWorkflow(
                    credentials=object(),
                    engineer=_OrderedAgent(
                        "Engineer", order, EngineerResult(event_log, "code", [])
                    ),
                    analyst=_OrderedAgent("Analyst", order, analyst),
                    verifier=_OrderedAgent(
                        "Verifier",
                        order,
                        VerificationResult(True, "Complete"),
                    ),
                    evidence_checker=_OrderedAgent(
                        "EvidenceChecker", order, evidence
                    ),
                    recorder=_Recorder(),
                )
                workflow.execute(state)
                self.assertEqual(order, expected_order)

    def test_verifier_failure_does_not_prevent_evidence_checker(self):
        with tempfile.TemporaryDirectory() as task_dir:
            event_log = pd.DataFrame(
                {
                    "case:concept:name": ["1"],
                    "concept:name": ["Start"],
                    "time:timestamp": pd.to_datetime(["2026-01-01"]),
                }
            )
            state = PMaxSession(
                "analyze",
                event_log,
                task_dir,
                response_validation_enabled=True,
                evidence_check_enabled=True,
            )
            order = []
            engineer = EngineerResult(event_log, "code", [])
            snapshot = {
                "request_index": 1,
                "artifacts": [],
                "context": [],
                "log_abstraction": state["log_abstraction"],
            }
            analyst = AnalystResult(
                [{"type": "text", "content": "Done"}],
                "report",
                [],
                (),
                snapshot,
            )
            evidence = EvidenceCheckResult(
                claims=(
                    EvidenceClaim(
                        "claim_1", "Done", ("part_1",), "non_verifiable", "Opinion"
                    ),
                ),
                question_parts=(
                    QuestionPartAssessment(
                        "part_1", "Analyze", None, "answered"
                    ),
                ),
                confidence="medium",
                confidence_explanation="No factual claim.",
            )
            workflow = PMaxWorkflow(
                credentials=object(),
                engineer=_OrderedAgent("Engineer", order, engineer),
                analyst=_OrderedAgent("Analyst", order, analyst),
                verifier=_OrderedAgent(
                    "Verifier", order, error=RuntimeError("verifier failed")
                ),
                evidence_checker=_OrderedAgent(
                    "EvidenceChecker", order, evidence
                ),
                recorder=_Recorder(),
            )

            result = workflow.execute(state)

            self.assertEqual(
                order, ["Engineer", "Analyst", "Verifier", "EvidenceChecker"]
            )
            self.assertTrue(state["response_verification"]["unavailable"])
            self.assertIs(result.evidence_check, evidence)
            self.assertEqual(state["final_report"], analyst.report)


if __name__ == "__main__":
    unittest.main()
