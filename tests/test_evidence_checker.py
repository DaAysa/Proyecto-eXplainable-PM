import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from promoai.agents.analyst import AnalystAgent
from promoai.agents.contracts import AnalystResult, EngineerResult, EvidenceCheckResult, VerificationResult
from promoai.agents.engineer import EngineerAgent
from promoai.agents.evidence import evidence_sources
from promoai.agents.evidence_checker import (
    EvidenceCheckerAgent,
)
from promoai.agents.evidence_replay import check_saved_session, load_evidence_session
from promoai.agents.persistence import AgentRunRecorder
from promoai.agents.state import PMaxSession
from promoai.agents.workflow import PMaxWorkflow
from promoai.general_utils.llm_connection import LLMConnection, query_llm


CREDENTIALS = LLMConnection("unused", "model", "provider", {"temperature": 0.7})


def location(source_id, row, column, value, unit="number"):
    return {"source_id": source_id, "row": row, "column": column, "value": value, "evidence_unit": unit}


def assessment(claims, status="answered", available=None):
    return {"claims": claims, "coverage": {"status": status, "explanation": "Coverage assessment."},
            "available_evidence": available or []}


def judged(claim_id, citation=None, status="supported"):
    return {"id": claim_id, "status": status, "explanation": "Evidence assessment.", "location": citation}


class EvidenceCheckerTests(unittest.TestCase):
    def session(self, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        event_log = pd.DataFrame({"case:concept:name": ["1", "2"], "concept:name": ["A", "B"]})
        return PMaxSession("Compare durations and acceptance and propose recommendations", event_log, directory.name, **kwargs)

    def saved_q15_session(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = Path(directory.name)
        (session / "dataframes").mkdir()
        (session / "reports").mkdir()

        table_name = (
            "017_comparison_of_case_duration_and_o_accepted_proportion_between_"
            "new_credit_and_limit_raise.csv"
        )
        pd.DataFrame({
            "case_ApplicationType": ["Limit raise", "New credit"],
            "total_cases": [248, 5925],
            "mean_duration_hours": [465.2103157594086, 634.0673159951712],
            "median_duration_hours": [442.34724958333334, 741.9690677777777],
            "std_duration_hours": [256.73724513898793, 235.1399129218262],
            "accepted_count": [127, 1268],
            "accepted_proportion": [0.5120967741935484, 0.2140084388185654],
        }).to_csv(session / "dataframes" / table_name, index=False)

        report = [
            {
                "type": "text",
                "content": (
                    "### Summary of Calculated Findings\n\n"
                    "New credit applications exhibit a mean duration of **634.07 hours**. "
                    "Limit raise applications achieve an acceptance proportion of **51.21%**."
                ),
            },
            {
                "type": "text",
                "content": (
                    "Implement automated pre-screening for New credit applications to reduce "
                    "bottlenecks in early processing stages and shorten overall lead time."
                ),
            },
        ]
        report_name = "027_assistant_report_request_15.json"
        (session / "reports" / report_name).write_text(
            json.dumps(report), encoding="utf-8"
        )
        question = (
            "How do observed case duration and the proportion of cases containing O_Accepted "
            "differ between New credit and Limit raise, and what two recommendations could be "
            "proposed, distinguishing calculated findings from suggestions that require further "
            "evaluation?"
        )
        manifest = [
            {
                "type": "dataframe",
                "file_path": f"dataframes/{table_name}",
                "description": (
                    "Comparison of case duration and O_Accepted proportion between New credit "
                    "and Limit raise application types."
                ),
                "data_preview": (
                    "case_ApplicationType mean_duration_hours accepted_proportion\n"
                    "Limit raise 465.2103 0.5121\nNew credit 634.0673 0.2140"
                ),
            },
            {
                "type": "generated_code",
                "file_path": "code/027_engineer_step_15.py",
                "description": "Engineer generated code for request 15.",
                "extra": {
                    "role": "engineer",
                    "request_index": 15,
                    "user_request": question,
                },
            },
            {
                "type": "report",
                "file_path": f"reports/{report_name}",
                "description": "Structured analyst report for request 15.",
            },
        ]
        (session / "manifest.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in manifest), encoding="utf-8"
        )
        return session

    def run_scripted(self, state, responses):
        events = []
        respond = Mock(side_effect=[json.dumps(response) if isinstance(response, dict) else response for response in responses])
        result = EvidenceCheckerAgent(CREDENTIALS, respond).execute(state, events.append)
        for call in respond.call_args_list:
            self.assertEqual(call.kwargs["llm_args"]["temperature"], 0)
        self.assertEqual(CREDENTIALS.args, {"temperature": 0.7})
        self.assertTrue(all(event["agent"] == "EvidenceChecker" for event in events))
        return result, respond

    def temp_plan(self, state, localized=False):
        sources = evidence_sources(state)
        table = next(source for source in sources if Path(source.get("file_path", "")).name.startswith("017_"))
        key = table["source_id"]
        if localized:
            state["final_report"] = [{"type": "text", "content": "New credit: media 634,07 h. Limit raise: 51,21 % aceptados. Recomiendo automatizar la revisión."}]
            snippets = [("New credit: media 634,07 h", "634,07"), ("Limit raise: 51,21 % aceptados", "51,21 %")]
            recommendation = "Recomiendo automatizar la revisión."
        else:
            snippets = [("**634.07 hours**", "634.07"), ("**51.21%**", "51.21%")]
            recommendation = "Implement automated pre-screening for New credit applications to reduce bottlenecks in early processing stages and shorten overall lead time."
        plan = {
            "parts": [
                {"id": "duration", "question": "Compare durations and propose recommendations", "data_types": ["time"], "source_ids": [key]},
                {"id": "acceptance", "question": "Compare acceptance proportions", "data_types": ["frequency"], "source_ids": [key]},
            ],
            "claims": [
                {"id": "mean", "part_id": "duration", "text": snippets[0][0], "kind": "fact", "claimed_value": snippets[0][1]},
                {"id": "rate", "part_id": "acceptance", "text": snippets[1][0], "kind": "fact", "claimed_value": snippets[1][1]},
                {"id": "advice", "part_id": "duration", "text": recommendation, "kind": "recommendation", "claimed_value": None},
            ],
        }
        return plan, key

    def temp_responses(self, plan, key):
        return [plan,
            assessment([judged("mean", location(key, 2, "mean_duration_hours", "634.0673")), judged("advice", status="non_verifiable")]),
            assessment([judged("rate", location(key, 1, "accepted_proportion", "0.5120967741935484", "fraction"))])]

    def test_temp_request_15_acceptance_and_localized_numbers(self):
        for localized in (False, True):
            with self.subTest(localized=localized):
                state = load_evidence_session(self.saved_q15_session(), 15)
                plan, key = self.temp_plan(state, localized)
                result, respond = self.run_scripted(state, self.temp_responses(plan, key))
                claims = {claim["id"]: claim for claim in result.claims}
                for claim_id in ("mean", "rate"):
                    self.assertEqual(claims[claim_id]["status"], "supported")
                    self.assertNotIn("numeric_check", claims[claim_id])
                    self.assertTrue(Path(claims[claim_id]["location"]["file_path"]).name.startswith("017_"))
                self.assertEqual(claims["advice"]["status"], "non_verifiable")
                self.assertEqual([part["status"] for part in result.question_parts], ["answered", "answered"])
                self.assertEqual(result.confidence, "high")
                self.assertEqual(result.provenance, "inferred_order_no_context")
                # Assessment calls only receive selected evidence for their part.
                for call in respond.call_args_list[1:]:
                    payload = json.loads(call.args[0][1]["content"])
                    self.assertEqual([source["source_id"] for source in payload["evidence"]], [key])
                self.assertNotIn("preview", json.loads(respond.call_args_list[0].args[0][1]["content"])["source_catalog"][0])

    def test_agents_request_english_output_even_with_non_english_input(self):
        from promoai.agents.prompt_builders import AnalystPromptBuilder, EngineerPromptBuilder
        from promoai.agents.verifier import VerifierAgent

        state = self.session()
        state["user_request"][-1] = "Comparar los casos y explicar los resultados."
        state["final_report"] = [{"type": "text", "content": "Datos del analisis anterior."}]
        prompts = [
            EngineerPromptBuilder().build(state, "API description")[0]["content"],
            AnalystPromptBuilder._initial_message(state, "Contexto anterior"),
            AnalystPromptBuilder._follow_up_message(state["user_request"][-1], "Contexto anterior"),
            VerifierAgent._build_messages(state["user_request"][-1], state["final_report"])[0]["content"],
            EvidenceCheckerAgent._plan_prompt(),
            EvidenceCheckerAgent._assessment_prompt(),
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt[:60]):
                self.assertIn("in English", prompt)
                self.assertNotIn("Use the user's language", prompt)
                self.assertNotIn("same language as the user's request", prompt)

        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state, localized=True)
        result, _ = self.run_scripted(state, self.temp_responses(plan, key))
        self.assertEqual(result.claims[0]["text"], plan["claims"][0]["text"])
        self.assertEqual(result.claims[0]["status"], "supported")
        self.assertEqual(result.confidence, "high")
        self.assertTrue(result.confidence_rule.startswith("High when"))

    def test_legacy_replay_excludes_failed_attempt_and_future_sources(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = Path(directory.name)
        for folder in ("requests", "dataframes", "code", "reports", "llm_requests"):
            (session / folder).mkdir()
        (session / "requests/001_request.md").write_text("Question one", encoding="utf-8")
        (session / "requests/002_request.md").write_text("Question two", encoding="utf-8")
        pd.DataFrame({"value": [13]}).to_csv(session / "dataframes/013_failed.csv", index=False)
        pd.DataFrame({"value": [14]}).to_csv(session / "dataframes/014_success.csv", index=False)
        pd.DataFrame({"value": [19]}).to_csv(session / "dataframes/019_future.csv", index=False)
        (session / "code/001_engineer.py").write_text("final_event_log = api.event_log", encoding="utf-8")
        (session / "reports/001_assistant_report_request_1.json").write_text(
            '[{"type":"text","content":"Value 14"}]', encoding="utf-8"
        )
        retry_trace = {
            "request": {
                "config": {"system_instruction": "You are a Process Mining Data Engineer."},
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "Executing your code led to an error!"}],
                }],
            }
        }
        (session / "llm_requests/001_retry.json").write_text(
            json.dumps(retry_trace), encoding="utf-8"
        )
        entries = [
            ("user_request", "requests/001_request.md", {}),
            ("dataframe", "dataframes/013_failed.csv", {}),
            ("llm_request", "llm_requests/001_retry.json", {}),
            ("dataframe", "dataframes/014_success.csv", {}),
            ("generated_code", "code/001_engineer.py", {
                "role": "engineer", "request_index": 1, "user_request": "Question one",
            }),
            ("report", "reports/001_assistant_report_request_1.json", {}),
            ("user_request", "requests/002_request.md", {}),
            ("dataframe", "dataframes/019_future.csv", {}),
        ]
        manifest = []
        for entry_type, file_path, extra in entries:
            manifest.append(json.dumps({
                "type": entry_type,
                "file_path": file_path,
                "description": Path(file_path).stem,
                "extra": extra,
            }))
        (session / "manifest.jsonl").write_text("\n".join(manifest), encoding="utf-8")

        state = load_evidence_session(session, 1)
        paths = [Path(source.get("file_path", "")).name for source in evidence_sources(state)]
        self.assertFalse(any(path.startswith("013_") for path in paths))
        self.assertFalse(any(path.startswith("019_") for path in paths))
        self.assertTrue(any(path.startswith("014_") for path in paths))
        snapshot = state["analyst_evidence"][0]
        self.assertFalse(snapshot["artifacts"][0]["eligible"])
        self.assertTrue(snapshot["artifacts"][1]["eligible"])

    def test_llm_verdicts_and_cited_values_are_preserved_without_python_comparison(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        for status in ("supported", "contradicted", "unsupported", "non_verifiable"):
            with self.subTest(status=status):
                responses = self.temp_responses(plan, key)
                # Both the claim and the supplied citation differ from the CSV.
                # Comparing them and issuing a verdict is exclusively the LLM's job.
                verdict = responses[1]["claims"][0]
                verdict.update(status=status, explanation="The model's comparison result.")
                verdict["location"] = location(key, 1, "mean_duration_hours", "999")
                result, _ = self.run_scripted(state, responses)
                self.assertEqual(result.claims[0]["status"], status)
                self.assertEqual(result.claims[0]["explanation"], verdict["explanation"])
                self.assertEqual(result.claims[0]["location"]["value"], "999")
                self.assertNotIn("numeric_check", result.claims[0])

    def test_invalid_reference_is_corrected_without_reclassifying_a_claim(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        responses = self.temp_responses(plan, key)
        responses[1]["claims"][0]["location"]["row"] = 100
        corrected = copy.deepcopy(responses[1])
        corrected["claims"][0]["location"]["row"] = 2
        responses.insert(2, corrected)
        result, respond = self.run_scripted(state, responses)
        self.assertEqual(result.claims[0]["location"]["row"], 2)
        self.assertEqual(responses[1]["claims"][0]["status"], "supported")
        correction = json.loads(respond.call_args_list[2].args[0][-1]["content"])
        self.assertIn("row 100 is outside the valid range 1-2", correction["validation_error"])
        self.assertEqual(correction["reference_constraints"][0]["valid_rows"], "1-2")

    def test_invalid_reference_fails_after_three_assessment_attempts(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        invalid = self.temp_responses(plan, key)[1]
        invalid["claims"][0]["location"]["row"] = 100
        responses = [plan, invalid, copy.deepcopy(invalid), copy.deepcopy(invalid)]
        with self.assertRaisesRegex(ValueError, "invalid evidence reference"):
            self.run_scripted(state, responses)
        self.assertEqual(invalid["claims"][0]["status"], "supported")

    def test_non_verbatim_claim_is_corrected_before_assessment(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        invalid = copy.deepcopy(plan)
        invalid["claims"][0]["text"] = "New credit mean duration is 634.07 hours."
        responses = [invalid, *self.temp_responses(plan, key)]
        result, respond = self.run_scripted(state, responses)
        self.assertEqual(result.claims[0]["text"], plan["claims"][0]["text"])
        correction = json.loads(respond.call_args_list[1].args[0][-1]["content"])
        self.assertIn("exact report excerpt", correction["validation_error"])
        self.assertIn("contiguous exact excerpt", correction["instruction"])

    def test_csv_lookup_finds_row_beyond_analyst_summary(self):
        state = self.session()
        path = Path(state["artifact_session_dir"]) / "full.csv"
        pd.DataFrame({"group": [f"g{i}" for i in range(70)], "hours": range(70)}).to_csv(path, index=False)
        state["saved_artifacts"] = {str(path): ("Duration per group", "Only a statistical summary is available.")}
        state["analyst_evidence"] = [{"request_index": 1, "user_request": "duration", "context": [], "artifacts": [
            {"id": "artifact_0", "file_path": str(path), "description": "Duration per group", "preview": "Only summary", "eligible": True}]}]
        state["final_report"] = [{"type": "text", "content": "Group g67 has 67 hours."}]
        key = "q1/artifact_0"
        plan = {"parts": [{"id": "p", "question": "Duration", "data_types": ["time"], "source_ids": [key]}],
                "claims": [{"id": "c", "part_id": "p", "text": "Group g67 has 67 hours.", "kind": "fact", "claimed_value": "67"}]}
        lookup = {"lookups": [{"source_id": key, "filters": {"group": "g67"}, "columns": ["group", "hours"]}]}
        result, respond = self.run_scripted(state, [plan, lookup, assessment([judged("c", location(key, 68, "hours", "67"))])])
        self.assertEqual(result.confidence, "high")
        rows = json.loads(respond.call_args_list[-1].args[0][-1]["content"])["lookup_results"][0]
        self.assertEqual(rows["rows"], [{"row": 68, "values": {"group": "g67", "hours": "67"}}])
        self.assertIn("Compare numbers, units, percentages and rounding yourself", respond.call_args_list[-1].args[0][0]["content"])
        self.assertNotIn("numeric_check", result.claims[0])

    def test_unanswered_part_probable_cause_uses_valid_evidence(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        for available, expected in (([location(key, 2, "mean_duration_hours", "634.0673159951712")], "Analyst"), ([], "Engineer")):
            with self.subTest(expected=expected):
                responses = self.temp_responses(plan, key)
                responses[1]["coverage"]["status"] = "unanswered"
                responses[1]["available_evidence"] = available
                result, _ = self.run_scripted(state, responses)
                self.assertIn(expected, result.question_parts[0]["probable_cause"])
                self.assertEqual(result.confidence, "low")

    def test_preserves_context_ids_and_prior_artifacts_after_flush_and_replay(self):
        state = self.session()
        path = Path(state["artifact_session_dir"]) / "dataframes" / "first.csv"
        path.parent.mkdir()
        pd.DataFrame({"hours": [2]}).to_csv(path, index=False)
        state["saved_artifacts"] = {str(path): ("Duration", "hours = 2")}
        state.add_context("Fitness: 0.92\nTotal cases: 2")
        initial_abstraction = state["log_abstraction"]

        def generate(messages, **kwargs):
            return "final_report=[]", [{"type": "text", "content": "Fitness is 0.92."}], messages

        workflow = PMaxWorkflow(CREDENTIALS, analyst=AnalystAgent(CREDENTIALS, result_generator=generate))
        workflow.run_analyst(state)
        self.assertEqual(state["context"], [])
        snapshot = state["analyst_evidence"][0]
        self.assertEqual(snapshot["context"], ["Fitness: 0.92\nTotal cases: 2"])
        self.assertEqual(snapshot["log_abstraction"], initial_abstraction)
        self.assertEqual(snapshot["artifacts"][0]["id"], "artifact_0")
        state.add_request("And precision?")
        state.add_context("Precision: 0.85")
        workflow.run_analyst(state)
        self.assertEqual(state["analyst_evidence"][1]["artifacts"], [])
        self.assertIsNone(state["analyst_evidence"][1]["log_abstraction"])
        sources = evidence_sources(state)
        self.assertEqual(sources[0]["source_id"], "q2/context_0")
        self.assertIn("q1/artifact_0", {source["source_id"] for source in sources})
        restored = load_evidence_session(state["artifact_session_dir"], 2)
        self.assertEqual(restored["analyst_evidence"], state["analyst_evidence"])
        self.assertEqual(restored["user_request"][-1], "And precision?")

        plan = {"parts": [{"id": "p", "question": "Fitness", "data_types": ["conformance"], "source_ids": ["q1/context_0"]}],
                "claims": [{"id": "c", "part_id": "p", "text": "Fitness is 0.92.", "kind": "fact", "claimed_value": "0.92"}]}
        result, _ = self.run_scripted(restored, [plan, assessment([judged("c", location("q1/context_0", 1, None, "0.92"))])])
        self.assertEqual(result.claims[0]["status"], "supported")
        self.assertEqual(result.confidence, "high")

    def test_engineer_failed_attempt_is_excluded_without_altering_analyst_input(self):
        state = self.session()
        failed_code = '''```python
api.save_dataframe(pd.DataFrame({'hours': [999]}), 'Failed duration')
raise ValueError('retry')
```'''
        valid_code = '''```python
api.save_dataframe(pd.DataFrame({'hours': [2]}), 'Successful duration')
final_event_log = api.event_log
```'''
        with patch("promoai.general_utils.llm_connection.query_llm", side_effect=[failed_code, valid_code]):
            EngineerAgent(CREDENTIALS).execute(state)
        self.assertEqual(len(state["saved_artifacts"]), 2)
        prompts = []

        def generate(messages, **kwargs):
            prompts.extend(messages)
            return "final_report=[]", [], messages

        PMaxWorkflow(CREDENTIALS, analyst=AnalystAgent(CREDENTIALS, result_generator=generate)).run_analyst(state)
        artifacts = state["analyst_evidence"][0]["artifacts"]
        self.assertEqual([artifact["eligible"] for artifact in artifacts], [False, True])
        self.assertEqual([artifact["id"] for artifact in artifacts], ["artifact_0", "artifact_1"])
        self.assertIn("Failed duration", prompts[-1]["content"])
        sources = evidence_sources(state)
        self.assertNotIn("q1/artifact_0", {source["source_id"] for source in sources})
        self.assertIn("q1/artifact_1", {source["source_id"] for source in sources})
        self.assertFalse(any("Failed duration" in source["preview"] for source in sources))

    def test_json_and_schema_validation_rejects_bad_model_output(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        bad_source = copy.deepcopy(plan)
        bad_source["parts"][0]["source_ids"] = ["q999/artifact_0"]
        for response in ("not json", "[]"):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    self.run_scripted(state, [response])
        for response in ({"parts": []}, bad_source):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    self.run_scripted(
                        state,
                        [response, copy.deepcopy(response), copy.deepcopy(response)],
                    )
        with self.assertRaises(ValueError):
            invalid = assessment([])
            self.run_scripted(state, [plan, invalid, invalid, invalid])
        with self.assertRaises(ValueError):
            self.run_scripted(state, [plan, {"lookups": [{"source_id": "q999/artifact_0"}]}])

    def test_assessment_cannot_replace_claim_value_or_kind(self):
        state = load_evidence_session(self.saved_q15_session(), 15)
        plan, key = self.temp_plan(state)
        responses = self.temp_responses(plan, key)
        responses[1]["claims"][0].update(claimed_value="465.21", kind="recommendation")
        responses[1]["claims"][0]["location"] = location(key, 1, "mean_duration_hours", "465.2103157594086")
        result, _ = self.run_scripted(state, responses)
        self.assertEqual(result.claims[0]["claimed_value"], "634.07")
        self.assertEqual(result.claims[0]["kind"], "fact")
        self.assertEqual(result.claims[0]["status"], "supported")

    def test_standalone_entry_point_and_result_persistence(self):
        fixture_session = self.saved_q15_session()
        state = load_evidence_session(fixture_session, 15)
        plan, key = self.temp_plan(state)
        responses = [json.dumps(response) for response in self.temp_responses(plan, key)]
        result = check_saved_session(
            fixture_session,
            15,
            CREDENTIALS,
            response_generator=Mock(side_effect=responses),
            persist=False,
        )
        self.assertEqual(result.confidence, "high")
        session = self.session()
        AgentRunRecorder().record_evidence_checker(session, result)
        path = next(Path(session["artifact_session_dir"]).glob("reports/*evidence_check_request_1.json"))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result.as_dict())

    def test_workflow_toggles_order_and_check_failures_preserve_report(self):
        for verify, check, fail_verifier, fail_checker in (
            (False, False, False, False), (True, False, False, False),
            (False, True, False, False), (True, True, False, False),
            (True, True, True, False), (True, True, False, True),
        ):
            with self.subTest(verify=verify, check=check, fail_verifier=fail_verifier, fail_checker=fail_checker):
                session = self.session(response_validation_enabled=verify, evidence_check_enabled=check)
                order = []

                def agent(name, result, fail=False):
                    def execute(*args):
                        order.append(name)
                        if fail:
                            raise ValueError(f"{name} failed")
                        return result
                    return Mock(execute=execute)

                analyst_result = AnalystResult([{"type": "text", "content": "Completed report"}], "final_report=[]", [], ())
                evidence_result = EvidenceCheckResult((), (), "low", "rule", "recorded")
                workflow = PMaxWorkflow(CREDENTIALS,
                    engineer=agent("engineer", EngineerResult(session["event_log"], "", [])),
                    analyst=agent("analyst", analyst_result),
                    verifier=agent("verifier", VerificationResult(True, "Done"), fail_verifier),
                    evidence_checker=agent("evidence", evidence_result, fail_checker), recorder=Mock())
                result = workflow.execute(session)
                self.assertEqual(order, ["engineer", "analyst"] + (["verifier"] if verify else []) + (["evidence"] if check else []))
                self.assertEqual(session["final_report"], analyst_result.report)
                self.assertEqual(result.evidence_check, evidence_result if check and not fail_checker else None)
                if fail_checker:
                    self.assertTrue(session["evidence_check"]["unavailable"])

    def test_temperature_is_forwarded_to_each_provider_only_when_passed(self):
        for provider, function in (("Google", "google"), ("Anthropic", "anthropic"), ("Cohere", "cohere")):
            with self.subTest(provider=provider):
                with patch(f"promoai.general_utils.llm_connection.generate_response_with_history_{function}", return_value="ok") as generate:
                    session = self.session()
                    args = {"artifact_session_dir": session["artifact_session_dir"], "temperature": 0, "temperature_override": 0}
                    query_llm([], "key", "model", provider, args)
                    self.assertEqual(generate.call_args.kwargs["temperature"], 0)
                    del args["temperature_override"]
                    args["temperature"] = 0.7
                    query_llm([], "key", "model", provider, args)
                    self.assertNotIn("temperature", generate.call_args.kwargs)

    def test_google_receives_zero_temperature_and_openai_filters_internal_override(self):
        session = self.session()
        args = {
            "artifact_session_dir": session["artifact_session_dir"],
            "temperature": 0,
            "temperature_override": 0,
        }
        with patch("promoai.general_utils.llm_connection.genai.Client") as client:
            client.return_value.models.generate_content.return_value = Mock(text="ok")
            query_llm([{"role": "system", "content": "Check evidence"}], "unused", "model", "Google", args)
            config = client.return_value.models.generate_content.call_args.kwargs["config"]
            self.assertEqual(config.temperature, 0)
        response = {"output": [{"content": [{"text": "ok"}]}]}
        with patch("promoai.general_utils.llm_connection._requests_post", return_value=response) as post:
            query_llm([], "unused", "model", "OpenAI", args)
            payload = post.call_args.kwargs["json_"]
            self.assertEqual(payload["temperature"], 0)
            self.assertNotIn("temperature_override", payload)

    def test_ui_has_independent_toggle_and_reference_evidence_format(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("from pmax import run_page\nrun_page()").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.toggle(key="pmax_evidence_check_enabled").value)
        app.toggle(key="pmax_response_validation_enabled").set_value(False).run(timeout=30)
        self.assertTrue(app.toggle(key="pmax_evidence_check_enabled").value)
        self.assertFalse(app.toggle(key="pmax_response_validation_enabled").value)
        blocks = [
            {"type": "verification", "content": {"answers_question": True, "explanation": "Complete"}},
            {
                "type": "evidence_check",
                "content": {
                    "confidence": "medium",
                    "question_parts": [{
                        "question": "How many cases contain activity A?",
                        "status": "answered",
                        "explanation": "The response includes the requested count.",
                    }],
                    "claims": [{
                        "text": "Activity A occurs in 42 cases.",
                        "kind": "fact",
                        "status": "supported",
                        "explanation": "The saved table contains the same count.",
                        "location": {
                            "source_id": "q1/artifact_0",
                            "row": 1,
                            "column": "case_count",
                            "value": "42",
                        },
                    }],
                },
            },
        ]
        script = "from pmax import display_chat_message\ndisplay_chat_message('assistant', " + repr(blocks) + ")"
        rendered = AppTest.from_string(script).run(timeout=30)
        self.assertEqual(len(rendered.exception), 0)
        self.assertEqual(len(rendered.expander), 2)
        self.assertEqual(
            rendered.expander[1].label,
            "🟡 Evidence check: Medium confidence",
        )
        self.assertNotIn("Confianza", rendered.expander[1].label)
        rendered_text = "\n".join(item.value for item in rendered.markdown)
        self.assertIn("#### Question coverage", rendered_text)
        self.assertIn("**answered** — How many cases contain activity A?", rendered_text)
        self.assertIn("#### Claims", rendered_text)
        self.assertIn("**supported** — Activity A occurs in 42 cases.", rendered_text)
        captions = "\n".join(item.value for item in rendered.caption)
        self.assertIn("Evidence: artifact_0, row 1, column case_count, value 42", captions)

    def test_plan_preserves_llm_claim_values_without_numeric_normalization(self):
        state = self.session()
        state["final_report"] = [{
            "type": "text",
            "content": "There are 6,173 cases and acceptance is 77.40%.",
        }]
        plan = {
            "parts": [{
                "id": "p1",
                "question": "Counts and acceptance",
                "data_types": ["frequency"],
                "source_ids": [],
            }],
            "claims": [
                {
                    "id": "c1",
                    "part_id": "p1",
                    "text": "There are 6,173 cases",
                    "kind": "fact",
                    "claimed_value": "6173",
                },
                {
                    "id": "c2",
                    "part_id": "p1",
                    "text": "acceptance is 77.40%",
                    "kind": "fact",
                    "claimed_value": "77.40",
                },
            ],
        }
        EvidenceCheckerAgent._validate_plan(
            plan,
            {},
            "There are 6,173 cases and acceptance is 77.40%.",
        )
        self.assertEqual(plan["claims"][0]["claimed_value"], "6173")
        self.assertEqual(plan["claims"][1]["claimed_value"], "77.40")

    def test_plan_accepts_only_inline_markdown_differences(self):
        plan = {
            "parts": [{
                "id": "p1",
                "question": "Activity frequency",
                "data_types": ["frequency"],
                "source_ids": [],
            }],
            "claims": [{
                "id": "c1",
                "part_id": "p1",
                "text": "A_Cancelled and O_Cancelled each occur in 4,778 cases",
                "kind": "fact",
                "claimed_value": "4778",
            }],
        }
        report = "**Results:** `A_Cancelled` and `O_Cancelled` each occur in **4,778** cases."
        EvidenceCheckerAgent._validate_plan(plan, {}, report)
        self.assertEqual(plan["claims"][0]["claimed_value"], "4778")

        plan["claims"][0]["text"] = "A_Cancelled occurs in 9,999 cases"
        with self.assertRaisesRegex(ValueError, "exact report excerpt"):
            EvidenceCheckerAgent._validate_plan(plan, {}, report)

    def test_plan_accepts_latex_math_delimiters_as_presentation_markup(self):
        plan = {
            "parts": [{
                "id": "p1",
                "question": "Case duration quartiles",
                "data_types": ["time"],
                "source_ids": [],
            }],
            "claims": [{
                "id": "c1",
                "part_id": "p1",
                "text": "First Quartile (Q_1): 436.985 hours",
                "kind": "fact",
                "claimed_value": "436.985",
            }],
        }
        report = "- **First Quartile ($Q_1$):** 436.985 hours"
        EvidenceCheckerAgent._validate_plan(plan, {}, report)


if __name__ == "__main__":
    unittest.main()
