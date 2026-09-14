import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from promoai.agents.contracts import AgentRetryPolicy
from promoai.agents.engineer import EngineerAgent
from promoai.agents.pm4py_wrapper import PM4PYWrapper
from promoai.agents.state import PMaxSession
from promoai.general_utils.llm_connection import LLMConnection


class EventLogIsolationTests(unittest.TestCase):
    def _event_log(self):
        return pd.DataFrame(
            {
                "case:concept:name": ["1", "2", "3"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03"]
                ),
                "amount": [5, 15, 25],
            }
        )

    def _session(self, event_log=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return PMaxSession(
            "analyze",
            self._event_log() if event_log is None else event_log,
            temp_dir.name,
        )

    @staticmethod
    def _credentials():
        return LLMConnection("key", "model", "openai", {})

    @staticmethod
    def _result_generator(response):
        def generate(messages, extraction_function, **kwargs):
            code, result = extraction_function(response, False)
            updated_messages = messages + [{"role": "assistant", "content": response}]
            return code, result, updated_messages

        return generate

    def test_session_keeps_source_and_working_logs_independent(self):
        uploaded_log = self._event_log()
        session = self._session(uploaded_log)

        self.assertIsNot(session["initial_event_log"], uploaded_log)
        self.assertIsNot(session["event_log"], uploaded_log)
        self.assertIsNot(session["event_log"], session["initial_event_log"])

        session["event_log"].loc[0, "concept:name"] = "Changed"

        self.assertEqual(uploaded_log.loc[0, "concept:name"], "A")
        self.assertEqual(session["initial_event_log"].loc[0, "concept:name"], "A")

    def test_failed_in_place_mutation_is_discarded_before_retry(self):
        session = self._session()
        api = PM4PYWrapper(session, None)

        with self.assertRaises(Exception):
            api.code_extraction(
                """```python
api.event_log.drop(index=0, inplace=True)
raise RuntimeError("late failure")
final_event_log = api.event_log
```"""
            )

        _, result = api.code_extraction(
            """```python
final_event_log = api.event_log
```"""
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(len(api.event_log), 3)
        self.assertEqual(len(session["event_log"]), 3)
        self.assertEqual(len(session["initial_event_log"]), 3)

    def test_failed_api_filter_is_discarded_before_retry(self):
        session = self._session()
        api = PM4PYWrapper(session, None)

        with self.assertRaises(Exception):
            api.code_extraction(
                """```python
api.filter_pandas_query("amount > 10")
raise RuntimeError("late failure")
final_event_log = api.event_log
```"""
            )

        _, result = api.code_extraction(
            """```python
final_event_log = api.event_log
```"""
        )

        self.assertEqual(result["amount"].tolist(), [5, 15, 25])
        self.assertEqual(session["event_log"]["amount"].tolist(), [5, 15, 25])

    def test_exhausted_retries_do_not_modify_source_or_session_log(self):
        uploaded_log = self._event_log()
        expected = uploaded_log.copy(deep=True)
        session = self._session(uploaded_log)
        failing_response = """```python
api.event_log.drop(index=0, inplace=True)
raise RuntimeError("late failure")
final_event_log = api.event_log
```"""
        agent = EngineerAgent(
            self._credentials(),
            retry_policy=AgentRetryPolicy(primary_attempts=2, fallback_attempts=0),
        )

        with patch(
            "promoai.general_utils.llm_connection.query_llm",
            return_value=failing_response,
        ):
            with self.assertRaises(Exception):
                agent.execute(session)

        pd.testing.assert_frame_equal(uploaded_log, expected)
        pd.testing.assert_frame_equal(session["initial_event_log"], expected)
        pd.testing.assert_frame_equal(session["event_log"], expected)

    def test_each_new_request_starts_from_original_log(self):
        session = self._session()
        first_response = """```python
final_event_log = api.event_log.query("amount > 10").copy()
```"""
        first_agent = EngineerAgent(
            self._credentials(), result_generator=self._result_generator(first_response)
        )

        first_result = first_agent.execute(session)
        session.apply_engineer_result(first_result)

        self.assertEqual(len(session["event_log"]), 2)
        self.assertIn("Total Events: 2", session["log_abstraction"])

        session.add_request("analyze again")
        second_response = """```python
final_event_log = api.event_log
```"""
        second_agent = EngineerAgent(
            self._credentials(), result_generator=self._result_generator(second_response)
        )

        second_result = second_agent.execute(session)
        session.apply_engineer_result(second_result)

        self.assertEqual(len(second_result.event_log), 3)
        self.assertEqual(len(session["event_log"]), 3)
        self.assertIn("Total Events: 3", session["log_abstraction"])
        self.assertEqual(
            session["initial_event_log"]["amount"].tolist(), [5, 15, 25]
        )


if __name__ == "__main__":
    unittest.main()
