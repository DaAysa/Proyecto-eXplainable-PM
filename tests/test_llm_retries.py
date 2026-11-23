import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from google.genai.errors import ClientError, ServerError

from promoai.general_utils.llm_connection import (
    AuthError,
    BadRequestError,
    LLM_REQUEST_ATTEMPTS,
    ServiceUnavailableError,
    generate_result_with_error_handling,
    query_llm,
)


MODULE = "promoai.general_utils.llm_connection"


def google_error(code, message="This model is currently experiencing high demand."):
    error_class = ServerError if code >= 500 else ClientError
    return error_class(code, {"error": {"code": code, "message": message}})


class LLMRetryTests(unittest.TestCase):
    def start_patch(self, *args, **kwargs):
        patcher = patch(*args, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.session = Path(directory.name)
        self.events = []
        self.args = {
            "artifact_session_dir": str(self.session),
            "agent_name": "Engineer",
            "progress_callback": self.events.append,
        }
        self.conversation = [{"role": "user", "content": "Count the cases."}]
        self.client = self.start_patch(f"{MODULE}.genai.Client").return_value
        self.sleep = self.start_patch(f"{MODULE}.time.sleep")
        self.start_patch(f"{MODULE}.random.uniform", return_value=0)
        self.start_patch(f"{MODULE}.constants.ENABLE_PRINTS", False)

    def query(self):
        return query_llm(self.conversation, "unused", "gemini-test", "Google", self.args)

    def test_engineer_recovers_from_503_without_consuming_code_attempts(self):
        self.client.models.generate_content.side_effect = [
            google_error(503), google_error(503), SimpleNamespace(text="working code")
        ]
        extract = Mock(return_value=("code", "result"))
        code, result, messages = generate_result_with_error_handling(
            self.conversation, extract, "unused", "gemini-test", "Google", self.args,
            max_iterations=1, additional_iterations=0,
        )
        self.assertEqual((code, result), ("code", "result"))
        extract.assert_called_once_with("working code", False)
        self.assertEqual(messages, [
            {"role": "user", "content": "Count the cases."},
            {"role": "assistant", "content": "working code"},
        ])
        self.assertEqual(self.sleep.call_args_list, [call(2.0), call(4.0)])
        self.assertEqual([event["event"] for event in self.events], [
            "request_started", "request_retry", "request_retry",
            "response_received", "attempt_completed",
        ])
        retries = [event for event in self.events if event["event"] == "request_retry"]
        self.assertTrue(all(event["agent"] == "Engineer" for event in retries))
        self.assertEqual([event["attempt"] for event in retries], [2, 3])
        self.assertTrue(all(event["model"] == "gemini-test" for event in retries))
        calls = self.client.models.generate_content.call_args_list
        self.assertEqual(calls, [calls[0]] * 3)
        requests = sorted((self.session / "llm_requests").glob("*.json"))
        responses = sorted((self.session / "llm_responses").glob("*.json"))
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(responses), 3)
        self.assertIn("503", json.loads(responses[0].read_text(encoding="utf-8"))["error"])
        self.assertIsNone(json.loads(responses[-1].read_text(encoding="utf-8"))["error"])

    def test_persistent_503_stops_at_limit_without_changing_prompt(self):
        original = copy.deepcopy(self.conversation)
        self.client.models.generate_content.side_effect = google_error(503)
        with self.assertRaises(ServiceUnavailableError):
            self.query()
        self.assertEqual(self.client.models.generate_content.call_count, LLM_REQUEST_ATTEMPTS)
        self.assertEqual(self.sleep.call_args_list, [call(2.0), call(4.0), call(8.0)])
        self.assertEqual(self.conversation, original)

    def test_auth_and_invalid_model_errors_fail_immediately(self):
        for status, error_class in ((400, BadRequestError), (401, AuthError),
                                    (403, AuthError), (404, BadRequestError)):
            with self.subTest(status=status):
                self.client.models.generate_content.reset_mock()
                self.client.models.generate_content.side_effect = google_error(status)
                with self.assertRaises(error_class):
                    self.query()
                self.client.models.generate_content.assert_called_once()
                self.sleep.assert_not_called()

    def test_transient_status_takes_priority_over_model_not_available_text(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.sleep.reset_mock()
                self.client.models.generate_content.side_effect = [
                    google_error(status, "Model not available"), SimpleNamespace(text="ok")
                ]
                self.assertEqual(self.query(), "ok")
                self.sleep.assert_called_once_with(2.0)

    def test_code_correction_remains_separate_from_service_retries(self):
        self.client.models.generate_content.side_effect = [
            SimpleNamespace(text="invalid code"), google_error(503), SimpleNamespace(text="fixed code")
        ]
        extract = Mock(side_effect=[ValueError("Invalid code"), ("code", "result")])
        generate_result_with_error_handling(
            self.conversation, extract, "unused", "gemini-test", "Google", self.args,
            max_iterations=2, additional_iterations=0,
        )
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(sum(m.get("type") == "error" for m in self.conversation), 1)
        self.sleep.assert_called_once_with(2.0)
        starts = [e["attempt"] for e in self.events if e["event"] == "request_started"]
        self.assertEqual(starts, [1, 2])

    def test_untyped_programming_errors_are_not_retried(self):
        with patch(f"{MODULE}._query_llm_once", side_effect=ValueError("bug")) as once:
            with self.assertRaisesRegex(ValueError, "bug"):
                self.query()
            once.assert_called_once()
        self.sleep.assert_not_called()

    def test_jitter_is_added_to_backoff(self):
        self.client.models.generate_content.side_effect = [google_error(503), SimpleNamespace(text="ok")]
        with patch(f"{MODULE}.random.uniform", return_value=0.5):
            self.assertEqual(self.query(), "ok")
        self.sleep.assert_called_once_with(2.5)


if __name__ == "__main__":
    unittest.main()
