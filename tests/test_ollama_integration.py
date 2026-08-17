import os
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

from promoai.general_utils.ai_providers import AI_MODEL_DEFAULTS, AIProviders
from promoai.general_utils.llm_connection import ServiceUnavailableError, query_llm


def test_ollama_provider_has_expected_default_model():
    assert AI_MODEL_DEFAULTS[AIProviders.OLLAMA.value] == "qwen3.5:4b"


def test_query_llm_uses_ollama_openai_compatible_endpoint():
    response = Mock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": "local response"}}]
    }
    conversation = [{"role": "user", "content": "Hello"}]

    with TemporaryDirectory() as trace_dir:
        with patch.dict(
            os.environ,
            {"OLLAMA_BASE_URL": "http://ollama:11434/v1/"},
            clear=False,
        ):
            with patch(
                "promoai.general_utils.llm_connection.requests.post",
                return_value=response,
            ) as post:
                result = query_llm(
                    conversation,
                    api_key="",
                    llm_name="qwen3.5:4b",
                    ai_provider=AIProviders.OLLAMA.value,
                    llm_args={"artifact_session_dir": trace_dir},
                )

    assert result == "local response"
    call = post.call_args
    assert call.args[0] == "http://ollama:11434/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer ollama"
    assert call.kwargs["timeout"] == (3.05, 600.0)
    assert call.kwargs["json"] == {
        "model": "qwen3.5:4b",
        "messages": conversation,
    }


def test_query_llm_maps_ollama_connection_errors():
    with TemporaryDirectory() as trace_dir:
        with patch(
            "promoai.general_utils.llm_connection.requests.post",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            try:
                query_llm(
                    [{"role": "user", "content": "Hello"}],
                    api_key="",
                    llm_name="qwen3.5:4b",
                    ai_provider=AIProviders.OLLAMA.value,
                    llm_args={"artifact_session_dir": trace_dir},
                )
                assert False, "Expected a ServiceUnavailableError"
            except ServiceUnavailableError as error:
                assert error.retryable is True


def test_setup_allows_ollama_without_api_key():
    app = AppTest.from_file("setup_page.py").run()

    assert AIProviders.OLLAMA.value in app.selectbox[0].options

    app.selectbox[0].set_value(AIProviders.OLLAMA.value).run()

    assert len(app.text_input) == 1
    assert app.text_input[0].value == "qwen3.5:4b"

    app.button[0].click().run()

    assert len(app.error) == 0
    credentials = app.session_state["llm_credentials"]
    assert credentials.ai_provider == AIProviders.OLLAMA.value
    assert credentials.api_key == "ollama"
