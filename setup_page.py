import streamlit as st
from promoai.general_utils.ai_providers import (
    AI_HELP_DEFAULTS,
    AI_MODEL_DEFAULTS,
    AIProviders,
    MAIN_HELP,
)

from promoai.general_utils.llm_connection import LLMConnection


def run_page():
    # Initialize State
    if "provider" not in st.session_state:
        st.session_state["provider"] = list(AI_MODEL_DEFAULTS.keys())[0]
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = AI_MODEL_DEFAULTS[st.session_state["provider"]]

    def on_provider_change():
        st.session_state["model_name"] = AI_MODEL_DEFAULTS[st.session_state["provider"]]

    # UI Elements
    st.write("### 🤖 AI Configuration")
    st.markdown('<div class="api-config-marker"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        provider = st.selectbox(
            "AI Provider",
            options=AI_MODEL_DEFAULTS.keys(),
            key="provider",
            on_change=on_provider_change,
            help=MAIN_HELP,
        )

        is_ollama = provider == AIProviders.OLLAMA.value
        api_key = ""

        if is_ollama:
            ai_model_name = st.text_input(
                "Model Name",
                key="model_name",
                help=AI_HELP_DEFAULTS.get(st.session_state["provider"], ""),
            )
            st.caption("Ollama runs locally; no API key is required.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ai_model_name = st.text_input(
                    "Model Name",
                    key="model_name",
                    help=AI_HELP_DEFAULTS.get(st.session_state["provider"], ""),
                )
            with col2:
                api_key = st.text_input(
                    "API Key", type="password", placeholder="my-precious-api-key"
                )
        azure_endpoint = None

        if provider == AIProviders.AZURE.value:
            azure_endpoint = st.text_input(
                "Azure Endpoint",
                key="azure_endpoint",
                placeholder="https://your-resource.openai.azure.com/",
            )
        if st.button("Save Credentials", type="primary", use_container_width=True):
            if not ai_model_name.strip():
                st.error("Please enter a model name.")
            elif not is_ollama and not api_key:
                st.error("Please enter an API key.")
            else:
                args = (
                    {"END_POINT": azure_endpoint}
                    if provider == AIProviders.AZURE.value
                    else {}
                )
                st.session_state["llm_credentials"] = LLMConnection(
                    api_key="ollama" if is_ollama else api_key,
                    llm_name=ai_model_name,
                    ai_provider=provider,
                    args=args,
                )
                st.success(
                    "Credentials saved! You can now navigate to ProMoAI or PMAx."
                )


if __name__ in {"__main__", "__page__"}:
    run_page()
