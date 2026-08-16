![ProMoAI logo](https://i.imgur.com/cu5OtpK.png)

# ProMoAI & PMAx

**ProMoAI** is an AI-powered suite for Process Mining that leverages Large Language Models (LLMs) to bridge the gap between natural language and technical process analysis. The framework now includes **PMAx**, an autonomous agentic system for data-driven insights.

The suite consists of two primary modules:

1. **ProMoAI (Model Generation):** Transforms text or event logs into formal process models (BPMN/Petri nets).
2. **PMAx (Agentic Analytics):** An autonomous multi-agent framework that functions as a virtual process analyst to query event logs and generate data-grounded reports.

---

## Features

### 1. ProMoAI: Model Generation & Refinement

- **Text-to-Model:** Generate BPMN or PNML models from natural language descriptions.
- **Model Refinement:** Upload an existing BPMN or Petri net and use AI to modify or extend it via chat.
- **Discovery Baseline:** Start with an XES event log to discover an initial model, then refine it using the LLM.

### 2. PMAx: Agentic Process Mining (New!)

- **Autonomous Reasoning:** Uses a "divide-and-conquer" architecture with specialized **Engineer** and **Analyst** agents.
- **Privacy-Preserving:** Only lightweight metadata (column names/types) is sent to the LLM. Raw event data never leaves your local environment.
- **Deterministic Accuracy:** The system generates and executes local Python code (using whitelisted data preprocessing libraries) to compute exact metrics, avoiding LLM hallucinations.
- **Comprehensive Reporting:** Automatically generates tables, statistical charts, and narrative insights from high-level business questions.

---

## Launching the App

### On the Cloud

Access the unified suite directly at: [https://promoai.streamlit.app/](https://promoai.streamlit.app/)

### Locally

1. Clone this repository.
2. Install the required environment and packages (see [Requirements](#requirements)).
3. Run the application:

- **Unified Suite (ProMoAI + PMAx):**
  ```bash
  streamlit run app.py
  ```
- **Standalone ProMoAI (Legacy Interface):**
  ```bash
  streamlit run promoai_standalone.py
  ```

### With Docker and a local LLM

The complete application can run with a local `qwen3.5:4b` model served by
Ollama. No API key is required for this provider.

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501), select
`Ollama (Local)` in the AI configuration, and save the configuration. The first
startup waits for Ollama to download the model (approximately 3.4 GB), so it can
take several minutes depending on the connection. Later startups reuse the
model stored in the `ollama-data` Docker volume.

Stop the application with `Ctrl+C` or `docker compose down`. To also remove the
downloaded model, run:

```bash
docker compose down -v
```

Ollama is only available to the application on the internal Compose network;
port `11434` is not exposed to the host. When running Ollama outside Docker, the
application defaults to `http://localhost:11434/v1`. This can be overridden with
the `OLLAMA_BASE_URL` environment variable.

### Installation as Python Library:

You can install the core ProMoAI components via pip:

```bash
pip install promoai
```

## Requirements

- _Environment:_ the app is tested on both Python 3.9 and 3.10.
- _Dependencies:_ all required dependencies are listed in the file 'requirements.txt'.
- _Packages:_ all required packages are listed in the file 'packages.txt'.

## Citation

If you use this suite in your research, please cite the relevant papers:

### ProMoAI (Process Modeling)

```bibtex
@inproceedings{DBLP:conf/ijcai/KouraniB0A24,
  author       = {Humam Kourani and
                  Alessandro Berti and
                  Daniel Schuster and
                  Wil M. P. van der Aalst},
  title        = {ProMoAI: Process Modeling with Generative {AI}},
  booktitle    = {Proceedings of the Thirty-Third International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2024},
  pages        = {8708--8712},
  publisher    = {ijcai.org},
  year         = {2024},
  url          = {https://www.ijcai.org/proceedings/2024/1014}
}
```
