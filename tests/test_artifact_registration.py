from types import SimpleNamespace

import pandas as pd

from promoai.agents.agents import analyst_node
from promoai.agents.pm4py_wrapper import PM4PYWrapper
from promoai.agents.state import ProcessState


def _build_state(tmp_path):
    session_dir = tmp_path / "artifacts"
    session_dir.mkdir()
    event_log = pd.DataFrame(
        {
            "case:concept:name": ["case-1"],
            "concept:name": ["A"],
            "time:timestamp": pd.to_datetime(["2026-08-23T16:00:07Z"]),
        }
    )
    return ProcessState(
        user_request="analyze the process",
        event_log=event_log,
        artifact_session_dir=str(session_dir),
    )


def test_save_visualization_does_not_register_failed_exports(tmp_path):
    state = _build_state(tmp_path)
    wrapper = PM4PYWrapper(state, client=None)

    class BrokenFigure:
        def savefig(self, *_args, **_kwargs):
            raise RuntimeError("render failed")

    try:
        wrapper.save_visualization(BrokenFigure(), "Broken visualization", data={})
        assert False, "Expected the export to fail"
    except RuntimeError as error:
        assert str(error) == "render failed"

    assert state["saved_artifacts"] == {}


def test_save_visualization_registers_only_existing_files(tmp_path):
    state = _build_state(tmp_path)
    wrapper = PM4PYWrapper(state, client=None)

    class WorkingFigure:
        def savefig(self, file_path, **_kwargs):
            with open(file_path, "wb") as handle:
                handle.write(b"png")

    wrapper.save_visualization(WorkingFigure(), "Valid visualization", data={})

    assert len(state["saved_artifacts"]) == 1
    [saved_path] = state["saved_artifacts"].keys()
    assert saved_path.endswith(".png")


def test_analyst_node_skips_missing_artifacts(tmp_path, monkeypatch):
    state = _build_state(tmp_path)
    existing_artifact = tmp_path / "existing.png"
    existing_artifact.write_bytes(b"png")
    missing_artifact = tmp_path / "missing.png"

    state.update_artifacts(str(missing_artifact), "Missing artifact", {})
    state.update_artifacts(str(existing_artifact), "Existing artifact", {})

    def fake_generate_result_with_error_handling(
        conversation,
        extraction_function,
        api_key,
        llm_name,
        ai_provider,
        llm_args=None,
        max_iterations=5,
        additional_iterations=5,
        standard_error_message=None,
        progress_callback=None,
    ):
        valid_artifact_ids = extraction_function.keywords["valid_artifact_ids"]
        assert valid_artifact_ids == ["artifact_0"]
        return (
            "```python\nfinal_report = [{'type': 'artifact', 'content': 'artifact_0'}]\n```",
            [{"type": "artifact", "content": "artifact_0"}],
            conversation,
        )

    monkeypatch.setattr(
        "promoai.agents.agents.generate_result_with_error_handling",
        fake_generate_result_with_error_handling,
    )

    credentials = SimpleNamespace(
        api_key="test-key",
        llm_name="test-model",
        ai_provider="test-provider",
        args={},
    )

    updated_state = analyst_node(state, credentials)

    assert updated_state["final_report"] == [
        {"type": "artifact", "content": str(existing_artifact)}
    ]
