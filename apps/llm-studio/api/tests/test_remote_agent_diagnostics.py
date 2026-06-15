from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_diagnostics() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "remote_agent" / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("llm_studio_remote_agent_diagnostics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_agent_diagnostics_redacts_generic_provider_error_values() -> None:
    diagnostics = _load_diagnostics()
    hf_token = "hf_0123456789abcdef0123456789abcdef"
    runpod_token = "rpa_0123456789abcdef0123456789abcdef"

    sanitized = diagnostics.sanitize_log_fields(
        {
            "generic_error": f"provider echoed {hf_token} and {runpod_token}",
            "nested": [{"message": f"retry used {hf_token}"}],
            "safe": "value",
        }
    )

    combined = str(sanitized)
    assert hf_token not in combined
    assert runpod_token not in combined
    assert "[redacted]" in combined


def test_remote_agent_diagnostics_redacts_raw_json_assignment_strings() -> None:
    diagnostics = _load_diagnostics()
    raw = '{"api_key": "arbitrary-secret", "token":"another-secret"}'

    redacted = diagnostics.redact_secrets(raw)

    assert "arbitrary-secret" not in redacted
    assert "another-secret" not in redacted
    assert redacted.count("[redacted]") == 2


def test_remote_agent_diagnostics_redacts_arbitrary_execution_token(monkeypatch) -> None:
    diagnostics = _load_diagnostics()
    secret = "legacy-private-token-without-provider-prefix"
    monkeypatch.setenv("LLM_STUDIO_HF_DATASET_TOKENS", f'["{secret}"]')

    redacted = diagnostics.redact_secrets(f"provider echoed {secret}")

    assert secret not in redacted
    assert "[redacted]" in redacted
