from __future__ import annotations

import json
import logging
from pathlib import Path

from app.logging_config import (
    RedactingJsonFormatter,
    clear_known_secrets,
    configure_backend_logging,
    redact_known_secrets,
    redact_secrets,
    register_known_secrets,
)


def test_secret_redaction_covers_runtime_and_provider_credentials() -> None:
    value = (
        "Authorization: Bearer top-secret "
        "runtime_token=desktop-secret runpod_api_key=provider-secret"
    )

    redacted = redact_secrets(value)

    assert "top-secret" not in redacted
    assert "desktop-secret" not in redacted
    assert "provider-secret" not in redacted
    assert redacted.count("[REDACTED]") == 3


def test_secret_redaction_covers_bare_provider_credentials() -> None:
    secrets = (
        "rpa_0123456789abcdef0123456789abcdef",
        "rps_0123456789abcdef0123456789abcdef",
        "hf_0123456789abcdef0123456789abcdef",
    )

    redacted = redact_secrets(f"provider echoed {' '.join(secrets)}")

    assert all(secret not in redacted for secret in secrets)
    assert redacted.count("[REDACTED]") == len(secrets)


def test_secret_redaction_covers_raw_json_assignment_strings() -> None:
    value = '{"api_key": "arbitrary-secret", "runtime_token":"desktop-secret"}'

    redacted = redact_secrets(value)

    assert "arbitrary-secret" not in redacted
    assert "desktop-secret" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_secret_redaction_covers_scoped_arbitrary_credentials() -> None:
    secret = "legacy-private-token-without-provider-prefix"
    scope = "test:arbitrary-credential"
    try:
        register_known_secrets(scope, [secret])

        assert secret not in redact_secrets(f"provider echoed {secret}")
        assert secret not in redact_known_secrets(f"retry echoed {secret}", [secret])
    finally:
        clear_known_secrets(scope)
    assert redact_secrets(secret) == secret


def test_structured_formatter_emits_event_id_without_secret() -> None:
    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "runtime_token=%s",
        ("secret-value",),
        None,
    )
    record.event_id = "test.event"

    payload = json.loads(RedactingJsonFormatter().format(record))

    assert payload["event_id"] == "test.event"
    assert "secret-value" not in payload["message"]


def test_structured_formatter_redacts_approved_event_fields_recursively() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "event", (), None)
    record.event_id = "test.fields"
    record.event_fields = {
        "job_id": "job-123",
        "api_key": "provider-secret",
        "nested": {"runtime_token": "runtime-secret"},
    }

    payload = json.loads(RedactingJsonFormatter().format(record))

    assert payload["fields"]["job_id"] == "job-123"
    assert payload["fields"]["api_key"] == "[REDACTED]"
    assert payload["fields"]["nested"]["runtime_token"] == "[REDACTED]"
    assert "provider-secret" not in json.dumps(payload)
    assert "runtime-secret" not in json.dumps(payload)


def test_backend_logging_rotates_to_managed_log_file(tmp_path: Path) -> None:
    path = configure_backend_logging(tmp_path)
    logging.getLogger("test").info("hello", extra={"event_id": "test.hello"})
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert path == tmp_path / "backend.jsonl"
    assert '"event_id":"test.hello"' in path.read_text(encoding="utf-8")
