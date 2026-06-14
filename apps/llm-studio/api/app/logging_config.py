from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import threading
from typing import Any, Iterable

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(x-llm-studio-token|runtime[_-]?token|"
    r"runpod[_-]?(?:api[_-]?)?key|pod[_-]?agent[_-]?token|api[_-]?key)"
    r"\b[\"']?(\s*[=:]\s*|\s+)(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_AUTHORIZATION = re.compile(r"(?i)\bauthorization\s*[=:]\s*(?:bearer\s+)?[^\s,;]+")
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_PROVIDER_CREDENTIAL = re.compile(r"(?i)\b(?:hf|rpa|rps)_[a-z0-9_-]{16,}\b")
_SECRET_KEY_MARKERS = ("token", "apikey", "authorization", "secret", "password", "credential")
_KNOWN_SECRET_SCOPES: dict[str, tuple[str, ...]] = {}
_KNOWN_SECRET_SCOPES_LOCK = threading.RLock()


def configure_backend_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "backend.jsonl"
    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(RedactingJsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    return log_path


class RedactingJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "event_id": getattr(record, "event_id", "backend.log"),
            "message": redact_secrets(record.getMessage()),
        }
        event_fields = getattr(record, "event_fields", None)
        if isinstance(event_fields, dict):
            payload["fields"] = redact_value(event_fields)
        if record.exc_info:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def redact_secrets(value: str) -> str:
    redacted = _AUTHORIZATION.sub("Authorization=[REDACTED]", value)
    redacted = _BEARER.sub("Bearer [REDACTED]", redacted)
    redacted = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        redacted,
    )
    redacted = _PROVIDER_CREDENTIAL.sub("[REDACTED]", redacted)
    for secret in _active_known_secrets():
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def redact_known_secrets(value: str, secrets: Iterable[str | None]) -> str:
    redacted = redact_secrets(value)
    for secret in _normalize_known_secrets(secrets):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def register_known_secrets(scope: str, secrets: Iterable[str | None]) -> tuple[str, ...]:
    normalized = _normalize_known_secrets(secrets)
    with _KNOWN_SECRET_SCOPES_LOCK:
        if normalized:
            _KNOWN_SECRET_SCOPES[scope] = normalized
        else:
            _KNOWN_SECRET_SCOPES.pop(scope, None)
    return normalized


def clear_known_secrets(scope: str) -> None:
    with _KNOWN_SECRET_SCOPES_LOCK:
        _KNOWN_SECRET_SCOPES.pop(scope, None)


def known_secrets_for_scope(scope: str) -> tuple[str, ...]:
    with _KNOWN_SECRET_SCOPES_LOCK:
        return _KNOWN_SECRET_SCOPES.get(scope, ())


def redact_value(value: Any, *, key: str = "") -> Any:
    normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
    if any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {
            str(item_key): redact_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_secrets(str(value))


def _active_known_secrets() -> tuple[str, ...]:
    with _KNOWN_SECRET_SCOPES_LOCK:
        return _normalize_known_secrets(
            secret
            for secrets in _KNOWN_SECRET_SCOPES.values()
            for secret in secrets
        )


def _normalize_known_secrets(secrets: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                secret.strip()
                for secret in secrets
                if isinstance(secret, str) and secret.strip()
            },
            key=len,
            reverse=True,
        )
    )
