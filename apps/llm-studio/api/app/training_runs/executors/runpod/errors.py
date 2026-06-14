from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError

from ....logging_config import redact_secrets, redact_value


class RemoteAgentError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(redact_secrets(message))
        self.status_code = status_code
        self.payload = redact_value(payload)
        self.retryable = retryable


def decode_http_error(exc: HTTPError) -> tuple[str, Any]:
    raw = exc.read()
    detail = raw.decode("utf-8", errors="replace") if raw else ""
    try:
        return detail, json.loads(detail)
    except Exception:
        return detail, None


def format_agent_http_error(status_code: int, detail: str, payload: Any) -> str:
    if isinstance(payload, dict):
        error_code = payload.get("error_code")
        error_name = payload.get("error_name")
        title = payload.get("title")
        if payload.get("cloudflare_error") is True and error_code == 1010:
            return (
                "RunPod proxy rejected the pod-agent request with Cloudflare 1010 "
                f"({error_name or title or 'browser signature blocked'})."
            )
    return f"Pod agent request failed with HTTP {status_code}: {detail}"


def is_retryable_agent_http_error(status_code: int, payload: Any) -> bool:
    if isinstance(payload, dict) and payload.get("cloudflare_error") is True:
        retryable = payload.get("retryable")
        if retryable is False:
            return False
    if status_code in {401, 403}:
        return False
    return status_code == 404 or status_code >= 500
