from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi

from .errors import (
    RemoteAgentError,
    decode_http_error,
    format_agent_http_error,
    is_retryable_agent_http_error,
)

DEFAULT_AGENT_TIMEOUT_SECONDS = 30.0
DEFAULT_BUNDLE_UPLOAD_TIMEOUT_SECONDS = 120.0
DEFAULT_FILE_DOWNLOAD_TIMEOUT_SECONDS = 120.0
DEFAULT_POD_AGENT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class RemoteAgentClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        job_id: str,
        *,
        timeout: float = DEFAULT_AGENT_TIMEOUT_SECONDS,
        user_agent: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._job_id = job_id
        self._timeout = timeout
        configured_user_agent = (
            user_agent
            or os.getenv("LLM_STUDIO_RUNPOD_AGENT_USER_AGENT")
            or DEFAULT_POD_AGENT_USER_AGENT
        )
        self._user_agent = configured_user_agent.strip() or DEFAULT_POD_AGENT_USER_AGENT
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health", include_job_header=False, include_auth=False)

    def system(self) -> dict[str, Any]:
        query = urlencode({"job_id": self._job_id})
        return self._json("GET", f"/v1/system?{query}")

    def upload_bundle(self, bundle_path: Path, *, content_type: str) -> dict[str, Any]:
        data = bundle_path.read_bytes()
        return self._json(
            "POST",
            f"/v1/jobs/{quote(self._job_id)}/bundle",
            body=data,
            content_type=content_type,
            timeout=max(self._timeout, DEFAULT_BUNDLE_UPLOAD_TIMEOUT_SECONDS),
        )

    def start(self) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{quote(self._job_id)}/start")

    def status(self) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{quote(self._job_id)}")

    def runtime_state(self) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{quote(self._job_id)}/runtime-state")

    def checkpoints(self) -> list[dict[str, Any]]:
        payload = self._json("GET", f"/v1/jobs/{quote(self._job_id)}/checkpoints")
        items = payload.get("checkpoints") if isinstance(payload, dict) else None
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def cancel(self) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{quote(self._job_id)}/cancel")

    def shutdown(self) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{quote(self._job_id)}/shutdown")

    def download_append_file(self, remote_kind: str, local_path: Path) -> None:
        offset = local_path.stat().st_size if local_path.exists() else 0
        query = urlencode({"offset": offset})
        raw = self._bytes("GET", f"/v1/jobs/{quote(self._job_id)}/{remote_kind}?{query}")
        if raw:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("ab") as handle:
                handle.write(raw)

    def download_file(self, relative_path: str, local_path: Path, *, offset: int = 0, optional: bool = False) -> bytes:
        query_payload: dict[str, str | int] = {"path": relative_path, "offset": offset}
        if optional:
            query_payload["optional"] = "1"
        query = urlencode(query_payload)
        try:
            raw = self._bytes(
                "GET",
                f"/v1/jobs/{quote(self._job_id)}/files?{query}",
                timeout=DEFAULT_FILE_DOWNLOAD_TIMEOUT_SECONDS,
            )
        except RemoteAgentError as exc:
            if optional and exc.status_code == 404:
                return b""
            raise
        if raw:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if offset else "wb"
            with local_path.open(mode) as handle:
                handle.write(raw)
        return raw

    def _json(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        include_job_header: bool = True,
        include_auth: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        raw = self._request(
            method,
            path,
            body=body,
            content_type=content_type,
            include_job_header=include_job_header,
            include_auth=include_auth,
            timeout=timeout,
        )
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RemoteAgentError("Pod agent returned a non-JSON response.") from exc
        return payload if isinstance(payload, dict) else {"value": payload}

    def _bytes(self, method: str, path: str, *, timeout: float | None = None) -> bytes:
        return self._request(method, path, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        include_job_header: bool = True,
        include_auth: bool = True,
        timeout: float | None = None,
    ) -> bytes:
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if include_auth:
            headers["Authorization"] = f"Bearer {self._token}"
        if include_job_header:
            headers["X-LLM-Studio-Job-Id"] = self._job_id
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(f"{self._base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout or self._timeout, context=self._ssl_context) as response:
                return response.read()
        except HTTPError as exc:
            detail, payload = decode_http_error(exc)
            retryable = is_retryable_agent_http_error(exc.code, payload)
            message = format_agent_http_error(exc.code, detail, payload)
            raise RemoteAgentError(message, status_code=exc.code, payload=payload, retryable=retryable) from exc
        except URLError as exc:
            raise RemoteAgentError(f"Pod agent is unreachable: {exc.reason}") from exc
