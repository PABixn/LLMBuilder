from __future__ import annotations

import time
import traceback
from typing import Any

from fastapi import FastAPI, Request

from .auth import configured_job_id
from .files import ranged_file
from .routes import agent_log, create_remote_agent_router, runner_probe
from .runner import RemoteTrainingRunner

app = FastAPI(title="LLM Studio Remote Training Agent", version="0.1.0")
runner = RemoteTrainingRunner()
manifests: dict[str, dict[str, Any]] = {}


@app.middleware("http")
async def log_failed_requests(request: Request, call_next):
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        agent_log(
            "request_exception",
            job_id=request.headers.get("x-llm-studio-job-id") or configured_job_id() or None,
            method=request.method,
            path=request.url.path,
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=8),
        )
        raise
    if response.status_code >= 400 and request.url.path.startswith("/v1/"):
        agent_log(
            "request_error_response",
            job_id=request.headers.get("x-llm-studio-job-id") or configured_job_id() or None,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    return response


app.include_router(create_remote_agent_router(runner=runner, manifests=manifests))

__all__ = ["app", "agent_log", "manifests", "ranged_file", "runner", "runner_probe"]
