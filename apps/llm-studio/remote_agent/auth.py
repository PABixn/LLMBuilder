from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


def configured_job_id() -> str:
    return os.getenv("LLM_STUDIO_REMOTE_JOB_ID", "").strip()


def configured_token() -> str:
    return os.getenv("LLM_STUDIO_REMOTE_AGENT_TOKEN", "").strip()


def require_agent_auth(
    job_id: str,
    authorization: str | None = Header(default=None),
    x_llm_studio_job_id: str | None = Header(default=None),
) -> None:
    expected_job_id = configured_job_id()
    expected_token = configured_token()
    if not expected_job_id or not expected_token:
        raise HTTPException(status_code=503, detail="Remote agent is not configured.")
    if not hmac.compare_digest(job_id, expected_job_id):
        raise HTTPException(status_code=403, detail="Job id is not allowed on this pod.")
    if not x_llm_studio_job_id or not hmac.compare_digest(x_llm_studio_job_id, expected_job_id):
        raise HTTPException(status_code=403, detail="Missing or invalid job id header.")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    provided = authorization[len(prefix) :].strip()
    if not hmac.compare_digest(provided, expected_token):
        raise HTTPException(status_code=401, detail="Invalid bearer token.")
