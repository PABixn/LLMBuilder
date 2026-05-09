from __future__ import annotations

import hashlib
import secrets


class RunPodTokenRegistry:
    def __init__(self) -> None:
        self.agent_tokens: dict[str, str] = {}
        self.api_keys: dict[str, str] = {}

    def create_agent_token(self, job_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.agent_tokens[job_id] = token
        return token

    def set_api_key(self, job_id: str, api_key: str) -> None:
        self.api_keys[job_id] = api_key

    def agent_token(self, job_id: str) -> str | None:
        return self.agent_tokens.get(job_id)

    def api_key(self, job_id: str) -> str | None:
        return self.api_keys.get(job_id)

    def clear(self, job_id: str) -> None:
        self.agent_tokens.pop(job_id, None)
        self.api_keys.pop(job_id, None)

    def missing_token_error(self) -> str:
        return (
            "Pod-agent token is not recoverable after API restart. Remote reattach is unavailable in this "
            "version; stop the pod from RunPod or launch a new run."
        )


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
