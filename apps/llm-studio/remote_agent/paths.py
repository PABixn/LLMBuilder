from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.getenv("LLM_STUDIO_REMOTE_WORKSPACE", "/workspace/llm-studio")).resolve()


def job_root(job_id: str) -> Path:
    return workspace_root() / "jobs" / job_id


def outputs_dir(job_id: str) -> Path:
    return job_root(job_id) / "outputs"
