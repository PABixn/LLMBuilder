from __future__ import annotations

from enum import StrEnum
from typing import Any

from ...schemas import TrainingJobState, TrainingJobStatus


class RunPodExecutorStatus(StrEnum):
    queued = "queued"
    provisioning = "provisioning"
    booting = "booting"
    checking_agent = "checking_agent"
    building_bundle = "building_bundle"
    uploading = "uploading"
    starting = "starting"
    running = "running"
    syncing = "syncing"
    cleaning_up = "cleaning_up"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    cleaned_up = "cleaned_up"


def remote_executor_status(state: dict[str, Any]) -> str:
    status = state.get("status")
    if status in {"completed", "failed", "cancelled"}:
        return str(status)
    stage = str(state.get("stage") or "").lower()
    if "sync" in stage:
        return RunPodExecutorStatus.syncing.value
    if "upload" in stage:
        return RunPodExecutorStatus.uploading.value
    return RunPodExecutorStatus.running.value


def coerce_status(value: Any) -> TrainingJobStatus | None:
    try:
        return TrainingJobStatus(str(value))
    except Exception:
        return None


def coerce_state(value: Any) -> TrainingJobState | None:
    try:
        return TrainingJobState(str(value))
    except Exception:
        return None
