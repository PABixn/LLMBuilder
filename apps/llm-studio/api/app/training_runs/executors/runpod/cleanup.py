from __future__ import annotations

from ...schemas import TrainingJobStatus
from ...store import StoredTrainingJob
from ..base import CleanupPolicy


def policy_from_job(job: StoredTrainingJob) -> CleanupPolicy:
    payload = job.runpod_cleanup_policy or {}
    return CleanupPolicy(
        pod=str(payload.get("pod") or "delete_after_sync"),
        network_volume=str(payload.get("network_volume") or "keep"),
    )


def terminal_cleanup_policy(job: StoredTrainingJob, status: TrainingJobStatus) -> CleanupPolicy:
    policy = policy_from_job(job)
    if status in {TrainingJobStatus.failed, TrainingJobStatus.cancelled} and policy.pod == "delete_after_sync":
        return CleanupPolicy(pod="stop_after_sync", network_volume=policy.network_volume)
    return policy


def policy_payload(policy: CleanupPolicy) -> dict[str, str]:
    return {"pod": policy.pod, "network_volume": policy.network_volume}
