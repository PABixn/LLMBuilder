from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ....logging_config import redact_secrets, redact_value
from ...store import StoredTrainingJob

_LAST_LIFECYCLE_LOG_AT: dict[tuple[str, str, str], float] = {}
EXECUTOR_KIND = "runpod_pod"
RUNPOD_LIFECYCLE_EVENT_PREFIXES = (
    "submit",
    "create_pod",
    "pod_ready",
    "agent",
    "bundle",
    "sync",
    "cleanup",
    "refresh",
    "stop",
)


def log_lifecycle(
    job: StoredTrainingJob,
    event: str,
    message: str,
    *,
    throttle_seconds: float | None = None,
    throttle_key: str | None = None,
    **fields: Any,
) -> None:
    if throttle_seconds is not None:
        key = (job.id, event, throttle_key or "")
        now_monotonic = time.monotonic()
        previous = _LAST_LIFECYCLE_LOG_AT.get(key)
        if previous is not None and now_monotonic - previous < throttle_seconds:
            return
        _LAST_LIFECYCLE_LOG_AT[key] = now_monotonic

    sanitized_message = redact_secrets(message)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job.id,
        "correlation_id": job.id,
        "executor": EXECUTOR_KIND,
        "event": event,
        "category": lifecycle_error_category(event, fields),
        "message": sanitized_message,
        **sanitize_log_fields(fields),
    }
    line = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)

    artifact_dir = Path(job.artifact_dir)
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "runpod_lifecycle.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        with (artifact_dir / "runpod_lifecycle.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[runpod:{event}] {sanitized_message}")
            detail = compact_log_detail(payload)
            if detail:
                handle.write(f" | {detail}")
            handle.write("\n")
    except Exception:
        pass

    print(f"[llm-studio-runpod] {line}", flush=True)


def read_lifecycle_events(artifact_dir: Path) -> list[dict[str, Any]]:
    path = artifact_dir / "runpod_lifecycle.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def lifecycle_error_category(event: str, fields: dict[str, Any]) -> str | None:
    event_lower = event.lower()
    error_text = str(fields.get("error") or "").lower()
    if "invalid api key" in error_text or "unauthorized" in error_text or "forbidden" in error_text:
        return "invalid_api_key"
    if "capacity" in error_text or "no instances currently available" in error_text:
        return "no_capacity"
    if event_lower.startswith("pod_ready") or "port" in event_lower:
        return "pod_no_port"
    if event_lower.startswith("agent_health") or event_lower.startswith("refresh_agent"):
        return "agent_unreachable"
    if "system" in event_lower or "compatibility" in event_lower or "too old" in error_text:
        return "stale_image"
    if event_lower.startswith("bundle_upload"):
        return "bundle_upload_rejected"
    if "trainer import" in error_text or "cannot import" in error_text:
        return "trainer_import_failure"
    if "runner exited" in error_text or "runtime" in event_lower:
        return "trainer_runtime_failure"
    if event_lower.startswith("sync") or "final artifact sync" in error_text:
        return "sync_failure"
    if event_lower.startswith("cleanup"):
        return "cleanup_failure"
    return None


def is_standard_lifecycle_event(event: str) -> bool:
    return any(event.startswith(prefix) for prefix in RUNPOD_LIFECYCLE_EVENT_PREFIXES)


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    sanitized = redact_value(fields)
    return sanitized if isinstance(sanitized, dict) else {}


def sanitize_log_value(value: Any) -> Any:
    return redact_value(value)


def compact_log_detail(payload: dict[str, Any]) -> str:
    details = {
        key: value
        for key, value in payload.items()
        if key not in {"timestamp", "job_id", "executor", "event", "message"}
        and value is not None
        and value != {}
        and value != []
    }
    if not details:
        return ""
    return json.dumps(details, ensure_ascii=True, default=str, sort_keys=True)
