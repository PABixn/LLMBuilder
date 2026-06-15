from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from .config import RuntimeSettings
from .data_migrations import DATA_SCHEMA_VERSION

BACKEND_VERSION = "0.1.0"
API_CONTRACT_VERSION = "1"
RUNTIME_MANIFEST_SCHEMA_VERSION = "1"
PROCESS_STARTED_MONOTONIC_ENV = "LLM_STUDIO_PROCESS_STARTED_MONOTONIC"


@dataclass(frozen=True)
class StartupStatus:
    stage: str
    detail: str
    ready: bool


@dataclass
class StartupProfiler:
    started_at: float
    stage_started_at: float
    current_stage: str
    stage_ms: dict[str, float]
    ready_in_ms: float | None = None

    @classmethod
    def begin(
        cls,
        stage: str,
        *,
        now: float | None = None,
        started_at: float | None = None,
    ) -> "StartupProfiler":
        timestamp = time.monotonic() if now is None else now
        origin = timestamp if started_at is None else min(timestamp, started_at)
        return cls(
            started_at=origin,
            stage_started_at=origin,
            current_stage=stage,
            stage_ms={},
        )

    def transition(
        self,
        stage: str,
        *,
        ready: bool = False,
        now: float | None = None,
    ) -> None:
        timestamp = time.monotonic() if now is None else now
        if stage != self.current_stage:
            elapsed = max(0.0, (timestamp - self.stage_started_at) * 1000)
            self.stage_ms[self.current_stage] = round(
                self.stage_ms.get(self.current_stage, 0.0) + elapsed,
                3,
            )
            self.current_stage = stage
            self.stage_started_at = timestamp
        if ready and self.ready_in_ms is None:
            self.ready_in_ms = round(max(0.0, (timestamp - self.started_at) * 1000), 3)

    def payload(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = time.monotonic() if now is None else now
        stage_ms = dict(self.stage_ms)
        if self.ready_in_ms is None:
            stage_ms[self.current_stage] = round(
                stage_ms.get(self.current_stage, 0.0)
                + max(0.0, (timestamp - self.stage_started_at) * 1000),
                3,
            )
        elapsed_ms = (
            self.ready_in_ms
            if self.ready_in_ms is not None
            else round(max(0.0, (timestamp - self.started_at) * 1000), 3)
        )
        return {
            "elapsed_ms": elapsed_ms,
            "ready_in_ms": self.ready_in_ms,
            "stage_ms": stage_ms,
        }


_status_lock = threading.Lock()
_status = StartupStatus(stage="initializing", detail="Backend process is starting.", ready=False)
_migration_status: dict[str, Any] | None = None
_startup_profiler = StartupProfiler.begin("initializing")


def begin_startup() -> None:
    global _status, _migration_status, _startup_profiler
    with _status_lock:
        _status = StartupStatus(
            stage="initializing",
            detail="Backend process is starting.",
            ready=False,
        )
        _migration_status = None
        process_started_at = read_process_started_at()
        if process_started_at is None:
            _startup_profiler = StartupProfiler.begin("initializing")
        else:
            _startup_profiler = StartupProfiler.begin(
                "process_imports",
                started_at=process_started_at,
            )
            _startup_profiler.transition("initializing")


def read_process_started_at(*, now: float | None = None) -> float | None:
    timestamp = time.monotonic() if now is None else now
    raw = os.getenv(PROCESS_STARTED_MONOTONIC_ENV, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    age = timestamp - value
    return value if value > 0 and 0 <= age <= 3600 else None


def set_startup_status(stage: str, detail: str, *, ready: bool = False) -> None:
    global _status
    with _status_lock:
        _startup_profiler.transition(stage, ready=ready)
        _status = StartupStatus(stage=stage, detail=detail, ready=ready)


def startup_status() -> StartupStatus:
    with _status_lock:
        return _status


def set_migration_status(status: dict[str, Any]) -> None:
    global _migration_status
    with _status_lock:
        _migration_status = dict(status)


def migration_status() -> dict[str, Any] | None:
    with _status_lock:
        return None if _migration_status is None else dict(_migration_status)


def startup_timing_payload() -> dict[str, Any]:
    with _status_lock:
        return _startup_profiler.payload()


def readiness_payload(settings: RuntimeSettings) -> dict[str, Any]:
    status = startup_status()
    return {
        "ok": status.ready,
        "ready": status.ready,
        "startup_stage": status.stage,
        "startup_detail": status.detail,
        "backend_version": BACKEND_VERSION,
        "api_contract_version": API_CONTRACT_VERSION,
        "runtime_version": settings.runtime_version,
        "data_schema_version": str(DATA_SCHEMA_VERSION),
        "runtime_manifest_schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "desktop_mode": settings.desktop_mode,
        "compute": compute_capabilities(),
        "migration_status": migration_status(),
        "startup_timing": startup_timing_payload(),
    }


def compute_capabilities() -> dict[str, Any]:
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "platform": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": True,
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def active_job_payload(app_state: Any) -> dict[str, Any]:
    tokenizer_jobs = app_state.tokenizer_store.list_jobs()
    training_jobs = app_state.training_store.list_jobs()
    active_tokenizer = [
        {"id": job.id, "status": job.status.value, "stage": job.stage}
        for job in tokenizer_jobs
        if job.status.value in {"pending", "running"}
    ]
    active_training = [
        {
            "id": job.id,
            "status": job.status.value,
            "stage": job.stage,
            "executor_kind": job.executor_kind,
        }
        for job in training_jobs
        if job.status.value in {"pending", "running"}
    ]
    return {
        "active": bool(active_tokenizer or active_training),
        "tokenizer_jobs": active_tokenizer,
        "training_jobs": active_training,
        "has_active_local_training": any(
            job["executor_kind"] == "local" for job in active_training
        ),
        "has_active_runpod_training": any(
            job["executor_kind"] == "runpod_pod" for job in active_training
        ),
    }
