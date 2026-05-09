from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..schemas import TrainingJobState, TrainingJobStatus
from ..store import StoredTrainingJob


@dataclass(slots=True)
class TrainingJobBundle:
    job_id: str
    job_dir: Path
    model_config_path: Path
    tokenizer_path: Path
    training_config_path: Path
    dataloader_config_path: Path
    resolved_preflight_path: Path
    stdout_path: Path
    stderr_path: Path
    stats_path: Path
    samples_path: Path
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionHandle:
    executor_kind: str
    started_at: datetime | None = None
    process_id: int | None = None
    remote_ids: dict[str, str] = field(default_factory=dict)
    agent_base_url: str | None = None
    updates: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionSnapshot:
    status: TrainingJobStatus | None = None
    state: TrainingJobState | None = None
    stage: str | None = None
    progress: float | None = None
    error: str | None = None
    finished_at: datetime | None = None
    process_id: int | None = None
    updates: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CleanupPolicy:
    pod: str = "delete_after_sync"
    network_volume: str = "keep"


class TrainingExecutor(Protocol):
    """Executor contract for a prepared training bundle.

    `submit()` owns launching local or remote compute for a persisted job row and
    returns only state fields the manager should merge into storage.
    `refresh()` must be idempotent because polling may call it repeatedly.
    `stop()` should return a terminal snapshot when cancellation is requested.
    `cleanup()` is the only method expected to perform external resource
    deletion or shutdown side effects.
    """

    kind: str

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        ...

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        ...

    def stop(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        ...

    def cleanup(self, job: StoredTrainingJob, policy: CleanupPolicy) -> None:
        ...
