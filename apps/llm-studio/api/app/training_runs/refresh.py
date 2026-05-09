from __future__ import annotations

import time

from .schemas import TrainingJobStatus
from .store import StoredTrainingJob


class ExecutorRefreshThrottle:
    def __init__(self, *, runpod_kind: str, interval_seconds: float) -> None:
        self._runpod_kind = runpod_kind
        self._interval_seconds = interval_seconds
        self._last_runpod_refresh_at: dict[str, float] = {}

    @property
    def last_runpod_refresh_at(self) -> dict[str, float]:
        return self._last_runpod_refresh_at

    def should_refresh(self, job: StoredTrainingJob) -> bool:
        if job.executor_kind != self._runpod_kind:
            return True
        if job.status not in {TrainingJobStatus.pending, TrainingJobStatus.running}:
            return True
        last_refresh = self._last_runpod_refresh_at.get(job.id)
        if last_refresh is None:
            return True
        return (time.monotonic() - last_refresh) >= self._interval_seconds

    def record_refresh(self, job: StoredTrainingJob) -> None:
        if job.executor_kind == self._runpod_kind:
            self._last_runpod_refresh_at[job.id] = time.monotonic()
