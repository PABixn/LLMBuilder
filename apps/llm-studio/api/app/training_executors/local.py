from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..training_models import TrainingJobState, TrainingJobStatus
from ..training_storage import StoredTrainingJob
from .base import CleanupPolicy, ExecutionHandle, ExecutionSnapshot, TrainingJobBundle

IMPORT_ROOT = Path(__file__).resolve().parents[5]


class LocalSubprocessExecutor:
    kind = "local"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        process = self._spawn_process(bundle)
        self._processes[job.id] = process
        return ExecutionHandle(
            executor_kind=self.kind,
            started_at=_utc_now(),
            process_id=process.pid,
            updates={
                "executor_kind": self.kind,
                "executor_status": "running",
            },
        )

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        process = self._processes.get(job.id)
        exit_code = None
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                self._processes.pop(job.id, None)
        elif job.process_id is not None and job.status in {TrainingJobStatus.pending, TrainingJobStatus.running}:
            if not process_exists(job.process_id):
                exit_code = 1

        if exit_code is None:
            return ExecutionSnapshot()

        if job.status not in {TrainingJobStatus.pending, TrainingJobStatus.running}:
            return ExecutionSnapshot()

        if exit_code == 0:
            return ExecutionSnapshot(
                status=TrainingJobStatus.completed,
                state=TrainingJobState.completed,
                stage="Completed",
                progress=1.0,
                finished_at=_utc_now(),
                updates={"executor_status": "completed", "process_id": None},
            )
        if exit_code == 2:
            return ExecutionSnapshot(
                status=TrainingJobStatus.cancelled,
                state=TrainingJobState.cancelled,
                stage="Cancelled",
                progress=1.0,
                error="Training was cancelled.",
                finished_at=_utc_now(),
                updates={"executor_status": "cancelled", "process_id": None},
            )
        return ExecutionSnapshot(
            status=TrainingJobStatus.failed,
            state=TrainingJobState.failed,
            stage="Failed",
            progress=1.0,
            error=f"Training subprocess exited with code {exit_code}.",
            finished_at=_utc_now(),
            updates={"executor_status": "failed", "process_id": None},
        )

    def stop(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        if job.process_id is not None:
            self._terminate_process(job.process_id)
        process = self._processes.pop(job.id, None)
        if process is not None and process.poll() is None:
            process.terminate()
        return ExecutionSnapshot(
            status=TrainingJobStatus.cancelled,
            state=TrainingJobState.cancelled,
            stage="Cancelled",
            progress=1.0,
            error="Training was cancelled by the user.",
            finished_at=_utc_now(),
            updates={"executor_status": "cancelled", "process_id": None},
        )

    def cleanup(self, job: StoredTrainingJob, policy: CleanupPolicy) -> None:
        return None

    def shutdown(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
        self._processes.clear()

    def _spawn_process(self, bundle: TrainingJobBundle) -> subprocess.Popen[bytes]:
        stdout_handle = bundle.stdout_path.open("ab")
        stderr_handle = bundle.stderr_path.open("ab")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-m",
            "training.runner",
            "--job-id",
            bundle.job_id,
            "--model-config-path",
            str(bundle.model_config_path),
            "--tokenizer-path",
            str(bundle.tokenizer_path),
            "--training-config-path",
            str(bundle.training_config_path),
            "--dataloader-config-path",
            str(bundle.dataloader_config_path),
            "--output-dir",
            str(bundle.job_dir),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=IMPORT_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
                start_new_session=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        return process

    def _terminate_process(self, pid: int) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
