from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from ..schemas import TrainingJobState, TrainingJobStatus
from ..store import StoredTrainingJob
from ...runtime_paths import ensure_source_root_on_path
from ...dataset_credentials import HF_DATASET_TOKENS_ENV, encode_dataset_hf_tokens
from .base import CleanupPolicy, ExecutionHandle, ExecutionSnapshot, TrainingJobBundle

IMPORT_ROOT = ensure_source_root_on_path()


class LocalSubprocessExecutor:
    kind = "local"

    def __init__(self, *, shutdown_grace_seconds: float = 5.0) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._shutdown_grace_seconds = max(0.0, float(shutdown_grace_seconds))

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
                self._terminate_owned_process(process)
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
        process = self._processes.pop(job.id, None)
        if process is not None:
            self._terminate_owned_process(process)
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
            self._terminate_owned_process(process)
        self._processes.clear()

    def _spawn_process(self, bundle: TrainingJobBundle) -> subprocess.Popen[bytes]:
        stdout_handle = bundle.stdout_path.open("ab")
        stderr_handle = bundle.stderr_path.open("ab")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        encoded_hf_tokens = encode_dataset_hf_tokens(
            bundle.manifest.get("dataset_hf_tokens", [])
            if isinstance(bundle.manifest.get("dataset_hf_tokens"), list)
            else []
        )
        if encoded_hf_tokens is not None:
            env[HF_DATASET_TOKENS_ENV] = encoded_hf_tokens
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
        process_group_options: dict[str, object] = {"start_new_session": True}
        if os.name == "nt":
            process_group_options = {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        try:
            process = subprocess.Popen(
                command,
                cwd=IMPORT_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
                **process_group_options,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        return process

    def _terminate_owned_process(self, process: subprocess.Popen[bytes]) -> None:
        pid = process.pid
        if os.name == "nt":
            self._terminate_windows_process_tree(process)
            return

        self._signal_unix_process_group(pid, signal.SIGTERM)
        self._wait_for_unix_process_group_exit(process, self._shutdown_grace_seconds)
        if process_group_exists(pid):
            self._signal_unix_process_group(pid, signal.SIGKILL)
            self._wait_for_unix_process_group_exit(process, self._shutdown_grace_seconds)
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=self._shutdown_grace_seconds)
        except subprocess.TimeoutExpired:
            pass

    def _terminate_windows_process_tree(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=self._shutdown_grace_seconds)
            except (OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=self._shutdown_grace_seconds)
        except subprocess.TimeoutExpired:
            pass

    def _wait_for_unix_process_group_exit(
        self,
        process: subprocess.Popen[bytes],
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        while process_group_exists(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.025)

    @staticmethod
    def _signal_unix_process_group(pid: int, signal_number: int) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal_number)
            else:
                os.kill(pid, signal_number)
        except ProcessLookupError:
            return


def process_group_exists(pid: int) -> bool:
    if not hasattr(os, "killpg"):
        return process_exists(pid)
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
