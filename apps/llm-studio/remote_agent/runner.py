from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import log_event


@dataclass(slots=True)
class RemoteProcess:
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path


class RemoteTrainingRunner:
    def __init__(self) -> None:
        self._processes: dict[str, RemoteProcess] = {}

    def start(self, *, job_id: str, job_root: Path, manifest: dict[str, Any]) -> int:
        existing = self._processes.get(job_id)
        if existing is not None and existing.process.poll() is None:
            runner_log("start_existing_process", job_id=job_id, process_id=existing.process.pid)
            return existing.process.pid
        runner = manifest.get("runner") if isinstance(manifest.get("runner"), dict) else {}
        args = runner.get("args") if isinstance(runner.get("args"), dict) else {}
        outputs = job_root / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        stdout_path = outputs / "stdout.log"
        stderr_path = outputs / "stderr.log"
        model_config_path = job_root / str(args.get("model_config_path", "inputs/model_config.json"))
        tokenizer_path = job_root / str(args.get("tokenizer_path", "inputs/tokenizer_artifact.json"))
        training_config_path = job_root / str(args.get("training_config_path", "inputs/training_config.json"))
        dataloader_config_path = job_root / str(args.get("dataloader_config_path", "inputs/dataloader_config.json"))
        input_report = input_file_report(
            {
                "model_config": model_config_path,
                "tokenizer": tokenizer_path,
                "training_config": training_config_path,
                "dataloader_config": dataloader_config_path,
            }
        )
        missing_inputs = [item["path"] for item in input_report.values() if item.get("exists") is not True]
        if missing_inputs:
            runner_log("input_files_missing", job_id=job_id, job_root=str(job_root), inputs=input_report)
            raise RuntimeError(f"Training subprocess cannot start because required input files are missing: {missing_inputs}")
        command = [
            sys.executable,
            "-m",
            "training.runner",
            "--job-id",
            job_id,
            "--model-config-path",
            str(model_config_path),
            "--tokenizer-path",
            str(tokenizer_path),
            "--training-config-path",
            str(training_config_path),
            "--dataloader-config-path",
            str(dataloader_config_path),
            "--output-dir",
            str(outputs),
        ]
        runner_log(
            "start_command",
            job_id=job_id,
            cwd=str(repo_root()),
            command=command,
            job_root=str(job_root),
            outputs=str(outputs),
            inputs=input_report,
            cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
            python_executable=sys.executable,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        stdout_handle = stdout_path.open("ab")
        stderr_handle = stderr_path.open("ab")
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root(),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:
                runner_log(
                    "process_spawn_failed",
                    job_id=job_id,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback_text(),
                )
                raise
        finally:
            stdout_handle.close()
            stderr_handle.close()
        self._processes[job_id] = RemoteProcess(process=process, stdout_path=stdout_path, stderr_path=stderr_path)
        runner_log(
            "process_spawned",
            job_id=job_id,
            process_id=process.pid,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        time.sleep(1.0)
        exit_code = process.poll()
        if exit_code is not None:
            stdout_tail = tail_text(stdout_path)
            stderr_tail = tail_text(stderr_path)
            detail = f"Training subprocess exited during startup with code {exit_code}."
            if stdout_tail:
                detail = f"{detail}\n\nstdout tail:\n{stdout_tail}"
            if stderr_tail:
                detail = f"{detail}\n\nstderr tail:\n{stderr_tail}"
            runner_log(
                "process_startup_exit",
                job_id=job_id,
                process_id=process.pid,
                exit_code=exit_code,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
            raise RuntimeError(detail)
        runner_log("process_startup_alive", job_id=job_id, process_id=process.pid)
        return process.pid

    def status(self, job_id: str) -> dict[str, Any]:
        remote_process = self._processes.get(job_id)
        if remote_process is None:
            return {"process_id": None, "exit_code": None, "running": False}
        exit_code = remote_process.process.poll()
        return {
            "process_id": remote_process.process.pid,
            "exit_code": exit_code,
            "running": exit_code is None,
        }

    def cancel(self, job_id: str) -> None:
        remote_process = self._processes.get(job_id)
        if remote_process is None or remote_process.process.poll() is not None:
            runner_log("cancel_no_running_process", job_id=job_id)
            return
        pid = remote_process.process.pid
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            runner_log("cancel_sigterm_sent", job_id=job_id, process_id=pid)
        except ProcessLookupError:
            runner_log("cancel_process_missing", job_id=job_id, process_id=pid)
            return
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if remote_process.process.poll() is not None:
                runner_log("cancel_process_exited", job_id=job_id, process_id=pid, exit_code=remote_process.process.poll())
                return
            time.sleep(0.25)
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
            runner_log("cancel_sigkill_sent", job_id=job_id, process_id=pid)
        except ProcessLookupError:
            runner_log("cancel_process_missing_before_sigkill", job_id=job_id, process_id=pid)
            return


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "training").is_dir() and (parent / "model").is_dir():
            return parent
    return current.parents[1]


def tail_text(path: Path, *, max_bytes: int = 4096) -> str:
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read(max_bytes)
    return data.decode("utf-8", errors="replace").strip()


def input_file_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
        if path.exists():
            try:
                item["size_bytes"] = path.stat().st_size
            except OSError as exc:
                item["stat_error"] = f"{type(exc).__name__}: {exc}"
        report[label] = item
    return report


def traceback_text() -> str:
    import traceback

    return traceback.format_exc(limit=8)


def runner_log(event: str, *, job_id: str | None = None, **fields: Any) -> None:
    log_event(
        service="llm-studio-remote-training-runner",
        event=event,
        job_id=job_id,
        file_name="runner.log",
        prefix="llm-studio-runner",
        **fields,
    )
