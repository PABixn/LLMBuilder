from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.training_runs.executors.base import TrainingJobBundle
from app.training_runs.executors.local import (
    IMPORT_ROOT,
    LocalSubprocessExecutor,
    process_group_exists,
)
from app.dataset_credentials import HF_DATASET_TOKENS_ENV
from app.training_runs.schemas import TrainingJobState, TrainingJobStatus
from app.training_runs.store import StoredTrainingJob


def make_bundle(tmp_path: Path) -> TrainingJobBundle:
    return TrainingJobBundle(
        job_id="job123456",
        job_dir=tmp_path,
        model_config_path=tmp_path / "model_config.json",
        tokenizer_path=tmp_path / "tokenizer.json",
        training_config_path=tmp_path / "training_config.json",
        dataloader_config_path=tmp_path / "dataloader_config.json",
        resolved_preflight_path=tmp_path / "resolved_preflight.json",
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stats_path=tmp_path / "stats.jsonl",
        samples_path=tmp_path / "samples.jsonl",
    )


def make_job(tmp_path: Path) -> StoredTrainingJob:
    return StoredTrainingJob(
        id="job123456",
        name="Local job",
        status=TrainingJobStatus.running,
        state=TrainingJobState.preflight,
        stage="Running",
        progress=0.0,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=None,
        project_id="project123",
        project_name="Project",
        tokenizer_job_id="tok123456",
        tokenizer_name="Tokenizer",
        model_config={},
        training_config={},
        dataloader_config={},
        resolved_runtime=None,
        memory_estimate=None,
        artifact_dir=str(tmp_path),
        artifact_bundle_file=None,
        stats_path=str(tmp_path / "stats.jsonl"),
        samples_path=str(tmp_path / "samples.jsonl"),
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
        last_step=0,
        max_steps=10,
        latest_loss=None,
        latest_grad_norm=None,
        latest_lr=None,
        latest_tokens_per_sec=None,
        checkpoint_count=0,
        sample_count=0,
        error=None,
        process_id=None,
        output_size_bytes=0,
        executor_kind="local",
        executor_status="running",
    )


def test_local_executor_constructs_training_runner_command(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakePopen:
        pid = 1234

        def __init__(self, command, *, cwd, stdout, stderr, env, start_new_session):  # noqa: ANN001
            captured["command"] = command
            captured["cwd"] = cwd
            captured["stdout_closed_during_call"] = stdout.closed
            captured["stderr_closed_during_call"] = stderr.closed
            captured["env"] = env
            captured["start_new_session"] = start_new_session

        def poll(self) -> None:
            return None

    monkeypatch.setattr("app.training_runs.executors.local.subprocess.Popen", FakePopen)
    bundle = make_bundle(tmp_path)

    handle = LocalSubprocessExecutor().submit(make_job(tmp_path), bundle)

    assert handle.process_id == 1234
    assert captured["command"] == [
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
    assert captured["cwd"] == IMPORT_ROOT
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["start_new_session"] is True
    assert captured["stdout_closed_during_call"] is False
    assert captured["stderr_closed_during_call"] is False


def test_local_executor_injects_hf_credentials_only_into_process_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    hf_token = "hf_0123456789abcdef0123456789abcdef"

    class FakePopen:
        pid = 1234

        def __init__(self, _command, *, cwd, stdout, stderr, env, start_new_session):  # noqa: ANN001
            captured["env"] = env

        def poll(self) -> None:
            return None

    monkeypatch.setattr("app.training_runs.executors.local.subprocess.Popen", FakePopen)
    bundle = make_bundle(tmp_path)
    bundle.manifest["dataset_hf_tokens"] = [None, hf_token]
    bundle.dataloader_config_path.write_text(
        '{"datasets":[{"name":"public"},{"name":"private"}]}',
        encoding="utf-8",
    )

    LocalSubprocessExecutor().submit(make_job(tmp_path), bundle)

    assert hf_token in captured["env"][HF_DATASET_TOKENS_ENV]
    assert hf_token not in bundle.dataloader_config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("exit_code", "status", "state", "stage", "executor_status"),
    [
        (0, TrainingJobStatus.completed, TrainingJobState.completed, "Completed", "completed"),
        (2, TrainingJobStatus.cancelled, TrainingJobState.cancelled, "Cancelled", "cancelled"),
        (7, TrainingJobStatus.failed, TrainingJobState.failed, "Failed", "failed"),
    ],
)
def test_local_executor_maps_process_exit_codes(
    monkeypatch,
    tmp_path: Path,
    exit_code: int,
    status: TrainingJobStatus,
    state: TrainingJobState,
    stage: str,
    executor_status: str,
) -> None:
    class FakeProcess:
        def poll(self) -> int:
            return exit_code

    executor = LocalSubprocessExecutor()
    terminated: list[object] = []
    monkeypatch.setattr(executor, "_terminate_owned_process", terminated.append)
    job = make_job(tmp_path)
    process = FakeProcess()
    executor._processes[job.id] = process  # type: ignore[assignment]

    snapshot = executor.refresh(job)

    assert snapshot.status == status
    assert snapshot.state == state
    assert snapshot.stage == stage
    assert snapshot.progress == 1.0
    assert snapshot.finished_at is not None
    assert snapshot.updates["executor_status"] == executor_status
    assert snapshot.updates["process_id"] is None
    assert terminated == [process]
    if exit_code == 7:
        assert snapshot.error == "Training subprocess exited with code 7."


@pytest.mark.skipif(os.name == "nt", reason="Unix process-group behavior")
def test_local_executor_stop_kills_owned_descendants_only(tmp_path: Path) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_script = tmp_path / "descendant.py"
    descendant_script.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "import signal",
                "import time",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                f"Path({str(descendant_pid_path)!r}).write_text(str(os.getpid()))",
                "time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )
    parent_script = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,{str(descendant_script)!r}]);"
        "time.sleep(60)"
    )
    owned = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 3
        while not descendant_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert descendant_pid_path.is_file()
        assert process_group_exists(owned.pid)

        executor = LocalSubprocessExecutor(shutdown_grace_seconds=0.1)
        job = make_job(tmp_path)
        job.process_id = owned.pid
        executor._processes[job.id] = owned

        snapshot = executor.stop(job)

        assert snapshot.status == TrainingJobStatus.cancelled
        assert owned.poll() is not None
        assert not process_group_exists(owned.pid)
        assert unrelated.poll() is None
    finally:
        if process_group_exists(owned.pid):
            os.killpg(owned.pid, signal.SIGKILL)
        if process_group_exists(unrelated.pid):
            os.killpg(unrelated.pid, signal.SIGKILL)
        owned.wait(timeout=3)
        unrelated.wait(timeout=3)


def test_local_executor_does_not_signal_untracked_persisted_pid(monkeypatch, tmp_path: Path) -> None:
    terminated: list[object] = []
    executor = LocalSubprocessExecutor()
    monkeypatch.setattr(executor, "_terminate_owned_process", terminated.append)
    job = make_job(tmp_path)
    job.process_id = 987654

    executor.stop(job)

    assert terminated == []
