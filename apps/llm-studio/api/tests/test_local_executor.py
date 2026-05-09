from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.training_runs.executors.base import TrainingJobBundle
from app.training_runs.executors.local import IMPORT_ROOT, LocalSubprocessExecutor
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
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["start_new_session"] is True
    assert captured["stdout_closed_during_call"] is False
    assert captured["stderr_closed_during_call"] is False


@pytest.mark.parametrize(
    ("exit_code", "status", "state", "stage", "executor_status"),
    [
        (0, TrainingJobStatus.completed, TrainingJobState.completed, "Completed", "completed"),
        (2, TrainingJobStatus.cancelled, TrainingJobState.cancelled, "Cancelled", "cancelled"),
        (7, TrainingJobStatus.failed, TrainingJobState.failed, "Failed", "failed"),
    ],
)
def test_local_executor_maps_process_exit_codes(
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
    job = make_job(tmp_path)
    executor._processes[job.id] = FakeProcess()  # type: ignore[assignment]

    snapshot = executor.refresh(job)

    assert snapshot.status == status
    assert snapshot.state == state
    assert snapshot.stage == stage
    assert snapshot.progress == 1.0
    assert snapshot.finished_at is not None
    assert snapshot.updates["executor_status"] == executor_status
    assert snapshot.updates["process_id"] is None
    if exit_code == 7:
        assert snapshot.error == "Training subprocess exited with code 7."
