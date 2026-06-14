from __future__ import annotations

import json
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.training_executors.base import ExecutionHandle, ExecutionSnapshot, TrainingJobBundle
from app.training_jobs import ResolvedPreflightContext, TrainingRunManager
from app.training_models import (
    CreateTrainingJobRequest,
    RunPodCleanupPolicy,
    TrainingAssetRef,
    TrainingBatchLrRecommendation,
    TrainingExecutorKind,
    TrainingJobState,
    TrainingJobStatus,
)
from app.training_storage import StoredTrainingJob


class FakeTrainingStore:
    def __init__(self) -> None:
        self.jobs: dict[str, StoredTrainingJob] = {}
        self.events: list[str] = []
        self.deleted: list[str] = []

    def create_job(self, job: StoredTrainingJob) -> None:
        self.events.append("create")
        self.jobs[job.id] = job

    def get_job(self, job_id: str) -> StoredTrainingJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[StoredTrainingJob]:
        return list(self.jobs.values())

    def update_job(self, job_id: str, **updates: Any) -> StoredTrainingJob | None:
        self.events.append(f"update:{updates.get('executor_status') or updates.get('status') or 'fields'}")
        job = self.jobs.get(job_id)
        if job is None:
            return None
        for key, value in updates.items():
            setattr(job, key, value)
        return job

    def delete_job(self, job_id: str) -> StoredTrainingJob | None:
        self.deleted.append(job_id)
        return self.jobs.pop(job_id, None)


class FakeLocalExecutor:
    kind = "local"

    def __init__(self) -> None:
        self.submitted: list[tuple[StoredTrainingJob, TrainingJobBundle]] = []
        self.stopped: list[str] = []

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        self.submitted.append((job, bundle))
        return ExecutionHandle(
            executor_kind=self.kind,
            process_id=4321,
            updates={"executor_status": "running"},
        )

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        return ExecutionSnapshot()

    def stop(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        self.stopped.append(job.id)
        return ExecutionSnapshot(status=TrainingJobStatus.cancelled)

    def cleanup(self, job: StoredTrainingJob, policy: object) -> None:
        return None


class FakeRunPodExecutor(FakeLocalExecutor):
    kind = "runpod_pod"

    def __init__(self, *, fail_submit: bool = False) -> None:
        super().__init__()
        self.fail_submit = fail_submit

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        if self.fail_submit:
            raise RuntimeError("remote launch exploded")
        return ExecutionHandle(executor_kind=self.kind, updates={"executor_status": "running"})

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        return ExecutionSnapshot(updates={"executor_status": job.executor_status})


class DeferredThread:
    created: list["DeferredThread"] = []

    def __init__(self, *, target, args, name: str, daemon: bool) -> None:  # noqa: ANN001
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False
        DeferredThread.created.append(self)

    def start(self) -> None:
        self.started = True


def make_context() -> ResolvedPreflightContext:
    return ResolvedPreflightContext(
        valid=True,
        model_project=TrainingAssetRef(id="project123", name="Project", artifact_file="model_config.json", status="READY"),
        tokenizer=TrainingAssetRef(id="tok123456", name="Tokenizer", artifact_file="tokenizer.json", status="completed"),
        model_config={"context_length": 8, "vocab_size": 16},
        normalized_training_config={"max_steps": 10},
        normalized_dataloader_config={"datasets": []},
        warnings=[],
        errors=[],
        recommended_fixes=[],
        compatibility=None,
        derived_runtime=None,
        memory_estimate=None,
        batch_and_lr_recommendation=None,
    )


def make_manager(tmp_path: Path, monkeypatch, *, runpod_fail_submit: bool = False) -> tuple[TrainingRunManager, FakeTrainingStore, FakeLocalExecutor, FakeRunPodExecutor]:
    settings = SimpleNamespace(training_jobs_dir=tmp_path / "jobs", runpod_default_gpu_count=1)
    monkeypatch.setattr("app.training_jobs.get_settings", lambda: settings)
    tokenizer_artifact = tmp_path / "tokenizer.json"
    tokenizer_artifact.write_text("{}", encoding="utf-8")

    store = FakeTrainingStore()
    local_executor = FakeLocalExecutor()
    runpod_executor = FakeRunPodExecutor(fail_submit=runpod_fail_submit)
    manager = TrainingRunManager.__new__(TrainingRunManager)
    manager._store = store
    manager._tokenizer_store = object()
    manager._local_executor = local_executor
    manager._runpod_executor = runpod_executor
    manager._executors = {local_executor.kind: local_executor, runpod_executor.kind: runpod_executor}
    manager._refresh_locks = {}
    manager._refresh_locks_guard = threading.Lock()
    manager._last_runpod_refresh_at = {}
    manager._resolve_preflight_context = lambda _request: make_context()
    manager._require_tokenizer_artifact_path = lambda _tokenizer_job_id: tokenizer_artifact
    return manager, store, local_executor, runpod_executor


def make_request(*, kind: TrainingExecutorKind = TrainingExecutorKind.local) -> CreateTrainingJobRequest:
    payload: dict[str, Any] = {
        "project_id": "project123",
        "tokenizer_job_id": "tok123456",
        "training_config": {},
        "dataloader_config": {},
        "execution_target": {"kind": kind.value},
    }
    if kind == TrainingExecutorKind.runpod_pod:
        payload["execution_target"].update(
            {
                "api_key": "ui-key",
                "cleanup_policy": RunPodCleanupPolicy().model_dump(mode="json"),
            }
        )
    return CreateTrainingJobRequest.model_validate(payload)


def make_stored_job(tmp_path: Path, *, status: TrainingJobStatus = TrainingJobStatus.running) -> StoredTrainingJob:
    failed = status == TrainingJobStatus.failed
    return StoredTrainingJob(
        id="job123456",
        name="Job",
        status=status,
        state=TrainingJobState.failed if failed else TrainingJobState.preflight,
        stage="Failed" if failed else "Running",
        progress=1.0 if failed else 0.0,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc) if failed else None,
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
        executor_kind="runpod_pod",
        executor_status="running",
    )


def test_create_local_training_job_uses_fake_executor_without_running_trainer(tmp_path: Path, monkeypatch) -> None:
    manager, store, local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)

    response = manager.create_job(make_request())

    assert response.status == TrainingJobStatus.running
    assert response.executor_kind == TrainingExecutorKind.local
    assert response.process_id == 4321
    assert len(local_executor.submitted) == 1
    stored, bundle = local_executor.submitted[0]
    assert store.events[0] == "create"
    assert stored.id == response.id == bundle.job_id
    assert Path(response.artifact_dir).exists()
    assert (Path(response.artifact_dir) / "resolved_preflight.json").exists()


def test_local_launch_failure_redacts_arbitrary_dataset_token_and_suppresses_raw_cause(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, store, local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    secret = "legacy-private-token-without-provider-prefix"
    local_executor.submit = lambda *_args: (_ for _ in ()).throw(
        RuntimeError(f"local provider echoed {secret}")
    )
    request = make_request()
    request.hf_token = secret
    request.dataloader_config = {
        "datasets": [
            {
                "name": "private-dataset",
                "split": "train",
                "streaming": True,
                "text_columns": ["text"],
            }
        ]
    }

    try:
        manager.create_job(request)
    except RuntimeError as exc:
        assert secret not in str(exc)
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
    else:
        raise AssertionError("Expected local launch failure")

    assert store.jobs == {}
    assert manager._token_registry() == {}


def test_create_runpod_job_persists_row_before_remote_submission(tmp_path: Path, monkeypatch) -> None:
    DeferredThread.created = []
    monkeypatch.setattr("app.training_jobs.threading.Thread", DeferredThread)
    manager, store, _local_executor, runpod_executor = make_manager(tmp_path, monkeypatch)

    response = manager.create_job(make_request(kind=TrainingExecutorKind.runpod_pod))

    assert response.status == TrainingJobStatus.running
    assert response.state == TrainingJobState.preflight
    assert response.executor_status == "provisioning"
    assert store.events[:2] == ["create", "update:provisioning"]
    assert len(DeferredThread.created) == 1
    assert DeferredThread.created[0].started is True
    assert runpod_executor.submitted == []


def test_remote_submission_failure_marks_failed_and_keeps_job_directory(tmp_path: Path, monkeypatch) -> None:
    manager, store, _local_executor, runpod_executor = make_manager(tmp_path, monkeypatch, runpod_fail_submit=True)
    job_dir = tmp_path / "remote-job"
    job_dir.mkdir()
    stored = make_stored_job(job_dir)
    store.create_job(stored)
    bundle = TrainingJobBundle(
        job_id=stored.id,
        job_dir=job_dir,
        model_config_path=job_dir / "model_config.json",
        tokenizer_path=job_dir / "tokenizer.json",
        training_config_path=job_dir / "training_config.json",
        dataloader_config_path=job_dir / "dataloader_config.json",
        resolved_preflight_path=job_dir / "resolved_preflight.json",
        stdout_path=job_dir / "stdout.log",
        stderr_path=job_dir / "stderr.log",
        stats_path=job_dir / "stats.jsonl",
        samples_path=job_dir / "samples.jsonl",
        manifest={},
    )

    manager._submit_remote_job(stored.id, stored, bundle)

    failed = store.get_job(stored.id)
    assert failed is not None
    assert failed.status == TrainingJobStatus.failed
    assert failed.state == TrainingJobState.failed
    assert failed.executor_status == "failed"
    assert failed.remote_error == "remote launch exploded"
    assert job_dir.exists()
    assert runpod_executor.submitted == []


def test_remote_submission_failure_redacts_arbitrary_dataset_token(tmp_path: Path, monkeypatch) -> None:
    manager, store, _local_executor, runpod_executor = make_manager(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "remote-job"
    job_dir.mkdir(parents=True)
    stored = make_stored_job(job_dir)
    store.create_job(stored)
    secret = "legacy-private-token-without-provider-prefix"
    runpod_executor.submit = lambda *_args: (_ for _ in ()).throw(
        RuntimeError(f"remote provider echoed {secret}")
    )
    bundle = TrainingJobBundle(
        job_id=stored.id,
        job_dir=job_dir,
        model_config_path=job_dir / "model_config.json",
        tokenizer_path=job_dir / "tokenizer.json",
        training_config_path=job_dir / "training_config.json",
        dataloader_config_path=job_dir / "dataloader_config.json",
        resolved_preflight_path=job_dir / "resolved_preflight.json",
        stdout_path=job_dir / "stdout.log",
        stderr_path=job_dir / "stderr.log",
        stats_path=job_dir / "stats.jsonl",
        samples_path=job_dir / "samples.jsonl",
        manifest={"dataset_hf_tokens": [secret]},
    )
    manager._remember_dataset_hf_tokens(stored.id, [secret])

    manager._submit_remote_job(stored.id, stored, bundle)

    failed = store.get_job(stored.id)
    assert failed is not None
    assert failed.error is not None and secret not in failed.error
    assert failed.remote_error is not None and secret not in failed.remote_error
    assert manager._known_dataset_hf_tokens(stored.id) == ()


def test_state_updates_from_runtime_sets_terminal_completed_and_failed(tmp_path: Path, monkeypatch) -> None:
    manager, _store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)

    completed = manager._state_updates_from_runtime({"status": "completed", "state": "completed", "stage": "Done"})
    failed = manager._state_updates_from_runtime({"status": "failed", "state": "failed", "error": "boom"})

    assert completed["status"] == TrainingJobStatus.completed
    assert completed["state"] == TrainingJobState.completed
    assert completed["executor_status"] == "completed"
    assert failed["status"] == TrainingJobStatus.failed
    assert failed["state"] == TrainingJobState.failed
    assert failed["executor_status"] == "failed"
    assert failed["error"] == "boom"


def test_terminal_db_row_is_not_reverted_by_stale_runtime_or_executor_snapshot(tmp_path: Path, monkeypatch) -> None:
    manager, store, _local_executor, runpod_executor = make_manager(tmp_path, monkeypatch)
    job = make_stored_job(tmp_path, status=TrainingJobStatus.completed)
    job.state = TrainingJobState.completed
    job.stage = "Completed"
    job.progress = 1.0
    job.executor_status = "completed"
    store.create_job(job)
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({"status": "running", "state": "training", "stage": "Training", "progress": 0.4}),
        encoding="utf-8",
    )

    refreshed = manager._refresh_job(job.id)

    assert refreshed is not None
    assert refreshed.status == TrainingJobStatus.completed
    assert refreshed.state == TrainingJobState.completed
    assert refreshed.stage == "Completed"
    assert refreshed.progress == 1.0


def test_get_logs_prepends_runpod_logs_and_limits_after_merge(tmp_path: Path, monkeypatch) -> None:
    manager, _store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job = make_stored_job(tmp_path)
    (tmp_path / "runpod_lifecycle.log").write_text("life\n", encoding="utf-8")
    (tmp_path / "runpod_startup.log").write_text("startup\n", encoding="utf-8")
    (tmp_path / "runpod_agent.log").write_text("agent\n", encoding="utf-8")
    (tmp_path / "runpod_runner.log").write_text("runner\n", encoding="utf-8")
    (tmp_path / "stdout.log").write_text("stdout-1\nstdout-2\n", encoding="utf-8")
    (tmp_path / "stderr.log").write_text("stderr\n", encoding="utf-8")
    manager.get_job = lambda _job_id: job

    logs = TrainingRunManager.get_logs(manager, job.id, lines=3)

    assert logs.stdout_lines == ["runner", "stdout-1", "stdout-2"]
    assert logs.stderr_lines == ["stderr"]


def test_get_logs_redacts_active_arbitrary_dataset_token(tmp_path: Path, monkeypatch) -> None:
    manager, _store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job = make_stored_job(tmp_path)
    secret = "legacy-private-token-without-provider-prefix"
    (tmp_path / "stdout.log").write_text(f"provider echoed {secret}\n", encoding="utf-8")
    (tmp_path / "stderr.log").write_text(f"retry used {secret}\n", encoding="utf-8")
    manager.get_job = lambda _job_id: job
    manager._remember_dataset_hf_tokens(job.id, [secret])

    try:
        logs = TrainingRunManager.get_logs(manager, job.id)
    finally:
        manager._forget_dataset_hf_tokens(job.id)

    assert secret not in "\n".join([*logs.stdout_lines, *logs.stderr_lines])
    assert "[REDACTED]" in "\n".join([*logs.stdout_lines, *logs.stderr_lines])


def test_terminal_refresh_scrubs_arbitrary_dataset_token_before_scope_clear(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "job123456"
    job_dir.mkdir(parents=True)
    job = make_stored_job(job_dir)
    job.executor_kind = "local"
    store.create_job(job)
    secret = "legacy-private-token-without-provider-prefix"
    (job_dir / "runtime_state.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "state": "failed",
                "stage": "Failed",
                "progress": 1.0,
                "error": f"provider echoed {secret}",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "metadata.json").write_text(
        json.dumps({"error": f"metadata echoed {secret}"}),
        encoding="utf-8",
    )
    (job_dir / "stdout.log").write_text(f"stdout echoed {secret}\n", encoding="utf-8")
    manager._remember_dataset_hf_tokens(job.id, [secret])

    refreshed = manager._refresh_job(job.id)

    assert refreshed is not None
    assert refreshed.status == TrainingJobStatus.failed
    assert refreshed.error is not None and secret not in refreshed.error
    assert secret not in (job_dir / "runtime_state.json").read_text(encoding="utf-8")
    assert secret not in (job_dir / "metadata.json").read_text(encoding="utf-8")
    assert secret not in (job_dir / "stdout.log").read_text(encoding="utf-8")
    assert manager._known_dataset_hf_tokens(job.id) == ()


def test_terminal_refresh_retains_redaction_scope_when_scrub_cannot_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "job123456"
    job_dir.mkdir(parents=True)
    job = make_stored_job(job_dir)
    job.executor_kind = "local"
    store.create_job(job)
    secret = "legacy-private-token-without-provider-prefix"
    (job_dir / "runtime_state.json").write_text(
        json.dumps({"status": "failed", "state": "failed", "error": f"echoed {secret}"}),
        encoding="utf-8",
    )
    manager._remember_dataset_hf_tokens(job.id, [secret])
    manager._scrub_job_text_files = lambda _job: False

    refreshed = manager._refresh_job(job.id)

    assert refreshed is not None and refreshed.status == TrainingJobStatus.failed
    assert manager._known_dataset_hf_tokens(job.id) == (secret,)
    manager._forget_dataset_hf_tokens(job.id)


def test_artifact_bundle_scrubs_arbitrary_dataset_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "job123456"
    job_dir.mkdir(parents=True)
    job = make_stored_job(job_dir, status=TrainingJobStatus.completed)
    job.executor_kind = "local"
    job.state = TrainingJobState.completed
    job.executor_status = "completed"
    store.create_job(job)
    secret = "legacy-private-token-without-provider-prefix"
    (job_dir / "stdout.log").write_text(f"provider echoed {secret}\n", encoding="utf-8")
    (job_dir / "metadata.json").write_text(
        json.dumps({"error": f"metadata echoed {secret}"}),
        encoding="utf-8",
    )
    exports_dir = tmp_path / "exports"
    monkeypatch.setattr("app.training_jobs.training_exports_dir", lambda: exports_dir)
    manager._remember_dataset_hf_tokens(job.id, [secret])

    try:
        archive = manager.build_artifact_bundle(job.id)
    finally:
        manager._forget_dataset_hf_tokens(job.id)

    with zipfile.ZipFile(archive) as bundle:
        combined = b"".join(bundle.read(name) for name in bundle.namelist())
    assert secret.encode() not in combined


def test_artifact_bundle_is_blocked_when_credential_scrub_cannot_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job_dir = tmp_path / "jobs" / "job123456"
    job_dir.mkdir(parents=True)
    job = make_stored_job(job_dir, status=TrainingJobStatus.completed)
    job.executor_kind = "local"
    job.state = TrainingJobState.completed
    job.executor_status = "completed"
    store.create_job(job)
    secret = "legacy-private-token-without-provider-prefix"
    manager._remember_dataset_hf_tokens(job.id, [secret])
    manager._scrub_job_text_files = lambda _job: False

    try:
        manager.build_artifact_bundle(job.id)
    except RuntimeError as exc:
        assert "credential-bearing managed output" in str(exc)
    else:
        raise AssertionError("Expected artifact export to fail closed")
    finally:
        manager._forget_dataset_hf_tokens(job.id)


def test_delete_job_preserves_stop_before_delete_rule(tmp_path: Path, monkeypatch) -> None:
    manager, store, local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    job = make_stored_job(tmp_path)
    job.executor_kind = "local"
    store.create_job(job)

    try:
        manager.delete_job(job.id)
    except RuntimeError as exc:
        assert "Stop the training job before deleting it" in str(exc)
    else:
        raise AssertionError("Expected deleting a running job to fail")

    assert store.get_job(job.id) is job
    assert store.deleted == []
    assert local_executor.stopped == []


def test_delete_job_rejects_stored_artifact_directory_escape(tmp_path: Path, monkeypatch) -> None:
    from app.storage_safety import UnsafeManagedPathError

    manager, store, _local_executor, _runpod_executor = make_manager(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    job = make_stored_job(outside, status=TrainingJobStatus.failed)
    store.create_job(job)

    try:
        manager.delete_job(job.id)
    except UnsafeManagedPathError:
        pass
    else:
        raise AssertionError("Expected a stored artifact directory escape to be rejected")

    assert marker.exists()
    assert store.get_job(job.id) is job
