from __future__ import annotations

import importlib
import io
import json
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from app.training_executors import remote_sync
from app.training_runs.executors.runpod import agent_client
from app.training_executors.base import CleanupPolicy, ExecutionSnapshot
from app.training_executors.remote_sync import (
    DEFAULT_POD_AGENT_USER_AGENT,
    RemoteAgentClient,
    RemoteAgentError,
    build_remote_bundle,
    rewrite_local_dataset_files,
)
from app.training_executors.runpod_client import CreatePodRequest, RunPodClientError
from app.training_executors.runpod_pod import (
    RunPodPodExecutor,
    _bundle_upload_error,
    build_agent_base_url,
    terminal_cleanup_policy,
)
from app.training_models import TrainingJobState, TrainingJobStatus
from app.training_storage import StoredTrainingJob, TrainingStudioStore

REPO_ROOT = Path(__file__).resolve().parents[4]


class ExitedProcess:
    pid = 12345

    def poll(self) -> int:
        return 1


class FakeCompatibleAgent:
    def system(self) -> dict[str, object]:
        return {"runner": {"import_ok": True}}


class FakeProtocolAwareAgent:
    def system(self) -> dict[str, object]:
        return {
            "agent_protocol_version": 1,
            "bundle_format_versions": ["llm-studio-training-bundle-v1"],
            "supports_optional_files": True,
            "supports_checkpoint_manifest": True,
            "runner": {"import_ok": True},
        }


class FakeProtocolAgentMissingCheckpointManifest:
    def system(self) -> dict[str, object]:
        return {
            "agent_protocol_version": 1,
            "bundle_format_versions": ["llm-studio-training-bundle-v1"],
            "supports_optional_files": True,
            "supports_checkpoint_manifest": False,
            "runner": {"import_ok": True},
        }


class FakeOldAgent:
    def system(self) -> dict[str, object]:
        return {"workspace": "/workspace/llm-studio"}


class FakeBrokenAgent:
    def system(self) -> dict[str, object]:
        return {"runner": {"import_ok": False, "error": "ModuleNotFoundError: No module named 'training.local_text_data'"}}


class FakeLegacyAgentWithoutSystem:
    def system(self) -> dict[str, object]:
        raise RemoteAgentError("Pod agent request failed with HTTP 404: ", status_code=404, retryable=True)


class FakeAgentWithLegacySystemAuthBug:
    def system(self) -> dict[str, object]:
        raise RemoteAgentError(
            'Pod agent request failed with HTTP 422: {"detail":[{"type":"missing","loc":["query","job_id"]}]}',
            status_code=422,
            payload={"detail": [{"type": "missing", "loc": ["query", "job_id"]}]},
            retryable=False,
        )


class FakeAgentWithBrokenSystemDiagnostics:
    def system(self) -> dict[str, object]:
        raise RemoteAgentError("Pod agent request failed with HTTP 500: Internal Server Error", status_code=500)


class FakeAgentWithAuthFailure:
    def system(self) -> dict[str, object]:
        raise RemoteAgentError("Pod agent request failed with HTTP 403: forbidden", status_code=403, retryable=False)


class FakeCleanupClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def stop_pod(self, pod_id: str) -> None:
        self.calls.append(("stop", pod_id))

    def delete_pod(self, pod_id: str) -> None:
        self.calls.append(("delete", pod_id))


class FakeRefreshStore:
    def __init__(self, job: StoredTrainingJob) -> None:
        self.job = job
        self.updates: list[dict[str, object]] = []

    def get_job(self, job_id: str) -> StoredTrainingJob | None:
        return self.job if job_id == self.job.id else None

    def update_job(self, job_id: str, **updates: object) -> StoredTrainingJob | None:
        if job_id != self.job.id:
            return None
        self.updates.append(updates)
        for key, value in updates.items():
            setattr(self.job, key, value)
        return self.job


class FakeRefreshExecutor:
    kind = "runpod_pod"

    def __init__(self) -> None:
        self.calls = 0

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        self.calls += 1
        return ExecutionSnapshot(updates={"executor_status": "running"})


def stored_runpod_job(tmp_path: Path, *, status: TrainingJobStatus = TrainingJobStatus.running) -> StoredTrainingJob:
    failed = status == TrainingJobStatus.failed
    return StoredTrainingJob(
        id="job123456",
        name="RunPod job",
        status=status,
        state=TrainingJobState.failed if failed else TrainingJobState.preflight,
        stage="RunPod launch failed" if failed else "Provisioning RunPod pod",
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
        error="RunPod launch failed" if failed else None,
        process_id=None,
        output_size_bytes=0,
        executor_kind="runpod_pod",
        executor_status="failed" if failed else "provisioning",
    )


def test_training_store_migrates_old_sqlite_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "training.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE llm_training_jobs (
            id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            status VARCHAR(32) NOT NULL,
            state VARCHAR(64) NOT NULL,
            stage VARCHAR(255) NOT NULL,
            progress FLOAT NOT NULL,
            created_at DATETIME NOT NULL,
            started_at DATETIME,
            finished_at DATETIME,
            project_id VARCHAR(64) NOT NULL,
            project_name VARCHAR(255) NOT NULL,
            tokenizer_job_id VARCHAR(64) NOT NULL,
            tokenizer_name VARCHAR(255) NOT NULL,
            model_config JSON NOT NULL,
            training_config JSON NOT NULL,
            dataloader_config JSON NOT NULL,
            resolved_runtime JSON,
            memory_estimate JSON,
            artifact_dir TEXT NOT NULL,
            artifact_bundle_file VARCHAR(512),
            stats_path TEXT NOT NULL,
            samples_path TEXT NOT NULL,
            stdout_path TEXT NOT NULL,
            stderr_path TEXT NOT NULL,
            last_step INTEGER NOT NULL DEFAULT 0,
            max_steps INTEGER NOT NULL DEFAULT 0,
            latest_loss FLOAT,
            latest_grad_norm FLOAT,
            latest_lr FLOAT,
            latest_tokens_per_sec FLOAT,
            checkpoint_count INTEGER NOT NULL DEFAULT 0,
            sample_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            process_id INTEGER,
            output_size_bytes INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        """
        INSERT INTO llm_training_jobs (
            id, name, status, state, stage, progress, created_at, project_id, project_name,
            tokenizer_job_id, tokenizer_name, model_config, training_config, dataloader_config,
            artifact_dir, stats_path, samples_path, stdout_path, stderr_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job123456",
            "Old job",
            "completed",
            "completed",
            "Completed",
            1.0,
            datetime.now(timezone.utc).isoformat(),
            "project123",
            "Project",
            "tok123456",
            "Tokenizer",
            "{}",
            "{}",
            "{}",
            str(tmp_path),
            str(tmp_path / "stats.jsonl"),
            str(tmp_path / "samples.jsonl"),
            str(tmp_path / "stdout.log"),
            str(tmp_path / "stderr.log"),
        ),
    )
    connection.commit()
    connection.close()

    store = TrainingStudioStore(url=f"sqlite:///{db_path}")
    store.initialize()
    store.initialize()
    job = store.get_job("job123456")
    with sqlite3.connect(db_path) as migrated_connection:
        migrated_columns = [row[1] for row in migrated_connection.execute("PRAGMA table_info(llm_training_jobs)")]

    assert job is not None
    assert job.executor_kind == "local"
    assert job.runpod_gpu_count == 1
    assert job.runpod_pod_id is None
    assert job.runpod_network_volume_id is None
    assert job.runpod_cost_per_hr is None
    assert job.runpod_agent_base_url is None
    assert job.runpod_cleanup_policy is None
    assert job.remote_workspace_path is None
    assert len(migrated_columns) == len(set(migrated_columns))
    assert "remote_error" in migrated_columns


def test_training_store_skips_sqlite_migrations_for_non_sqlite_urls(monkeypatch) -> None:
    called = False

    def fake_apply_sqlite_migrations(_engine: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("app.training_runs.store.apply_sqlite_migrations", fake_apply_sqlite_migrations)
    store = TrainingStudioStore.__new__(TrainingStudioStore)
    store._url = "postgresql://example.test/db"
    store._engine = object()

    store._migrate_schema()

    assert called is False


def test_incomplete_runpod_restart_recovery_is_explicit_without_failing_job(tmp_path: Path) -> None:
    store = TrainingStudioStore(url=f"sqlite:///{tmp_path / 'training.db'}")
    store.initialize()
    job = stored_runpod_job(tmp_path / "artifacts", status=TrainingJobStatus.running)
    store.create_job(job)

    count = store.mark_incomplete_runpod_jobs_recovery_limited("token unavailable after restart")
    recovered = store.get_job(job.id)

    assert count == 1
    assert recovered is not None
    assert recovered.status == TrainingJobStatus.running
    assert recovered.executor_status == "running"
    assert recovered.remote_error == "token unavailable after restart"


def test_runpod_create_pod_payload_uses_current_image_name_field() -> None:
    payload = CreatePodRequest(
        name="llm-studio-test",
        image_name="ghcr.io/example/llm-builder-training:latest",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        cloud_type="SECURE",
        data_center_id="US-KS-2",
        container_disk_gb=50,
        volume_gb=100,
        volume_mount_path="/workspace",
        ports=["8021/http"],
        env={"LLM_STUDIO_REMOTE_JOB_ID": "job123"},
    ).to_payload()

    assert payload["imageName"] == "ghcr.io/example/llm-builder-training:latest"
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["ports"] == ["8021/http"]
    assert "image" not in payload


def test_runpod_cleanup_policy_normalizes_unsupported_network_volume_delete() -> None:
    from app.training_models import RunPodCleanupPolicy

    policy = RunPodCleanupPolicy.model_validate(
        {"pod": "delete_after_sync", "network_volume": "delete_after_sync"}
    )

    assert policy.network_volume == "keep"


def test_runpod_agent_url_prefers_http_proxy_for_declared_http_port() -> None:
    pod = {
        "id": "abc123xyz",
        "publicIp": "100.65.0.119",
        "ports": ["8021/http"],
        "portMappings": {},
    }

    assert build_agent_base_url(pod, 8021) == "https://abc123xyz-8021.proxy.runpod.net"


def test_runpod_agent_url_uses_proxy_for_runtime_http_mapping() -> None:
    pod = {
        "id": "ldl1dxirsim64n",
        "runtime": {
            "ports": [
                {
                    "ip": "100.65.0.101",
                    "isIpPublic": False,
                    "privatePort": 8021,
                    "publicPort": 60141,
                    "type": "http",
                }
            ]
        },
    }

    assert build_agent_base_url(pod, 8021) == "https://ldl1dxirsim64n-8021.proxy.runpod.net"


def test_runpod_agent_url_uses_direct_tcp_port_mapping() -> None:
    pod = {
        "id": "tcp123",
        "publicIp": "213.173.109.39",
        "ports": ["8021/tcp"],
        "portMappings": {"8021": 13007},
    }

    assert build_agent_base_url(pod, 8021) == "http://213.173.109.39:13007"


def test_runpod_agent_url_uses_public_ip_for_runtime_tcp_mapping() -> None:
    pod = {
        "id": "tcp123",
        "publicIp": "213.173.109.39",
        "runtime": {
            "ports": [
                {
                    "ip": "100.65.0.101",
                    "isIpPublic": False,
                    "privatePort": 8021,
                    "publicPort": 13007,
                    "type": "tcp",
                }
            ]
        },
    }

    assert build_agent_base_url(pod, 8021) == "http://213.173.109.39:13007"


def test_runpod_agent_url_prefers_explicit_runtime_url_over_declared_port() -> None:
    pod = {
        "id": "ldl1dxirsim64n",
        "ports": ["8021/http"],
        "runtime": {
            "ports": [
                {
                    "privatePort": 8021,
                    "type": "http",
                    "url": "https://agent.example.test/",
                }
            ]
        },
    }

    assert build_agent_base_url(pod, 8021) == "https://agent.example.test"


def test_remote_agent_client_uses_certifi_ssl_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(request: object, **kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        captured["user_agent"] = request.get_header("User-agent")  # type: ignore[attr-defined]
        captured["authorization"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        return FakeResponse()

    monkeypatch.setattr(agent_client, "urlopen", fake_urlopen)

    payload = RemoteAgentClient("https://example.test", "token", "job123").health()

    assert payload == {"ok": True}
    assert captured["context"] is not None
    assert captured["user_agent"] == DEFAULT_POD_AGENT_USER_AGENT
    assert captured["authorization"] is None


def test_remote_agent_client_system_supports_legacy_query_job_auth(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"runner": {"import_ok": true}}'

    def fake_urlopen(request: object, **kwargs: object) -> FakeResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["authorization"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        captured["job_header"] = request.get_header("X-llm-studio-job-id")  # type: ignore[attr-defined]
        return FakeResponse()

    monkeypatch.setattr(agent_client, "urlopen", fake_urlopen)

    payload = RemoteAgentClient("https://example.test", "token", "job123").system()

    assert payload == {"runner": {"import_ok": True}}
    assert captured["url"] == "https://example.test/v1/system?job_id=job123"
    assert captured["authorization"] == "Bearer token"
    assert captured["job_header"] == "job123"


def test_remote_agent_client_download_optional_file_sends_optional_query(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"optional": true}'

    def fake_urlopen(request: object, **_kwargs: object) -> FakeResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        return FakeResponse()

    monkeypatch.setattr(agent_client, "urlopen", fake_urlopen)

    target = tmp_path / "optional_status.json"
    raw = RemoteAgentClient("https://example.test", "token", "job123").download_file(
        "optional_status.json",
        target,
        optional=True,
    )

    assert raw == b'{"optional": true}'
    assert target.read_text(encoding="utf-8") == '{"optional": true}'
    assert captured["url"] == (
        "https://example.test/v1/jobs/job123/files?path=optional_status.json&offset=0&optional=1"
    )


def test_remote_agent_client_optional_file_404_is_empty(monkeypatch, tmp_path: Path) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise HTTPError(
            "https://example.test/v1/jobs/job123/files?path=optional_status.json&optional=1",
            404,
            "Not Found",
            hdrs=None,
            fp=io.BytesIO(b'{"detail": "File is not available."}'),
        )

    monkeypatch.setattr(agent_client, "urlopen", fake_urlopen)

    target = tmp_path / "optional_status.json"
    raw = RemoteAgentClient("https://example.test", "token", "job123").download_file(
        "optional_status.json",
        target,
        optional=True,
    )

    assert raw == b""
    assert not target.exists()


def test_remote_agent_client_marks_cloudflare_1010_as_non_retryable(monkeypatch) -> None:
    cloudflare_payload = {
        "title": "Error 1010: Access denied",
        "status": 403,
        "error_code": 1010,
        "error_name": "browser_signature_banned",
        "cloudflare_error": True,
        "retryable": False,
    }

    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise HTTPError(
            "https://pod-8021.proxy.runpod.net/health",
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(json.dumps(cloudflare_payload).encode("utf-8")),
        )

    monkeypatch.setattr(agent_client, "urlopen", fake_urlopen)

    try:
        RemoteAgentClient("https://pod-8021.proxy.runpod.net", "token", "job123").health()
    except RemoteAgentError as exc:
        assert exc.status_code == 403
        assert exc.retryable is False
        assert "Cloudflare 1010" in str(exc)
    else:
        raise AssertionError("Expected Cloudflare 1010 to raise RemoteAgentError")


def test_runpod_agent_health_stops_on_non_retryable_error(monkeypatch, tmp_path: Path) -> None:
    class NonRetryableHealthAgent:
        attempts = 0

        def health(self) -> dict[str, object]:
            self.attempts += 1
            raise RemoteAgentError("Cloudflare 1010", status_code=403, retryable=False)

    agent = NonRetryableHealthAgent()
    monkeypatch.setattr("app.training_executors.runpod_pod.time.sleep", lambda _seconds: None)

    try:
        RunPodPodExecutor()._wait_for_agent(agent, job=stored_runpod_job(tmp_path))  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "failed permanently" in str(exc)
    else:
        raise AssertionError("Expected permanent agent health failure")
    assert agent.attempts == 1


def test_runpod_refresh_does_not_revert_terminal_failure_to_provisioning(tmp_path: Path) -> None:
    snapshot = RunPodPodExecutor().refresh(stored_runpod_job(tmp_path, status=TrainingJobStatus.failed))

    assert snapshot.status is None
    assert snapshot.updates == {}


def test_training_manager_coalesces_back_to_back_runpod_refreshes(tmp_path: Path) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from app.training_jobs import TrainingRunManager

    job = stored_runpod_job(tmp_path, status=TrainingJobStatus.running)
    store = FakeRefreshStore(job)
    executor = FakeRefreshExecutor()
    manager = TrainingRunManager.__new__(TrainingRunManager)
    manager._store = store
    manager._runpod_executor = executor
    manager._executors = {executor.kind: executor}
    manager._refresh_locks = {}
    manager._refresh_locks_guard = threading.Lock()
    manager._last_runpod_refresh_at = {}

    first = manager._refresh_job(job.id)
    second = manager._refresh_job(job.id)

    assert first is job
    assert second is job
    assert executor.calls == 1


def test_training_manager_marks_executor_status_from_terminal_runtime_state(tmp_path: Path) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from app.training_jobs import TrainingRunManager

    job = stored_runpod_job(tmp_path, status=TrainingJobStatus.running)
    job.executor_status = "running"
    (tmp_path / "runtime_state.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "state": "failed",
                "stage": "Failed",
                "progress": 1.0,
                "error": "TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'",
            }
        ),
        encoding="utf-8",
    )
    store = FakeRefreshStore(job)
    executor = FakeRefreshExecutor()
    manager = TrainingRunManager.__new__(TrainingRunManager)
    manager._store = store
    manager._runpod_executor = executor
    manager._executors = {executor.kind: executor}
    manager._refresh_locks = {}
    manager._refresh_locks_guard = threading.Lock()
    manager._last_runpod_refresh_at = {job.id: float("inf")}

    refreshed = manager._refresh_job(job.id)

    assert refreshed is job
    assert refreshed.status == TrainingJobStatus.failed
    assert refreshed.executor_status == "failed"
    assert executor.calls == 0


def test_runpod_submit_failure_cleanup_stops_instead_of_deleting_by_default(tmp_path: Path) -> None:
    client = FakeCleanupClient()

    RunPodPodExecutor()._cleanup_after_submit_failure(  # type: ignore[arg-type]
        client,
        stored_runpod_job(tmp_path),
        "pod123",
        cleanup_policy={"pod": "delete_after_sync"},
    )

    assert client.calls == [("stop", "pod123")]


def test_runpod_submit_failure_cleanup_honors_keep_policy(tmp_path: Path) -> None:
    client = FakeCleanupClient()

    RunPodPodExecutor()._cleanup_after_submit_failure(  # type: ignore[arg-type]
        client,
        stored_runpod_job(tmp_path),
        "pod123",
        cleanup_policy={"pod": "keep"},
    )

    assert client.calls == []


def test_runpod_cleanup_uses_ui_api_key_kept_in_memory(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}
    client = FakeCleanupClient()

    class FakeRunPodClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        def stop_pod(self, pod_id: str) -> None:
            client.stop_pod(pod_id)

        def delete_pod(self, pod_id: str) -> None:
            client.delete_pod(pod_id)

    monkeypatch.setattr("app.training_executors.runpod_pod.RunPodClient", FakeRunPodClient)
    executor = RunPodPodExecutor()
    job = stored_runpod_job(tmp_path)
    job.runpod_pod_id = "pod123"
    executor._api_keys[job.id] = "ui-key"

    executor.cleanup(job, CleanupPolicy(pod="delete_after_sync"))

    assert captured["api_key"] == "ui-key"
    assert client.calls == [("delete", "pod123")]


def test_runpod_failed_terminal_cleanup_stops_pod_for_inspection(tmp_path: Path) -> None:
    job = stored_runpod_job(tmp_path, status=TrainingJobStatus.failed)
    job.runpod_cleanup_policy = {"pod": "delete_after_sync", "network_volume": "keep"}

    policy = terminal_cleanup_policy(job, TrainingJobStatus.failed)

    assert policy.pod == "stop_after_sync"
    assert policy.network_volume == "keep"


def test_runpod_completed_terminal_cleanup_honors_delete_policy(tmp_path: Path) -> None:
    job = stored_runpod_job(tmp_path, status=TrainingJobStatus.completed)
    job.runpod_cleanup_policy = {"pod": "delete_after_sync", "network_volume": "keep"}

    policy = terminal_cleanup_policy(job, TrainingJobStatus.completed)

    assert policy.pod == "delete_after_sync"


def test_runpod_refresh_cleans_up_terminal_job_with_stale_executor_status(monkeypatch, tmp_path: Path) -> None:
    client = FakeCleanupClient()

    class FakeRunPodClient:
        def __init__(self, _api_key: str) -> None:
            pass

        def stop_pod(self, pod_id: str) -> None:
            client.stop_pod(pod_id)

        def delete_pod(self, pod_id: str) -> None:
            client.delete_pod(pod_id)

    monkeypatch.setattr("app.training_executors.runpod_pod.RunPodClient", FakeRunPodClient)
    executor = RunPodPodExecutor()
    job = stored_runpod_job(tmp_path, status=TrainingJobStatus.failed)
    job.executor_status = "running"
    job.runpod_pod_id = "pod123"
    job.runpod_cleanup_policy = {"pod": "delete_after_sync", "network_volume": "keep"}
    executor._api_keys[job.id] = "ui-key"

    snapshot = executor.refresh(job)

    assert snapshot.updates["executor_status"] == "failed"
    assert client.calls == [("stop", "pod123")]
    assert job.id not in executor._api_keys


def test_runpod_create_pod_retries_transient_capacity_errors(monkeypatch, tmp_path: Path) -> None:
    class FlakyCreatePodClient:
        def __init__(self) -> None:
            self.calls = 0

        def create_pod(self, _request: CreatePodRequest) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise RunPodClientError("create pod: There are no instances currently available", status_code=500)
            return {"id": "pod123"}

    client = FlakyCreatePodClient()
    request = CreatePodRequest(
        name="llm-studio-test",
        image_name="ghcr.io/example/llm-builder-training:latest",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        cloud_type="SECURE",
        container_disk_gb=50,
        volume_gb=100,
        volume_mount_path="/workspace",
        ports=["8021/tcp"],
        env={},
    )
    monkeypatch.setattr("app.training_executors.runpod_pod.time.sleep", lambda _seconds: None)

    pod = RunPodPodExecutor()._create_pod_with_retries(  # type: ignore[arg-type]
        client,
        request,
        job=stored_runpod_job(tmp_path),
    )

    assert pod == {"id": "pod123"}
    assert client.calls == 2


def test_runpod_bundle_upload_proxy_404_points_to_tcp_transport() -> None:
    error = _bundle_upload_error(
        RemoteAgentError("Pod agent request failed with HTTP 404: ", status_code=404),
        agent_base_url="https://pod-8021.proxy.runpod.net",
    )

    assert isinstance(error, RuntimeError)
    assert "LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL=tcp" in str(error)


def test_runpod_bundle_upload_direct_404_points_to_training_image() -> None:
    error = _bundle_upload_error(
        RemoteAgentError("Pod agent request failed with HTTP 404: ", status_code=404),
        agent_base_url="http://213.173.109.39:13007",
    )

    assert isinstance(error, RuntimeError)
    assert "stale or wrong training image" in str(error)


def test_training_image_includes_shared_local_text_module() -> None:
    dockerfile = (REPO_ROOT / "docker" / "training" / "Dockerfile").read_text(encoding="utf-8")

    assert "build-essential" in dockerfile
    assert "CC=/usr/bin/gcc" in dockerfile
    assert "CXX=/usr/bin/g++" in dockerfile
    assert "LLM_STUDIO_TORCH_COMPILE=0" in dockerfile
    assert "assert shutil.which('gcc')" in dockerfile
    assert "COPY training ./training" in dockerfile
    assert "import training.local_text_data; import training.runner" in dockerfile


def test_training_entrypoint_runs_startup_diagnostics() -> None:
    entrypoint = (REPO_ROOT / "docker" / "training" / "entrypoint.sh").read_text(encoding="utf-8")
    diagnostics = (REPO_ROOT / "apps" / "llm-studio" / "remote_agent" / "diagnostics.py").read_text(encoding="utf-8")

    assert "python -m remote_agent.diagnostics startup" in entrypoint
    assert "startup.log" in entrypoint
    assert "uvicorn_start" in entrypoint
    assert "system_tools" in diagnostics
    assert '"gcc"' in diagnostics
    assert '"tokenizers": True' in diagnostics
    assert "transformers" not in diagnostics


def test_default_runpod_training_image_does_not_point_at_stale_import_broken_tag() -> None:
    config_source = (REPO_ROOT / "apps" / "llm-studio" / "api" / "app" / "config.py").read_text(encoding="utf-8")
    env_example = (REPO_ROOT / "apps" / "llm-studio" / "api" / ".env.example").read_text(encoding="utf-8")

    assert "ghcr.io/pabixn/llm-builder-training:sha-7037615" not in config_source
    assert "ghcr.io/pabixn/llm-builder-training:sha-7037615" not in env_example
    assert "ghcr.io/pabixn/llm-builder-training:latest" in config_source
    assert "LLM_STUDIO_RUNPOD_TRAINING_IMAGE=ghcr.io/pabixn/llm-builder-training:latest" in env_example
    assert 'default="tcp"' in config_source
    assert "LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL=tcp" in env_example


def test_runpod_executor_rejects_agent_without_runner_compatibility_report() -> None:
    executor = RunPodPodExecutor()

    try:
        executor._verify_agent_compatibility(FakeOldAgent())  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "too old" in str(exc)
    else:
        raise AssertionError("Expected old agent compatibility check to fail")


def test_runpod_executor_rejects_agent_with_broken_runner_import() -> None:
    executor = RunPodPodExecutor()

    try:
        executor._verify_agent_compatibility(FakeBrokenAgent())  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "cannot import the training runner" in str(exc)
        assert "training.local_text_data" in str(exc)
    else:
        raise AssertionError("Expected broken runner compatibility check to fail")


def test_runpod_executor_accepts_agent_with_runner_compatibility_report() -> None:
    RunPodPodExecutor()._verify_agent_compatibility(FakeCompatibleAgent())  # type: ignore[arg-type]


def test_runpod_executor_accepts_protocol_aware_agent() -> None:
    RunPodPodExecutor()._verify_agent_compatibility(FakeProtocolAwareAgent())  # type: ignore[arg-type]


def test_runpod_executor_rejects_protocol_agent_without_checkpoint_manifest() -> None:
    try:
        RunPodPodExecutor()._verify_agent_compatibility(
            FakeProtocolAgentMissingCheckpointManifest()  # type: ignore[arg-type]
        )
    except RuntimeError as exc:
        assert "checkpoint manifests" in str(exc)
    else:
        raise AssertionError("Expected incompatible protocol capabilities to fail")


def test_runpod_executor_allows_legacy_agent_without_system_endpoint(tmp_path: Path) -> None:
    RunPodPodExecutor()._verify_agent_compatibility(
        FakeLegacyAgentWithoutSystem(),  # type: ignore[arg-type]
        job=stored_runpod_job(tmp_path),
    )


def test_runpod_executor_allows_legacy_agent_system_query_bug(tmp_path: Path) -> None:
    RunPodPodExecutor()._verify_agent_compatibility(
        FakeAgentWithLegacySystemAuthBug(),  # type: ignore[arg-type]
        job=stored_runpod_job(tmp_path),
    )


def test_runpod_executor_allows_broken_system_diagnostics(tmp_path: Path) -> None:
    RunPodPodExecutor()._verify_agent_compatibility(
        FakeAgentWithBrokenSystemDiagnostics(),  # type: ignore[arg-type]
        job=stored_runpod_job(tmp_path),
    )


def test_runpod_executor_rejects_system_auth_failure(tmp_path: Path) -> None:
    try:
        RunPodPodExecutor()._verify_agent_compatibility(
            FakeAgentWithAuthFailure(),  # type: ignore[arg-type]
            job=stored_runpod_job(tmp_path),
        )
    except RuntimeError as exc:
        assert "authenticated system check failed" in str(exc)
    else:
        raise AssertionError("Expected system auth failure to fail compatibility check")


def test_remote_agent_system_auth_accepts_header_without_query_job_id(monkeypatch, tmp_path: Path) -> None:
    llm_studio_root = REPO_ROOT / "apps" / "llm-studio"
    if str(llm_studio_root) not in sys.path:
        sys.path.insert(0, str(llm_studio_root))
    job_id = "job-system-auth"
    monkeypatch.setenv("LLM_STUDIO_REMOTE_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("LLM_STUDIO_REMOTE_JOB_ID", job_id)
    monkeypatch.setenv("LLM_STUDIO_REMOTE_AGENT_TOKEN", "token")
    import remote_agent.app as remote_app

    remote_app = importlib.reload(remote_app)
    client = TestClient(remote_app.app)
    headers = {"Authorization": "Bearer token", "X-LLM-Studio-Job-Id": job_id}

    system_response = client.get("/v1/system", headers=headers)
    mismatched_job_response = client.get("/v1/jobs/not-this-job/runtime-state", headers=headers)

    assert system_response.status_code == 200
    assert system_response.json()["job_id"] == job_id
    assert system_response.json()["agent_protocol_version"] == 1
    assert "llm-studio-training-bundle-v1" in system_response.json()["bundle_format_versions"]
    assert system_response.json()["supports_optional_files"] is True
    assert mismatched_job_response.status_code == 403


def test_remote_agent_runtime_state_reports_early_process_exit(monkeypatch, tmp_path: Path) -> None:
    llm_studio_root = REPO_ROOT / "apps" / "llm-studio"
    if str(llm_studio_root) not in sys.path:
        sys.path.insert(0, str(llm_studio_root))
    import remote_agent.app as remote_app
    import remote_agent.runner as remote_runner

    remote_runner = importlib.reload(remote_runner)
    remote_app = importlib.reload(remote_app)
    job_id = "job-early-exit"
    monkeypatch.setenv("LLM_STUDIO_REMOTE_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("LLM_STUDIO_REMOTE_JOB_ID", job_id)
    monkeypatch.setenv("LLM_STUDIO_REMOTE_AGENT_TOKEN", "token")
    outputs = tmp_path / "workspace" / "jobs" / job_id / "outputs"
    outputs.mkdir(parents=True)
    remote_app.runner._processes[job_id] = remote_runner.RemoteProcess(
        process=ExitedProcess(),
        stdout_path=outputs / "stdout.log",
        stderr_path=outputs / "stderr.log",
    )

    client = TestClient(remote_app.app)
    response = client.get(
        f"/v1/jobs/{job_id}/runtime-state",
        headers={"Authorization": "Bearer token", "X-LLM-Studio-Job-Id": job_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["state"] == "failed"
    assert payload["process"]["exit_code"] == 1
    assert "stderr.log" in payload["error"]


def test_remote_agent_exposes_container_diagnostic_logs(monkeypatch, tmp_path: Path) -> None:
    llm_studio_root = REPO_ROOT / "apps" / "llm-studio"
    if str(llm_studio_root) not in sys.path:
        sys.path.insert(0, str(llm_studio_root))
    import remote_agent.app as remote_app

    remote_app = importlib.reload(remote_app)
    job_id = "job-diagnostics"
    workspace = tmp_path / "workspace"
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "startup.log").write_text("startup ready\n", encoding="utf-8")
    (logs_dir / "agent.log").write_text("agent ready\n", encoding="utf-8")
    (logs_dir / "runner.log").write_text("runner ready\n", encoding="utf-8")
    monkeypatch.setenv("LLM_STUDIO_REMOTE_WORKSPACE", str(workspace))
    monkeypatch.setenv("LLM_STUDIO_REMOTE_JOB_ID", job_id)
    monkeypatch.setenv("LLM_STUDIO_REMOTE_AGENT_TOKEN", "token")

    client = TestClient(remote_app.app)
    headers = {"Authorization": "Bearer token", "X-LLM-Studio-Job-Id": job_id}

    assert client.get(f"/v1/jobs/{job_id}/logs/startup", headers=headers).text == "startup ready\n"
    assert client.get(f"/v1/jobs/{job_id}/logs/agent", headers=headers).text == "agent ready\n"
    assert client.get(f"/v1/jobs/{job_id}/logs/runner", headers=headers).text == "runner ready\n"


def test_remote_agent_optional_missing_file_returns_empty_200(monkeypatch, tmp_path: Path) -> None:
    llm_studio_root = REPO_ROOT / "apps" / "llm-studio"
    if str(llm_studio_root) not in sys.path:
        sys.path.insert(0, str(llm_studio_root))
    job_id = "job-optional-file"
    monkeypatch.setenv("LLM_STUDIO_REMOTE_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("LLM_STUDIO_REMOTE_JOB_ID", job_id)
    monkeypatch.setenv("LLM_STUDIO_REMOTE_AGENT_TOKEN", "token")
    import remote_agent.app as remote_app

    remote_app = importlib.reload(remote_app)
    client = TestClient(remote_app.app)
    headers = {"Authorization": "Bearer token", "X-LLM-Studio-Job-Id": job_id}

    optional_response = client.get(
        f"/v1/jobs/{job_id}/files",
        params={"path": "optional_status.json", "optional": "true"},
        headers=headers,
    )
    required_response = client.get(
        f"/v1/jobs/{job_id}/files",
        params={"path": "optional_status.json"},
        headers=headers,
    )

    assert optional_response.status_code == 200
    assert optional_response.text == ""
    assert required_response.status_code == 404


def test_remote_runner_log_persists_to_workspace(monkeypatch, tmp_path: Path) -> None:
    llm_studio_root = REPO_ROOT / "apps" / "llm-studio"
    if str(llm_studio_root) not in sys.path:
        sys.path.insert(0, str(llm_studio_root))
    import remote_agent.runner as remote_runner

    remote_runner = importlib.reload(remote_runner)
    monkeypatch.setenv("LLM_STUDIO_REMOTE_WORKSPACE", str(tmp_path / "workspace"))

    remote_runner.runner_log("diagnostic_probe", job_id="job-runner-log", api_key="secret")

    payload = json.loads((tmp_path / "workspace" / "logs" / "runner.log").read_text(encoding="utf-8"))
    assert payload["event"] == "diagnostic_probe"
    assert payload["job_id"] == "job-runner-log"
    assert payload["api_key"] == "[redacted]"


def test_remote_agent_start_reports_immediate_subprocess_failure(monkeypatch, tmp_path: Path) -> None:
    llm_studio_root = REPO_ROOT / "apps" / "llm-studio"
    if str(llm_studio_root) not in sys.path:
        sys.path.insert(0, str(llm_studio_root))
    import remote_agent.runner as remote_runner

    remote_runner = importlib.reload(remote_runner)
    job_id = "job-start-exit"
    job_root = tmp_path / "workspace" / "jobs" / job_id
    monkeypatch.setenv("LLM_STUDIO_REMOTE_WORKSPACE", str(tmp_path / "workspace"))
    inputs = job_root / "inputs"
    inputs.mkdir(parents=True)
    for name in ("model_config.json", "tokenizer_artifact.json", "training_config.json", "dataloader_config.json"):
        (inputs / name).write_text("{}", encoding="utf-8")
    outputs = job_root / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "stderr.log").write_text(
        "ModuleNotFoundError: No module named 'training.local_text_data'\n",
        encoding="utf-8",
    )

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 45678

        def poll(self) -> int:
            return 1

    monkeypatch.setattr(remote_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(remote_runner.time, "sleep", lambda _seconds: None)

    runner = remote_runner.RemoteTrainingRunner()
    try:
        runner.start(job_id=job_id, job_root=job_root, manifest={"runner": {"args": {}}})
    except RuntimeError as exc:
        message = str(exc)
        assert "exited during startup" in message
        assert "training.local_text_data" in message
    else:
        raise AssertionError("Expected immediate subprocess failure to be reported")


def test_rewrite_local_dataset_files_copies_and_rewrites_paths(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("hello", encoding="utf-8")
    inputs_dir = tmp_path / "bundle" / "inputs"
    payload = {
        "datasets": [
            {
                "name": "local-text",
                "data_files": {"train": str(source)},
            }
        ]
    }

    rewritten, metadata = rewrite_local_dataset_files(
        payload,
        source_base=tmp_path,
        target_inputs_dir=inputs_dir,
    )

    train_path = rewritten["datasets"][0]["data_files"]["train"]
    assert train_path.startswith("inputs/local_files/000-local-text/")
    assert (tmp_path / "bundle" / train_path).exists()
    assert metadata[0]["original_path"] == str(source)
    assert metadata[0]["remote_path"] == train_path
