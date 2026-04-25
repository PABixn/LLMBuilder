from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.training_executors import remote_sync
from app.training_executors.remote_sync import RemoteAgentClient, build_remote_bundle, rewrite_local_dataset_files
from app.training_executors.runpod_client import CreatePodRequest
from app.training_executors.runpod_pod import RunPodPodExecutor, build_agent_base_url
from app.training_storage import TrainingStudioStore

REPO_ROOT = Path(__file__).resolve().parents[4]


class ExitedProcess:
    pid = 12345

    def poll(self) -> int:
        return 1


class FakeCompatibleAgent:
    def system(self) -> dict[str, object]:
        return {"runner": {"import_ok": True}}


class FakeOldAgent:
    def system(self) -> dict[str, object]:
        return {"workspace": "/workspace/llm-studio"}


class FakeBrokenAgent:
    def system(self) -> dict[str, object]:
        return {"runner": {"import_ok": False, "error": "ModuleNotFoundError: No module named 'llm_builder'"}}


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
    job = store.get_job("job123456")

    assert job is not None
    assert job.executor_kind == "local"
    assert job.runpod_gpu_count == 1
    assert job.runpod_pod_id is None


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
        return FakeResponse()

    monkeypatch.setattr(remote_sync, "urlopen", fake_urlopen)

    payload = RemoteAgentClient("https://example.test", "token", "job123").health()

    assert payload == {"ok": True}
    assert captured["context"] is not None


def test_training_image_includes_shared_local_text_module() -> None:
    dockerfile = (REPO_ROOT / "docker" / "training" / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY llm_builder ./llm_builder" in dockerfile
    assert "import llm_builder.local_text_data; import training.runner" in dockerfile


def test_training_entrypoint_runs_startup_diagnostics() -> None:
    entrypoint = (REPO_ROOT / "docker" / "training" / "entrypoint.sh").read_text(encoding="utf-8")
    diagnostics = (REPO_ROOT / "apps" / "llm-studio" / "remote_agent" / "diagnostics.py").read_text(encoding="utf-8")

    assert "python -m remote_agent.diagnostics startup" in entrypoint
    assert "startup.log" in entrypoint
    assert "uvicorn_start" in entrypoint
    assert '"tokenizers": True' in diagnostics
    assert "transformers" not in diagnostics


def test_default_runpod_training_image_does_not_point_at_stale_import_broken_tag() -> None:
    config_source = (REPO_ROOT / "apps" / "llm-studio" / "api" / "app" / "config.py").read_text(encoding="utf-8")
    env_example = (REPO_ROOT / "apps" / "llm-studio" / "api" / ".env.example").read_text(encoding="utf-8")

    assert "ghcr.io/pabixn/llm-builder-training:sha-7037615" not in config_source
    assert "ghcr.io/pabixn/llm-builder-training:sha-7037615" not in env_example
    assert "ghcr.io/pabixn/llm-builder-training:latest" in config_source
    assert "LLM_STUDIO_RUNPOD_TRAINING_IMAGE=ghcr.io/pabixn/llm-builder-training:latest" in env_example


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
        assert "llm_builder" in str(exc)
    else:
        raise AssertionError("Expected broken runner compatibility check to fail")


def test_runpod_executor_accepts_agent_with_runner_compatibility_report() -> None:
    RunPodPodExecutor()._verify_agent_compatibility(FakeCompatibleAgent())  # type: ignore[arg-type]


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
    (outputs / "stderr.log").write_text("ModuleNotFoundError: No module named 'llm_builder'\n", encoding="utf-8")

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
        assert "llm_builder" in message
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
