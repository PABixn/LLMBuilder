from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.training_executors.base import TrainingJobBundle
from app.training_executors.remote_sync import BundleBuildResult
from app.training_executors.runpod_client import CreatePodRequest, RunPodClientError
from app.training_executors.runpod_pod import RunPodPodExecutor
from app.training_models import TrainingJobState, TrainingJobStatus
from app.training_storage import StoredTrainingJob


class CapturingRunPodClient:
    api_keys: list[str] = []
    create_requests: list[CreatePodRequest] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        CapturingRunPodClient.api_keys.append(api_key)

    def create_pod(self, request: CreatePodRequest) -> dict[str, Any]:
        CapturingRunPodClient.create_requests.append(request)
        return {"id": "pod123", "name": "llm-studio-job123456"}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return {
            "id": pod_id,
            "name": "llm-studio-job123456",
            "publicIp": "203.0.113.10",
            "runtime": {
                "ports": [
                    {
                        "type": "tcp",
                        "privatePort": 8021,
                        "publicPort": 18021,
                        "ip": "203.0.113.10",
                    }
                ]
            },
        }

    def stop_pod(self, pod_id: str) -> None:
        return None

    def delete_pod(self, pod_id: str) -> None:
        return None


class CapturingAgentClient:
    instances: list["CapturingAgentClient"] = []

    def __init__(self, base_url: str, token: str, job_id: str) -> None:
        self.base_url = base_url
        self.token = token
        self.job_id = job_id
        self.calls: list[str] = []
        CapturingAgentClient.instances.append(self)

    def health(self) -> dict[str, Any]:
        self.calls.append("health")
        return {"ok": True}

    def system(self) -> dict[str, Any]:
        self.calls.append("system")
        return {"runner": {"import_ok": True}, "workspace": "/workspace/llm-studio"}

    def upload_bundle(self, bundle_path: Path, *, content_type: str) -> dict[str, Any]:
        self.calls.append("upload_bundle")
        return {"ok": True, "content_type": content_type}

    def start(self) -> dict[str, Any]:
        self.calls.append("start")
        return {"ok": True, "process_id": 7654}


def make_job(tmp_path: Path) -> StoredTrainingJob:
    return StoredTrainingJob(
        id="job123456",
        name="RunPod job",
        status=TrainingJobStatus.running,
        state=TrainingJobState.preflight,
        stage="Provisioning RunPod pod",
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
        executor_kind="runpod_pod",
        executor_status="provisioning",
        runpod_cleanup_policy={"pod": "delete_after_sync", "network_volume": "keep"},
    )


def make_bundle(
    tmp_path: Path,
    *,
    api_key: str = "ui-secret-api-key",
    hf_token: str | None = None,
) -> TrainingJobBundle:
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
        manifest={
            "execution_target": {
                "kind": "runpod_pod",
                "api_key": api_key,
                "gpu_type_id": "NVIDIA GeForce RTX 4090",
                "gpu_count": 2,
                "cloud_type": "COMMUNITY",
                "data_center_id": "EU-RO-1",
                "network_volume_size_gb": 123,
                "cleanup_policy": {"pod": "delete_after_sync", "network_volume": "keep"},
            },
            "dataset_hf_tokens": [hf_token],
        },
    )


def fake_bundle_result(tmp_path: Path) -> BundleBuildResult:
    bundle_path = tmp_path / "bundle.tar.gz"
    bundle_path.write_bytes(b"bundle")
    return BundleBuildResult(
        path=bundle_path,
        manifest={"files": [{"path": "manifest.json", "sha256": "abc", "size_bytes": 1}]},
        content_type="application/gzip",
    )


def test_runpod_submit_verifies_agent_compatibility_before_upload(monkeypatch, tmp_path: Path) -> None:
    CapturingRunPodClient.api_keys = []
    CapturingRunPodClient.create_requests = []
    CapturingAgentClient.instances = []
    monkeypatch.setattr("app.training_executors.runpod_pod.RunPodClient", CapturingRunPodClient)
    monkeypatch.setattr("app.training_executors.runpod_pod.RemoteAgentClient", CapturingAgentClient)
    monkeypatch.setattr("app.training_executors.runpod_pod.build_remote_bundle", lambda _bundle: fake_bundle_result(tmp_path))

    handle = RunPodPodExecutor().submit(make_job(tmp_path), make_bundle(tmp_path))

    agent = CapturingAgentClient.instances[0]
    assert agent.calls == ["health", "system", "upload_bundle", "start"]
    assert handle.updates["executor_status"] == "running"


def test_runpod_submit_pod_request_defaults_and_secret_handling(monkeypatch, tmp_path: Path) -> None:
    CapturingRunPodClient.api_keys = []
    CapturingRunPodClient.create_requests = []
    CapturingAgentClient.instances = []
    monkeypatch.setattr("app.training_executors.runpod_pod.RunPodClient", CapturingRunPodClient)
    monkeypatch.setattr("app.training_executors.runpod_pod.RemoteAgentClient", CapturingAgentClient)
    monkeypatch.setattr("app.training_executors.runpod_pod.build_remote_bundle", lambda _bundle: fake_bundle_result(tmp_path))

    hf_token = "hf_0123456789abcdef0123456789abcdef"
    executor = RunPodPodExecutor()
    handle = executor.submit(make_job(tmp_path), make_bundle(tmp_path, hf_token=hf_token))
    request = CapturingRunPodClient.create_requests[0]
    payload = request.to_payload()

    assert CapturingRunPodClient.api_keys == ["ui-secret-api-key"]
    assert payload["imageName"]
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["gpuCount"] == 2
    assert payload["cloudType"] == "COMMUNITY"
    assert payload["dataCenterIds"] == ["EU-RO-1"]
    assert payload["ports"] == ["8021/tcp"]
    assert payload["env"]["LLM_STUDIO_REMOTE_JOB_ID"] == "job123456"
    assert payload["env"]["LLM_STUDIO_REMOTE_AGENT_TOKEN"]
    assert payload["env"]["HF_HOME"].endswith("/cache/huggingface")
    assert payload["env"]["HF_DATASETS_CACHE"].endswith("/cache/huggingface/datasets")
    assert hf_token in payload["env"]["LLM_STUDIO_HF_DATASET_TOKENS"]
    assert "ui-secret-api-key" not in str(payload["env"])
    assert handle.updates["runpod_agent_token_hash"]
    assert executor._agent_tokens["job123456"] == payload["env"]["LLM_STUDIO_REMOTE_AGENT_TOKEN"]
    lifecycle = (tmp_path / "runpod_lifecycle.jsonl").read_text(encoding="utf-8")
    assert "ui-secret-api-key" not in lifecycle
    assert hf_token not in lifecycle


def test_runpod_create_pod_auth_errors_do_not_retry(tmp_path: Path) -> None:
    class AuthErrorClient:
        calls = 0

        def create_pod(self, _request: CreatePodRequest) -> dict[str, Any]:
            AuthErrorClient.calls += 1
            raise RunPodClientError("Unauthorized", status_code=401)

    request = CreatePodRequest(
        name="job",
        image_name="image",
        gpu_type_id="gpu",
        gpu_count=1,
        cloud_type="SECURE",
        container_disk_gb=50,
        volume_gb=100,
        volume_mount_path="/workspace",
        ports=["8021/tcp"],
        env={},
    )

    try:
        RunPodPodExecutor()._create_pod_with_retries(AuthErrorClient(), request, job=make_job(tmp_path))  # type: ignore[arg-type]
    except RunPodClientError as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected auth failures to be non-retryable")

    assert AuthErrorClient.calls == 1
