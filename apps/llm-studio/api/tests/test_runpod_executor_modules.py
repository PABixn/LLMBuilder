from __future__ import annotations

import sys
import hashlib
from types import SimpleNamespace

import pytest

from app.schemas import write_json
from app.training_runs.schemas import TrainingJobStatus
from app.training_runs.executors.base import TrainingJobBundle
from app.training_runs.executors.runpod.bundle import build_remote_bundle
from app.training_runs.executors.runpod.client import (
    RunPodClientError,
    _error_message,
    _extract_items,
    _extract_object,
    _sanitize_error_payload,
)
from app.training_runs.executors.runpod.config import resolve_runpod_target
from app.training_runs.executors.runpod.executor import RunPodPodExecutor
from app.training_runs.executors.runpod.lifecycle_log import (
    is_standard_lifecycle_event,
    lifecycle_error_category,
    log_lifecycle,
    read_lifecycle_events,
    sanitize_log_fields,
)
from app.training_runs.executors.runpod.state import RunPodExecutorStatus, remote_executor_status
from app.training_runs.executors.runpod.sync import sync_final_outputs, sync_remote_checkpoints
from app.training_runs.executors.runpod.tokens import RunPodTokenRegistry, hash_token


def make_settings(**overrides):
    values = {
        "runpod_api_key": "env-key",
        "runpod_training_image": "training-image",
        "runpod_default_gpu_type": "NVIDIA RTX 4090",
        "runpod_default_gpu_count": 1,
        "runpod_default_cloud_type": "SECURE",
        "runpod_default_data_center_id": None,
        "runpod_default_volume_size_gb": 100,
        "runpod_container_disk_gb": 50,
        "runpod_volume_mount_path": "/workspace",
        "runpod_agent_port": 8021,
        "runpod_agent_port_protocol": "tcp",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_remote_bundle(tmp_path):
    bundle = TrainingJobBundle(
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
    write_json(bundle.model_config_path, {"vocab_size": 16, "context_length": 8})
    bundle.tokenizer_path.write_text("{}", encoding="utf-8")
    write_json(bundle.training_config_path, {"max_steps": 1})
    write_json(bundle.dataloader_config_path, {"datasets": [{"name": "dataset"}]})
    write_json(bundle.resolved_preflight_path, {"valid": True})
    return bundle


def test_resolve_runpod_target_prefers_ui_key_and_normalizes_defaults() -> None:
    target = resolve_runpod_target(
        {
            "api_key": " ui-key ",
            "gpu_count": 2,
            "cloud_type": "community",
            "data_center_id": "EU-RO-1",
            "network_volume_size_gb": 123,
            "cleanup_policy": {"pod": "keep", "network_volume": "keep"},
            "interruptible": True,
        },
        make_settings(),
    )

    assert target.api_key == "ui-key"
    assert target.api_key_source == "ui"
    assert target.gpu_count == 2
    assert target.cloud_type == "COMMUNITY"
    assert target.data_center_id == "EU-RO-1"
    assert target.volume_size_gb == 123
    assert target.agent_port_protocol == "tcp"
    assert target.cleanup_policy == {"pod": "keep", "network_volume": "keep"}
    assert target.interruptible is True


def test_resolve_runpod_target_uses_env_key_when_ui_key_absent() -> None:
    target = resolve_runpod_target({}, make_settings(runpod_api_key="env-secret"))

    assert target.api_key == "env-secret"
    assert target.api_key_source == "env"


def test_token_registry_keeps_raw_secrets_process_local() -> None:
    registry = RunPodTokenRegistry()

    token = registry.create_agent_token("job123456")
    registry.set_api_key("job123456", "api-key")

    assert registry.agent_token("job123456") == token
    assert registry.api_key("job123456") == "api-key"
    assert hash_token(token) != token
    assert len(hash_token(token)) == 64
    assert "not recoverable" in registry.missing_token_error()
    registry.clear("job123456")
    assert registry.agent_token("job123456") is None
    assert registry.api_key("job123456") is None


def test_lifecycle_sanitizer_recursively_redacts_secrets() -> None:
    sanitized = sanitize_log_fields(
        {
            "api_key": "secret",
            "headers": {"Authorization": "Bearer secret"},
            "nested": [{"agent_token": "secret"}, {"hf_token": "secret"}],
            "safe": "value",
        }
    )

    assert sanitized == {
        "api_key": "[redacted]",
        "headers": {"Authorization": "[redacted]"},
        "nested": [{"agent_token": "[redacted]"}, {"hf_token": "[redacted]"}],
        "safe": "value",
    }


def test_lifecycle_log_includes_correlation_id_category_and_is_parseable(tmp_path) -> None:
    job = SimpleNamespace(id="job123456", artifact_dir=str(tmp_path))

    log_lifecycle(job, "cleanup_failed", "Cleanup failed.", error="delete failed")
    events = read_lifecycle_events(tmp_path)

    assert events[0]["job_id"] == "job123456"
    assert events[0]["correlation_id"] == "job123456"
    assert events[0]["category"] == "cleanup_failure"
    assert is_standard_lifecycle_event(events[0]["event"]) is True
    assert lifecycle_error_category("create_pod_failed", {"error": "no instances currently available"}) == "no_capacity"


def test_runpod_status_labels_map_from_remote_state() -> None:
    assert [status.value for status in RunPodExecutorStatus] == [
        "queued",
        "provisioning",
        "booting",
        "checking_agent",
        "building_bundle",
        "uploading",
        "starting",
        "running",
        "syncing",
        "cleaning_up",
        "completed",
        "failed",
        "cancelled",
        "cleaned_up",
    ]
    assert remote_executor_status({"status": "completed"}) == "completed"
    assert remote_executor_status({"stage": "Syncing outputs"}) == "syncing"
    assert remote_executor_status({"stage": "Uploading bundle"}) == "uploading"
    assert remote_executor_status({"stage": "Training"}) == "running"


def test_runpod_client_extracts_list_and_object_response_shapes() -> None:
    assert _extract_items([{"id": "pod"}, "ignored"]) == [{"id": "pod"}]
    assert _extract_items({"items": [{"id": "item"}]}) == [{"id": "item"}]
    assert _extract_items({"pods": [{"id": "pod"}]}) == [{"id": "pod"}]
    assert _extract_items({"networkVolumes": [{"id": "volume"}]}) == [{"id": "volume"}]
    assert _extract_items({"data": [{"id": "data"}]}) == [{"id": "data"}]

    assert _extract_object({"pod": {"id": "pod"}}) == {"id": "pod"}
    assert _extract_object({"networkVolume": {"id": "volume"}}) == {"id": "volume"}
    assert _extract_object({"data": {"id": "data"}}) == {"id": "data"}
    assert _extract_object({"id": "raw"}) == {"id": "raw"}
    with pytest.raises(RunPodClientError):
        _extract_object([{"id": "unexpected"}])


def test_runpod_client_error_payloads_are_sanitized() -> None:
    payload = _sanitize_error_payload(
        {
            "message": "capacity unavailable",
            "api_key": "secret",
            "nested": {"Authorization": "Bearer secret"},
        }
    )

    assert payload == {
        "message": "capacity unavailable",
        "api_key": "[redacted]",
        "nested": {"Authorization": "[redacted]"},
    }
    assert _error_message(payload) == "capacity unavailable"


def test_remote_bundle_falls_back_to_gzip_when_zstandard_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "zstandard", None)
    bundle = make_remote_bundle(tmp_path)

    result = build_remote_bundle(bundle)

    assert result.path.name == "bundle.tar.gz"
    assert result.path.exists()
    assert result.content_type == "application/gzip"
    assert result.manifest["format"] == "llm-studio-training-bundle-v1"


def test_remote_bundle_uses_zstandard_when_available(tmp_path) -> None:
    pytest.importorskip("zstandard")
    bundle = make_remote_bundle(tmp_path)

    result = build_remote_bundle(bundle)

    assert result.path.name == "bundle.tar.zst"
    assert result.path.exists()
    assert result.content_type == "application/zstd"
    assert result.manifest["format"] == "llm-studio-training-bundle-v1"


def test_remote_bundle_removes_staging_directory_after_success(tmp_path) -> None:
    bundle = make_remote_bundle(tmp_path)

    result = build_remote_bundle(bundle)

    assert result.path.exists()
    assert not (tmp_path / ".remote_bundle").exists()


class FakeRemoteAgent:
    def __init__(self, files, checkpoints):
        self.files = dict(files)
        self._checkpoints = checkpoints
        self.downloads = []

    def checkpoints(self):
        return self._checkpoints

    def download_file(self, relative_path, local_path, *, offset=0, optional=False):
        if relative_path not in self.files:
            if optional:
                return b""
            raise AssertionError(f"unexpected remote path: {relative_path}")
        data = self.files[relative_path]
        self.downloads.append(relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if offset else "wb"
        with local_path.open(mode + ("" if "b" in mode else "b")) as handle:
            handle.write(data)
        return data

    def download_append_file(self, remote_kind, local_path):
        return None


def test_sync_remote_checkpoints_downloads_and_verifies_manifest_files(tmp_path) -> None:
    payload = b"checkpoint-weights"
    digest = hashlib.sha256(payload).hexdigest()
    agent = FakeRemoteAgent(
        {"checkpoints/3/model.pt": payload},
        [
            {
                "step": 3,
                "directory": "checkpoints/3",
                "files": [
                    {
                        "path": "checkpoints/3/model.pt",
                        "size_bytes": len(payload),
                        "sha256": digest,
                    }
                ],
            }
        ],
    )
    job = SimpleNamespace(id="job123456", artifact_dir=str(tmp_path))

    assert sync_remote_checkpoints(agent, job) == 1
    assert (tmp_path / "checkpoints" / "3" / "model.pt").read_bytes() == payload

    assert sync_remote_checkpoints(agent, job) == 1
    assert agent.downloads == ["checkpoints/3/model.pt"]


def test_sync_final_outputs_requires_verified_artifact_manifest(tmp_path) -> None:
    manifest = b'{"job_id":"job123456","files":[]}'
    agent = FakeRemoteAgent({"artifact_manifest.json": manifest}, [])
    job = SimpleNamespace(id="job123456", artifact_dir=str(tmp_path))

    result = sync_final_outputs(agent, job)

    assert result.final_manifest_verified is True
    assert (tmp_path / "artifact_manifest.json").read_bytes() == manifest


def test_terminal_delete_after_sync_waits_for_final_manifest(tmp_path) -> None:
    class ExecutorThatWouldFailIfCleanupRuns(RunPodPodExecutor):
        def cleanup(self, job, policy):  # pragma: no cover - should not be reached
            raise AssertionError("cleanup should wait for final sync")

    job = SimpleNamespace(
        id="job123456",
        artifact_dir=str(tmp_path),
        status=TrainingJobStatus.completed,
        executor_status="syncing",
        runpod_pod_id="pod123",
        runpod_agent_base_url="http://agent",
        runpod_cleanup_policy={"pod": "delete_after_sync", "network_volume": "keep"},
    )

    snapshot = ExecutorThatWouldFailIfCleanupRuns().refresh(job)

    assert snapshot.updates["executor_status"] == "syncing"
    assert "final artifact sync is incomplete" in snapshot.updates["remote_error"]
