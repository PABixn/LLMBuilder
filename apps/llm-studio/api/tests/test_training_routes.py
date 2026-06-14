from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from app.main import app
from app.training_runs.executors.runpod.client import RunPodClientError
from app.training_runs.routes import _stream_generation_events
from app.training_runs.schemas import TrainingGenerateRequest
from app.training_models import TrainingExecutorKind, TrainingJobResponse, TrainingJobState, TrainingJobStatus


def test_training_public_routes_are_registered() -> None:
    routes = {
        (next(iter(route.methods - {"HEAD", "OPTIONS"})), route.path)
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/api/v1/training")
    }

    expected = {
        ("GET", "/api/v1/training/health"),
        ("GET", "/api/v1/training/config/templates"),
        ("GET", "/api/v1/training/config/schemas"),
        ("POST", "/api/v1/training/validate/dataloader"),
        ("POST", "/api/v1/training/validate/training-config"),
        ("POST", "/api/v1/training/validate/preflight"),
        ("GET", "/api/v1/training/providers/runpod/defaults"),
        ("GET", "/api/v1/training/providers/runpod/status"),
        ("GET", "/api/v1/training/providers/runpod/catalog"),
        ("POST", "/api/v1/training/providers/runpod/validate-key"),
        ("GET", "/api/v1/training/providers/runpod/pods"),
        ("GET", "/api/v1/training/providers/runpod/network-volumes"),
        ("POST", "/api/v1/training/jobs"),
        ("GET", "/api/v1/training/jobs"),
        ("GET", "/api/v1/training/jobs/{job_id}"),
        ("DELETE", "/api/v1/training/jobs/{job_id}"),
        ("GET", "/api/v1/training/jobs/{job_id}/metrics"),
        ("GET", "/api/v1/training/jobs/{job_id}/samples"),
        ("GET", "/api/v1/training/jobs/{job_id}/logs"),
        ("GET", "/api/v1/training/jobs/{job_id}/checkpoints"),
        ("POST", "/api/v1/training/jobs/{job_id}/generate"),
        ("POST", "/api/v1/training/jobs/{job_id}/generate/stream"),
        ("POST", "/api/v1/training/jobs/{job_id}/stop"),
        ("POST", "/api/v1/training/jobs/{job_id}/remote/resync"),
        ("POST", "/api/v1/training/jobs/{job_id}/remote/cleanup"),
        ("POST", "/api/v1/training/jobs/{job_id}/remote/reattach"),
        ("GET", "/api/v1/training/jobs/{job_id}/artifact"),
    }

    assert expected <= routes


def test_runpod_catalog_exposes_selectable_gpu_choices() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/training/providers/runpod/catalog")

    assert response.status_code == 200
    payload = response.json()
    gpu_options = payload["gpu_options"]
    assert gpu_options
    ids = {option["id"] for option in gpu_options}
    assert "NVIDIA GeForce RTX 4090" in ids
    assert any(option["memory_gb"] for option in gpu_options)


def test_runpod_resource_routes_redact_provider_echoed_credentials(monkeypatch) -> None:
    hf_token = "hf_0123456789abcdef0123456789abcdef"
    runpod_token = "rpa_0123456789abcdef0123456789abcdef"

    class FakeRunPodClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "memory-key"

        def list_pods(self) -> list[dict[str, object]]:
            return [
                {
                    "id": "pod-1",
                    "env": {"LLM_STUDIO_HF_DATASET_TOKENS": hf_token},
                    "message": f"provider echoed {runpod_token}",
                }
            ]

        def list_network_volumes(self) -> list[dict[str, object]]:
            return [
                {
                    "id": "volume-1",
                    "metadata": {"hf_token": hf_token},
                    "message": f"provider echoed {runpod_token}",
                }
            ]

    monkeypatch.setattr("app.training_runs.routes.RunPodClient", FakeRunPodClient)
    previous = getattr(app.state, "runpod_api_key_override", None)
    app.state.runpod_api_key_override = "memory-key"
    try:
        client = TestClient(app)
        responses = [
            client.get("/api/v1/training/providers/runpod/pods"),
            client.get("/api/v1/training/providers/runpod/network-volumes"),
        ]
    finally:
        app.state.runpod_api_key_override = previous

    for response in responses:
        assert response.status_code == 200
        serialized = response.text
        assert hf_token not in serialized
        assert runpod_token not in serialized
        assert "[REDACTED]" in serialized


def test_runpod_key_validation_never_echoes_provider_credentials(monkeypatch) -> None:
    runpod_token = "rpa_0123456789abcdef0123456789abcdef"

    class FailingRunPodClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == runpod_token

        def validate_key(self) -> dict[str, object]:
            raise RunPodClientError(f"provider echoed {runpod_token}")

    monkeypatch.setattr("app.training_runs.routes.RunPodClient", FailingRunPodClient)
    response = TestClient(app).post(
        "/api/v1/training/providers/runpod/validate-key",
        json={"api_key": runpod_token},
    )

    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert runpod_token not in response.text
    assert "[REDACTED]" in response.text


def make_training_response(job_id: str, *, executor_status: str = "running", remote_error: str | None = None) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=job_id,
        name="RunPod job",
        status=TrainingJobStatus.running,
        state=TrainingJobState.preflight,
        stage="Running",
        progress=0.0,
        created_at=datetime.now(timezone.utc),
        started_at=None,
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
        artifact_dir="/tmp/job",
        artifact_bundle_file=None,
        stats_path="/tmp/job/stats.jsonl",
        samples_path="/tmp/job/samples.jsonl",
        stdout_path="/tmp/job/stdout.log",
        stderr_path="/tmp/job/stderr.log",
        last_step=0,
        max_steps=10,
        elapsed_seconds=None,
        eta_seconds=None,
        latest_loss=None,
        latest_grad_norm=None,
        latest_lr=None,
        latest_tokens_per_sec=None,
        checkpoint_count=0,
        sample_count=0,
        error=None,
        process_id=None,
        output_size_bytes=0,
        executor_kind=TrainingExecutorKind.runpod_pod,
        executor_status=executor_status,
        remote_error=remote_error,
    )


class FakeRemoteActionManager:
    def __init__(self) -> None:
        self.cleanup_ids: list[str] = []
        self.reattach_ids: list[str] = []

    def cleanup_remote_job(self, job_id: str) -> TrainingJobResponse:
        self.cleanup_ids.append(job_id)
        return make_training_response(job_id, executor_status="cleaned_up")

    def reattach_remote_job(self, job_id: str) -> TrainingJobResponse:
        self.reattach_ids.append(job_id)
        return make_training_response(
            job_id,
            remote_error="Remote reattach is unavailable in this version.",
        )


def test_remote_cleanup_and_reattach_routes_delegate_to_manager() -> None:
    manager = FakeRemoteActionManager()
    previous = getattr(app.state, "training_jobs", None)
    app.state.training_jobs = manager
    client = TestClient(app)
    try:
        cleanup_response = client.post("/api/v1/training/jobs/job123456/remote/cleanup")
        reattach_response = client.post("/api/v1/training/jobs/job123456/remote/reattach")
    finally:
        if previous is None:
            del app.state.training_jobs
        else:
            app.state.training_jobs = previous

    assert cleanup_response.status_code == 200
    assert cleanup_response.json()["executor_status"] == "cleaned_up"
    assert reattach_response.status_code == 200
    assert "unavailable" in reattach_response.json()["remote_error"]
    assert manager.cleanup_ids == ["job123456"]
    assert manager.reattach_ids == ["job123456"]


class FakeStreamingTokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return " ".join(str(token_id) for token_id in token_ids)


class DisconnectingRequest:
    def __init__(self, states: list[bool]) -> None:
        self._states = iter(states)

    async def is_disconnected(self) -> bool:
        return next(self._states, True)


class TrackingStreamingModel:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.closed = False
        self.failure = failure

    def generate(self, **_kwargs):
        try:
            yield 11
            if self.failure is not None:
                raise self.failure
            yield 12
        finally:
            self.closed = True


class DirectFailureStreamingModel:
    def generate(self, **_kwargs):
        raise RuntimeError("direct deterministic failure")


def collect_stream_events(**kwargs) -> list[dict[str, object]]:
    async def collect() -> list[dict[str, object]]:
        return [
            json.loads(event)
            async for event in _stream_generation_events(**kwargs)
        ]

    return asyncio.run(collect())


def stream_kwargs(
    *,
    request: DisconnectingRequest,
    model: TrackingStreamingModel,
    tmp_path: Path,
) -> dict[str, object]:
    return {
        "request": request,
        "job_id": "job123456",
        "checkpoint_step": 3,
        "checkpoint_path": tmp_path / "model.pt",
        "tokenizer_job_id": "tok123456",
        "tokenizer": FakeStreamingTokenizer(),
        "prompt_token_ids": [1, 2],
        "model": model,
        "payload": TrainingGenerateRequest(prompt="hello", max_tokens=2),
    }


def test_stream_generation_stops_and_closes_model_iterator_after_disconnect(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="llm_studio.training_routes")
    model = TrackingStreamingModel()

    events = collect_stream_events(
        **stream_kwargs(
            request=DisconnectingRequest([False, True]),
            model=model,
            tmp_path=tmp_path,
        )
    )

    assert [event["type"] for event in events] == ["start", "token"]
    assert events[1]["token_id"] == 11
    assert model.closed is True
    assert any(
        getattr(record, "event_id", None) == "training.inference.stream.cancelled"
        for record in caplog.records
    )


def test_stream_generation_emits_sanitized_error_and_closes_model_iterator(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="llm_studio.training_routes")
    model = TrackingStreamingModel(failure=RuntimeError("deterministic failure"))

    events = collect_stream_events(
        **stream_kwargs(
            request=DisconnectingRequest([False, False, False]),
            model=model,
            tmp_path=tmp_path,
        )
    )

    assert [event["type"] for event in events] == ["start", "token", "error"]
    assert events[-1]["detail"] == "Inference failed: RuntimeError: deterministic failure"
    assert model.closed is True
    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event_id", None) == "training.inference.stream.failed"
    )
    assert getattr(failure_record, "event_fields")["error_type"] == "RuntimeError"
    assert "deterministic failure" not in str(getattr(failure_record, "event_fields"))


def test_stream_generation_emits_error_when_model_rejects_generation_setup(
    tmp_path: Path,
) -> None:
    events = collect_stream_events(
        **stream_kwargs(
            request=DisconnectingRequest([False]),
            model=DirectFailureStreamingModel(),  # type: ignore[arg-type]
            tmp_path=tmp_path,
        )
    )

    assert [event["type"] for event in events] == ["start", "error"]
    assert events[-1]["detail"] == (
        "Inference failed: RuntimeError: direct deterministic failure"
    )
