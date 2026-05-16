from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from app.main import app
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
