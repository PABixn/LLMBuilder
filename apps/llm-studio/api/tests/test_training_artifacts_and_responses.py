from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.training_runs.artifacts import prepare_training_job
from app.training_runs.preflight import ResolvedPreflightContext
from app.training_runs.responses import job_to_response
from app.training_runs.schemas import CreateTrainingJobRequest, TrainingAssetRef, TrainingExecutorKind


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


def test_prepare_training_job_writes_inputs_and_builds_stored_row(tmp_path: Path) -> None:
    tokenizer_source = tmp_path / "tokenizer.json"
    tokenizer_source.write_text("{}", encoding="utf-8")
    request = CreateTrainingJobRequest.model_validate(
        {
            "project_id": "project123",
            "tokenizer_job_id": "tok123456",
            "training_config": {},
            "dataloader_config": {},
            "execution_target": {"kind": TrainingExecutorKind.local.value},
        }
    )

    prepared = prepare_training_job(
        request=request,
        context=make_context(),
        preflight_payload={"valid": True},
        tokenizer_source_path=tokenizer_source,
        training_jobs_root=tmp_path / "jobs",
        runpod_default_gpu_count=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        job_id="job123456",
    )

    assert prepared.stored.id == "job123456"
    assert prepared.stored.name == "Project x Tokenizer"
    assert prepared.stored.executor_kind == "local"
    assert prepared.bundle.manifest["format"] == "llm-studio-training-bundle-v1"
    assert prepared.bundle.manifest["job_id"] == "job123456"
    assert prepared.bundle.model_config_path.read_text(encoding="utf-8")
    assert prepared.bundle.tokenizer_path.read_text(encoding="utf-8") == "{}"
    assert prepared.bundle.resolved_preflight_path.read_text(encoding="utf-8")


def test_job_to_response_maps_runtime_elapsed_fields(tmp_path: Path) -> None:
    tokenizer_source = tmp_path / "tokenizer.json"
    tokenizer_source.write_text("{}", encoding="utf-8")
    request = CreateTrainingJobRequest.model_validate(
        {
            "project_id": "project123",
            "tokenizer_job_id": "tok123456",
            "training_config": {},
            "dataloader_config": {},
            "execution_target": {"kind": TrainingExecutorKind.local.value},
        }
    )
    prepared = prepare_training_job(
        request=request,
        context=make_context(),
        preflight_payload={"valid": True},
        tokenizer_source_path=tokenizer_source,
        training_jobs_root=tmp_path / "jobs",
        runpod_default_gpu_count=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        job_id="job123456",
    )

    response = job_to_response(
        prepared.stored,
        runtime_state={"elapsed_seconds": "12.5", "eta_seconds": "-1"},
    )

    assert response.id == "job123456"
    assert response.elapsed_seconds == 12.5
    assert response.eta_seconds is None
